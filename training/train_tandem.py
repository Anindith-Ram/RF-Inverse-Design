"""
Phase 4b: Tandem inverse model training.

The tandem network addresses non-uniqueness by adding a forward-consistency
loss term. The differentiable ABCD synthesis evaluates whether the predicted
LC values reproduce the target S21 curve.

Two-phase training schedule
---------------------------
Phase 1 — warm-start (WARM_EPOCHS epochs):
    L = MSE(I(x), y_LC)
    Supervised LC loss only. Gives the inverse model a reasonable initialization
    before the forward-consistency gradient is turned on.

Phase 2 — tandem (remaining epochs):
    L = α·MSE(I(x), y_LC) + β·MSE(ABCD(I(x)), x_S21)
    Uses exact differentiable ABCD synthesis (not the neural ForwardMLP surrogate)
    so the forward-consistency gradient is physically exact.

Gradient path
-------------
    x_input → TandemInverseMLP.forward() → y_log_pred (3 heads)
    y_log_pred → DifferentiableInverseTransform → y_log_raw (log10 LC)
    y_log_raw → 10^y_log_raw = L_pred, C_pred
    L_pred, C_pred, Q_L_gt, Q_C_gt → abcd_s21_batch() → S21_pred (101-dim)
    S21_pred vs S21_target → L_fwd

Usage:
    source rf_env/bin/activate
    python training/train_tandem.py

Outputs:
    results/tandem_best.pt          — trained TandemInverseMLP checkpoint
    results/figures/tandem_loss.png — train/val loss curves (LC + fwd losses)
"""

import os
import sys
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.tandem import TandemInverseMLP
from evaluation.metrics import (
    component_mse, component_mae, r2_per_component,
    roundtrip_mse_lc, FREQ_HZ,
)
from evaluation.visualize import plot_scatter, plot_r2_bars, plot_roundtrip

# Reuse dataset + evaluation helpers from train_mlp
from training.train_mlp import (
    FilterDataset, predict_all, evaluate_model,
    build_roundtrip_samples,
)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

BATCH_SIZE   = 256
MAX_EPOCHS   = 500
WARM_EPOCHS  = 100      # Phase 1: supervised LC loss only — allow full convergence
LR           = 1e-3
WEIGHT_DECAY = 1e-4
LR_PATIENCE  = 20
LR_FACTOR    = 0.5
ES_PATIENCE  = 50

# Tandem loss weights (active only in Phase 2, after warm-start).
# Differentiable ABCD synthesis gives fwd_loss in dB². At warm-start convergence,
# lc_loss ≈ 0.003 (normalized), fwd_loss ≈ 24-46 dB².
# BETA is ramped from 0 → BETA_TARGET over BETA_RAMP_EPOCHS to avoid destroying
# the warm-start initialization on the first tandem epoch (fwd_loss starts ~300 dB²).
ALPHA       = 1.0      # weight on supervised LC loss (normalized MSE)
BETA_TARGET = 0.001    # final weight on ABCD S21 loss (dB²) — 0.001*30=0.03, ratio ~10:1 LC
BETA_RAMP_EPOCHS = 50  # linearly ramp BETA from 0 → BETA_TARGET over this many tandem epochs

# Floor applied to S11 targets in forward-consistency loss (mirrors train_forward.py)
S11_FLOOR_DB = -60.0


# ── Differentiable ABCD synthesis ────────────────────────────────────────────

# Frequency grid registered as a buffer-like constant (not a parameter)
_FREQ_HZ = torch.from_numpy(
    __import__('numpy').linspace(40e9, 90e9, 101)
).float()
_Z0 = 50.0


def _abcd_cascade_vectorized(
    L: torch.Tensor,    # (B, n_res)  in SI (Henries) — differentiable
    C: torch.Tensor,    # (B, n_res)  in SI (Farads)  — differentiable
    Q_L: torch.Tensor,  # (B, n_res)  float, NaN→fallback applied externally
    Q_C: torch.Tensor,  # (B, n_res)  float
    fc_GHz: torch.Tensor,  # (B,)
    freq: torch.Tensor,    # (F,) Hz
    device: torch.device,
) -> torch.Tensor:
    """
    Vectorized ABCD cascade for a group of samples all with the same N.

    Args:
        L, C:    (B, N) tensors — SI values with gradient
        Q_L, Q_C: (B, N) float tensors — no gradient needed
        fc_GHz:  (B,) center frequencies
        freq:    (F,) frequency grid in Hz

    Returns:
        s21_db: (B, F) normalized S21 in dB
    """
    B, n_res = L.shape
    F = len(freq)

    f0  = (fc_GHz * 1e9).view(B, 1)       # (B, 1)
    w0  = 2.0 * torch.pi * f0             # (B, 1)
    w   = (2.0 * torch.pi * freq).view(1, F)  # (1, F)

    # Initialize ABCD as identity: M[b,f] = I  → shape (B, F, 2, 2) complex
    M = torch.zeros(B, F, 2, 2, dtype=torch.complex64, device=device)
    M[:, :, 0, 0] = 1.0
    M[:, :, 1, 1] = 1.0

    for k in range(n_res):
        Lk  = L[:, k].view(B, 1)   # (B, 1)
        Ck  = C[:, k].view(B, 1)
        qLk = Q_L[:, k].view(B, 1)
        qCk = Q_C[:, k].view(B, 1)

        # Shape: (B, F) real → cast to complex for arithmetic
        R_s   = (w0 * Lk / qLk) * torch.sqrt(freq.view(1, F) / f0)   # (B, F) skin-effect
        R_esr = 1.0 / (w0 * Ck * qCk)                                  # (B, 1)

        Z_L = R_s.to(torch.complex64)   + (1j * w * Lk).to(torch.complex64)   # (B, F)
        Z_C = R_esr.to(torch.complex64) + (1.0 / (1j * w * Ck)).to(torch.complex64)
        Z_res = Z_L + Z_C    # (B, F) complex

        # Build (B*F, 2, 2) ABCD element matrix
        Mk = torch.zeros(B, F, 2, 2, dtype=torch.complex64, device=device)
        Mk[:, :, 0, 0] = 1.0
        Mk[:, :, 1, 1] = 1.0
        if (k + 1) % 2 == 1:   # series branch
            Mk[:, :, 0, 1] = Z_res
        else:                    # shunt branch
            Mk[:, :, 1, 0] = 1.0 / Z_res

        # Cascade: (B*F, 2, 2) bmm
        M_flat  = M.view(B * F, 2, 2)
        Mk_flat = Mk.view(B * F, 2, 2)
        M = torch.bmm(M_flat, Mk_flat).view(B, F, 2, 2)

    # Extract S21
    A  = M[:, :, 0, 0]
    B_ = M[:, :, 0, 1]
    C_ = M[:, :, 1, 0]
    D_ = M[:, :, 1, 1]
    denom  = A + B_ / _Z0 + _Z0 * C_ + D_    # (B, F)
    s21_c  = 2.0 / denom                       # (B, F) complex
    s21_lin = s21_c.abs()                       # (B, F) real

    # Normalize per sample (peak = 0 dB) and convert to dB
    peak    = s21_lin.max(dim=1, keepdim=True).values.clamp(min=1e-30)
    s21_norm = s21_lin / peak
    s21_db   = 20.0 * torch.log10(s21_norm.clamp(min=1e-5))   # clamp → -100 dB floor
    return s21_db


def abcd_s21_batch(
    L_log: torch.Tensor,  # (B, 5) log10(L) — interleaved, unused cols = 0
    C_log: torch.Tensor,  # (B, 5) log10(C) — interleaved, unused cols = 0
    Q_L:   torch.Tensor,  # (B, 5) ground-truth Q, NaN-padded
    Q_C:   torch.Tensor,
    fc_GHz: torch.Tensor,  # (B,)
    N:      torch.Tensor,  # (B,) long
    device: torch.device,
) -> torch.Tensor:
    """
    Differentiable ABCD cascade → S21 in dB, normalized so peak = 0 dB.

    Vectorized: processes all samples with the same N simultaneously using
    (B×F, 2, 2) matrix multiplications rather than looping over B.

    Returns:
        s21_db: (B, 101) float tensor
    """
    freq  = _FREQ_HZ.to(device)
    B     = L_log.shape[0]
    s21_out = torch.zeros(B, 101, device=device)

    for Nval in [3, 4, 5]:
        mask = (N == Nval)
        if not mask.any():
            continue

        # Convert log10 → SI (gradient flows through here)
        L_si = 10.0 ** L_log[mask, :Nval]   # (B_N, Nval)
        C_si = 10.0 ** C_log[mask, :Nval]

        # Replace NaN Q with safe fallback (shouldn't occur for k < Nval)
        qL = Q_L[mask, :Nval].nan_to_num(nan=10.0)
        qC = Q_C[mask, :Nval].nan_to_num(nan=100.0)

        s21_out[mask] = _abcd_cascade_vectorized(
            L_si, C_si, qL, qC, fc_GHz[mask], freq, device
        )

    return s21_out


# ── Differentiable inverse-transform ─────────────────────────────────────────

class DifferentiableInverseTransform(nn.Module):
    """
    Applies the inverse StandardScaler transform as a differentiable affine op.

    y_raw = y_norm * std + mean

    Registered as buffers so the operation is on the correct device and gradients
    flow through y_norm → y_raw without numpy round-trips.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.register_buffer('mean', torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer('std',  torch.from_numpy(std.astype(np.float32)))

    def forward(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * self.std + self.mean


# ── Tandem loss computation ───────────────────────────────────────────────────

def compute_tandem_loss(
    model:   TandemInverseMLP,
    inv_3:   DifferentiableInverseTransform,
    inv_4:   DifferentiableInverseTransform,
    inv_5:   DifferentiableInverseTransform,
    batch:   dict,
    device:  torch.device,
    use_fwd: bool,
    beta:    float = BETA_TARGET,
) -> tuple[torch.Tensor, float, float]:
    """
    Compute the combined tandem loss for one batch.

    Forward-consistency uses differentiable ABCD synthesis (exact physics)
    rather than the neural ForwardMLP surrogate to avoid forward model bias.

    Args:
        model:     TandemInverseMLP (being trained)
        inv_3/4/5: differentiable inverse transforms for each N group
        batch:     dict from FilterDataset
        device:    compute device
        use_fwd:   if False, only the LC supervision term is computed (warm-start)

    Returns:
        total_loss: scalar tensor
        lc_loss_val:  float (for logging)
        fwd_loss_val: float (for logging, 0.0 if use_fwd=False)
    """
    x       = batch['x'].to(device)
    y_norm  = batch['y_norm'].to(device)
    N_batch = batch['N']

    out_3, out_4, out_5 = model(x)

    # ── Phase 1/2: supervised LC loss ────────────────────────────────────────
    lc_losses = []
    mask3 = N_batch == 3
    mask4 = N_batch == 4
    mask5 = N_batch == 5

    if mask3.any():
        lc_losses.append(nn.functional.mse_loss(out_3[mask3], y_norm[mask3, :6]))
    if mask4.any():
        lc_losses.append(nn.functional.mse_loss(out_4[mask4], y_norm[mask4, :8]))
    if mask5.any():
        lc_losses.append(nn.functional.mse_loss(out_5[mask5], y_norm[mask5, :10]))

    lc_loss = torch.stack(lc_losses).mean()

    if not use_fwd:
        return lc_loss, lc_loss.item(), 0.0

    # ── Phase 2: forward-consistency loss (differentiable ABCD synthesis) ────
    # Denormalize predictions to log10(LC) space via differentiable affine op.
    batch_size = x.size(0)
    # Interleaved layout: columns 0,2,4,... = log10(L1..L5),
    #                     columns 1,3,5,... = log10(C1..C5)
    y_log_pred = torch.zeros(batch_size, 10, device=device)

    if mask3.any():
        y_log_pred[mask3, :6]  = inv_3(out_3[mask3])
    if mask4.any():
        y_log_pred[mask4, :8]  = inv_4(out_4[mask4])
    if mask5.any():
        y_log_pred[mask5, :10] = inv_5(out_5[mask5])

    # Extract per-resonator L and C (interleaved: L1,C1,L2,C2,...)
    # y_log_pred[:, 0::2] = log10(L1..L5),  y_log_pred[:, 1::2] = log10(C1..C5)
    L_log = y_log_pred[:, 0::2]   # (batch, 5)
    C_log = y_log_pred[:, 1::2]   # (batch, 5)

    Q_L    = batch['Q_L'].to(device)    # (batch, 5), NaN-padded
    Q_C    = batch['Q_C'].to(device)
    fc_GHz = batch['fc_GHz'].to(device)
    N_long = batch['N'].to(device)

    # Compute exact differentiable S21 from predicted LC + ground-truth Q
    s21_pred = abcd_s21_batch(L_log, C_log, Q_L, Q_C, fc_GHz, N_long, device)

    s21_target = batch['s21_target'].to(device)
    fwd_loss = nn.functional.mse_loss(s21_pred, s21_target)

    total_loss = ALPHA * lc_loss + beta * fwd_loss
    return total_loss, lc_loss.item(), fwd_loss.item()


# ── Training loop ─────────────────────────────────────────────────────────────

def train_tandem(
    model:        TandemInverseMLP,
    inv_3:        DifferentiableInverseTransform,
    inv_4:        DifferentiableInverseTransform,
    inv_5:        DifferentiableInverseTransform,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
    checkpoint_path: str,
) -> tuple[list, list, list, list]:
    """
    Two-phase tandem training.

    Returns:
        (train_lc_losses, val_lc_losses, train_fwd_losses, val_fwd_losses)
        All are per-epoch lists. Forward losses are 0.0 during warm-start.
    """
    model.to(device)
    inv_3.to(device)
    inv_4.to(device)
    inv_5.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=LR_PATIENCE, factor=LR_FACTOR,
    )

    best_val = float('inf')
    no_improve = 0
    tr_lc, vl_lc, tr_fwd, vl_fwd = [], [], [], []

    phase2_started = False
    current_beta   = 0.0

    pbar = tqdm(range(1, MAX_EPOCHS + 1), desc='Training TandemInverseMLP')
    for epoch in pbar:
        use_fwd = epoch > WARM_EPOCHS

        # Reset early stopping and scheduler when transitioning to tandem phase.
        # The combined loss (ALPHA*lc + beta*fwd) has a different scale from the
        # warm-start LC loss, so the Phase 1 best cannot serve as Phase 2 baseline.
        if use_fwd and not phase2_started:
            phase2_started = True
            best_val = float('inf')
            no_improve = 0
            # Re-initialize scheduler with a lower LR for fine-tuning
            for pg in optimizer.param_groups:
                pg['lr'] = LR * 0.1
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=LR_PATIENCE, factor=LR_FACTOR,
            )
            # Save warm-start checkpoint as Phase 1 baseline (for comparison)
            warm_ckpt = checkpoint_path.replace('.pt', '_warm.pt')
            torch.save(model.state_dict(), warm_ckpt)
            print(f'\n  Phase 2 start: reset ES, save warm checkpoint → {warm_ckpt}')

        # Linearly ramp beta from 0 → BETA_TARGET over the first BETA_RAMP_EPOCHS
        # tandem epochs. This prevents the ~300 dB² initial fwd loss from destroying
        # the warm-start initialization before the model has had a chance to adapt.
        if use_fwd:
            tandem_epoch = epoch - WARM_EPOCHS  # 1-indexed
            ramp_frac = min(tandem_epoch / BETA_RAMP_EPOCHS, 1.0)
            current_beta = BETA_TARGET * ramp_frac

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        t_lc_total = t_fwd_total = 0.0
        for batch in train_loader:
            total, lc_v, fwd_v = compute_tandem_loss(
                model, inv_3, inv_4, inv_5, batch, device, use_fwd, current_beta
            )
            optimizer.zero_grad()
            total.backward()
            # Clip gradients — ABCD synthesis creates large gradients at stopband
            # frequencies where s21_lin → 0 and d(log)/ds21_lin → ∞
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_lc_total  += lc_v
            t_fwd_total += fwd_v

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        v_lc_total = v_fwd_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                _, lc_v, fwd_v = compute_tandem_loss(
                    model, inv_3, inv_4, inv_5, batch, device, use_fwd, current_beta
                )
                v_lc_total  += lc_v
                v_fwd_total += fwd_v

        n_tr = len(train_loader)
        n_vl = len(val_loader)
        t_lc  = t_lc_total  / n_tr
        t_fwd = t_fwd_total / n_tr
        v_lc  = v_lc_total  / n_vl
        v_fwd = v_fwd_total / n_vl

        val_combined = ALPHA * v_lc + current_beta * v_fwd if use_fwd else v_lc
        scheduler.step(val_combined)

        tr_lc.append(t_lc)
        vl_lc.append(v_lc)
        tr_fwd.append(t_fwd)
        vl_fwd.append(v_fwd)

        if val_combined < best_val:
            best_val = val_combined
            no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            no_improve += 1

        phase_label = 'tandem' if use_fwd else 'warm  '
        pbar.set_postfix({
            'phase': phase_label,
            'lc':    f'{v_lc:.5f}',
            'fwd':   f'{v_fwd:.4f}' if use_fwd else '—',
            'beta':  f'{current_beta:.5f}' if use_fwd else '—',
            'best':  f'{best_val:.5f}',
            'pat':   no_improve,
        })

        if no_improve >= ES_PATIENCE:
            print(f'\n  Early stopping at epoch {epoch} (best={best_val:.5f})')
            break

    ckpt = torch.load(checkpoint_path, weights_only=True)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    return tr_lc, vl_lc, tr_fwd, vl_fwd


# ── Plots ─────────────────────────────────────────────────────────────────────

def save_tandem_loss_plot(
    tr_lc, vl_lc, tr_fwd, vl_fwd, warm_epochs, save_path
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(tr_lc) + 1)
    axes[0].plot(epochs, tr_lc, label='Train LC')
    axes[0].plot(epochs, vl_lc, label='Val LC')
    axes[0].axvline(warm_epochs, color='k', linestyle='--', alpha=0.5, label='Tandem start')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE (normalized log-LC)')
    axes[0].set_title('Tandem — LC Supervision Loss')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs[warm_epochs:], tr_fwd[warm_epochs:], label='Train Fwd')
    axes[1].plot(epochs[warm_epochs:], vl_fwd[warm_epochs:], label='Val Fwd')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE (dB²)')
    axes[1].set_title('Tandem — Forward-Consistency Loss')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path    = os.path.join(project_root, 'data', 'dataset_realistic.pkl')
    results_dir  = os.path.join(project_root, 'results')
    figures_dir  = os.path.join(results_dir, 'figures')
    tandem_ckpt  = os.path.join(results_dir, 'tandem_best.pt')
    os.makedirs(figures_dir, exist_ok=True)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}')
    print('Using differentiable ABCD synthesis for forward-consistency loss (exact physics).')

    # ── Load dataset ──────────────────────────────────────────────────────────
    with open(data_path, 'rb') as f:
        ds = pickle.load(f)
    n_total = len(ds['N'])
    print(f'Loaded {n_total} samples')

    # ── Train/val/test split (same seed as train_mlp.py for comparability) ───
    all_idx = np.arange(n_total)
    train_idx, temp_idx = train_test_split(
        all_idx, test_size=0.20, stratify=ds['N'], random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=ds['N'][temp_idx], random_state=42
    )
    print(f'Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}')

    # ── Fit scalers ───────────────────────────────────────────────────────────
    y_log = ds['y_log']

    # X scaler on X_full (207-dim)
    x_scaler = StandardScaler().fit(ds['X_full'][train_idx])

    # Per-N y scalers on log10(LC)
    def fit_y_scaler(mask_train, n_cols):
        s = StandardScaler().fit(y_log[mask_train, :n_cols])
        return s

    mask3_tr = ds['N'][train_idx] == 3
    mask4_tr = ds['N'][train_idx] == 4
    mask5_tr = ds['N'][train_idx] == 5

    y_scaler_3 = fit_y_scaler(train_idx[mask3_tr], 6)
    y_scaler_4 = fit_y_scaler(train_idx[mask4_tr], 8)
    y_scaler_5 = fit_y_scaler(train_idx[mask5_tr], 10)

    # Differentiable inverse transforms for gradient path
    inv_3 = DifferentiableInverseTransform(y_scaler_3.mean_, y_scaler_3.scale_)
    inv_4 = DifferentiableInverseTransform(y_scaler_4.mean_, y_scaler_4.scale_)
    inv_5 = DifferentiableInverseTransform(y_scaler_5.mean_, y_scaler_5.scale_)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    def make_loader(idx, shuffle):
        dataset = FilterDataset(
            idx, ds, x_scaler, y_scaler_3, y_scaler_4, y_scaler_5,
            use_scalar_x=False,
        )
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader   = make_loader(val_idx,   shuffle=False)
    test_loader  = make_loader(test_idx,  shuffle=False)

    # ── Train ─────────────────────────────────────────────────────────────────
    model = TandemInverseMLP(input_dim=207, dropout=0.2)
    print(f'\n=== Phase 1: warm-start for {WARM_EPOCHS} epochs (LC loss only) ===')
    print(f'=== Phase 2: tandem for up to {MAX_EPOCHS - WARM_EPOCHS} more epochs '
          f'(α={ALPHA}, β ramp 0→{BETA_TARGET} over {BETA_RAMP_EPOCHS} epochs) ===\n')

    tr_lc, vl_lc, tr_fwd, vl_fwd = train_tandem(
        model, inv_3, inv_4, inv_5,
        train_loader, val_loader, device, tandem_ckpt,
    )

    # ── Save with full checkpoint (matching mlp_realistic format) ─────────────
    torch.save({
        'model_state_dict': model.state_dict(),
        'y_scaler_3_mean':  y_scaler_3.mean_,
        'y_scaler_3_std':   y_scaler_3.scale_,
        'y_scaler_4_mean':  y_scaler_4.mean_,
        'y_scaler_4_std':   y_scaler_4.scale_,
        'y_scaler_5_mean':  y_scaler_5.mean_,
        'y_scaler_5_std':   y_scaler_5.scale_,
        'x_scaler_mean':    x_scaler.mean_,
        'x_scaler_std':     x_scaler.scale_,
        'warm_epochs':      WARM_EPOCHS,
        'alpha':            ALPHA,
        'beta':             BETA_TARGET,
    }, tandem_ckpt)
    print(f'\nCheckpoint saved: {tandem_ckpt}')

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print('\nEvaluating on test set...')
    y_pred_log, y_true_log, meta = predict_all(
        model, test_loader, device, y_scaler_3, y_scaler_4, y_scaler_5
    )
    results = evaluate_model(y_pred_log, y_true_log, meta)
    header = (f"{'Model':<18} | {'N':>2} | {'comp_mse(log)':>13} | "
              f"{'comp_mae(log)':>13} | {'r2_mean':>8} | {'rt_mse(dB²)':>12}")
    sep = '-' * len(header)
    print('\n' + sep)
    print(header)
    print(sep)
    for N in [3, 4, 5]:
        if N not in results:
            continue
        r = results[N]
        print(f"{'Tandem':<18} | {N:>2} | "
              f"{r['comp_mse']:>13.6f} | {r['comp_mae']:>13.6f} | "
              f"{r['r2_mean']:>8.4f} | {r['rt_mse']:>12.4f}")
    print(sep)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print('\nSaving plots...')
    save_tandem_loss_plot(
        tr_lc, vl_lc, tr_fwd, vl_fwd, WARM_EPOCHS,
        os.path.join(figures_dir, 'tandem_loss.png'),
    )
    plot_scatter(
        y_pred_log, y_true_log, meta['N'],
        os.path.join(figures_dir, 'tandem_scatter.png'),
    )
    rt_samples = build_roundtrip_samples(y_pred_log, meta)
    plot_roundtrip(rt_samples, os.path.join(figures_dir, 'tandem_roundtrip.png'),
                   model_label='Tandem')

    for name in ['tandem_loss.png', 'tandem_scatter.png', 'tandem_roundtrip.png']:
        print(f'  Saved {os.path.join(figures_dir, name)}')

    print('\nPhase 4b complete.')


if __name__ == '__main__':
    main()
