"""
Phase 4a: Forward surrogate model training.

Trains ForwardMLP: (LC, Q, specs) → (S21, S11) on dataset_realistic.pkl.
The forward mapping is unique and well-posed, so this trains cleanly and
serves as a prerequisite for Phase 4b (tandem) and Phase 4d (CMA-ES).

Normalization buffers are embedded into the saved checkpoint so ForwardMLP
is self-contained when loaded by train_tandem.py or experiments/cmaes_design.py.

Usage:
    source rf_env/bin/activate
    python training/train_forward.py

Outputs:
    results/forward_model_best.pt   — checkpoint with weights + scaler buffers
    results/figures/fwd_loss.png    — train/val loss curves
    results/figures/fwd_scatter.png — predicted vs true S21 at 3 freq points
"""

import os
import sys
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.forward_model import ForwardMLP, build_forward_input, FORWARD_INPUT_DIM, FORWARD_OUTPUT_DIM
from evaluation.metrics import r2_per_component

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

BATCH_SIZE   = 512
MAX_EPOCHS   = 300
LR           = 1e-3
WEIGHT_DECAY = 1e-4
LR_PATIENCE  = 15
LR_FACTOR    = 0.5
ES_PATIENCE  = 30


# ── Dataset ───────────────────────────────────────────────────────────────────

class ForwardDataset(Dataset):
    """
    Dataset for forward model training.

    X: 24-dim [log10(LC), Q_L, Q_C, fc_GHz, fbw, N3_flag, N5_flag] — pre-normalized
    y: 202-dim [S21(101), S11(101)] — pre-normalized

    Both are normalized by external StandardScalers fitted on the training set.
    """

    def __init__(
        self,
        indices:   np.ndarray,
        X_fwd_raw: np.ndarray,   # (n_total, 24) un-normalized forward inputs
        y_sp_raw:  np.ndarray,   # (n_total, 202) un-normalized S-param targets
        x_scaler:  StandardScaler,
        y_scaler:  StandardScaler,
    ):
        X_norm = x_scaler.transform(X_fwd_raw[indices])
        y_norm = y_scaler.transform(y_sp_raw[indices])
        self.X = torch.from_numpy(X_norm).float()
        self.y = torch.from_numpy(y_norm).float()

        # Keep raw S-params for evaluation (in dB)
        self.y_raw = torch.from_numpy(y_sp_raw[indices]).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {'x': self.X[idx], 'y': self.y[idx], 'y_raw': self.y_raw[idx]}


# ── Training ──────────────────────────────────────────────────────────────────

def run_epoch(
    model:     ForwardMLP,
    loader:    DataLoader,
    device:    torch.device,
    optimizer: torch.optim.Optimizer = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total = 0.0
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
    return total / len(loader)


def train_model(
    model:       ForwardMLP,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
    checkpoint_path: str,
) -> tuple[list, list]:
    """AdamW + ReduceLROnPlateau + early stopping."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=LR_PATIENCE, factor=LR_FACTOR,
    )

    best_val = float('inf')
    no_improve = 0
    train_losses, val_losses = [], []

    pbar = tqdm(range(1, MAX_EPOCHS + 1), desc='Training ForwardMLP')
    for epoch in pbar:
        t_loss = run_epoch(model, train_loader, device, optimizer)
        v_loss = run_epoch(model, val_loader,   device)
        scheduler.step(v_loss)

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        if v_loss < best_val:
            best_val  = v_loss
            no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            no_improve += 1

        pbar.set_postfix({
            'train': f'{t_loss:.5f}', 'val': f'{v_loss:.5f}',
            'best':  f'{best_val:.5f}', 'pat': no_improve,
        })

        if no_improve >= ES_PATIENCE:
            print(f'\n  Early stopping at epoch {epoch} (best val={best_val:.5f})')
            break

    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    return train_losses, val_losses


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_forward(
    model:    ForwardMLP,
    loader:   DataLoader,
    device:   torch.device,
    y_scaler: StandardScaler,
) -> dict:
    """
    Evaluate forward model on a DataLoader.

    Computes in true dB space (after inverse-transforming predictions):
        s21_mse_db2: MSE of S21 predictions (dB²)
        s11_mse_db2: MSE of S11 predictions (dB²)
        r2_s21:      mean R² across 101 frequency points for S21
        r2_s11:      mean R² across 101 frequency points for S11
    """
    model.eval()
    all_pred_db, all_true_db = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            pred_norm = model(x).cpu().numpy()
            pred_db   = y_scaler.inverse_transform(pred_norm)
            all_pred_db.append(pred_db)
            all_true_db.append(batch['y_raw'].numpy())

    pred_db = np.concatenate(all_pred_db, axis=0)
    true_db = np.concatenate(all_true_db, axis=0)

    s21_pred, s21_true = pred_db[:, :101],  true_db[:, :101]
    s11_pred, s11_true = pred_db[:, 101:],  true_db[:, 101:]

    s21_mse = float(np.mean((s21_pred - s21_true) ** 2))
    s11_mse = float(np.mean((s11_pred - s11_true) ** 2))

    # R² per frequency point, then average
    r2_s21_arr = r2_per_component(s21_pred, s21_true)
    r2_s11_arr = r2_per_component(s11_pred, s11_true)

    return {
        's21_mse_db2': s21_mse,
        's11_mse_db2': s11_mse,
        'r2_s21':      float(np.nanmean(r2_s21_arr)),
        'r2_s11':      float(np.nanmean(r2_s11_arr)),
        'pred_db':     pred_db,
        'true_db':     true_db,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def save_loss_plot(train_losses, val_losses, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Train')
    ax.plot(epochs, val_losses,   label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss (normalized S-params)')
    ax.set_title('Forward Surrogate — Training Loss')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_scatter_plot(pred_db, true_db, save_path):
    """
    3×2 scatter: predicted vs true S-params at 6 representative frequency points.
    Row 0: S21.  Row 1: S11.  Columns: 40, 65, 90 GHz.
    """
    FREQ_GHZ = np.linspace(40, 90, 101)
    freq_targets = [40, 65, 90]
    fi = [np.argmin(np.abs(FREQ_GHZ - f)) for f in freq_targets]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for col, (fidx, fghz) in enumerate(zip(fi, freq_targets)):
        for row, (name, offset) in enumerate([('S21', 0), ('S11', 101)]):
            ax = axes[row][col]
            t = true_db[:,  offset + fidx]
            p = pred_db[:, offset + fidx]
            ax.scatter(t, p, s=3, alpha=0.3)
            lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
            margin = (hi - lo) * 0.05
            ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                    'k--', linewidth=1.0, alpha=0.6)
            r2 = 1 - np.sum((t - p)**2) / max(np.sum((t - t.mean())**2), 1e-12)
            ax.set_title(f'{name} @ {fghz} GHz  R²={r2:.4f}')
            ax.set_xlabel(f'True {name} (dB)')
            ax.set_ylabel(f'Pred {name} (dB)')
            ax.grid(True, alpha=0.3)

    fig.suptitle('Forward Surrogate — Predicted vs True S-params', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_curve_overlay_plot(pred_db, true_db, N_arr, save_path):
    """
    3-panel overlay: one random sample per N, target (solid) vs predicted (dashed).
    """
    FREQ_GHZ = np.linspace(40, 90, 101)
    rng = np.random.RandomState(7)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax_idx, N in enumerate([3, 4, 5]):
        ax = axes[ax_idx]
        idxs = np.where(N_arr == N)[0]
        if len(idxs) == 0:
            ax.set_visible(False)
            continue
        i = rng.choice(idxs)
        ax.plot(FREQ_GHZ, true_db[i, :101],  'b-',  linewidth=2.0, label='Target S21')
        ax.plot(FREQ_GHZ, pred_db[i, :101],  'b--', linewidth=1.5, label='Pred S21')
        ax.plot(FREQ_GHZ, true_db[i, 101:],  'r-',  linewidth=2.0, label='Target S11', alpha=0.7)
        ax.plot(FREQ_GHZ, pred_db[i, 101:],  'r--', linewidth=1.5, label='Pred S11',   alpha=0.7)
        ax.set_title(f'N={N}')
        ax.set_xlabel('Freq (GHz)')
        ax.set_ylabel('S-param (dB)')
        ax.set_xlim(40, 90)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Forward Surrogate — S-param Curve Overlay', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path    = os.path.join(project_root, 'data', 'dataset_realistic.pkl')
    results_dir  = os.path.join(project_root, 'results')
    figures_dir  = os.path.join(results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load dataset ──────────────────────────────────────────────────────────
    with open(data_path, 'rb') as f:
        ds = pickle.load(f)
    n_total = len(ds['N'])
    print(f'Loaded {n_total} samples')

    # ── Build forward model inputs ────────────────────────────────────────────
    X_fwd = build_forward_input(
        y_log  = ds['y_log'],
        Q_L    = ds['Q_L'],
        Q_C    = ds['Q_C'],
        fc_GHz = ds['fc_GHz'],
        fbw    = ds['fbw'],
        N      = ds['N'],
    )
    # S-param targets: X_full[5:207] = [S21(101), S11(101)]
    y_sp = ds['X_full'][:, 5:207].astype(np.float64)

    # Clip S11 at -60 dB floor. S11 is derived via the lossless identity
    # S11 = sqrt(1 - |S21|^2), which gives -120 dB at the passband peak where
    # S21 ≈ 0 dB. Values below -60 dB are physically indistinguishable (perfect
    # matching) and create unreliable training targets that dominate MSE loss.
    S11_FLOOR_DB = -60.0
    y_sp[:, 101:] = np.maximum(y_sp[:, 101:], S11_FLOOR_DB)

    assert X_fwd.shape == (n_total, 24), f'X_fwd shape {X_fwd.shape}'
    assert y_sp.shape  == (n_total, 202), f'y_sp shape {y_sp.shape}'
    print(f'X_fwd: {X_fwd.shape}  y_sp: {y_sp.shape}')
    print(f'S11 clipped to floor {S11_FLOOR_DB} dB  '
          f'(fraction clipped: {(ds["X_full"][:, 106:207] < S11_FLOOR_DB).mean():.1%})')

    # ── Stratified split by N ─────────────────────────────────────────────────
    all_idx = np.arange(n_total)
    train_idx, temp_idx = train_test_split(
        all_idx, test_size=0.20, stratify=ds['N'], random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=ds['N'][temp_idx], random_state=42
    )
    print(f'Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}')

    # ── Fit scalers ───────────────────────────────────────────────────────────
    x_scaler = StandardScaler().fit(X_fwd[train_idx])
    y_scaler = StandardScaler().fit(y_sp[train_idx])

    # ── DataLoaders ───────────────────────────────────────────────────────────
    def make_loader(idx, shuffle):
        ds_ = ForwardDataset(idx, X_fwd, y_sp, x_scaler, y_scaler)
        return DataLoader(ds_, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader   = make_loader(val_idx,   shuffle=False)
    test_loader  = make_loader(test_idx,  shuffle=False)

    # ── Train ─────────────────────────────────────────────────────────────────
    model = ForwardMLP()
    ckpt_path = os.path.join(results_dir, 'forward_model_best.pt')

    print(f'\n=== Training ForwardMLP ({FORWARD_INPUT_DIM}-dim → {FORWARD_OUTPUT_DIM}-dim) ===')
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, device, ckpt_path
    )

    # ── Embed normalization buffers in model ──────────────────────────────────
    # This makes the checkpoint self-contained: tandem/CMA-ES can call model.predict()
    # without needing external scaler objects.
    model.set_normalization(
        x_mean = x_scaler.mean_,
        x_std  = x_scaler.scale_,
        y_mean = y_scaler.mean_,
        y_std  = y_scaler.scale_,
    )
    # Re-save with buffers included
    torch.save({
        'model_state_dict': model.state_dict(),
        'x_scaler_mean': x_scaler.mean_,
        'x_scaler_std':  x_scaler.scale_,
        'y_scaler_mean': y_scaler.mean_,
        'y_scaler_std':  y_scaler.scale_,
    }, ckpt_path)
    print(f'Checkpoint saved (with normalization buffers): {ckpt_path}')

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print('\nEvaluating on test set...')
    metrics = evaluate_forward(model, test_loader, device, y_scaler)
    print(f'\nForward model test metrics:')
    print(f'  S21 MSE: {metrics["s21_mse_db2"]:.4f} dB²')
    print(f'  S11 MSE: {metrics["s11_mse_db2"]:.4f} dB²')
    print(f'  R²(S21): {metrics["r2_s21"]:.4f}')
    print(f'  R²(S11): {metrics["r2_s11"]:.4f}')

    # ── Plots ─────────────────────────────────────────────────────────────────
    print('\nSaving plots...')
    save_loss_plot(train_losses, val_losses,
                   os.path.join(figures_dir, 'fwd_loss.png'))
    save_scatter_plot(metrics['pred_db'], metrics['true_db'],
                      os.path.join(figures_dir, 'fwd_scatter.png'))
    save_curve_overlay_plot(
        metrics['pred_db'], metrics['true_db'], ds['N'][test_idx],
        os.path.join(figures_dir, 'fwd_overlay.png'),
    )

    for name in ['fwd_loss.png', 'fwd_scatter.png', 'fwd_overlay.png']:
        print(f'  Saved {os.path.join(figures_dir, name)}')

    # ── Sanity assertions ─────────────────────────────────────────────────────
    assert metrics['r2_s21'] > 0.90, f'R²(S21) too low: {metrics["r2_s21"]:.4f}'
    assert metrics['r2_s11'] > 0.80, f'R²(S11) too low: {metrics["r2_s11"]:.4f}'
    assert metrics['s21_mse_db2'] < 5.0, f'S21 MSE too high: {metrics["s21_mse_db2"]:.4f}'
    print('\nAll assertions passed.')
    print('\nPhase 4a complete.')


if __name__ == '__main__':
    main()
