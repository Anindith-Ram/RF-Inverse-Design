"""
MLP inverse model training on the OTFL301v2 LC dataset.

Trains InverseMLP (207-dim input) and SpecsOnlyMLP (5-dim input) on
dataset_otfl301v2.pkl. Targets are log10(LC) values (continuous regression).

Key differences from the realistic-dataset version:
  - Loads dataset_otfl301v2.pkl (N∈{3,5} only), trains on y_log (log10 scale)
  - Two N-specific heads: N=3 (6 outputs), N=5 (10)
  - Two y-scalers, one per N group
  - Round-trip evaluation uses parasitic-aware ABCD synthesis (honest eval)
  - X_full is 207-dim (5 scalar + 101×S21 + 101×S11)

Usage:
    source rf_env/bin/activate
    python training/train_mlp.py
"""

import os
import sys
import random
import pickle
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mlp import InverseMLP, SpecsOnlyMLP
from evaluation.metrics import (
    component_mse, component_mae, r2_per_component,
    roundtrip_mse_lc, FREQ_HZ_OTFL301 as FREQ_HZ,
)
from evaluation.visualize import (
    plot_loss_curves, plot_scatter, plot_roundtrip, plot_r2_bars,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

BATCH_SIZE   = 256
MAX_EPOCHS   = 300
LR           = 1e-3
WEIGHT_DECAY = 1e-4
LR_PATIENCE  = 15
LR_FACTOR    = 0.5
ES_PATIENCE  = 30
RT_THRESHOLD = 5.0   # dB² — honest eval floor is ~0 for perfect predictor


# ── Dataset ───────────────────────────────────────────────────────────────────

class FilterDataset(Dataset):
    """
    PyTorch Dataset for the OTFL301v2 LC filter dataset (N∈{3,5}).

    Training targets are normalized y_log (log10 LC values).
    Stores Q_L/Q_C and parasitic metadata for honest round-trip evaluation.
    S-param targets are at X_full[5:106] (S21) and [106:207] (S11).

    Args:
        indices:     sample indices into the full dataset
        dataset:     full dataset dict from dataset_otfl301v2.pkl
        x_scaler:    fitted StandardScaler for X features
        y_scaler_3:  fitted StandardScaler for N=3 log10-LC (6-dim)
        y_scaler_4:  unused (kept for API compatibility)
        y_scaler_5:  fitted StandardScaler for N=5 log10-LC (10-dim)
        use_scalar_x: if True, use X_scalar (5-dim) instead of X_full (207-dim)
    """

    def __init__(
        self,
        indices: np.ndarray,
        dataset: dict,
        x_scaler: StandardScaler,
        y_scaler_3: StandardScaler,
        y_scaler_4: StandardScaler,
        y_scaler_5: StandardScaler,
        use_scalar_x: bool = False,
    ):
        x_key  = 'X_scalar' if use_scalar_x else 'X_full'
        X_raw  = dataset[x_key][indices]
        y_log  = dataset['y_log'][indices]      # log10(LC), NaN-padded
        y_lc   = dataset['y'][indices]          # actual LC in SI, NaN-padded
        N_arr  = dataset['N'][indices]

        self.N      = torch.from_numpy(N_arr).long()
        self.fc_GHz = torch.from_numpy(dataset['fc_GHz'][indices]).float()
        self.fbw    = torch.from_numpy(dataset['fbw'][indices]).float()

        self.X = torch.from_numpy(x_scaler.transform(X_raw)).float()

        # Normalize y_log per N group; store as (n, 10) with NaN → 0
        y_norm = np.full_like(y_log, np.nan)
        mask3  = N_arr == 3
        mask5  = N_arr == 5
        if mask3.any():
            y_norm[mask3, :6]  = y_scaler_3.transform(y_log[mask3, :6])
        if mask5.any():
            y_norm[mask5, :10] = y_scaler_5.transform(y_log[mask5, :10])

        self.y_norm = torch.from_numpy(np.nan_to_num(y_norm, nan=0.0)).float()
        self.y_log  = torch.from_numpy(y_log).float()   # for eval (inverse transform)
        self.y_lc   = torch.from_numpy(y_lc).float()    # actual LC for diagnostics

        # S-param targets from X_full (always 207-dim regardless of use_scalar_x)
        x_full = dataset['X_full'][indices]
        self.s21_target = torch.from_numpy(x_full[:, 5:106]).float()
        self.s11_target = torch.from_numpy(x_full[:, 106:207]).float()

        # Q metadata for round-trip evaluation
        self.Q_L = torch.from_numpy(dataset['Q_L'][indices]).float()  # (n, N_max), NaN-padded
        self.Q_C = torch.from_numpy(dataset['Q_C'][indices]).float()

        # Parasitic metadata for honest eval
        self.k_m       = torch.from_numpy(dataset['k_m'][indices]).float()       # (n, N-1)
        self.C_sub_frac = torch.from_numpy(dataset['C_sub_frac'][indices]).float()  # (n, N)
        self.alpha_C   = torch.from_numpy(dataset['alpha_C'][indices]).float()   # (n, N)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'x':           self.X[idx],
            'y_norm':      self.y_norm[idx],
            'y_log':       self.y_log[idx],
            'y_lc':        self.y_lc[idx],
            'N':           self.N[idx],
            'fc_GHz':      self.fc_GHz[idx],
            'fbw':         self.fbw[idx],
            's21_target':  self.s21_target[idx],
            's11_target':  self.s11_target[idx],
            'Q_L':         self.Q_L[idx],
            'Q_C':         self.Q_C[idx],
            'k_m':         self.k_m[idx],
            'C_sub_frac':  self.C_sub_frac[idx],
            'alpha_C':     self.alpha_C[idx],
        }


# ── Training ──────────────────────────────────────────────────────────────────

def compute_batch_loss(
    model: nn.Module,
    batch: dict,
    device: torch.device,
) -> torch.Tensor:
    """
    MSE loss on normalized y_log using N-specific heads.

    N=3 → head_3 vs y_norm[:,0:6]
    N=4 → head_4 vs y_norm[:,0:8]
    N=5 → head_5 vs y_norm[:,0:10]
    Loss is the mean of per-N MSEs (equal weighting across orders).
    """
    x      = batch['x'].to(device)
    y_norm = batch['y_norm'].to(device)
    N_batch = batch['N']

    out_3, out_4, out_5 = model(x)

    losses  = []
    mask3   = N_batch == 3
    mask5   = N_batch == 5

    if mask3.any():
        losses.append(nn.functional.mse_loss(out_3[mask3], y_norm[mask3, :6]))
    if mask5.any():
        losses.append(nn.functional.mse_loss(out_5[mask5], y_norm[mask5, :10]))

    return torch.stack(losses).mean()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            loss = compute_batch_loss(model, batch, device)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
    return total_loss / len(loader)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    model_name: str,
    checkpoint_path: str,
) -> tuple[list, list]:
    """AdamW + ReduceLROnPlateau + early stopping."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=LR_PATIENCE, factor=LR_FACTOR,
    )

    best_val = float('inf')
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    pbar = tqdm(range(1, MAX_EPOCHS + 1), desc=f'Training {model_name}')
    for epoch in pbar:
        t_loss = run_epoch(model, train_loader, device, optimizer)
        v_loss = run_epoch(model, val_loader, device)
        scheduler.step(v_loss)

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        if v_loss < best_val:
            best_val = v_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1

        pbar.set_postfix({
            'train': f'{t_loss:.5f}',
            'val':   f'{v_loss:.5f}',
            'best':  f'{best_val:.5f}',
            'pat':   epochs_no_improve,
        })

        if epochs_no_improve >= ES_PATIENCE:
            print(f'\n  Early stopping at epoch {epoch} (best val={best_val:.5f})')
            break

    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    return train_losses, val_losses


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_all(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_scaler_3: StandardScaler,
    y_scaler_4: StandardScaler,
    y_scaler_5: StandardScaler,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Run inference; return denormalized log10-LC predictions and ground truth.

    Returns:
        y_pred_log: (n, 10) predicted log10(LC), NaN for undefined positions
        y_true_log: (n, 10) ground-truth log10(LC), NaN for undefined positions
        meta: dict with N, fc_GHz, fbw, s21_target, s11_target, Q_L, Q_C
    """
    model.eval()
    all_pred, all_true = [], []
    meta_keys = ['N', 'fc_GHz', 'fbw', 's21_target', 's11_target', 'Q_L', 'Q_C',
                 'k_m', 'C_sub_frac', 'alpha_C', 'y_lc']
    meta = {k: [] for k in meta_keys}

    with torch.no_grad():
        for batch in loader:
            x       = batch['x'].to(device)
            N_batch = batch['N'].numpy()
            out_3, out_4, out_5 = model(x)

            pred_log = np.full((len(x), 10), np.nan)
            mask3 = N_batch == 3
            mask5 = N_batch == 5

            if mask3.any():
                pred_log[mask3, :6]  = y_scaler_3.inverse_transform(
                    out_3[mask3].cpu().numpy()
                )
            if mask5.any():
                pred_log[mask5, :10] = y_scaler_5.inverse_transform(
                    out_5[mask5].cpu().numpy()
                )

            all_pred.append(pred_log)
            all_true.append(batch['y_log'].numpy())
            for k in meta_keys:
                meta[k].append(batch[k].numpy())

    y_pred_log = np.concatenate(all_pred, axis=0)
    y_true_log = np.concatenate(all_true, axis=0)
    for k in meta_keys:
        meta[k] = np.concatenate(meta[k], axis=0)

    return y_pred_log, y_true_log, meta


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(
    y_pred_log: np.ndarray,
    y_true_log: np.ndarray,
    meta: dict,
) -> dict:
    """
    Compute metrics per N in {3, 5}.

    Component MSE/MAE/R² are in log10-space (what the model is trained to predict).
    Round-trip MSE is in dB² (physical S-param error, parasitic-aware, honest eval).

    Returns:
        results: {N: {comp_mse, comp_mae, r2, r2_mean, rt_mse, acc_frac}}
    """
    results = {}
    for N in [3, 5]:
        mask = meta['N'] == N
        if mask.sum() == 0:
            continue

        n_lc = 2 * N   # number of valid LC columns
        yp = y_pred_log[mask, :n_lc]
        yt = y_true_log[mask, :n_lc]

        c_mse  = component_mse(yp, yt)
        c_mae  = component_mae(yp, yt)
        r2     = r2_per_component(yp, yt)
        r2_mean = float(np.nanmean(r2))

        # Convert log10 → SI for round-trip synthesis
        L_pred_all = 10.0 ** y_pred_log[mask, 0::2][:, :N]   # (n_N, N)
        C_pred_all = 10.0 ** y_pred_log[mask, 1::2][:, :N]
        # Ground-truth LC for computing coupling_M and C_sub
        y_lc_all = meta['y_lc'][mask]
        L_true_all = y_lc_all[:, 0::2][:, :N]
        C_true_all = y_lc_all[:, 1::2][:, :N]

        Q_L_all    = meta['Q_L'][mask, :N]
        Q_C_all    = meta['Q_C'][mask, :N]
        s21_all    = meta['s21_target'][mask]
        s11_all    = meta['s11_target'][mask]
        fc_all     = meta['fc_GHz'][mask]
        k_m_all    = meta['k_m'][mask, :N - 1]
        csf_all    = meta['C_sub_frac'][mask, :N]
        alphaC_all = meta['alpha_C'][mask, :N]

        rt_vals = []
        for i in range(mask.sum()):
            coup_M = k_m_all[i] * np.minimum(L_true_all[i, :-1], L_true_all[i, 1:])
            C_sub  = csf_all[i] * C_true_all[i]
            rt = roundtrip_mse_lc(
                L_pred=L_pred_all[i],
                C_pred=C_pred_all[i],
                N=N,
                fc_GHz=float(fc_all[i]),
                Q_L=Q_L_all[i],
                Q_C=Q_C_all[i],
                target_s21_db=s21_all[i],
                target_s11_db=s11_all[i],
                freq_hz=FREQ_HZ,
                coupling_M=coup_M,
                C_sub=C_sub,
                alpha_C=alphaC_all[i],
            )
            if not np.isinf(rt):
                rt_vals.append(rt)

        rt_arr = np.array(rt_vals) if rt_vals else np.array([float('inf')])
        acc_frac = float(np.mean(rt_arr < RT_THRESHOLD))

        results[N] = {
            'comp_mse': c_mse,
            'comp_mae': c_mae,
            'r2':       r2,
            'r2_mean':  r2_mean,
            'rt_mse':   float(np.mean(rt_arr)),
            'acc_frac': acc_frac,
        }

    return results


def build_roundtrip_samples(
    y_pred_log: np.ndarray,
    meta: dict,
    n_per_N: int = 2,
) -> list:
    """
    Build sample dicts for the round-trip overlay plot.

    Selects n_per_N random samples per N. Synthesizes S21 from predicted LC
    using ground-truth Q and parasitics (honest eval).
    """
    from evaluation.metrics import synthesize_from_lc, roundtrip_mse_lc
    rng = np.random.RandomState(0)
    samples = []

    for N in [3, 5]:
        mask = meta['N'] == N
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        chosen = rng.choice(len(idxs), size=min(n_per_N, len(idxs)), replace=False)

        for c in chosen:
            i = idxs[c]
            L_pred = 10.0 ** y_pred_log[i, 0::2][:N]
            C_pred = 10.0 ** y_pred_log[i, 1::2][:N]
            fc     = float(meta['fc_GHz'][i])
            Q_L    = meta['Q_L'][i, :N]
            Q_C    = meta['Q_C'][i, :N]

            y_lc_i   = meta['y_lc'][i]
            L_true   = y_lc_i[0::2][:N]
            C_true   = y_lc_i[1::2][:N]
            coup_M   = meta['k_m'][i, :N - 1] * np.minimum(L_true[:-1], L_true[1:])
            C_sub    = meta['C_sub_frac'][i, :N] * C_true
            alpha_C  = meta['alpha_C'][i, :N]

            s21_pred, _ = synthesize_from_lc(L_pred, C_pred, N, fc, Q_L, Q_C, FREQ_HZ,
                                              coupling_M=coup_M, C_sub=C_sub, alpha_C=alpha_C)
            if s21_pred is None:
                s21_pred = np.zeros(101)

            rt = roundtrip_mse_lc(
                L_pred, C_pred, N, fc, Q_L, Q_C,
                meta['s21_target'][i], meta['s11_target'][i], FREQ_HZ,
                coupling_M=coup_M, C_sub=C_sub, alpha_C=alpha_C,
            )
            samples.append({
                's21_target': meta['s21_target'][i],
                's21_pred':   s21_pred,
                'N':          N,
                'ripple_dB':  0.0,
                'fc_GHz':     fc,
                'fbw':        float(meta['fbw'][i]),
                'rt_mse':     rt,
            })

    return samples


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_results_table(results_full: dict, results_scalar: dict) -> None:
    """Print benchmark table for all models and filter orders."""
    header = (f"{'Model':<18} | {'N':>2} | {'comp_mse(log)':>13} | "
              f"{'comp_mae(log)':>13} | {'r2_mean':>8} | {'rt_mse(dB²)':>12} | {'acc_frac':>9}")
    sep = '-' * len(header)
    print('\n' + sep)
    print(header)
    print(sep)
    for label, res in [('MLP (full)', results_full), ('MLP (scalar)', results_scalar)]:
        for N in [3, 5]:
            if N not in res:
                continue
            r = res[N]
            print(
                f"{label:<18} | {N:>2} | "
                f"{r['comp_mse']:>13.6f} | "
                f"{r['comp_mae']:>13.6f} | "
                f"{r['r2_mean']:>8.4f} | "
                f"{r['rt_mse']:>12.4f} | "
                f"{r['acc_frac']:>9.4f}"
            )
    print(sep)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path    = os.path.join(project_root, 'data', 'dataset_otfl301v2.pkl')
    results_dir  = os.path.join(project_root, 'results')
    figures_dir  = os.path.join(results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load dataset ──────────────────────────────────────────────────────────
    with open(data_path, 'rb') as f:
        ds = pickle.load(f)
    n_total = len(ds['N'])
    print(f'Loaded {n_total} samples from {data_path}')
    print(f'N distribution: { {n: int((ds["N"]==n).sum()) for n in [3,5]} }')

    # ── Stratified 80/10/10 split by N (N∈{3,5} only) ───────────────────────
    all_idx = np.arange(n_total)
    train_idx, temp_idx = train_test_split(
        all_idx, test_size=0.20, stratify=ds['N'], random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=ds['N'][temp_idx], random_state=42
    )
    print(f'Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}')

    # ── Fit scalers on training set only ─────────────────────────────────────
    x_full_scaler   = StandardScaler().fit(ds['X_full'][train_idx])
    x_scalar_scaler = StandardScaler().fit(ds['X_scalar'][train_idx])

    train_N   = ds['N'][train_idx]
    train_log = ds['y_log'][train_idx]

    y_scaler_3 = StandardScaler().fit(train_log[train_N == 3, :6])
    # Dummy scaler for N=4 (API compatibility; no N=4 samples in this dataset)
    y_scaler_4 = StandardScaler().fit(train_log[train_N == 3, :6])
    y_scaler_5 = StandardScaler().fit(train_log[train_N == 5, :10])

    # ── Build DataLoaders ─────────────────────────────────────────────────────
    def make_loaders(use_scalar_x, x_scaler):
        kw = dict(use_scalar_x=use_scalar_x)
        scalers = (x_scaler, y_scaler_3, y_scaler_4, y_scaler_5)
        train_ds = FilterDataset(train_idx, ds, *scalers, **kw)
        val_ds   = FilterDataset(val_idx,   ds, *scalers, **kw)
        test_ds  = FilterDataset(test_idx,  ds, *scalers, **kw)
        train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
        val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_ld  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        return train_ld, val_ld, test_ld

    full_train, full_val, full_test = make_loaders(False, x_full_scaler)
    scal_train, scal_val, scal_test = make_loaders(True,  x_scalar_scaler)

    # ── Train InverseMLP (full) ───────────────────────────────────────────────
    print('\n=== Training InverseMLP (full 207-dim input) ===')
    mlp_full = InverseMLP(input_dim=207)
    ckpt_full = os.path.join(results_dir, 'mlp_otfl301v2_best.pt')
    t_losses_full, v_losses_full = train_model(
        mlp_full, full_train, full_val, device, 'MLP-full', ckpt_full
    )

    # ── Train SpecsOnlyMLP (scalar) ───────────────────────────────────────────
    print('\n=== Training SpecsOnlyMLP (5-dim scalar input) ===')
    mlp_scalar = SpecsOnlyMLP()
    ckpt_scalar = os.path.join(results_dir, 'mlp_otfl301v2_scalar_best.pt')
    t_losses_scal, v_losses_scal = train_model(
        mlp_scalar, scal_train, scal_val, device, 'MLP-scalar', ckpt_scalar
    )

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print('\nEvaluating on test set...')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        yp_full, yt_full, meta_full = predict_all(
            mlp_full, full_test, device, y_scaler_3, y_scaler_4, y_scaler_5
        )
        res_full = evaluate_model(yp_full, yt_full, meta_full)

        yp_scal, yt_scal, meta_scal = predict_all(
            mlp_scalar, scal_test, device, y_scaler_3, y_scaler_4, y_scaler_5
        )
        res_scal = evaluate_model(yp_scal, yt_scal, meta_scal)

    print_results_table(res_full, res_scal)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print('\nSaving plots...')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, t_l, v_l, label in [
        (axes[0], t_losses_full,  v_losses_full,  'MLP (full)'),
        (axes[1], t_losses_scal, v_losses_scal, 'MLP (scalar)'),
    ]:
        epochs = range(1, len(t_l) + 1)
        ax.plot(epochs, t_l, label='Train')
        ax.plot(epochs, v_l, label='Val')
        ax.set_title(label)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss (normalized log₁₀ LC)')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle('Training Loss Curves — OTFL301v2 Dataset', fontsize=13)
    plt.tight_layout()
    loss_path = os.path.join(figures_dir, 'mlp_otfl301v2_loss.png')
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {loss_path}')

    scatter_path = os.path.join(figures_dir, 'mlp_otfl301v2_scatter.png')
    plot_scatter(yp_full, yt_full, meta_full['N'], scatter_path)
    print(f'  Saved {scatter_path}')

    rt_samples = build_roundtrip_samples(yp_full, meta_full)
    rt_path = os.path.join(figures_dir, 'mlp_otfl301v2_roundtrip.png')
    plot_roundtrip(rt_samples, rt_path, model_label='MLP (full, OTFL301v2)')
    print(f'  Saved {rt_path}')

    r2_data = {
        'MLP (full)':   {N: res_full[N]['r2']  for N in res_full},
        'MLP (scalar)': {N: res_scal[N]['r2']  for N in res_scal},
    }
    r2_path = os.path.join(figures_dir, 'mlp_otfl301v2_r2.png')
    plot_r2_bars(r2_data, r2_path)
    print(f'  Saved {r2_path}')

    print('\nTraining complete.')


if __name__ == '__main__':
    main()
