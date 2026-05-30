"""
Phase 4c+: Conditional INN training for RF filter inverse design.
Joint LC+Q posterior: p(LC, Q_L, Q_C | S-params).

Trains three independent cINNs (one per N ∈ {3, 4, 5}) using maximum likelihood.
Each cINN models the full posterior p(y|x) over log10([LC, Q_L, Q_C]) given
S-param + spec inputs, allowing multiple physically valid designs to be sampled
at inference time.

Output target layout:
    y = [log10(L1), log10(C1), ..., log10(LN), log10(CN),   ← LC  (0 : 2N)
         log10(Q_L1), ..., log10(Q_LN),                      ← Q_L (2N: 3N)
         log10(Q_C1), ..., log10(Q_CN)]                      ← Q_C (3N: 4N)
    D_y = 4N  →  N=3: 12,  N=4: 16,  N=5: 20

Evaluation uses the model's own predicted Q (not ground-truth Q), giving an
honest rt_mse that holds at real inference time (EMX data, unknown Q).

Loss: pure NLL = 0.5 * ||z||² - log|det J|

Evaluation
----------
PRIMARY metric: acceptable_fraction — fraction of K=10 sampled designs with
rt_mse < 50 dB² on a 500-sample val subset. Checkpointing is on this metric.

Final test: best-of-K=50 rt_mse, mean rt_mse, acceptable fraction, diversity,
mode count via DBSCAN, posterior NLL on y_true, Q accuracy (MAE in log10-space),
and ForwardMLP cross-check.

Usage:
    source rf_env/bin/activate
    python -u training/train_inn.py

Outputs:
    results/inn_q_N{3,4,5}_best.pt     — cINN + embedder checkpoints
    results/inn_q_N{3,4,5}_log.csv     — per-epoch diagnostics
    results/inn_q_metrics_table.txt    — final benchmark table
    results/figures/inn_q_*.png        — all diagnostic plots
"""

import os
import sys
import csv
import json
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.inn import ConditionEmbedder, make_cinn, verify_bijection, fix_mps_contiguity
from evaluation.metrics import (
    synthesize_from_lc, roundtrip_mse_lc, component_mse, r2_per_component, FREQ_HZ
)
from training.train_mlp import FilterDataset

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_BLOCKS        = 8
SUBNET_DIM      = 128
COND_DIM        = 64
AFFINE_CLAMP    = 2.0
LR              = 1e-3
WEIGHT_DECAY    = 1e-5
BATCH_SIZE      = 256
MAX_EPOCHS      = 300
ES_PATIENCE     = 30
LR_PATIENCE     = 15
LR_FACTOR       = 0.5
GRAD_CLIP       = 10.0
K_EVAL_EVERY    = 10       # epochs between rt_mse evaluations
K_EVAL          = 10       # samples per val condition during training eval
K_EVAL_NSAMPLE  = 500      # val subset size for fast rt_mse eval
K_INFERENCE     = 50       # final test best-of-K
K_DIVERSITY     = 20       # samples for diversity / mode-count analysis
NLL_NAN_GUARD   = 1e6      # divergence threshold
SEED            = 42
RT_THRESHOLD    = 50.0     # dB² — "acceptable" design criterion


# ── Joint target builder ──────────────────────────────────────────────────────

def build_y_full(ds: dict, idx: np.ndarray, N_val: int) -> np.ndarray:
    """
    Build joint [log10(LC), log10(Q_L), log10(Q_C)] target.

    Returns:
        (n, 4N) float32 array
    """
    y_log_lc = ds['y_log'][idx, :2*N_val].astype(np.float64)          # (n, 2N)
    q_L = np.log10(np.clip(ds['Q_L'][idx, :N_val], 1e-6, None))       # (n, N)
    q_C = np.log10(np.clip(ds['Q_C'][idx, :N_val], 1e-6, None))       # (n, N)
    return np.concatenate([y_log_lc, q_L, q_C], axis=1).astype(np.float32)


# ── Per-N sampling: invert K latent vectors for a batch ───────────────────────

@torch.no_grad()
def sample_lc_batch(
    inn: nn.Module,
    embedder: nn.Module,
    x_norm: torch.Tensor,   # (B, 207) normalized
    K: int,
    D_y: int,
    y_scaler_mean: np.ndarray,
    y_scaler_std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Sample K candidates for a batch of B conditions.

    Returns:
        y_log_samples: (B, K, D_y) in raw (unscaled) space
            First 2N dims: log10(LC), next N: log10(Q_L), last N: log10(Q_C)
    """
    inn.eval(); embedder.eval()
    B = x_norm.shape[0]
    c = embedder(x_norm.to(device))           # (B, D_c)

    # Expand condition for K samples: (B*K, D_c)
    c_rep = c.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)

    z = torch.randn(B * K, D_y, device=device)
    y_norm_flat, _ = inn(z, c=[c_rep], rev=True)   # (B*K, D_y)

    y_norm_np = y_norm_flat.cpu().numpy().reshape(B, K, D_y)
    # Denormalize: y = y_norm * std + mean
    y_log = y_norm_np * y_scaler_std[None, None, :] + y_scaler_mean[None, None, :]
    return y_log   # (B, K, D_y)


# ── Fast rt_mse eval on a val subset ─────────────────────────────────────────

def eval_rt_mse_fast(
    inn: nn.Module,
    embedder: nn.Module,
    val_subset: dict,   # pre-extracted numpy arrays for K_EVAL_NSAMPLE samples
    K: int,
    D_y: int,
    N_val: int,
    y_scaler_mean: np.ndarray,
    y_scaler_std: np.ndarray,
    device: torch.device,
) -> tuple:
    """
    Compute best-of-K rt_mse and acceptable_fraction on val_subset.
    Uses the model's own predicted Q (not ground-truth Q).

    Returns:
        (best_of_k_rt_mse, acceptable_fraction, mean_rt_mse, median_rt_mse)
    """
    D_lc = 2 * N_val
    x_norm = torch.from_numpy(val_subset['x_norm']).float()
    # Sample in mini-batches to avoid OOM
    MBSZ = 64
    n = x_norm.shape[0]
    all_y_log = []
    for start in range(0, n, MBSZ):
        xb = x_norm[start:start+MBSZ]
        yl = sample_lc_batch(inn, embedder, xb, K, D_y,
                              y_scaler_mean, y_scaler_std, device)
        all_y_log.append(yl)
    y_log_samples = np.concatenate(all_y_log, axis=0)  # (n, K, D_y)

    rt_best = []
    rt_all  = []
    for i in range(n):
        fc  = float(val_subset['fc_GHz'][i])
        s21 = val_subset['s21_target'][i]
        s11 = val_subset['s11_target'][i]

        rts = []
        for k in range(K):
            y_samp = y_log_samples[i, k]            # (D_y,)
            lc_block  = y_samp[:D_lc]               # (2N,) interleaved L1,C1,...
            L = 10.0 ** lc_block[0::2]              # (N,)
            C = 10.0 ** lc_block[1::2]              # (N,)
            q_L_pred = 10.0 ** y_samp[D_lc:D_lc+N_val]    # (N,)
            q_C_pred = 10.0 ** y_samp[D_lc+N_val:]         # (N,)
            rt = roundtrip_mse_lc(L, C, N_val, fc,
                                  q_L_pred, q_C_pred, s21, s11, FREQ_HZ)
            rts.append(rt if np.isfinite(rt) else RT_THRESHOLD * 10)

        rt_best.append(min(rts))
        rt_all.extend(rts)

    rt_best = np.array(rt_best)
    rt_all  = np.array(rt_all)
    best_of_k   = float(np.mean(rt_best))
    acc_frac    = float(np.mean(rt_all < RT_THRESHOLD))
    mean_rt     = float(np.mean(rt_all))
    median_rt   = float(np.median(rt_all))
    return best_of_k, acc_frac, mean_rt, median_rt


# ── Training loop (per N) ─────────────────────────────────────────────────────

def train_one_cinn(
    N_val: int,
    train_idx_N: np.ndarray,
    val_idx_N: np.ndarray,
    ds: dict,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,      # 4N-dim joint scaler
    device: torch.device,
    checkpoint_path: str,
    log_path: str,
    results_dir: str,
    figures_dir: str,
) -> dict:
    """
    Train one cINN for a specific filter order N.
    y target is 4N-dimensional: [log10(LC), log10(Q_L), log10(Q_C)].

    Returns:
        summary dict with final metrics and best checkpoint info
    """
    D_lc = 2 * N_val
    D_y  = 4 * N_val
    y_scaler_mean = y_scaler.mean_.astype(np.float32)   # (4N,)
    y_scaler_std  = y_scaler.scale_.astype(np.float32)  # (4N,)

    print(f'\n{"="*60}')
    print(f'  Training cINN for N={N_val}  (D_lc={D_lc}, D_y={D_y}, '
          f'{len(train_idx_N)} train / {len(val_idx_N)} val samples)')
    print(f'{"="*60}')

    # ── Datasets / loaders ────────────────────────────────────────────────────
    # FilterDataset normalizes y_log with per-N scalers (for y_norm key).
    # We don't use batch['y_norm'] — we build y_b from y_log + Q_L + Q_C directly.
    # Pass identity 2*n scalers to satisfy FilterDataset's interface.
    def _get_dummy_scaler(n):
        d = StandardScaler()
        d.mean_  = np.zeros(2*n, dtype=np.float32)
        d.scale_ = np.ones(2*n, dtype=np.float32)
        d.var_   = np.ones(2*n, dtype=np.float32)
        d.n_features_in_ = 2*n
        return d

    train_ds = FilterDataset(
        train_idx_N, ds, x_scaler,
        _get_dummy_scaler(3), _get_dummy_scaler(4), _get_dummy_scaler(5),
        use_scalar_x=False,
    )
    val_ds = FilterDataset(
        val_idx_N, ds, x_scaler,
        _get_dummy_scaler(3), _get_dummy_scaler(4), _get_dummy_scaler(5),
        use_scalar_x=False,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Pre-extract val subset for fast rt_mse eval ───────────────────────────
    rng = np.random.RandomState(SEED)
    eval_n = min(K_EVAL_NSAMPLE, len(val_idx_N))
    eval_idx = rng.choice(len(val_idx_N), size=eval_n, replace=False)
    val_sub_x    = x_scaler.transform(
        (ds['X_full'] if 'X_full' in ds else ds['X_scalar'])[val_idx_N[eval_idx]]
    ).astype(np.float32)
    val_sub_fc   = ds['fc_GHz'][val_idx_N[eval_idx]].astype(np.float32)
    val_sub_s21  = ds['X_full'][val_idx_N[eval_idx], 5:106].astype(np.float32)
    val_sub_s11  = ds['X_full'][val_idx_N[eval_idx], 106:207].astype(np.float32)
    val_subset = {
        'x_norm':     val_sub_x,
        'fc_GHz':     val_sub_fc,
        's21_target': val_sub_s21,
        's11_target': val_sub_s11,
    }

    # ── Model + optimizer ─────────────────────────────────────────────────────
    inn      = make_cinn(D_y, D_c=COND_DIM, n_blocks=N_BLOCKS,
                         subnet_dim=SUBNET_DIM, affine_clamping=AFFINE_CLAMP)
    embedder = ConditionEmbedder(input_dim=207, cond_dim=COND_DIM)
    inn.to(device); embedder.to(device)
    fix_mps_contiguity(inn)  # fixes w_perm_inv non-contiguous layout on MPS

    # ── Bijection sanity check ────────────────────────────────────────────────
    print(f'  Bijection check (D_y={D_y}) ...', end=' ', flush=True)
    err = verify_bijection(inn, embedder, D_y, device, tol=1e-4)
    print(f'max err = {err:.2e}  ✓')

    params = list(inn.parameters()) + list(embedder.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=LR_PATIENCE, factor=LR_FACTOR,
    )

    # Normalization tensors (stay on device for batch construction)
    mean_t = torch.tensor(y_scaler_mean, dtype=torch.float32, device=device)
    std_t  = torch.tensor(y_scaler_std,  dtype=torch.float32, device=device)

    best_acc_frac      = -1.0
    best_val_rt_mse    = float('inf')
    no_improve         = 0
    diverged           = False
    last_good_state    = {'inn': inn.state_dict(), 'emb': embedder.state_dict()}

    history = {
        'train_nll': [], 'val_nll': [],
        'z_mean': [], 'z_std': [], 'logdet_mean': [],
        'grad_norm': [], 'lr': [],
        'val_rt_mse': [], 'val_acc_frac': [],
    }

    # ── Open CSV log ──────────────────────────────────────────────────────────
    with open(log_path, 'w', newline='') as lf:
        writer = csv.writer(lf)
        writer.writerow(['epoch', 'train_nll', 'val_nll', 'z_mean', 'z_std',
                         'logdet_mean', 'grad_norm', 'lr',
                         'val_rt_mse', 'val_acc_frac'])

        pbar = tqdm(range(1, MAX_EPOCHS + 1), desc=f'cINN N={N_val}')
        for epoch in pbar:
            # ── Train ─────────────────────────────────────────────────────────
            inn.train(); embedder.train()
            t_nll_total = 0.0
            ep_grad_norm = 0.0
            for batch in train_loader:
                x_b = batch['x'].to(device)

                # Build 4N-dim target: [log10(LC), log10(Q_L), log10(Q_C)]
                y_log_lc = batch['y_log'][:, :D_lc].to(device)          # (B, 2N)
                q_L_raw  = batch['Q_L'][:, :N_val].clamp(min=1e-6).to(device)
                q_C_raw  = batch['Q_C'][:, :N_val].clamp(min=1e-6).to(device)
                y_log_q  = torch.cat([torch.log10(q_L_raw),
                                      torch.log10(q_C_raw)], dim=1)     # (B, 2N)
                y_raw    = torch.cat([y_log_lc, y_log_q], dim=1)         # (B, 4N)
                y_b      = (y_raw - mean_t) / std_t                       # normalized

                c = embedder(x_b)
                z, logdet = inn(y_b, c=[c])
                nll = 0.5 * (z ** 2).sum(dim=1).mean() - logdet.mean()

                # Divergence guard
                if not torch.isfinite(nll) or nll.item() > NLL_NAN_GUARD:
                    print(f'\n  DIVERGENCE detected at epoch {epoch}: '
                          f'NLL={nll.item():.3e}. Saving last-good checkpoint.')
                    diverge_path = checkpoint_path.replace('.pt', '_diverged.pt')
                    torch.save({'inn_state_dict': last_good_state['inn'],
                                'embedder_state_dict': last_good_state['emb'],
                                'epoch': epoch, 'nll': nll.item()}, diverge_path)
                    with open(diverge_path.replace('.pt', '_diag.json'), 'w') as df:
                        json.dump({'epoch': epoch, 'nll': float(nll.item()),
                                   'N': N_val, 'D_y': D_y}, df, indent=2)
                    diverged = True
                    break

                optimizer.zero_grad()
                nll.backward()
                gn = torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP).item()
                optimizer.step()
                t_nll_total  += nll.item()
                ep_grad_norm += gn

            if diverged:
                break

            # ── Validate ──────────────────────────────────────────────────────
            inn.eval(); embedder.eval()
            v_nll_total = 0.0
            z_list = []
            ld_list = []
            with torch.no_grad():
                for batch in val_loader:
                    x_b = batch['x'].to(device)

                    y_log_lc = batch['y_log'][:, :D_lc].to(device)
                    q_L_raw  = batch['Q_L'][:, :N_val].clamp(min=1e-6).to(device)
                    q_C_raw  = batch['Q_C'][:, :N_val].clamp(min=1e-6).to(device)
                    y_log_q  = torch.cat([torch.log10(q_L_raw),
                                          torch.log10(q_C_raw)], dim=1)
                    y_raw    = torch.cat([y_log_lc, y_log_q], dim=1)
                    y_b      = (y_raw - mean_t) / std_t

                    c = embedder(x_b)
                    z_v, ld_v = inn(y_b, c=[c])
                    v_nll_total += (0.5*(z_v**2).sum(1).mean() - ld_v.mean()).item()
                    z_list.append(z_v.cpu())
                    ld_list.append(ld_v.cpu())

            z_cat = torch.cat(z_list)
            ld_cat = torch.cat(ld_list)
            t_nll = t_nll_total / len(train_loader)
            v_nll = v_nll_total / len(val_loader)
            z_mean = float(z_cat.mean())
            z_std  = float(z_cat.std())
            ld_mean = float(ld_cat.mean())
            ep_gn   = ep_grad_norm / len(train_loader)
            cur_lr  = optimizer.param_groups[0]['lr']

            scheduler.step(v_nll)

            # ── Fast rt_mse eval every K_EVAL_EVERY epochs ────────────────────
            val_rt_mse_ep   = None
            val_acc_frac_ep = None
            if epoch % K_EVAL_EVERY == 0:
                bk, af, _, _ = eval_rt_mse_fast(
                    inn, embedder, val_subset, K_EVAL, D_y, N_val,
                    y_scaler_mean, y_scaler_std, device
                )
                val_rt_mse_ep   = bk
                val_acc_frac_ep = af

                # Checkpoint on acceptable_fraction (then break ties with rt_mse)
                improved = (
                    val_acc_frac_ep > best_acc_frac or
                    (val_acc_frac_ep == best_acc_frac and bk < best_val_rt_mse)
                )
                if improved:
                    best_acc_frac   = val_acc_frac_ep
                    best_val_rt_mse = bk
                    no_improve      = 0
                    last_good_state = {
                        'inn': {k: v.cpu().clone() for k, v in inn.state_dict().items()},
                        'emb': {k: v.cpu().clone() for k, v in embedder.state_dict().items()},
                    }
                    torch.save({
                        'inn_state_dict':      inn.state_dict(),
                        'embedder_state_dict': embedder.state_dict(),
                        'y_scaler_mean': y_scaler_mean,
                        'y_scaler_std':  y_scaler_std,
                        'x_scaler_mean': x_scaler.mean_.astype(np.float32),
                        'x_scaler_std':  x_scaler.scale_.astype(np.float32),
                        'N': N_val, 'D_y': D_y, 'D_lc': D_lc, 'D_c': COND_DIM,
                        'n_blocks': N_BLOCKS, 'subnet_dim': SUBNET_DIM,
                        'affine_clamping': AFFINE_CLAMP,
                        'best_val_rt_mse': best_val_rt_mse,
                        'best_acc_frac': best_acc_frac,
                        'epoch': epoch,
                    }, checkpoint_path)
                else:
                    no_improve += 1

            # Record history
            history['train_nll'].append(t_nll)
            history['val_nll'].append(v_nll)
            history['z_mean'].append(z_mean)
            history['z_std'].append(z_std)
            history['logdet_mean'].append(ld_mean)
            history['grad_norm'].append(ep_gn)
            history['lr'].append(cur_lr)
            history['val_rt_mse'].append(val_rt_mse_ep)
            history['val_acc_frac'].append(val_acc_frac_ep)

            writer.writerow([epoch, t_nll, v_nll, z_mean, z_std,
                             ld_mean, ep_gn, cur_lr,
                             val_rt_mse_ep or '', val_acc_frac_ep or ''])
            lf.flush()

            pbar.set_postfix({
                'tr_nll': f'{t_nll:.3f}',
                'vl_nll': f'{v_nll:.3f}',
                'z_std':  f'{z_std:.2f}',
                'best_af': f'{best_acc_frac:.2f}',
                'pat': no_improve,
            })

            if no_improve >= ES_PATIENCE:
                print(f'\n  Early stopping at epoch {epoch} '
                      f'(best acc_frac={best_acc_frac:.3f}, '
                      f'best rt_mse={best_val_rt_mse:.1f})')
                break

    # ── Post-training bijection check ─────────────────────────────────────────
    if not diverged and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, weights_only=False)
        inn.load_state_dict(ckpt['inn_state_dict'])
        embedder.load_state_dict(ckpt['embedder_state_dict'])
        inn.to(device); embedder.to(device)
        fix_mps_contiguity(inn)
        print(f'  Post-training bijection check ...', end=' ', flush=True)
        try:
            err = verify_bijection(inn, embedder, D_y, device, tol=1e-3)
            print(f'max err = {err:.2e}  ✓')
        except AssertionError as e:
            print(f'WARN: {e}')

    return {
        'history': history,
        'best_val_rt_mse': best_val_rt_mse,
        'best_acc_frac': best_acc_frac,
        'diverged': diverged,
    }


# ── Final test evaluation ─────────────────────────────────────────────────────

def evaluate_test_set(
    N_val: int,
    test_idx_N: np.ndarray,
    ds: dict,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    checkpoint_path: str,
    device: torch.device,
) -> dict:
    """
    Full multi-metric test evaluation for one N using the saved checkpoint.

    Metrics: best-of-K rt_mse (with predicted Q), mean rt_mse, median rt_mse,
             acceptable_fraction, diversity, mode count (DBSCAN), posterior NLL
             on y_true, Q accuracy (q_L_mae, q_C_mae in log10-space),
             comp_mse on LC portion, r2_mean.
    """
    D_lc = 2 * N_val
    D_y  = 4 * N_val
    ckpt = torch.load(checkpoint_path, weights_only=False)

    inn      = make_cinn(D_y, D_c=COND_DIM, n_blocks=N_BLOCKS,
                         subnet_dim=SUBNET_DIM, affine_clamping=AFFINE_CLAMP)
    embedder = ConditionEmbedder(input_dim=207, cond_dim=COND_DIM)
    inn.load_state_dict(ckpt['inn_state_dict'])
    embedder.load_state_dict(ckpt['embedder_state_dict'])
    inn.to(device); embedder.to(device)
    fix_mps_contiguity(inn)
    inn.eval(); embedder.eval()

    y_scaler_mean = ckpt['y_scaler_mean']   # (4N,)
    y_scaler_std  = ckpt['y_scaler_std']    # (4N,)

    x_raw  = ds['X_full'][test_idx_N]
    x_norm = x_scaler.transform(x_raw).astype(np.float32)

    # Full 4N ground-truth target (for NLL evaluation)
    y_true_full = build_y_full(ds, test_idx_N, N_val)      # (n_test, 4N)
    # LC-only ground truth (for comp_mse / R²)
    y_log_true_lc = ds['y_log'][test_idx_N, :D_lc].astype(np.float32)  # (n_test, 2N)
    # True Q in log10-space (for Q accuracy metric)
    q_L_true_log = np.log10(np.clip(ds['Q_L'][test_idx_N, :N_val], 1e-6, None))  # (n_test, N)
    q_C_true_log = np.log10(np.clip(ds['Q_C'][test_idx_N, :N_val], 1e-6, None))  # (n_test, N)

    fc_arr  = ds['fc_GHz'][test_idx_N].astype(np.float32)
    s21_arr = x_raw[:, 5:106].astype(np.float32)
    s11_arr = x_raw[:, 106:207].astype(np.float32)
    n_test  = len(test_idx_N)

    # ── Sample K=K_INFERENCE candidates per test sample ──────────────────────
    MBSZ = 32
    y_log_K   = []  # (n_test, K_INFERENCE, D_y)
    y_log_div = []  # (n_test, K_DIVERSITY, D_y) — for diversity/mode analysis

    for start in range(0, n_test, MBSZ):
        xb = torch.from_numpy(x_norm[start:start+MBSZ]).float()
        yl = sample_lc_batch(inn, embedder, xb, K_INFERENCE, D_y,
                              y_scaler_mean, y_scaler_std, device)
        y_log_K.append(yl)

    for start in range(0, n_test, MBSZ):
        xb = torch.from_numpy(x_norm[start:start+MBSZ]).float()
        yl = sample_lc_batch(inn, embedder, xb, K_DIVERSITY, D_y,
                              y_scaler_mean, y_scaler_std, device)
        y_log_div.append(yl)

    y_log_K   = np.concatenate(y_log_K,   axis=0)   # (n_test, K_INFERENCE, D_y)
    y_log_div = np.concatenate(y_log_div, axis=0)   # (n_test, K_DIVERSITY, D_y)

    # ── Posterior NLL on y_true (full 4N target) ─────────────────────────────
    nll_true_list = []
    with torch.no_grad():
        for start in range(0, n_test, MBSZ):
            xb = torch.from_numpy(x_norm[start:start+MBSZ]).float().to(device)
            yb_true_norm = torch.from_numpy(
                ((y_true_full[start:start+MBSZ] - y_scaler_mean) / y_scaler_std)
            ).float().to(device)
            c = embedder(xb)
            z_t, ld_t = inn(yb_true_norm, c=[c])
            nll_t = (0.5*(z_t**2).sum(1) - ld_t).cpu().numpy()
            nll_true_list.append(nll_t)
    nll_true = np.concatenate(nll_true_list)

    # ── rt_mse per sample using predicted Q ──────────────────────────────────
    rt_best_list  = []
    rt_all_list   = []

    for i in range(n_test):
        fc  = float(fc_arr[i])
        s21 = s21_arr[i]
        s11 = s11_arr[i]

        rts = []
        for k in range(K_INFERENCE):
            y_samp    = y_log_K[i, k]                              # (D_y,)
            lc_block  = y_samp[:D_lc]                              # (2N,)
            L = 10.0 ** lc_block[0::2]                             # (N,)
            C = 10.0 ** lc_block[1::2]                             # (N,)
            q_L_pred = 10.0 ** y_samp[D_lc:D_lc+N_val]            # (N,)
            q_C_pred = 10.0 ** y_samp[D_lc+N_val:]                 # (N,)
            rt = roundtrip_mse_lc(L, C, N_val, fc,
                                  q_L_pred, q_C_pred, s21, s11, FREQ_HZ)
            rts.append(rt if np.isfinite(rt) else RT_THRESHOLD * 10)
        rt_best_list.append(min(rts))
        rt_all_list.extend(rts)

    rt_best = np.array(rt_best_list)
    rt_all  = np.array(rt_all_list)

    # ── z=0 prediction (modal estimate) with predicted Q ─────────────────────
    rt_z0_list = []
    with torch.no_grad():
        for start in range(0, n_test, MBSZ):
            xb = torch.from_numpy(x_norm[start:start+MBSZ]).float().to(device)
            c  = embedder(xb)
            bsz = xb.shape[0]
            z0  = torch.zeros(bsz, D_y, device=device)
            y_z0_norm, _ = inn(z0, c=[c], rev=True)
            y_z0_log = (y_z0_norm.cpu().numpy() * y_scaler_std[None, :]
                        + y_scaler_mean[None, :])
            for bi in range(bsz):
                i = start + bi
                lc_block  = y_z0_log[bi, :D_lc]
                L = 10.0 ** lc_block[0::2]
                C = 10.0 ** lc_block[1::2]
                q_L_pred = 10.0 ** y_z0_log[bi, D_lc:D_lc+N_val]
                q_C_pred = 10.0 ** y_z0_log[bi, D_lc+N_val:]
                rt = roundtrip_mse_lc(
                    L, C, N_val, float(fc_arr[i]),
                    q_L_pred, q_C_pred,
                    s21_arr[i], s11_arr[i], FREQ_HZ
                )
                rt_z0_list.append(rt if np.isfinite(rt) else RT_THRESHOLD * 10)
    rt_z0 = np.array(rt_z0_list)

    # ── Component MSE / R² on LC portion (mean-of-K prediction) ──────────────
    y_log_lc_mean = y_log_K[:, :, :D_lc].mean(axis=1)   # (n_test, 2N)
    comp_mse_val = component_mse(y_log_lc_mean, y_log_true_lc)
    r2_val = r2_per_component(y_log_lc_mean, y_log_true_lc)
    r2_mean = float(np.nanmean(r2_val))

    # ── Q accuracy: MAE in log10-space (mean-of-K prediction) ────────────────
    q_L_pred_mean_log = y_log_K[:, :, D_lc:D_lc+N_val].mean(axis=1)   # (n_test, N)
    q_C_pred_mean_log = y_log_K[:, :, D_lc+N_val:].mean(axis=1)        # (n_test, N)
    q_L_mae = float(np.abs(q_L_pred_mean_log - q_L_true_log).mean())
    q_C_mae = float(np.abs(q_C_pred_mean_log - q_C_true_log).mean())

    # ── Diversity (std of samples in normalized space) ────────────────────────
    y_log_div_norm = (y_log_div - y_scaler_mean[None, None, :]) / y_scaler_std[None, None, :]
    diversity_per_sample = y_log_div_norm.std(axis=1).mean(axis=1)  # (n_test,)
    diversity_mean = float(diversity_per_sample.mean())

    # ── Mode count via DBSCAN on 5 random test conditions ─────────────────────
    rng = np.random.RandomState(SEED)
    mode_counts = []
    mode_samples_for_plot = []
    chosen_for_modes = rng.choice(n_test, size=min(5, n_test), replace=False)
    for ci in chosen_for_modes:
        samples_norm = (y_log_div[ci] - y_scaler_mean) / y_scaler_std  # (K_DIV, D_y)
        db = DBSCAN(eps=0.3, min_samples=3).fit(samples_norm)
        n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        mode_counts.append(max(n_clusters, 1))
        mode_samples_for_plot.append({
            'y_log_samples': y_log_div[ci],   # (K_DIV, D_y) — first 2N are LC
            'fc_GHz': float(fc_arr[ci]),
            'q_L_pred_mean': 10.0 ** q_L_pred_mean_log[ci],   # (N,)
            'q_C_pred_mean': 10.0 ** q_C_pred_mean_log[ci],   # (N,)
            's21_target': s21_arr[ci],
            'N': N_val,
        })
    mode_count_mean = float(np.mean(mode_counts))

    return {
        'N': N_val,
        'rt_best_of_K': float(np.mean(rt_best)),
        'rt_mean':       float(np.mean(rt_all)),
        'rt_median':     float(np.median(rt_all)),
        'rt_z0':         float(np.mean(rt_z0)),
        'acc_frac':      float(np.mean(rt_all < RT_THRESHOLD)),
        'diversity':     diversity_mean,
        'mode_count':    mode_count_mean,
        'nll_true_mean': float(np.mean(nll_true)),
        'nll_true_std':  float(np.std(nll_true)),
        'comp_mse':      comp_mse_val,
        'r2_mean':       r2_mean,
        'q_L_mae':       q_L_mae,
        'q_C_mae':       q_C_mae,
        # For plots
        'y_log_K':       y_log_K,
        'y_log_true_lc': y_log_true_lc,
        'mode_samples':  mode_samples_for_plot,
        'rt_best_arr':   rt_best,
        'rt_z0_arr':     rt_z0,
        'fc_arr':        fc_arr,
        'q_L_pred_mean_log': q_L_pred_mean_log,
        'q_C_pred_mean_log': q_C_pred_mean_log,
        'q_L_true_log':  q_L_true_log,
        'q_C_true_log':  q_C_true_log,
        's21_arr':       s21_arr,
        's11_arr':       s11_arr,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def save_loss_plot(history: dict, N_val: int, figures_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    epochs = range(1, len(history['train_nll']) + 1)

    axes[0].plot(epochs, history['train_nll'], label='Train NLL')
    axes[0].plot(epochs, history['val_nll'],   label='Val NLL')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('NLL')
    axes[0].set_title(f'N={N_val} — NLL Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history['z_std'], label='z_std (val)', color='tab:orange')
    axes[1].axhline(1.0, color='k', linestyle='--', alpha=0.5, label='target=1')
    axes[1].plot(epochs, history['z_mean'], label='z_mean (val)', color='tab:blue')
    axes[1].axhline(0.0, color='k', linestyle=':', alpha=0.5, label='target=0')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('z statistic')
    axes[1].set_title(f'N={N_val} — z statistics'); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history['logdet_mean'], color='tab:green')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('mean log|det J|')
    axes[2].set_title(f'N={N_val} — log det J'); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(figures_dir, f'inn_q_N{N_val}_loss.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()


def save_z_hist(history: dict, N_val: int, figures_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    z_means = [v for v in history['z_mean'] if v is not None]
    z_stds  = [v for v in history['z_std']  if v is not None]
    ep      = range(1, len(z_means) + 1)
    ax.plot(ep, z_means, label='z_mean'); ax.axhline(0,  color='k', ls='--', alpha=0.4)
    ax.plot(ep, z_stds,  label='z_std');  ax.axhline(1.0, color='r', ls='--', alpha=0.4)
    ax.set_xlabel('Epoch'); ax.set_title(f'N={N_val} — z statistics convergence')
    ax.legend(); ax.grid(alpha=0.3)
    out = os.path.join(figures_dir, f'inn_q_N{N_val}_z_hist.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()


def save_logdet_plot(history: dict, N_val: int, figures_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(history['logdet_mean']) + 1), history['logdet_mean'])
    ax.set_xlabel('Epoch'); ax.set_ylabel('mean log|det J|')
    ax.set_title(f'N={N_val} — log|det J|'); ax.grid(alpha=0.3)
    out = os.path.join(figures_dir, f'inn_q_N{N_val}_logdet.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()


def save_scatter_plot(results_by_N: dict, figures_dir: str) -> None:
    from evaluation.visualize import plot_scatter
    all_pred = []
    all_true = []
    all_N    = []
    for N_val, res in results_by_N.items():
        D_lc = 2 * N_val
        n    = res['y_log_true_lc'].shape[0]
        y_pred_padded = np.full((n, 10), np.nan)
        y_true_padded = np.full((n, 10), np.nan)
        y_pred_padded[:, :D_lc] = res['y_log_K'][:, :, :D_lc].mean(axis=1)
        y_true_padded[:, :D_lc] = res['y_log_true_lc']
        all_pred.append(y_pred_padded)
        all_true.append(y_true_padded)
        all_N.append(np.full(n, N_val, dtype=int))
    y_pred_all = np.concatenate(all_pred)
    y_true_all = np.concatenate(all_true)
    N_all      = np.concatenate(all_N)
    plot_scatter(y_pred_all, y_true_all, N_all,
                 os.path.join(figures_dir, 'inn_q_scatter.png'))


def save_roundtrip_plot(results_by_N: dict, ds: dict, figures_dir: str) -> None:
    from evaluation.visualize import plot_roundtrip
    from evaluation.metrics import synthesize_from_lc
    samples = []
    rng = np.random.RandomState(0)
    for N_val, res in results_by_N.items():
        D_lc = 2 * N_val
        n = res['y_log_true_lc'].shape[0]
        chosen = rng.choice(n, size=min(2, n), replace=False)
        for ci in chosen:
            fc      = float(res['fc_arr'][ci])
            s21_tgt = res['s21_arr'][ci]
            rt_best_k = res['rt_best_arr'][ci]
            # Use mean-of-K prediction with predicted Q
            y_log_mean = res['y_log_K'][ci].mean(axis=0)   # (D_y,)
            lc_block = y_log_mean[:D_lc]
            L = 10.0 ** lc_block[0::2]
            C = 10.0 ** lc_block[1::2]
            q_L_pred = 10.0 ** y_log_mean[D_lc:D_lc+N_val]
            q_C_pred = 10.0 ** y_log_mean[D_lc+N_val:]
            s21_pred, _ = synthesize_from_lc(L, C, N_val, fc, q_L_pred, q_C_pred, FREQ_HZ)
            if s21_pred is None:
                s21_pred = np.zeros(101)
            samples.append({
                's21_target': s21_tgt,
                's21_pred':   s21_pred,
                'N': N_val,
                'ripple_dB': 0.0,
                'fc_GHz': fc,
                'fbw': 0.2,
                'rt_mse': rt_best_k,
            })
    if samples:
        plot_roundtrip(samples, os.path.join(figures_dir, 'inn_q_roundtrip.png'),
                       model_label='cINN (LC+Q)')


def save_diversity_plot(results_by_N: dict, figures_dir: str) -> None:
    """
    For up to 3 test conditions per N, overlay K_DIVERSITY synthesized S21 curves
    plus ground-truth S21. Uses each sample's own predicted Q for synthesis.
    """
    from evaluation.metrics import synthesize_from_lc
    for N_val, res in results_by_N.items():
        D_lc = 2 * N_val
        mode_samples = res['mode_samples'][:3]
        if not mode_samples:
            continue
        fig, axes = plt.subplots(1, len(mode_samples),
                                 figsize=(6 * len(mode_samples), 4), sharey=True)
        if len(mode_samples) == 1:
            axes = [axes]
        for ax, ms in zip(axes, mode_samples):
            fc  = ms['fc_GHz']
            s21_tgt = ms['s21_target']

            ax.plot(FREQ_HZ / 1e9, s21_tgt, 'k-', lw=2, label='Target', zorder=5)
            for k in range(ms['y_log_samples'].shape[0]):
                y_log = ms['y_log_samples'][k]          # (D_y,)
                lc_block = y_log[:D_lc]
                L = 10.0 ** lc_block[0::2]
                C = 10.0 ** lc_block[1::2]
                q_L = 10.0 ** y_log[D_lc:D_lc+N_val]
                q_C = 10.0 ** y_log[D_lc+N_val:]
                s21, _ = synthesize_from_lc(L, C, N_val, fc, q_L, q_C, FREQ_HZ)
                if s21 is not None:
                    ax.plot(FREQ_HZ / 1e9, s21, alpha=0.3, lw=0.8, color='tab:blue')
            ax.set_xlabel('Freq (GHz)'); ax.set_ylabel('S21 (dB)')
            ax.set_ylim(-60, 3)
            ax.set_title(f'N={N_val}, fc={fc:.1f} GHz  ({K_DIVERSITY} samples)')
            ax.grid(alpha=0.3)

        handles = [
            plt.Line2D([0],[0], color='k', lw=2, label='Target'),
            plt.Line2D([0],[0], color='tab:blue', alpha=0.5, lw=1, label='cINN samples'),
        ]
        axes[-1].legend(handles=handles)
        plt.tight_layout()
        out = os.path.join(figures_dir, f'inn_q_diversity_N{N_val}.png')
        plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()


# ── ForwardMLP cross-check ────────────────────────────────────────────────────

def forwardmlp_crosscheck(
    N_val: int,
    test_idx_N: np.ndarray,
    ds: dict,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    checkpoint_path: str,
    fwd_ckpt_path: str,
    device: torch.device,
    n_samples: int = 500,
) -> float:
    """
    Compute best-of-K=10 rt_mse using the ForwardMLP as a cross-check simulator.
    Uses predicted Q from the cINN samples (not ground-truth Q).
    Returns mean best-of-K rt_mse via ForwardMLP, or None if checkpoint not found.
    """
    if not os.path.exists(fwd_ckpt_path):
        print(f'  ForwardMLP checkpoint not found at {fwd_ckpt_path} — skipping cross-check.')
        return None

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.forward_model import ForwardMLP

    D_lc = 2 * N_val
    D_y  = 4 * N_val
    ckpt = torch.load(checkpoint_path, weights_only=False)
    inn  = make_cinn(D_y, D_c=COND_DIM, n_blocks=N_BLOCKS,
                     subnet_dim=SUBNET_DIM, affine_clamping=AFFINE_CLAMP)
    embedder = ConditionEmbedder(input_dim=207, cond_dim=COND_DIM)
    inn.load_state_dict(ckpt['inn_state_dict'])
    embedder.load_state_dict(ckpt['embedder_state_dict'])
    inn.to(device); embedder.to(device)
    fix_mps_contiguity(inn)
    inn.eval(); embedder.eval()

    y_scaler_mean = ckpt['y_scaler_mean']
    y_scaler_std  = ckpt['y_scaler_std']

    fwd_ckpt = torch.load(fwd_ckpt_path, map_location='cpu', weights_only=False)
    fwd_model = ForwardMLP()
    fwd_model.load_state_dict(fwd_ckpt['model_state_dict'])
    fwd_model.eval()

    rng = np.random.RandomState(SEED)
    n = min(n_samples, len(test_idx_N))
    sub_idx = rng.choice(len(test_idx_N), size=n, replace=False)
    sub_test = test_idx_N[sub_idx]

    x_raw  = ds['X_full'][sub_test]
    x_norm = x_scaler.transform(x_raw).astype(np.float32)
    fc_arr = ds['fc_GHz'][sub_test].astype(np.float32)
    s21_arr = x_raw[:, 5:106].astype(np.float32)
    s11_arr = x_raw[:, 106:207].astype(np.float32)

    MBSZ = 32
    y_log_K = []
    for start in range(0, n, MBSZ):
        xb = torch.from_numpy(x_norm[start:start+MBSZ]).float()
        yl = sample_lc_batch(inn, embedder, xb, 10, D_y,
                              y_scaler_mean, y_scaler_std, device)
        y_log_K.append(yl)
    y_log_K = np.concatenate(y_log_K, axis=0)

    from models.forward_model import build_forward_input
    rt_best_fwd = []
    for i in range(n):
        rts = []
        for k in range(10):
            y_samp = y_log_K[i, k]                            # (D_y,)
            lc_block  = y_samp[:D_lc]
            q_L_pred  = 10.0 ** y_samp[D_lc:D_lc+N_val]      # (N,)
            q_C_pred  = 10.0 ** y_samp[D_lc+N_val:]           # (N,)

            # Pad LC to 10-dim for build_forward_input
            y_log_pad = np.zeros(10, dtype=np.float32)
            y_log_pad[:D_lc] = lc_block

            # Pad Q to 5-dim
            q_L_pad = np.zeros((1, 5), dtype=np.float32)
            q_C_pad = np.zeros((1, 5), dtype=np.float32)
            q_L_pad[0, :N_val] = q_L_pred
            q_C_pad[0, :N_val] = q_C_pred

            fwd_in = build_forward_input(
                y_log_pad[None, :], q_L_pad, q_C_pad,
                fc_arr[i:i+1], ds['fbw'][sub_test[i:i+1]],
                np.array([N_val]),
            )
            fwd_in_t = torch.from_numpy(fwd_in.astype(np.float32))
            with torch.no_grad():
                s_pred = fwd_model.predict(fwd_in_t)[0].cpu().numpy()   # (202,)
            s21_pred = s_pred[:101]; s11_pred = s_pred[101:]
            mse = float(np.mean((np.concatenate([s21_pred, s11_pred])
                                 - np.concatenate([s21_arr[i], s11_arr[i]])) ** 2))
            rts.append(mse)
        rt_best_fwd.append(min(rts))

    return float(np.mean(rt_best_fwd))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path    = os.path.join(project_root, 'data', 'dataset_realistic.pkl')
    results_dir  = os.path.join(project_root, 'results')
    figures_dir  = os.path.join(results_dir, 'figures')
    fwd_ckpt     = os.path.join(results_dir, 'forward_model_best.pt')
    os.makedirs(figures_dir, exist_ok=True)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load dataset ──────────────────────────────────────────────────────────
    with open(data_path, 'rb') as f:
        ds = pickle.load(f)
    n_total = len(ds['N'])
    print(f'Loaded {n_total} samples')

    # ── Stratified 80/10/10 split — identical seed as train_mlp.py ───────────
    all_idx = np.arange(n_total)
    train_idx, temp_idx = train_test_split(
        all_idx, test_size=0.20, stratify=ds['N'], random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=ds['N'][temp_idx], random_state=42
    )
    print(f'Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}')

    # ── Fit scalers ───────────────────────────────────────────────────────────
    x_scaler = StandardScaler().fit(ds['X_full'][train_idx])

    # Joint 4N-dim scaler: [log10(LC), log10(Q_L), log10(Q_C)]
    y_scalers = {}
    for N_val in [3, 4, 5]:
        mask_tr = ds['N'][train_idx] == N_val
        idx_tr_N = train_idx[mask_tr]
        y_full = build_y_full(ds, idx_tr_N, N_val)    # (n, 4N)
        y_scalers[N_val] = StandardScaler().fit(y_full)

    # ── Train one cINN per N sequentially ────────────────────────────────────
    train_summaries = {}
    for N_val in [3, 4, 5]:
        train_idx_N = train_idx[ds['N'][train_idx] == N_val]
        val_idx_N   = val_idx[ds['N'][val_idx]   == N_val]

        ckpt_path = os.path.join(results_dir, f'inn_q_N{N_val}_best.pt')
        log_path  = os.path.join(results_dir, f'inn_q_N{N_val}_log.csv')

        if os.path.exists(ckpt_path):
            print(f'\n  N={N_val}: checkpoint already exists at {ckpt_path} — skipping training.')
            ckpt = torch.load(ckpt_path, weights_only=False)
            train_summaries[N_val] = {
                'history': {'train_nll': [], 'val_nll': [], 'z_mean': [], 'z_std': [],
                            'logdet_mean': [], 'grad_norm': [], 'lr': [],
                            'val_rt_mse': [], 'val_acc_frac': []},
                'best_val_rt_mse': ckpt.get('best_val_rt_mse', float('inf')),
                'best_acc_frac':   ckpt.get('best_acc_frac', -1.0),
                'diverged': False,
            }
            continue

        summary = train_one_cinn(
            N_val, train_idx_N, val_idx_N,
            ds, x_scaler, y_scalers[N_val],
            device, ckpt_path, log_path, results_dir, figures_dir,
        )
        train_summaries[N_val] = summary

        # Save training plots immediately after each N
        save_loss_plot(summary['history'], N_val, figures_dir)
        save_z_hist(summary['history'],   N_val, figures_dir)
        save_logdet_plot(summary['history'], N_val, figures_dir)

    # ── Final test evaluation ─────────────────────────────────────────────────
    print('\n' + '='*60)
    print('  Final test-set evaluation (K=50 best-of-K, predicted Q)')
    print('='*60)

    results_by_N = {}
    for N_val in [3, 4, 5]:
        ckpt_path = os.path.join(results_dir, f'inn_q_N{N_val}_best.pt')
        if not os.path.exists(ckpt_path):
            print(f'  N={N_val}: no checkpoint found (training diverged?), skipping.')
            continue
        test_idx_N = test_idx[ds['N'][test_idx] == N_val]
        print(f'  Evaluating N={N_val} on {len(test_idx_N)} test samples ...')
        res = evaluate_test_set(
            N_val, test_idx_N, ds, x_scaler, y_scalers[N_val],
            ckpt_path, device
        )
        results_by_N[N_val] = res

    # ── ForwardMLP cross-check ────────────────────────────────────────────────
    fwd_rt = {}
    for N_val in [3, 4, 5]:
        ckpt_path = os.path.join(results_dir, f'inn_q_N{N_val}_best.pt')
        if not os.path.exists(ckpt_path):
            continue
        test_idx_N = test_idx[ds['N'][test_idx] == N_val]
        print(f'  ForwardMLP cross-check N={N_val} ...')
        frt = forwardmlp_crosscheck(
            N_val, test_idx_N, ds, x_scaler, y_scalers[N_val],
            ckpt_path, fwd_ckpt, device, n_samples=500,
        )
        fwd_rt[N_val] = frt

    # ── Print and save metrics table ──────────────────────────────────────────
    # Phase 4c baselines (with GT-Q evaluation)
    phase4c_rt = {3: 0.46, 4: 1.86, 5: 0.73}
    mlp_rt     = {3: 36.49, 4: 23.86, 5: 39.69}

    header = (f"{'Metric':<28} | {'N=3':>10} | {'N=4':>10} | {'N=5':>10}")
    sep    = '-' * len(header)
    lines  = [sep, header, sep]

    def row(label, fn):
        vals = [f'{fn(results_by_N[N]):.4f}' if N in results_by_N else 'N/A'
                for N in [3, 4, 5]]
        return f'{label:<28} | {vals[0]:>10} | {vals[1]:>10} | {vals[2]:>10}'

    lines.append(f"{'MLP baseline rt_mse':<28} | {mlp_rt[3]:>10.2f} | {mlp_rt[4]:>10.2f} | {mlp_rt[5]:>10.2f}")
    lines.append(f"{'Phase 4c (GT-Q eval)':<28} | {phase4c_rt[3]:>10.2f} | {phase4c_rt[4]:>10.2f} | {phase4c_rt[5]:>10.2f}")
    lines.append(sep)
    lines.append(row('cINN+Q best-of-50',      lambda r: r['rt_best_of_K']))
    lines.append(row('cINN+Q mean rt_mse',     lambda r: r['rt_mean']))
    lines.append(row('cINN+Q median rt_mse',   lambda r: r['rt_median']))
    lines.append(row('cINN+Q z=0 rt_mse',      lambda r: r['rt_z0']))
    lines.append(row('acc_frac (< 50 dB²)',    lambda r: r['acc_frac']))
    lines.append(row('diversity (norm std)',    lambda r: r['diversity']))
    lines.append(row('mode count (DBSCAN)',     lambda r: r['mode_count']))
    lines.append(row('NLL on y_true',           lambda r: r['nll_true_mean']))
    lines.append(row('comp_mse LC (log)',       lambda r: r['comp_mse']))
    lines.append(row('r2_mean LC',              lambda r: r['r2_mean']))
    lines.append(row('q_L_mae (log10)',         lambda r: r['q_L_mae']))
    lines.append(row('q_C_mae (log10)',         lambda r: r['q_C_mae']))

    fwd_row = [f'{fwd_rt.get(N, float("nan")):.2f}' if fwd_rt.get(N) is not None else 'N/A'
               for N in [3, 4, 5]]
    lines.append(f"{'FwdMLP best-of-10':<28} | {fwd_row[0]:>10} | {fwd_row[1]:>10} | {fwd_row[2]:>10}")
    lines.append(sep)

    table_str = '\n'.join(lines)
    print('\n' + table_str)

    table_path = os.path.join(results_dir, 'inn_q_metrics_table.txt')
    with open(table_path, 'w') as tf:
        tf.write(table_str + '\n')
    print(f'\nMetrics table saved → {table_path}')

    # ── Plots ─────────────────────────────────────────────────────────────────
    print('\nSaving evaluation plots ...')
    if results_by_N:
        save_scatter_plot(results_by_N, figures_dir)
        save_roundtrip_plot(results_by_N, ds, figures_dir)
        save_diversity_plot(results_by_N, figures_dir)

    for fname in ['inn_q_scatter.png', 'inn_q_roundtrip.png']:
        full = os.path.join(figures_dir, fname)
        if os.path.exists(full):
            print(f'  Saved {full}')
    for N_val in [3, 4, 5]:
        for suffix in ['loss', 'z_hist', 'logdet', 'diversity']:
            p = os.path.join(figures_dir, f'inn_q_N{N_val}_{suffix}.png')
            if os.path.exists(p):
                print(f'  Saved {p}')

    print('\nPhase 4c+ complete.')


if __name__ == '__main__':
    main()
