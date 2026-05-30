"""
Phase 4c-V2: Training script for cINN with parasitic-conditioned ConditionEmbedderV2.

Dataset: dataset_otfl301v2.pkl — 24–40 GHz SOI, N∈{3,5}, 4 physics improvements.

Architecture (Bundle 1+4+5):
  - ConditionEmbedderV2: [231 → 256 → 128 → cond_dim] (~80k params)
  - INN: n_blocks=8, subnet_dim=128, cond_dim=128 (~340k params)
  - D_y = 2*N (LC only) — Q is NOT part of the cINN target.
  - Condition input = X_full(207) + parasitics(24):
      Q_L(5) + Q_C(5) + k_m(4) + C_sub_frac(5) + alpha_C(5) = 24.
    For N=3 samples, unused slots are zero-padded.
  - z_std regularizer: λ·(mean(std(z)) - 1)² added to NLL loss.

Why D_y=2N (not 4N):
  Q_L ~ Uniform[15,30], Q_C ~ Uniform[100,300] are sampled independently of the
  spec/S-param conditioning input. Including them in D_y forces the cINN to fit 2N
  uninformative dimensions, causing z_std drift. With Q as CONDITIONING (not target),
  the cINN sees Q as a known confounder — posterior on LC becomes near-deterministic.

Why parasitic conditioning:
  S = f(LC, Q, k_m, C_sub, alpha_C). With only S-params as input, the cINN must
  marginalize over 24 unknown parasitic dimensions, producing a genuinely broad
  posterior. With parasitics as conditioning, the posterior p(LC | S, parasitics)
  is sharp (only residual process spread remains). At eval time, ground-truth
  parasitics from the dataset are used — consistent with real EMX design where
  the process stack is known at design time.

Usage:
    source rf_env/bin/activate
    python -u training/train_inn_v2.py

Outputs:
    results/inn_v2_otfl301v2_N{3,5}_best.pt   — checkpoints
    results/inn_v2_otfl301v2_N{3,5}_log.csv   — per-epoch diagnostics
    results/inn_v2_metrics_table.txt           — benchmark table
    results/figures/inn_v2_*.png               — diagnostic plots
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
from models.inn_v2 import (
    ConditionEmbedderV2, make_cinn_v2, fix_mps_contiguity, verify_bijection
)
from evaluation.metrics import (
    synthesize_from_lc, roundtrip_mse_lc, component_mse, r2_per_component,
    FREQ_HZ_OTFL301 as FREQ_HZ,
)
from training.train_mlp import FilterDataset

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_BLOCKS        = 8        # Bundle 5: was 12 — smaller model, less overfit
SUBNET_DIM      = 128      # Bundle 5: was 256 — ~5x param reduction
COND_DIM        = 128
AFFINE_CLAMP    = 1.5
LR              = 1e-3
WEIGHT_DECAY    = 1e-4     # Bundle 5: was 1e-5 — stronger reg with smaller model
BATCH_SIZE      = 256
MAX_EPOCHS      = 400
COND_INPUT_DIM  = 231      # Bundle 1: 207 (X_full) + 24 (parasitics) = 231
Z_REG_LAMBDA    = 1.0      # Bundle 4: λ·(mean(std(z)) - 1)² penalty
ES_PATIENCE     = 30
LR_PATIENCE     = 15
LR_FACTOR       = 0.5
GRAD_CLIP       = 10.0
K_EVAL_EVERY    = 10
K_EVAL          = 10
K_EVAL_NSAMPLE  = 500
K_INFERENCE     = 50
K_DIVERSITY     = 20
NLL_NAN_GUARD   = 1e4
SEED            = 42
RT_THRESHOLD    = 5.0      # dB² — honest threshold; eval now uses true parasitics so floor ≈ 0

PREFIX          = 'inn_v2_otfl301v2'  # checkpoint/log/plot filename prefix


# ── Bundle 1: parasitic-conditioned input ─────────────────────────────────────

def build_full_x(ds, indices):
    """Build the 231-dim conditioning input: X_full(207) + parasitics(24).

    Parasitics packed as [Q_L(5), Q_C(5), k_m(4), C_sub_frac(5), alpha_C(5)] = 24.
    NaN-padded slots (for N=3 samples in 5/4-shaped arrays) are replaced with 0.
    """
    x_full = ds['X_full'][indices]                                        # (n, 207)
    Q_L = np.nan_to_num(ds['Q_L'][indices],        nan=0.0)              # (n, 5)
    Q_C = np.nan_to_num(ds['Q_C'][indices],        nan=0.0)              # (n, 5)
    k_m = np.nan_to_num(ds['k_m'][indices],        nan=0.0)              # (n, 4)
    csf = np.nan_to_num(ds['C_sub_frac'][indices], nan=0.0)              # (n, 5)
    aC  = np.nan_to_num(ds['alpha_C'][indices],    nan=0.0)              # (n, 5)
    parasitics = np.concatenate([Q_L, Q_C, k_m, csf, aC], axis=1)        # (n, 24)
    return np.concatenate([x_full, parasitics], axis=1).astype(np.float32)


# ── Per-N sampling ────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_batch(
    inn: nn.Module,
    embedder: nn.Module,
    x_norm: torch.Tensor,
    K: int,
    D_y: int,
    y_scaler_mean: np.ndarray,
    y_scaler_std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Sample K candidates for a batch of B conditions. Returns (B, K, D_y)."""
    inn.eval(); embedder.eval()
    B = x_norm.shape[0]
    c = embedder(x_norm.to(device))
    c_rep = c.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
    z = torch.randn(B * K, D_y, device=device)
    y_norm_flat, _ = inn(z, c=[c_rep], rev=True)
    y_norm_np = y_norm_flat.cpu().numpy().reshape(B, K, D_y)
    return y_norm_np * y_scaler_std[None, None, :] + y_scaler_mean[None, None, :]


# ── Fast val rt_mse ───────────────────────────────────────────────────────────

def eval_rt_mse_fast(
    inn, embedder, val_subset, K, D_y, N_val,
    y_scaler_mean, y_scaler_std, device
):
    """best-of-K rt_mse and acc_frac using ground-truth Q (D_y = 2*N = LC only)."""
    D_lc = 2 * N_val  # D_y == D_lc since we dropped Q from the cINN target
    x_norm = torch.from_numpy(val_subset['x_norm']).float()
    MBSZ = 64
    n = x_norm.shape[0]
    all_y = []
    for s in range(0, n, MBSZ):
        all_y.append(sample_batch(inn, embedder, x_norm[s:s+MBSZ], K, D_y,
                                  y_scaler_mean, y_scaler_std, device))
    y_log_samples = np.concatenate(all_y, axis=0)  # (n, K, D_lc)

    rt_best, rt_all = [], []
    for i in range(n):
        fc  = float(val_subset['fc_GHz'][i])
        s21 = val_subset['s21_target'][i]
        s11 = val_subset['s11_target'][i]

        # Ground-truth Q for honest synthesis
        qL = val_subset['Q_L'][i, :N_val]
        qC = val_subset['Q_C'][i, :N_val]

        # Ground-truth parasitics for honest eval
        y_true  = val_subset['y'][i]
        L_true  = y_true[0::2][:N_val]
        C_true  = y_true[1::2][:N_val]
        k_m_row = val_subset['k_m'][i][:N_val - 1]
        csf_row = val_subset['C_sub_frac'][i][:N_val]
        aC_row  = val_subset['alpha_C'][i][:N_val]
        coup_M  = k_m_row * np.minimum(L_true[:-1], L_true[1:])
        C_sub_i = csf_row * C_true

        rts = []
        for k in range(K):
            lc = y_log_samples[i, k]   # D_lc-dim LC prediction
            L = 10.0 ** lc[0::2]; C = 10.0 ** lc[1::2]
            rt = roundtrip_mse_lc(L, C, N_val, fc, qL, qC, s21, s11, FREQ_HZ,
                                   coupling_M=coup_M, C_sub=C_sub_i, alpha_C=aC_row)
            rts.append(rt if np.isfinite(rt) else RT_THRESHOLD * 10)
        rt_best.append(min(rts)); rt_all.extend(rts)

    rt_best = np.array(rt_best); rt_all = np.array(rt_all)
    return (float(np.mean(rt_best)), float(np.mean(rt_all < RT_THRESHOLD)),
            float(np.mean(rt_all)), float(np.median(rt_all)))


# ── Training loop (per N) ─────────────────────────────────────────────────────

def train_one_cinn(
    N_val, train_idx_N, val_idx_N, ds,
    x_scaler, y_scaler, device,
    checkpoint_path, log_path, figures_dir,
):
    D_lc = 2 * N_val
    D_y  = D_lc   # LC only — Q dropped from cINN target (see module docstring)
    y_scaler_mean = y_scaler.mean_.astype(np.float32)
    y_scaler_std  = y_scaler.scale_.astype(np.float32)

    print(f'\n{"="*60}')
    print(f'  Training V2 cINN  N={N_val}  D_y={D_y} (LC only)  '
          f'blocks={N_BLOCKS}  subnet={SUBNET_DIM}  cond={COND_DIM}')
    print(f'  {len(train_idx_N)} train / {len(val_idx_N)} val')
    print(f'{"="*60}')

    def _dummy_scaler(n):
        d = StandardScaler()
        d.mean_  = np.zeros(2*n, dtype=np.float32)
        d.scale_ = np.ones(2*n, dtype=np.float32)
        d.var_   = np.ones(2*n, dtype=np.float32)
        d.n_features_in_ = 2*n
        return d

    # FilterDataset is shared with MLP and uses 207-dim X_full. For cINN with
    # Bundle 1 parasitic conditioning, we build a 231-dim X separately and
    # override the dataset's self.X after construction.
    # _dummy_x_scaler ensures FilterDataset's 207-dim transform is bypassed.
    _dummy_x_scaler = StandardScaler()
    _dummy_x_scaler.mean_  = np.zeros(207, dtype=np.float32)
    _dummy_x_scaler.scale_ = np.ones(207, dtype=np.float32)
    _dummy_x_scaler.var_   = np.ones(207, dtype=np.float32)
    _dummy_x_scaler.n_features_in_ = 207

    train_ds = FilterDataset(
        train_idx_N, ds, _dummy_x_scaler,
        _dummy_scaler(3), _dummy_scaler(4), _dummy_scaler(5),
        use_scalar_x=False,
    )
    val_ds = FilterDataset(
        val_idx_N, ds, _dummy_x_scaler,
        _dummy_scaler(3), _dummy_scaler(4), _dummy_scaler(5),
        use_scalar_x=False,
    )
    # Bundle 1: override self.X with 231-dim parasitic-conditioned input
    train_ds.X = torch.from_numpy(
        x_scaler.transform(build_full_x(ds, train_idx_N)).astype(np.float32)
    )
    val_ds.X = torch.from_numpy(
        x_scaler.transform(build_full_x(ds, val_idx_N)).astype(np.float32)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Pre-extract val subset
    rng = np.random.RandomState(SEED)
    eval_n = min(K_EVAL_NSAMPLE, len(val_idx_N))
    eval_idx = rng.choice(len(val_idx_N), size=eval_n, replace=False)
    ev_idx_abs = val_idx_N[eval_idx]

    val_subset = {
        'x_norm':     x_scaler.transform(
            build_full_x(ds, ev_idx_abs)
        ).astype(np.float32),
        'fc_GHz':     ds['fc_GHz'][ev_idx_abs].astype(np.float32),
        's21_target': ds['X_full'][ev_idx_abs, 5:106].astype(np.float32),
        's11_target': ds['X_full'][ev_idx_abs, 106:207].astype(np.float32),
        # Ground-truth Q for synthesis (not predicted — Q is process metadata)
        'Q_L':        ds['Q_L'][ev_idx_abs].astype(np.float32),
        'Q_C':        ds['Q_C'][ev_idx_abs].astype(np.float32),
        # Ground-truth parasitics for honest eval
        'k_m':        ds['k_m'][ev_idx_abs],
        'C_sub_frac': ds['C_sub_frac'][ev_idx_abs],
        'alpha_C':    ds['alpha_C'][ev_idx_abs],
        'y':          ds['y'][ev_idx_abs],      # actual LC in SI for parasitic calc
    }

    # Build model — Bundle 1: embedder takes 231-dim input (X_full + 24 parasitics)
    inn      = make_cinn_v2(D_y, D_c=COND_DIM, n_blocks=N_BLOCKS,
                             subnet_dim=SUBNET_DIM, affine_clamping=AFFINE_CLAMP)
    embedder = ConditionEmbedderV2(input_dim=COND_INPUT_DIM, cond_dim=COND_DIM)
    inn.to(device); embedder.to(device)
    fix_mps_contiguity(inn)

    emb_p = sum(p.numel() for p in embedder.parameters())
    inn_p = sum(p.numel() for p in inn.parameters())
    print(f'  Params: embedder={emb_p:,}  inn={inn_p:,}  total={emb_p+inn_p:,}')

    print(f'  Bijection check (D_y={D_y}) ...', end=' ', flush=True)
    err = verify_bijection(inn, embedder, D_y, device, tol=1e-4)
    print(f'max err = {err:.2e}  ✓')

    params = list(inn.parameters()) + list(embedder.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=LR_PATIENCE, factor=LR_FACTOR,
    )

    mean_t = torch.tensor(y_scaler_mean, dtype=torch.float32, device=device)
    std_t  = torch.tensor(y_scaler_std,  dtype=torch.float32, device=device)

    best_acc_frac = -1.0; best_val_rt_mse = float('inf')
    no_improve = 0; diverged = False
    last_good_state = {'inn': inn.state_dict(), 'emb': embedder.state_dict()}

    history = {k: [] for k in ['train_nll','val_nll','z_mean','z_std',
                                 'logdet_mean','grad_norm','lr',
                                 'val_rt_mse','val_acc_frac']}

    with open(log_path, 'w', newline='') as lf:
        writer = csv.writer(lf)
        writer.writerow(['epoch','train_nll','val_nll','z_mean','z_std',
                         'logdet_mean','grad_norm','lr','val_rt_mse','val_acc_frac'])

        pbar = tqdm(range(1, MAX_EPOCHS + 1), desc=f'V2 N={N_val}')
        for epoch in pbar:
            # Train
            inn.train(); embedder.train()
            t_nll_total = 0.0; ep_grad_norm = 0.0
            for batch in train_loader:
                x_b      = batch['x'].to(device)
                y_b      = (batch['y_log'][:, :D_lc].to(device) - mean_t) / std_t

                c = embedder(x_b)
                z, logdet = inn(y_b, c=[c])
                nll = 0.5 * (z ** 2).sum(dim=1).mean() - logdet.mean()

                # Bundle 4: z_std regularization — pushes latent to N(0, I)
                # Computed per-dim std across batch, averaged across dims.
                z_std_batch = z.std(dim=0).mean()
                z_reg = (z_std_batch - 1.0) ** 2
                loss = nll + Z_REG_LAMBDA * z_reg

                if not torch.isfinite(loss) or loss.item() > NLL_NAN_GUARD:
                    print(f'\n  DIVERGENCE epoch={epoch} loss={loss.item():.3e}')
                    dp = checkpoint_path.replace('.pt', '_diverged.pt')
                    torch.save({'inn_state_dict': last_good_state['inn'],
                                'embedder_state_dict': last_good_state['emb'],
                                'epoch': epoch}, dp)
                    with open(dp.replace('.pt', '_diag.json'), 'w') as df:
                        json.dump({'epoch': epoch, 'nll': float(nll.item()),
                                   'N': N_val, 'D_y': D_y}, df, indent=2)
                    diverged = True; break

                optimizer.zero_grad()
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP).item()
                optimizer.step()
                t_nll_total += nll.item(); ep_grad_norm += gn

            if diverged:
                break

            # Validate
            inn.eval(); embedder.eval()
            v_nll_total = 0.0; z_list = []; ld_list = []
            with torch.no_grad():
                for batch in val_loader:
                    x_b  = batch['x'].to(device)
                    y_b  = (batch['y_log'][:, :D_lc].to(device) - mean_t) / std_t
                    c = embedder(x_b)
                    z_v, ld_v = inn(y_b, c=[c])
                    v_nll_total += (0.5*(z_v**2).sum(1).mean() - ld_v.mean()).item()
                    z_list.append(z_v.cpu()); ld_list.append(ld_v.cpu())

            z_cat = torch.cat(z_list); ld_cat = torch.cat(ld_list)
            t_nll = t_nll_total / len(train_loader)
            v_nll = v_nll_total / len(val_loader)
            z_mean = float(z_cat.mean()); z_std = float(z_cat.std())
            ld_mean = float(ld_cat.mean())
            ep_gn = ep_grad_norm / len(train_loader)
            cur_lr = optimizer.param_groups[0]['lr']
            scheduler.step(v_nll)

            val_rt_mse_ep = None; val_acc_frac_ep = None
            if epoch % K_EVAL_EVERY == 0:
                bk, af, _, _ = eval_rt_mse_fast(
                    inn, embedder, val_subset, K_EVAL, D_y, N_val,
                    y_scaler_mean, y_scaler_std, device
                )
                val_rt_mse_ep = bk; val_acc_frac_ep = af
                improved = (val_acc_frac_ep > best_acc_frac or
                            (val_acc_frac_ep == best_acc_frac and bk < best_val_rt_mse))
                if improved:
                    best_acc_frac = val_acc_frac_ep; best_val_rt_mse = bk
                    no_improve = 0
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

            for k, v in [('train_nll', t_nll), ('val_nll', v_nll),
                          ('z_mean', z_mean), ('z_std', z_std),
                          ('logdet_mean', ld_mean), ('grad_norm', ep_gn),
                          ('lr', cur_lr), ('val_rt_mse', val_rt_mse_ep),
                          ('val_acc_frac', val_acc_frac_ep)]:
                history[k].append(v)

            writer.writerow([epoch, t_nll, v_nll, z_mean, z_std, ld_mean,
                             ep_gn, cur_lr, val_rt_mse_ep or '', val_acc_frac_ep or ''])
            lf.flush()

            pbar.set_postfix({'tr_nll': f'{t_nll:.2f}', 'vl_nll': f'{v_nll:.2f}',
                              'z_std': f'{z_std:.2f}', 'best_af': f'{best_acc_frac:.2f}',
                              'pat': no_improve})

            if no_improve >= ES_PATIENCE:
                print(f'\n  Early stopping epoch={epoch} '
                      f'acc_frac={best_acc_frac:.3f} rt_mse={best_val_rt_mse:.2f}')
                break

    # Post-training bijection check
    if not diverged and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, weights_only=False)
        inn.load_state_dict(ckpt['inn_state_dict'])
        embedder.load_state_dict(ckpt['embedder_state_dict'])
        inn.to(device); embedder.to(device)
        fix_mps_contiguity(inn)
        print('  Post-training bijection check ...', end=' ', flush=True)
        try:
            err = verify_bijection(inn, embedder, D_y, device, tol=1e-3)
            print(f'max err = {err:.2e}  ✓')
        except AssertionError as e:
            print(f'WARN: {e}')

    return {'history': history, 'best_val_rt_mse': best_val_rt_mse,
            'best_acc_frac': best_acc_frac, 'diverged': diverged}


# ── Final test evaluation ─────────────────────────────────────────────────────

def evaluate_test_set(N_val, test_idx_N, ds, x_scaler, y_scaler,
                      checkpoint_path, device):
    D_lc = 2 * N_val
    D_y  = D_lc   # LC only — Q dropped from cINN target
    ckpt = torch.load(checkpoint_path, weights_only=False)

    inn      = make_cinn_v2(D_y, D_c=COND_DIM, n_blocks=N_BLOCKS,
                             subnet_dim=SUBNET_DIM, affine_clamping=AFFINE_CLAMP)
    embedder = ConditionEmbedderV2(input_dim=COND_INPUT_DIM, cond_dim=COND_DIM)
    inn.load_state_dict(ckpt['inn_state_dict'])
    embedder.load_state_dict(ckpt['embedder_state_dict'])
    inn.to(device); embedder.to(device)
    fix_mps_contiguity(inn)
    inn.eval(); embedder.eval()

    y_scaler_mean = ckpt['y_scaler_mean']
    y_scaler_std  = ckpt['y_scaler_std']

    x_raw  = ds['X_full'][test_idx_N]
    # Bundle 1: build 231-dim parasitic-conditioned input
    x_norm = x_scaler.transform(build_full_x(ds, test_idx_N)).astype(np.float32)
    y_log_true_lc = ds['y_log'][test_idx_N, :D_lc].astype(np.float32)
    y_true_vals   = ds['y'][test_idx_N]    # actual LC in SI
    q_L_true      = ds['Q_L'][test_idx_N, :N_val].astype(np.float32)
    q_C_true      = ds['Q_C'][test_idx_N, :N_val].astype(np.float32)
    fc_arr  = ds['fc_GHz'][test_idx_N].astype(np.float32)
    s21_arr = x_raw[:, 5:106].astype(np.float32)
    s11_arr = x_raw[:, 106:207].astype(np.float32)
    k_m_test      = ds['k_m'][test_idx_N]
    c_sub_frac_test = ds['C_sub_frac'][test_idx_N]
    alpha_C_test  = ds['alpha_C'][test_idx_N]
    n_test = len(test_idx_N)

    MBSZ = 32
    y_log_K = []; y_log_div = []
    for s in range(0, n_test, MBSZ):
        xb = torch.from_numpy(x_norm[s:s+MBSZ]).float()
        y_log_K.append(sample_batch(inn, embedder, xb, K_INFERENCE, D_y,
                                    y_scaler_mean, y_scaler_std, device))
    for s in range(0, n_test, MBSZ):
        xb = torch.from_numpy(x_norm[s:s+MBSZ]).float()
        y_log_div.append(sample_batch(inn, embedder, xb, K_DIVERSITY, D_y,
                                      y_scaler_mean, y_scaler_std, device))
    y_log_K   = np.concatenate(y_log_K,   axis=0)  # (n_test, K_INFERENCE, D_lc)
    y_log_div = np.concatenate(y_log_div, axis=0)  # (n_test, K_DIVERSITY, D_lc)

    # Posterior NLL on y_true (LC only)
    nll_list = []
    with torch.no_grad():
        for s in range(0, n_test, MBSZ):
            xb = torch.from_numpy(x_norm[s:s+MBSZ]).float().to(device)
            yb = torch.from_numpy(
                ((y_log_true_lc[s:s+MBSZ] - y_scaler_mean) / y_scaler_std)
            ).float().to(device)
            c = embedder(xb)
            z_t, ld_t = inn(yb, c=[c])
            nll_list.append((0.5*(z_t**2).sum(1) - ld_t).cpu().numpy())
    nll_true = np.concatenate(nll_list)

    def _parasitic_kwargs(i):
        """Ground-truth parasitic kwargs for sample i."""
        y_i    = y_true_vals[i]
        L_i    = y_i[0::2][:N_val]
        C_i    = y_i[1::2][:N_val]
        coup_M = k_m_test[i][:N_val-1] * np.minimum(L_i[:-1], L_i[1:])
        C_sub  = c_sub_frac_test[i][:N_val] * C_i
        return dict(coupling_M=coup_M, C_sub=C_sub, alpha_C=alpha_C_test[i][:N_val])

    # rt_mse: use predicted LC + ground-truth Q + ground-truth parasitics
    rt_best_list = []; rt_all_list = []
    for i in range(n_test):
        fc = float(fc_arr[i]); s21 = s21_arr[i]; s11 = s11_arr[i]
        pk = _parasitic_kwargs(i)
        qL = q_L_true[i]; qC = q_C_true[i]
        rts = []
        for k in range(K_INFERENCE):
            lc = y_log_K[i, k]   # D_lc-dim LC prediction
            L = 10.0 ** lc[0::2]; C = 10.0 ** lc[1::2]
            rt = roundtrip_mse_lc(L, C, N_val, fc, qL, qC, s21, s11, FREQ_HZ, **pk)
            rts.append(rt if np.isfinite(rt) else RT_THRESHOLD * 10)
        rt_best_list.append(min(rts)); rt_all_list.extend(rts)
    rt_best = np.array(rt_best_list); rt_all = np.array(rt_all_list)

    # z=0 prediction (best-of-1, deterministic)
    rt_z0_list = []
    with torch.no_grad():
        for s in range(0, n_test, MBSZ):
            xb = torch.from_numpy(x_norm[s:s+MBSZ]).float().to(device)
            c  = embedder(xb); bsz = xb.shape[0]
            y_z0_norm, _ = inn(torch.zeros(bsz, D_y, device=device), c=[c], rev=True)
            y_z0 = y_z0_norm.cpu().numpy() * y_scaler_std + y_scaler_mean
            for bi in range(bsz):
                i  = s + bi
                L  = 10.0 ** y_z0[bi, 0::2]; C = 10.0 ** y_z0[bi, 1::2]
                pk = _parasitic_kwargs(i)
                rt = roundtrip_mse_lc(L, C, N_val, float(fc_arr[i]),
                                      q_L_true[i], q_C_true[i],
                                      s21_arr[i], s11_arr[i], FREQ_HZ, **pk)
                rt_z0_list.append(rt if np.isfinite(rt) else RT_THRESHOLD * 10)
    rt_z0 = np.array(rt_z0_list)

    # LC comp_mse / R²
    y_log_lc_mean = y_log_K.mean(axis=1)   # (n_test, D_lc)
    comp_mse_val = component_mse(y_log_lc_mean, y_log_true_lc)
    r2_val   = r2_per_component(y_log_lc_mean, y_log_true_lc)
    r2_mean  = float(np.nanmean(r2_val))

    # Diversity (in normalized LC space)
    y_norm = (y_log_div - y_scaler_mean[None,None,:]) / y_scaler_std[None,None,:]
    diversity_mean = float(y_norm.std(axis=1).mean(axis=1).mean())

    # Mode count
    rng = np.random.RandomState(SEED)
    mode_counts = []
    mode_samples_for_plot = []
    for ci in rng.choice(n_test, size=min(5, n_test), replace=False):
        sn = (y_log_div[ci] - y_scaler_mean) / y_scaler_std
        db = DBSCAN(eps=0.3, min_samples=3).fit(sn)
        nc = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        mode_counts.append(max(nc, 1))
        mode_samples_for_plot.append({
            'y_log_samples': y_log_div[ci],  # (K_DIVERSITY, D_lc)
            'q_L_true': q_L_true[ci],
            'q_C_true': q_C_true[ci],
            'fc_GHz': float(fc_arr[ci]),
            's21_target': s21_arr[ci],
            'N': N_val,
        })

    return {
        'N': N_val,
        'rt_best_of_K': float(np.mean(rt_best)),
        'rt_mean':       float(np.mean(rt_all)),
        'rt_median':     float(np.median(rt_all)),
        'rt_z0':         float(np.mean(rt_z0)),
        'acc_frac':      float(np.mean(rt_all < RT_THRESHOLD)),
        'diversity':     diversity_mean,
        'mode_count':    float(np.mean(mode_counts)),
        'nll_true_mean': float(np.mean(nll_true)),
        'nll_true_std':  float(np.std(nll_true)),
        'comp_mse':      comp_mse_val,
        'r2_mean':       r2_mean,
        'y_log_K':       y_log_K,       # (n_test, K_INFERENCE, D_lc)
        'y_log_true_lc': y_log_true_lc,
        'mode_samples':  mode_samples_for_plot,
        'rt_best_arr':   rt_best,
        'fc_arr':        fc_arr,
        's21_arr':       s21_arr,
        's11_arr':       s11_arr,
        'q_L_true':      q_L_true,
        'q_C_true':      q_C_true,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def save_loss_plot(history, N_val, figures_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    ep = range(1, len(history['train_nll']) + 1)
    axes[0].plot(ep, history['train_nll'], label='Train NLL')
    axes[0].plot(ep, history['val_nll'],   label='Val NLL')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('NLL')
    axes[0].set_title(f'V2 N={N_val} — NLL'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(ep, history['z_std'],  color='tab:orange', label='z_std')
    axes[1].axhline(1.0, color='k', ls='--', alpha=0.5)
    axes[1].plot(ep, history['z_mean'], color='tab:blue',   label='z_mean')
    axes[1].axhline(0.0, color='k', ls=':', alpha=0.5)
    axes[1].set_title(f'V2 N={N_val} — z stats'); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot(ep, history['logdet_mean'], color='tab:green')
    axes[2].set_title(f'V2 N={N_val} — log|det J|'); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'{PREFIX}_N{N_val}_loss.png'),
                dpi=150, bbox_inches='tight'); plt.close()


def save_diversity_plot(results_by_N, figures_dir):
    for N_val, res in results_by_N.items():
        D_lc = 2 * N_val
        mode_samples = res['mode_samples'][:3]
        if not mode_samples: continue
        fig, axes = plt.subplots(1, len(mode_samples),
                                 figsize=(6*len(mode_samples), 4), sharey=True)
        if len(mode_samples) == 1: axes = [axes]
        for ax, ms in zip(axes, mode_samples):
            fc = ms['fc_GHz']; s21_tgt = ms['s21_target']
            qL = ms['q_L_true']; qC = ms['q_C_true']
            ax.plot(FREQ_HZ/1e9, s21_tgt, 'k-', lw=2, label='Target', zorder=5)
            for k in range(ms['y_log_samples'].shape[0]):
                lc = ms['y_log_samples'][k]   # D_lc-dim LC
                L = 10.0**lc[0::2]; C = 10.0**lc[1::2]
                s21, _ = synthesize_from_lc(L, C, N_val, fc, qL, qC, FREQ_HZ)
                if s21 is not None:
                    ax.plot(FREQ_HZ/1e9, s21, alpha=0.3, lw=0.8, color='tab:blue')
            ax.set_xlabel('Freq (GHz)'); ax.set_ylabel('S21 (dB)')
            ax.set_ylim(-60, 3)
            ax.set_title(f'N={N_val}, fc={fc:.1f} GHz')
            ax.grid(alpha=0.3)
        handles = [plt.Line2D([0],[0], color='k', lw=2, label='Target'),
                   plt.Line2D([0],[0], color='tab:blue', alpha=0.5, lw=1, label='V2 samples')]
        axes[-1].legend(handles=handles)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f'{PREFIX}_diversity_N{N_val}.png'),
                    dpi=150, bbox_inches='tight'); plt.close()


def save_scatter_plot(results_by_N, figures_dir):
    from evaluation.visualize import plot_scatter
    all_pred, all_true, all_N = [], [], []
    for N_val, res in results_by_N.items():
        D_lc = 2 * N_val; n = res['y_log_true_lc'].shape[0]
        yp = np.full((n, 10), np.nan); yt = np.full((n, 10), np.nan)
        yp[:, :D_lc] = res['y_log_K'][:, :, :D_lc].mean(axis=1)
        yt[:, :D_lc] = res['y_log_true_lc']
        all_pred.append(yp); all_true.append(yt)
        all_N.append(np.full(n, N_val, dtype=int))
    plot_scatter(np.concatenate(all_pred), np.concatenate(all_true),
                 np.concatenate(all_N),
                 os.path.join(figures_dir, f'{PREFIX}_scatter.png'))


def save_roundtrip_plot(results_by_N, figures_dir):
    from evaluation.visualize import plot_roundtrip
    samples = []; rng = np.random.RandomState(0)
    for N_val, res in results_by_N.items():
        for ci in rng.choice(res['y_log_true_lc'].shape[0], size=min(2, len(res['fc_arr'])), replace=False):
            fc  = float(res['fc_arr'][ci])
            lc  = res['y_log_K'][ci].mean(axis=0)  # mean of K samples, D_lc-dim
            L   = 10.0 ** lc[0::2]; C = 10.0 ** lc[1::2]
            qL  = res['q_L_true'][ci]; qC = res['q_C_true'][ci]
            s21p, _ = synthesize_from_lc(L, C, N_val, fc, qL, qC, FREQ_HZ)
            samples.append({'s21_target': res['s21_arr'][ci],
                            's21_pred': s21p if s21p is not None else np.zeros(101),
                            'N': N_val, 'ripple_dB': 0.0, 'fc_GHz': fc, 'fbw': 0.2,
                            'rt_mse': res['rt_best_arr'][ci]})
    if samples:
        plot_roundtrip(samples, os.path.join(figures_dir, f'{PREFIX}_roundtrip.png'),
                       model_label='cINN V2')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path   = os.path.join(project_root, 'data', 'dataset_otfl301v2.pkl')
    results_dir = os.path.join(project_root, 'results')
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Architecture: N_BLOCKS={N_BLOCKS}  SUBNET_DIM={SUBNET_DIM}  COND_DIM={COND_DIM}')

    with open(data_path, 'rb') as f:
        ds = pickle.load(f)
    n_total = len(ds['N'])
    print(f'Loaded {n_total} samples')

    all_idx = np.arange(n_total)
    all_idx = all_idx[ds['N'][all_idx] != 4]   # keep N=3 and N=5 only (N=4 physically impractical)
    train_idx, temp_idx = train_test_split(
        all_idx, test_size=0.20, stratify=ds['N'][all_idx], random_state=42)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=ds['N'][temp_idx], random_state=42)
    print(f'Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}')

    # Bundle 1: fit x_scaler on 231-dim parasitic-conditioned input
    x_scaler = StandardScaler().fit(build_full_x(ds, train_idx))
    assert x_scaler.mean_.shape == (COND_INPUT_DIM,), \
        f'x_scaler dim {x_scaler.mean_.shape} != expected ({COND_INPUT_DIM},)'

    y_scalers = {}
    for N_val in [3, 5]:
        mask = ds['N'][train_idx] == N_val
        # Fit on LC only (D_y = 2*N); Q is no longer part of the cINN target
        y_scalers[N_val] = StandardScaler().fit(
            ds['y_log'][train_idx[mask], :2*N_val].astype(np.float32)
        )

    train_summaries = {}
    for N_val in [3, 5]:
        train_idx_N = train_idx[ds['N'][train_idx] == N_val]
        val_idx_N   = val_idx[ds['N'][val_idx]   == N_val]
        ckpt_path   = os.path.join(results_dir, f'{PREFIX}_N{N_val}_best.pt')
        log_path    = os.path.join(results_dir, f'{PREFIX}_N{N_val}_log.csv')

        if os.path.exists(ckpt_path):
            print(f'\n  N={N_val}: checkpoint exists at {ckpt_path} — skipping.')
            ckpt = torch.load(ckpt_path, weights_only=False)
            train_summaries[N_val] = {
                'history': {k: [] for k in ['train_nll','val_nll','z_mean','z_std',
                                             'logdet_mean','grad_norm','lr',
                                             'val_rt_mse','val_acc_frac']},
                'best_val_rt_mse': ckpt.get('best_val_rt_mse', float('inf')),
                'best_acc_frac':   ckpt.get('best_acc_frac', -1.0),
                'diverged': False,
            }
            continue

        summary = train_one_cinn(
            N_val, train_idx_N, val_idx_N,
            ds, x_scaler, y_scalers[N_val],
            device, ckpt_path, log_path, figures_dir,
        )
        train_summaries[N_val] = summary
        save_loss_plot(summary['history'], N_val, figures_dir)

    # Final test evaluation
    print('\n' + '='*60)
    print('  Final test-set evaluation (K=50, ground-truth Q)')
    print('='*60)

    results_by_N = {}
    for N_val in [3, 5]:
        ckpt_path = os.path.join(results_dir, f'{PREFIX}_N{N_val}_best.pt')
        if not os.path.exists(ckpt_path):
            print(f'  N={N_val}: no checkpoint — skipping.'); continue
        test_idx_N = test_idx[ds['N'][test_idx] == N_val]
        print(f'  Evaluating N={N_val} ({len(test_idx_N)} samples) ...')
        results_by_N[N_val] = evaluate_test_set(
            N_val, test_idx_N, ds, x_scaler, y_scalers[N_val], ckpt_path, device)

    # Print metrics table
    phase4cq_rt = {3: 0.9754, 5: 0.8808}
    mlp_rt      = {3: 36.49,  5: 39.69}

    header = f"{'Metric':<28} | {'N=3':>10} | {'N=5':>10}"
    sep    = '-' * len(header)
    lines  = [sep, header, sep]

    def row(label, fn):
        vals = [f'{fn(results_by_N[N]):.4f}' if N in results_by_N else 'N/A'
                for N in [3, 5]]
        return f'{label:<28} | {vals[0]:>10} | {vals[1]:>10}'

    lines.append(f"{'MLP baseline rt_mse':<28} | {mlp_rt[3]:>10.2f} | {mlp_rt[5]:>10.2f}")
    lines.append(f"{'cINN+Q (V1) best-of-50':<28} | {phase4cq_rt[3]:>10.4f} | {phase4cq_rt[5]:>10.4f}")
    lines.append(sep)
    lines.append(row('cINN V2 best-of-50',      lambda r: r['rt_best_of_K']))
    lines.append(row('cINN V2 mean rt_mse',     lambda r: r['rt_mean']))
    lines.append(row('cINN V2 median rt_mse',   lambda r: r['rt_median']))
    lines.append(row('cINN V2 z=0 rt_mse',      lambda r: r['rt_z0']))
    lines.append(row(f'acc_frac (<{RT_THRESHOLD:.0f} dB², gnd-trth Q)', lambda r: r['acc_frac']))
    lines.append(row('diversity (norm std)',     lambda r: r['diversity']))
    lines.append(row('mode count (DBSCAN)',      lambda r: r['mode_count']))
    lines.append(row('NLL on y_true (LC)',       lambda r: r['nll_true_mean']))
    lines.append(row('comp_mse LC (log)',        lambda r: r['comp_mse']))
    lines.append(row('r2_mean LC',               lambda r: r['r2_mean']))
    lines.append(sep)

    table_str = '\n'.join(lines)
    print('\n' + table_str)

    table_path = os.path.join(results_dir, 'inn_v2_metrics_table.txt')
    with open(table_path, 'w') as tf:
        tf.write(table_str + '\n')
    print(f'\nMetrics table saved → {table_path}')

    if results_by_N:
        save_scatter_plot(results_by_N, figures_dir)
        save_roundtrip_plot(results_by_N, figures_dir)
        save_diversity_plot(results_by_N, figures_dir)

    print('\nPhase 4c-V2 complete.')


if __name__ == '__main__':
    main()
