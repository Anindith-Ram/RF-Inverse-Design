"""
Generate final demo figures and benchmark table.

Loads: dataset_otfl301v2.pkl, mlp_otfl301v2_best.pt,
       inn_v2_otfl301v2_N3_best.pt, inn_v2_otfl301v2_N5_best.pt

Produces 5 figures + results/final_benchmark_table.txt

Usage:
    source rf_env/bin/activate
    python evaluation/make_final_figures.py
"""

import os
import sys
import pickle
import warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mlp import InverseMLP, SpecsOnlyMLP
from models.inn_v2 import ConditionEmbedderV2, make_cinn_v2, fix_mps_contiguity
from evaluation.metrics import (
    synthesize_from_lc, roundtrip_mse_lc, component_mse, r2_per_component,
    FREQ_HZ_OTFL301 as FREQ_HZ,
)
from training.train_mlp import FilterDataset, predict_all, evaluate_model
from training.train_inn_v2 import build_full_x

# ── Config ────────────────────────────────────────────────────────────────────
N_BLOCKS    = 8         # Bundle 5
SUBNET_DIM  = 128       # Bundle 5
COND_DIM    = 128
COND_INPUT_DIM = 231    # Bundle 1: X_full + parasitics
AFFINE_CLAMP = 1.5
K_INFERENCE = 50
K_DIVERSITY = 20
RT_THRESHOLD = 5.0
SEED        = 42

FIGSIZE_WIDE = (14, 5)


def setup_paths():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path   = os.path.join(root, 'data', 'dataset_otfl301v2.pkl')
    results_dir = os.path.join(root, 'results')
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    ckpt = lambda name: os.path.join(results_dir, name)
    fig  = lambda name: os.path.join(figures_dir, name)
    return data_path, results_dir, figures_dir, ckpt, fig


# ── Dataset / split ───────────────────────────────────────────────────────────

def load_dataset_and_split(data_path):
    with open(data_path, 'rb') as f:
        ds = pickle.load(f)
    n = len(ds['N'])
    all_idx = np.arange(n)
    train_idx, temp = train_test_split(all_idx, test_size=0.20, stratify=ds['N'], random_state=42)
    val_idx, test_idx = train_test_split(temp, test_size=0.50, stratify=ds['N'][temp], random_state=42)

    # MLP scalers fit on 207/5-dim inputs; cINN scaler fits on 231-dim parasitic-conditioned input
    x_scaler_mlp  = StandardScaler().fit(ds['X_full'][train_idx])
    xs_scaler     = StandardScaler().fit(ds['X_scalar'][train_idx])
    x_scaler_inn  = StandardScaler().fit(build_full_x(ds, train_idx))
    tN = ds['N'][train_idx]; tlog = ds['y_log'][train_idx]
    ys3 = StandardScaler().fit(tlog[tN==3, :6])
    ys4 = StandardScaler().fit(tlog[tN==3, :6])  # dummy (no N=4)
    ys5 = StandardScaler().fit(tlog[tN==5, :10])

    return ds, train_idx, val_idx, test_idx, x_scaler_mlp, xs_scaler, x_scaler_inn, ys3, ys4, ys5


# ── MLP inference ─────────────────────────────────────────────────────────────

def run_mlp(ds, test_idx, x_scaler, xs_scaler, ys3, ys4, ys5, ckpt_full, ckpt_scalar, device):
    from torch.utils.data import DataLoader
    full_test_ds = FilterDataset(test_idx, ds, x_scaler, ys3, ys4, ys5, use_scalar_x=False)
    scal_test_ds = FilterDataset(test_idx, ds, xs_scaler, ys3, ys4, ys5, use_scalar_x=True)
    full_ld = DataLoader(full_test_ds, batch_size=256, shuffle=False)
    scal_ld = DataLoader(scal_test_ds, batch_size=256, shuffle=False)

    mlp_full = InverseMLP(input_dim=207).to(device)
    mlp_full.load_state_dict(torch.load(ckpt_full, weights_only=True, map_location=device))

    mlp_scal = SpecsOnlyMLP().to(device)
    mlp_scal.load_state_dict(torch.load(ckpt_scalar, weights_only=True, map_location=device))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yp_full, yt_full, meta_full = predict_all(mlp_full, full_ld, device, ys3, ys4, ys5)
        yp_scal, yt_scal, meta_scal = predict_all(mlp_scal, scal_ld, device, ys3, ys4, ys5)

    res_full = evaluate_model(yp_full, yt_full, meta_full)
    res_scal = evaluate_model(yp_scal, yt_scal, meta_scal)

    return (yp_full, yt_full, meta_full, res_full,
            yp_scal, yt_scal, meta_scal, res_scal)


# ── cINN inference ────────────────────────────────────────────────────────────

def _sample_inn(inn, embedder, x_norm, K, D_y, y_mean, y_std, device, mbsz=32):
    """Sample K candidates for each row in x_norm. Returns (n, K, D_y)."""
    inn.eval(); embedder.eval()
    n = x_norm.shape[0]
    out = []
    with torch.no_grad():
        for s in range(0, n, mbsz):
            xb = torch.from_numpy(x_norm[s:s+mbsz]).float().to(device)
            bsz = xb.shape[0]
            c = embedder(xb).unsqueeze(1).expand(-1, K, -1).reshape(bsz*K, -1)
            z = torch.randn(bsz*K, D_y, device=device)
            y_norm_flat, _ = inn(z, c=[c], rev=True)
            y_flat = y_norm_flat.cpu().numpy() * y_std + y_mean
            out.append(y_flat.reshape(bsz, K, D_y))
    return np.concatenate(out, axis=0)  # (n, K, D_y)


def run_cinn(N_val, ds, test_idx, x_scaler, ckpt_path, device):
    """Run cINN inference on test set for one N_val. Returns results dict.

    D_y = 2*N (LC only). Q is ground-truth from dataset — not predicted by cINN.
    """
    D_lc = 2 * N_val
    D_y  = D_lc   # LC only

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    inn  = make_cinn_v2(D_y, D_c=COND_DIM, n_blocks=N_BLOCKS,
                        subnet_dim=SUBNET_DIM, affine_clamping=AFFINE_CLAMP)
    emb  = ConditionEmbedderV2(input_dim=COND_INPUT_DIM, cond_dim=COND_DIM)
    inn.load_state_dict(ckpt['inn_state_dict'])
    emb.load_state_dict(ckpt['embedder_state_dict'])
    inn.to(device); emb.to(device)
    fix_mps_contiguity(inn)
    y_mean = ckpt['y_scaler_mean']; y_std = ckpt['y_scaler_std']

    test_N = ds['N'][test_idx] == N_val
    idx_N  = test_idx[test_N]
    n_test = len(idx_N)

    # Bundle 1: 231-dim parasitic-conditioned input
    x_norm        = x_scaler.transform(build_full_x(ds, idx_N)).astype(np.float32)
    fc_arr        = ds['fc_GHz'][idx_N]
    s21_arr       = ds['X_full'][idx_N, 5:106].astype(np.float32)
    s11_arr       = ds['X_full'][idx_N, 106:207].astype(np.float32)
    y_true_vals   = ds['y'][idx_N]
    y_log_true_lc = ds['y_log'][idx_N, :D_lc].astype(np.float32)
    q_L_true      = ds['Q_L'][idx_N, :N_val].astype(np.float32)
    q_C_true      = ds['Q_C'][idx_N, :N_val].astype(np.float32)
    k_m_t         = ds['k_m'][idx_N]
    csf_t         = ds['C_sub_frac'][idx_N]
    aC_t          = ds['alpha_C'][idx_N]

    y_log_K   = _sample_inn(inn, emb, x_norm, K_INFERENCE, D_y, y_mean, y_std, device)
    y_log_div = _sample_inn(inn, emb, x_norm, K_DIVERSITY, D_y, y_mean, y_std, device)

    def _pk(i):
        yi = y_true_vals[i]; L = yi[0::2][:N_val]; C = yi[1::2][:N_val]
        coup_M = k_m_t[i][:N_val-1] * np.minimum(L[:-1], L[1:])
        return dict(coupling_M=coup_M, C_sub=csf_t[i][:N_val]*C, alpha_C=aC_t[i][:N_val])

    # Best-of-K with ground-truth Q
    rt_best_list = []; rt_all_list = []
    for i in range(n_test):
        pk = _pk(i); fc = float(fc_arr[i])
        qL = q_L_true[i]; qC = q_C_true[i]
        rts = []
        for k in range(K_INFERENCE):
            lc = y_log_K[i, k]   # D_lc-dim LC prediction
            L = 10.0**lc[0::2]; C = 10.0**lc[1::2]
            rt = roundtrip_mse_lc(L, C, N_val, fc, qL, qC,
                                   s21_arr[i], s11_arr[i], FREQ_HZ, **pk)
            rts.append(rt if np.isfinite(rt) else RT_THRESHOLD * 10)
        rt_best_list.append(min(rts)); rt_all_list.extend(rts)

    # z=0 (best-of-1) evaluation
    rt_z0_list = []
    with torch.no_grad():
        inn.eval(); emb.eval()
        mbsz = 32
        for s in range(0, n_test, mbsz):
            xb = torch.from_numpy(x_norm[s:s+mbsz]).float().to(device)
            c  = emb(xb); bsz = xb.shape[0]
            y_z0_norm, _ = inn(torch.zeros(bsz, D_y, device=device), c=[c], rev=True)
            y_z0 = y_z0_norm.cpu().numpy() * y_std + y_mean
            for bi in range(bsz):
                ii = s + bi
                L  = 10.0**y_z0[bi, 0::2]; C = 10.0**y_z0[bi, 1::2]
                pk = _pk(ii)
                rt = roundtrip_mse_lc(L, C, N_val, float(fc_arr[ii]),
                                      q_L_true[ii], q_C_true[ii],
                                      s21_arr[ii], s11_arr[ii], FREQ_HZ, **pk)
                rt_z0_list.append(rt if np.isfinite(rt) else RT_THRESHOLD * 10)

    rt_best = np.array(rt_best_list)
    rt_all  = np.array(rt_all_list)
    rt_z0   = np.array(rt_z0_list)
    acc_best_of_K = float(np.mean(rt_best < RT_THRESHOLD))
    acc_z0 = float(np.mean(rt_z0 < RT_THRESHOLD))

    y_log_lc_mean = y_log_K.mean(axis=1)   # (n_test, D_lc)
    comp_mse_val  = component_mse(y_log_lc_mean, y_log_true_lc)
    r2_mean = float(np.nanmean(r2_per_component(y_log_lc_mean, y_log_true_lc)))

    return {
        'N': N_val, 'n_test': n_test,
        'rt_best_of_K':  float(np.mean(rt_best)),
        'rt_mean':       float(np.mean(rt_all)),
        'rt_median':     float(np.median(rt_all)),
        'rt_z0':         float(np.mean(rt_z0)),
        'acc_best_of_K': acc_best_of_K,
        'acc_z0':        acc_z0,
        'comp_mse':      comp_mse_val,
        'r2_mean':       r2_mean,
        'rt_best_arr':   rt_best,
        'rt_all':        rt_all,
        'rt_z0_arr':     rt_z0,
        'y_log_K':       y_log_K,       # (n_test, K_INFERENCE, D_lc)
        'y_log_true_lc': y_log_true_lc,
        'fc_arr':        fc_arr,
        's21_arr':       s21_arr,
        's11_arr':       s11_arr,
        'y_true_vals':   y_true_vals,
        'y_log_div':     y_log_div,     # (n_test, K_DIVERSITY, D_lc)
        'q_L_true':      q_L_true,
        'q_C_true':      q_C_true,
        'k_m':           k_m_t,
        'C_sub_frac':    csf_t,
        'alpha_C':       aC_t,
    }


# ── Figure 1: Benchmark bar chart ─────────────────────────────────────────────

def fig_benchmark_bar(mlp_res, inn_res_by_N, fig_path):
    N_vals = [3, 5]
    x = np.arange(len(N_vals)); w = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: rt_mse
    ax = axes[0]
    mlp_rt = [mlp_res[N]['rt_mse'] for N in N_vals]
    inn_z0 = [inn_res_by_N[N]['rt_z0'] for N in N_vals]
    inn_b50= [inn_res_by_N[N]['rt_best_of_K'] for N in N_vals]

    b1 = ax.bar(x - w,   mlp_rt, w*0.9, label='MLP (best-of-1)', color='#e74c3c', alpha=0.85)
    b2 = ax.bar(x,       inn_z0, w*0.9, label='cINN best-of-1 (z=0)', color='#3498db', alpha=0.85)
    b3 = ax.bar(x + w,   inn_b50, w*0.9, label=f'cINN best-of-{K_INFERENCE}', color='#2ecc71', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([f'N={n}' for n in N_vals], fontsize=12)
    ax.set_ylabel('Round-trip MSE (dB²)', fontsize=11)
    ax.set_title('Round-trip MSE: MLP vs cINN', fontsize=12)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')
    for bars in [b1, b2, b3]:
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2, h*1.08, f'{h:.1f}',
                    ha='center', va='bottom', fontsize=8)

    # Right: acc_frac
    ax = axes[1]
    mlp_acc = [mlp_res[N]['acc_frac'] for N in N_vals]
    inn_acc_z0 = [inn_res_by_N[N]['acc_z0'] for N in N_vals]
    inn_acc_b50= [inn_res_by_N[N]['acc_best_of_K'] for N in N_vals]

    ax.bar(x - w,   mlp_acc, w*0.9, label='MLP (best-of-1)', color='#e74c3c', alpha=0.85)
    ax.bar(x,       inn_acc_z0, w*0.9, label='cINN best-of-1 (z=0)', color='#3498db', alpha=0.85)
    ax.bar(x + w,   inn_acc_b50, w*0.9, label=f'cINN best-of-{K_INFERENCE}', color='#2ecc71', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([f'N={n}' for n in N_vals], fontsize=12)
    ax.set_ylabel(f'Acc. fraction (rt_mse < {RT_THRESHOLD} dB²)', fontsize=11)
    ax.set_title('Fraction of valid designs produced', fontsize=12)
    ax.set_ylim(0, 1.12); ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    for bars in [ax.containers[0], ax.containers[1], ax.containers[2]]:
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2, min(h+0.02, 1.05), f'{h:.2f}',
                    ha='center', va='bottom', fontsize=8)

    fig.suptitle('MLP vs cINN — OTFL301v2 Dataset (24–40 GHz, N∈{3,5})', fontsize=13)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fig_path}')


# ── Figure 2: Posterior samples overlay ───────────────────────────────────────

def fig_posterior_samples(ds, inn_res_by_N, mlp_pred_full, meta_full, fig_path):
    rng = np.random.RandomState(SEED)
    n_samples = 3

    fig, axes = plt.subplots(2, n_samples*2, figsize=(18, 8))

    col = 0
    for N_val in [3, 5]:
        res = inn_res_by_N[N_val]
        D_lc = 2 * N_val
        n_t = res['n_test']
        chosen = rng.choice(n_t, size=n_samples, replace=False)

        for ci in chosen:
            row = 0 if N_val == 3 else 1
            ax = axes[row, col % (n_samples*2)]
            fc = float(res['fc_arr'][ci])
            s21_tgt = res['s21_arr'][ci]
            ax.plot(FREQ_HZ/1e9, s21_tgt, 'k-', lw=2.5, label='Target', zorder=5)

            # Ground-truth Q for synthesis (cINN predicts LC only)
            qL_gt = res['q_L_true'][ci]
            qC_gt = res['q_C_true'][ci]

            # cINN K=20 posterior samples
            for k in range(min(20, K_DIVERSITY)):
                lc = res['y_log_div'][ci, k]   # D_lc-dim
                L = 10.0**lc[0::2]; C = 10.0**lc[1::2]
                s21, _ = synthesize_from_lc(L, C, N_val, fc, qL_gt, qC_gt, FREQ_HZ)
                if s21 is not None:
                    ax.plot(FREQ_HZ/1e9, s21, alpha=0.25, lw=0.8, color='#2980b9')

            # Best-of-10 prediction (using ground-truth Q)
            best_k = np.argmin([
                roundtrip_mse_lc(
                    10.0**res['y_log_K'][ci,k,0::2],
                    10.0**res['y_log_K'][ci,k,1::2],
                    N_val, fc,
                    qL_gt, qC_gt,
                    res['s21_arr'][ci], res['s11_arr'][ci], FREQ_HZ
                ) for k in range(min(10, K_INFERENCE))
            ])
            lc_best = res['y_log_K'][ci, best_k]
            L = 10.0**lc_best[0::2]; C = 10.0**lc_best[1::2]
            s21_best, _ = synthesize_from_lc(L, C, N_val, fc, qL_gt, qC_gt, FREQ_HZ)
            if s21_best is not None:
                ax.plot(FREQ_HZ/1e9, s21_best, 'g--', lw=1.5, label='cINN best', zorder=4)

            ax.set_xlabel('Freq (GHz)', fontsize=9)
            ax.set_ylabel('S21 (dB)', fontsize=9)
            ax.set_ylim(-65, 5); ax.set_title(f'N={N_val}, fc={fc:.1f} GHz', fontsize=9)
            ax.grid(alpha=0.25)
            if col == 0:
                handles = [
                    plt.Line2D([0],[0], color='k', lw=2, label='Target'),
                    plt.Line2D([0],[0], color='#2980b9', alpha=0.6, lw=1, label='cINN samples (K=20)'),
                    plt.Line2D([0],[0], color='g', ls='--', lw=1.5, label='cINN best'),
                ]
                ax.legend(handles=handles, fontsize=7)
            col += 1

    fig.suptitle('cINN Posterior Samples — target S21 + K=20 candidate designs', fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fig_path}')


# ── Figure 3: Diversity scatter ────────────────────────────────────────────────

def fig_diversity_scatter(inn_res_by_N, fig_path):
    rng = np.random.RandomState(SEED)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, N_val in zip(axes, [3, 5]):
        res = inn_res_by_N[N_val]
        D_lc = 2 * N_val
        ci = int(rng.choice(res['n_test'], 1)[0])
        # L1, C1 for all K_DIVERSITY samples
        lc = res['y_log_div'][ci, :, :D_lc]  # (K, D_lc)
        L1 = 10.0**lc[:, 0] * 1e12   # pH
        C1 = 10.0**lc[:, 1] * 1e15   # fF

        sc = ax.scatter(L1, C1, c=range(len(L1)), cmap='viridis', s=40, alpha=0.85, zorder=3)
        plt.colorbar(sc, ax=ax, label='Sample index')
        ax.set_xlabel('L₁ (pH)', fontsize=11); ax.set_ylabel('C₁ (fF)', fontsize=11)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_title(f'N={N_val}, fc={float(res["fc_arr"][ci]):.1f} GHz — '
                     f'K={K_DIVERSITY} posterior LC samples', fontsize=10)
        ax.grid(True, alpha=0.25, which='both')

    fig.suptitle('cINN Posterior Diversity — multiple valid (L₁, C₁) configurations per spec', fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fig_path}')


# ── Figure 4: Dataset samples ─────────────────────────────────────────────────

def fig_dataset_samples(ds, fig_path):
    rng = np.random.RandomState(SEED)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for row, N_val in enumerate([3, 5]):
        idx = np.where(ds['N'] == N_val)[0]
        chosen = rng.choice(len(idx), 3, replace=False)
        for col, ci in enumerate(chosen):
            i = idx[ci]
            ax = axes[row, col]
            s21 = ds['X_full'][i, 5:106]
            s11 = ds['X_full'][i, 106:207]
            ax.plot(FREQ_HZ/1e9, s21, label='S21', color='#2980b9', lw=1.5)
            ax.plot(FREQ_HZ/1e9, s11, label='S11', color='#e74c3c', lw=1, ls='--', alpha=0.8)
            fc = float(ds['fc_GHz'][i]); fbw = float(ds['fbw'][i])
            ripple = float(ds['ripple_dB'][i])
            ax.axvline(fc, color='gray', ls=':', alpha=0.5)
            ax.set_xlabel('Freq (GHz)', fontsize=9); ax.set_ylabel('S (dB)', fontsize=9)
            ax.set_ylim(-65, 5); ax.set_title(
                f'N={N_val}  fc={fc:.1f} GHz  FBW={fbw:.2f}  r={ripple:.2f} dB',
                fontsize=8)
            ax.grid(alpha=0.25)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    fig.suptitle('OTFL301v2 Dataset Samples — realistic physics (mutual coupling, substrate parasitics)',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fig_path}')


# ── Figure 5: RT MSE CDF ───────────────────────────────────────────────────────

def fig_rt_cdf(mlp_res, inn_res_by_N, fig_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bins = np.logspace(-2, 3, 200)

    for ax, N_val in zip(axes, [3, 5]):
        res = inn_res_by_N[N_val]
        mlp_rt_mean = mlp_res[N_val]['rt_mse']

        inn_rt_best = res['rt_best_arr']
        inn_rt_z0   = res['rt_z0_arr']

        # CDF
        for data, label, color, ls in [
            (inn_rt_best, f'cINN best-of-{K_INFERENCE}', '#2ecc71', '-'),
            (inn_rt_z0,   'cINN best-of-1 (z=0)',        '#3498db', '--'),
        ]:
            sorted_data = np.sort(data)
            cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
            ax.plot(sorted_data, cdf, color=color, ls=ls, lw=2, label=label)

        ax.axvline(mlp_rt_mean, color='#e74c3c', ls='-.', lw=2, alpha=0.8,
                   label=f'MLP mean ({mlp_rt_mean:.0f} dB²)')
        ax.axvline(RT_THRESHOLD, color='k', ls=':', lw=1.5, alpha=0.6,
                   label=f'Threshold ({RT_THRESHOLD} dB²)')
        ax.set_xscale('log')
        ax.set_xlabel('Round-trip MSE (dB²)', fontsize=11)
        ax.set_ylabel('CDF', fontsize=11)
        ax.set_title(f'N={N_val}', fontsize=12)
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
        ax.set_xlim(0.01, 200)

    fig.suptitle('Round-trip MSE Distribution — cINN on OTFL301v2 Test Set', fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fig_path}')


# ── Benchmark table ────────────────────────────────────────────────────────────

def write_benchmark_table(mlp_res, inn_res_by_N, table_path):
    lines = [
        "RF Inverse Design — Final Benchmark (OTFL301v2, 24–40 GHz, N∈{3,5})",
        "=" * 90,
        f"{'Model':<35} | {'N':>2} | {'rt_mse (dB²)':>12} | {'acc_frac':>9} | {'comp_mse':>10} | {'R²':>6}",
        "-" * 90,
    ]

    for N in [3, 5]:
        r = mlp_res[N]
        lines.append(
            f"{'MLP baseline (best-of-1)':<35} | {N:>2} | {r['rt_mse']:>12.2f} | "
            f"{r['acc_frac']:>9.4f} | {r['comp_mse']:>10.6f} | {r['r2_mean']:>6.4f}"
        )

    lines.append("-" * 90)

    for N in [3, 5]:
        res = inn_res_by_N[N]
        lines.append(
            f"{'cINN best-of-1 (z=0)':<35} | {N:>2} | {res['rt_z0']:>12.2f} | "
            f"{res['acc_z0']:>9.4f} | {res['comp_mse']:>10.6f} | {res['r2_mean']:>6.4f}"
        )
        lines.append(
            f"{'cINN best-of-50':<35} | {N:>2} | {res['rt_best_of_K']:>12.2f} | "
            f"{res['acc_best_of_K']:>9.4f} | {'':>10} | {'':>6}"
        )
        lines.append(
            f"{'cINN median rt_mse':<35} | {N:>2} | {res['rt_median']:>12.2f} | "
            f"{'—':>9} | {'':>10} | {'':>6}"
        )

    lines.append("=" * 90)
    lines.append(f"  acc_frac = fraction of test samples with rt_mse < {RT_THRESHOLD:.0f} dB²")
    lines.append(f"  rt_mse uses ground-truth parasitics (honest eval — perfect predictor → rt_mse ≈ 0)")

    text = "\n".join(lines) + "\n"
    with open(table_path, 'w') as f:
        f.write(text)
    print(f'\n{text}')
    print(f'  Saved {table_path}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    data_path, results_dir, figures_dir, ckpt, fig = setup_paths()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load dataset and build split (deterministic, same seed as training)
    print('Loading dataset...')
    (ds, train_idx, val_idx, test_idx, x_scaler_mlp, xs_scaler, x_scaler_inn,
     ys3, ys4, ys5) = load_dataset_and_split(data_path)
    print(f'Test set: {len(test_idx)} samples')

    # MLP evaluation
    print('\n=== Evaluating MLP ===')
    mlp_ckpt_full   = ckpt('mlp_otfl301v2_best.pt')
    mlp_ckpt_scalar = ckpt('mlp_otfl301v2_scalar_best.pt')
    if not os.path.exists(mlp_ckpt_full):
        print(f'WARN: {mlp_ckpt_full} not found — run train_mlp.py first')
        return
    (yp_full, yt_full, meta_full, mlp_res_full,
     yp_scal, yt_scal, meta_scal, mlp_res_scal) = run_mlp(
        ds, test_idx, x_scaler_mlp, xs_scaler, ys3, ys4, ys5,
        mlp_ckpt_full, mlp_ckpt_scalar, device
    )
    for N in [3, 5]:
        r = mlp_res_full[N]
        print(f'  MLP N={N}: rt_mse={r["rt_mse"]:.2f} dB², acc_frac={r["acc_frac"]:.3f}')

    # cINN evaluation
    print('\n=== Evaluating cINN ===')
    inn_res_by_N = {}
    for N_val in [3, 5]:
        ckpt_path = ckpt(f'inn_v2_otfl301v2_N{N_val}_best.pt')
        if not os.path.exists(ckpt_path):
            print(f'WARN: {ckpt_path} not found — run train_inn_v2.py first')
            return
        print(f'  N={N_val}...')
        inn_res_by_N[N_val] = run_cinn(N_val, ds, test_idx, x_scaler_inn, ckpt_path, device)
        r = inn_res_by_N[N_val]
        print(f'    rt_best_of_{K_INFERENCE}={r["rt_best_of_K"]:.2f} dB², '
              f'rt_z0={r["rt_z0"]:.2f}, acc_best={r["acc_best_of_K"]:.3f}')

    # Generate figures
    print('\n=== Generating figures ===')
    fig_benchmark_bar(mlp_res_full, inn_res_by_N, fig('final_benchmark_bar.png'))
    fig_posterior_samples(ds, inn_res_by_N, yp_full, meta_full, fig('final_posterior_samples.png'))
    fig_diversity_scatter(inn_res_by_N, fig('final_diversity.png'))
    fig_dataset_samples(ds, fig('final_dataset_samples.png'))
    fig_rt_cdf(mlp_res_full, inn_res_by_N, fig('final_rt_cdf.png'))

    # Benchmark table
    write_benchmark_table(mlp_res_full, inn_res_by_N,
                          os.path.join(results_dir, 'final_benchmark_table.txt'))

    print('\nAll done.')


if __name__ == '__main__':
    main()
