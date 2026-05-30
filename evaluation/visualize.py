"""
Visualization utilities for the RF inverse design pipeline.

All functions save to results/figures/ and return the save path.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FREQ_GHZ = np.linspace(40, 90, 101)
G_LABELS  = ['g1', 'g2', 'g3', 'g4', 'g5', 'g6']
LC_LABELS = ['L1', 'C1', 'L2', 'C2', 'L3', 'C3', 'L4', 'C4', 'L5', 'C5']


def plot_loss_curves(
    train_losses: list,
    val_losses: list,
    save_path: str,
    title: str = "Training Loss",
) -> None:
    """
    Save train/val loss curves.

    Args:
        train_losses: list of per-epoch training loss values
        val_losses: list of per-epoch validation loss values
        save_path: output PNG path
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Train', linewidth=1.5)
    ax.plot(epochs, val_losses, label='Val', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss (normalized g-values)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_scatter(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    N_values: np.ndarray,
    save_path: str,
) -> None:
    """
    Predicted vs actual scatter plots per LC position (log10-space values).

    Layout: 2 rows × 5 cols.  Row 0 = inductors (L1..L5), row 1 = capacitors (C1..C5).
    Columns that are NaN for a given N are skipped. Log-scale axes since L/C span
    orders of magnitude.

    Args:
        y_pred:   predicted log10(LC), shape (n_samples, 10); NaN where undefined
        y_true:   ground-truth log10(LC), shape (n_samples, 10); NaN where undefined
        N_values: filter order per sample, shape (n_samples,)
        save_path: output PNG path
    """
    colors = {3: '#1f77b4', 4: '#2ca02c', 5: '#ff7f0e'}
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))

    for col in range(5):               # L1..L5 in row 0, C1..C5 in row 1
        for row in range(2):
            k = col * 2 + row          # column index in y: L1=0,C1=1,L2=2,...
            ax = axes[row][col]
            label = LC_LABELS[k]

            for N in [3, 4, 5]:
                mask = (N_values == N) & ~np.isnan(y_true[:, k])
                if mask.sum() == 0:
                    continue
                ax.scatter(
                    y_true[mask, k], y_pred[mask, k],
                    c=colors[N], alpha=0.3, s=5, label=f'N={N}',
                )

            all_valid = ~np.isnan(y_true[:, k])
            if all_valid.sum() > 0:
                lo = min(y_true[all_valid, k].min(), y_pred[all_valid, k].min())
                hi = max(y_true[all_valid, k].max(), y_pred[all_valid, k].max())
                margin = (hi - lo) * 0.05
                ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                        'k--', linewidth=1.0, alpha=0.6)
                ax.set_xlabel(f'True log₁₀({label})')
                ax.set_ylabel(f'Pred log₁₀({label})')
            else:
                ax.set_visible(False)
                continue

            ax.set_title(label)
            ax.legend(fontsize=6, markerscale=2)
            ax.grid(True, alpha=0.3)

    fig.suptitle('Predicted vs Actual LC values (log₁₀ space)', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roundtrip(
    samples: list,
    save_path: str,
    model_label: str = "MLP",
) -> None:
    """
    6-panel overlay: target S21 (solid) vs predicted S21 (dashed).

    samples is a list of dicts with keys:
        s21_target, s21_pred, N, ripple_dB, fc_GHz, fbw, rt_mse

    Args:
        samples: list of 6 sample dicts (3 for N=3, 3 for N=5)
        save_path: output PNG path
        model_label: model name for the title
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for idx, s in enumerate(samples[:6]):
        ax = axes[idx // 3][idx % 3]
        ax.plot(FREQ_GHZ, s['s21_target'], 'b-', linewidth=1.5, label='Target')
        ax.plot(FREQ_GHZ, s['s21_pred'], 'r--', linewidth=1.5, label='Predicted')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('S21 (dB)')
        ax.set_title(
            f"N={s['N']}, r={s['ripple_dB']}dB, fc={s['fc_GHz']:.0f}GHz\n"
            f"rt_mse={s['rt_mse']:.4f}"
        )
        ax.set_xlim(40, 90)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{model_label} — Round-Trip S21 Overlay', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_r2_bars(
    r2_by_model: dict,
    save_path: str,
) -> None:
    """
    Bar chart of R² per LC position, grouped by model, one panel per N.

    Args:
        r2_by_model: dict mapping label → {N: r2_array}
            r2_array has shape (10,); NaN for positions not valid for that N
        save_path: output PNG path
    """
    n_cols_per_N = {3: 6, 4: 8, 5: 10}   # number of valid LC positions per N
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax_idx, N in enumerate([3, 4, 5]):
        ax = axes[ax_idx]
        n_g = n_cols_per_N[N]
        x = np.arange(n_g)
        width = 0.8 / max(len(r2_by_model), 1)

        for i, (label, r2_dict) in enumerate(r2_by_model.items()):
            r2 = r2_dict.get(N, np.full(10, np.nan))
            vals = [float(r2[k]) if k < len(r2) and not np.isnan(r2[k]) else 0.0
                    for k in range(n_g)]
            offset = (i - len(r2_by_model) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width=width * 0.9, label=label, alpha=0.85)

        ax.set_xlabel('LC position')
        ax.set_ylabel('R²')
        ax.set_title(f'R² per LC position — N={N}')
        ax.set_xticks(x)
        ax.set_xticklabels(LC_LABELS[:n_g], fontsize=7)
        ax.set_ylim(-0.1, 1.05)
        ax.axhline(1.0, color='k', linestyle='--', linewidth=0.8, alpha=0.4)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
