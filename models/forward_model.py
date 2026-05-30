"""
Forward surrogate model for RF filter design (Phase 4a).

Maps physical LC + Q parameters → S-parameter curves.
Used as a frozen differentiable component in the tandem network (Phase 4b)
and as the objective function for CMA-ES optimization (Phase 4d).

Input layout (24-dim):
    [0:10]  log10(L1,C1,...,L5,C5)  interleaved, NaN-padded positions → 0
    [10:15] Q_L1..Q_L5              NaN-padded positions → 0
    [15:20] Q_C1..Q_C5              NaN-padded positions → 0
    [20]    fc_GHz
    [21]    fbw
    [22]    N3_flag  (1 if N==3, else 0)
    [23]    N5_flag  (1 if N==5, else 0)

Output (202-dim): [S21_dB(101), S11_dB(101)] at 40–90 GHz, 101 points.

Checkpoint format (results/forward_model_best.pt):
    model_state_dict  — nn.Module state dict
    x_scaler_mean     — (24,) float64 numpy array
    x_scaler_std      — (24,) float64 numpy array
    y_scaler_mean     — (202,) float64 numpy array
    y_scaler_std      — (202,) float64 numpy array

This format lets train_tandem.py reconstruct normalization as a torch operation
so gradients flow through the forward model during tandem training.
"""

import numpy as np
import torch
import torch.nn as nn

FORWARD_INPUT_DIM  = 24
FORWARD_OUTPUT_DIM = 202   # S21(101) + S11(101)


class ForwardMLP(nn.Module):
    """
    Forward surrogate: (LC+Q+specs) → S-params.

    Architecture: [24 → 512 → 512 → 256 → 202] with BatchNorm + ReLU + Dropout(0.1).
    Lower dropout than InverseMLP — the forward mapping is deterministic, no mode
    averaging needed, so less regularization is required.

    forward() expects PRE-NORMALIZED inputs (normalized by the training StandardScaler).
    Use predict() for raw un-normalized inputs — it applies normalization internally
    using buffers stored in the checkpoint.

    Args:
        input_dim:  input dimension (default 24)
        output_dim: output dimension (default 202)
        dropout:    dropout probability (default 0.1)
    """

    def __init__(
        self,
        input_dim:  int   = FORWARD_INPUT_DIM,
        output_dim: int   = FORWARD_OUTPUT_DIM,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
        )

        # Normalization buffers — set via set_normalization() after training.
        # Stored in checkpoint so the model is self-contained for tandem/CMA-ES.
        self.register_buffer('x_mean', torch.zeros(input_dim))
        self.register_buffer('x_std',  torch.ones(input_dim))
        self.register_buffer('y_mean', torch.zeros(output_dim))
        self.register_buffer('y_std',  torch.ones(output_dim))

    def set_normalization(
        self,
        x_mean: np.ndarray, x_std:  np.ndarray,
        y_mean: np.ndarray, y_std:  np.ndarray,
    ) -> None:
        """Store scaler statistics as buffers for use in predict() and tandem training."""
        self.x_mean.copy_(torch.from_numpy(x_mean.astype(np.float32)))
        self.x_std.copy_(torch.from_numpy(x_std.astype(np.float32)))
        self.y_mean.copy_(torch.from_numpy(y_mean.astype(np.float32)))
        self.y_std.copy_(torch.from_numpy(y_std.astype(np.float32)))

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        Forward pass expecting pre-normalized input, returning normalized output.

        Args:
            x_norm: (batch, 24) normalized input

        Returns:
            y_norm: (batch, 202) normalized S-param predictions
        """
        return self.net(x_norm)

    def predict(self, x_raw: torch.Tensor) -> torch.Tensor:
        """
        Predict from raw (un-normalized) input. Returns S-params in dB.

        Applies x normalization internally, runs forward(), then denormalizes output.
        Gradient-safe: used in tandem training to backpropagate through the inverse
        model's predicted y_log → x_raw → forward model → S-params → tandem loss.

        Args:
            x_raw: (batch, 24) raw input tensor

        Returns:
            s_params_db: (batch, 202) S-params in dB
        """
        x_norm = (x_raw - self.x_mean) / self.x_std
        y_norm = self.net(x_norm)
        return y_norm * self.y_std + self.y_mean


# ── Input construction helpers ────────────────────────────────────────────────

def build_forward_input(
    y_log:   np.ndarray,
    Q_L:     np.ndarray,
    Q_C:     np.ndarray,
    fc_GHz:  np.ndarray,
    fbw:     np.ndarray,
    N:       np.ndarray,
) -> np.ndarray:
    """
    Build the 24-dim forward model input matrix from dataset arrays (numpy).

    NaN-padded positions (unused LC/Q slots for smaller N) are replaced with 0.
    N is one-hot encoded as [N3_flag, N5_flag]: N=4 → [0, 0].

    Args:
        y_log:  (n, 10) log10 LC values from dataset['y_log']
        Q_L:    (n, 5)  inductor Q from dataset['Q_L']
        Q_C:    (n, 5)  capacitor Q from dataset['Q_C']
        fc_GHz: (n,)
        fbw:    (n,)
        N:      (n,)

    Returns:
        X_fwd: (n, 24) float64
    """
    n = len(fc_GHz)
    X = np.zeros((n, FORWARD_INPUT_DIM), dtype=np.float64)
    X[:, 0:10]  = np.nan_to_num(y_log[:, :10], nan=0.0)
    X[:, 10:15] = np.nan_to_num(Q_L,            nan=0.0)
    X[:, 15:20] = np.nan_to_num(Q_C,            nan=0.0)
    X[:, 20]    = fc_GHz
    X[:, 21]    = fbw
    X[:, 22]    = (N == 3).astype(np.float64)
    X[:, 23]    = (N == 5).astype(np.float64)
    return X


def build_forward_input_torch(
    y_log_pred: torch.Tensor,
    Q_L:        torch.Tensor,
    Q_C:        torch.Tensor,
    fc_GHz:     torch.Tensor,
    fbw:        torch.Tensor,
    N:          torch.Tensor,
) -> torch.Tensor:
    """
    Build the 24-dim forward model input tensor from torch tensors.

    Gradient-safe: gradients flow through y_log_pred (the inverse model output).
    All other inputs are constants from the batch metadata.

    Used in train_tandem.py for the tandem forward-consistency loss.

    Args:
        y_log_pred: (batch, 10) predicted log10 LC — already NaN→0 from DataLoader
        Q_L:        (batch, 5)  ground-truth Q_L — NaN→0 applied internally
        Q_C:        (batch, 5)  ground-truth Q_C — NaN→0 applied internally
        fc_GHz:     (batch,)
        fbw:        (batch,)
        N:          (batch,) long

    Returns:
        x_fwd: (batch, 24)
    """
    q_l = torch.nan_to_num(Q_L, nan=0.0)
    q_c = torch.nan_to_num(Q_C, nan=0.0)
    n3  = (N == 3).float().unsqueeze(1)
    n5  = (N == 5).float().unsqueeze(1)

    return torch.cat([
        y_log_pred,
        q_l,
        q_c,
        fc_GHz.unsqueeze(1),
        fbw.unsqueeze(1),
        n3,
        n5,
    ], dim=1)
