"""
Tandem inverse model for RF filter design (Phase 4b).

Architecture
------------
The tandem network addresses the non-uniqueness problem by adding a forward-
consistency term to the loss. Instead of minimizing only ||predicted_LC - true_LC||²
(which forces the model to pick one mode of a multi-modal distribution), the tandem
loss also minimizes ||F(I(x)) - x||², where:
    I = inverse model (the model being trained)
    F = frozen forward surrogate (ForwardMLP from Phase 4a)
    x = target S-param curve

This enforces physical consistency: the predicted LC values, when passed through the
forward model, must reproduce the target S-params. This constraint is compatible with
ALL valid solutions to the inverse problem, not just the ground-truth mode — so the
model is free to find any physically valid LC combination that produces the target
S-params, rather than being forced to match the specific ground-truth values.

Loss during training
--------------------
    L = α · MSE(I(x), y_LC) + β · MSE(F(I(x)), x_Spaaram)

    Phase 1 (warm-start, epochs 1–warm_epochs):   α=1.0, β=0.0
        Train the inverse model with supervised LC loss only.
        This gives a reasonable initialization before adding forward loss.

    Phase 2 (tandem, epochs warm_epochs+1 onward): α=alpha, β=beta (beta >> alpha)
        The forward-consistency term dominates, pushing predicted LC values toward
        physically valid regions. α > 0 is kept as a weak anchor to prevent the
        model from drifting to degenerate solutions that satisfy F(I(x))=x trivially.

Input/output
------------
TandemInverseMLP has the same interface as InverseMLP (models/mlp.py):
    Input:  (batch, input_dim) — default 207: [fc, fbw, ripple, N3_flag, N5_flag, S21×101, S11×101]
    Output: tuple(head_3, head_4, head_5) — same 3-head design

The forward model F is loaded from checkpoint results/forward_model_best.pt and
frozen (requires_grad=False). Gradient flows:
    x_input → I(x) → y_log_pred → build_forward_input_torch() → F.predict() → S_pred → L_fwd

build_forward_input_torch() is gradient-safe: it inserts y_log_pred into columns [0:10]
of the 24-dim forward input while treating Q and spec columns as constants.

Checkpoint format (results/tandem_best.pt):
    model_state_dict  — TandemInverseMLP state dict
    x_scaler_*        — 3-scaler dicts (for y_3, y_4, y_5) mirroring mlp checkpoint
    (same format as mlp_realistic_best.pt, so evaluate_model from train_mlp.py reusable)
"""

import torch
import torch.nn as nn


def _build_trunk(input_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
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
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


class TandemInverseMLP(nn.Module):
    """
    Inverse model with the same 3-head architecture as InverseMLP (models/mlp.py).

    This is a standalone module — the tandem forward-consistency loss is computed
    in train_tandem.py using the frozen ForwardMLP, not inside this class.

    Args:
        input_dim: input dimension (default 207 = full S-param + specs)
        dropout:   dropout probability (default 0.2)
    """

    def __init__(self, input_dim: int = 207, dropout: float = 0.2):
        super().__init__()
        self.trunk  = _build_trunk(input_dim, dropout)
        self.head_3 = nn.Linear(128, 6)    # N=3: log10(L1,C1,L2,C2,L3,C3)
        self.head_4 = nn.Linear(128, 8)    # N=4
        self.head_5 = nn.Linear(128, 10)   # N=5

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, input_dim) normalized input

        Returns:
            (h3, h4, h5): normalized log10(LC) predictions for each N-head
        """
        feat = self.trunk(x)
        return self.head_3(feat), self.head_4(feat), self.head_5(feat)
