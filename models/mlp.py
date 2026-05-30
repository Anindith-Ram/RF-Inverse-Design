"""
MLP inverse models for RF filter design (realistic LC dataset).

InverseMLP:    207-dim input (S-param curves + 5 scalar specs) → LC values
SpecsOnlyMLP:  5-dim input (scalar specs only) → LC values (ablation baseline)

Output heads are N-specific (N=3: 6 LC values, N=4: 8, N=5: 10).
Targets are log10(L/C) values — caller handles the log/exp transform.

Architecture: shared trunk [input→512→256→128] with BatchNorm+ReLU+Dropout(0.2),
then three N-specific linear heads.
"""

import torch
import torch.nn as nn


def _build_trunk(input_dim: int, dropout: float = 0.2) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, 512),
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


class InverseMLP(nn.Module):
    """
    MLP inverse model: S-param features → log10(LC) values.

    Architecture: shared trunk [input→512→256→128] with BatchNorm+ReLU+Dropout(0.2),
    then three N-specific linear heads.

    Output head sizes match 2×N LC pairs:
        head_3: 6 outputs  (L1,C1,L2,C2,L3,C3)
        head_4: 8 outputs  (L1,C1,...,L4,C4)
        head_5: 10 outputs (L1,C1,...,L5,C5)

    Args:
        input_dim: input feature dimension (default 207)
        dropout: dropout probability (default 0.2)
    """

    def __init__(self, input_dim: int = 207, dropout: float = 0.2):
        super().__init__()
        self.trunk = _build_trunk(input_dim, dropout)
        self.head_3 = nn.Linear(128, 6)
        self.head_4 = nn.Linear(128, 8)
        self.head_5 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through trunk and all three heads.

        Args:
            x: input tensor, shape (batch, input_dim)

        Returns:
            out_3: head-3 output, shape (batch, 6)
            out_4: head-4 output, shape (batch, 8)
            out_5: head-5 output, shape (batch, 10)
        """
        feat = self.trunk(x)
        return self.head_3(feat), self.head_4(feat), self.head_5(feat)


class SpecsOnlyMLP(nn.Module):
    """
    Ablation baseline: 5-dim scalar specs → log10(LC) values.

    Input: [fc_GHz, fbw, ripple_dB, N3_flag, N5_flag]
    Same trunk architecture as InverseMLP but without S-parameter curves.

    Args:
        dropout: dropout probability (default 0.2)
    """

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.trunk = _build_trunk(input_dim=5, dropout=dropout)
        self.head_3 = nn.Linear(128, 6)
        self.head_4 = nn.Linear(128, 8)
        self.head_5 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: scalar specs tensor, shape (batch, 5)

        Returns:
            out_3: head-3 output, shape (batch, 6)
            out_4: head-4 output, shape (batch, 8)
            out_5: head-5 output, shape (batch, 10)
        """
        feat = self.trunk(x)
        return self.head_3(feat), self.head_4(feat), self.head_5(feat)
