"""
Phase 4c-V2: Improved cINN architecture for RF filter inverse design.

Changes from inn.py / Phase 4c+
---------------------------------
1. ConditionEmbedderV2 — deeper, wider MLP: [207 → 512 → 512 → 256 → 128]
   V1 used [207 → 128 → 64] (2 layers, 64-dim output, 35k params).
   V2 uses 3 hidden layers, 128-dim output, 400k params — 11× more capacity.

   Why not a CNN: x_scaler (StandardScaler) normalizes each of the 101 frequency
   points independently by that point's own mean/std across the training set.
   This destroys the local continuity that a CNN exploits — after normalization,
   neighboring frequency points share no statistical structure. A plain MLP reading
   all 207 features simultaneously is the right architecture for per-feature-
   normalized inputs.

2. Larger INN — N_BLOCKS=12 (was 8), SUBNET_DIM=256 (was 128), COND_DIM=128 (was 64).

Parameter counts (approximate):
    ConditionEmbedderV2:            ~400k
    make_cinn_v2(D_y=12, blocks=12): ~1.2M
    Total per N=3:                  ~1.6M   (vs 255k in V1)
"""

import torch
import torch.nn as nn
from FrEIA.framework import SequenceINN
from FrEIA.modules import AllInOneBlock


# ── Subnet constructor ────────────────────────────────────────────────────────

def _make_subnet_v2(c_in: int, c_out: int, subnet_dim: int) -> nn.Module:
    """
    2-hidden-layer MLP inside each coupling block (same depth as V1, wider).
    Final Linear is zero-initialized → identity warm-start.
    """
    return nn.Sequential(
        nn.Linear(c_in, subnet_dim),
        nn.ReLU(),
        nn.Linear(subnet_dim, subnet_dim),
        nn.ReLU(),
        nn.Linear(subnet_dim, c_out),
    )


# ── Deeper MLP condition embedder ─────────────────────────────────────────────

class ConditionEmbedderV2(nn.Module):
    """
    MLP embedder. Default input_dim=231 for parasitic-conditioned cINN:
        [5 scalar specs + 101 S21 + 101 S11 + 24 parasitic features]
    Parasitic features: Q_L(5) + Q_C(5) + k_m(4) + C_sub_frac(5) + alpha_C(5) = 24.

    Smaller architecture for Bundle 1+4+5: [input → 256 → 128 → cond_dim].
    Reduced from V1 [input → 512 → 512 → 256 → cond_dim] (533k → ~80k params)
    to combat overfitting at 20k training samples per N.
    """

    def __init__(self, input_dim: int = 231, cond_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, cond_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Larger cINN factory ───────────────────────────────────────────────────────

def make_cinn_v2(
    D_y: int,
    D_c: int = 128,
    n_blocks: int = 12,
    subnet_dim: int = 256,
    affine_clamping: float = 2.0,
) -> SequenceINN:
    """
    Larger conditional INN: n_blocks=12, subnet_dim=256, D_c=128.
    ~5× more INN parameters than V1 (8 blocks, 128 dim, 64 cond).
    """
    inn = SequenceINN(D_y)

    def _subnet(c_in: int, c_out: int) -> nn.Module:
        return _make_subnet_v2(c_in, c_out, subnet_dim)

    for _ in range(n_blocks):
        inn.append(
            AllInOneBlock,
            cond=0,
            cond_shape=(D_c,),
            subnet_constructor=_subnet,
            affine_clamping=affine_clamping,
            permute_soft=True,
        )

    # Zero-init final Linear of every coupling subnet → identity at init.
    for module in inn.modules():
        if isinstance(module, nn.Sequential):
            children = list(module.children())
            if children and isinstance(children[-1], nn.Linear):
                nn.init.zeros_(children[-1].weight)
                nn.init.zeros_(children[-1].bias)

    return inn


# ── Re-export utilities from inn.py ──────────────────────────────────────────
from models.inn import fix_mps_contiguity, verify_bijection  # noqa: F401
