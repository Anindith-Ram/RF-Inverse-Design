"""
Phase 4c: Conditional INN (cINN) for RF filter inverse design.

Architecture
------------
Three independent cINNs, one per filter order N ∈ {3, 4, 5}, with output
dimensions D_y ∈ {6, 8, 10} (log10 LC values). Keeping separate models per N
avoids padding artifacts in log|det J| and isolates failure modes.

Each cINN pairs with a shared ConditionEmbedder that compresses the 207-dim
input (S-params + specs) down to a 64-dim conditioning vector fed into every
coupling block.

Coupling blocks: FrEIA AllInOneBlock (affine coupling + ActNorm + soft
permutation). Soft-clamping on the log-scale exponent (affine_clamping=2.0)
prevents NLL explosions. Subnet last-layer zero-init makes each block start as
the identity — effectively a free supervised warm-start.

Loss
----
Pure NLL (maximum likelihood):
    loss = 0.5 * ||z||² - log|det J|
No auxiliary supervised LC term. Zero-init already makes z ≈ y_norm at step 0.

Inference
---------
Sample z ~ N(0, I), invert: y_pred = INN^{-1}(z | c=embedder(x)).
Best-of-K: sample K candidates, pick lowest rt_mse via synthesize_from_lc.

Checkpoint format (results/inn_N{N}_best.pt):
    inn_state_dict, embedder_state_dict,
    y_scaler_mean, y_scaler_std,
    x_scaler_mean, x_scaler_std,
    N, D_y, D_c, n_blocks, subnet_dim, affine_clamping, best_val_rt_mse
"""

import torch
import torch.nn as nn
from FrEIA.framework import SequenceINN
from FrEIA.modules import AllInOneBlock


# ── Subnet constructor ────────────────────────────────────────────────────────

def _make_subnet(c_in: int, c_out: int, subnet_dim: int) -> nn.Module:
    """
    Small MLP used inside each coupling block.
    Final Linear is zero-initialized by make_cinn so each block starts as identity.
    """
    return nn.Sequential(
        nn.Linear(c_in, subnet_dim),
        nn.ReLU(),
        nn.Linear(subnet_dim, subnet_dim),
        nn.ReLU(),
        nn.Linear(subnet_dim, c_out),
    )


# ── Condition embedder ────────────────────────────────────────────────────────

class ConditionEmbedder(nn.Module):
    """
    Compresses the 207-dim X_full input to a D_c-dim conditioning vector.
    Trained jointly with the cINN. Shared across coupling blocks.

    Input layout (X_full, 207-dim):
        [0]     fc_GHz
        [1]     fbw
        [2]     ripple_dB
        [3]     N3_flag
        [4]     N5_flag
        [5:106] S21_dB (101 pts)
        [106:207] S11_dB (101 pts)
    """

    def __init__(self, input_dim: int = 207, cond_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, cond_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── cINN factory ──────────────────────────────────────────────────────────────

def make_cinn(
    D_y: int,
    D_c: int = 64,
    n_blocks: int = 8,
    subnet_dim: int = 128,
    affine_clamping: float = 2.0,
) -> SequenceINN:
    """
    Build a conditional INN for D_y-dimensional LC outputs.

    Args:
        D_y:             output / input dimension (6, 8, or 10)
        D_c:             conditioning dimension (must match ConditionEmbedder output)
        n_blocks:        number of AllInOneBlock coupling layers
        subnet_dim:      hidden width of each coupling block's subnet MLP
        affine_clamping: soft-clamp on log-scale exponent (prevents NLL blow-up)

    Returns:
        FrEIA SequenceINN with gradient-enabled parameters.
    """
    inn = SequenceINN(D_y)

    def _subnet(c_in: int, c_out: int) -> nn.Module:
        return _make_subnet(c_in, c_out, subnet_dim)

    for _ in range(n_blocks):
        inn.append(
            AllInOneBlock,
            cond=0,
            cond_shape=(D_c,),
            subnet_constructor=_subnet,
            affine_clamping=affine_clamping,
            permute_soft=True,
        )

    # Zero-init the final Linear of every coupling subnet → identity at init.
    # Walk all Sequential submodules; zero the last child if it's a Linear.
    for module in inn.modules():
        if isinstance(module, nn.Sequential):
            children = list(module.children())
            if children and isinstance(children[-1], nn.Linear):
                nn.init.zeros_(children[-1].weight)
                nn.init.zeros_(children[-1].bias)

    return inn


# ── MPS contiguity fix ────────────────────────────────────────────────────────

def fix_mps_contiguity(model: nn.Module) -> None:
    """
    FrEIA AllInOneBlock stores w_perm_inv as a non-contiguous tensor (it is
    initialized as w.T which is a view). On MPS, non-contiguous parameters are
    read with wrong strides, breaking the reverse permutation. Call this after
    model.to('mps') to force all parameters to be contiguous in memory.
    """
    for param in model.parameters():
        if not param.data.is_contiguous():
            param.data = param.data.contiguous()


# ── Bijection sanity check ────────────────────────────────────────────────────

def verify_bijection(
    inn: SequenceINN,
    embedder: ConditionEmbedder,
    D_y: int,
    device: torch.device,
    tol: float = 1e-4,
) -> float:
    """
    Assert f^{-1}(f(y | c), c) ≈ y (round-trip reconstruction).

    Call this before training starts. If the assertion fires the INN is broken
    and training must be aborted — do not suppress the error.

    Args:
        inn:      cINN to check
        embedder: companion ConditionEmbedder
        D_y:      output dimension
        device:   compute device
        tol:      max abs reconstruction error tolerated

    Returns:
        max abs reconstruction error (float)
    """
    inn.eval()
    embedder.eval()
    with torch.no_grad():
        y = torch.randn(32, D_y, device=device)
        in_dim = next(embedder.parameters()).shape[1]
        x = torch.randn(32, in_dim, device=device)
        c = embedder(x)
        z, _ = inn(y, c=[c])
        y_rec, _ = inn(z, c=[c], rev=True)
    err = (y - y_rec).abs().max().item()
    assert err < tol, (
        f"Bijection broken for D_y={D_y}: max reconstruction error = {err:.2e} "
        f"(tolerance {tol:.2e}). Do not proceed with training."
    )
    inn.train()
    embedder.train()
    return err
