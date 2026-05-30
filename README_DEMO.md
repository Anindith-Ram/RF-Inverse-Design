# RF Filter Inverse Design — cINN Demo

A conditional Invertible Neural Network (cINN) pipeline for RF bandpass filter inverse
design: given desired S-parameter targets, produce multiple valid LC filter configurations.

---

## Problem Statement

Designing on-chip RF bandpass filters for mm-wave SoC applications (24–40 GHz SOI
CMOS) requires mapping desired transmission/reflection specs to physical LC resonator
values. Classical Chebyshev synthesis assumes ideal components; real on-chip components
have lossy inductors (Q_L ≈ 15–30), lossy capacitors (Q_C ≈ 100–300), mutual inductive
coupling between resonators, substrate parasitic capacitances, and process spread. These
effects make the synthesis problem non-invertible: multiple LC configurations produce
nearly identical S-parameter responses.

A naive MLP minimizes E[||ŷ − y||²]. For multi-modal posteriors p(y|X), the minimizer
is the **conditional mean** — a weighted average of valid modes that satisfies none of
them. A cINN models the full posterior p(LC | S-params, specs) and can produce K diverse
candidate designs per specification.

---

## Dataset

**OTFL301v2** — 50,000 samples, N ∈ {3, 5} resonators, 24–40 GHz, generated with 4
physics improvements over ideal Chebyshev synthesis:

| Physics effect | Implementation |
|---|---|
| Mutual inductance | k_m ∈ Uniform[0.02, 0.12]; M = k_m · min(L_i, L_{i+1}) |
| Substrate parasitic caps | C_sub = c_frac · C; c_frac ∈ Uniform[0.03, 0.15] |
| Frequency-dependent Q_C | Q_C(f) = Q_C0 · (f0/f)^α; α ∈ Uniform[0.1, 0.5] |
| Process spread | σ_L = 4%, σ_C = 3% log-normal per element |

Deliberately simplified (and documented):
- Tuning states not included (discrete conditioning — documented extension)
- Nearest-neighbor coupling only (no cross-coupling)
- ABCD cascade approximates EM response (proxy for EMX data)

Input features: `X_full` (207-dim): `[fc_GHz, fbw, ripple_dB, N3_flag, N5_flag, S21×101, S11×101]`

Target: `y_log` = log₁₀[L₁, C₁, …, L_N, C_N] in SI units (Henries/Farads)

---

## Results

Evaluation uses **honest synthesis**: round-trip MSE is computed by synthesizing S-params
from predicted LC using the **ground-truth parasitics** (k_m, C_sub_frac, α_C) stored per
sample in the dataset. A perfect LC predictor achieves rt_mse ≈ 0 dB².

```
Model                          | N=3 rt_mse  | N=5 rt_mse  | acc_frac (<5 dB²)
                               |             |             |   N=3    /   N=5
-------------------------------|-------------|-------------|------------------
MLP baseline (best-of-1)       |  100.7 dB²  |  128.3 dB²  |  0.16  /  0.06
cINN best-of-1 (z=0)           |   46.1 dB²  |   98.9 dB²  |  0.55  /  0.18
cINN best-of-50                |    2.7 dB²  |    8.0 dB²  |  0.97  /  0.89
```

**Why the MLP fails**: rt_mse = 100–128 dB² despite R² ≈ 0.995 on log₁₀(LC). Mode
averaging produces LC values that lie between valid resonator configurations — the
synthesized filter matches none of the target modes. Acc_frac of 6–16% means < 1 in 6
MLP designs is physically valid at the 5 dB² threshold.

**Why the cINN succeeds**: by modeling p(LC | spec, S-params, parasitics), sampling
K=50 candidates and selecting the best gives the designer multiple valid configurations.
With ground-truth parasitics as conditioning, the posterior sharpens to near-deterministic;
**97% of N=3 specs and 89% of N=5 specs yield at least one < 5 dB² candidate**.

A single z=0 mode estimate (best-of-1) already outperforms MLP rt_mse by 2–3× — the
diversity of K=50 sampling drives the 16–37× rt_mse improvement headline.

---

## Architecture

Three models share the same dataset:

### MLP Baseline (`models/mlp.py`)
`InverseMLP`: 207-dim → [512 → 256 → 128] → 2 N-specific heads (N=3: 6-dim, N=5: 10-dim)

### cINN (`models/inn_v2.py` + FrEIA)
- `ConditionEmbedderV2`: [231 → 256 → 128 → 128] — 109k params
- `make_cinn_v2`: 8× `AllInOneBlock` (FrEIA), subnet_dim=128, affine_clamping=1.5 — 274k params
- **D_y = 2·N** (LC only): Q is sampled independently of S-params; including Q as cINN target caused z_std drift to 2.5+. Q is **conditioned on** instead (added to embedder input).
- **Condition input = 231 dims**: 5 scalar specs + 101 S21 + 101 S11 + 24 parasitics
  (Q_L×5, Q_C×5, k_m×4, C_sub_frac×5, α_C×5). NaN-padded slots zero-filled for N=3.
- **z_std regularizer**: `loss = NLL + 1.0·(mean(std(z)) − 1)²` — prevents the degenerate
  affine-scaling mode that inflates `log|det J|` at the cost of val NLL.
- Total: ~383k parameters per N (4.6× smaller than initial design — overfitting fix)

Training: pure NLL + z-reg, AdamW(lr=1e-3, wd=1e-4), 400 epochs, ReduceLROnPlateau, MPS.

---

## Figures

| Figure | What it shows |
|---|---|
| `final_benchmark_bar.png` | rt_mse and acc_frac: MLP vs cINN best-of-1 vs best-of-50 |
| `final_posterior_samples.png` | K=20 cINN S21 overlays + ground truth, 3 samples each for N=3 and N=5 |
| `final_diversity.png` | Scatter of K=50 (L₁, C₁) samples per spec — posterior diversity |
| `final_dataset_samples.png` | 6 representative dataset S-params showing realistic asymmetric passbands |
| `final_rt_cdf.png` | CDF of rt_mse: MLP vs cINN, shows heavy tail of MLP |

---

## How to Run

```bash
# Setup
source rf_env/bin/activate

# Regenerate dataset (optional — already exists)
python data/generate_otfl301v2.py

# Retrain MLP baseline (~15 min)
python -u training/train_mlp.py 2>&1 | tee /tmp/mlp_train.log

# Retrain cINN N=3 + N=5 (~35 min, runs sequentially)
python -u training/train_inn_v2.py 2>&1 | tee /tmp/inn_train.log

# Generate all figures + benchmark table
python evaluation/make_final_figures.py

# View results
cat results/final_benchmark_table.txt
open results/figures/final_benchmark_bar.png
```

---

## Transferability to Otava / EMX Data

This pipeline uses ABCD-synthesized LC values as a physics-faithful proxy for EMX
EM simulation data. Swapping in EMX data requires:

1. New data loader (CSV/HDF5 from EMX sweep runs)
2. `y` = physical geometry parameters (resonator length, gap width) instead of LC
3. Round-trip eval uses a trained `ForwardMLP` surrogate instead of ABCD formula
4. Dataset will be smaller (thousands vs 50k) — more regularization, fewer epochs
5. Non-uniqueness will be more severe in real EM data — cINN is the most relevant model

Scale: 10,000 EMX simulations × 2h/sim = 20,000 CPU-hours. Build ML pipeline on
synthetic proxy data in parallel with data collection.

---

## Limitations

- **Synthetic dataset**: ABCD cascade is an approximation of real EM behavior. S11 is
  computed from the lossless identity (S11 = √(1−|S21|²)) rather than full EM simulation.
- **Basic ABCD model**: nearest-neighbor mutual coupling only; no substrate via coupling,
  no port discontinuities, no metal layer parasitics beyond Q_L/Q_C.
- **No tuning states**: the 3-bit switched-capacitor tuning state is documented in
  `data/generate_otfl301v2.py` and trivially added by conditioning on bit pattern.
- **N=4 not included**: N ∈ {3, 5} only in this demo dataset; N=4 was dropped for
  cleaner demonstration and symmetry with typical Chebyshev order selection.
- **Tail behavior**: approximately 10–15% of cINN samples have rt_mse > 20 dB² even
  with K=50. These correspond to high-k_m + high-C_sub samples where parasitic effects
  dominate. This is a known limitation, not a metric artifact.
