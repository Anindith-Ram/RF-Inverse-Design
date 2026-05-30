# RF Inverse Design ML Pipeline

## Project Overview
End-to-end ML pipeline for RF bandpass filter inverse design. Trains models to map
desired S-parameter specifications → filter component values (physical LC elements),
mirroring the workflow used in Cadence EMX-based on-chip mm-wave filter design at
60–90 GHz.

The classical synthesis procedure (Pozar Ch. 8) gives exact LC values for ideal
filters. On-chip CMOS at mm-wave frequencies, components are not ideal — parasitic
inductance, substrate coupling, skin-effect loss, and fringing fields mean the math
no longer closes. EMX EM simulation is required to compute accurate S-parameters,
but each run takes hours. This pipeline trains neural networks to approximate the
inverse mapping: S-params → LC geometry, collapsing the iterative simulation loop
into a single forward pass.

**Current dataset state:** The active dataset is **OTFL301v2**
(`data/dataset_otfl301v2.pkl`): 50k samples, N ∈ {3, 5}, 24–40 GHz SOI, 207-dim X_full,
4 physics improvements (mutual coupling, substrate caps, freq-dep Q_C, process spread),
no tuning bits. Legacy datasets are in `data/archive/`.

---

## RF Background (essential for understanding this codebase)

### S-parameters
Two-port network measurement standard. All measurements at 50 Ω terminations.
- S11: reflection at port 1 = V1⁻/V1⁺
- S21: transmission from port 1 to port 2 = V2⁻/V1⁺
- Expressed in dB: 20·log₁₀(voltage ratio)
- Passband target: S21 near 0 dB (low IL), S11 below −20 dB (low reflection)
- Stopband target: S21 below −30 to −40 dB (strong rejection)

### Filter specs
- fc: center frequency (GHz) — always stored as fc_nominal (design target)
- FBW: fractional bandwidth = BW / fc (dimensionless, 0–1)
- IL: insertion loss = S21 drop below 0 dB in passband (spec: < 2 dB ideal, up to 8 dB lossy)
- RL: return loss = S11 in passband (spec: < −20 dB ideal)

### fc_nominal — CRITICAL DISTINCTION
The active datasets use fc_nominal (the design parameter) exclusively.
- **Right**: store fc_nominal = the random float used to compute L/C values
- **Wrong**: store fc_actual = grid-snapped S21 peak from FREQ_HZ
- Why: ±0.25 GHz grid snap causes 80–250 dB² round-trip MSE even with perfect predictions

### g-values and the prototype filter
g-values (g₁..gN+1) are dimensionless normalized element values. For this pipeline,
they are an intermediate step: g-values are computed from (N, ripple_dB), then
scaled to physical L/C via bandpass prototype transformation:

```
Series branch k (odd, 1-indexed):  L_k = g_k·Z0/(ω0·fbw),  C_k = fbw/(g_k·Z0·ω0)
Shunt  branch k (even, 1-indexed): L_k = fbw·Z0/(g_k·ω0),  C_k = g_k/(fbw·Z0·ω0)
```

where ω0 = 2π·fc, Z0 = 50 Ω, and the ladder starts with a series branch.

**Key property**: L_k · C_k = 1/ω0² for all k — every LC pair resonates at fc.
This is a useful sanity check: after computing L/C from g-values, verify
1/(2π√(L_k·C_k)) ≈ fc for each resonator.

### Non-uniqueness problem — CRITICAL FOR MODEL SELECTION
Multiple different LC configurations can produce near-identical S-parameter responses.
A naive MLP minimizes E[||ŷ − y||²]; for multi-modal posteriors p(y|X), the minimizer
is the conditional mean — a blend of modes that is physically invalid for each.

**Two sources of non-uniqueness in the realistic dataset:**
1. **Q as unobserved confounder**: S-params are generated from (L, C, Q). The model
   sees S-params and predicts L/C, but Q is hidden. Different (L/C, Q) combinations
   can produce similar S-param shapes (lower Q + slightly different L/C = same passband
   depth). The model cannot distinguish these from S-params alone.
2. **Mode averaging**: The MLP averages across modes → predicted L/C satisfies none.

**Evidence from results**: scalar model (5-dim, no S-params) ≥ full model (207-dim)
on component metrics. S-param curves don't help because they carry ambiguous information
when Q is unobserved. This motivates tandem networks (Phase 4b) and INNs (Phase 4c).

**Result**: comp_mse ≈ 0.9994 R² in log-space but rt_mse = 36–46 dB² — the model
predicts LC values that are near the ground truth in log-space, but the ABCD synthesis
is highly sensitive near resonance. A 1% error in L and C shifts resonances by ~0.5%
(~325 MHz at 65 GHz), causing large S-param deviations in steep transition regions.

### Round-trip error — PRIMARY METRIC
Procedure for the **realistic** dataset (LC targets):
1. Model predicts log10(LC) → exponentiate → L_pred, C_pred
2. Synthesize S-params via lossy ABCD cascade using ground-truth Q values from dataset
3. Compute MSE between synthesized S-params and original target S-params (dB²)

Procedure for the **synthetic** dataset (g-value targets, legacy):
1. Model predicts g-values
2. Infer ripple_dB from predicted g1 via inverse formula
3. Synthesize via scipy.signal.cheby1
4. MSE between synthesized and target S-params

Always use rt_mse as PRIMARY for model selection. comp_mse/R² are secondary/diagnostic.

### Passband Mask — CRITICAL DETAIL
Always use a **10% INSET** from nominal passband edges:
```python
f_inset = (f_high - f_low) * 0.10
mask = (FREQ_HZ >= f_low + f_inset) & (FREQ_HZ <= f_high - f_inset)
```
Never extend outward (f_low*0.95 to f_high*1.05) — for N=5, this captures transition
frequencies where S21 is already −10 to −25 dB, causing false IL failures.

---

## Dataset Architecture

### OTFL301v2 — ACTIVE (`data/dataset_otfl301v2.pkl`)
Generated by `data/generate_otfl301v2.py`. All active training uses this dataset.

| Property | Value |
|----------|-------|
| Samples | 50,000 |
| N values | {3, 5} only |
| ripple_dB | Uniform[0.05, 2.0] (continuous) |
| fc_GHz | Uniform[24, 40] (SOI 24–40 GHz band) |
| fbw | Uniform[0.08, 0.50] |
| Q_L per inductor | Uniform[15, 30] (per-element random) |
| Q_C per capacitor | Uniform[100, 300] (per-element random) |
| k_m (mutual coupling) | Uniform[0.02, 0.12] per adjacent pair |
| C_sub_frac | Uniform[0.03, 0.15] per element |
| alpha_C | Uniform[0.1, 0.5] per element (freq-dep Q_C exponent) |
| Process spread | σ_L=4%, σ_C=3% log-normal |
| S-param synthesis | Enhanced lossy ABCD (v2) |
| X_full dims | 207 (no tuning bits) |

**Dataset dict keys:**
```
X_full      (50000, 207)  [fc_GHz, fbw, ripple_dB, N3_flag, N5_flag, S21×101, S11×101]
X_scalar    (50000, 5)    [fc_GHz, fbw, ripple_dB, N3_flag, N5_flag]
y           (50000, 10)   [L1,C1,...,L5,C5] in SI, NaN-padded for N=3
y_log       (50000, 10)   log10(y), NaN-padded — USE THIS AS ML TARGET
N           (50000,)      filter order (3 or 5 only)
fc_GHz      (50000,)      fc_nominal
fbw         (50000,)
ripple_dB   (50000,)
Q_L         (50000, 5)    per-inductor Q (NaN-padded) — metadata for honest eval
Q_C         (50000, 5)    per-capacitor Q (NaN-padded) — metadata for honest eval
k_m         (50000, 4)    mutual coupling coefficients (NaN-padded for N=3)
C_sub_frac  (50000, 5)    substrate cap fraction (NaN-padded for N=3)
alpha_C     (50000, 5)    freq-dep Q_C exponent (NaN-padded for N=3)
IL_dB       (50000,)
RL_dB       (50000,)
```

**S-param offsets in X_full:**
- X_full[:,5:106]   = S21_dB (101 points, 24–40 GHz)
- X_full[:,106:207] = S11_dB (101 points)

### Legacy Datasets — ARCHIVED (`data/archive/`)
- `dataset_realistic.pkl`: 50–85 GHz, N∈{3,4,5}, no mutual coupling/substrate caps
- `dataset_otfl301.pkl`: 24–40 GHz V1 (without alpha_C and process spread)
- `dataset.pkl`: synthetic g-value targets (degenerate 6-class, do not use)

---

## Architecture Decisions (Realistic Dataset)

### Inverse models (`models/mlp.py`)
- **InverseMLP**: 207-dim input → [512→256→128] trunk → three N-specific heads
  - head_3: 128 → 6  (L1,C1,L2,C2,L3,C3)
  - head_4: 128 → 8  (L1,C1,...,L4,C4)
  - head_5: 128 → 10 (L1,C1,...,L5,C5)
- **SpecsOnlyMLP**: 5-dim input → same trunk → same three heads (ablation baseline)
- Dropout: 0.2, BatchNorm at each layer
- Targets: y_log (log10-scale LC), normalized by separate StandardScaler per N group
- Three scalers: y_scaler_3 (6-dim), y_scaler_4 (8-dim), y_scaler_5 (10-dim)

**X_full (207-dim) layout:**
```
[0]     fc_GHz (fc_nominal)
[1]     fbw
[2]     ripple_dB
[3]     N3_flag  (1 if N==3, else 0)
[4]     N5_flag  (1 if N==5, else 0)
[5:106] S21_dB at 101 freq points (40–90 GHz)
[106:207] S11_dB at 101 freq points
```

### Forward surrogate (`models/forward_model.py`)
- **ForwardMLP**: 24-dim input → [512→512→256] → 202-dim output
- Input layout: [log10(L1,C1,...,L5,C5), Q_L1..5, Q_C1..5, fc_GHz, fbw, N3_flag, N5_flag]
  - NaN-padded positions (unused slots for smaller N) replaced with 0
  - N one-hot as [N3_flag, N5_flag]: N=4 → [0, 0]
- Output: [S21_dB(101), S11_dB(101)] = 202-dim
- Dropout: 0.1 (less than inverse — forward mapping is deterministic)
- **Checkpoint embeds normalization buffers**: `model.predict(x_raw)` is self-contained
  for use in tandem training without external scaler objects
- Key helpers:
  - `build_forward_input(y_log, Q_L, Q_C, fc_GHz, fbw, N)` → (n, 24) numpy
  - `build_forward_input_torch(...)` → (batch, 24) tensor with gradient path through y_log

### Normalization strategy
- X_full, X_scalar: StandardScaler on training set
- y_log targets: separate StandardScaler per N group (y_scaler_3/4/5)
- No zero-variance issue: LC values are fully continuous; no gN+1=1.0 degeneracy
- y_scaler fit on `y_log` (log10 values), not raw SI values
- Random seed: 42 everywhere

### Stratified split
With continuous ripple_dB (no discrete groups), stratify by N alone:
```python
train_test_split(all_idx, test_size=0.20, stratify=ds['N'], random_state=42)
```
80/10/10 split → train=40k, val=5k, test=5k.

---

## Phase Status

### Phase 1 — Synthetic Data Generation ✅ COMPLETE (legacy)
- File: `data/generate_synthetic.py`
- 6000 samples (1000 per group × 6 groups), g-value targets, 0 discarded
- Key fixes: passband mask inset, fc_nominal storage

### Phase 1b — Realistic Dataset Generation ✅ COMPLETE
- File: `data/generate_realistic.py`
- 50,000 samples, physical LC targets, lossy ABCD synthesis
- Per-sample Q randomization creates genuine non-uniqueness
- 50,000 unique output vectors (verified in log10-space)

### Phase 2 — MLP Baseline on Synthetic Dataset ✅ COMPLETE (legacy)
Results were misleadingly good due to dataset degeneracy (6-class classification):
```
Model        | N | comp_mse  | comp_mae  | r2_mean | rt_mse (dB²)
MLP (full)   | 3 | 0.000017  | 0.002295  | 0.9998  | 0.0015
MLP (full)   | 5 | 0.000011  | 0.001993  | 0.9999  | 0.0009
MLP (scalar) | 3 | 0.000008  | 0.001794  | 0.9998  | 0.0006
MLP (scalar) | 5 | 0.000010  | 0.002252  | 0.9999  | 0.0014
```
Full and scalar models match — S-params add no information over ripple_dB because
g-values depend ONLY on (N, ripple_dB). Task reduces to 6-class classification.

### Phase 2b — MLP Baseline on Realistic Dataset ✅ COMPLETE
Files: `models/mlp.py`, `training/train_mlp.py`, `evaluation/metrics.py`,
       `evaluation/visualize.py`

Results (test set, 5000 samples). comp_mse/mae/R² in log10-space:
```
Model        | N | comp_mse(log) | comp_mae(log) | r2_mean | rt_mse (dB²)
MLP (full)   | 3 | 0.000034      | 0.004168      | 0.9994  | 36.49
MLP (full)   | 4 | 0.000032      | 0.004156      | 0.9994  | 23.86
MLP (full)   | 5 | 0.000034      | 0.004208      | 0.9994  | 39.69
MLP (scalar) | 3 | 0.000024      | 0.003624      | 0.9996  | 42.24
MLP (scalar) | 4 | 0.000017      | 0.003141      | 0.9997  | 23.87
MLP (scalar) | 5 | 0.000020      | 0.003345      | 0.9996  | 45.78
```
**Key findings:**
- R²=0.9994 in log-space but rt_mse=24–46 dB² — ABCD synthesis is highly sensitive
  near resonance; 1% LC error → ~325 MHz resonance shift → large S-param deviation
- Scalar model ≥ full model on component metrics → S-param curves don't help because
  Q is an unobserved confounder causing multi-modal posterior p(y|X)
- Checkpoints: `results/mlp_realistic_best.pt`, `results/mlp_realistic_scalar_best.pt`

### Phase 3 — Non-uniqueness Demonstration ⏳ PENDING
File: `experiments/nonuniqueness_demo.py`
Evidence already visible in Phase 2b (scalar ≥ full; rt_mse >> comp_mse).
Formal demo: find test samples where multiple (L/C, Q) pairs produce near-identical
S-params but different L/C → show MLP averages between them.

### Phase 4a — Forward Surrogate ✅ COMPLETE
Files: `models/forward_model.py`, `training/train_forward.py`
Maps (LC, Q, specs) → S-params. Prerequisite for tandem (4b) and CMA-ES (4d).

Architecture: ForwardMLP [24 → 512 → 512 → 256 → 202], dropout=0.1
- Input: [log10(L1,C1,...), Q_L1..5, Q_C1..5, fc_GHz, fbw, N3_flag, N5_flag]
- Output: [S21_dB(101), S11_dB(101)], S11 clipped at -60 dB floor during training
- Normalization buffers embedded in checkpoint for self-contained tandem use
- Gradient-safe: `predict()` and `build_forward_input_torch()` maintain gradient
  flow from y_log_pred through the forward model to the tandem loss

Results (test set):
  S21 MSE = 0.032 dB²   R²(S21) = 0.9965
  S11 MSE = 8.97 dB²    R²(S11) = 0.889
S11 quality is lower due to sharp notches in the passband (see Known Bug #7 below).
S21 quality is excellent and sufficient for tandem training.
Checkpoint: `results/forward_model_best.pt`

### Phase 4b — Tandem Network ❌ FAILED (5 training runs)
Files: `models/tandem.py`, `training/train_tandem.py`
Root cause of failure: differentiable ABCD loss landscape is too non-smooth for
gradient descent. Resonance peaks create large discontinuous gradients — even with
BETA_TARGET=0.001 + linear ramp over 50 tandem epochs, the loss diverged at epoch 110.

All 5 runs outcome summary:
- Runs 1-3: Neural ForwardMLP surrogate — rt_mse 72-253 dB² (surrogate bias amplified)
- Run 4: Differentiable ABCD, BETA=0.01 → phase 2 immediately destroyed LC predictions
- Run 5: Differentiable ABCD, BETA_TARGET=0.001 + ramp → diverged at epoch 110

Best result: warm-start checkpoint (phase 1 only) rt_mse = 37.8/36.6/49.0 dB² (N=3/4/5).
No improvement over MLP baseline via tandem approach. Approach abandoned.

### Phase 4c-V2 — cINN with Bundle 1+4+5 (OTFL301v2) ✅ COMPLETE
Files: `models/inn_v2.py`, `training/train_inn_v2.py`

Final architecture (Bundle 1+4+5):
- `ConditionEmbedderV2`: [231 → 256 → 128 → 128], 109k params
- `make_cinn_v2`: 8 × AllInOneBlock (FrEIA), subnet_dim=128, COND_DIM=128, 274k params
- **D_y = 2·N (LC only)** — Q is conditioned on, not predicted
- **Bundle 1 — parasitic conditioning** (the headline fix): condition vector = X_full(207) +
  parasitics(24): Q_L(5) + Q_C(5) + k_m(4) + C_sub_frac(5) + α_C(5). Built by
  `training/train_inn_v2.py::build_full_x()`. NaN-padded slots zero-filled for N=3.
  Effect: posterior p(LC | S, parasitics) becomes near-deterministic; eliminates the
  parasitic confounding that capped baseline acc_frac at ~0.18 (N=3) / ~0.05 (N=5).
- **Bundle 4 — z-reg loss**: `loss = NLL + 1.0·(mean(std(z, batch)) − 1)²` prevents the
  affine-scaling degenerate mode that inflated `log|det J|` and collapsed val NLL.
- **Bundle 5 — smaller model**: N_BLOCKS 12→8, SUBNET_DIM 256→128, weight_decay 1e-5→1e-4.
  Total params 1.75M → 383k per N. Matches data scale (20k samples per N).
- `AFFINE_CLAMP = 1.5`, `RT_THRESHOLD = 5.0 dB²`
- MPS fix: `fix_mps_contiguity(inn)` after `.to('mps')`

**Final test-set results (5000 samples, K=50 best-of-50):**
| Metric | MLP baseline | cINN best-of-1 (z=0) | cINN best-of-50 |
|---|---|---|---|
| N=3 rt_mse | 100.7 dB² | 46.1 dB² | **2.69 dB²** |
| N=3 acc_frac | 0.162 | 0.554 | **0.975** |
| N=5 rt_mse | 128.3 dB² | 98.9 dB² | **8.01 dB²** |
| N=5 acc_frac | 0.062 | 0.184 | **0.890** |

cINN best-of-50 achieves **37× rt_mse improvement** over MLP on N=3 and **16×** on N=5;
**97.5% of N=3 specs and 89.0% of N=5 specs** yield at least one < 5 dB² candidate.

Deferred improvements (see `PROPOSALS_FUTURE.md`):
- Proposal 2 — predict only L/C ratio (encode fc constraint structurally)
- Proposal 3 — local refinement via differentiable ABCD from cINN starting points
- Proposal 6 — scale dataset to 200k–500k samples for further N=5 gains

### Phase 4a (V1) — Phase 4b archived
- Phase 4a ForwardMLP and Phase 4b tandem experiments are in `results/archive/`
- All legacy 50–85 GHz V1 checkpoints in `results/archive/`

### Phase 5 — Final Demo Figures
File: `evaluation/make_final_figures.py`
Produces 5 figures + `results/final_benchmark_table.txt` from trained checkpoints.
Run after Phase 4c-V2 training completes.

---

## Project Structure (Active Files)
```
RF-Inverse-Design/
├── CLAUDE.md
├── README_DEMO.md                     # one-page demo summary
├── requirements.txt
├── rf_env/                            # Python virtual environment (not committed)
├── data/
│   ├── generate_otfl301v2.py          # ACTIVE: generates dataset_otfl301v2.pkl
│   ├── dataset_otfl301v2.pkl          # ACTIVE dataset (gitignored)
│   └── archive/                       # legacy generators + datasets
├── models/
│   ├── mlp.py                         # InverseMLP + SpecsOnlyMLP (2 heads: N=3, N=5)
│   ├── inn.py                         # FrEIA utilities (fix_mps_contiguity, verify_bijection)
│   └── inn_v2.py                      # Phase 4c-V2: ConditionEmbedderV2 + make_cinn_v2
├── training/
│   ├── train_mlp.py                   # MLP baseline on OTFL301v2
│   └── train_inn_v2.py                # cINN V2 training, D_y=2N, honest eval
├── evaluation/
│   ├── metrics.py                     # synthesize_from_lc (with parasitic kwargs),
│   │                                  # roundtrip_mse_lc, component_mse, r2_per_component
│   ├── visualize.py                   # loss curves, LC scatter, S-param overlays
│   └── make_final_figures.py          # 5 demo figures + benchmark table
└── results/
    ├── figures/                       # all saved plots
    ├── mlp_otfl301v2_best.pt          # InverseMLP checkpoint (OTFL301v2)
    ├── mlp_otfl301v2_scalar_best.pt   # SpecsOnlyMLP checkpoint (OTFL301v2)
    ├── inn_v2_otfl301v2_N3_best.pt    # cINN V2 checkpoint N=3
    ├── inn_v2_otfl301v2_N5_best.pt    # cINN V2 checkpoint N=5
    ├── final_benchmark_table.txt      # headline metrics (after training)
    └── archive/                       # legacy checkpoints (V1, 50-85 GHz, tandem, etc.)
```

## Running the Demo Pipeline
```bash
source rf_env/bin/activate

# (Optional) Regenerate dataset — already exists
python data/generate_otfl301v2.py

# Retrain MLP baseline (~15 min)
python -u training/train_mlp.py 2>&1 | tee /tmp/mlp_train.log

# Retrain cINN N=3 + N=5 (~35 min)
python -u training/train_inn_v2.py 2>&1 | tee /tmp/inn_train.log

# Generate all figures + benchmark table (requires trained checkpoints)
python evaluation/make_final_figures.py

cat results/final_benchmark_table.txt
```

---

## Evaluation Metrics Reference

### For realistic dataset (LC targets)
| Metric | Description | Priority |
|--------|-------------|----------|
| comp_mse (log) | MSE on log10(LC) predictions | secondary |
| comp_mae (log) | MAE on log10(LC) predictions | secondary |
| r2_per_pos | R² per LC position (log-space) | diagnostic |
| roundtrip_mse_lc | MSE: target S-params vs ABCD(LC_pred, Q_true) | **PRIMARY** |

Round-trip uses ground-truth Q from dataset — isolates LC prediction error from Q uncertainty.

### For synthetic dataset (g-value targets, legacy)
| Metric | Description | Priority |
|--------|-------------|----------|
| comp_mse | MSE on g-value predictions | secondary |
| roundtrip_mse | MSE: target S-params vs cheby1(g_pred) | **PRIMARY** |

### For forward model
| Metric | Description | Target |
|--------|-------------|--------|
| s21_mse_db2 | S21 MSE in dB² | < 1.0 dB² |
| s11_mse_db2 | S11 MSE in dB² | < 1.0 dB² |
| r2_s21 | mean R² across 101 freq points | > 0.99 | achieved: 0.9965 |
| r2_s11 | mean R² across 101 freq points | > 0.80 (after S11 floor, see Bug #7) | achieved: 0.889 |

---

## Known Bugs Fixed (do not reintroduce)

### 1. Passband mask extension direction
- **Wrong**: `mask = (freq >= f_low*0.95) & (freq <= f_high*1.05)`
- **Right**: `f_inset = (f_high - f_low)*0.10; mask = (freq >= f_low+f_inset) & (freq <= f_high-f_inset)`
- Why: outer extension captures steep transition for N=5 → false IL failures

### 2. fc_actual vs fc_nominal (Phase 1 → all phases)
- **Wrong**: store fc_actual = grid-snapped S21 peak
- **Right**: store fc_nominal = design parameter fed to synthesis
- Why: ±0.25 GHz error causes 80–250 dB² round-trip MSE even with perfect predictions

### 3. Zero-std StandardScaler (synthetic dataset only)
- gN+1 = 1.0 always (odd N) → scale_=0 → division by zero
- Fix: `scaler.scale_ = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)`
- Not an issue for realistic dataset (all LC values are continuously varying)

### 4. ReduceLROnPlateau verbose kwarg removed in PyTorch 2.x
- Remove `verbose=False` from constructor call

### 5. Uniqueness check on SI-scale values (generate_realistic.py)
- **Wrong**: `np.round(y, decimals=6)` — rounds L~1e-12 to 0, gives 1 unique vector/N
- **Right**: `np.round(y_log, decimals=4)` — log10-space has appropriate precision

### 6. ForwardMLP predict() / forward() mode interaction
- `forward()` expects pre-normalized input; `predict()` normalizes internally
- Normalization consistency test must use `.eval()` mode to ensure deterministic BN
- In train mode, BN uses batch statistics; in eval mode, uses running statistics
- For tandem training, call `predict()` from train mode — gradients flow correctly

### 7. S11 floor: lossless identity creates -120 dB numerical artifacts
- `generate_realistic.py` computes S11 from S21 using the lossless identity:
  `S11 = sqrt(max(1 - |S21|^2, 1e-12))`. At the passband peak, S21 → 0 dB,
  so 1 - |S21|^2 → 0, giving S11 → -120 dB (numerical floor from `1e-12`).
- These -120 dB values are physically meaningless (perfect matching), but
  dominate MSE loss (S11 MSE = 54.9 dB² vs 8.97 dB² with floor).
- **Fix**: clip S11 at -60 dB in `train_forward.py` and `train_tandem.py`:
  `y_sp[:, 101:] = np.maximum(y_sp[:, 101:], -60.0)`
  `s11_target.clamp(min=-60.0)` in the tandem batch loss
- The assertion threshold for R²(S11) is relaxed to 0.80 (vs 0.90 for S21)
- Note: if `generate_realistic.py` is regenerated to use the actual ABCD-computed
  S11 (not the lossless identity), this workaround is no longer needed.

---

## Pretrained Models Assessment
No publicly available pretrained models exist for 40–90 GHz on-chip filter inverse
design. Searched: Hugging Face, GitHub (InvDesignNet, CloakingNet, photonics repos),
IEEE papers. Findings:
- FrEIA framework exists (Ardizzone 2019) — provides INN training infrastructure,
  no pretrained weights for RF domain
- Closest public work: microstrip filter CNN (IEEE 2024) at 0.5–2 GHz — 30× frequency
  mismatch, not transferable
- FNO/RNN surrogates (research papers only, no weights)
- **Decision: train all models from scratch on dataset_realistic.pkl**

---

## Phase 4 Architecture Guide (for new sessions)

### Why MLP fails (non-uniqueness)
Mode averaging: argmin E[||I(x)-y||²] = E[y|x] = mean of all valid modes.
Mean of two valid LC configurations is a third point that satisfies neither.
Evidence: scalar model ≥ full model; rt_mse 24–46 dB² despite R²=0.9994.

### Phase 4a: Forward surrogate — PURPOSE
- Unique mapping: (LC, Q, specs) → S-params, no ambiguity, trains cleanly
- Used as **frozen differentiable objective** in tandem training
- Used as **fast objective function** in CMA-ES optimization
- train_tandem.py loads `results/forward_model_best.pt` and calls `model.predict()`

### Phase 4b: Tandem network — HOW IT ADDRESSES NON-UNIQUENESS
Loss = α·||I(x)-y||² + β·||ABCD(I(x))-x_S21||² with α=1.0, β=0.01
Forward-consistent loss: penalizes predictions that DON'T synthesize the target S-params,
not predictions that differ from the "true" (single-mode) y. Any valid mode scores zero.
Training: (1) warm-start 50 epochs with LC supervision only; (2) add differentiable ABCD loss.

**Use differentiable ABCD, not neural ForwardMLP, for the tandem loss.**
The ForwardMLP has 0.032 dB² S21 MSE — small, but the tandem loss magnifies this into
systematic bias: the inverse model learns to satisfy the surrogate's errors, not true physics.
Result: neural-forward tandem rt_mse = 72-98 dB² vs MLP baseline 24-46 dB².
Differentiable ABCD (exact physics) avoids this. Gradient chain: y_log_pred → 10^y →
complex ABCD cascade → S21 magnitude → dB normalization → MSE.

**Gradient concerns**: ABCD S21 loss creates large gradients at stopband (20·log10 near 0).
Fix: clamp s21_norm at 1e-5 (= -100 dB floor) before log10, and clip gradients at norm=1.0.

**Loss scale balance**: Converged LC loss ≈ 0.003 normalized; ABCD fwd loss ≈ 24-46 dB².
BETA=0.01 gives combined ≈ 0.003 + 0.01·30 = 0.3, keeping both terms meaningful.

**Early stopping reset**: Must reset best_val and no_improve when Phase 2 starts.
Without this, Phase 2 combined loss > Phase 1 LC loss → immediate early stopping.

### Phase 4c: INN — HOW IT ADDRESSES NON-UNIQUENESS MORE FUNDAMENTALLY
cINN models full posterior p(y|x) via bijection (y, x) ↔ (z, x), z~N(0,I).
Inference: sample z, compute y = f⁻¹(z|x). Multiple samples → multiple valid designs.
ONLY approach that enumerates diverse solutions — critical for Otava use case where
designers need multiple candidate geometries, not just one answer.
Framework: FrEIA (vislearn/FrEIA on GitHub). Requires more tuning than tandem.

### Phase 4d: CMA-ES — GOLD STANDARD BENCHMARK
No training needed for inverse step. Uses ForwardMLP as objective.
Achieves near-optimal rt_mse by optimization, not learning.
Too slow for batch inference but establishes the performance ceiling for comparison.

---

## Connection to Otava Internship
At Otava, input X = desired S-param specs from EMX simulation targets.
Output y = physical geometry parameters: resonator length, coupling gap width,
metal layer dimensions — approximately 10–20 continuous parameters per filter.

EMX dataset collection: run thousands of simulations across swept geometry parameter
grids. This pipeline uses LC values as geometry proxy — swapping in EMX data requires:
1. New data loader (CSV/HDF5 from EMX)
2. `y` = geometry params (not LC values)
3. Round-trip eval uses trained ForwardMLP (not ABCD formula — EMX responses are not
   analytically synthesizable)
4. Dataset will be smaller (~thousands) — more regularization, fewer epochs
5. Non-uniqueness will be more severe — INN is the most relevant Phase 4 model

**Scale reality check**: 10,000 EMX simulations × 2 hours/sim = 20,000 CPU-hours.
Start data collection early; build ML pipeline in parallel on synthetic data.
