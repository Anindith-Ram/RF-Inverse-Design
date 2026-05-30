# Deferred cINN Improvement Proposals

Proposals 2, 3, and 6 from the May 2026 cINN improvement analysis. Bundle 1+4+5
was implemented immediately; these three are recorded here for follow-up iteration.

---

## Proposal 2 — Structural fc constraint via L/C ratio prediction

**Idea**: Replace the 2N-dim target `[log L_1, log C_1, ..., log L_N, log C_N]`
with an N-dim target `[log ρ_1, ..., log ρ_N]` where `ρ_k = L_k / C_k` (impedance²).

**Decoding** at inference:
- From spec fc, compute `ω0 = 2π fc`
- Constrain `L_k · C_k = 1/ω0²` (every resonator tunes to fc — classical Pozar synthesis)
- Then `L_k = √(ρ_k) / ω0`, `C_k = 1 / (√(ρ_k) · ω0)`

**Why it helps**:
- Halves the prediction dimension (N=3: D_y 6→3, N=5: 10→5)
- Encodes the fc invariant *structurally* instead of forcing the cINN to learn it
- Removes the most common failure mode: "predicted LC pair doesn't resonate at fc"

**Expected impact**: acc_frac +0.10–0.20 standalone.

**Implementation cost**: ~30 lines
- `data/generate_otfl301v2.py`: store `rho = L/C` alongside L, C
- `training/train_inn_v2.py`: target = log10(rho), D_y = N
- `evaluation/metrics.py`: add `decode_rho_to_LC(rho, fc)` helper
- All eval paths: decode rho to LC before synthesis

**Gotchas**:
- Process spread on L and C is INDEPENDENT in the current dataset. Predicting only
  ρ implicitly assumes L and C are perfectly correlated (always tuned to fc). True
  for nominal LC; *not* true after process spread is applied. So the cINN now sees
  a NOISIER target (rho has more spread than nominal). Net effect TBD empirically.
- Alternative: predict (log ρ, log(LC product)) — 2 dims per resonator instead of 1,
  but `log(LC product)` should be very tight around `−log(ω0²)`. This keeps process
  spread as a small auxiliary signal.

---

## Proposal 3 — Local refinement via differentiable ABCD

**Idea**: After sampling K=50 LC candidates from the cINN, treat each as a starting
point for 20 Adam steps of gradient descent on the differentiable ABCD round-trip loss
(using ground-truth parasitics). Return the polished candidate per starting point;
take best-of-K across polished candidates.

**Why it helps**:
- Phase 4b tandem training failed because differentiable ABCD gradients are *unstable
  from random init* — large discontinuities near resonance peaks
- Starting *near a resonance* (from a cINN sample), the gradient is well-defined and
  points sharply uphill
- Exact physics objective beats any neural surrogate

**Expected impact**: acc_frac → 0.7–0.85 at K=50 if cINN candidates are within 5–10%
of true LC (which they should be after Bundle 1+4+5).

**Implementation cost**: ~80 lines
- Reuse the differentiable ABCD code from the failed tandem experiment
  (`training/train_tandem.py` — archived but recoverable)
- New function `refine_lc(L_init, C_init, target_s21, target_s11, parasitics, n_steps=20)`
- Apply per sample in `make_final_figures.py` and `evaluate_test_set`
- Adds ~5–10 ms per sample at inference (acceptable)

**Gotchas**:
- Need to clamp s21_norm at 1e-5 (-100 dB floor) before log10 to avoid stopband NaNs
- Gradient clip at norm=1.0 to prevent the resonance-peak spike from blowing up Adam
- Learning rate is the critical hyperparameter — start at 1e-3, anneal to 1e-4

---

## Proposal 6 — Larger training dataset (50k → 200k or 500k)

**Idea**: Regenerate `data/dataset_otfl301v2.pkl` at 200k or 500k samples.

**Why it helps**:
- Current 50k samples × 80/20 split = 40k train, 20k per-N — only 20k per cINN
- 1.7M parameter model / 20k samples = 85 samples per parameter — severe overfit ratio
- Even after Bundle 5 (smaller model, 500k params), 40 samples/param is borderline

**Expected impact**: +0.03–0.10 acc_frac, primarily by reducing the train/val NLL gap.

**Implementation cost**: 1-line change in generator + ~30 min regen time at 200k.

**Gotchas**:
- The generator runs sequentially. At ~6 ms/sample, 200k samples ≈ 20 min, 500k ≈ 50 min.
  Could parallelize with multiprocessing but adds complexity.
- Disk: pickle file grows from ~80 MB → 320 MB / 800 MB. Fine on local disk.
- All scaler stats and benchmarks need to be recomputed — should be transparent.

---

## Ordering recommendation

If acc_frac after Bundle 1+4+5 is still below 0.50:

1. **Try Proposal 6 first** (cheap data scaling) — diagnoses whether the bottleneck is
   data quantity vs model architecture.
2. **Then Proposal 2** (structural fc constraint) — architecturally cleanest fix.
3. **Proposal 3 last** (local refinement) — most powerful but most code complexity.
   Reserve for when cINN is already producing decent candidates.

If acc_frac after Bundle 1+4+5 is **above 0.60**:

- Skip Proposals 2 and 6 — diminishing returns
- Go directly to Proposal 3 for the final 0.60 → 0.85 push
