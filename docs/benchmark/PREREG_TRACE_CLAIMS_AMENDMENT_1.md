# Amendment 1 to the TRACE-claims pre-registration (H5–H8)

Date: 2026-10-12. Amends [PREREG_TRACE_CLAIMS.md](PREREG_TRACE_CLAIMS.md) (frozen at `32943cc`),
which stays in the repository unchanged; results will be reported under this amendment, and
under the original wording wherever that can still be computed.

**State at amendment time: no TRACE results exist.** `traceot.shift_decompose` does not exist in
any branch (the shift v2 implementation, PR #8, exposes no such binding yet), the
`TraceMethod` adapter raises, and no TRACE record exists under `benchmarks/results/`. Nothing
about TRACE's behaviour on ShiftBench was known when these changes were made. Rival results on
the frozen tracks exist (the ShiftBench run) but none of the new scenarios below had been run
for any method.

**Reason.** An independent referee report ([REFEREE_REPORT.md](../theory/REFEREE_REPORT.md),
merged as PR #9, `879d087`) found that H5.1 is tautological, the H7 planted channels are impure
and its rival test is on the wrong statistic, and the H8(a) bar is not a 5% test and its
Clopper–Pearson bound was unspecified and inconsistent with the stated count. Each change
below cites the referee item. Everything not listed here is unchanged.

## A1. H5.1: independent barycenter reconstruction (replaces the original item 1)

*Referee:* `Σ_k Σ_j Σ_r φ_{kjr} − Σ_k (Δc̄_k)²/(2k+1)` holds by construction (it is the stage-1
efficiency identity E1 composed with Parseval), so it tests the code's arithmetic, not the
budget. It is kept, renamed **H5.1a (implementation identity)**, and no longer counts as
evidence.

**H5.1b (new, primary).** Compute the target W2² without the coefficients: for each period t
evaluate TRACE's fitted conditional quantile function `Q̂_t(u|x_i)` for every row i of period t
on the 256-node Gauss–Legendre grid on [0,1], average over rows to get `Q̄̂_t(u)` (the Vincent
barycenter), and integrate `W_rec = Σ_m ω_m (Q̄̂_B(u_m) − Q̄̂_A(u_m))²`. Monotone rearrangement,
if TRACE applies one, is applied before averaging exactly as in prediction. Compare with
`Σ φ_{kjr}`. Bar: `|W_rec − Σφ| ≤ 1e-8 · max(1, W_rec)` in every replication. A gap above the
bar, including one caused by rearrangement (which leaves the polynomial span), is a recorded
failure of the exact-budget claim on the fitted model. Needs the adapter call
`predict_quantile` (see [TRACE_ADAPTER_SPEC.md](TRACE_ADAPTER_SPEC.md)).

## A2. H7: shape shifts with mean **and** L-scale fixed (replaces §1 table and §4)

*Referee:* standardising to variance 1 does not fix λ₂ (lognormal s=0.6 moves λ₂ by 0.065,
scaled t₅ by 0.030), so the planted channel sets were impure; `e_k ≤ 0.05·sd` was loose; the
paired rival test was on `sd`, not on the shape channels.

**Scenarios** (`benchmarks/shiftbench/sb_dgp_shape.py`; the six original scenarios are
withdrawn, not run). Baseline as before: p features, active β = (1.0, −0.8, 0.6, 0.4, −0.3),
intercept 1, independent N(0,1) features, N(0,1) noise. Every new law is centred (mean 0).
`skew` = frozen `_skew_noise`; `skewneg` = −`skew`; `t5` = Student-t(5)·√(3/5);
`chi3` = frozen `_skew_feature` (standardised χ²₃).

| scenario | period B change | graded channels K* (frozen) | planted pieces |
|---|---|---|---|
| `skew_struct` | noise → `skew`·σ_B | {2} | structure only |
| `tail_struct` | noise → `t5`·σ_B | {3} | structure only |
| `skew_comp` | x0 → `chi3`; noise σ_B | {2} | composition on k=2 (σ_B offsets k=1, not graded) |
| `tail_comp` | x1 → `t5`; noise σ_B | {3} | composition on k=3 |
| `shape_mixed` | x0 → `chi3` **and** noise → `skewneg`·σ_B | {2} | composition (+) and structure (−) on k=2 |

**Calibration (fixes λ₁ and λ₂ exactly up to Monte Carlo).** All new laws have mean 0 and the
features keep mean 0, so `E[y]` = 1 in both periods (λ₁ fixed). σ_B is the root, found by
bisection on [0.3, 3] to relative tolerance 1e-6, of `λ₂(Y_B(σ)) = λ₂(Y_A)` on the oracle sample
(2,000,000 draws per period, seed 99, common random numbers across σ). σ_B is stored with the
truth and used for all replications. The achieved `|Δλ₁|, |Δλ₂|` of the oracle are reported and
must be ≤ 2e-3 (else the scenario is fixed before any method runs and the fix is logged).

**Truth.** For k = 0..3: unbiased sample L-moments of `Y_A, Y_B, Y_c = (X_B, structure A),
Y_c' = (X_A, structure B)` on the oracle sample; TRACE estimand = order-free split
(SHIFT_V2 §4); rival estimand = A-reference split (`comp = λ(Y_c) − λ(Y_A)`). Both stored in
`benchmarks/results/shiftbench_truth_shape.json`.

**Metric.** For graded k ∈ K*: `err_k = |comp̂_k − comp_k| + |struct̂_k − struct_k|` (L-moment scale,
each method against its own estimand), and the relative error `r_k = err_k / (|comp_k| + |struct_k|)`.
**Recovery** in a replication: `r_k ≤ 0.25` for every k ∈ K*, every planted nonzero piece with the
correct sign, and `|comp̂_0| + |struct̂_0| ≤ 0.05·sd(y_A)`. (`f_k` ratios are no longer used.)
**Bar:** recovery rate ≥ 0.80 at (20,000, 20, 20 reps) and ≥ 0.60 at (2,000, 5, 50 reps) in each
scenario. Coverage is graded only in H8(b) (single bar; the duplicate H7 coverage bar is dropped).

**Rivals on the shape channels (primary rival test).** DFL and CFM produce counterfactual laws;
their L-moments are read from the frozen fits without changing them: DFL from the reweighted
period-A sample (weighted L-moments via the weighted quantile function on the same 256-node grid),
CFM from the rearranged discrete law (atoms `vals`, masses `diff(F)`). Implemented as read-outs in
`benchmarks/shiftbench/sb_shape_readout.py`, which copies the frozen fitting code verbatim and
adds only the read-out. OB, RIF and SHAP have no L-moment output and are n/a. Paired test per
scenario, primary at (20,000, 20): `d_r = err_k^TRACE − err_k^best` (best of DFL/CFM by mean),
k the single graded channel. **Supported** if the 95% interval is below 0 in ≥ 4 of 5 scenarios.
Holm family: these 5 tests plus H6's 4 and H8's 2 primaries.

## A3. H8: one joint 5% test per replication and an exact CP bar

*Referee:* "composition band or structure band excludes 0" is a union of two 95% tests (≈ 0.08 under
negative correlation), not a 5% test; the CP bound was not stated as one- or two-sided and
the two-sided upper at 10/200 is 0.0900 > 0.09.

**Test (replaces the union rule).** In each null replication, with B bootstrap replicates of the
statistic vector θ (mean channel: θ = (comp_0, struct_0)), compute `t_i = θ̂_i / sd*(θ_i)` and the
bootstrap null quantile `c* = q_0.95( max_i |θ*_i − θ̂_i| / sd*(θ_i) )`. Reject iff `max_i |t_i| > c*`.
This is one test at nominal 5% for the whole vector. The same construction is used for the
channel-wide test (θ over k = 0..3, both roles: 8 entries) and the per-feature test (θ over
2p mean-channel feature pieces). The same test is applied to every rival from its own bootstrap
replicates (rivals without a quantity are n/a).

**Bar (mean channel, primary).** On the 200-replication extended null (B = 100), with x rejections:
pass iff the **exact one-sided 95% Clopper–Pearson upper bound** `Beta⁻¹(0.95; x+1, 200−x) ≤ 0.09`,
which is equivalent to **x ≤ 11** (upper bounds: x=10 → 0.0833, x=11 → 0.0894, x=12 → 0.0954).
The point estimate x/200 is reported. Note: with a true rate of exactly 0.05, P(x ≤ 11) ≈ 0.70, so
an exactly calibrated method fails ~30% of the time; this is accepted and stated now.
Channel-wide and per-feature: same bar (x ≤ 11), secondary.
H8(b) coverage bar unchanged (pooled ≥ 0.90, each scenario × channel ≥ 0.85), now including the five
A2 scenarios. H8(c) placebo unchanged (secondary), using the joint test.

## A4. Failure rules

Unchanged: every miss is recorded with scenario, size, replication and value, and stays on
record. The withdrawn H7 scenarios and the union-band H8(a) are reported as "withdrawn by
Amendment 1 before any result".

## Deviations logged after this amendment (before any method ran on the new scenarios)

* **D1 (scenario code, 2026-10-12).** Building the oracle truth, the λ₁ check `|Δλ₁| ≤ 2e-3` failed
  for `tail_comp` (−2.03e-3) and `shape_mixed` (−2.24e-3). λ₁ is equal by construction (every law
  has mean 0), so this is Monte Carlo noise (SE ≈ √(2·sd_A²/2·10⁶) ≈ 1.8e-3). The check is
  replaced by `|Δλ₁| ≤ 4·SE`; λ₂ is matched by calibration to ≤ 1.1e-7 in all five scenarios.
  No scenario or bar was changed. Only the oracle truth was looked at; no method ran.
* **D2 (disclosure, no change).** The oracle shows small planted signals on the graded channels
  (Shapley truth, L-moment scale): `skew_struct` struct λ₃ = 0.050; `tail_struct` struct λ₄ = 0.0097;
  `skew_comp` comp λ₃ = 0.040; `tail_comp` comp λ₄ = 0.0043; `shape_mixed` λ₃ comp +0.040 / struct
  −0.054. With `r_k ≤ 0.25` the `tail_*` scenarios require absolute errors of about 0.002 and 0.001,
  which may be infeasible at these sample sizes for any method. The bars are kept as frozen; a miss
  there will be reported as a failure, with this note. Ungraded channels also move (e.g. λ₄ in
  `skew_struct` and `shape_mixed`, ±0.007 on λ₂ pieces that cancel in `skew_comp`, `shape_mixed`)
  and are reported.
* **D3 (H8 test calibration note).** The joint max-|t| test with B = 100 rejected 5.75% on a toy
  Gaussian null (2,000 draws, `sb_h8.py`), consistent with small-B bootstrap quantile noise. No change.
