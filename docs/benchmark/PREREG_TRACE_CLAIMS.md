# Pre-registration addendum: claims about TRACE (H5–H8)

Addendum to [PREREG_SHIFTBENCH.md](PREREG_SHIFTBENCH.md) (frozen at `c0d7c0f`). Written and
committed **before TRACE has produced any output on ShiftBench**: the `TraceMethod` adapter in
`benchmarks/shiftbench/sb_adapter.py` still raises, and no TRACE record exists under
`benchmarks/results/`. The commit that adds this file is the freeze point for H5–H8.
Nothing in PREREG_SHIFTBENCH.md (H1–H4, primary endpoints, rival specifications, truth file,
seeds) is changed by this addendum. The theory being tested is
[SHIFT_V2.md](../theory/SHIFT_V2.md) (branch `shift-v2`, `913f3e0`) and
[UNIQUENESS.md](../theory/UNIQUENESS.md).

## 0. Fixed conventions

* **Rivals**: OB, DFL, CFM, RIF, SHAP exactly as in SHIFTBENCH.md §4. No tuning.
* **TRACE configuration (fixed now)**: K = 4 Legendre channels (k = 0..3) plus the
  "unresolved" residual channel; cross-fitted period-odds classifier with 5 folds; weights
  trimmed at the 0.99 quantile; barycenter target for per-feature pieces, marginal-law target
  for channel-level pieces (SHIFT_V2 §2); order-free Shapley split (SHIFT_V2 §4–5). Bands: the
  harness's percentile bootstrap (rows resampled within period), plus TRACE's own uniform
  max-|t| band for H8(b). Any change to these is a deviation (§6).
* **Seeds**: the frozen `seed_of(*parts) = crc32("|".join(parts))` from
  `run_synthetic.py`. Data: `seed_of("data", scen, n, p, rep)`; method: `seed_of("m", rep)`;
  harness bootstrap: `seed_of("boot", scen, rep)` / `seed_of("mb", rep, b)`; permutations:
  `seed_of("perm", scen, n, p, rep)`; real-panel resamples: `seed_of("real", kind, k)`.
  New streams introduced here: `seed_of("half", scen, n, p, rep)` (synthetic split-half),
  `seed_of("sboot", scen, n, p, rep, b)` (synthetic stability bootstrap),
  `seed_of("placebo", k)` (real-panel placebo), `seed_of("null200", rep)` (extended null).
  Oracle truth for new scenarios: 2,000,000 draws per law, `seed = 99` (as `Scenario.truth`).
* **Scale**: every error is divided by `sd(y_A)` as in the frozen metrics. W2² quantities are
  divided by `sd(y_A)²`.
* **Alpha**: 0.05 two-sided per comparison. "Paired per-replication test" = for each
  replication r compute d_r = metric(TRACE) − metric(rival) on the same data and seed; report
  mean(d) with a 95% t-interval over replications (and a 95% percentile interval from 10,000
  resamples of the d_r, seed 2026; the claim needs both to exclude zero on the predicted side).
  H6, H7, H8 are the inferential family; the "supported" label uses Holm adjustment across
  their primary tests (listed below); unadjusted results are also reported. H5 is a
  deterministic exactness check, not a test.
* **Best rival** for a metric is chosen on the same runs as the rival with the best mean on that
  metric (favours the rivals; TRACE is not given a choice).

## 1. New planted scenarios (H7), fully specified

Added as a new module `benchmarks/shiftbench/sb_dgp_shape.py` that imports `sb_dgp` and does
not modify it. Every scenario starts from the frozen baseline (`p` features, active
coefficients (1.0, −0.8, 0.6, 0.4, −0.3), intercept 1.0, independent N(0,1) features, N(0,1)
noise, σ = 1), so **the mean of y is 1.0 in both periods** in every row below. New noise laws are
standardised (mean 0, variance 1): `skew` = the frozen `_skew_noise` (centred lognormal,
s = 0.6); `t5` = Student-t(5)·√(3/5); `skewneg` = −`_skew_noise`.

| scenario | period B change (mean of y unchanged) | channel(s) that move | planted composition / structure |
|---|---|---|---|
| `sd_struct` | noise σ 1.0 → 1.4 | k = 1 (scale) | structure only |
| `sd_comp` | x0 scaled ×1.6 about its mean 0 | k = 1 | composition only |
| `skew_struct` | noise N(0,1) → `skew` | k = 2 (skew) (+ small k ≥ 3) | structure only |
| `tail_struct` | noise N(0,1) → `t5` | k = 3 (kurtosis) | structure only |
| `shape_mixed` | x0 ×1.6 (composition) **and** noise → `skewneg` (structure) | k = 1 comp, k = 2 struct | both |
| `sd_offset` | x0 scaled ×1.3 (adds 0.69 to var y, composition) **and** noise σ 1.0 → √0.31 (removes 0.69, structure); var(y) = 3.25 in both periods | k = 1 both, total sd change 0 | offsetting |

Planted values (oracle, A-reference counterfactual Y_c = (X_B, structure A) **and** the
swapped counterfactual Y_c' = (X_A, structure B)): for each k = 0..3, L-moment
λ_{k+1} of Y_A, Y_B, Y_c, Y_c' (unbiased sample L-moments on 2·10⁶ draws), then the
order-free Shapley split comp_k, struct_k of SHIFT_V2 §4, and the W2² shares of §5
(a = (2k+1)·(comp_k, struct_k)). The frozen harness statistics (mean, sd, q10, q50, q90; A-reference
split) are also stored so rivals are scored on their own frozen estimands. Truth file:
`benchmarks/results/shiftbench_truth_shape.json`. Sizes: the frozen accuracy grid
(2,000, 5, 50), (20,000, 20, 20) and the frozen band design (n = 2,000, p = 5, 40 replications,
30 bootstrap resamples).

## 2. H5 — exact budget and order invariance

Run on all 11 frozen scenarios + 6 new scenarios, every replication of the accuracy grid.

**Metrics (per replication, TRACE):**
1. Barycenter budget gap `G_bar = |Σ_k Σ_{j∈{0..p}} Σ_{r∈{comp,struct}} φ_{kjr} − Σ_k (Δc̄_k)²/(2k+1)|`
   (j = 0 is the intercept player).
2. Marginal-law budget gap `G_mar = |Σ_k (φ_{k,comp} + φ_{k,struct}) + φ_{res,comp} + φ_{res,struct} − W2²(F̂_A, F̂_B)|`
   where W2² is computed directly as ∫(Q̂_B − Q̂_A)² on the empirical quantile functions.
3. Channel gap: for every k, `|φ_{k,comp} + φ_{k,struct} − (Δc_k)²/(2k+1)|` and coefficient
   gap `|comp_k + struct_k − Δλ_{k+1}|`.
4. Ordering invariance: run with the composition/structure roles evaluated in the opposite
   order (and with periods swapped A↔B, mapping comp_k → −comp_k etc.); largest absolute change
   in any φ.
5. Row/column permutation invariance: the frozen invariance design (INV_GRID, 5 reps, same seed)
   — largest change in any output after mapping features back.

**Pass bar:** items 1–4 ≤ 1e-9 · max(1, W2²) in **every** replication of every scenario
(relative to sd(y_A)²-scaled W2²); item 5 ≤ 1e-9 for the algebraic layer given identical fitted
inputs, and ≤ 1e-6 end to end.
**Contrast (reported, not tested):** for each rival and statistic, the additivity gap
|comp + struct − (θ(Y_B) − θ(Y_A))| and the order gap |split(A-reference) − split(B-reference, sign-flipped)|,
where B-reference reruns the frozen rival code with the period roles swapped; plus the frozen
permutation invariance numbers.
**Failure:** any single replication above a bar. A failure is recorded with scenario, size,
replication and gap, and stays on record even if later fixed.

## 3. H6 — stability of per-feature rankings

Attribution vector per method: mean-channel per-feature `|comp_j| + |struct_j|` (the frozen
metric 7 quantity). TRACE additionally reports its W2² per-feature vector
`Σ_k |φ_{kj,comp} + φ_{kj,struct}|` as a secondary (no rival equivalent). Rivals eligible:
those with per-feature output (OB, RIF, SHAP).

**Real panel** (frozen resamples, `run_real.py`):
* Bootstrap stability: for b = 1..100, s_b = Spearman(attribution on resample b, full-sample attribution).
* Split-half: for k = 1..50, h_k = Spearman(half 1, half 2).
* Paired test: d_b = s_b(TRACE) − s_b(best rival), d_k likewise.

**Synthetic** (new, frozen scenarios, n = 20,000, p = 20, all 10 non-null scenarios, 20 reps each):
per replication, split-half (`seed_of("half", …)`, rows of each period split at random into
halves) and 20 bootstrap resamples (`seed_of("sboot", …)`); per-replication stability =
Spearman(half 1, half 2) and mean Spearman(bootstrap, full). Paired over the 200
scenario × replication units.

**Primary tests (4):** {real, synthetic} × {bootstrap, split-half}.
**Pass bar:** mean paired difference ≥ +0.05 **and** 95% interval (both t and percentile)
strictly above 0, in all four. **Partial:** 95% interval above 0 but mean < 0.05, or passes in
only some of the four (reported as such, not as support).
**Failure:** any of the four intervals strictly below 0 (TRACE less stable than the best rival)
is a recorded failure; an interval straddling 0 is "not supported".

## 4. H7 — shape shifts at fixed mean

On the 6 new scenarios.
**Primary metric (TRACE):** channel split error `e_k = (|comp_k^hat − comp_k| + |struct_k^hat − struct_k|)/2 / sd(y_A)`
on the L-moment scale for every planted channel k ∈ {1,2,3}, and the structure fraction
`f_k = struct_k / (comp_k + struct_k)` (not defined for `sd_offset`; there the sign pattern is used).
**Recovery** in a replication: for every channel the scenario plants (table §1), e_k ≤ 0.05 and,
for scenarios other than `sd_offset`, |f̂_k − f_k| ≤ 0.10; for `sd_offset`, sign(comp̂_1) =
sign(comp_1), sign(struct̂_1) = sign(struct_1). Plus the mean channel: |comp̂_0| + |struct̂_0| ≤ 0.05·sd(y_A)
(no spurious location shift).
**Pass bar:** recovery rate ≥ 0.80 in each scenario at (20,000, 20), and ≥ 0.60 at (2,000, 5);
95% band coverage of the planted comp_k and struct_k ≥ 0.90 (band design).
**Rivals:** DFL, CFM, RIF errors on sd, q10, q50, q90 against their frozen A-reference truth
(and spurious mean-channel mass for all five rivals) are reported per scenario. Paired test
(primary, one per scenario, 6 tests): d_r = e^{TRACE}_{sd-equivalent} − e^{best rival}_{sd}, where
TRACE's sd-split is read from its "sd" output in the frozen `split` contract. Supported if
the 95% interval is below 0 in ≥ 4 of 6 scenarios.
**Failure:** recovery rate below bar in any scenario, or coverage below 0.90, recorded per
scenario.

## 5. H8 — no-shift control

* **(a) Null false-positive rate.** Frozen band design on `null` (40 reps), plus an extended
  null: 200 replications, n = 2,000, p = 5, B = 100 bootstrap resamples, data seed
  `seed_of("null200", rep)`, all methods. FPR = share of replications in which any composition or
  structure band (mean channel; separately: any of k = 0..3 using TRACE's uniform band; and
  per-feature) excludes 0.
  **Bar:** TRACE mean-channel FPR ≤ 0.05 point estimate on the 200-rep null **and**
  Clopper–Pearson 95% upper bound ≤ 0.09; uniform-band FPR over k = 0..3 ≤ 0.05; per-feature
  FPR (any of p features) ≤ 0.10. Rivals' FPRs on the same replications reported; paired McNemar
  test vs each rival on the mean-channel indicator.
* **(b) Coverage.** Frozen band design scenarios (`mixed`, `tail`, `hetero`, `entry_exit`) and the
  6 new ones: share of replications whose 95% band contains the planted comp/struct.
  **Bar:** pooled coverage ≥ 0.90, every scenario × channel ≥ 0.85.
* **(c) Real-panel placebo.** 100 placebo splits: within period A only, units randomly assigned
  to pseudo-periods (`seed_of("placebo", k)`, half/half). FPR of mean-channel bands
  (B = 100). **Bar:** ≤ 0.10 (units are not independent over quarters; reported, secondary).
**Primary tests:** (a) TRACE FPR bar; (b) pooled coverage bar.
**Failure:** any bar missed.

## 6. Reporting and deviations

All four hypotheses are reported in full whether supported or not, with every failure listed
(scenario, size, replication, value). Results go to `docs/benchmark/TRACE_CLAIMS_RESULTS.md`;
raw records to `benchmarks/results/trace_claims_*.jsonl`. Any change after this commit (a bug
fix in the new scenarios, a TRACE configuration change, a different bootstrap count) is listed
here with commit and reason, and the result under the frozen specification is still reported
when it can be computed.
