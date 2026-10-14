# FIXES2: diagnosis and fixes for the shift-v2 pre-registration failures

Status: plan and bars frozen **before** the fixed code is run on the validation seeds 6000–6019.
The commit that adds this file is the freeze point; the runner records its sha. Bars below are not
edited after that commit. Failed bars stay failed. `PREREG_SHIFT_v2.md`, its runner and its results
are not changed.

All diagnostics below used fresh seeds (5000–5019, 5100, 5200–5599). None used 1000–1019 data
other than the already-public per-seed lines of `verify/PREREG_SHIFT_v2_RESULTS.txt`.
Scripts: small throwaway simulations with the same data-generating process as the v2 runner.

## 1. Failures, causes, evidence

### B1 (S1 structure share 14/20) and B2 (S2 composition share 17/20)
**Cause 1: the Monte Carlo truth is noisy where the true value is exactly zero.** In S1 the
structure is identical in both periods, so `C =d A` and `C' =d B`: every structure term is exactly 0.
In S2 the covariate laws are identical, so `C =d B`, `C' =d A`: every composition term is exactly
0. The v2 truth drew A, B, C, C' independently (2·10⁶ each), giving S1 truth −0.0097 and S2 truth
−0.0032. For S2 the estimator's sampling sd is ≈ 0.006, so the truth was offset by about half a
standard deviation, enough to turn a 95% band into a materially lower hit rate.

**Cause 2: weight capping biases the counterfactual toward the unweighted sample.** Capping ψ at
its 0.995 quantile (≈ 0.8% of weight mass in S1/S3) moves `F_C` toward `F_B`, so part of the
composition change is booked as structure. On seeds 5000–5019 (S1): mean structure share
0.0090 with trim 0.995 vs −0.0106 untrimmed (sd across seeds 0.10). In S3 the trim moves comp
k=1 W2² by −1.4·10⁻³ (≈ 0.5 sd); untrimmed it is unbiased (0.0·10⁻³). In S2 the trimmed mass is
≈ 7·10⁻⁵, so trimming is not what failed B2.

**Ruled out.** The marginal bootstrap does re-run the cross-fitted period classifier on every
replicate (`bootstrap_marginal` → `marginal_view` with a new fold seed), so "bootstrap does not
resample the reweighting step" is not a cause. Percentile and basic (reverse-percentile) bands
both covered in 19/20 S1 seeds 5000–5019, for both trims. The v2 S1 seeds had a mean share of 0.059
(≈ 2.3 SE above the trimmed expectation); 14/20 is consistent with the bias above plus chance.

### B3 (S3 channel pieces, min 16/20)
Same two causes, smaller. Untrimmed biases on seeds 5000–5019 are ≤ 0.15 sd except struct k=1
(−0.004, ≈ 0.45 sd, both trims). That residual is of the size of the independent-draw MC error in
the v2 truth for the largest term, which common random numbers remove. The remaining failures
are expected to be coverage noise; no estimator change beyond the trim.

### B4 (S3 per-feature composition shares, min 13/20) and B9 (entry/exit, min 0/20)
**Cause: the TRACE fit's shape channels estimate a different target.** The fit minimizes integrated
pinball loss. Its first-order condition is `∫ (F(Q(u|x)) − u) e_k(u) du = 0`, i.e. to first order
`∫ f(Q*(u|x)) (Q − Q*) e_k du = 0`: a *density-weighted* projection of the conditional quantile.
The barycenter truth (and the definition in SHIFT_V2 §2) is the unweighted L²[0,1] Legendre
projection. The weighting discounts the tails, which carry the higher L-moments, so channels
k ≥ 1 are shrunk. Evidence, no-covariate-signal fit (seed 5100, skew-normal ε, 8 noise features),
fitted/true `c_k`, k = 1..5:

| n | fitted / true (pinball fit) | sample L-moments / true |
|---|---|---|
| 5,000 | 0.95 0.84 0.76 0.50 0.41 | 1.00 1.01 1.08 1.20 1.06 |
| 20,000 | 0.97 0.89 0.85 0.66 0.52 | 0.99 1.00 1.01 1.01 1.00 |
| 80,000 | 0.99 0.94 0.86 0.60 0.56 | 1.00 1.02 1.00 0.96 1.00 |

The shrinkage does not vanish with n (it is a target difference, not noise), so the bootstrap,
which refits TRACE and therefore reproduces the same shrinkage, centres its band on the wrong
value. S5 seed 5000: entry k=3..5 fitted 0.124 0.032 0.049 vs truth 0.151 0.044 0.084. For B4 the
channel-0 pieces are right (feature 1 comp α 0.3558 vs truth 0.3564); the deficit (feature 1 ratio
≈ 0.93 on seeds 5000–5003, independent of rounds 12 vs 40) comes from the scale/shape channels.
Replacing the Newton score by a density-corrected (Bahadur) score was tried and only partly
helps (k=5 ratio 0.50 → 0.62), because the linearization fails in the tails.

### B5 (S4 permutation floor, 3/20 false positives, bar ≤ 1/20)
**Not a calibration defect.** On 400 fresh null datasets (seeds 5200–5599, R = 200) the
false-positive rate is 21/400 = 5.25% (SE 1.1%). For an exactly calibrated 5% test,
P(≥ 3 of 20) = 7.5% and the bar "≤ 1/20" fails with probability 26%. The v2 failure is
consistent with chance. No estimator change; the bar is kept and a calibration bar is added (B5b).

### Speed (explain 184 s at 1e6 × 100 vs barycenter 3.9 s)
**Cause:** `explain` = barycenter + marginal view; the marginal view's cross-fitted logistic
regression is serial, cold-started on each of 5 folds, and each Newton step costs O(n p²)
(2·10⁶ × 100² per step). Barycenter cost is small. Profile at 2·10⁵ × 100, 2 threads:
explain 51.3 s, of which the marginal view is essentially all.

## 2. Fixes (Rust core)

**F1. PIT pseudo-outcome refit (`V1Config::pit_rounds`, opt-in, default 0).** For a continuous
conditional law, `U = F(Y|x)` is Uniform(0,1) and independent of `x`, and `Y = Q(U|x)`. Hence
`c_k(x) = (2k+1) ∫ Q(u|x) P*_k(u) du = (2k+1) E[Q(U|x) P*_k(U) | x] = E[(2k+1) Y P*_k(U) | x]`.
The pseudo-outcome `(2k+1) Y P*_k(U)` therefore has conditional mean exactly the L²-Legendre
coefficient, and an ordinary least-squares additive backfit of it (same bins, effects centred
under the training law) estimates the additive projection of `c_k(x)`, which is the B4 truth's
definition. `U` is estimated by the within-sample mid-rank of `(Y − ĉ₀(x)) / ĉ₁(x)` from the
pinball fit; this is exact under a location-scale conditional law (S1–S5) and its error enters
only through `E[Y P*_k'(U)(Û − U)]`. Channel 0 is the plain least-squares mean. Prototype on S5
seeds 5000–5002: entry/true k=0..3 within 0.91–1.06 (was 0.82 at k=3); k=4,5 unbiased but noisy.
S3 feature-1 ratio 0.98 (was 0.93). The bootstrap refits with the same config, so its band now
contains the fit's variance around the right centre.

**F2. Trim.** No capping (`trim_quantile = 1.0`) in the validation; the trimmed mass is still
reported. Justification: the cap's bias is first order in the trimmed mass, while with
cross-fitting and self-normalization the uncapped estimator only needs `E_B[ψ²] < ∞`, which
holds for these Gaussian mean shifts. (Library default unchanged; callers choose.)

**F3. Truth by common random numbers** (validation harness only). Couple A with C through
`(x_A, ε₁)` and B with C' through `(x_B, ε₂)`; reuse `x_A` for `x_B` when the covariate laws are
equal. Each law is still sampled exactly, so every law functional is unchanged; MC noise in
differences is removed, and S1-structure / S2-composition truths become exactly 0.

**F4. Speed.** Logistic gradient/Hessian accumulated in parallel over fixed 4096-row chunks
(summed in chunk order: deterministic for any thread count), lower triangle only, and each fold
warm-started from the previous fold's solution. Same estimating equation, same stopping rule
(step norm < 1e-8). At 2·10⁵ × 100, 2 threads: explain 51.3 s → 10.9 s, identical reported
W2², composition and order gap.

**Invariants kept.** All identities (Prop. 1, 3, 4, §7) are algebraic on any additive model and
on any weights, so they hold unchanged: period swap negates, players sum to the barycenter W2² and
entry/exit pieces sum to Δc̄ at ≤ 1e-9 (B7, B8 re-checked on every validation seed).

## 3. Validation (pre-registered)

Runner `docs/theory/verify/prereg_fixes2.py`, a copy of the v2 runner with only these changes:
seeds **6000–6019**; same scenarios S1–S5, n = 20,000, p = 8, K = 6, 5-fold cross-fit, R = 200,
B = 200 for the marginal bootstrap; **B_refit = 100** for the bootstraps that refit TRACE
(compute cap: 2 cores on a shared machine; percentile ends interpolated); TRACE fit = v2 FIT plus
`pit_rounds = 20`; trim 1.0; MC truth with common random numbers and new truth seeds
(7007–7012). Output `verify/FIXES2_RESULTS.txt`.

| Bar | Statistic | Pass criterion (same as v2 unless noted) |
|-----|-----------|----------------|
| B1 | S1 structure share | truth (0) inside band ≥ 18/20 |
| B2 | S2 composition share | truth (0) inside band ≥ 18/20 |
| B3 | S3 comp and struct W2², k = 0..2 | each component ≥ 18/20 |
| B4 | S3 per-feature comp share, features 1–3 | each ≥ 18/20 |
| B5 | S4 total W2² above permutation 95% floor | ≤ 1/20 (unchanged; 26% fail chance even if exact) |
| B5b | **new**: false-positive rate over 400 null datasets, seeds 6100–6499 | Wilson 95% CI contains 0.05 and rate ≤ 0.075 |
| B6 | S4 any player resolved (uniform band) | ≤ 1/20 |
| B7 | order invariance | ≤ 1e-9 every seed |
| B8 | player sums | ≤ 1e-9 every seed |
| B9 | S5 entry, exit (truth) and stay (0), k = 0..5 | each ≥ 18/20 |
| B10 | `verify/shift_v2_checks.py` | all PASS |
| B11 | speed, 1e6 × 100, `explain`, 2 threads | target ≤ 60 s; reported, not a correctness gate |

Expectations stated before running: B1, B2, B6–B10 pass; B3 and B4 likely improve but may still
miss a component (no fix aimed at coverage of quadratic terms); B9 k = 4, 5 is the weakest point
(pseudo-outcomes are noisy and the additive model is evaluated on groups cut inside bins).

## Deviation log

* **2026-10-14: validation stopped early at user request** (logged before scoring). The frozen
  runner was stopped to use the progress so far. Completed seeds (from the runner's per-seed cache):
  S1 20/20, S2 20/20, S3 8/20 (6000–6007); S4, S5 and the B5b calibration (seeds 6100–6499) not run.
  Scoring: the frozen bar logic of `prereg_fixes2.py` at `cc699b2` applied to the completed seeds by
  `verify/score_fixes2_partial.py` (no bar edited). Each bar is reported as k/n completed. A bar is
  PASS or FAIL only when its outcome is determined by the completed seeds whatever the missing seeds
  would show (e.g. more than 2 misses already, for an 18/20 bar); otherwise INCOMPLETE. B7/B8 are
  reported on the completed seeds of every scenario run. B10 is run as specified.
* **2026-10-14: B11 speed thread count.** The plan specifies 2 threads (shared machine). The machine is
  now free and the run uses all 8 cores (`RAYON_NUM_THREADS=8`); the thread count is recorded with
  the result. B11 is not a correctness gate.
