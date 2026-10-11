# ADOPT audit

Status: research, unvalidated. This note separates three kinds of finding:

* **Bug in our code.** Something our reproduction did that the paper's procedure, as we read it, does not say to do.
* **Ambiguity in the paper.** A choice the text does not fix. We did not retune the simulation to chase the published numbers.
* **Flaw in the paper's method.** A property of Algorithm 1 itself, with a short derivation or counterexample and a numerical check under `benchmarks/adopt_audit/`.

The paper is Song and Müller, "ADOPT: Additive Optimal Transport Regression", AISTATS 2026, PMLR 300:1558–1566, https://proceedings.mlr.press/v300/song26c.html. An arXiv preprint (2512.08118) was withdrawn on 15 December 2025. The withdrawal comment is: "Not all co-authors had agreed in advance to the public release." That is a consent issue, not a stated technical correction. The conference paper is the version this repository reproduces.

Paper numbers below are copied from `benchmarks/adopt_paper/paper_values.json`, which was transcribed from the PMLR PDF. They are not re-estimated.

Optional controls live on `ADOPTRegressor`. The defaults are Algorithm 1: `composition="given"`, `centering="frechet"`, `damping=1`, `certificate=False`, `project_prediction=False`, `bins=None`, `boundary="none"`. The 200-replicate tables use those defaults.

## 1. Bug in our code

### Bandwidth cross-validation used the held-out fold's covariate range

`select_bandwidth_cv` fits on each training fold and scores the held-out fold. The bandwidth is `h_j = c * range(X_j)`. The old code computed that range on the **whole sample**, so the held-out fold's covariate values set the scale of `h` even though their responses were not in the fit. The final fit, after `c` is chosen, still uses the range of the full training sample, which is what we want.

The fix computes the range on the training rows of the fold only. `tests/python/test_audit.py::test_cv_bandwidth_uses_training_range_only` builds a sample where one fold sits far to the right and checks that the Rust score matches a hand-rolled training-range loop and not the full-sample loop.

This bug cannot explain the table gaps. A uniform margin on `[0, 1]` barely changes its range when one fifth of the rows is removed, and the MISE grid was never inside the CV. The before/after tables are in Section 4.

### Checks that are not bugs, and the tests that keep them that way

| Check | Result | Test |
|---|---|---|
| Training rows vs MISE grid | Disjoint. The grid is the fixed midpoint lattice `(k+0.5)/10` on `[0,1]^3`. `reproduce.py` now aborts a replicate whose Chebyshev gap is 0, and records the minimum gap in the JSON meta. | `test_mise_grid_is_disjoint_midpoint_cube` |
| CV folds | A partition of `{0,…,n-1}`, each fold sorted, no shared index | `test_kfold_is_a_partition` |
| Noise | W2 responses use `f + ξ sin(2π f)/(2π)`, `ξ ~ Unif(-1,1)`. SPD noise `N(0, 0.01²)` hits the strict lower Cholesky block only; the diagonal is unchanged. The MISE target is the noise-free regression function, as in eq. (12). | `test_responses_carry_the_stated_noise` |
| Integral | Table 1: mean over the cube of the mean squared quantile error. Table 2: mean over the cube of the **sum** of squared log-Cholesky coordinates. A sum over quantile levels, or a mean over SPD coordinates, would change the scale. | `test_mise_quadrature_matches_the_paper_integral` |
| Seeds | Data: `SeedSequence([table, case, n, m, rep])` with table ∈ {1, 2}. CV fold seed: the replicate index only (ambiguity A5). The two generators are not the same stream. | `test_seeds_are_separated` |

## 2. Ambiguity in the paper

These are unchanged by the audit. They are copied here so the table gaps have a place to sit that is not "our code leaked".

* **A5.** Kernel, candidate grid, whether each predictor has its own fraction, and the CV loss are not stated. Our Gaussian kernel and the 10 log-spaced fractions in `[0.03, 0.6]` are a choice. Table 1 Case II is an additive location model (the component maps are translations, and they commute). FM matches the paper (204.7 vs 207.2 at n = 100, ×10⁻³), so the signal scale is right, and GF matches as well (6.36 vs 6.64). ADOPT and our L2-additive comparator are both 3–6× below the paper. A flexible additive fit of `x`, `x²` and `exp(-x)` at these sample sizes can be that accurate. The paper's FM standard errors are several times ours, which suggests their responses vary more across replicates than the formula we implemented (A10). We did not add noise the paper does not write down.
* **A15.** Table 2 Case I, m = 10: FM matches (134.3 vs 135.3, ×10⁻²), so the marginal signal matches, while GF is about 10× too low and ADOPT about 47× too low. Off-diagonal Cholesky entries are linear in `x`; only the log diagonal is nonlinear. With `σ = 0.01` a linear fit's error is tiny. Case I m = 20 and Case II do not match even for FM, so the factor design or the distance normalisation is still unresolved. The existing `σ` diagnostic in `docs/REPRODUCTION_NOTES.md` shows that a larger noise level does not fix GF and FM together. It is not a substitute for the table.

## 3. Flaws in the paper's method

Each experiment is `benchmarks/adopt_audit/run_audit.py`. Numbers below are from the release-build run in `benchmarks/results/adopt_audit.json`.

### (a) Order dependence — flaw, fix `composition="average"` or `"tangent"`

Composition of monotone maps on `R` is not commutative. At `y = 0`, `f(y) = y²` and `g(y) = (y+1)/2` give `f(g(0)) = 0.25` and `g(f(0)) = 0.5`.

ADOPT's `⊕` applies maps in feature order (Algorithm 1). Swapping the columns and swapping them back at prediction time changes the fit whenever the component maps do not commute. On a two-feature power-then-fractional-linear truth (n = 70, q = 24, fixed bandwidth 0.28), the mean W2² between the two orders' predictions on a 6×6 grid is `2.85×10⁻⁴` for `composition="given"`, `0` for `"average"` (the two orders are the same set of permutations), and `2.0×10⁻¹⁶` for `"tangent"`.

On a commuting location truth the three holdout W2² values agree to at least six digits (`8.84×10⁻⁴`). Averaging does cost something when the truth really has one order: holdout W2² rises from `4.51×10⁻⁵` (given) to `1.24×10⁻⁴` (average) and `1.55×10⁻⁴` (tangent). That is the bias of pretending the composition is order-free. The option is off in the reproduction, where the paper's order is part of the design (note A14).

`composition="average"` refits every permutation (at most 5 features; `p!` fits) and averages the composed predictions. It does not compose the averaged maps. `composition="tangent"` is additive backfitting of the quantile vectors, then isotonic projection of the sum.

### (b) Identifiability and centering — the barycentric identity already holds; the split of a shared effect does not

**Derivation.** The piecewise-linear map `T_{M, g}` is affine in the target knots `g`, including the translation used outside the knot range. If `M = mean_i g_i`, then `mean_i T_{M, g_i}(z) = z` for every `z`. Equation (11), which sets the anchor to the sample mean of `g_j(X_j)`, therefore already makes the average pushforward the identity. A numerical check on 18 random increasing curves and a random test vector gives a max absolute gap of `6.7×10⁻¹⁶`.

So "the components are not centered" is not a flaw of eq. (11). What remains is the usual additive non-identifiability. If both columns are the same covariate and the truth is a location shift, Gauss–Seidel in the given order puts the whole effect on the first column: mean squared gap between the two component curves is `0.201`. `centering="pushforward"` does not move that gap (`0.201`). `composition="average"` splits it evenly (gap `0`) and the holdout W2² stays at numerical zero for all three (`4.5×10⁻³¹` given and average, `1.5×10⁻³⁰` pushforward). Pushforward centering is the option that recenters pushed quantiles onto the global barycenter and projects; on this problem it is redundant, and it does not hurt the prediction.

### (c) Backfitting convergence — flaw, fix `certificate=True` (damping optional)

Nothing in Algorithm 1 makes the in-sample loss descend. The update is a nonlinear Gauss–Seidel step (smooth, project, compose). On correlated columns, a sine-plus-square location signal, n = 40, q = 12 and bandwidth 0.05, the in-sample W2² after iterations 1…8 is

```
0.000950, 0.002216, 0.003365, 0.004269, 0.004937, 0.005415, 0.005753, 0.005992
```

Every increment is positive. With `certificate=True` the fitter rejects a cycle whose in-sample mean W2² exceeds the last accepted value by more than `1e-12`, halves the step, and stops when the step falls below `1e-8`. Here it rejected 27 proposals, kept iteration 1, and finished at `0.000950`, which is the best of the undamped path.

The stopping threshold matters for the iteration count and almost not at all for holdout, once it is small. On a smooth location problem, `tol` in `{1e-2, 1e-4, 1e-8, 1e-12}` took 3, 4, 6 and 9 iterations. Holdout W2² went from `3.20×10⁻⁴` to `3.41×10⁻⁴` and then changed by less than 0.1% between `1e-8` and `1e-12`. Looser stopping was slightly **better** out of sample on this noise-free problem; we do not claim a tighter threshold is more accurate. Damping `0.5` together with the certificate raised that holdout from `3.41×10⁻⁴` to `3.49×10⁻⁴` (about 2%).

### (d) Off-grid maps and quantile crossing — projection fixes the tangent sum; the paper path does not cross

**Extension.** Outside the knots, `T` is a translation by the boundary displacement (`docs/REPRODUCTION_NOTES.md`, A6). A squared map on the midpoint grid, evaluated one unit below the first knot and one unit above the last, matches that formula, and the two images stay ordered. An increasing knot map extended this way stays increasing. The off-grid error that remains is interpolation of the composed residual (A7), which shrinks as `q` grows. It is not a crossing.

**Crossing.** A sum of centered increasing curves need not be increasing. `u + 0.2 sin(4π u)` decreases on 12 of 29 consecutive pairs; PAVA removes every decrease, and the L2 distance to the increasing grid `u` does not rise (the projection property). A tangent-space fit of a wiggly-but-projected response produced 60 decreases in the raw additive prediction and 0 after the final PAVA that `composition="tangent"` always applies. The paper path (`composition="given"`, each component already projected) had 0 decreases on the same inputs, and turning on `project_prediction` left those predictions unchanged. Negative local-linear weights can make a weighted mean of increasing curves decrease; PAVA on each component is what stops that from surviving into the paper's composition.

### (e) Scaling — flaw of the dense algorithm, fix `bins`

Supplement C of the paper gives `O(n² p)` (and `O(n² p q)` with a quantile grid) per cycle, because each of `n` residuals is smoothed at `n` sites. Each row has to be a whole distribution: the response is a quantile vector or an SPD matrix, not a scalar.

`bins=B` replaces the `n×n` weight matrix by `B` equal-width bin means and a local-linear smooth on the occupied centres. Cost is linear in `n` plus a `B×B` smooth. On the release build (4 CPUs, x86_64, Python 3.12), p = 2, q = 12, 3 iterations:

| n | dense (s) | bins=32 (s) | dense holdout | binned holdout |
|---:|---:|---:|---:|---:|
| 200 | 0.00288 | 0.00146 | 2.85×10⁻⁴ | 2.70×10⁻⁴ |
| 400 | 0.00883 | 0.00213 | 2.61×10⁻⁴ | 2.68×10⁻⁴ |
| 800 | 0.0257 | 0.00363 | 2.70×10⁻⁴ | 2.76×10⁻⁴ |
| 1600 | 0.0987 | 0.00739 | 2.67×10⁻⁴ | 2.69×10⁻⁴ |

Eight times as many rows multiplied dense time by 34 and binned time by 5.1. From 800 to 1600 the ratios were 3.83 (dense; quadratic is 4) and 2.03 (binned; linear is 2). Holdout W2² stayed within about 5% of the dense fit. `bins=None` remains the default used in the paper tables.

### (f) Boundary bias — local linear is the fix; reflection is not

Nadaraya–Watson has an `O(h)` bias at the edge of the sample because the kernel is cut off. Local linear cancels that term: the bias returns to the interior order `O(h²)`. On `exp(3x)`, n = 80, h = 0.18, 16 replicates, edge means (`x≤0.08` or `x≥0.92`) and interior means (`0.3≤x≤0.7`):

| smoother | edge MSE | interior MSE |
|---|---:|---:|
| Nadaraya–Watson | 12.15 | 0.459 |
| local linear (the paper's smoother) | 0.392 | 0.470 |
| even reflection | 11.47 | 0.514 |

Local linear cuts the edge error by about 30× and does not raise the interior error. Even reflection (`boundary="reflect"`) folds ghost points `2 lo − x` and `2 hi − x` back onto the parent, with the parent's response. That is the right correction for a flat derivative. It is the wrong one for a line: on `y = 2x+1` the local-linear error at `{0, 0.5, 1}` is `6.7×10⁻¹⁶`, and the reflected error is `0.41`. On `exp(3x)` reflection is about as bad as Nadaraya–Watson at the edge. The option stays available and off. The reproduction keeps local linear.

## 4. Tables before and after the CV-range fix

"Before" is `benchmarks/results/adopt_reproduction_before_audit.json` (B = 200, the run reported in the previous `docs/ADOPT_RESULTS.md`). "After" is the re-run with training-fold ranges only. Both use paper defaults for every new option. Entries are mean (standard error), Table 1 ×10⁻³, Table 2 ×10⁻². Paper columns are verbatim.

### Before (full-sample CV ranges)

| setting | paper ADOPT | ours ADOPT | paper GF | ours GF | paper FM | ours FM | ours L2-additive |
|---|---|---|---|---|---|---|---|
| I/100 | 0.317 (0.010) | 0.391 (0.014) | 0.591 (0.008) | 0.740 (0.011) | 11.019 (0.068) | 11.691 (0.010) | 0.640 (0.012) |
| I/300 | 0.181 (0.004) | 0.179 (0.005) | 0.499 (0.003) | 0.576 (0.005) | 11.119 (0.040) | 11.602 (0.003) | 0.436 (0.006) |
| II/100 | 1.685 (0.057) | 0.503 (0.010) | 6.636 (0.057) | 6.362 (0.022) | 207.205 (1.681) | 204.700 (0.219) | 0.579 (0.014) |
| II/300 | 1.121 (0.041) | 0.178 (0.003) | 6.701 (0.037) | 5.975 (0.008) | 211.297 (1.080) | 203.277 (0.064) | 0.194 (0.004) |

| setting | paper ADOPT | ours ADOPT | paper GF | ours GF | paper FM | ours FM |
|---|---|---|---|---|---|---|
| I/10/100 | 2.769 (0.025) | 0.059 (0.000) | 9.495 (0.060) | 0.950 (0.003) | 135.344 (0.647) | 134.279 (0.231) |
| I/10/300 | 1.655 (0.012) | 0.021 (0.000) | 9.714 (0.037) | 0.895 (0.001) | 136.122 (0.366) | 132.593 (0.066) |
| I/20/100 | 5.712 (0.070) | 0.220 (0.001) | 14.110 (0.102) | 1.943 (0.005) | 192.688 (0.92) | 272.178 (0.516) |
| I/20/300 | 3.788 (0.048) | 0.079 (0.000) | 14.254 (0.063) | 1.806 (0.002) | 193.796 (0.521) | 268.959 (0.191) |
| II/10/100 | 4.146 (0.088) | 0.086 (0.001) | 18.507 (0.090) | 4.675 (0.013) | 96.562 (0.490) | 206.037 (0.112) |
| II/10/300 | 2.702 (0.036) | 0.031 (0.000) | 18.808 (0.052) | 4.481 (0.004) | 97.047 (0.303) | 205.286 (0.033) |
| II/20/100 | 8.921 (0.167) | 0.340 (0.002) | 28.237 (0.14) | 10.124 (0.029) | 143.382 (0.706) | 434.257 (0.264) |
| II/20/300 | 6.435 (0.100) | 0.115 (0.000) | 28.512 (0.078) | 9.723 (0.008) | 144.088 (0.437) | 432.502 (0.092) |

### After (training-fold CV ranges)

<!-- AFTER_TABLES -->

The gap with the paper is still open in the same two places: Table 1 Case II ADOPT, and Table 2 ADOPT and GF. See Section 2.

## 5. Still unresolved

* Table 1 Case II. FM and GF match; ADOPT and L2-additive do not. Leakage does not explain it. The remaining candidates are the bandwidth recipe (A5) and response noise the text does not state (A10). We did not invent either.
* Table 2. FM matches only Case I, m = 10. The factor design or the distance normalisation for larger m, and the size of the GF/ADOPT errors at `σ = 0.01`, are still ambiguities (A15).
* A fixed point of transport backfitting is not guaranteed. The certificate stops the in-sample loss from rising; it does not prove the stopped fit is a global minimiser, and on the smooth location problem a looser threshold was slightly better out of sample.
* `composition="average"` removes order dependence and, on a non-commuting truth, raises the error. It is not a free replacement for the paper's order.
* Even reflection is not a boundary fix for local linear. It remains an option because some targets are flat at the edge; the experiments above say not to turn it on for a curved component.
