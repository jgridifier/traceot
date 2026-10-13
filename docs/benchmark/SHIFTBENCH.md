# ShiftBench: a benchmark for explaining distribution shift

Status: design, frozen by [PREREG_SHIFTBENCH.md](PREREG_SHIFTBENCH.md). Harness:
`benchmarks/shiftbench/`. Results: [SHIFTBENCH_RESULTS.md](SHIFTBENCH_RESULTS.md).

## 1. The question every method answers

An outcome `y` and features `X` are observed in two periods, A (reference) and B. A
statistic of the outcome distribution moved, for example its mean or its 90th
percentile. How much of the move is **composition** (the feature mix changed) and how
much is **structure** (the map from features to outcome changed)? And which features
carry each part?

The planted target is the standard counterfactual-distribution split. Let `Y_c` be the
outcome you would see with the features of period B and every other part of period A
(conditional mean, noise scale, noise law). For a statistic `theta`:

    composition = theta(Y_c) - theta(Y_A)
    structure   = theta(Y_B) - theta(Y_c)

Statistics (the "channels"): mean and median (location), standard deviation (scale),
10th and 90th percentiles (tails).

Per feature, truth exists for the mean because every scenario's conditional mean is
additive, `mu_t(x) = a_t + sum_j g_tj(x_j)`:

    composition_j = E_B[g_Aj] - E_A[g_Aj]
    structure_j   = E_B[g_Bj - g_Aj] - E_P[g_Bj - g_Aj]
    intercept     = mean structure - sum_j structure_j

`P` is the 50/50 pooled feature law. The centring removes the usual identification
problem of detailed decompositions (a slope change moves the mean only relative to a
reference point). The harness centres every method's features at the pooled sample mean,
so every method is scored against the same convention.

## 2. Synthetic suite

Five active features with coefficients (1.0, -0.8, 0.6, 0.4, -0.3); every other feature
is inert (zero effect, same law in both periods). Default features are independent
standard normals, default noise is standard normal. Eleven scenarios:

| scenario | what is planted | channel that should move |
|---|---|---|
| `null` | A and B identical | none |
| `loc_comp` | feature means shift (x0 +0.5, x1 -0.3, x2 +0.2) | location, composition |
| `loc_struct` | slopes change (x0 +0.3, x2 -0.2), intercept +0.1 | location, structure |
| `mixed` | x0, x3 shift; x1, x3 slopes change; intercept -0.1 | both |
| `scale` | x0 spread x1.5 (composition); noise sd x1.3 (structure) | scale |
| `tail` | x1 becomes skewed with the same mean and variance; noise becomes skewed with the same mean and variance | tails only |
| `correlated` | AR(0.8) correlation among active features; only x0 is translated; x1 slope changes | composition on x0 alone |
| `offsetting` | x0 shift +0.4 (composition) cancelled by intercept -0.4 (structure) | both, total near 0 |
| `hetero` | noise sd = exp(0.4 x0) in both periods; x0 shifts | scale and tails via composition |
| `nonlinear` | g0 = x0^2, g1 = sin(2 x1); x0, x1 shift; x2 slope changes | linear methods misspecified |
| `entry_exit` | 20% of period-B units are entrants from a shifted law (x0, x2 +0.8); x1 slope changes | mixture composition |

Planted values come from an oracle Monte Carlo with 2,000,000 draws per law (fixed seed),
stored in `benchmarks/results/shiftbench_truth.json`.

Sizes. Accuracy grid (n rows per period, p features, replications): (2,000, 5, 50),
(20,000, 20, 20), (200,000, 20, 5). Band grid: n = 2,000, p = 5, scenarios `null`,
`mixed`, `tail`, `hetero`, `entry_exit`, 40 replications, 30 bootstrap resamples each.
Scale grid: n in {10^4, 10^5, 10^6} x p in {10, 100}, scenario `mixed`, one run per cell,
4 threads, 1,200 s and 8 GiB caps per cell.

## 3. Metrics

All errors are divided by the standard deviation of `y` in period A.

1. **Channel split error.** For each statistic, `(|comp_hat - comp| + |struct_hat - struct|) / 2`,
   averaged over replications, then over the 11 scenarios. Reported per scenario with a
   95% interval over replications.
2. **Per-feature error** (mean channel). `sum_j |comp_hat_j - comp_j| + |struct_hat_j - struct_j|`.
   Also the **inert mass**, the same sum over inert features only (false attribution).
3. **Band coverage.** Share of replications whose 95% percentile bootstrap band (rows
   resampled within each period, centring redone) contains the planted composition or
   structure value. Also mean band width.
4. **Null false-positive rate.** On `null`, the share of replications where the
   composition band or the structure band excludes 0; and the per-feature version.
5. **Order invariance.** Largest change in any output when rows and feature columns are
   permuted (same seed), after mapping features back.
6. **Runtime and memory.** Wall seconds and peak resident memory per scale cell.

Real-data track (no ground truth):

7. **Bootstrap stability.** 100 cluster bootstrap resamples (units resampled with all
   their rows). Standard deviation of the composition and structure values; mean
   Spearman correlation of per-feature |attribution| with the full-sample ranking.
8. **Held-out agreement.** 50 random half/half splits of units. Mean gap between the two
   halves' mean-channel composition and structure values; mean correlation of their
   per-feature vectors.

A method that cannot produce a quantity reports n/a. Nothing is imputed.

## 4. Methods

Reference period A everywhere. Specifications are fixed in advance; no method is tuned on
the suite.

* **OB**: Oaxaca-Blinder with OLS on X. Mean only; per feature.
* **DFL**: DiNardo-Fortin-Lemieux reweighting. Logit of period on standardised [X, X^2];
  period-A outcomes reweighted by the odds. All statistics; aggregate only.
* **CFM**: Chernozhukov-Fernandez-Val-Melly distribution regression. Logit of 1{y <= t}
  on standardised [X, X^2] at 39 thresholds (pooled quantiles); counterfactual CDF
  averages the period-A conditional CDF over period-B features; monotone
  rearrangement; statistics read from each CDF on the same grid. All statistics; aggregate.
* **RIF**: Firpo-Fortin-Lemieux recentred influence-function regression on [X, X^2]
  (Gaussian kernel density, Silverman bandwidth, for quantile RIFs), then Oaxaca-Blinder.
  All statistics; per feature (a feature's two columns summed).
* **SHAP**: Shapley drift attribution. A LightGBM model per period (300 trees, learning
  rate 0.05, 15 leaves); TreeSHAP values (tree-path-dependent) of each model.
  Composition_j is the change in mean SHAP value of the period-A model from A rows to B
  rows; structure_j is the mean over B rows of the difference between the two models'
  SHAP values, pooled-centred. Totals use full-sample predictions; per-feature values use
  5,000 explained rows per period. Mean only; per feature.
* **TRACE**: slot defined by `benchmarks/shiftbench/sb_adapter.py::TraceMethod`, filled
  when TRACE shift decomposition v2 lands. Same grid, seeds and metrics.

## 5. Real data

A public quarterly financial panel: 96 units observed quarterly. Period A is 2017-2019
(12 quarters), period B is 2023-2025 (12 quarters); 1,152 unit-quarters in each. The
outcome is a return ratio (percent); eight anonymised ratio features `x1..x8` (`x1` is log
scale). Pooled 1st/99th percentile winsorising. Units are anonymised integer ids.
This panel was chosen among the candidate datasets in the project's data store because
it has the most units with complete coverage of both windows, a continuous outcome, and
public-domain status, so the derived file can be committed
(`benchmarks/shiftbench/data/real_panel.csv.gz`). The raw extract and preparation script
are kept with the source data, outside the repository.

## 6. Plugging in TRACE

Implement `traceot.shift_decompose(XA, yA, XB, yB, seed)` in the bindings (returning
`split`, optionally `feat_comp`, `feat_struct`, `intercept`), add `TraceMethod()` to
`RIVALS` in `sb_methods.py` (or a `--with-trace` switch), and rerun every track. The
truth file and seeds do not change.
