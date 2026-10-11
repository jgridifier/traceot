# TRACE simulations

Status: research, unvalidated. Fixed seeds. The success criteria in this section were written before any TRACE fit on these designs was inspected. Thresholds are not revised after seeing the numbers. If a criterion fails, the result is reported as a miss.

Data generators and the runner live in `benchmarks/trace_sims/`. Outcomes are drawn by inverse transform from the stated quantile function, with `z(u) = Phi^{-1}(u)` and `He3(z) = z^3 - 3z`.

## Criteria

### S1 — planted shifts

Seed 11. Two periods, `n = 2000` rows each, 8 features on `[0, 1]`: `x_loc`, `x_scale`, `x_shape`, and five noise columns.

- Period 1: `Q(u | x) = z(u)`.
- Period 2: `Q(u | x) = 1.2 x_loc + (0.4 + 1.2 x_scale) z(u) + 0.12 x_shape He3(z(u))`. The shape multiplier `0.12` is below `1/3`, so the conditional quantile stays increasing in `u`.

Both periods are binned with one shared set of edges (the pooled columns) and fit with `fit_codes` so the coefficient arrays share bins. Config: `max_bins=16`, `u_levels=31`, `n_channels=3`, `learning_rate=0.4`, `max_rounds=80`, `patience=12`, `val_frac=0.2`, `n_bags=1`, `max_interactions=0`, `min_count=5`, `seed=11`.

Let `E(j, m)` be the mean square of the period-2 minus period-1 coefficient of feature `j`, channel `m`, on the value bins (bin 0, the missing bin, is skipped).

- `E(x_loc, location) >= 3 * max_{j ≠ x_loc} E(j, location)`.
- `E(x_scale, scale) >= 3 * max_{j ≠ x_scale} E(j, scale)`.
- `E(x_shape, shape) >= 3 * max_{j ≠ x_shape} E(j, shape)`.
- Pearson correlation of the bin index with the location-channel difference of `x_loc` is `> 0.85`. The same correlation for the scale channel of `x_scale` and the shape channel of `x_shape` is `> 0.7`.

### S2 — planted leaks

Seed 22, `n = 1800`. Let `t = 0, …, n-1`.

```
y = 0.8 x_signal + sin(2 π t / n) + 0.05 Normal(0, 1)
x_leak = sin(2 π t / n)
x_proxy = y + 0.01 Normal(0, 1)
x_future[t] = y[t+1]   (the last row copies y[n-1])
x_dup = x_signal       (exact copy)
```

plus one noise column. Proxy threshold `0.9`.

- The target-proxy screen flags `x_proxy` at shift 0 and `x_future` at a non-zero shift.
- It does not flag `x_signal`, `x_leak` or the noise column.
- The duplicate screen flags the exact pair `(x_signal, x_dup)` and no other pair.
- Placebo, 20 permutations, seed 22: the maximum absolute correlation under the shuffled target is `< 0.25`, while the real absolute correlation of `x_proxy` is `> 0.95`.

Time split: first 70% train, last 30% test. Random split: 50/50 with seed 23. Fit config for both models: `max_rounds=25`, `learning_rate=0.35`, `max_bins=16`, `val_frac=0.2`, `patience=6`, `u_levels=19`, `n_bags=1`, `max_interactions=0`, `seed=0`. W2 grid size 64.

- `W2(x_leak, time split) / W2(x_leak, random split) > 3`.
- `W2(x_signal, time) / W2(x_signal, random) < 2`.
- The feature with the largest time/random W2 ratio is `x_leak`.
- Permutation importance of `x_leak` on the random-split model, evaluated on the random test set, is `> 2` times its importance on the time-split model, evaluated on the time test set.

### S3 — oddness

Seed 33, `n = 2400`, 4 uniform features. `y = x0 + 0.5 x1 + Normal(0, 0.5)`. Train is the first 1200 rows, calibration the next 600, test the last 600.

Anomalies: test-local rows `0..29` (30 rows), `y += 6`. Segment: test-local rows `200..299`, `y += 2.5`. The two sets do not overlap.

Fit on the train window only: `val_frac=0.15`, `max_rounds=30`, `learning_rate=0.35`, `max_bins=12`, `n_channels=3`, `u_levels=19`, `n_bags=1`, `max_interactions=0`, `patience=8`, `seed=33`. Scores are per-row mean pinball. The conformal p-value uses the calibration scores. Level `alpha = 0.1`.

- False-alarm rate on the clean test rows (p < 0.1) is `<= 0.20`.
- Power on the anomalies (p < 0.1) is `>= 0.80`.
- Fraction of the segment with p < 0.1 is `>= 0.50`.

### S4 — scale

Config, every cell: `max_rounds=3`, `patience=0`, `val_frac=0`, `n_bags=1`, `bag_frac=1`, `max_interactions=0`, `max_bins=32`, `u_levels=19`, `n_channels=3`, `learning_rate=0.3`, `min_count=1`, `seed=0`.

Grid: rows in `{1e4, 1e5, 1e6, 1e7}` and features in `{10, 100}`. Columns are generated and binned one at a time to `uint8`; the fit receives codes, never an `n × p` float matrix.

- All eight cells finish.
- For each feature count, `time(1e6) / time(1e5) < 25` and `time(1e5) / time(1e4) < 25`.
- Peak resident memory at `1e7 × 100` is reported. There is no numeric cap. Running out of memory is a failed criterion and is reported as such.

### Projection blur (not a fit)

Locked from the isotonic map alone, before any booster was run. Grid `K = 21`.

- Location interaction `Q = z + 1.2 x1 x2`: the contrast ratio (RMS after PAV) / (RMS before) is `>= 0.99`.
- Shape interaction `Q = z + 0.6 x1 x2 He3(z)`: the ratio is `< 0.90`, and each of the four projected curves is monotone.

### Vendor spend (descriptive)

No winner criterion. Features: `log_units`, `unit_price`, `deliveries`, `distance`, `quality`, `season`, and three noise columns.

```
Q(u) = 10 + 4 log(units) + 2 unit_price + (0.8 + 0.5 log(units)) z(u) + 0.05 quality He3(z)
```

The metrics table is recorded. When a mean EBM wins a metric, that is stated.

### EBM comparison

Optional, benchmark-only. `interpret` is not a runtime dependency. On S1–S4 and the vendor-spend draw, compare:

- TRACE;
- EBM regression (InterpretML), when the package imports;
- that EBM plus split-conformal residual quantiles (the distributional baseline);
- an EBM quantile objective, only if the installed `interpret` exposes one. If it does not, say so and skip it.

Metrics: CRPS / pinball, coverage of 80% and 90% intervals, MSE of the median and of the mean, fit time, peak memory, and whether the shape plots recover the planted channel. Report the cases where EBM wins.

## Results

Two runs, same machine as `docs/BENCHMARKS.md` (8 vCPU, x86_64, about 15 GB RAM), release build, 11 October 2026. The criteria section above was not edited after either run. v0 is the first fit. The second run is the fitter that keeps a shared channel step, with the defaults from the separate tuning grid.

### v0

Full precision of this run is the v0 block that used to be the only contents of `benchmarks/results/trace_sims.json` and `benchmarks/results/trace_ebm.json`. Those files now hold the second run. The v0 numbers below are the record.

#### Projection blur

Passes. Location interaction, contrast ratio `1.000` (criterion `>= 0.99`). Shape interaction, ratio `0.825` (criterion `< 0.90`). The curve at `(x1, x2) = (1, 1)` decreases in `u` before the projection and is monotone after it. The other three corner curves were already monotone.

#### S1 — planted shifts

**Partial.** Location and scale pass. Shape fails.

| Check | Value | Criterion | |
|---|---:|---|---|
| `E(x_loc, location)` / next location energy | 0.0697 / 0.00503 = 13.9 | `>= 3` | pass |
| `E(x_scale, scale)` / next scale energy | 0.0138 / 0.00096 = 14.4 | `>= 3` | pass |
| `E(x_shape, shape)` | 0 | `>= 3` times the next shape energy | fail |
| corr(bin index, location of `x_loc`) | 0.967 | `> 0.85` | pass |
| corr(bin index, scale of `x_scale`) | 0.966 | `> 0.7` | pass |
| corr(bin index, shape of `x_shape`) | undefined (all zeros) | `> 0.7` | fail |

`x_shape` is the zero vector in both periods, so the difference is zero. A separate look at the same period-2 fit (same seed and the locked hyperparameters) shows why: its held-out pinball gain is `-0.00020`, and the rule keeps a feature only when the gain is strictly above 0. The tail is real but small (`0.12 x He3`), and early stopping halted that bag at round 22 of 80, before the shape channel had earned its keep on the validation slice.

This is not a retune. The threshold stays 0.

#### S2 — planted leaks

**Pass**, all nine checks.

| Check | Measured |
|---|---|
| Proxy screen | `x_proxy` at shift 0, correlation 1.000; `x_future` at shift +1, correlation 1.000. No flag on `x_signal` (0.737), `x_leak` (0.664) or noise (0.008). |
| Duplicate screen | Only `(x_signal, x_dup)`, exact. |
| Placebo, 20 shuffles | maximum absolute correlation 0.086 (criterion `< 0.25`). Real `x_proxy` correlation 1.000 (criterion `> 0.95`). |
| W2 time / random | `x_leak` 25.4 (criterion `> 3`); `x_signal` 1.55 (criterion `< 2`). Largest ratio is `x_leak`. `x_future` is 17.2 and `x_proxy` is 13.7; both move, and the leak still ranks first. |
| Permutation importance of `x_leak` | random-split test 0.0478, time-split test 0.00786, ratio 6.1 (criterion `> 2`). Five shuffles, seed 0. |

#### S3 — oddness

**Pass.** False-alarm rate on the clean test rows `0.077` (criterion `<= 0.20`). Power on the 30 spiked rows `1.00` (criterion `>= 0.80`). Fraction of the shifted segment with p < 0.1 is `1.00` (criterion `>= 0.50`).

#### S4 — scale

**Pass.** All eight cells finished. For each feature count, `time(1e6) / time(1e5)` is 7.1 (10 features) and 7.7 (100 features); `time(1e5) / time(1e4)` is 6.8 and 6.3. The criterion is `< 25`. Ten times the rows costs about 7 to 12 times the wall time (binning plus the three-round fit), which is in the linear range once fixed overhead is included. From 1e6 to 1e7 the ratios are 9.2 (10 features) and 11.8 (100 features); those two were not part of the locked check and are reported anyway.

Wall time is binning plus fit. Columns are generated one at a time and never stored as an `n × p` float matrix. Peak RSS is `ru_maxrss` of the child process.

| rows | features | bin (s) | fit (s) | wall (s) | peak RSS (MB) |
|---:|---:|---:|---:|---:|---:|
| 1e4 | 10 | 0.002 | 0.014 | 0.015 | 39 |
| 1e4 | 100 | 0.011 | 0.068 | 0.079 | 40 |
| 1e5 | 10 | 0.021 | 0.082 | 0.103 | 47 |
| 1e5 | 100 | 0.173 | 0.326 | 0.498 | 55 |
| 1e6 | 10 | 0.145 | 0.583 | 0.728 | 125 |
| 1e6 | 100 | 1.16 | 2.67 | 3.83 | 211 |
| 1e7 | 10 | 1.26 | 5.43 | 6.69 | 838 |
| 1e7 | 100 | 12.0 | 33.1 | 45.2 | 1696 |

No cell ran out of memory. At 1e7 × 100 the peak is 1.7 GB.

#### EBM comparison

`interpret` 0.7.8 imported. `ExplainableBoostingRegressor(..., objective="quantile")` raises `Unrecognized objective type`, so there is no quantile EBM in this build and that column is absent. The point-mass EBM (the same prediction at every quantile level) has coverage 0, which is what a single number does; it is not the distributional baseline. The distributional baseline is that EBM plus split-conformal residual quantiles.

EBM wins the mean, and often the CRPS, when the job is a conditional mean plus a global band. TRACE is much faster at these sizes and its location channel tracks the planted location feature (correlation of bin index with the location coefficient: 0.95 on S1 period 2, 0.996 on the vendor draw). TRACE's intervals are conservative: coverage sits above the nominal 80/90% on the vendor draw and on the homoscedastic scale cell.

| Case | Model | CRPS | 80% cover | 90% cover | MSE mean | Fit (s) |
|---|---|---:|---:|---:|---:|---:|
| S1 period 2 | TRACE | 0.522 | 0.858 | 0.930 | 0.993 | 0.005 |
| S1 period 2 | EBM + conformal | 0.522 | 0.822 | 0.922 | 0.972 | 0.766 |
| S1 period 2 | EBM point | 0.692 | 0 | 0 | 0.971 | 0.766 |
| Vendor spend | TRACE | 1.187 | 0.996 | 1.000 | 3.022 | 0.049 |
| Vendor spend | EBM + conformal | **0.743** | 0.814 | 0.917 | **1.799** | 1.84 |
| Vendor spend | EBM point | 1.021 | 0 | 0 | 1.798 | 1.84 |
| n=2e4, p=10, 3 rounds | TRACE | 0.104 | 0.998 | 1.000 | 0.0213 | 0.014 |
| n=2e4, p=10, 3 rounds | EBM + conformal | **0.057** | 0.795 | 0.904 | **0.0102** | 1.86 |
| n=2e4, p=10, 3 rounds | EBM point | 0.081 | 0 | 0 | 0.0102 | 1.86 |

On S1 period 2 the CRPS values agree to three digits. The conformal band is closer to 80/90% coverage. The EBM point prediction wins MSE by a small margin (0.971 against 0.993), which is the loss it trains. On the vendor draw and on the homoscedastic cell, EBM plus residual quantiles wins CRPS and MSE, and its coverage lands on the nominal levels. TRACE's location plot recovers the planted shift in every case above. The shape channel does not: S1 dropped that feature. An EBM has no scale or shape channel; its term for the first feature is a nonzero step (range 0.95 on S1 period 2, 7.9 on the vendor draw), which is a mean effect, not a tail.

S4 at 1e7 rows was not refit with EBM. The small scale cell is the EBM timing comparison.

### After the shared channel step

The v0 logs are `benchmarks/results/trace_diagnosis.json`. That file is the fitter from before this round. The diagnosis script refuses to overwrite it.

#### What the v0 logs showed

On the n=2e4 homoscedastic cell, the published comparison was capped at 3 rounds (`patience=0`, `val_frac=0`, learning rate 0.35). Training pinball fell 0.0875 → 0.0619 → 0.0544 → 0.0509 and was still falling. Median MSE on the test slice was 0.0213, about twice the EBM's 0.0102. The mean absolute location step was 0.0076. A fully one-sided location residual would step by `lr/2 = 0.175`, and round 1's location step was 0.012, so the update is not being divided by 3 channels times `p`. Location centering kept essentially the whole contrast (fraction 1.000 on round 1, mean 0.969 over the three rounds).

The width did not move. Predicted 10–90 spread was 0.816 against a marginal `Q0` spread of 0.818 (ratio 0.998). Sixty rounds with patience 0 brought median MSE to 0.01045, which is already next to the EBM, and the spread ratio was still 1.000. The same cell with learning rate 0.3, 40 rounds and patience 8 ran all 40 rounds: validation pinball fell the whole way, so patience and `tol=0` were not what stopped it, and the spread ratio was still 1.000. The median can converge while the band stays at the marginal quantile, because centering throws away the channel mean that every bin agrees on. That mean is a scale (and shape) correction, not a location contrast.

The vendor draw, same v0 fitter, learning rate 0.35, 40 rounds, patience 8, also ran all 40 rounds. Validation pinball fell from 1.021 to 0.585. Median MSE was 3.02. Spread ratio 1.004, location centering kept 0.985 of the contrast. Eighty rounds, patience 0, still had falling validation loss (0.538 at the end) and median MSE 2.01, against the EBM's 1.80. The band stayed on `Q0`.

S1 period 2 on a 70% slice of the locked config stopped at round 17 of the logged path. The scored v0 fit, on the full period, stopped at round 22 and dropped `x_shape` on a held-out gain of −0.00020.

#### Tuning, on a different draw

Defaults were chosen on seed 101, `n=4000`, `p=5`, `y = 3(x0−0.5) + 1.2(x1−0.5) + 0.25 N(0,1)`, last 30% held out, `max_bins=32`, 19 quantile levels, one bag, no interactions. Bayes MSE of the mean is 0.0625. The grid is learning rate `{1.0, 1.5, 2.0, 2.5, 3.0}` times `max_rounds` `{40, 80}` times `patience` `{8, 20}` times `min_rounds` `{0, 10, 20}`: 60 rows in `benchmarks/results/trace_tune_grid.json`. This is the fitter that adds the shared channel step. The best held-out median MSE is 0.06610, at learning rate 1.0. Learning rate 1.5 is 0.0673. Learning rate 2.5 is about 0.088 to 0.100. Ties at learning rate 1.0 share that same MSE, including `min_rounds` 0, 10 and 20, because the restored checkpoint is after round 20. The chosen defaults are learning rate 1.0, `max_rounds=40`, `patience=8`, `min_rounds=20`. The last of those does not change the tuning MSE and is the floor for the feature screen. The vendor draw, S1, and the n=2e4 cell were not in the grid.

#### S1 — planted shifts

**Fail. Location and scale pass. Shape fails.** The locked hyperparameters are unchanged. `min_rounds` is not in that list, so the class default 20 applies: early stopping cannot restore a round before 20, and the screen cannot drop a feature before the checkpoint round reaches 20.

| Check | Value | Criterion | |
|---|---:|---|---|
| `E(x_loc, location)` / next location energy | 0.114 / 0.00782 = 14.6 | `>= 3` | pass |
| `E(x_scale, scale)` / next scale energy | 0.0250 / 0.00152 = 16.4 | `>= 3` | pass |
| `E(x_shape, shape)` | 0 | `>= 3` times the next shape energy | fail |
| corr(bin index, location of `x_loc`) | 0.968 | `> 0.85` | pass |
| corr(bin index, scale of `x_scale`) | 0.959 | `> 0.7` | pass |
| corr(bin index, shape of `x_shape`) | undefined (all zeros) | `> 0.7` | fail |

Period 2 ran 32 rounds. `x_shape` held-out pinball gain is −0.000537. Its shape-channel gain, tail-weighted by `(u−0.5)^2`, is −0.000044. Both are below 0, so the channel-aware rule still drops it. `x_loc` and `x_scale` are kept. `noise0` is also kept (overall gain +0.00203); that is not a criterion.

A diagnostic that is **not** the scored run, same seed, `gain_threshold=-1`, so nothing is dropped: at the same 32-round checkpoint the shape energy of `x_shape` is 0.00292, the next shape energy is `noise1` at 0.00337, and `x_scale`'s shape energy is 0.0195. The ratio is below 1, not above 3. Eighty rounds with patience 0 makes the shape-channel gain of `x_shape` positive (+0.000140), which would keep the feature, and the shape energy is 0.00231 against `noise1` at 0.00421 and `x_scale` at 0.0144. The bases are not orthogonal, so the scale feature leaks into `He3`. Keeping the planted tail feature does not pass the unchanged 3× test. The threshold stays 0.

#### S2 — planted leaks

**Pass**, all nine checks. The screens and the W2 ratios are the same statistics as v0 (`x_leak` time/random 25.4, `x_signal` 1.55, placebo maximum 0.086). Permutation importance moved with the fitter: random-split test 0.140, time-split test 0.0358, ratio 3.90 (criterion `> 2`; v0 was 6.1).

#### S3 — oddness

**Pass.** False-alarm rate 0.072 (v0 was 0.077; criterion `<= 0.20`). Power 1.00. Segment fraction 1.00.

#### S4 — scale

**Pass.** The locked cell is still three rounds, `patience=0`, `val_frac=0`. `min_rounds=20` does not add rounds when patience is 0 and `max_rounds` is 3, and with no validation rows the screen does not run. All eight cells finished. Growth ratios stay under 25: `time(1e6)/time(1e5)` is 6.2 and 9.1; `time(1e5)/time(1e4)` is 4.6 and 5.2.

The largest wall-time increase against the v0 table is the 1e4 × 10 cell, 0.015 s to 0.017 s (+14%). Its fit time went from 0.014 s to 0.016 s (+18%). Every larger cell is faster than the v0 measurement on this machine. Nothing exceeds the 25% regression bar. Peak RSS at 1e7 × 100 is 1696 MB, the same 1.7 GB.

| rows | features | bin (s) | fit (s) | wall (s) | peak RSS (MB) | v0 wall (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1e4 | 10 | 0.0015 | 0.016 | 0.017 | 39 | 0.015 |
| 1e4 | 100 | 0.0086 | 0.051 | 0.059 | 40 | 0.079 |
| 1e5 | 10 | 0.011 | 0.068 | 0.079 | 47 | 0.103 |
| 1e5 | 100 | 0.109 | 0.200 | 0.309 | 56 | 0.498 |
| 1e6 | 10 | 0.095 | 0.399 | 0.494 | 125 | 0.728 |
| 1e6 | 100 | 0.853 | 1.97 | 2.82 | 211 | 3.83 |
| 1e7 | 10 | 0.845 | 4.50 | 5.34 | 838 | 6.69 |
| 1e7 | 100 | 8.16 | 30.2 | 38.4 | 1696 | 45.2 |

#### EBM comparison

`interpret` 0.7.8. `objective="quantile"` still raises `Unrecognized objective type`. The point-mass EBM has coverage 0. The distributional baseline is that EBM plus the same residual-quantile procedure as v0 (`floor(u n)` on the calibration residuals). TRACE raw is a fit on every training row, with the tuned defaults (learning rate 1, 40 rounds, patience 8, `min_rounds=20`, `val_frac=0.2`). TRACE calibrated is a second fit on the first 75% of those training rows, then the Rust conformal offsets on the last 25%, the same cut the EBM uses. The two TRACE rows are not the same fit.

On n=2e4 the median MSE is 0.01024 against the EBM's 0.01018. That is about 0.6% higher, inside the 20% band (the band is about 0.0122). v0 at 3 rounds was 0.0213. The stored `Q0` 10–90 spread on this cell moved from 0.818 to 0.401. The conditional 10–90 width of `N(0, 0.1^2)` is 0.256, so the raw band is still wide and raw coverage is 0.952 / 0.999. Calibrated coverage is 0.790 / 0.905. EBM conformal coverage is 0.795 / 0.904, and EBM conformal still has the lower CRPS (0.0571 against 0.0588 calibrated and 0.0605 raw) and the lower MSE of the mean (0.01018 against 0.01024).

On the vendor draw EBM conformal still wins CRPS (0.743 against 0.770 raw) and MSE of the mean (1.799 against 1.842). Raw coverage is 0.919 / 0.995. Calibrated coverage is 0.808 / 0.908, next to the EBM's 0.814 / 0.917. The calibrated MSE is worse (1.958), which is the cost of fitting on 75% of the rows and then shifting the curve.

On S1 period 2, this comparison uses the tuned defaults, not the locked S1 hyperparameters above. TRACE raw CRPS is 0.518 against the EBM's 0.522. EBM still wins MSE of the mean (0.972 against 0.983). Location-bin correlation of the first feature is 0.958. The EBM term range on that feature is 0.95, the same mean effect as v0.

| Case | Model | CRPS | 80% cover | 90% cover | MSE mean | Fit (s) |
|---|---|---:|---:|---:|---:|---:|
| S1 period 2 | TRACE raw | 0.518 | 0.822 | 0.912 | 0.983 | 0.009 |
| S1 period 2 | TRACE calibrated | 0.524 | 0.827 | 0.940 | 0.998 | — |
| S1 period 2 | EBM + conformal | 0.522 | 0.822 | 0.922 | **0.972** | 0.70 |
| S1 period 2 | EBM point | 0.692 | 0 | 0 | 0.971 | 0.70 |
| Vendor spend | TRACE raw | 0.770 | 0.919 | 0.995 | 1.842 | 0.039 |
| Vendor spend | TRACE calibrated | 0.773 | 0.808 | 0.908 | 1.958 | — |
| Vendor spend | EBM + conformal | **0.743** | 0.814 | 0.917 | **1.799** | 1.67 |
| Vendor spend | EBM point | 1.021 | 0 | 0 | 1.798 | 1.67 |
| n=2e4, p=10 | TRACE raw | 0.0605 | 0.952 | 0.999 | 0.01024 | 0.067 |
| n=2e4, p=10 | TRACE calibrated | 0.0588 | 0.790 | 0.905 | 0.01050 | — |
| n=2e4, p=10 | EBM + conformal | **0.0571** | 0.795 | 0.904 | **0.01018** | 1.57 |
| n=2e4, p=10 | EBM point | 0.0807 | 0 | 0 | 0.01018 | 1.57 |

S4 at 1e7 rows was not refit with EBM. Bold marks the distributional row with the lower CRPS or the lower MSE of the mean, which is the EBM row in every case except S1 period 2 CRPS, where TRACE raw is lower (0.518 against 0.522) and the EBM still has the lower MSE.
