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

## v1 round

The v0 fitter is unchanged. S1–S4 below are a re-run of that fitter. The new checks are T1–T14 in `docs/theory/DESIGN_DELTA.md` §7, scored by `benchmarks/trace_sims/run_v1.py` into `benchmarks/results/trace_v1.json`. Those pass criteria were not edited after the results.

### S1–S4, v0 fitter, re-run

**S1 still fails the unchanged shape test.** Location energy 0.114 against the next location energy 0.00782 (ratio 14.6). Scale energy 0.0250 against 0.00152 (ratio 16.4). Shape energy of `x_shape` is 0, because the screen still drops it (shape-channel gain −0.000044). Correlations: location 0.968, scale 0.959, shape undefined. This is the same failure as the round above.

**S2 pass. S3 pass.** S2 false-alarm and power checks are unchanged in kind. S3 false-alarm rate, power and segment fraction stay inside their bars.

**S4 pass. No wall-time regression above 25%.** The locked cell is still three rounds, `patience=0`, `val_frac=0`, learning rate 0.3. Growth ratios stay under 25. Against the round-2 walls on this machine, the largest increase is 1e7 × 10, 5.34 s to 6.07 s (+14%). Against the original v0 walls, 1e4 × 10 goes from 0.015 s to 0.016 s (+6%) and every larger cell is faster than that v0 table.

| rows | features | bin (s) | fit (s) | wall (s) | peak RSS (MB) | round-2 wall (s) | v0 wall (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e4 | 10 | 0.0012 | 0.015 | 0.016 | 39 | 0.017 | 0.015 |
| 1e4 | 100 | 0.0086 | 0.044 | 0.052 | 40 | 0.059 | 0.079 |
| 1e5 | 10 | 0.011 | 0.068 | 0.080 | 47 | 0.079 | 0.103 |
| 1e5 | 100 | 0.112 | 0.215 | 0.327 | 56 | 0.309 | 0.498 |
| 1e6 | 10 | 0.094 | 0.375 | 0.469 | 125 | 0.494 | 0.728 |
| 1e6 | 100 | 0.824 | 2.12 | 2.95 | 211 | 2.82 | 3.83 |
| 1e7 | 10 | 0.991 | 5.08 | 6.07 | 838 | 5.34 | 6.69 |
| 1e7 | 100 | 9.71 | 28.8 | 38.5 | 1696 | 38.4 | 45.2 |

### T1–T14

| Test | Result | What the bar asked | What happened |
|---|---|---|---|
| T1 | pass | attribution sums to W2² within 1e-10 | max absolute gap 5.6e-16 on 1000 rows |
| T2 | pass | discrete Gram within 1e-12 of I | max gap 1.4e-15 |
| T3 | fail | He3 energy of `x_shape` ≥ 3× the next feature, and the He3 slope within 30% of 0.12√6 | ratio 0.50; slope 0.083 against 0.294 (relative error 0.72) |
| T4 | fail | each channel-3 energy within 2 Monte Carlo SEs of 200 reps, or within 25% of its L2 oracle | oracle shape/scale ratio 0.753; no feature met either bar. Seed-11 shape energy 0.00103 against oracle 0.00303 |
| T5 | fail | Legendre-native shape energy ≥ 3× others, scale leak ≤ 10% of that energy, planted energies within 25% | ratio 1.31; leak 0.76. The three planted energies are within 25% (relative errors 0.22, 0.19, 0.11). The separation bars fail |
| T6 | fail | Newton He3 energy ≤ 0.05× the 19-point v0 energy, and ≤ the permutation cutoff | ratio 0.50. The permutation half holds: 0.00092 ≤ cutoff 0.00290 |
| T7 | pass | K=3 exponential flagged; K=4 monotone with PAV move ≤ 1e-8; Bernstein grid clean | flagged; PAV distance 0; worst derivative on the 10⁴ grid is +1.10 |
| T8 | pass | PAV distances and the adjustment bound on 500 rows | worst slack 1.1e-16 |
| T9 | pass | structure inside a stratified permutation floor (rejection ≤ 0.08 at α=0.05, 200 reps); identity to 1e-8 | rejection 0.055; identity gap 1.4e-17 |
| T10 | pass | null rejection at α=0.1 in [0.048, 0.152], 300 reps, R=199 | rejection 0.100 |
| T11 | fail | RMSE ratio in [0.15, 0.40]; GLS slope variance ≤ 1.05× the better of equal and ∝m | ratio 0.206. GLS variance 1.15e-3 against 1.05× the m-proportional variance 9.21e-4 (ratio 1.31) |
| T12 | pass | exact dual equals a brute-force LP to 1e-8; max-jump bound ≥ exact | max gap 1.3e-15; the bound held |
| T13 | fail | robust period-B pinball ≤ plain in ≥ 60% of 100 reps; certificate in ≥ 95% | wins 0/100. Certificate held in 100/100. Matched-row W1 was inside the ball in 54/100 |
| T14 | pass | outcome-only W1 gap equals ε ρ̄/κ_y to 1e-9 | max gap 0 on 20 draws |

T3 is the S1 replacement test in a Hermite-in-z reporting basis. T4 is the same data in plain Legendre, where the population channel-3 ratio is 0.753 and a 3× rule is not the bar. Both fail against the pre-written oracles. The reason is written under Theory disputes; the bars were not relaxed.

### EBM comparison, this round

`interpret` 0.7.8. `objective="quantile"` still raises `Unrecognized objective type`. The table is a re-run of the same v0 comparison. EBM plus conformal residuals still wins CRPS and MSE of the mean on the vendor draw and on n=2e4. On S1 period 2, TRACE raw still has the lower CRPS (0.518 against 0.522) and EBM still has the lower MSE (0.972 against 0.983).

| Case | Model | CRPS | 80% cover | 90% cover | MSE mean | Fit (s) |
|---|---|---:|---:|---:|---:|---:|
| S1 period 2 | TRACE raw | 0.518 | 0.822 | 0.912 | 0.983 | 0.009 |
| S1 period 2 | TRACE calibrated | 0.524 | 0.827 | 0.940 | 0.998 | — |
| S1 period 2 | EBM + conformal | 0.522 | 0.822 | 0.922 | **0.972** | 0.70 |
| Vendor spend | TRACE raw | 0.770 | 0.919 | 0.995 | 1.842 | 0.023 |
| Vendor spend | TRACE calibrated | 0.773 | 0.808 | 0.908 | 1.958 | — |
| Vendor spend | EBM + conformal | **0.743** | 0.814 | 0.917 | **1.799** | 1.70 |
| n=2e4, p=10 | TRACE raw | 0.0605 | 0.952 | 0.999 | 0.01024 | 0.063 |
| n=2e4, p=10 | TRACE calibrated | 0.0588 | 0.790 | 0.905 | 0.01050 | — |
| n=2e4, p=10 | EBM + conformal | **0.0571** | 0.795 | 0.904 | **0.01018** | 1.59 |

### A comparison an EBM cannot make

These are capability differences. They are not accuracy wins.

A two-period shift with the same conditional law and a different covariate law, scored by the mixture decomposition: total W2² 0.0672 = composition 0.0850 + structure 0.00136 + cross −0.0191. The structure term is the small one. An EBM returns a mean; it has no composition/structure split of a change in the distribution.

Distribution-valued rows: each row is an exponential sample with scale linear in a discrete covariate. The Hosking fit's λ₂ RMSE on that draw is 0.00865. Collapsing each row to a mean and fitting an EBM would answer a different question.

### Theory disputes

**The T3 oracle is an L2 coefficient. The fitter minimises pinball.** T3 asks for the slope of `x_shape` in the orthonormal He3 channel to be within 30% of 0.12√6 = 0.294. On an unconditional draw from `0.8 z + 0.12 He3(z)`, a Newton fit in the degree-7 Legendre span converges (the loss stops moving, and an independent Newton in NumPy reaches the same point) to a reported He3 coefficient of about 0.118. That point has a lower pinball loss than the L2 projection of the true curve, whose reported coefficient is about 0.18. The report matrix matches a direct Gauss–Legendre inner product to 1e-10, so the gap is not a mis-scaled basis. THEORY.md §1.5 already records that a truncated pinball projection is not the L2 projection (`v1:1.17`). T3's pass bar treats them as the same number. T4 and T6 use the same identification: T4's channel-3 energies are the pinball fit's, compared with L2 oracles, and T6 asks the pinball fit of a pure scale `z` to put at most 5% of the v0 He3 energy into He3. The measured ratio is 0.50. The criteria were not changed.

## v2 baseline, before any fitter change

The corrected tests are frozen in `docs/theory/PREREG_TAIL_v2.md`. The v1 T3, T4 and T6 failures above stay failed. This section scores the fitter that was already on main (`TraceV1`, Newton pinball). The constructor is the one in the pre-registration: `k_fit=8`, `k_report=4`, `n_u=0`, `max_bins=16`, `mono="none"`, `max_rounds=200`, `tol=1e-10`. Checked against the code before the run: `n_u=0` is 12 Gauss–Legendre nodes, the bins are the order statistics at `floor(k n / 16)`, an empty reference basis is `z`, and `channel_energy` is the mean square of the value-bin coefficients. Those settings match, so nothing was adapted. Every fit used 12 nodes. No seed hit the round cap.

| Test | Result | Bar | What happened |
|---|---|---|---|
| T3v2 | pass | ≥ 9 of 20 seeds separate in the reference basis | 17 of 20. The other three ratios were 2.98, 2.42 and 2.65. Secondary bands, not a pass condition: 20 of 20 |
| T3v2-null | pass | false positives ≤ 1 of 20 | 0 of 20. The largest ratio was 0.69 |
| T4v2 | pass | ≥ 18 of 20 seeds inside the Legendre bands | 20 of 20. 3× separation is impossible in Legendre: the oracle ratio is 0.497 |
| T6v2 | pass | ≥ 19 of 20 seeds inside the channel-3 band and with ratio ≤ 0.0402 | 19 of 20. Seed 6115 had ratio 0.063 |

Raw rows are in `benchmarks/results/trace_v2_baseline.json`.

## v2 landscape items

The bars are the ones in `docs/research/LANDSCAPE.md` §4. They were not edited after these runs. None of the three items passed, so none of them replaced the default. `TraceRegressor` is still the v0 path. `TraceV1` is still the Newton path. `TraceHybrid`, the smoothed curvature, and the channel ridge are options.

### Item 1 — squared-loss mean, CRPS residual

**Fail.** The mean booster uses 1024 bins, learning rate 0.04, a cap of 50,000 rounds, patience 100, tolerance 1e-5, and a 15% validation slice. Bins thinner than 4 rows are pooled with their neighbours, because a hard leaf gate at 1024 bins takes no steps. There are no bags and no smoothing pass. Residual channels are `k_fit=6`, `k_report=4`, 12 Gauss–Legendre nodes, 16 bins, 40 rounds. Fresh seeds 81001–81020. Historical seeds are in the JSON and are not in the pass set.

| Case | Hybrid MSE | Booster MSE | Ratio | Hybrid CRPS | Booster+conformal CRPS | Raw TRACE CRPS |
|---|---:|---:|---:|---:|---:|---:|
| S1 period 2 | 1.386 | 1.213 | 1.14 | 0.649 | 0.578 | 0.576 |
| Vendor spend | 8.540 | 1.793 | 4.76 | 1.984 | 0.739 | 0.773 |
| n=2e4, p=10 | 0.0384 | 0.0101 | 3.81 | 0.128 | 0.0566 | 0.0601 |

The MSE bar is ≤ 1.01× on all three cases. The CRPS bar is ≤ the conformal booster on at least two cases, and no case worse than raw TRACE by more than 1%. Every case misses both. The landscape's own reading of an MSE miss this large is that the booster or the binning differs, not that squared loss is the wrong objective for the mean channel. This booster does not smooth adjacent bins and does not bag. The bar was not changed to fit that difference.

#### Follow-up: is the 4× gap a bug?

Checked before merge, on the same generators and the same split, without editing the booster or the bar. The recorded table above is unchanged. There is no corrected re-run. The measurements are in `benchmarks/results/trace_v2_item1_diagnosis.json`.

The mean channel alone, with no residual stage, reproduces the hybrid MSE. On vendor seed 44 the booster's test MSE is 8.45 and `hybrid.mean` is 8.45, against a TraceV1 quadrature mean of 1.98 and the additive booster at 1.80. On the n=2e4 seed the four numbers are 0.0397, 0.0397, 0.0105 and 0.0102. The residual stage is not where the MSE goes.

The residual cannot move \(c_0\). On 12 Gauss–Legendre nodes, \(e_0\) integrates to 1 and \(e_k\) for \(k \ge 1\) integrates to 0 (gap below \(10^{-12}\); covered by `higher_channels_integrate_to_zero_on_the_nodes`). Channel-0 energy of the locked residual is 0. The monotone projection changes the node values and leaves the quadrature mean where it was: the largest shift on these two seeds is about \(10^{-14}\). `hybrid.mean` matches that quadrature, with and without the projection, to about \(10^{-14}\). The reported mean is the booster, not a differently weighted node average.

The squared-loss step is the learning rate times the pooled bin residual. On the n=2e4 seed the first step has mean absolute residual 0.249 and mean absolute update 0.00998, ratio 0.040, and the count-weighted centre is 0. `one_step_is_learning_rate_times_residual` locks that arithmetic. A one-feature draw of \(x + 0.1\,\varepsilon\) reaches validation MSE 0.0135 against a noise floor of 0.01. The same draw with nine noise features stops at validation MSE 0.056.

Training and validation MSE on the two bad seeds:

| Round | n=2e4 train | n=2e4 val | Vendor train | Vendor val |
|---:|---:|---:|---:|---:|
| 1 | 0.079 | 0.089 | 11.05 | 12.65 |
| 21 | 0.011 | 0.051 | 0.996 | 10.22 |
| 41 | 0.005 | 0.045 | 0.322 | 10.19 |
| 101 | 0.002 | 0.042 | 0.046 | 10.49 |

The noise floors are about 0.010 and 1.76. Training falls through them. Validation never does, and early stopping restores that valley (best round 99 and 31). On the official fresh seeds, 10 of 20 S1 fits stop at exactly 100 rounds, which is the patience expiring while the intercept is still the best validation model.

The same step on 16 bins, one seed only, is not 4× off: n=2e4 test MSE 0.0106, vendor 2.00. That is a check, not a new item-1 result, and the 1024-bin booster was not retuned. An additive booster with one bag, no smoothing, and a pure cyclic schedule still lands at 0.0103 on the n=2e4 seed, because each of its steps is a two-leaf split inside the histogram. This channel writes a level into every pool on every visit.

The 200-round cap is a slow residual, not a step that increases the loss. A negative relative drop is already below the tolerance, so a step that raised pinball would stop by round 2. On vendor seed 44 the locked residual meets \(10^{-10}\) at round 168; on an 8,000-row scale draw it meets it at round 88. The unlocked TraceV1 fits in the baseline meet it in about 50 to 80 rounds. The lock sets the channel-0 component of the joint Newton step to zero after the solve, which keeps \(c_0\) fixed and leaves a shorter step. Thirty-eight of the eighty hybrid tail fits are still above the tolerance at round 200. Items 2 and 3 were not refit.

#### Item 1, edges fixed

Quantile cuts were taken from every row passed to `fit_mean`, including the internal validation slice. They are now taken from the training rows only. Same seeds, same bar, same full-pool booster otherwise. The original row above stays in place. JSON: `benchmarks/results/trace_v2_item1_edges.json`. On the n=2e4 historical seed the test MSE moves from 0.03967 to 0.03827.

| Case | Hybrid MSE | Booster MSE | Ratio | Hybrid CRPS | Booster+conformal CRPS | Raw TRACE CRPS |
|---|---:|---:|---:|---:|---:|---:|
| S1 period 2 | 1.309 | 1.213 | 1.08 | 0.608 | 0.578 | 0.576 |
| Vendor spend | 8.110 | 1.793 | 4.52 | 1.923 | 0.739 | 0.773 |
| n=2e4, p=10 | 0.0381 | 0.0101 | 3.78 | 0.129 | 0.0566 | 0.0601 |

**Fail.** The MSE ratios are still above 1.01 on every case, and CRPS is still worse than both comparators on every case.

### Item 1b — pre-registered before the run

Attempt 2 of the item-1 hypothesis. Written down before the run. No result is recorded in this section yet.

- **Change:** replace the full-pool write in `step_feature` with one two-leaf split per feature visit. Keep 1,024 cuts, learning rate 0.04, min leaf 4, the count-weighted centre, and the restore logic. Nothing else changes, and there is no tuning on the three report cases.
- **Bar:** the same as item 1. MSE ≤ 1.01× the additive booster on all 3 cases, and CRPS ≤ booster+conformal on at least 2 of 3. Use the same seeds. The item-1 scorer also requires CRPS ≤ 1.01× raw TRACE on every case, and a one-sided paired t-test at p < 0.05 on each of those contrasts. A CRPS case counts toward the 2-of-3 only when the mean and the t-test agree.
- **Disclosure:** this is attempt 2 of the item-1 hypothesis, so a pass is weaker evidence than a first-try pass.

### Item 2 — smoothed pinball curvature

**Fail.** Candidate: `TraceV1`, `k_fit=8`, learning rate 1.0, 40 rounds, logistic pinball density with bandwidth 0.05 annealed to 0.02 then 0.01, Levenberg damping `1e-3`. Baseline: the same constructor with those two knobs off. The bandwidth and the damping were fixed before the run.

Leakage is He3 energy of `y = (0.5+x)z` divided by the recorded v0 He3 energy 0.0018508010614343975. Seed 6, the historical T6 seed, gives 0.623. The mean over seeds 83001–83020 is 0.816. The bar is 0.10. The v1 Newton ratio on seed 6 was 0.496.

| Case | Candidate CRPS | Baseline CRPS | Ratio | Better |
|---|---:|---:|---:|---|
| S1 period 2 | 0.5993 | 0.5964 | 1.0049 | no |
| Vendor spend | 0.7642 | 0.7637 | 1.0006 | no |
| n=2e4, p=10 | 0.057810 | 0.057811 | 1.0000 | yes, paired p = 0.11 |

No case is worse by more than 0.5%. The improvement bar, better on at least two cases with a one-sided paired p < 0.05, is 0 of 3.

### Item 3 — channel-index ridge and tail weight

**Fail.** The span is `k_fit=8`, `k_report=4`, Hermite reporting. The grid, chosen only by validation CRPS, is `s ∈ {1, 2}`, `λ ∈ {1e-4, 1e-3, 1e-2, 1e-1}`, `γ ∈ {0, 1, 4}`. The baseline is `k_fit=3` with no penalty.

The screen keeps `x_shape` on 14 of 50 seeds (84001–84050). The bar is 40 of 50. The He3 energy ratio of `x_shape` to the next feature has mean 0.174 and median 0.155. The bar is a mean of at least 2. Validation CRPS usually selects `s=2` and `λ=0.1`, which is the heavy end of the grid, and that is where the tail energy goes.

| Case | Ridge CRPS | Baseline CRPS | Ratio |
|---|---:|---:|---:|
| S1 period 2 | 0.5923 | 0.5996 | 0.988 |
| Vendor spend | 0.7598 | 0.7664 | 0.991 |
| n=2e4, p=10 | 0.05777 | 0.05812 | 0.994 |

CRPS is better on all three cases and inside the 0.5% allowance. That part passes. Retention and the energy ratio do not, so the item fails.

### Tail tests after each change

Same frozen bars as the step-1 baseline. Reported separately.

| Fitter | T3v2 | T3v2-null | T4v2 | T6v2 |
|---|---|---|---|---|
| Unchanged `TraceV1` (step 1) | pass, 17/20 | pass, 0/20 | pass, 20/20 | pass, 19/20 |
| Hybrid | pass, 13/20 | pass, 0/20 | pass, 18/20 | pass, 19/20 |
| Smoothed curvature | pass, 17/20 | pass, 0/20 | pass, 20/20 | pass, 19/20 |
| Ridge, middle of the grid (`s=1`, `λ=1e-3`, `γ=1`) | pass, 17/20 | pass, 0/20 | pass, 20/20 | pass, 19/20 |

The hybrid residual fit hits the 200-round cap on 38 of 80 runs. The other two fitters do not. JSON: `trace_v2_tail_hybrid.json`, `trace_v2_tail_smooth.json`, `trace_v2_tail_ridge.json`.

## S1–S4 and the booster comparison, re-run

The v0 fitter is the one these four designs call, and that path was not edited. S1 still fails because the screen still drops `x_shape` (shape energy 0; location ratio 14.6; scale ratio 16.4). S2 passes. S3 passes. S4 passes. Wall time for the eight S4 cells is 52.9 s.

**No S4 cell is more than 25% slower than the previous re-run on this machine.** The largest increase is 1e6 × 10, 0.469 s to 0.494 s (+5%). The 1e7 × 100 cell is 38.5 s to 39.0 s (+1%). Peak RSS at 1e7 × 100 is 1696 MB, the same 1.7 GB.

| rows | features | bin (s) | fit (s) | wall (s) | peak RSS (MB) | previous wall (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1e4 | 10 | 0.0057 | 0.0077 | 0.0135 | 39 | 0.016 |
| 1e4 | 100 | 0.0088 | 0.041 | 0.050 | 40 | 0.052 |
| 1e5 | 10 | 0.011 | 0.065 | 0.077 | 47 | 0.080 |
| 1e5 | 100 | 0.112 | 0.214 | 0.326 | 56 | 0.327 |
| 1e6 | 10 | 0.102 | 0.392 | 0.494 | 125 | 0.469 |
| 1e6 | 100 | 0.825 | 2.12 | 2.94 | 211 | 2.95 |
| 1e7 | 10 | 0.958 | 5.00 | 5.96 | 838 | 6.07 |
| 1e7 | 100 | 9.78 | 29.2 | 39.0 | 1696 | 38.5 |

The booster comparison on the same three draws matches the table already in this file to the printed digits. EBM plus conformal residuals still has the lower CRPS on vendor spend (0.743 against 0.770) and on n=2e4 (0.0571 against 0.0605), and the lower MSE of the mean on all three. TRACE raw still has the lower CRPS on S1 period 2 (0.518 against 0.522).

### Disputes

No new dispute. The v1 T3 dispute stays accepted, and the v2 tail bars were not touched. Item 1's miss is the outcome the landscape already described for a booster that is not the same booster. Items 2 and 3 miss their magnitude bars; those bars were not relaxed.
