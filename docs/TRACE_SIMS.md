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

Run on the same cloud VM as `docs/BENCHMARKS.md` (8 vCPU, x86_64, about 15 GB RAM), release build, 11 October 2026. Full precision is in `benchmarks/results/trace_sims.json` and `benchmarks/results/trace_ebm.json`. Nothing in the criteria section above was edited after this run.

### Projection blur

Passes. Location interaction, contrast ratio `1.000` (criterion `>= 0.99`). Shape interaction, ratio `0.825` (criterion `< 0.90`). The curve at `(x1, x2) = (1, 1)` decreases in `u` before the projection and is monotone after it. The other three corner curves were already monotone.

### S1 — planted shifts

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

### S2 — planted leaks

**Pass**, all nine checks.

| Check | Measured |
|---|---|
| Proxy screen | `x_proxy` at shift 0, correlation 1.000; `x_future` at shift +1, correlation 1.000. No flag on `x_signal` (0.737), `x_leak` (0.664) or noise (0.008). |
| Duplicate screen | Only `(x_signal, x_dup)`, exact. |
| Placebo, 20 shuffles | maximum absolute correlation 0.086 (criterion `< 0.25`). Real `x_proxy` correlation 1.000 (criterion `> 0.95`). |
| W2 time / random | `x_leak` 25.4 (criterion `> 3`); `x_signal` 1.55 (criterion `< 2`). Largest ratio is `x_leak`. `x_future` is 17.2 and `x_proxy` is 13.7; both move, and the leak still ranks first. |
| Permutation importance of `x_leak` | random-split test 0.0478, time-split test 0.00786, ratio 6.1 (criterion `> 2`). Five shuffles, seed 0. |

### S3 — oddness

**Pass.** False-alarm rate on the clean test rows `0.077` (criterion `<= 0.20`). Power on the 30 spiked rows `1.00` (criterion `>= 0.80`). Fraction of the shifted segment with p < 0.1 is `1.00` (criterion `>= 0.50`).

### S4 — scale

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

### EBM comparison

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
