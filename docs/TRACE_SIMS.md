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

Not filled in yet. The runner writes `benchmarks/results/trace_sims.json`. This section is updated from that file after the run, without changing the criteria above.
