# TRACE v0

Status: research, unvalidated. TRACE is our own unpublished method. ADOPT (Song & Müller, AISTATS 2026) stays in this repository as a reproduction only. TRACE does not use ADOPT's pairwise transport backfitting.

Background, not part of the method: cyclic gradient boosting of per-feature shape functions is the idea behind explainable boosting machines (Nori, Jenkins, Koch, Caruana, *InterpretML: A Unified Framework for Machine Learning Interpretability*, arXiv:1909.09223, 2019). Pairwise terms screened from residual structure are the idea behind GA2M / FAST (Lou, Caruana, Gehrke, Hooker, *Accurate Intelligible Models with Pairwise Interactions*, KDD 2013). The screen below is a residual-gradient energy on a 2-D histogram. It is FAST-style. It is not a claim to reproduce that paper's statistic.

## Model

Work on the quantile function of a scalar outcome `Y` given features `x`:

```
Q(u | x) = Q0(u) + sum_j f_j(x_j, u) + sum_{(j,k) in S} f_jk(x_j, x_k, u)
```

`u` is a probability level. `Q0` is the marginal empirical quantile of the training outcomes. Each `f_j` is constant on histogram bins of feature `j` and is a short expansion in `u`:

```
f_j(x_j, u) = sum_{m=0}^{M-1} theta_{j, bin(x_j), m} * b_m(u)
```

with `M` in `{2, 3, 4}`. Let `z(u) = Phi^{-1}(u)` (Acklam's approximation). Before fitting, each basis column is divided by its root-mean-square on the grid, so a unit coefficient has a comparable pinball gradient. A column whose RMS is numerically zero is left at zero.

| index | name | raw direction | role |
|---:|---|---|---|
| 0 | location | `1` | shift, constant in `u` |
| 1 | scale | `z(u)` | widening |
| 2 | shape | `He3(z) = z^3 - 3z` | tail / skew |
| 3 | (only if `M=4`) | `He2(z) = z^2 - 1` | even departure; indices 0–2 stay as above |

The grid is the midpoint rule `u_k = (k + 0.5) / K`, `K` in `[3, 401]` (default 19). Training minimises the average pinball loss on that grid, which is a CRPS approximation: `CRPS ≈ 2 * mean_u rho_u`. Reported `crps` is exactly `2 * pinball` on this grid.

After the additive sum, each predicted curve is projected onto non-decreasing functions of `u` by the pool-adjacent-violators algorithm. The projection is monotone in `u` by construction. It is also where an interaction that bends a curve backwards gets blurred; see `projection_blur` and the simulation notes.

## Why the fit does not depend on column order

ADOPT composes transport maps, and composition does not commute, so the feature order is part of the model. TRACE adds functions. Addition of a fixed list of numbers is commutative, and the implementation makes the list itself independent of the input order:

1. One pinball subgradient is computed per round, from the current quantile fit. Every feature is updated from that same snapshot (a Jacobi step). A Gauss–Seidel step, refreshing the gradient after each feature, would make the result depend on the order the features were visited.
2. Dense `fit` accumulates those updates in the canonical order of the raw columns under `f64::total_cmp` (a stable sort). A permutation of the columns produces the same sequence of additions.
3. Two bitwise-equal columns are tied. Their relative order can follow the input, but the two updates are equal and the snapshot does not change between them, so the sum is the same.
4. The isotonic projection is a function of the summed curve only.

`fit_codes` does not reorder. It trusts the caller to pass columns in a fixed index order (used when two periods must share bin edges). Order invariance is a property of dense `fit`.

Each feature's bin update is centered by its count-weighted mean, so one feature cannot sneak a sample-wide shift into its own table. The average of those removed channel means, over features that had positive weight, is added **once** to the working quantile level and folded into the stored `Q0`. A common scale or shape correction (every bin wants the marginal band narrower) therefore survives centering, and it is not added once per feature. Dropped features are zeroed after bag coefficients are averaged. The per-bin standard deviation is the sample standard deviation across bags, computed before that zeroing.

## Algorithm

```
bin each column (code 0 = non-finite; else 1 + count of edges <= x)
for bag in bags:
    mask <- train / validation    # SplitMix64; out-of-bag when bags > 1, see below
    level <- empirical quantile of the in-bag outcomes   # lower pinball endpoint
    theta <- 0
    best <- zeros, and the starting level
    for round in 1..max_rounds:
        G, row_loss <- pinball subgradient of level + theta  # once
        for feature j in canonical order:
            for each bin with count >= min_count:
                delta[bin, m] <- lr * sum(G[m] in bin) / (count * K)
            center delta; remember the removed channel means
            add the centered delta to the coefficient and to every row's theta
        add the average removed mean, once, into level
        if early stopping is on and round <= min_rounds:
            this round becomes the checkpoint
        else if validation improved by more than tol:
            save coefficients and level; reset patience
        else if patience exhausted:
            restore the best coefficients and level
            break
    if patience is 0: keep the last round
    record overall pinball gain and tail-weighted shape-channel gain
average q0 (the learned level) and coefficients across bags
drop features selected in fewer than select_frac of the bags that voted
screen pairs; boost them; purify both margins into the main effects
report train and validation pinball
```

`Q0`. The marginal piece is the exact lower pinball minimiser. Sort the finite training outcomes. The index is `clamp(ceil(u * n) - 1, 0, n - 1)`. When `u * n` is an integer the pinball minimisers are the closed interval between order statistics `y_(u n)` and `y_(u n + 1)` (1-based); the lower end is `sorted[u*n - 1]` and the upper end is `sorted[u*n]`, which is what `floor(u * n)` returns. The two agree whenever `u * n` is not an integer. TRACE stores the lower end, then adds the learned channel correction described above. Example: `n = 4`, outcomes `1, 2, 3, 4`, `u = 0.5` returns `2`, and the upper convention would return `3`.

The step is not divided by the number of channels or by `p`. A bin whose residuals are entirely one-sided moves its location channel by about `lr / 2`. Mixed residuals move less. Round logs on bag 0 (`round_train`, `round_val`, `round_step`, `round_center_kept`) record the pinball before round 1 and after every round, the mean absolute post-center location step, and the fraction of location L2 that centering kept. The restored model is the checkpoint, which can be earlier than the last logged round.

Masks. `n_bags = 1` and `bag_frac = 1` uses one shuffle, seed `seed + 20011 * (bag + 1)`, and marks `round(val_frac * n)` rows as validation (Rust `f64::round`, half away from zero). When `n_bags > 1` and `bag_frac < 1`, there is no second shuffle: seed `seed + 10007 * (bag + 1)` draws `round(bag_frac * n)` in-bag rows (clamped so the complement is nonempty when `n >= 4`), and the complement is the validation set. That is a true holdout for that bag. `bag_frac = 1` with several bags has no complement, so it keeps the older path (draw the bag, then carve `val_frac` inside it). The report split is bag 0's out-of-bag mask in the out-of-bag mode, and otherwise a fresh split with seed `seed + 20011`. Bag 0's out-of-bag rows may have trained in another bag, so the reported validation loss is a holdout for bag 0 only. Shuffles are Fisher–Yates driven by SplitMix64.

Feature screen. On the validation rows, the overall gain is the pinball with that feature zeroed, minus the full pinball. The shape gain uses weights `w(u) ∝ (u - 0.5)^2`, normalised to sum to 1, and zeroes only channel 2 (the `He3` channel). It is zero when `M < 3`. A feature is kept when either gain is strictly above `gain_threshold`. If the checkpoint round is below `min_rounds`, the screen keeps every feature. Bags with no validation split do not vote and do not drop features. A feature is kept across bags when `votes = 0` or `count + 1e-12 >= select_frac * votes` (default `select_frac = 0.7`).

Interactions. Eligible features are the ones that were kept, visited in canonical order. On a training subsample of size at most `screen_rows` (shuffled with seed `seed + 42042`, then sorted so the accumulation order is fixed), each pair gets a score `sum over cells of count * ||mean(G) / K||^2`. Positive scores are sorted by score descending (`total_cmp`), then by canonical position, and truncated to `max_interactions`. During the pair rounds each table is grand-mean centered and the average of the active pairs' means is added once to the level, the same rule as the main effects. After the pair checkpoint is restored, both margins are removed into the main effects. Histogram counts are not an outer product, so the one-shot formula `cell - row mean - column mean + grand mean` leaves a residual margin. Alternating weighted centering (at most 24 passes, stop when every margin is below `1e-12`) does zero them. Each removed row mean is added to the left feature and each removed column mean to the right feature. The grand mean travels with those margins, so the level is not given a second copy. Predictions are unchanged by the purification. Empty cells stay zero.

Early stopping. It runs when `patience > 0` and the mask has validation rows, which requires `val_frac > 0` or the out-of-bag mode. While the round index is at most `min_rounds`, that round becomes the checkpoint and the stale counter resets, so a restored model has at least `min_rounds` updates whenever `max_rounds` allows it. After that, strict improvement is `loss < best - tol`. `patience = 0` keeps the last round. `bag_rounds` is how many rounds ran, not which round was restored. Pair boosting uses the same early-stopping switch and does not force `min_rounds`.

Calibration. Optional, and off unless `calibrate` is called. On a held-out slice, the residual at level `u` is `y - Q_projected(u | x)`. The offset is the conformal quantile of those residuals: sort them and take index `clamp(ceil((n + 1) * u) - 1, 0, n - 1)`. When that rank would pass `n`, the largest residual is used, and there is no finite-sample coverage claim above `n / (n + 1)`. This index is not the pinball index used for `Q0`. `predict(..., calibrated=True)` adds the offsets and runs PAV again, so the delivered curve can sit slightly off the raw offset. `score` reports the raw pinball, CRPS, coverage and MSE. When a calibration vector is present it also reports the same quantities with a `_calibrated` suffix. The two are not mixed.

## Defaults

Chosen on a separate homoscedastic draw (seed 101, `n = 4000`, `p = 5`, `y = 3(x0-0.5) + 1.2(x1-0.5) + 0.25 N(0,1)`, last 30% held out), never on the vendor, S1, or `n = 2e4` cells. The grid is `benchmarks/results/trace_tune_grid.json`. The rule was the lowest held-out median MSE; ties go to the setting that stops sooner. `min_rounds = 20` matches that MSE at learning rate 1 and keeps the shape screen from dropping a feature before the checkpoint has had those rounds.

| Knob | Default |
|---|---|
| `learning_rate` | 1.0 |
| `max_rounds` | 40 |
| `patience` | 8 |
| `tol` | 0 |
| `min_rounds` | 20 |
| `val_frac` | 0.2 |
| `max_bins` | 32 |
| `u_levels` | 19 |
| `n_channels` | 3 |
| `n_bags` | 1 |
| `gain_threshold` | 0 |

## Checks

All of these are Rust routines. The Python package only validates arrays.

| Check | What it flags |
|---|---|
| Target-proxy screen | `|corr(x, y)|` at shift 0, and at leads/lags up to `max_shift`. Shift `+s` correlates `x[:-s]` with `y[s:]` (the column leads `y`). |
| Duplicate screen | Bit-identical columns (FNV-style hash of `to_bits`, then a confirm) and pairs with `|corr| >= near`. |
| Placebo target | The same correlations after `n_perm` Fisher–Yates shuffles of `y`, each shuffle continuing one SplitMix64 stream. Reports the real absolute correlations, the maximum over permutations, and the mean of those per-permutation maxima. |
| Feature W2 | Per column, the square root of the mean squared difference of empirical quantiles on a midpoint grid. Used as adversarial validation between a train window and a test window. |
| Permutation importance | Mean pinball increase when a column's *bin codes* are shuffled. Time-respecting use is the caller's split: fit on the past, score the increase on the future. |
| Conformal oddness | Score each row by its mean pinball. The p-value against calibration scores is `(1 + #{cal >= s}) / (n_cal + 1)`. |

Pearson correlations are two-pass, in input order, dropping non-finite pairs. The result is `NaN` when fewer than two finite pairs remain or either side has zero variance.

## Complexity

Let `n` be rows, `p` features, `K` quantile levels, `M` channels, `R` rounds.

| Step | Cost | Memory |
|---|---|---|
| Bin edges | one sort of each finite column | one column at a time |
| Codes | `O(n p)` | `n p` bytes (`u8` when `max_bins <= 255`, else `u16`) |
| One round | `O(n K M)` for the gradient, then `O(n p M)` to histogram and apply | `O(n M)` for `theta` and `G`; the `n × K` quantile matrix is never stored |
| Pair screen | `O(|eligible|^2 * screen_rows)` once | one pair table, `bins^2 * M` floats, only for selected pairs |
| Predict | `O(n (p + |S|) M K)` plus one PAV per row | output `n × K` |

The main path does not build an `n × n` matrix and does not refit a pairwise transport. Recomputing the gradient once per feature would be `O(R p n K)` and would reintroduce order dependence; the Jacobi snapshot is the scalable choice and the invariance argument at the same time.

Histogram accumulation inside a feature walks rows in order, so a parallel and a serial build of that histogram would not need atomics and would still match. The pinball fill and the coefficient apply split rows into disjoint slices under rayon. Both are bit-identical to the serial code. `--no-default-features` builds the serial core.

## What the plots are

For feature `j` and channel `m`, the binned coefficient is the contribution of that bin to `Q(u)`, in units of the normalised basis. The location plot is a step function of `x_j`. The scale plot says how that bin widens `z(u)`. The shape plot says how it moves the `He3` direction. Bag standard deviations are pointwise bands, computed before a dropped feature is zeroed. Pair plots are `bins × bins` tables per channel.

## Open risks

- The location, scale and shape bases are not orthogonal on the pinball loss. A strong scale feature leaks into the `He3` channel. A weak planted tail can then lose the shape-energy comparison even when the feature is kept. S1 is the measured case; the criterion was not relaxed.
- With `n_bags > 1`, bag 0's out-of-bag rows can still have trained in another bag, so the single reported `val_loss` is a holdout for bag 0 only. `n_bags = 1` with `val_frac > 0` is a genuine holdout of that one fit.
- A pure scale or shape effect is identified only up to the learned `Q0`. Coefficients are on the RMS-normalised basis, so they are not the raw multipliers in a simulation formula. Recovery tests use energy ratios and correlations, not those multipliers.
- The shape channel can make `Q(u)` decrease in `u` before the projection. The projection then hides part of an interaction. That blur is measured; it is not removed.
- Calibration is a per-level additive correction, then a second PAV. It is not a single two-sided CQR score, and the second projection can move the curve off the exact residual quantile. Coverage above `n / (n + 1)` is not guaranteed when the conformal rank is clamped.
