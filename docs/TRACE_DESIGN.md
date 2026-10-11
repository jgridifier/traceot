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

Bin updates are centered by their count-weighted mean, so a feature cannot move the sample-mean quantile. That global level stays in `Q0`. Dropped features are zeroed after bag coefficients are averaged. The per-bin standard deviation is the sample standard deviation across bags, computed before that zeroing.

## Algorithm

```
bin each column (code 0 = non-finite; else 1 + count of edges <= x)
Q0 <- empty
for bag in bags:
    mask <- train / validation / out-of-bag   # SplitMix64, see below
    q0_bag <- empirical quantile of the training outcomes
    theta <- 0
    best <- zeros
    for round in 1..max_rounds:
        G, row_loss <- pinball subgradient of Q0 + theta  # once
        for feature j in canonical order:
            for each bin with count >= min_count:
                delta[bin, m] <- lr * sum(G[m] in bin) / (count * K)
            center delta by the count-weighted mean
            add delta to the coefficient and to every row's theta
        if validation improved by more than tol:
            save coefficients; reset patience
        else if patience exhausted:
            restore the best coefficients (zeros if the null model won)
            break
    record held-out pinball gain of each feature (loss with that feature zeroed, minus the full loss)
average q0 and coefficients across bags
drop features selected in fewer than select_frac of the bags that voted
screen pairs on a subsample of residual gradients; boost the selected pairs the same way
report train and validation pinball on a fresh split of the full sample
```

Masks. `n_bags = 1` and `bag_frac = 1` uses one shuffle, seed `seed + 20011 * (bag + 1)`, and marks `round(val_frac * n)` rows as validation (Rust `f64::round`, half away from zero). Otherwise a shuffle with seed `seed + 10007 * (bag + 1)` draws the bag, then a second shuffle holds out the validation rows. The report split uses seed `seed + 20011`. Shuffles are Fisher–Yates driven by SplitMix64. A feature is selected in a bag when its gain is strictly above `gain_threshold`. Bags with no validation split do not vote and do not drop features. A feature is kept when `votes = 0` or `count + 1e-12 >= select_frac * votes` (default `select_frac = 0.7`).

Interactions. Eligible features are the ones that were kept, visited in canonical order. On a training subsample of size at most `screen_rows` (shuffled with seed `seed + 42042`, then sorted so the accumulation order is fixed), each pair gets a score `sum over cells of count * ||mean(G) / K||^2`. Positive scores are sorted by score descending (`total_cmp`), then by canonical position, and truncated to `max_interactions`. Pair tables are centered by their grand mean only. They are not purified into pure interactions; a pair can still carry an additive margin. That is an open identifiability gap, recorded in the risks below.

Early stopping. It runs only when `patience > 0`, `val_frac > 0` and the split actually produced validation rows. The search starts from the null model. Strict improvement is `loss < best - tol`. `bag_rounds` is how many rounds ran, not which round was restored.

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

- Pair tables are only grand-mean centered, so they can absorb main-effect margins. Purification (a two-way ANOVA removal of the additive part) is not done in v0.
- With `n_bags > 1`, the report split's validation rows may have been training rows in some bags, so `val_loss` is slightly optimistic. `n_bags = 1` with `val_frac > 0` is a genuine holdout (the report split matches that bag's split).
- `Q0` uses rank `floor(u * n)`, which is not the exact pinball minimiser on every grid. The leftover is `O(1/n)` and is not corrected by the centered feature updates.
- A pure scale or shape effect is identified only up to `Q0`. Coefficients are on the RMS-normalised basis, so they are not the raw multipliers in a simulation formula. Recovery tests use energy ratios and correlations, not those multipliers.
- The shape channel can make `Q(u)` decrease in `u` before the projection. The projection then hides part of an interaction. That blur is measured; it is not removed.
