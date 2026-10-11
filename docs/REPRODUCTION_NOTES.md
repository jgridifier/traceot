# ADOPT reproduction notes

**Paper.** Song, W. & Müller, H.-G. (2026). *ADOPT: Additive Optimal Transport Regression.*
Proceedings of the 29th International Conference on Artificial Intelligence and Statistics (AISTATS),
PMLR 300:1558–1566. <https://proceedings.mlr.press/v300/song26c.html>

**Rules we followed.** The implementation was written from the paper (main text and supplement in the
PMLR PDF) only. No authors' code was available or used. Paper numbers are copied verbatim into
`benchmarks/adopt_paper/paper_values.json`, together with the SHA-256 of the PDF they came from, and
are never edited. Nothing was tuned towards the paper's numbers. Every place where the paper leaves a
choice open is listed below together with what we did. In code these places are marked `# CHOICE`.

Status: research, unvalidated.

## 1. What is implemented

| Paper element | Where | Notes |
|---|---|---|
| W2 geometry of 1-D distributions via quantile functions | `crates/core/src/w2.rs` (reference: `tests/python/reference/w2.py`) | quantile functions on a midpoint grid |
| Transport maps `T_{a,b} = F_b^{-1} ∘ F_a` and inverses | same | piecewise-linear, see A6 |
| Local Fréchet regression, eq. (3) | `smoother.rs` (reference: `smoother.py`) | local-linear weights, then L2 projection onto monotone functions (PAVA) |
| Partial transport residuals, eq. (10) | `adopt.rs` (reference: `adopt.py`) | see A7 |
| Transport centring with the Fréchet mean, eq. (11) | same | pointwise mean of quantile functions |
| Transport backfitting, Algorithm 1 (Gauss–Seidel) | same | stopping rule A4 |
| Log-Cholesky metric (Lin 2019), Example 2 | `logchol.rs` (reference: `logchol.py`) | see A12 |
| Global Fréchet regression (GF), Fréchet mean (FM) | `estimators.rs` (reference: `estimators.py`) | comparators in the paper |
| ADR (Han et al., 2020) | **not implemented** | see A11 |
| Bandwidth CV, fit/predict dispatch, Fréchet R² | `estimators.rs` (reference: `estimators.py`) | |
| Simulation designs of Sections 5.1 and 5.2 | `benchmarks/adopt_paper/dgp.py` | |

All numerics run in Rust (`crates/core`). Every routine also has a test-only pure-numpy
reference under `tests/python/reference/`. `tests/python/test_parity.py` and
`test_estimators.py` check agreement to `1e-9`/`1e-10`, and observed differences are around `1e-14`.

**Complexity.** Supplement C of the paper gives `O(n²p + np²)` per backfitting cycle, which becomes
`O(n²pq)` for quantile functions on `q` points and `O(n²pm²)` for `m × m` SPD matrices. Our
implementation has the same order. Measured: a fit with p = 3, q = 100 takes 12 ms at n = 100,
45 ms at n = 300 and 299 ms at n = 1000 (Rust, single thread), which is quadratic. ADOPT is therefore
kept separate from the linear-time TRACE core (see `BENCHMARKS.md`).

## 2. Ambiguities and the choices we made

| # | Topic | What the paper fixes | Our choice | Possible impact |
|---|---|---|---|---|
| A1 | Number of Monte Carlo replicates, Table 1 | B = 200 is stated for Table 2. We did not find B stated for Table 1. | B = 200 for both tables | SE only |
| A2 | Quantile grid | Distributions handled as quantile functions. Grid size and placement are not given. | q = 100 midpoints `u_k = (k − 0.5)/q`; W2² = mean squared difference on the grid | small |
| A3 | MISE integral over `[0,1]³`, eq. (12) | integral over the unit cube | 10³ midpoint grid, ×1/B average over replicates | small |
| A4 | Stopping rule (Algorithm 1, step 8) | stop when `(1/n) Σ_i d²(Ŷ_i^t, Ŷ_i^{t−1}) < ε`; ε not given | stop when `mean_i d²(Ŷ_i^t, Ŷ_i^{t−1}) ≤ max(tol · mean_i d²(Y_i, µ̂), 1e-30)`, tol = 1e-10, at most 200 cycles | negligible: converges well before |
| A5 | Kernel and bandwidth | kernel `K` a symmetric density (assumption A1); "h is selected by using 5-fold cross validation". Kernel, candidate grid, per-predictor vs common bandwidth and CV loss are not given | Gaussian kernel. Bandwidth `h_j = c · range(X_j)` with a common fraction `c` chosen by 5-fold CV (folds shuffled with SplitMix64, seed = replicate index) from 10 log-spaced fractions in [0.03, 0.6], minimising held-out mean `d²` | **likely the main source of differences in Table 1 Case II** |
| A6 | Transport maps between discretised quantile functions | `T = F_b^{-1} ∘ F_a` | piecewise-linear through knots `(Q_a(u_k), Q_b(u_k))`. Tied `a` knots are collapsed by averaging `b`. Outside the knot range the map is extended as a translation, so it stays increasing and invertible. Inverse = swap knots | small |
| A7 | Evaluating eq. (10) off the grid | composition of maps applied to `µ̂` | applied exactly as written: inverses of `T_{j−1}…T_1`, then `T_{µ̂,Y_i}`, then inverses of `T_p…T_{j+1}`. This telescopes exactly for j = 1 and for pure translations. For j > 1 with non-translation maps there is an `O(1/q²)` interpolation error (tested: it shrinks as q grows) | small |
| A8 | Local-linear weights when the local design is degenerate | not discussed | if `1 − u1²/(u0 u2) < 1e-10`, use Nadaraya–Watson weights, then the nearest neighbour | only at tiny bandwidths |
| A9 | Projection onto valid quantile functions | the minimiser over the Wasserstein space | L2 projection on the grid via PAVA. No support bounds (optional `bounds=` exists, unused in the reproduction) | small |
| A10 | Responses observed | fully observed distributions | responses observed exactly on the grid (no sampling step from the distributions) | lower noise than a sampling design would have |
| A11 | ADR comparator | Han et al. (2020) | not re-implemented. We added our own `L2AdditiveQuantileRegressor` (backfitting on quantile vectors, then PAVA) as an extra additive reference, clearly labelled | — |
| A12 | Log-Cholesky geometry | Lin (2019) metric | Coordinates = (strict lower part of `L`, log of diagonal of `L`). This space is flat, so transports are translations in coordinates, and ADOPT reduces exactly to classical additive backfitting with the same smoother. GF is a weighted least-squares fit in the same coordinates | none (exact) |
| A13 | True maps: centring | `T_j(x_j) = T_{E⊕ g_j(X_j), g_j(x_j)}` | `E⊕ g_j(X_j)` computed with `X_j ~ Unif(0,1)` (the marginal of `Φ(V_j)`): Gauss–Legendre with 128 nodes in coordinates (SPD, W2 Case II), or a fine quantile grid (20001 points) for the Beta family | negligible |
| A14 | Case I (W2) Beta family composition | `Y_x = [⊕_j T_j(x_j)](U)`, `U = Unif(0,1)` | applied in the order `T_1`, then `T_2`, then `T_3`, starting from the Unif(0,1) quantile function | the order matters for non-commuting maps; this is the paper's convention |
| A15 | SPD design | as transcribed in `benchmarks/adopt_paper/dgp.py` (Section 5.2) | factor entries `exp(−|k−l|/a)·f(x)` for k ≥ l. The diagonal (`|k−l| = 0`) becomes `log f(x)` in coordinates. Noise `N(0, 0.01²)` on the strict lower part of `L` after the transports. MISE uses `d²` = squared Euclidean distance of log-Cholesky coordinates (no 1/m normalisation) | **see Section 4: levels do not match** |

## 3. Results

Run: B = 200 replicates per setting, fixed seeds, Rust backend, 845 s wall time on 7 worker processes.
Paper values are copied verbatim from the PMLR PDF (Tables 1 and 2) and labelled "paper". Entries are mean (standard error).
*L2-additive is our extra comparator (not in the paper). ADR was not run (A11).
Full per-replicate data: `benchmarks/results/adopt_reproduction_reps.csv`. Generated copy: `docs/ADOPT_RESULTS.md`.

### Table 1: distributional responses (W2), MISE ×10⁻³

| setting | paper ADOPT | ours ADOPT | paper GF | ours GF | paper ADR | ours ADR | paper FM | ours FM | ours L2-additive* |
|---|---|---|---|---|---|---|---|---|---|
| I/100 | 0.317 (0.010) | 0.391 (0.014) | 0.591 (0.008) | 0.740 (0.011) | 0.761 (0.008) | not run | 11.019 (0.068) | 11.691 (0.010) | 0.640 (0.012) |
| I/300 | 0.181 (0.004) | 0.179 (0.005) | 0.499 (0.003) | 0.576 (0.005) | 0.586 (0.004) | not run | 11.119 (0.040) | 11.602 (0.003) | 0.436 (0.006) |
| II/100 | 1.685 (0.057) | 0.503 (0.010) | 6.636 (0.057) | 6.362 (0.022) | 80.411 (0.702) | not run | 207.205 (1.681) | 204.700 (0.219) | 0.579 (0.014) |
| II/300 | 1.121 (0.041) | 0.178 (0.003) | 6.701 (0.037) | 5.975 (0.008) | 81.951 (0.455) | not run | 211.297 (1.080) | 203.277 (0.064) | 0.194 (0.004) |

### Table 2: SPD responses (log-Cholesky), MISE ×10⁻²

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

Bandwidth fractions selected by CV for ADOPT (counts over 200 replicates) are stored in
`benchmarks/results/adopt_reproduction.json`, field `bw_fraction_counts`.

## 4. What does not reproduce, and likely causes

**Summary.** The qualitative claims reproduce in every setting: ADOPT has the lowest MISE, ahead of GF and then FM, and the ADOPT MISE falls as n grows. Numerically, Table 1 Case I and the Table 1 baselines are close. Our ADOPT in Table 1 Case II and all levels of Table 2 do not match.

| Setting | Match? | Observation | Likely cause (unconfirmed) |
|---|---|---|---|
| T1 Case I, ADOPT | n = 300: yes (0.179 vs 0.181). n = 100: no, ours about 23% higher | | bandwidth selection at small n (A5). CV picks wide fractions (mostly 0.22–0.31) |
| T1 Case I, GF / FM | close, but outside the SEs (GF +15–25%, FM +5%) | FM does not depend on any tuning, so a 5% FM gap means the target distributions or the MISE approximation differ slightly | A2/A3 (grid and integration), A13 (centring of the Beta maps), A10 (see the SE remark below) |
| T1 Case II, GF / FM | yes, within 1–11% | | |
| T1 Case II, ADOPT | **no**: ours 3–6× lower (0.50 vs 1.685, 0.18 vs 1.121) | our L2-additive comparator is just as low, so this is not a quirk of our ADOPT code | the paper's bandwidth/kernel choices (A5) or extra response noise in their design (A10) |
| T2, FM | only Case I, m = 10 matches | under our reading, FM grows about in proportion to m (Case I: 134 to 272, Case II: 206 to 434), while the paper's grows about 1.4× (135 to 193, 97 to 143). Paper Case II, m = 10 is half of ours | a different reading of the factor design `L_j(x)`, or a normalised distance (A15) |
| T2, ADOPT / GF | **no**: ours 10–50× lower | our errors with σ = 0.01 are dominated by smoothing bias, because the noise is tiny | noise level or placement (A15); see the diagnostic below |

**Standard-error remark.** The paper's FM standard errors are several times ours (Table 1: 0.068 vs 0.010 at I/100; 1.681 vs 0.219 at II/100). FM depends only on the sample mean of the responses. Larger FM standard errors therefore suggest that the paper's responses vary more between replicates than ours do. One example would be distributions observed through finite samples instead of exactly on the grid (A10). Extra variability of that kind would also raise the paper's ADOPT and GF errors relative to ours.

**Noise diagnostic for Table 2** (`benchmarks/adopt_paper/sensitivity_spd.py`, 20 replicates, MISE ×10⁻²; *not* used in the table above). Raising σ moves ADOPT towards the paper's values for m = 10 (σ = 0.1 gives 2.97 and 4.09 vs the paper's 2.769 and 4.146 at n = 100), but not consistently across n and m. GF stays 1.5–5× below the paper, and FM is unaffected. A different noise level alone therefore does not explain Table 2. The SPD design itself must differ from our reading in some way we could not identify from the text.

| setting | σ | ours ADOPT | ours GF | ours FM |
|---|---|---|---|---|
| I/10/100 | 0.01 | 0.059 | 0.944 | 134.18 |
| I/10/100 | 0.05 | 0.898 | 1.524 | 134.29 |
| I/10/100 | 0.1 | 2.973 | 3.336 | 134.63 |
| I/10/300 | 0.01 | 0.021 | 0.898 | 133.13 |
| I/10/300 | 0.05 | 0.325 | 1.081 | 133.17 |
| I/10/300 | 0.1 | 1.069 | 1.655 | 133.28 |
| I/20/100 | 0.01 | 0.219 | 1.942 | 273.45 |
| I/20/100 | 0.05 | 3.332 | 4.402 | 273.89 |
| I/20/100 | 0.1 | 11.610 | 12.091 | 275.28 |
| I/20/300 | 0.01 | 0.082 | 1.810 | 269.18 |
| I/20/300 | 0.05 | 1.300 | 2.621 | 269.33 |
| I/20/300 | 0.1 | 4.266 | 5.156 | 269.81 |
| II/10/100 | 0.01 | 0.090 | 4.639 | 207.09 |
| II/10/100 | 0.05 | 1.295 | 5.232 | 207.18 |
| II/10/100 | 0.1 | 4.088 | 7.075 | 207.51 |
| II/10/300 | 0.01 | 0.031 | 4.471 | 205.09 |
| II/10/300 | 0.05 | 0.482 | 4.659 | 205.13 |
| II/10/300 | 0.1 | 1.571 | 5.242 | 205.24 |
| II/20/100 | 0.01 | 0.341 | 10.200 | 435.33 |
| II/20/100 | 0.05 | 4.755 | 12.665 | 435.78 |
| II/20/100 | 0.1 | 14.809 | 20.395 | 437.21 |
| II/20/300 | 0.01 | 0.116 | 9.700 | 432.94 |
| II/20/300 | 0.05 | 1.758 | 10.486 | 433.09 |
| II/20/300 | 0.1 | 5.604 | 12.941 | 433.56 |

**What we did not do.** We did not change any setting to move our numbers towards the paper's. The reproduction table uses the paper's stated design under the readings listed in Section 2.

## 5. How to re-run

```bash
maturin develop --release
python benchmarks/adopt_paper/reproduce.py --reps 200 --workers 6   # writes benchmarks/results/
python docs/build_docs.py                                         # tables, figures, learning page
```

Seeds: replicate `r` of a setting uses `numpy.random.SeedSequence([table, case, n, m, r])`, where
table ∈ {1, 2}, case ∈ {1 (I), 2 (II)}, and m = 0 for Table 1. Per-replicate values are stored in
`benchmarks/results/adopt_reproduction_reps.csv`.
