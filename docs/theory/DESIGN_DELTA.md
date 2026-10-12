# DESIGN_DELTA: Rust-core implications of THEORY.md

Scope: what to change in `crates/core` and the bindings once these modifications are built. This document contains no code.
Section references (§) point to THEORY.md and check IDs to `verify/RESULTS.txt`.
All new simulation pass criteria in §7 are fixed here, **before** any results exist.

## 1. Parameterisation

| Item | v0 (today) | Proposed |
|---|---|---|
| u-grid | midpoint `u_k=(k+0.5)/K`, 19 levels | Gauss–Legendre nodes `n_u ≥ K_fit` on (0,1) with weights `w_u`. Optionally also report on a dense display grid. |
| Fitting basis | `1, z, He3 (,He2)`, RMS-normalised on the grid | Orthonormal shifted Legendre `e_k = √(2k+1) P*_k`, k < K_fit. Coefficients are `a_k` (Parseval with unit weights, §1.1). |
| K | 3 channels | `K_fit` (default 6–8) for fitting, `K_report` (default 4) for attribution, plus the reported truncation residual (§1.2, `v1:1.17`). |
| Reporting basis | same as fitting | A separate orthonormal basis chosen by the user: Legendre (default), Hermite-in-z, or reference-adapted (Gram–Schmidt with e_1 = centred reference). Report coefficients = L² projections computed with GL quadrature on the fitted curve (§1.5). |
| Storage per feature | `B_j × M` shape values | `B_j × K_fit` coefficients `θ_{j b k}`; intercept `θ_{0k}`. Centring convention and evaluation law stored with the model. |

Notes.
* GL nodes make the discrete Gram matrix exactly diagonal for polynomial channels (`v1:1.13`). They are not exact for `z`/He3 channels, so a Hermite-in-z *reporting* basis needs a high-order quadrature (≥ 1000 GL nodes, or a tanh-sinh/erf substitution) for the projection step. That cost is per model, not per row.
* Store the conversion matrices: `a ↔ c ↔ λ` (diagonal) and fitting → reporting basis (a `K_fit × K_report` matrix computed once).

## 2. Fitting objective

* **Pinball on GL nodes:** `L(θ) = Σ_i Σ_u w_u ρ_u(y_i − Σ_k a_k(x_i) e_k(u))`. This is a quadrature of the CRPS-type integrated pinball.
* **Steps.** Replace Jacobi gradient steps with Newton / preconditioned steps per feature block, using the curvature `H = Σ_u w_u e(u)e(u)ᵀ f̂(Q̂(u|x))`, which is K_fit × K_fit per bin. `f̂` comes from the current fit's quantile spacing, `1/Q̂'(u)`, clipped. Alternatively, run to convergence. Rationale: under the pinball metric, early-stopped gradient steps leak scale into shape channels (`v1:1.14`, `v1:1.15`: peak leak 0.408; zero with Newton).
  * Cost per round: O(n·n_u·K_fit) for gradients, plus O(Σ_j B_j·K_fit³) for the block solves. That is negligible for K_fit ≤ 8.
* **Early stopping:** if kept, apply it to the whole coefficient vector of a bin (Newton direction), not channel by channel.
* **Fréchet mode (§3):** if rows carry samples rather than scalars, the objective switches to per-channel weighted least squares on Hosking unbiased sample L-moment targets `t_{ik} = √(2k+1)·ℓ_{k+1}` in `a` units.
  * Weights are `1/(τ_k² + v_k/m_i)`, estimated from a first pass (`v3:3.5`).
  * Without monotonicity constraints there is one independent LS per channel. The `1/(2k+1)` weights matter only if constraints couple the channels.
  * Plug-in empirical-quantile targets must not be used for fixed small m (`v3:3.2`, `v3:3.3`).

## 3. Monotonicity constraints

Option A (default, cheap): **post-hoc weighted PAV** on GL nodes with GL weights (§1.4).
* Report `d = ‖Q̃ − ΠQ̃‖` per row (or its average) and the monotonisation adjustment `D_Π² − D²` with the bound `[−2D(d_A+d_B), 0]`.
* Attribution is computed pre-projection (exact) and the adjustment is shown as its own line.
* Cost: O(n_u) per row.

Option B (in-fit): **Bernstein linear constraints.** Bernstein coefficients of `Q'` at degree `K_fit−2+e` (e = 4 default) must be ≥ 0 for each bin combination. That is exponential in p, so it is only feasible as:
* (i) per-feature constraints under a sufficient decomposition (e.g. require the intercept to be monotone with a margin, and bound each shape function's derivative); or
* (ii) constraints enforced only at observed rows.

Both are sufficient and conservative (`v1:1.11`: 759/817 certified at e = 16). Exact SOS/SDP (Markov–Lukács) is not recommended for the core: it needs an SDP solver per row.

Hard fact to surface in docs: with `K = 3`, laws with |L-skew| > 1/5 cannot be represented monotonically (`v1:1.9`, `v1:1.11b`). Default `K_fit ≥ 4`.

## 4. Attribution and shift APIs

* `attribute(model, X_eval)`: per row, per channel, per player Shapley values. In orthonormal units this is `φ_{jk} = a_{jk}·Σ_i a_{ik}` (§1.3; equivalent to `Δc_{jk}·Σ_iΔc_{ik}/(2k+1)` in expansion units).
  * Returns the per-row array `n × (p+1) × K_report`, plus the truncation residual and the monotonisation adjustment.
  * Cost: O(n p K).
* `shift_conditional(model_A, model_B, X_eval)`: exact channel × player decomposition (§2.1). The evaluation law is a required argument, and the centring is redone under it.
* `shift_marginal(model_A, model_B, X_A, X_B)`: mixture quantile functions by quantile averaging on a fine u-grid. Returns the total, composition, structure and cross terms, and both path orders plus the two-player Shapley (§2.3). It flags bins populated in X_B but empty in A's training data (overlap failure).
  * Cost: O(n·n_u + n_grid·log) for the mixture inversion.
* Noise floors (§2.4–2.5):
  * `cross_fit = true` splits each period in two and reports `⟨Δ̂_1, Δ̂_2⟩`-based energies, unclipped.
  * `permutation_test(stat, R=199, strata=bins|none, block=None|len)`: global or within-cell permutation; block permutation for time-ordered rows (approximate).
  * `bootstrap(…, block=…)` only for magnitude intervals of non-null shifts.
  * Per-feature floors are labelled "approximate" in the output.

## 5. DRO mode

* Cost specification: per-feature `κ_j` per bin step (default: bin width in standardised x units), optional `κ_y`, and ball type `W1 | W2 | Winf`.
* **W1/W2, covariate or joint:** the exact dual (Theorem 4.1 / Prop 4.6–4.7). Per λ, the separable inner max costs O(n Σ_j B_j). λ is found by a breakpoint search: O(n Σ_j B_j log) if done by sorting candidate slopes, or bisection on the subgradient.
* Fitting: either
  * (a) the LP (epigraph form, O(n Σ_j B_j) constraints, a HiGHS-class solver via FFI); or
  * (b) subgradient / bundle methods on the convex robust risk (`v4:4.6`).

  The **max-jump bound** `emp + ε ρ̄ max_j G_j/κ_j` can be offered as a cheap surrogate penalty. It must be labelled an upper bound (`v4:4.2`, ratio 0.22–1.00).
* **W∞:** exact closed form via per-feature window min/max (O(npB) with sliding-window min/max). A bin-mass-weighted TV penalty is the corresponding upper bound/surrogate (`v4:4.5`).
* **Outcome-only W1** adds the constant `ε ρ̄/κ_y`. Do not expose it as a regulariser (`v4:4.4`). Expose W2 outcome-only only as an evaluated quantity (exact via 1-D dual, `v4:4.10`).
* Do not calibrate ε from per-feature marginal W1/W2 distances; they are lower bounds on the joint distance (`v4:4.7`). Provide a joint W1 estimate between periods on the bin grid (an LP over occupied cells, or a sliced/greedy upper bound) instead.
* Raw-x costs: supported only via exact point-to-bin distances. Document that there is no Lipschitz-type surrogate (`v4:4.3`).

## 6. Complexity summary

| Operation | Cost |
|---|---|
| Fit, one Newton round | O(n·n_u·K_fit + Σ_j B_j K_fit³) |
| PAV per row | O(n_u) |
| Attribution | O(n p K_report) |
| Marginal shift | O(n n_u + G log G), G = display grid |
| DRO evaluation, one λ | O(n Σ_j B_j) |
| DRO λ-search | O(log(1/tol)) evaluations, or exact breakpoints |
| DRO LP fit | O(n Σ_j B_j) constraints and variables |

## 7. New simulation tests (pass criteria fixed before results)

All tests use fixed seeds and are run with the defaults in §1–2 unless stated otherwise. "Oracle" means the projection of the *true generator* onto the stated reporting basis, computed by ≥ 2000-node GL quadrature (or exact formula).

1. **T1 Parseval / attribution identity.** Any fitted model, 1000 random rows: `|Σ_{j,k} φ_{jk} − W2²(pre-proj)| ≤ 1e-10·max(1, W2²)`, and `W2²(pre-proj)` by GL equals `Σ_k Δa_k²` to 1e-10 relative.
2. **T2 GL orthogonality.** Discrete Gram of `e_0..e_{K_fit−1}` on the fitting nodes: max |G − I| ≤ 1e-12.
3. **T3 S1-replacement, tail identification in a Hermite-in-z reporting basis.** Use S1's generator and sizes (n = 2000 per period, seed 11) with K_fit = 8 and reporting basis {1, z, He2, He3}/√m!.
   * Pass: x_shape's estimated energy in the He3 channel ≥ 3× the largest of (noise features, x_scale, x_loc) in that channel;
   * and the estimated He3 coefficient slope is within 30% of the oracle (0.12 in He3 units, i.e. 0.12·√6 in orthonormal units).
4. **T4 S1 in plain Legendre reporting basis (expected-negative control).** Same data, Legendre K_report = 4.
   * Pass: every feature's estimated channel-3 energy is within 2 Monte-Carlo SEs (200 reps) **or** within 25% relative of its oracle projection (oracle ratio x_shape/x_scale = 0.753).
   * The test checks agreement with the oracle, not a 3× ratio.
5. **T5 Legendre-native tail feature.** Generator `Q(u|x) = 1.2x_loc + (0.5+x_scale)·e_1(u) + 0.15·x_shape·e_3(u)`, plus two pure-noise features, with `Y = Q(U|x)` and U uniform, x ~ U[0,1]³ (sizes as S1).
   * Monotone for all x: `Q' ≥ 2√3·0.5 − 3·0.15·√7 = 1.732 − 1.191 > 0`, because `min P*_3' = −3` (at u = 1/2) and `P*_1' = 2`.
   * Pass: x_shape channel-3 energy ≥ 3× every other feature's channel-3 energy;
   * x_scale's channel-3 energy ≤ 10% of x_shape's;
   * per-feature energies are within 25% of the oracle.
6. **T6 Grid leakage.** Pure-scale generator `Q = (0.5+x)z`, fitted (a) on a 19-point midpoint grid with v0 steps and (b) on GL nodes with Newton steps.
   * Pass: (b) shape-channel energy ≤ 0.05 × (a)'s, and (b) ≤ permutation noise floor at α = 0.05.
7. **T7 Monotonicity.**
   * Exponential outcome, intercept-only: K_fit = 3 must be flagged non-representable (|τ_3| > 1/5).
   * K_fit = 4 must produce a monotone fit without PAV changes > 1e-8.
   * Option B constraints must yield zero monotonicity violations on a 10⁴ grid for all rows.
8. **T8 PAV bound.** For 500 fitted rows: `D_Π ≤ D` and `D_Π ≥ D − d_A − d_B` (tolerance 1e-10), and the reported adjustment lies in `[−2D(d_A+d_B), 0]`.
9. **T9 Shift decomposition.** Two-period generator with a planted covariate-only shift (P_B ≠ P_A, same conditional law).
   * Pass: the structure term is within its permutation noise floor (stratified by bins, α = 0.05, rejection ≤ 0.08 over 200 reps);
   * composition + structure + cross equals the total to 1e-8.
10. **T10 Permutation level.** No-shift generator, 300 reps, R = 199: rejection at α = 0.1 lies in [0.048, 0.152].
11. **T11 Fréchet consistency.** Rows of size m_i ∈ {5, 50, 500} (thirds), exponential rows with scale linear in x.
    * Pass: RMSE(n = 3200)/RMSE(n = 200) ∈ [0.15, 0.40] with Hosking targets;
    * GLS-weighted slope variance ≤ 1.05 × min(equal, ∝m).
12. **T12 DRO exactness.** For n = 50, p = 3, B = 6, random models:
    * the exact dual equals a brute-force LP to 1e-8 for ε ∈ {0.01, 0.1, 0.5}, under W1, W2 and joint costs;
    * the max-jump surrogate is ≥ exact in all cases.
13. **T13 DRO usefulness (descriptive with pre-set criterion).** Train on period A; evaluate pinball on a period B generated by moving each row by one bin in feature 1 with probability 0.2 (a W1 move of 0.2·κ_1).
    * Pass: the robust fit with ε = 0.2κ_1 has B-period pinball ≤ the non-robust fit's in ≥ 60% of 100 reps.
    * The certificate `E_B ℓ ≤ R(ε)` for the robust fit holds in ≥ 95% of reps. This needs B's empirical law to lie in the ball; check it by computing the realised W1 on the grid.
14. **T14 Outcome-only W1 is a constant.** For 20 random fits: `R(ε) − emp = ε ρ̄/κ_y` to 1e-9.

## 8. Things not to build (from the theory)

* No Lipschitz/TV penalty presented as "the W1-DRO objective". The exact LP or a labelled upper bound only.
* No per-feature split of marginal shift presented as exact.
* No clipping of negative attributions or cross-fit energies.
* No ε calibration from per-feature marginal distances.
