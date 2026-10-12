# TRACE: mathematical backbone for the four approved modifications

Status: proofs plus numerical verification. This is not package code.
Repository examined: `jgridifier/traceot` at commit d45500a ("TRACE v0"); read-only clone.
Every numbered claim carries a check ID (`v1:1.4`, etc.). The scripts are in `verify/` and the results in `verify/RESULTS.txt`.
Labels used throughout:
**Theorem/Proposition** means proved here (or a standard result with a citation).
**Numerical** means supported by computation only.
**UNVERIFIED** means a citation detail or claim I could not confirm against a primary source.

---

## 0. Setting, notation, and the conventions every result depends on

* `u ∈ (0,1)` is the quantile level. `Q(u|x)` is the conditional quantile function of `Y` given covariates `x = (x_1,…,x_p)`.
* Each feature `x_j` is discretised into ordinal bins `b_j ∈ {0,…,B_j−1}`.
* Inner product and norm: `⟨g,h⟩ = ∫_0^1 g(u)h(u) du` and `‖g‖² = ⟨g,g⟩`.
* For two laws on ℝ with finite second moments, `W2²(F,G) = ‖Q_F − Q_G‖²`, the one-dimensional quantile representation. All W2 statements below rest on this identity. It applies only to scalar `Y`.
* **Shifted Legendre polynomials.** `P*_k(u) = P_k(2u−1)`, so `P*_k(1) = 1` and `P*_k(u) = Σ_r p*_{k,r} u^r` with `p*_{k,r} = (−1)^{k−r} C(k,r) C(k+r,r)`. This is the convention of Hosking (1990, JRSS-B 52(1):105–124, doi 10.1111/j.2517-6161.1990.tb01775.x).
* **Orthogonality.** `⟨P*_j, P*_k⟩ = δ_{jk}/(2k+1)` (`v1:1.1`).
* **L-moments.** `λ_{r} = ∫ Q(u) P*_{r−1}(u) du` for r ≥ 1. This equals Hosking's order-statistic definition (`v1:1.2`, checked for the uniform, exponential, normal and Gumbel laws).
* **Three coordinate systems for the same function** `Q = Σ_k c_k P*_k`:
  * expansion coefficients `c_k = (2k+1) λ_{k+1}` (`v1:1.3`);
  * L-moments `λ_{k+1} = c_k/(2k+1)`;
  * orthonormal coefficients `a_k = √(2k+1) λ_{k+1} = c_k/√(2k+1)`. These are the coefficients on the orthonormal basis `e_k = √(2k+1) P*_k`.
* **TRACE v0 (what exists today, read from `crates/core/src/basis.rs` and `docs/TRACE_DESIGN.md`).**
  * Channels are `1`, `z`, `He3(z) = z³−3z` (and `He2` when M = 4), with `z = Φ^{-1}(u)`.
  * They are evaluated on the midpoint grid `u_k = (k+0.5)/K` and RMS-normalised on that grid.
  * The fit is boosting on pinball loss with Jacobi-type steps (lr 1.0, 40 rounds), followed by PAV in `u`.

---

## 1. The shifted-Legendre / L-moment basis

### 1.1 Parseval and the correct W2 formula

*Intuition.* The shifted Legendre polynomials are perpendicular in the space of quantile functions, like the axes of a coordinate system. The squared Wasserstein-2 distance between two distributions is a squared length in that space. So it splits into one term per axis, like Pythagoras. The only subtlety is bookkeeping: the axes are not unit length, so the weight of each term depends on which coefficients you write down.

**Assumptions.** `Q_A, Q_B ∈ L²(0,1)`, i.e. both laws have finite second moment.

**Theorem 1.1 (Parseval).** With `Δc_k = c_k^A − c_k^B` and `Δλ_r` defined the same way,
`W2²(F_A,F_B) = Σ_{k≥0} Δc_k²/(2k+1) = Σ_{k≥0} (2k+1) Δλ_{k+1}² = Σ_{k≥0} Δa_k²`.

*Proof.* `{√(2k+1)P*_k}` is an orthonormal basis of L²(0,1): orthonormality is the identity above, and completeness follows from Weierstrass density of polynomials in C[0,1] together with density of C[0,1] in L². Parseval's identity for orthonormal bases applied to `Q_A − Q_B` gives `‖Q_A−Q_B‖² = Σ_k ⟨Q_A−Q_B, √(2k+1)P*_k⟩² = Σ_k (2k+1) Δλ_{k+1}²`. Substituting `λ_{k+1} = c_k/(2k+1)` gives the other forms. ∎ (`v1:1.4`, relative error 1.8e-16.)

**Correction to a stated claim.** "W2² = Σ (2k+1) Δb_k²" is correct **if and only if `b_k` are the L-moments `λ_{k+1}`**. With `b_k` the expansion coefficients `c_k` the weight must be `1/(2k+1)`. Using `(2k+1)` with `c_k` is wrong: on the test pair it gives 1.5103 against the true 1.2388 (`v1:1.5`). The orthonormal coefficients `a_k` avoid the issue (unit weights). I recommend `a_k` as the internal parameterisation.

### 1.2 Truncation residual

*Intuition.* If you keep only the first K channels, you lose whatever part of the difference lives on the higher axes. Because the axes are perpendicular, that loss is exactly the squared length of the discarded part. It never makes the total bigger, and it shrinks as K grows. How fast it shrinks depends on how smooth the quantile difference is, so there is no universal rate.

**Proposition 1.2.** Let `Π_K` denote the orthogonal projection onto `span{P*_0,…,P*_{K−1}}` and `R_K = (I−Π_K)(Q_A−Q_B)`. Then:

`W2²(F_A,F_B) = Σ_{k<K} (2k+1) Δλ_{k+1}² + ‖R_K‖²`, with `‖R_K‖² = Σ_{k≥K} (2k+1)Δλ_{k+1}²`.

The residual is non-negative, non-increasing in K, and tends to 0 as K → ∞.

*Proof.* Pythagoras for the orthogonal decomposition `Π_K ⊕ (I−Π_K)`, plus Theorem 1.1 for the tail sum. ∎ (`v1:1.6`: residual 0.194, 0.0424, 0.0185, 0.0062 at K = 1, 3, 5, 10.)

**What fails.**
* There is no rate without smoothness assumptions.
* For laws with unbounded support, `Q` is unbounded at 0 or 1, so the Legendre coefficients decay only polynomially; the residual can be material at K = 3–4.
* The residual is a population quantity. Estimating it from data needs the higher sample L-moments, which are noisy.

### 1.3 Exact additive per-feature attribution (Shapley of a squared sum)

*Intuition.* In an additive model, the change in channel k for one row is a sum of contributions, one per feature (plus the intercept). The energy in that channel is the square of the sum, which contains cross terms between features. The Shapley value splits each cross term equally between the two features involved. Every feature therefore gets "its own contribution times the total". The shares add up exactly, but a feature that pushes against the overall change gets a negative share.

**Setting.** Fix a row `x` and channel `k`. Let `a_{jk}(x)` be the change in feature j's contribution, in L-moment units, for players `j = 0` (intercept) through `p`. Let `A_k(x) = Σ_j a_{jk}(x)`. The channel energy is `(2k+1) A_k(x)²`; in expansion units it is `A_k^c(x)²/(2k+1)`.

**Theorem 1.3.** For the cooperative game `v(S) = (Σ_{j∈S} a_j)² · w` (w > 0 fixed), the Shapley value is `φ_j = w · a_j · Σ_i a_i`. It is efficient, i.e. `Σ_j φ_j = v(N)`. In particular, with `a` in expansion units and `w = 1/(2k+1)`, `φ_{jk} = Δf_{jk}·(Σ_i Δf_{ik})/(2k+1)` exactly as stated in the task.

*Proof.* Expand `v(S) = w Σ_{i∈S} a_i² + w Σ_{i≠l; i,l∈S} a_i a_l`. The Shapley value is linear in the game.
* The game `S ↦ a_i² 1[i∈S]` gives all its value to i.
* The game `S ↦ 2 a_i a_l 1[{i,l}⊆S]` is a scaled unanimity game on `{i,l}`, so i and l each get `a_i a_l`.
* Summing gives `φ_j = w(a_j² + Σ_{l≠j} a_j a_l) = w a_j Σ_l a_l`.

Efficiency is `Σ_j w a_j Σ_l a_l = w(Σ a)²`. ∎ (`v1:1.7`: brute force over all coalitions, 50 games, p ≤ 7; negative values occur.)

**Aggregate attribution.** Average over rows with a stated evaluation law `x ~ P`:
`E φ_{jk} = w [ E a_{jk}² + Σ_{i≠j} E a_{jk}a_{ik} ]`. The shares sum exactly to `E_x` of the channel energy.

**Proposition 1.4 (functional-ANOVA purification).** Assume:
1. the features are independent under P;
2. every shape-function change is centred, `E_P a_{jk}(x_j) = 0` for j ≥ 1, so the intercept carries the mean.

Then the cross terms vanish, and `E φ_{jk} = w Var_P(a_{jk})` for j ≥ 1 and `E φ_{0k} = w a_{0k}²`. Under (1)–(2) the additive components are the purified functional-ANOVA main effects in the sense of Lengerich, Tan, Chang, Hooker & Caruana (AISTATS 2020, PMLR 108:2402–2412).

*Proof.* Under independence and centring, `E a_{jk} a_{ik} = E a_{jk} E a_{ik} = 0` for i ≠ j ≥ 1. Also `E a_{0k} a_{jk} = a_{0k} E a_{jk} = 0`. ∎ (`v1:1.8`; with dependence `v1:1.8b` shows a non-zero covariance term, while the sum is still exact.)

**What fails.**
1. Dependent features: the covariance terms remain and are split equally. The equal split is a Shapley convention, not a causal statement.
2. The intercept is a player. Changing the centring convention (centring under P_A vs P_B vs a pooled law) moves mass between the intercept and the features. The evaluation law P must be stated with every attribution.
3. Negative shares are legitimate and must not be clipped.
4. All of this is exact **before** the isotonic projection (§1.4).

### 1.4 Monotonicity of truncated expansions and the isotonic projection

*Intuition.* A quantile function must be nondecreasing, but an arbitrary polynomial is not. With very few channels the condition is a simple inequality: with three channels, the L-skewness must lie between −1/5 and 1/5. So a strongly skewed law such as the exponential cannot be represented monotonically with three channels at all. For more channels the exact condition is a semidefinite constraint, and a simpler linear condition is safe but slightly conservative. If you instead fix monotonicity after the fact with PAV, the result is no longer a sum of per-feature pieces. The additive split then holds only approximately, with an error you can bound and should report.

**Proposition 1.5 (K = 3).** `Q = c_0 + c_1 P*_1 + c_2 P*_2` is nondecreasing on [0,1] iff `|c_2| ≤ c_1/3`. When `λ_2 > 0` this is equivalent to `|τ_3| ≤ 1/5`, where `τ_3 = λ_3/λ_2`.

*Proof.* `Q'(u) = 2c_1 + c_2(12u−6)` is affine, so it is nonnegative on [0,1] iff it is nonnegative at the endpoints: `2c_1 ∓ 6c_2 ≥ 0`. Then `τ_3 = (c_2/5)/(c_1/3) = 3c_2/(5c_1)`. ∎ (`v1:1.9`, 20000/20000.)

*Consequence.* The exponential law has τ_3 = 1/3, so its 3-term truncation is not monotone: min Q' = −2. Its 4-term truncation is monotone: min Q' = 0.536 (`v1:1.11b`).

**Proposition 1.6 (K = 4).** With `c_3` included, `Q' (u)= 2c_1 + c_2(12u−6) + c_3(60u²−60u+12)`. Q is nondecreasing iff all of the following hold:
* `Q'(0) = 2c_1 − 6c_2 + 12c_3 ≥ 0`;
* `Q'(1) = 2c_1 + 6c_2 + 12c_3 ≥ 0`;
* if `c_3 > 0` and `u* = 1/2 − c_2/(10c_3) ∈ (0,1)`, then `Q'(u*) ≥ 0`.

At τ_3 = 0 this is `−1/14 ≤ τ_4 ≤ 2/7`.

*Proof.* Q' is quadratic. Its minimum over [0,1] is at an endpoint, or at the interior vertex when it is convex (`c_3 > 0`). The vertex solves `Q'' = 12c_2 + c_3(120u−60) = 0`. With c_2 = 0, write τ_4 = 3c_3/(7c_1):
* `Q'(0) ≥ 0` ⇔ `c_3 ≥ −c_1/6` ⇔ `τ_4 ≥ −1/14`;
* `Q'(1/2) = 2c_1 − 3c_3 ≥ 0` ⇔ `τ_4 ≤ 2/7`. ∎ (`v1:1.10`, 20000/20000.)

**Proposition 1.7 (general K).**
* *(Exact.)* `Q' ≥ 0` on [0,1] iff `Q'(u) = σ_0(u) + u(1−u) σ_1(u)` (even degree), or `Q' = u σ_0 + (1−u) σ_1` (odd degree), with `σ_i` sums of squares of the appropriate degrees. This is the Markov–Lukács theorem, see Powers & Reznick (Trans. AMS, 2000) for the interval form. Membership is a semidefinite (LMI) constraint on the coefficients.
* *(Sufficient, linear.)* Write `Q'` in the Bernstein basis of degree `K−2+e` (degree elevation e). If all Bernstein coefficients are ≥ 0, then `Q' ≥ 0`, because Bernstein polynomials are nonnegative on [0,1]. The converse fails. For strictly positive Q' it holds after enough degree elevation; this is a classical fact about Bernstein coefficients, not re-proved here, and its rate is not used.
* (`v1:1.11`: no false certificates in 5000 random cases. Of 817 monotone cases, 255/618/759 are certified at elevation 0/4/16.)

**Isotonic projection.** Let `C` be the closed convex cone of nondecreasing functions in L²(0,1), and `Π` the metric projection onto it. On a grid, PAV with quadrature weights computes it; with Gauss–Legendre nodes, use the GL weights.

**Proposition 1.8 (projection bounds).** For `Q̃_1, Q̃_2 ∈ L²` with `d_i = ‖Q̃_i − ΠQ̃_i‖`:
1. **Non-expansive:** `‖ΠQ̃_1 − ΠQ̃_2‖ ≤ ‖Q̃_1 − Q̃_2‖`.
2. **Lower bound:** `‖ΠQ̃_1 − ΠQ̃_2‖ ≥ ‖Q̃_1 − Q̃_2‖ − d_1 − d_2`.
3. **Size of the correction:** for absolutely continuous `Q̃`, `d ≤ ‖∫_0^· (Q̃')_−‖ ≤ ∫_0^1 (Q̃')_−`, the total negative variation.
4. **Moreau:** `⟨Q̃−ΠQ̃, ΠQ̃⟩ = 0` and `⟨Q̃−ΠQ̃, g⟩ ≤ 0` for every `g ∈ C`.

*Proof.*
1. Standard for projections onto closed convex sets in Hilbert space.
2. Triangle inequality.
3. `M(u) = Q̃(0) + ∫_0^u (Q̃')_+` is in C, and `Q̃ − M = −∫_0^u (Q̃')_−`. So `d ≤ ‖Q̃ − M‖`, and the sup bound gives the second inequality.
4. Moreau's decomposition for a closed convex cone.

∎ (`v1:1.12`, 300 random pairs.)

**How the projection breaks the additive split, and the bound.** Π is nonlinear: `Π(Σ_j f_j) ≠ Σ_j Π(f_j)`, and the coefficients of `ΠQ̃` are not a sum of per-feature coefficients. Theorems 1.1/1.3 therefore apply exactly to the *pre-projection* curves only. By Prop. 1.8(1)–(2), with `D = ‖Q̃_A − Q̃_B‖` (exactly attributed) and `D_Π` the post-projection distance:
`0 ≤ D² − D_Π² ≤ D² − (D − d_A − d_B)_+² ≤ 2 D (d_A + d_B)`.
Report the pre-projection attribution, plus a single "monotonisation adjustment" term `D_Π² − D²` with this bound. Do not redistribute it across features: no exact rule exists.

### 1.5 Why orthogonality should fix tail identification, and when it will not

*Intuition.* The hope is that a feature acting only on the tails shows up only in the "tail" channels if the channels are perpendicular. That is true only if two other things hold:
* the fitting procedure measures distance the same way the channels are perpendicular;
* the feature's effect points along one of the chosen axes.

TRACE v0's channels were already perpendicular in the Wasserstein sense. What mixed them was a coarse evaluation grid, the pinball loss's own geometry combined with early-stopped boosting, and noise. Worse, in the plain Legendre basis the simulated "scale" feature has a genuine L-kurtosis component. No noise-free fit can then pass the current simulation criterion.

**Finding 1 (corrects the premise "channels are not orthogonal").** In continuous L²(0,1), `{1, z, He3(z)}` with `z = Φ^{-1}(u)` are orthogonal, with `‖He_m‖² = m!`. Hence `W2² = Σ_m m! Δθ_m²` for v0's Hermite channels (`v1:1.13`, continuous ⟨z,He3⟩ = −6.4e-15; `v1:1.16e`). Non-orthogonality in v0 comes from three sources.

1. **The midpoint u-grid.**
   * Measured `corr(z, He3)` is −0.4546 (19 levels), −0.2942 (31) and −0.1008 (101) (`v1:1.13` INFO).
   * Gauss–Legendre nodes with n ≥ K make the discrete Gram matrix of `P*_0..P*_{K−1}` exactly diagonal: the GL rule with n nodes is exact for degree ≤ 2n−1 (`v1:1.13`: max |corr| 7.8e-16).
   * The midpoint rule has O(N⁻²) error for polynomials (Legendre K = 4 max |corr| 1.5e-2, 5.6e-3, 5.2e-4 at N = 19, 31, 101).
   * No polynomial rule is exact for `z`, `He3`, which are unbounded.
2. **The pinball/CRPS metric.**
   * Integrated pinball loss near the truth has curvature `H_{jk} = ∫ e_j(u) e_k(u) f(Q(u)) du`; it weights u by the density at the quantile, so it is not the W2 metric.
   * For a Gaussian outcome: Legendre `corr(P0,P2) = −0.460`, `corr(P1,P3) = −0.528`; v0 Hermite `corr(z,He3) = −0.7746 = −√0.6`.
   * For an exponential outcome: `corr(P0,P1) = −0.577` (`v1:1.14`).
   * Consequence: gradient-type boosting with early stopping leaks a pure scale effect into He3. On the local quadratic, the peak shape coefficient is 0.408 for a scale truth of 1.2; at convergence or with a Newton step it is zero (`v1:1.15`). The same metric also correlates the estimation errors across channels.
3. **Misspecification.** When the truth is not in the K-channel span, the truncated pinball/CRPS projection is not the L² projection. Lognormal(0, 0.6): max coefficient bias 0.284 at K = 3. Fitting K_fit = 8 and reporting the first 3 reduces it to 0.057 (`v1:1.17`).

**Finding 2 (plain Legendre does not isolate S1's tail feature, even without noise).** S1 has `Q = 1.2x_loc + (0.4+1.2x_scale) z + 0.12 x_shape He3`.
* `z` itself has L-kurtosis load `λ_4(z) = 0.06917` (`v1:1.2`).
* `He3` has `λ_2(He3) = −0.282` and `λ_4(He3) = 0.600`.
* In channel k = 3 (L-kurtosis) the population energy ratio x_shape/x_scale is **0.753**, below the 3× criterion (`v1:1.16`).
* Other reference-adapted orthonormal bases (e_0 = 1, e_1 = centred reference, the rest Gram–Schmidt on P*_k):

| e_1 reference | x_shape / x_scale shape-channel energy | check |
|---|---|---|
| S1 marginal | 1.67 | `v1:1.16c` |
| mean L-scale-standardised conditional | 2.00 | `v1:1.16d` |
| `z` | exact isolation: scale load 3.8e-16 | `v1:1.16b` |

* I had hypothesised that the standardised reference would reach ≥ 3×. The test shows that hypothesis is **false**.

**Proposition 1.9 (when orthogonality identifies a channel, population level).** Let `{e_k}` be orthonormal in L²(0,1). Let the true additive model be `Q(u|x) = Σ_j g_j(x_j, u)`, and let the fit be the population L² (W2) projection of each `g_j` on `span{e_k}`. Feature j has zero energy in channel k iff `⟨g_j(x_j,·), e_k⟩ = 0` for P-a.e. x_j. In particular, a feature whose effect lies in `span{e_k : k ∈ T}` has zero energy outside T.

*Proof.* Coefficients of an orthogonal projection are inner products. ∎

The proposition says identification is a property of **the pair (truth, basis)**. Orthogonality alone does not deliver it. It fails when:
* the generator's directions are not basis directions (Finding 2);
* the estimator is not the W2 projection (pinball metric + early stopping, or misspecification);
* the quadrature is inexact;
* the signal is below the noise floor (§2.5).

**Recommendation (details in DESIGN_DELTA.md).**
* Fit in a rich polynomial basis on GL nodes (K_fit > K_report) with converged or Newton-type steps.
* Report by L² projection onto an explicitly chosen orthonormal reporting basis, which may be Legendre, reference-adapted, or Hermite-in-z. Parseval is then exact for the reported channels, plus the truncation residual.
* Write simulation pass criteria as oracle projections of the true generator in the reporting basis.

---

## 2. Shift attribution between periods A and B

Notation:
* `Q_m(u|x)` is the model for period m ∈ {A, B}.
* `P_m` is the covariate law in period m.
* `F^{(m,P)}` is the mixture law `∫ F_m(·|x) dP(x)` with quantile function `Q^{(m,P)}`.
* All curves are pre-projection unless stated otherwise.

### 2.1 Conditional shift

*Intuition.* For a given covariate profile, compare the two periods' predicted distributions and measure the Wasserstein distance. Then average over a stated population of profiles. Each comparison is a sum over channels and features (Section 1), so the average is too. This decomposition is exact, but it is only as meaningful as the population you average over, so state that population.

**Theorem 2.1.** Let P be any covariate law (it must be stated; P_B by default). Then
`D_cond(P) = E_{x~P} W2²(Q_A(·|x), Q_B(·|x)) = Σ_k Σ_j E_P φ_{jk}(x)`,
with `φ` from Theorem 1.3 applied to the per-row coefficient changes. The decomposition is exact by channel and by player.

*Proof.* Theorem 1.1 per row, Theorem 1.3 per row and channel, then linearity of expectation. ∎ (`v2:2.1`, |diff| 4.4e-16.)

**Assumptions and failures.**
* Evaluation law P: the value and the split depend on P.
* Centring of shape functions: done under the same P.
* Exact only before the isotonic projection (bound §1.4).
* Extrapolation: if P puts mass on bins never observed in period A or B, the model's value there is an extrapolation.

### 2.2 Marginal shift is not an average

*Intuition.* The overall distribution in a period is a mixture over covariate profiles. Mixing can cancel shifts: if one group moves up and another moves down by the same amount, the overall distribution may not change at all. So the marginal shift is never larger than the average conditional shift (with the same covariates), but it can be much smaller. It also cannot be split exactly feature by feature.

**Proposition 2.2.** For the same covariate law P, `W2²(F^{(A,P)}, F^{(B,P)}) ≤ D_cond(P)`.

*Proof.* Let `X ~ P` and `U ~ Unif(0,1)` be independent. The pair `(Q_A(U|X), Q_B(U|X))` is a coupling of `F^{(A,P)}` and `F^{(B,P)}` with expected squared cost `D_cond(P)`. W2² is the infimum over couplings. ∎ (`v2:2.2`: 1.099 ≤ 3.583.)

**Counterexample (no averaging identity).** Two equally likely groups with `Q_A = (z, z+2)` and `Q_B = (z+2, z)`. The marginal shift is 0 while the conditional shift is 4 (`v2:2.2b`).

**What fails.** There is no exact feature-level decomposition of the marginal W2². The mixture quantile function is a nonlinear functional of the per-row curves. A Shapley attribution over features is possible but requires recomputing a mixture for each coalition (2^p mixtures, or sampled permutations). It also depends on how a "partial" switch of a feature is defined. Channel-wise *linear* attribution of marginal L-moment differences is exact, because λ_k is linear in Q; squared energies are not.

### 2.3 Covariate-composition vs conditional-structure (Oaxaca–Blinder / DFL)

*Intuition.* The change between periods has two sources: the covariates changed (composition), or the relationship between covariates and outcome changed (structure). Build a counterfactual period, period A's model with period B's covariates, and compare each real period to it. The change in quantile functions then splits exactly into two pieces. The squared distance has a cross term, and the order in which you switch the two sources matters.

**Definition.** The counterfactual is `Q_C = Q^{(A,P_B)}`. This is the model-based analogue of the DiNardo–Fortin–Lemieux (Econometrica, 1996) reweighting counterfactual and of Oaxaca (1973) / Blinder (1973) for means. Inference for such counterfactual distributions is treated by Chernozhukov, Fernández-Val & Melly ("Inference on counterfactual distributions", Econometrica 2013; journal volume/pages UNVERIFIED here).

**Proposition 2.3.**
1. `Q_B − Q_A = (Q_B − Q_C) + (Q_C − Q_A)` exactly (structure + composition), in every channel: `Δλ_k = Δλ_k^{str} + Δλ_k^{comp}`.
2. `W2²(F_A,F_B) = ‖Q_B−Q_C‖² + ‖Q_C−Q_A‖² + 2⟨Q_B−Q_C, Q_C−Q_A⟩`.
3. The two-player Shapley split over {composition, structure} is efficient. It requires the other counterfactual `Q^{(B,P_A)}`.

*Proof.*
1. Telescoping, plus linearity of λ_k in Q.
2. Expand the square.
3. Efficiency of Shapley.

∎ (`v2:2.3`: total 1.599 = 1.099 + 0.049 + 0.451.)

**What fails.**
* The cross term can be large; in the check it is 28% of the total.
* The Shapley composition share can be negative (−0.015).
* The split is path dependent: composition-first gives 0.049, structure-first gives −0.079 (`v2:2.3` INFO).
* **Assumption (overlap / no extrapolation):** model A must be valid on the support of P_B. Bins populated in B but empty in A make `Q_C` an extrapolation.
* Feature-level splitting of the composition term has the same non-closed-form problem as §2.2.

### 2.4 Bias of squared statistics and cross-fitting

*Intuition.* Squared distances computed from noisy estimates are biased upward: even with no real shift, noise produces a positive distance. If you split the data into two independent halves and multiply the estimate from one half by the estimate from the other, the noise no longer multiplies itself. The product is then unbiased, provided each half's estimate is itself unbiased.

**Proposition 2.4.** Let `Δ̂ = Δ + ε` with `E ε = 0` and `Cov ε = Σ`. Then:
* `E‖Δ̂‖² = ‖Δ‖² + tr Σ`;
* if `Δ̂_1, Δ̂_2` are independent with means Δ, then `E⟨Δ̂_1, Δ̂_2⟩ = ‖Δ‖²`;
* for Shapley products, `E[ ½(a_{j}^{(1)} A^{(2)} + a_{j}^{(2)} A^{(1)}) ] = a_j A`.

*Proof.* Expand; the expectations of the cross products of independent zero-mean errors vanish. ∎ (`v2:2.4`: plug-in bias 0.1238 vs theory 0.1250; cross-fit 0.2839 ± 0.0022 vs truth 0.2843.)

**What fails.** Boosted, binned and early-stopped estimates are not unbiased. Cross-fitting removes the *noise × noise* term, not the regularisation bias. Cross-fit estimates of a nonnegative quantity can be negative; report them unclipped.

### 2.5 Noise floor: permutation, bootstrap, conformal

*Intuition.* To decide whether a shift is real, compare it with what pure relabelling of rows would produce. If the two periods really come from the same distribution, shuffling period labels does not change the distribution of the statistic, so the shuffled values form an exact reference. The bootstrap is useful for the size of a clearly non-zero shift. For a squared quantity near zero it is unreliable, so it should not be used as the null floor.

**Proposition 2.5 (permutation test, exact level).** H_0: the 2n rows are exchangeable across the period label, e.g. iid from the same joint law of (x, y) in both periods. Let `T` be any statistic and let `p = (1 + #{T_π ≥ T_obs})/(1 + R)` with R uniform random permutations. Then `P_{H_0}(p ≤ α) ≤ α`. This is the standard randomisation-test result (e.g. Lehmann & Romano, *Testing Statistical Hypotheses*, ch. 15 or 17 by edition; chapter number UNVERIFIED). (`v2:2.5`: rejection rate 0.103 at α = 0.10; power 0.28 for the planted tail at n = 300 per period.)

**Variants and failures.**
* To test "no conditional shift" while allowing covariate shift, permute labels *within covariate cells*. This is exact when x is discrete and the cells are the bins; it is approximate when the bins are coarsenings of a continuous x.
* Time-dependent rows (serial correlation within a period) violate exchangeability. Use block permutations of contiguous time blocks, which are approximate.
* Per-feature floors (e.g. permuting only feature j's contribution) are approximate. There is no exact null for "feature j contributes nothing" in a fitted additive model with dependent features.

**Bootstrap.**
* The row bootstrap within each period gives an SD for the statistic. In the check, the median ratio of bootstrap SD to Monte-Carlo SD is 0.71, within the pre-set 35% band (`v2:2.6`), so it underestimates.
* Bootstrap intervals for a squared statistic near 0 are unreliable because the parameter is on the boundary of a non-regular problem.
* Use the permutation test for the null floor and the bootstrap (block bootstrap for dependent rows) only for the magnitude of a clearly non-null shift.

**Conformal.** Split-conformal methods give coverage for *individual predictions*, not for a population distance. The exchangeability-based exact procedure for "is there a shift" is the permutation test above. I did not derive a conformal procedure for the shift statistic itself.

---

## 3. Distribution-valued responses: Fréchet regression in W2

*Intuition.* Sometimes each row is a whole sample (a distribution), not one number. The natural fit minimises the average squared Wasserstein distance between each row's distribution and the model's. In the Legendre coordinates this is ordinary least squares, done separately per channel. The targets are each row's L-moments. Hosking's unbiased sample L-moments let even small per-row samples give consistent fits. The naive approach of plugging in the empirical quantile function does not, because it is biased in the shape channels.

**Setting.**
* Row i has covariates `x_i` and an iid sample of size `m_i` from `F_i` with quantile function `Q_i`. Row weights are `w_i > 0`.
* The model is `Q_θ(u|x) = Σ_{k<K} c_k(x;θ) P*_k(u)`, with `c_k(x;θ) = D(x)θ_k` linear (intercept + bin dummies per feature).
* Global Fréchet regression in W2 (Petersen & Müller, Ann. Statist. 47(2):691–719, 2019) takes a weighted average of quantile functions and then projects onto the monotone functions.

**Theorem 3.1 (reduction).** With target `t_{ik} = (2k+1) ∫ Q_i P*_k`, i.e. `(2k+1)λ_{k+1}(F_i)`:
`Σ_i w_i W2²(F_i, Q_θ(·|x_i)) = Σ_k (1/(2k+1)) Σ_i w_i (t_{ik} − D(x_i)θ_k)² + Σ_i w_i ‖(I−Π_K)Q_i‖²`.
Without cross-channel constraints, the minimiser is K separate weighted least-squares fits. The weight `1/(2k+1)` does not change any per-channel argmin. It matters only when constraints couple channels (monotonicity, shared penalties).

*Proof.* Apply Proposition 1.2 to each row. Theorem 1.1 turns the projected part into `Σ_k (t_{ik} − c_k)²/(2k+1)`. The residual does not depend on θ. ∎ (`v3:3.1`: objectives 13.69944898 in both forms; parameters agree to 8.5e-14.)

**Proposition 3.2 (targets).**
* Hosking's sample L-moments `ℓ_r` (PWM form) are unbiased for `λ_r` whenever `m ≥ r` and the mean exists. The proof is in Hosking (1990).
* The plug-in coefficient `∫ Q̂_m P*_k` (empirical quantile function) is biased at O(1/m) for k ≥ 1. Exponential law: bias −0.100 in λ_2 at m = 5 and −0.025 at m = 20 (`v3:3.2`).
* The plug-in coincides with what one gets by inserting empirical measures into the W2 Fréchet objective. **Naive Fréchet regression on raw row samples is therefore biased in the shape channels for fixed m.**

**Proposition 3.3 (consistency with fixed per-row size).** Assume:
1. rows are iid draws of `(x_i, m_i, F_i)`, with sampling within a row iid given F_i;
2. `m_i ≥ k+1`;
3. `E t̃_{ik}² < ∞` for the unbiased target `t̃ = (2k+1) ℓ_{k+1}`, which needs the outcome to have finite variance;
4. `E[w D(x)ᵀD(x)]` is nonsingular.

Then weighted LS on `t̃_{ik}` is consistent and √n-asymptotically normal for the best linear (in D) approximation of `E[(2k+1)λ_{k+1}(F)|x]`, as n → ∞ with m_i bounded.

*Proof.* `E[t̃_{ik}|x_i, F_i] = (2k+1)λ_{k+1}(F_i)`, so the row-sampling noise has conditional mean zero. Standard M-estimation (least squares with iid rows and finite second moments) gives consistency and asymptotic normality for the projection parameter. ∎

Numerically, at m = 5 the RMSE ratio between n = 200 and n = 3200 is 0.260 (√n predicts 0.25), while plug-in targets plateau at 0.152 (`v3:3.3`). With plug-in targets, consistency requires `min_i m_i → ∞` as well.

**Proposition 3.4 (W2 error of a row's empirical law; used to choose m).** Bobkov & Ledoux (Memoirs AMS 261, no. 1259, 2019) give `E W2²(F_m, F) ≤ 2 J_2(F)/(m+1)`, where `J_2(F) = ∫ F(1−F)/f`.
* UNVERIFIED against the primary text: both the constant and the theorem number. The constant is from a secondary source.
* Checked numerically for the uniform law (J_2 = 1/6) at m = 5, 20, 100 (`v3:3.4`).
* For the normal law J_2 = ∞, and m·E W2² grows: 2.61, 3.03, 3.33 at m = 10, 100, 1000 (`v3:3.4b`). The 1/m rate fails for Gaussian rows.
* This affects plug-in W2 errors only. Unbiased L-moment targets do not need it.

**Proposition 3.5 (unequal m_i).** Suppose `t̃_{ik} = μ_k(x_i) + η_{ik} + ε_{ik}`, where:
* `η_{ik}` is between-row heterogeneity with variance `τ_k²`;
* `ε_{ik}` is sampling noise with `Var(ε_{ik}|m_i) ≈ v_k/m_i`, the leading-order variance of sample L-moments.

Then the Gauss–Markov-optimal weights are `w_{ik} = 1/(τ_k² + v_k/m_i)`. They differ by channel, which is allowed because the channels are separate fits. The two extremes are:
* `τ_k = 0` gives `w ∝ m_i`;
* `τ_k² ≫ v_k/m_i` gives equal weights.

`τ_k²` and `v_k` can be estimated by method of moments from first-pass residuals. *Proof:* Aitken's theorem for heteroskedastic linear models. ∎ (`v3:3.5`: slope variance GLS 3.97e-3 vs equal 8.26e-3 vs ∝m 5.35e-3; all unbiased.) *Approximation:* `v_k/m_i` is the leading-order variance only, and `v_k` depends on F_i.

**Proposition 3.6 (monotonicity in Fréchet mode).** If the target `Q(·|x)` is nondecreasing, then `‖ΠQ̂ − Q‖ ≤ ‖Q̂ − Q‖`. This follows from Prop. 1.8(1) with `ΠQ = Q`, so projecting a fitted curve never hurts (`v3:3.6`). If monotonicity is imposed *during* the fit (Prop. 1.7 constraints), the channels couple and the `1/(2k+1)` weights matter.

---

## 4. Wasserstein distributionally robust fitting

### 4.0 Precisely stated duality results (citations verified to the level stated)

* **Mohajerin Esfahani & Kuhn**, "Data-driven distributionally robust optimization using the Wasserstein metric", Math. Program. 171:115–166 (2018), arXiv:1505.05116.
  * Type-1 Wasserstein balls around the empirical law.
  * Finite-sample guarantee (their Thm 3.5) from measure concentration (Thm 3.4), under a light-tail condition (Assumption 3.3).
  * Asymptotic consistency (Thm 3.6).
  * Convex reduction for losses that are pointwise maxima of finitely many concave pieces (Thm 4.2).
  * LP reformulations for piecewise-affine losses with 1-/∞-norm transport costs (Cor 5.1).
  * The numbering of their Lipschitz-loss results is UNVERIFIED.
* **Gao & Kleywegt**, "Distributionally robust stochastic optimization with Wasserstein distance", arXiv:1604.02199; Math. Oper. Res. (2023; volume/pages UNVERIFIED).
  * Strong duality on a Polish space for any measurable loss Ψ (their Thm 3.14): `sup_{W_p(μ,ν)≤θ} E_μ Ψ = min_{λ≥0} { λθ^p − ∫ inf_ξ [λ d^p(ξ,ζ) − Ψ(ξ)] ν(dζ) }`.
  * Data-driven form (Cor 3.19).
  * Growth-rate condition (Prop 3.6, Lemma 3.4).
  * The discrete metric yields total variation (Remark 3.23).
* **Blanchet & Murthy**, "Quantifying distributional model risk via optimal transport", Math. Oper. Res. 44(2) (2019), doi 10.1287/moor.2018.0936. Strong duality for lower-semicontinuous costs on Polish spaces (Thm 1; page range UNVERIFIED).
* **Blanchet, Kang & Murthy**, "Robust Wasserstein profile inference and applications to machine learning", J. Appl. Probab. 56(3):830–857 (2019): sqrt-LASSO and regularised logistic regression as DRO.
* **Gao, Chen & Kleywegt**, "Wasserstein distributionally robust optimization and variation regularization", Oper. Res. 72(3):1177–1191 (2024), doi 10.1287/opre.2022.2383, arXiv:1712.06050. The worst-case risk is approximated or bounded by empirical risk plus a "variation" regulariser that generalises Lipschitz and gradient-norm penalties, including for nonsmooth losses. Verified at abstract level; theorem-level statements UNVERIFIED.
* **Shafieezadeh-Abadeh, Kuhn & Mohajerin Esfahani**, "Regularization via mass transportation", JMLR 20 (2019). For losses `L(⟨w,x⟩ − y)` with L Lipschitz on an unbounded space, the W1 worst case equals `empirical + ρ·lip(L)·‖(w,−1)‖_*`. Theorem number and exact equality conditions UNVERIFIED (seen only via a secondary summary).

For finite support, which is the case for TRACE's binned covariates, everything below is **plain LP duality**. None of the general theorems above is needed for exactness.

### 4.1 Setting for TRACE

* Fix a quantile level u. The model is additive and binned: `q(b) = Σ_j f_j(b_j)`, where in TRACE `f_j(b) = Σ_k θ_{jbk} e_k(u)`.
* Loss: `ℓ(b,y) = ρ_u(y − q(b))` with `ρ_u(r) = max(u r, (u−1) r)`, and `ρ̄ = max(u, 1−u)`.
* The empirical law `P_n` sits on `(b_i, y_i)`.
* **Ordinal grid cost:** `d((b,y),(b',y')) = Σ_j κ_j |b_j − b'_j| + κ_y |y − y'|`. κ_j is the "bin-width cost" per step; κ_y = ∞ means covariate-only perturbation.
* `R(ε) = sup{E_Q ℓ : W_d(Q,P_n) ≤ ε}` (type 1) and `R_2(ε)`, the same with `W_{2,d}`.

*Intuition (whole section).* The adversary moves a fraction of the data to other bins (or other outcome values) within a transport budget, to make the loss as large as possible. Because the model is a step function of the bins, the cheapest damaging move is "shift a row one or more bins over". The worst case can be computed exactly by a small linear programme that splits feature by feature. A smoothness (Lipschitz) penalty is the wrong tool for step functions. On the bin grid the honest analogue is the largest jump per unit bin cost, and it gives only an upper bound. Moving outcomes alone, under W1, adds the same constant to every model, so it does not regularise at all.

### 4.2 Exact robust objective on the bin grid (covariate perturbation)

**Theorem 4.1 (exact, finite LP).** With κ_y = ∞, and with Q restricted to the grid support (the model is defined only on bins),
`R(ε) = min_{λ≥0} { λε + (1/n) Σ_i max_c [ ℓ(c, y_i) − λ d(c, b_i) ] }`.
The inner maximum separates over features:
`max_c [ℓ(c,y_i) − λ d(c,b_i)] = max_{s ∈ {u, u−1}} { s y_i + Σ_j max_{c_j} [ −s f_j(c_j) − λ κ_j |c_j − b_{ij}| ] }`.
It costs O(Σ_j B_j) per row and per λ. The value as a function of λ is convex and piecewise linear, so it can be minimised by a 1-D search over breakpoints.

*Proof.* The primal is the LP `max Σ_{i,c} π_{ic} ℓ(c,y_i)` subject to:
* `Σ_c π_{ic} = 1/n` for each i;
* `Σ π_{ic} d(c,b_i) ≤ ε`;
* `π ≥ 0`.

Every coupling with first marginal P_n has this form. The LP is feasible (put π on c = b_i) and bounded (finite ℓ), so LP strong duality gives the dual. Separability: `ρ_u(r) = max_s s·r`, and `y − q(c)` is additively separable in c, so the max over c of a sum of per-feature terms is the sum of per-feature maxima, for each s. This is also the finite-support case of Gao–Kleywegt Thm 3.14 and Mohajerin Esfahani–Kuhn Thm 4.2. ∎ (`v4:4.1`: primal LP = dual = separable form to 1e-8.)

**Proposition 4.2 (convexity / LP form for fitting).** `θ ↦ R(ε; θ)` is convex. The robust fitting problem `min_θ R(ε;θ)` (summed over levels, with a common or per-level adversary, see 4.7) is an LP with O(n Σ_j B_j) constraints, using the epigraph variables `g_{ijs} ≥ −s f_j(c) − λκ_j|c − b_{ij}|` for all c, and `t_i ≥ s y_i + Σ_j g_{ijs}`.

*Proof.* For fixed (c, s), `s y − s q_θ(c) − λ d` is affine in (θ, λ). A maximum of affine functions is jointly convex, and partial minimisation over λ preserves convexity. ∎ (`v4:4.6`: midpoint convexity in 60 random pairs.)

### 4.3 The Lipschitz question: the correct grid analogue is an upper bound

**Proposition 4.3 (upper bound, not exact).** Let `G_j = max_b |f_j(b+1) − f_j(b)|`, the largest adjacent jump. Then
`R(ε) ≤ (1/n)Σ_i ℓ(b_i, y_i) + ε ρ̄ max_j G_j/κ_j`.

*Proof.* `|ℓ(c,y) − ℓ(b,y)| ≤ ρ̄ Σ_j |f_j(c_j) − f_j(b_j)| ≤ ρ̄ Σ_j G_j |c_j − b_j| ≤ ρ̄ (max_j G_j/κ_j) d(c,b)`. So ℓ is Lipschitz on the grid metric with that constant, and Kantorovich–Rubinstein duality for W1 gives the bound. ∎

* With the ℓ1-additive cost the dual norm is a **max** over features. With an ℓ∞ (max) cost it would be a **sum** `Σ_j G_j/κ_j`.
* Tightness: the observed ratio (exact gain)/(bound gain) ranges from 0.22 to 1.00 (`v4:4.2`).
* The bound is attained only if some row has a residual of the right sign at a bin adjacent to the maximal jump, so that the adversary can actually realise the rate.

**Conclusion.**
* The "adjacent-difference" penalty `max_j G_j/κ_j` (an ℓ∞ norm of adjacent differences) is the W1-correct form. It is an upper bound on the exact robust risk.
* A *total-variation* penalty (sum of adjacent differences) is the correct form for **W∞ / local** perturbations (4.6), not for W1.
* Use the exact LP (Theorem 4.1) when exactness matters.

**Proposition 4.4 (raw-x cost: no Lipschitz equivalence).** Measure transport in raw covariate units, `d = Σ_j κ_j |x_j − x'_j|`. The worst case is still exact via Theorem 4.1, with `d(c, x_i) = Σ_j κ_j dist(x_{ij}, bin c_j)`. But the initial slope `dR/dε|_{0+} = max_{i,c} [ℓ(c,y_i) − ℓ(b_i,y_i)]/d(c,x_i)` is unbounded as a data point approaches a bin edge. The step function has no finite Lipschitz constant, so `emp + ε·Lip` bounds of Shafieezadeh-Abadeh et al. type do not apply. (`v4:4.3`: slope 77.7, 77.7, 297.4, 1486.9 for edge distances δ = 0.05, 0.01, 0.002, 0.0004. The slope stays ≥ gain/δ for the planted point; at the two larger δ a different point/feature attains the maximum.)

### 4.4 Outcome perturbation, kept separate from covariate perturbation

**Proposition 4.5 (outcome-only W1 is a constant).** With covariates fixed and `d = κ_y |y − y'|`:
`R(ε) = emp + ε ρ̄ / κ_y`.
This holds exactly whenever at least one row's residual has the sign that attains ρ̄ (or is zero), and the support of y is unbounded in that direction. It is the same for every model, so it does **not** regularise the fit.

*Proof.* ρ_u is ρ̄-Lipschitz, which gives ≤ by Kantorovich–Rubinstein. For equality, suppose u ≥ 1/2 (the other case is symmetric). Take a row with residual `y_i − q(b_i) ≥ 0` and move a fraction π ≤ 1/n of mass from it upward by t, with `π κ_y t = ε`. On that path ρ_u has slope u = ρ̄, so the loss increases by `π ρ̄ t = ε ρ̄/κ_y`. ∎ (`v4:4.4`.)

* Integrated pinball, `∫_0^1 ρ_u(y − q(u)) du`, has derivative in y equal to `∫(u − 1{y < q(u)}) du ∈ [−1/2, 1/2]`. So one adversary common to all levels gains at rate at most 1/2 per unit cost/κ_y; the rate approaches 1/2 as y moves past all of q's range. If instead each level has its own adversary, the rates add up to `∫max(u,1−u) du = 3/4`. Summing per-level worst cases is therefore an upper bound on the common-adversary value (`v4:4.8`: 0.570 ≤ 0.642).

**Proposition 4.6 (joint (x,y) perturbation, additive cost).**
`R(ε) = min_{λ ≥ ρ̄/κ_y} { λε + (1/n)Σ_i max_c [ℓ(c,y_i) − λ d_x(c,b_i)] }`.
The covariate-only dual is restricted to `λ ≥ ρ̄/κ_y`.

*Proof.* In the dual, the inner sup over `y'` of `ρ(y' − q) − λκ_y|y' − y|` is +∞ if λ < ρ̄/κ_y. Otherwise it is attained at `y' = y`, since `ρ(y'−q) − ρ(y−q) ≤ ρ̄|y'−y|`. ∎ (`v4:4.4b`.) Hence the joint value equals the covariate-only value when the covariate optimum `λ* ≥ ρ̄/κ_y`; otherwise it is strictly larger. In the check, κ_y = 1: equal (0.5302, 2.1356); κ_y = 0.05, ε = 0.4: 6.79 vs 2.14.

**Proposition 4.7 (W2 balls).**
* Covariate W2 on the grid: replace `d` by `d²` and ε by ε² in Theorem 4.1. The result is still an exact LP and its dual, and it is ≤ the W1 value, because `W_1 ≤ W_2` and so the W2 ball is contained in the W1 ball (`v4:4.9`).
* Outcome-only W2 (`d = κ_y²|Δy|²`, budget ε²): the exact value is the same LP/dual with a 1-D inner sup per row. It is ≤ `emp + ρ̄ε/κ_y` by Cauchy–Schwarz (`E|Δy| ≤ ε/κ_y`). Unlike W1 it **depends on the residuals**: a shifted fit gave gain 0.0150 vs 0.0269 at ε = 0.05 (`v4:4.10`).
* I have not derived a closed form for the outcome-only W2 worst case. It is computable exactly per Gao–Kleywegt duality, but its regularisation character is uncharacterised (open).

**Guarantee under a W1/W2 ball.** If the deployment law Q of (b, y) satisfies `W_{1,d}(Q, P_n) ≤ ε` for the stated cost d, then `E_Q ℓ ≤ R(ε)` deterministically; the same holds for W2 with R_2. The finite-sample statement "the population law lies in the ball with probability 1−β" needs the concentration results of Mohajerin Esfahani & Kuhn (Thm 3.4/3.5, light tails). As I recall it, their radius scales like `n^{−1/max(d,2)}` in the data dimension d (exponent UNVERIFIED). That rate is pessimistic for p features. That statement is about the *population* law, not a later period's law. **Calibrating ε from per-feature marginal W1 distances under-covers**: `Σ_j κ_j W1(marginal_j) ≤ W_1(joint)` for the additive cost, with joint/marginal-sum ratios up to 1.64 (`v4:4.7`). The repo's per-feature "Feature W2" style checks therefore give lower bounds on the needed radius, not valid radii.

### 4.5 W∞ (local) perturbations: where TV is the right form

**Proposition 4.8.** Suppose each row can move at most r bins per feature and `|Δy| ≤ δ` (a W∞ ball with ℓ∞-type cost). Then the worst case is exactly
`(1/n) Σ_i max{ ρ(y_i + δ − q_min,i), ρ(y_i − δ − q_max,i) }`,
where `q_min,i = Σ_j min_{|c−b_{ij}|≤r} f_j(c)` and similarly for q_max. This costs O(npB). A total-variation bound follows:
`≤ emp + ρ̄ (δ + (1/n) Σ_i Σ_j TV_{window}(f_j; b_{ij}, r))`.
Averaging over rows, the TV term is a **bin-mass-weighted total variation** of each shape function.

*Proof.* Pointwise maximisation over a product of windows. ρ is monotone in each direction of its argument, so the extreme is at the joint min or max of the additive q. TV bound: `|f(c) − f(b)| ≤` TV over the window. ∎ (`v4:4.5`: exact 0.891 ≤ TV bound 1.911.)

### 4.6 What Wasserstein balls do not cover

* Selection or reweighting shifts, i.e. a change in the mixture weights of existing rows with no movement of rows, are not small in W1 unless the rows are close. φ-divergence (e.g. CVaR-type / χ²) ambiguity sets model these; not covered here.
* Integrated loss with a common adversary vs per-level adversaries: summing per-level worst cases is a valid upper bound, not exact (`v4:4.8`).

---

## 5. Summary of corrections to the claims in the brief

1. "W2² = Σ(2k+1)Δb_k²" holds for L-moments b_k = λ_{k+1}. With expansion coefficients the weight is 1/(2k+1) (§1.1).
2. "v0's failure is because the channels are not orthogonal" is incorrect as stated: {1, z, He3} are W2-orthogonal. The causes are the midpoint grid, the pinball metric with early stopping, misspecification and noise (§1.5).
3. "Orthogonality (switching to Legendre) will fix tail identification" is false for S1 as written: the population channel-3 ratio is 0.753 < 3. Identification needs the reporting basis to contain the generator's directions, and tests must use oracle projections (§1.5).
4. "Binned step functions with a Lipschitz/TV penalty give the DRO objective": under W1 the exact form is an LP. The Lipschitz-type analogue (max adjacent jump per bin cost) is only an upper bound. TV is the W∞ form. Raw-x costs have no Lipschitz equivalent (§4.3–4.5).
5. Fréchet W2 regression "reduces to LS on L-moment coefficients with 1/(2k+1) weights": true. The weights are irrelevant without cross-channel constraints, and naive plug-in targets are inconsistent for fixed m (§3).
6. Marginal shift is not decomposable by feature in closed form. Composition/structure splits are exact as differences but path dependent as squared energies (§2).
