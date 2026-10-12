# Shift decomposition v2: explaining distribution change between two periods

Status: theory, verified numerically by `verify/shift_v2_checks.py` (checks C1–C6).
Notation follows THEORY.md: `P*_k` is the shifted Legendre polynomial of degree k on [0,1],
`∫ P*_j P*_k du = δ_jk/(2k+1)`; channel k carries the L-moment `λ_{k+1} = ∫ Q(u) P*_k(u) du`.
Write `c_k = (2k+1) λ_{k+1}` for the coefficient of `Q` in the basis, so `Q = Σ_k c_k P*_k`.

## 1. Per-channel W2² (Prop. 1)
For quantile functions `Q_A, Q_B` in span{P*_0..P*_{K-1}} (or the projection of any L² quantile
function onto it), `W2²(A,B) = ∫ (Q_B−Q_A)² du = Σ_k (Δc_k)²/(2k+1) = Σ_k (2k+1)(Δλ_{k+1})²`.
*Proof.* In 1-D, W2 is the L²[0,1] distance between quantile functions; expand and use orthogonality:
cross terms vanish and `∫ P*_k² = 1/(2k+1)`. Beyond K, the residual `∫ (Q_B−Q_A − proj)²` is
reported as a separate "unresolved" channel, so the total is exact. ∎ (C1)

## 2. Targets: what "the distribution of a period" means
Two complementary targets, both reported:

* **Marginal law** `F_t(y) = ∫ F_t(y|x) dF_t(x)`. Model-free, but L-moments are *not* linear in
  the mixture (only λ₁ is), so only channel-level splits are exact.
* **Barycenter (Vincent average)** `Q̄_t(u) = E_{x~F_t}[Q_t(u|x)]`. This is the W2 Fréchet mean of
  the conditional laws (in 1-D the W2 barycenter averages quantile functions). Coefficients are
  linear: `c̄_{t,k} = E_{x~F_t}[c_{t,k}(x)]`, and with TRACE's additive channels
  `c_{t,k}(x) = b_{t,k} + Σ_j f_{t,kj}(x_j)`, per-feature splits are exact.
The gap between the two targets (a mixing/Jensen term) is reported, never hidden.

## 3. Counterfactuals via DiNardo–Fortin–Lemieux reweighting
Let `C = (structure B, covariates A)`: `F_C(y) = ∫ F_B(y|x) dF_A(x)`. With pooled sample and period
label T, Bayes gives `dF_A/dF_B (x) = [P(A|x)/P(B|x)]·[P(B)/P(A)] =: ψ(x)`, so
`F_C(y) = E_B[ψ(X) 1{Y≤y}]` (Prop. 2; requires overlap: supp F_A ⊆ supp F_B). Symmetrically
`C' = (structure A, covariates B)` reweights A by `1/ψ`. ψ is estimated by a period-odds classifier,
**cross-fitted** (K folds; each unit's weight from a model not trained on it), self-normalized to
mean 1. Weights are trimmed at a pre-declared quantile and the trimmed mass is reported. (C5)
For the barycenter target, counterfactuals need no weights: `c̄_C = E_A[c_B(x)]` directly.

## 4. Order-free two-player Shapley split (Prop. 3)
Per channel k, with values `v(∅)=λ^A, v({comp})=λ^{C'}, v({struct})=λ^C, v({both})=λ^B`:
`comp_k = ½[(λ^B−λ^C) + (λ^{C'}−λ^A)]`, `struct_k = ½[(λ^C−λ^A) + (λ^B−λ^{C'})]`.
Then `comp_k + struct_k = λ^B − λ^A` identically, and the result does not depend on which ordering
is "first" (it is the average over both). The interaction `λ^B−λ^C−λ^{C'}+λ^A` is split equally,
and is reported too. ∎ (C3)

## 5. From channel deltas to W2² shares (Prop. 4)
Let players i have additive coefficient-scale contributions `a_i` with `Σ_i a_i = Δc_k`. The game
`v(S) = (Σ_{i∈S} a_i)²/(2k+1)` has Shapley values `φ_i = a_i·(Σ a)/(2k+1)`.
*Proof.* For player i and coalition S not containing i, `v(S∪i)−v(S) = (a_i² + 2a_i s_S)/(2k+1)`,
`s_S = Σ_{j∈S} a_j`. Averaging over uniformly random orderings, each j≠i precedes i with
probability ½, so `E[s_S] = ½ Σ_{j≠i} a_j`; hence `φ_i = (a_i² + a_i Σ_{j≠i} a_j)/(2k+1) =
a_i Σ a/(2k+1)`. Efficiency: `Σ φ_i = (Σa)²/(2k+1)`, i.e. exactly channel k's W2² term. ∎ (C2, C4)
Note φ_i can be negative (a player pulling against the net shift); this is reported, not clipped.

**Per-feature players (barycenter target).** With `ḡ_t^{s}(j) = E_{x~F_s}[f_{t,kj}(x_j)]`:
`comp_{kj} = ½[(ḡ_A^B−ḡ_A^A) + (ḡ_B^B−ḡ_B^A)]`, `struct_{kj} = ½[(ḡ_B^A−ḡ_A^A) + (ḡ_B^B−ḡ_A^B)]`
(superscript = covariate period, subscript = structure period), plus an intercept player
`Δb_k`. These 2p+1 players sum exactly to `Δc̄_k` (C4). The channel-level marginal-law split
(§4) is reported alongside, using `a = (comp_k, struct_k)·(2k+1)`.

## 6. Inference
**Bootstrap bands (Chernozhukov–Fernández-Val–Melly style).** Resample units within each period
(stratified, B ≥ 200 replicates), re-run the whole pipeline including the cross-fitted classifier
and the TRACE fit (or, at scale, the multiplier/exchangeable-weight bootstrap with Exp(1) weights,
which avoids refitting the sample split). Pointwise percentile bands per channel and per player;
a **uniform** band across channels uses the max-|t| statistic `max_k |θ*_k−θ̂_k|/σ̂*_k`.
Validity rests on Hadamard differentiability of the quantile/L-moment and reweighting maps (CFM 2013)
and on the classifier converging fast enough (cross-fitting removes own-observation bias).
**Noise floor.** (i) Permutation: shuffle period labels R times; under exchangeability the
decomposition has zero mean; the 95th percentile of each statistic is the floor. (ii) Cross-fit
floor: variability across fold assignments. Shares below the floor are flagged "not resolved".

## 7. Optional: units entering/leaving (unbalanced mass)
Split each period into stayers S and movers: `F_A = (1−e_A)F_A^S + e_A F^X` (exits),
`F_B = (1−e_B)F_B^S + e_B F^N` (entries). The barycenter is linear in unit weights, so
`Δc̄ = stay + mix + entry + exit` exactly with
`stay = ½[(1−e_A)+(1−e_B)](c̄_B^S−c̄_A^S)`, `mix = ½(c̄_A^S+c̄_B^S)(e_A−e_B)`,
`entry = e_B c̄^N`, `exit = −e_A c̄^X` (C6); feed these as players into Prop. 4.
If total mass changes (`m_A ≠ m_B`, counts not shares), report separately the pure-mass term of the
Hellinger–Kantorovich distance, `HK²_mass = 4(√m_A − √m_B)²` (HK between two masses at the same point, under the convention HK² = 4·Hellinger² at zero transport; scale convention-dependent), also shown normalized by `m_A+m_B`;
it is **not** added into W2² (different units), only shown next to it.

## 8. Claims and limits
Exact: Props 1, 3, 4 and §7 identities (algebraic, verified to ≤1e-10). Approximate: DFL
counterfactual (consistency under overlap + correct/consistent classifier), bootstrap coverage
(asymptotic). Shapley is one fair-division convention, not causal identification; "composition"
means covariate mix under the model's conditional structure, under overlap.
