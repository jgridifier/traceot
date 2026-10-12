# Uniqueness of the TRACE shift budget

Status: theorem with full proofs; every lemma and counterexample is checked numerically by
[`verify/uniqueness_checks.py`](verify/uniqueness_checks.py) (checks U1–U7, X0–XO).
Builds on [SHIFT_V2.md](SHIFT_V2.md) (notation: `P*_k` shifted Legendre on [0,1],
`w_k := ∫(P*_k)² = 1/(2k+1)`, `Q = Σ_k c_k P*_k`, raw coefficients `c_k`).

## 0. Summary

TRACE splits the squared 2-Wasserstein distance between two periods into pieces indexed by
channel k, feature j and role r ∈ {composition, structure}:
`φ_{kjr} = w_k · a_{kjr} · Δc̄_k`, where `a_{kjr}` is the order-free (two-player Shapley)
coefficient-scale contribution. We show this is the **only** rule that (i) factors through
coefficient-scale attributions, (ii) uses Shapley's axioms at the coefficient level, and
(iii) uses Shapley's axioms for the induced W2² game. Channel additivity is **derived**
(from orthogonality), not assumed. Uniqueness is relative to the factorisation axiom (F): a
different but also exact, symmetric rule exists without it (§5, XF). The per-feature result
holds for the barycenter target only; for the marginal law only the channel-level result holds.

## 1. Setting

**Periods and roles.** Two periods A (=0) and B (=1). A "hybrid" pairs a covariate period
s and a structure period t; (A,A) and (B,B) are the observed periods, (B,A) is SHIFT_V2's C
(structure A, covariates B) and (A,B) is C'.

**Barycenter target (B).** Each period's conditional quantile function is
`Q_t(u|x) = Σ_{k<K} c_{t,k}(x) P*_k(u)` with additive coefficients
`c_{t,k}(x) = b_{t,k} + Σ_{j=1..p} f_{t,kj}(x_j)` (TRACE's model; f centred by a fixed convention,
see §6). The barycenter of hybrid (s,t) is `Q̄[s,t](u) = E_{x~F_s} Q_t(u|x)`, with coefficients
`c̄_k[s,t] = Σ_{j=0..p} G_{kj}[s,t]`, where

    G_{kj}[s,t] = E_{x~F_s} f_{t,kj}(x_j)   (j ≥ 1),     G_{k0}[s,t] = b_{t,k}   (intercept player j = 0).

`G_{kj}[s,t]` depends on `F_s` only through the marginal law of `x_j`, so each 2×2 table is well
defined. The data of the problem is the array `G = (G_{kj}[s,t])`. Write
`Δc̄_k = c̄_k[B,B] − c̄_k[A,A]` and `W := ∫(Q̄[B,B] − Q̄[A,A])²du`, the squared W2 distance between
the two period barycenters (in 1-D W2 is the L² distance of quantile functions).

**Marginal-law target (M).** `F_t` is the marginal law of y in period t, and `F_C`, `F_{C'}` are the
DFL counterfactuals. `λ^{s,t}_{k+1}` are L-moments of the four laws, `c_k[s,t] = (2k+1)λ^{s,t}_{k+1}`.
Players are only the roles r (no features).

**Games.** A TU game on a finite player set N is `v: 2^N → R` with `v(∅) = 0`. `Γ(N)` is the space
of all of them. A value is a map `ψ: Γ(N) → R^N`.

## 2. Axioms

Stage 1 (coefficient scale). For each (k,j), the **role game** on players {comp, struct} is
`g_{kj}(S) = G_{kj}[1{comp∈S}, 1{struct∈S}] − G_{kj}[A,A]`. A stage-1 rule α maps a 2-player game
to `(α_comp, α_struct)` and must satisfy

* **E1 (efficiency).** `α_comp + α_struct = g(N)` = the observed change.
* **S1 (order symmetry).** If the two roles are interchangeable in g (`g({comp}) = g({struct})`), they
  get equal amounts. Equivalently, no role is privileged as "first" in the path A → hybrid → B.
* **N1 (null role).** If switching a role never changes the value, it gets 0.
* **L1 (additivity).** `α(g + h) = α(g) + α(h)` on all 2-player games.

Stage 2 (W2² scale). Let N = {(k,j,r)}. Given coefficient attributions `a ∈ R^N`, the **induced
game** is `v_a(S) = ‖Σ_{(k,j,r)∈S} a_{kjr} P*_k‖²` (L²[0,1]). A stage-2 value ψ on Γ(N) satisfies
Shapley's axioms:

* **E2 (efficiency / exact budget).** `Σ_i ψ_i(v) = v(N)`.
* **S2 (symmetry).** Interchangeable players get equal amounts.
* **N2 (null player).** A player with `v(S∪i) = v(S)` for all S gets 0.
* **L2 (additivity).** `ψ(v + w) = ψ(v) + ψ(w)`.

Linking axiom:

* **F (factorisation through coefficient attributions).** `Φ(G) = ψ(v_{α(G)})`, where
  `α(G)_{kjr} = α(g_{kj})_r`. In words: first explain each coefficient's change, then share the
  squared distance among those coefficient pieces; the W2² allocation sees the data only through
  the coefficient attributions and the geometry of the basis.

Note: no channel-additivity axiom is imposed; it is a consequence (Lemma 4).

## 3. Lemmas

**Lemma 1 (Parseval; SHIFT_V2 Prop. 1).** For `D = Σ_{k<K} d_k P*_k`, `‖D‖² = Σ_k w_k d_k²`.
*Proof.* `‖D‖² = Σ_{k,l} d_k d_l ∫P*_k P*_l = Σ_k d_k² w_k` since `∫P*_k P*_l = δ_{kl} w_k`. ∎ (U1)

**Lemma 2 (Shapley's theorem).** On Γ(N), exactly one value satisfies E2, S2, N2, L2:
`Sh_i(v) = Σ_{S ⊆ N∖i} |S|!(n−|S|−1)!/n! · [v(S∪i) − v(S)]`.
*Proof.* Existence: Sh is linear (L2); summing marginal contributions along any ordering telescopes
to v(N), and Sh averages over orderings (E2); a null player's marginal contributions are all 0 (N2);
the formula is invariant under relabelling (S2). Uniqueness: the unanimity games
`u_T(S) = 1{T ⊆ S}`, T ≠ ∅, form a basis of Γ(N) (Möbius inversion: `v = Σ_T m_v(T) u_T` with
`m_v(T) = Σ_{R⊆T} (−1)^{|T|−|R|} v(R)`; there are 2^n − 1 of them and they are independent because
the triangular system `u_T(S)`, ordered by inclusion, has unit diagonal). For any value ψ obeying
the axioms and scalar m, ψ(m·u_T) is fixed: players outside T are null (N2, so 0); players in T
are interchangeable (S2, so equal); they sum to m (E2). So `ψ_i(m u_T) = m/|T|` for i ∈ T. Then L2 (additivity; it also gives ψ(0) = 0) fixes ψ on every `v = Σ m_v(T) u_T`. ∎
For n = 2 this is the explicit statement used in stage 1: the basis is `u_{comp}, u_{struct}, u_{both}`
with forced values (1,0), (0,1), (½,½). (U2b)

**Lemma 3 (stage-1 formula).** The unique α satisfying E1, S1, N1, L1 is

    a_comp   = ½[(G[B,A] − G[A,A]) + (G[B,B] − G[A,B])],
    a_struct = ½[(G[A,B] − G[A,A]) + (G[B,B] − G[B,A])],

i.e. SHIFT_V2 §4/§5, the average of the comp-first and struct-first sequential splits. The
interaction `I = G[B,B] − G[B,A] − G[A,B] + G[A,A]` is split ½/½.
*Proof.* Lemma 2 with n = 2, evaluating Sh on the role game. ∎ (U2, U6d)
Summing over j: `Σ_{j,r} a_{kjr} = c̄_k[B,B] − c̄_k[A,A] = Δc̄_k` (E1 per player, intercept included). (U5)

**Lemma 4 (quadratic game in a Hilbert space).** Let `x_1..x_n` be vectors in a real inner-product
space and `v(S) = ‖Σ_{i∈S} x_i‖²`. Then `Sh_i(v) = ⟨x_i, Σ_{l} x_l⟩`.
*Proof.* `v(S) = Σ_{i∈S}‖x_i‖² + Σ_{i<l; i,l∈S} 2⟨x_i,x_l⟩ = Σ_i ‖x_i‖² u_{{i}}(S) + Σ_{i<l} 2⟨x_i,x_l⟩ u_{{i,l}}(S)`.
By Lemma 2, `Sh(u_{{i}})` gives 1 to i, `Sh(u_{{i,l}})` gives ½ to each of i, l. By linearity
`Sh_i(v) = ‖x_i‖² + Σ_{l≠i}⟨x_i,x_l⟩ = ⟨x_i, Σ_l x_l⟩`. ∎ (U3)
SHIFT_V2 Prop. 4 is the 1-dimensional case.

**Corollary 4′ (channel separation).** With `x_{kjr} = a_{kjr} P*_k`,
`Sh_{kjr}(v_a) = ⟨a_{kjr}P*_k, Σ_{k'} Δc̄_{k'}P*_{k'}⟩ = w_k a_{kjr} Δc̄_k`; summing over (j,r)
gives `w_k (Δc̄_k)²`, channel k's Parseval term. Cross-channel interaction terms
`⟨P*_k, P*_{k'}⟩` vanish, so channel-level efficiency is a theorem, not an assumption. (U4, U4b)

**Lemma 5 (invariances).** For the rule of Theorem 1:
(a) swapping the labels A↔B maps every a to −a and every Δc̄ to −Δc̄, so φ is unchanged;
(b) relabelling features permutes φ accordingly;
(c) y ↦ σy + m (σ > 0) adds m to every `c̄_0[s,t]` (via b_{t,0}) and multiplies all coefficients by σ,
so `a ↦ σa` (the constant cancels in every difference), `Δc̄ ↦ σΔc̄`, `φ ↦ σ²φ`;
(d) any bijection `x_j ↦ h(x_j)` with f re-expressed accordingly leaves every G, hence φ, unchanged.
*Proof.* Direct substitution into the formulas of Lemma 3 and Cor. 4′. ∎ (U6, U6b, U6c)

**Lemma 6 (scope of the marginal law).** λ₁ is linear in mixtures of laws; λ_{k+1}, k ≥ 1, is not.
*Proof.* λ₁ is the mean, linear in F. For λ₂ = ½E|Y − Y′| (Y, Y′ iid F) and F = ½F₁ + ½F₂,
`E|Y − Y′| = ¼E|Y₁−Y₁′| + ¼E|Y₂−Y₂′| + ½E|Y₁−Y₂|`, hence
`λ₂(F) − ½(λ₂(F₁)+λ₂(F₂)) = ⅛[2E|Y₁−Y₂| − E|Y₁−Y₁′| − E|Y₂−Y₂′|] = ⅛·ED(F₁,F₂)`,
the energy distance, which is > 0 whenever F₁ ≠ F₂. For N(0,1), N(3,1) it is ≈ 0.47. ∎ (U7, U7b)
Consequence: the marginal law's L-moments are not sums of per-feature terms, so no per-feature
"stage-1" table exists for target M; only roles (comp, struct) are players there.

## 4. Theorems

**Theorem 1 (barycenter target: uniqueness of the TRACE budget).**
Under the setting of §1 (target B, additive coefficients, K channels in the orthogonal shifted
Legendre basis), there is exactly one allocation rule `Φ: G ↦ (Φ_{kjr})` satisfying F with a stage-1
rule obeying E1, S1, N1, L1 and a stage-2 value on Γ(N) obeying E2, S2, N2, L2. It is

    Φ_{kjr} = a_{kjr} · Δc̄_k / (2k+1),

with `a_{kjr}` from Lemma 3. It satisfies exactly (algebraic identities, no approximation):
`Σ_{j,r} Φ_{kjr} = (Δc̄_k)²/(2k+1)` for every k, and `Σ_{k,j,r} Φ_{kjr} = W = W2²(Q̄_A, Q̄_B)`;
it is unchanged by swapping the period labels or the order in which composition and structure
are switched, is equivariant under feature relabelling, invariant to y-translation and to
reparametrisation of features, and scales by σ² under y-scaling.

*Proof.* By L1–E1–S1–N1 and Lemma 3, α is unique and equals the stated a. By E2–S2–N2–L2 and
Lemma 2, ψ = Sh on Γ(N); by F, Φ = Sh(v_a), and by Lemma 4 and Cor. 4′,
`Φ_{kjr} = w_k a_{kjr} Δc̄_k`. Existence: Sh and the stage-1 Shapley satisfy their axioms (Lemma 2).
Budget: by Cor. 4′ the channel sums are `w_k(Δc̄_k)²`, and by Lemma 1 these add to
`‖Σ_k Δc̄_k P*_k‖² = ‖Q̄[B,B] − Q̄[A,A]‖² = W` (Lemma 3 gives `Σ_{j,r} a_{kjr} = Δc̄_k`).
Invariances: Lemma 5. ∎ (U5)

**Theorem 2 (marginal-law target: channel level only).** For target M with players
N = {(k,r)} plus, optionally, the residual channel, the unique rule satisfying F (with the role
game `g_k(S) = c_k[1{comp∈S},1{struct∈S}] − c_k[A,A]`), E1–L1 and E2–L2 is
`Φ_{kr} = a_{kr} Δc_k/(2k+1)`, with `Σ_{k<K,r} Φ_{kr} = Σ_k (Δc_k)²/(2k+1)`. If the residual
`R[s,t] = Q[s,t] − Σ_{k<K} c_k[s,t]P*_k` is added as one more function-valued channel with role pieces
`R_comp, R_struct` (Lemma 3 applied pointwise in u), then by Lemma 4
`Φ_{res,r} = ⟨R_r, R[B,B] − R[A,A]⟩` and the full sum equals `W2²(F_A, F_B)` exactly.
*Proof.* As Theorem 1 with p = 0 and the extra vector `R_r` (orthogonal to span{P*_k, k<K} because
it is a projection residual, so Cor. 4′ still separates it). ∎
No per-feature statement is made for target M (Lemma 6).

## 5. Each axiom is needed (counterexamples)

Each rule below satisfies all axioms of its stage except the one named, and differs from TRACE
(checked on random games with n = 2 and n = 3: X0 and X-lines).

| dropped | rule | why it satisfies the others | effect |
|---|---|---|---|
| E1/E2 | ½·Shapley | linear, symmetric, null-respecting | pieces sum to half the budget |
| S1/S2 | fixed-order path (comp first, then struct; the classic A-reference counterfactual split) | telescopes (E), null contributes 0 (N), linear (L) | result depends on the order; this is the order dependence the rivals show |
| N1/N2 | equal split `v(N)/n` | E, S, L hold | an inert feature receives budget |
| L1/L2 | Shapley²-proportional: `ψ_i = Sh_i² · v(N)/Σ_l Sh_l²` | E by construction, S and N inherited from Sh | not additive across games |
| F | Shapley of the single **switch game** `v(S) = w_k(Σ_j g_{kj}(S ∩ {(j,comp),(j,struct)}))²` | E2–L2 all hold; also exact | differs from TRACE whenever some feature has a comp×struct interaction `I_{kj} ≠ 0` (XF: max diff 0.92 on a random case); coincides when every `I_{kj} = 0` |
| orthogonality | same rule in a non-orthogonal basis (monomials) with per-channel terms `(Δc_k)²‖P_k‖²` | — | per-channel terms no longer sum to W2² (XO); Cor. 4′ fails because cross terms `⟨P_k,P_l⟩ ≠ 0` |

The F row is the important one: **E, S, N, L alone do not single out TRACE's budget**; they single
out the Shapley value *of whichever game you write down*. TRACE's choice is to explain coefficient
changes first (where the target is linear) and square afterwards. The switch game instead lets
the squared distance attribute interaction terms across features; both are exact and order-free.

## 6. Assumptions and scope (read before citing)

1. **Model, not truth.** Theorem 1 is about the barycenter of TRACE's *additive* coefficient model.
   If the true conditional quantile coefficients have feature interactions, the per-feature pieces
   explain the fitted model's barycenter, not the data.
2. **Barycenter ≠ marginal law.** W in Theorem 1 is `W2²` between the period barycenters (Vincent
   averages), not between the marginal laws. The gap (a mixing/Jensen term) is reported separately.
   Per-feature exactness exists only for the barycenter (Lemma 6).
3. **Truncation.** Theorem 1 is exact for `W2²` of the K-channel projections. Theorem 2 adds the
   residual to recover the full marginal `W2²`.
4. **Centring convention.** The split of structure between a feature's `f_{t,kj}` and the intercept
   depends on how f is centred (adding a period-specific constant to f_{t,kj} and subtracting it from
   b_{t,k} moves mass between (k,j,struct) and (k,0,struct)). Composition pieces are invariant to it.
   TRACE uses the pooled-mean centring (same as the benchmark harness); the uniqueness is conditional
   on that convention.
5. **Population vs estimate.** The theorems are algebraic; they hold exactly for plug-in estimates
   (that is what H5 tests), but statistical accuracy of DFL counterfactuals, fits and bands is a
   separate, asymptotic matter (SHIFT_V2 §3, §6).
6. **Shapley is a convention.** The axioms define a fair division, not a causal effect.
   Pieces can be negative (a feature pulling against the net shift) and are reported unclipped.
7. **Joint law of features.** For target B only the marginals of each x_j enter (additivity); changes in
   dependence between features with unchanged marginals produce no composition piece in the barycenter.
   They can still move the marginal law (Theorem 2).
