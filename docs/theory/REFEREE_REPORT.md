# Referee report: uniqueness of the TRACE shift budget

Independent adversarial review, written to a mathematics-journal standard.
I tried to break Theorems 1 and 2. I did not succeed in falsifying the stated
theorems. I did succeed in showing that the *slogan* of uniqueness is stronger
than the theorems, that one supporting identification is easy to misread, and
that several pre-registration bars are tautological, ambiguous, or slightly
incoherent as calibration targets.

**Reviewed.** `docs/theory/UNIQUENESS.md`, `docs/theory/verify/uniqueness_checks.py`,
`docs/benchmark/PREREG_TRACE_CLAIMS.md`.
**Context.** `docs/theory/SHIFT_V2.md` on `shift-v2` (`913f3e0`) and
`docs/benchmark/PREREG_SHIFTBENCH.md`.
**Independent numerics.** `docs/theory/verify/referee_checks.py` (seed 20261013).
Permutation-average Shapley, Hosking’s explicit `P*_k` sum, a 401-point open
grid. The file does not import or copy `uniqueness_checks.py`. All referee
assertions passed.

I have not edited `UNIQUENESS.md` or either pre-registration note.

---

## 0. Recommendation

**Accept the theorems as characterisations, not as uniqueness of “the” shift
budget.** Rewrite the title, the first paragraph, and Theorem 1 so that axiom F
is in the hypothesis, not in a footnote. State Lemma 6 only for `λ₂` of a
balanced two-component mixture. Do not let §6.2 be read as “the barycenter
versus marginal `W2²` gap equals one-eighth of the energy distance” — that
statement is false (R8f, R8g). Keep the honesty of §5 and §6; they already
contain most of the necessary caveats.

---

## 1. Verdict per claim

| claim | verdict | one-line reason |
|---|---|---|
| Lemma 1 (Parseval) | **correct** | orthogonal expansion in `P*_k`; R1 |
| Lemma 2 (Shapley’s theorem) | **correct** | standard; additivity plus E/S/N fixes the unanimity basis, including scalar multiples, without a separate homogeneity axiom; R2b |
| Lemma 3 (stage-1 formula) | **correct** | 2-player Shapley; R2, R2d |
| Lemma 4 (quadratic Hilbert game) | **correct** | Möbius support on pairs; `Sh_i = ⟨x_i, Σ x⟩`; R3 |
| Corollary 4′ (channel separation) | **correct** | orthogonality, not an extra axiom; R4, R4b |
| Lemma 5(a) period swap | **correct** | `a ↦ −a`, `Δc̄ ↦ −c̄`, bilinear `Φ` unchanged; R6a |
| Lemma 5(b) feature relabelling | **correct** | equivariance, not invariance; R6b |
| Lemma 5(c) `y ↦ σy + m` (`σ>0`) | **correct** | translation cancels in every difference; `Φ ↦ σ² Φ`; R6c–R6e. The `+m` lands only on `c_0` because `∫ P*_k = 0` for `k≥1` |
| Lemma 5(d) feature reparametrisation | **correct with fixes** | true for a bijection of a *single* `x_j` with `f` re-expressed, as a statement about `Φ(G)` (R6f). Not an estimator claim. A joint map of several features is out of scope and does change the per-feature split (R6g–h) |
| Lemma 6, `λ₂(½F₁+½F₂) − ½(λ₂F₁+λ₂F₂) = ED/8` | **correct** | algebra is right; `ED(N(0,1),N(3,1))/8 = 0.472217`; sample gap `0.47231` (R8–R8b). Identity is 50-50 and `λ₂` only (R8d, R8e) |
| “barycenter vs marginal gap `= ED/8`” | **wrong** | if that is how §6.2 is read. `W2²` of the 50-50 Gaussian mixture to its Vincent barycenter is `0.6846`, not `0.4722` (R8f). Two Gaussian *periods* have gap `0` (R8g) |
| Theorem 1, formula and budget | **correct** | `Φ_{kjr} = a_{kjr} Δc̄_k/(2k+1)` sums to barycenter `W2²`; R5 |
| Theorem 1, uniqueness | **correct with fixes** | unique *among maps of the form `ψ ∘ v_α ∘ α` with Shapley at both stages*. Not unique among exact, symmetric, efficient rules (R9) |
| Theorem 1, invariance list | **correct with fixes** | algebraic invariances hold; per-feature *structure* is unique only after a centring convention (R7) |
| Theorem 2 | **correct with fixes** | same F caveat; residual channel is the same lemma in an orthogonal complement; no per-feature claim, correctly (R12, R13) |
| Axiom independence table | **correct** | each dropped axiom has a rule that keeps the others (R10). The S1 sentence “equivalently, no role is first” is false as logic (R17) |
| F is a primitive fairness axiom | **correct with fixes** | F is architectural. It is the axiom that *selects the game*, reverse-engineered from SHIFT_V2’s two-stage formula |
| Switch-game Shapley undermines “unique” | **correct** as a critique of the slogan | both rules are exact and order-free; they differ iff some `I_{kj} ≠ 0` (R9). Explicit `p=1` gap: TRACE `(2,1)` vs switch `(5/3,4/3)` on `w_k=1/3` |
| Centring dependence, §6.4 | **correctly disclosed** | composition is invariant, structure moves between feature and intercept (R7). Uniqueness of per-feature structure is conditional |
| Joint law of features, §6.7 | **correctly disclosed** | same marginals, different copula: barycenter budget `≈0`, marginal `W2² = 0.476` (R13) |
| H5 bars | **correct with fixes** | coherent as a software exactness check; item 1 is tautological given E1 (R5d); mislabelled as a hypothesis |
| H6 bars | **correct** | conservative; best-rival-on-the-same-runs favours rivals; `+0.05` Spearman is arbitrary but not rigged |
| H7 bars | **correct with fixes** | planted-`k` lists are impure; `e_k ≤ 0.05` is loose relative to several signals; primary rival test is on `sd`, not on the planted shape channels |
| H8 bars | **correct with fixes** | union FPR of two 95% bands is not a 5% object (R15c, union `0.080` at `ρ=−0.75`); Clopper–Pearson convention unspecified; two-sided 95% upper at `10/200` is `0.09003 > 0.09` |

---

## 2. Lemma-by-lemma

### Lemma 1

Correct. Shifted Legendre `P*_k(u) = P_k(2u−1)` is orthogonal on `[0,1]` with
`‖P*_k‖² = 1/(2k+1)`. For `D = Σ_k d_k P*_k`, `‖D‖² = Σ_k w_k d_k²` is
Parseval. The authors’ U1 and the referee’s R1 agree (grid gap `4×10^{-6}` on
401 nodes; the identity is exact).

### Lemma 2

Correct. The unanimity games `u_T` are a basis of `{v : v(∅)=0}` by Möbius
inversion. Efficiency, symmetry and the null-player axiom fix
`ψ_i(m u_T) = m/|T|` for `i ∈ T` *without* a separate homogeneity axiom,
because `m u_T` is itself a game whose worth at `N` is `m`. Additivity then
extends the value to every game. R2b reproduces the forced 2-player values
`(1,0)`, `(0,1)`, `(½,½)`.

### Lemma 3

Correct. On the role game `g(S) = G[1_{comp∈S}, 1_{struct∈S}] − G[A,A]`,
Shapley is the displayed average of the two paths. The interaction
`I = G[B,B]−G[B,A]−G[A,B]+G[A,A]` is exactly the path disagreement on each
coordinate, and is split `½/½` (R2d). Summing over features including the
intercept recovers `Δc̄_k` by E1 (R5d).

### Lemma 4 and Corollary 4′

Correct. Expand `‖Σ_{i∈S} x_i‖²` in the unanimity basis (singletons and pairs
only). Shapley of `u_{{i}}` and `u_{{i,ℓ}}` gives
`Sh_i = ‖x_i‖² + Σ_{ℓ≠i} ⟨x_i,x_ℓ⟩ = ⟨x_i, Σ x⟩`. With
`x_{kjr} = a_{kjr} P*_k` and `Σ_{j,r} a_{kjr} = Δc̄_k`, orthogonality kills
cross-channel inner products and yields `Φ_{kjr} = w_k a_{kjr} Δc̄_k`.
Channel-level efficiency is a theorem, not an axiom. R4/R4b confirm. XO / R11
confirm that a non-orthogonal basis (monomials) makes the naive per-channel
sum fail.

A remark, not a correction: once F has replaced the data by coefficient pieces
`a_{kjr}` in an orthogonal basis, stage-2 Shapley on the induced game is the
same rule as *signed proportionality* of the Parseval term,
`Φ_{kjr} = a_{kjr} / Δc̄_k · w_k (Δc̄_k)²`. The Hilbert-space argument is the
right one, but it is heavier than the formula it produces.

### Lemma 5

**(a)(b)(c)** hold as algebraic identities of the closed form (R6a–e).
Translation invariance is because `m` is added to every `c̄_0[s,t]` and cancels
in every difference; scaling `σ>0` is bilinear in `(a, Δc̄)`.

**(d)** holds for the map `G ↦ Φ(G)` under a bijection of one coordinate with
the same `f` rewritten (R6f: `x ↦ x³` on `(0,1)`, tables agree to `10^{-12}`).
Two qualifications the text should make:

1. This is not a claim about any estimator of `f`. A spline or a linear fit on
   the raw `x_j` will move under a monotone reparametrisation.
2. A *joint* reparametrisation is outside the claim and does change the
   per-feature split. A 45° rotation of two features with `y = x_0^2 + x_1^2`
   moved the per-feature `a` by `0.012` while leaving `Σ_j a` invariant (R6g,h).
   That is the usual rotation problem of additive decompositions, not a
   contradiction.

### Lemma 6

The identity as written is correct, and is only about `λ₂` of a *balanced*
mixture. Write `λ₂(F) = ½ E|Y−Y′|` and `F = ½ F₁ + ½ F₂`. Expanding the four
pair-types gives

```
λ₂(F) − ½(λ₂(F₁)+λ₂(F₂))
  = ⅛ ( 2 E|Y₁−Y₂| − E|Y₁−Y₁′| − E|Y₂−Y₂′| )
  = ⅛ ED(F₁,F₂).
```

Closed form: `ED(N(0,1), N(3,1)) = 3.777733`, so `ED/8 = 0.472217`, matching
the text’s “≈ 0.47”. An 80,000-draw PWM estimator of the `λ₂` gap is
`0.47231` (R8b). `λ₁` is linear (R8c). The identity fails for unequal mixture
weights (5:1 gap `0.2634 ≠ 0.4722`, R8d) and is not a statement about `λ₃`
(unequal-scale pair: `λ₃` gap `0.1853` vs `ED/8 = 0.3734`, R8e).

**What Lemma 6 is not.** It is not the gap between barycenter `W2²` and
marginal `W2²`.

- The Vincent barycenter of `{N(0,1), N(3,1)}` is `N(1.5,1)`. The 50-50
  mixture is a different law. `W2²(mixture, barycenter) = 0.6846 ≠ 0.4722`
  (R8f).
- If the two *periods* are themselves `N(0,1)` and `N(3,1)`, barycenters
  coincide with the laws and the gap is `0`, while `ED/8` is still `0.47`
  (R8g).

§6.2 says the barycenter–marginal gap is “a mixing/Jensen term” and is
“reported separately.” That is the right object, and it is not `ED/8`. The
authors’ own U7 only checks that the `λ₂` gap is `> 0.1`; it never checks the
factor `1/8` or the `W2²` reading. Anyone citing Lemma 6 as “the” gap between
the two targets will be wrong.

Consequence of Lemma 6 for per-feature statements on the marginal law: correct.
`λ_{k+1}` for `k≥1` is not linear in mixtures, so there is no per-feature
stage-1 table for target M.

---

## 3. Theorem 1

**Formula and budget: correct.** Under the additive coefficient model,
`Σ_{j,r} a_{kjr} = Δc̄_k` and `Φ_{kjr} = a_{kjr} Δc̄_k/(2k+1)` sums to
`Σ_k (Δc̄_k)²/(2k+1) = W2²(Q̄_A, Q̄_B)` of the *K-channel barycenters* (R5).
Negative pieces occur and must stay unclipped (R16). Period swap and
comp/struct order-average leave `Φ` unchanged (R6a, Lemma 3).

**Uniqueness: correct relative to F; overstated as a slogan.**

The proof is: stage-1 axioms ⇒ `α` is 2-player Shapley (Lemma 3); stage-2
axioms on *all* of `Γ(N)` ⇒ `ψ` is Shapley (Lemma 2); F ⇒ `Φ = Sh(v_α(G))`;
Lemma 4 + orthogonality ⇒ the formula. Existence is immediate. I do not have a
counterexample inside this class.

Uniqueness of `ψ` on all TU games is stronger than needed. The induced games
`v_a` have Möbius support on sets of size at most 2, and that linear family is
already closed under addition, so E/S/N/L uniquely determine Shapley *on the
games that arise*. That is a repair, not a break.

What *does* break the slogan is the authors’ own §5. Efficiency, symmetry,
null-player and additivity do not select a game. They select the Shapley value
of whichever game one writes down. TRACE’s game is “Shapley-then-square”:
collapse each `2×2` to `(a_comp, a_struct)`, then play
`v_a(S) = ‖Σ_{i∈S} a_i P*_k(i)‖²`. The switch game

```
v(S) = w_k ( Σ_j ( G_j[comp on in S, struct on in S] − G_j[A,A] ) )²
```

is the `W2²` of the barycenter after a subset of `(feature, role)` switches.
It satisfies E2–L2, is exact, and is order-free. It differs from TRACE
whenever some interaction `I_{kj} ≠ 0`, and coincides when every `I_{kj} = 0`
(authors’ XF; referee R9).

A closed counter-example, `p = 1`, `w_k = 1/3`,

```
G = [[0, 0], [1, 3]],   I = 2.
```

TRACE: `a = (2, 1)`, `Φ = (2, 1)`.
Switch-Shapley: `(5/3, 4/3)`.
Both sum to `3`. The gap is `1/3`, not noise (R9b,c). For two random features
the coordinatewise max gap was `0.71` (R9e). The Owen value of the same switch
game (coalition-consecutive orders, one block per feature) is a third exact
rule and differed from both, max `|Owen−TRACE| = 0.42` (R9h).

So F is doing almost all of the uniqueness work. F is not a fairness axiom in
the sense of E/S/N/L. It is the decision to attribute on the linear
coefficient scale and square afterwards. That decision is already the
architecture of SHIFT_V2 §§4–5. The characterisation is valid. It is
reverse-engineered.

Two further limits, both already in §6, that a reader of “the unique budget”
will miss:

1. **Model, not truth.** The pieces explain the barycenter of the *fitted
   additive model*. Feature interactions in the true conditional quantiles are
   outside the theorem.
2. **Barycenter, not the observed law.** With identical one-dimensional
   marginals and a changed copula, `y = x_0^2 + x_1^2` has barycenter
   `Δc̄ ≈ 0` (`0.002` on 60,000 draws) and marginal `W2² = 0.476` (R13).
   Theorem 1’s exact budget can be numerically zero while the outcome law
   moves. §6.7 states this; the uniqueness claim should repeat it in the
   theorem box.

**Stage-2 additivity is not additivity in the tables.** `Φ(G+G′) ≠ Φ(G)+Φ(G′)`
(R18, gap `4.8` on a random draw). L2 constrains a linear extension to
fictional games, not the allocation’s behaviour on data. That is the usual
critique of Shapley’s additivity axiom; it is not a proof error.

---

## 4. Theorem 2

**Correct, with the same F caveat, and correctly weaker.**

Players are roles per channel, not features. Lemma 6 is the right reason: the
marginal law’s L-moments are not sums of per-feature terms. Adding the residual
`R[s,t] = Q[s,t] − Π_{<K} Q[s,t]` as one more function-valued pair of role
pieces, with Lemma 3 applied pointwise, gives
`Φ_{res,r} = ⟨R_r, ΔR⟩` and restores `W2²(F_A, F_B)` of the truncated-plus-
residual expansion (R12). Because each `R[s,t]` is a projection residual,
linear combinations `R_comp, R_struct` stay orthogonal to `span{P*_k : k<K}`,
so Corollary 4′ still separates the residual from the fitted channels (R12c).

The word “optionally” in the player set is inelegant (two theorems, two player
sets) but not wrong. The residual must be included if the claim is exact
marginal `W2²`; it must be omitted if the claim is the K-channel projection.
H5 item 2, which tests against empirical `∫(Q̂_B−Q̂_A)²`, is the right
empirical counterpart *only* when the residual is included.

---

## 5. Axioms: independence, S1 wording, naturalness of F

The dropped-axiom table is correct on `n=2` under an independent permutation
Shapley (R10): half-Shapley drops E; a fixed-order path drops S; equal split
`v(N)/n` drops N and keeps E, S, L; Shapley-squared-proportional drops L and
keeps E, S, N. I did not find a hidden dependence.

S1 is stated as: interchangeable roles get equal amounts. The next sentence,
“Equivalently, no role is privileged as first in the path `A → hybrid → B`,”
is **not** equivalent. On

```
G = [[0, 1], [1, 1]]
```

one has `g({comp}) = g({struct}) = 1`, so S1 applies and Shapley gives
`(½,½)`. The composition-first path gives `(1, 0)` (R17). The path-average is
the *solution* of the four axioms, not a restatement of S1. Delete
“Equivalently”.

F is necessary for the uniqueness claim the authors want, and they say so.
I would go further: F is the only non-standard axiom, and it is chosen so that
the object being characterised is exactly the rule of SHIFT_V2. A cleaner
paper treats F as a *definition of the game*, then cites Shapley once.

---

## 6. Centring

§6.4 is accurate and important. Adding a period-specific constant `γ_t` to
`f_{t,kj}` and subtracting it from `b_{t,k}` leaves every composition piece
unchanged and moves structure between `(k,j,struct)` and `(k,0,struct)` (R7:
budget invariant; composition tables agree to `10^{-10}`; structure moved by
`0.28` on the random draw, with the feature-plus-intercept sum conserved).
Pooled-mean centring of each `f_{t,kj}` (one reference measure, both periods)
pins the constants down. Period-specific centring would not. The harness
convention of centring the *features* `x` at the pooled sample mean is the
same convention only for maps that are linear in `x`. For nonlinear `f` the
two operations differ. The uniqueness theorem should say “unique given a
stated centring of each `f_{t,kj}`,” not “TRACE uses the same convention as
the harness.”

Composition of the intercept is identically zero if `G_{k0}` does not depend
on the covariate period (R5b). That is a sanity check, not extra content.

---

## 7. Pre-registration H5–H8

H1–H4 and the frozen rival specifications are untouched, as the addendum
claims. I am reviewing only whether H5–H8 are coherent and not rigged. I do
not treat a bar that is hard for TRACE as a virtue, or a bar that is easy as
proof of rigging; the question is whether the bar measures the claim.

### H5 — exact budget and order invariance

**Correct with fixes. Not rigged. Partly tautological.**

Items 1–4 at `1e-9 · max(1, W2²)` (after the frozen `/sd(y_A)²` scaling) are
the right tolerance for float64 algebra. A single-replication failure rule is
appropriate for an identity. Item 5’s split (`1e-9` algebraic, `1e-6`
end-to-end) is reasonable.

Item 1, `G_bar = |Σ Φ_{kjr} − Σ_k (Δc̄_k)²/(2k+1)|`, is an E1 bookkeeping
test: once `Σ_{j,r} a_{kjr} = Δc̄_k` is implemented, the gap is identically
zero (R5d). It does **not** independently reconstruct `∫(Q̄_B−Q̄_A)²` from
the barycenter quantile functions. A wrong barycenter that is internally
additive still passes. Item 2, against empirical marginal `W2²`, *is*
independent — if and only if the residual channel is in the sum, which the
formula does include.

Item 4 bundles two invariances (role order and period swap) that the theory
separates. Both should hold. Item 5 tests permutation of rows and feature
columns, not the reparametrisation of Lemma 5(d). That is a narrower
empirical claim than the theorem’s wording, and is the right one to test.

H5 is a deterministic software check and is labelled as such (“not a test”).
Calling it H5 in a family of inferential hypotheses is cosmetic, not rigging.

### H6 — stability of rankings

**Correct, not rigged.** The primary vector is the frozen mean-channel
`|comp_j| + |struct_j|`, which rivals also produce. The W2² per-feature vector
is secondary and is not used in the paired test — the honest choice. Best
rival is chosen on the same runs, which favours rivals. The “supported”
label requires mean paired Spearman difference `≥ +0.05` *and* both 95%
intervals above 0 in all four designs. That is conservative. `+0.05` is
arbitrary; say so. Failure is defined as any of the four intervals strictly
below 0; a straddle is “not supported.” Those three labels are distinct and
usable.

### H7 — shape shifts at fixed mean

**Correct with fixes. Not rigged in TRACE’s favour; several bars are loose or
impure.**

The new scenarios keep `E[y] = 1` in both periods by starting from the frozen
linear baseline and scaling `x_0` about mean 0. That part is clean. Oracle
truth uses both counterfactuals and the order-free split; rivals remain on
their frozen A-reference estimands. That is the right pairing.

Problems:

1. **Planted channels are impure.** Standardising a law to mean 0 and variance
   1 does not freeze `λ₂`. Referee samples (`n = 120,000`):

   | law | `λ₂` | `λ₃` | `λ₄` | `|Δλ₂|` vs `N(0,1)` |
   |---|---|---|---|---|
   | `N(0,1)` | `0.5640` | `~0` | `0.0690` | — |
   | centred lognormal `s=0.6` | `0.4988` | `0.1435` | `0.0933` | **`0.0652`** |
   | scaled `t_5` | `0.5346` | `~0` | `0.1030` | **`0.0295`** |

   So `skew_struct` is not “`k=2` plus small `k≥3`”: `k=1` moves by `0.065`,
   which is larger than H7’s `e_k ≤ 0.05` bar *before* `/sd(y_A)` and is not
   small relative to `Δλ₃ = 0.143`. `tail_struct` is not a pure `k=3` shift:
   `Δλ₂ = 0.030` and `Δλ₄ = 0.034`. The table’s “(+ small `k ≥ 3`)” for skew
   is ambiguous about whether `k=3` is a planted recovery target. Freeze the
   exact set `{k}` per scenario before looking at output.

2. **The `e_k ≤ 0.05` bar is loose after `/sd(y_A)`.** `sd(y_A) = √3.25 ≈ 1.80`.
   Several planted L-moment moves, divided by that scale, are below `0.05`. A
   zero estimator then passes the *error* half of “recovery” and is only
   caught by the structure-fraction or sign requirement. Say that recovery is
   carried by `f_k` (or by signs on `sd_offset`), not by `e_k`.

3. **`f_k = struct_k/(comp_k+struct_k)` is a signed ratio.** It is undefined
   when the net channel is near 0. Excluding `sd_offset` is necessary and
   done. Any other near-null channel (including an unplanted `k=1` on
   `tail_struct`) has the same pathology and is still exposed if someone later
   reports `f_k` there.

4. **The primary paired test against rivals is on the frozen `sd` split**, not
   on the L-moment channels H7 is about. `skew_struct` and `tail_struct` are
   designed so that variance is essentially unchanged; the `sd` test is then
   a near-null comparison. Support requires the interval below 0 in at least
   4 of 6 scenarios — i.e. TRACE must win the four scale-moving scenarios and
   treat the two shape scenarios as coin-flips on `sd`. That is fragile, and
   it is not a test of shape recovery. It is conservative for a “shape”
   claim, not a gift.

5. The mean-channel guard `|comp̂_0| + |struct̂_0| ≤ 0.05·sd(y_A)` is the
   right “no spurious location” check for these DGPs (both period means are
   1, and the maps are linear, so both mean-pieces are truly near 0, not
   merely offsetting).

Recovery rates `≥ 0.80` at `(20,000, 20)` and `≥ 0.60` at `(2,000, 5)` are
lenient, especially with (2). Coverage `≥ 0.90` (H7) versus H8(b)’s pooled
`≥ 0.90` and per-cell `≥ 0.85` is a duplicated, slightly different bar for
overlapping designs. Pick one.

### H8 — no-shift control

**Correct with fixes. The mean-channel FPR bar is not a coherent 5% target.**

**(a)** Three FPR objects are defined: mean-channel (union of composition and
structure bands); uniform band over `k = 0..3`; per-feature union over `p`
features. The mean-channel bar is: point estimate `≤ 0.05` on 200 null
replications *and* a Clopper–Pearson 95% upper bound `≤ 0.09`.

- “Any composition or structure band excludes 0” is a *union* of two 95%
  bands. Even with `corr(comp, struct) = −0.75` (they sum to a near-zero
  `Δ` under the null), a 20,000-draw Gaussian simulation gives union FPR
  `0.080` (R15c). A properly calibrated pair of 95% bands is not a 5%
  union test. Either the bar should be `≈ 0.08–0.10`, or the object should
  be a single 2-vector uniform band.
- “Clopper–Pearson 95% upper bound” does not say one-sided versus the upper
  end of a two-sided interval. At the design point `10/200 = 0.05`,
  two-sided CP upper `= 0.09003 > 0.09` and one-sided CP upper `= 0.0833`
  (R15). A bar that the exact 5% operating point fails under the two-sided
  reading is not a usable calibration target. Fix the convention and the
  number together. `≤ 11/200` already fails the point-estimate half, so the
  intended operating point is `≤ 10/200` plus a bound that `10/200` must
  satisfy.

The uniform-band FPR `≤ 0.05` over four channels is the one object that
matches a 95% uniform band. The per-feature union bar `≤ 0.10` for `p = 5`
is stricter than a naïve union of five 5% tests and will fail unless the
per-feature bands are conservative or a uniform-over-features band is used.
That is hard, not rigged.

**(b)** Pooled coverage `≥ 0.90` with per-scenario-and-channel `≥ 0.85`, on
95% bands, is a slightly loosened coverage claim and is fine if *joint*
coverage of `(comp, struct)` is the object. If the indicators are pooled
across roles as well, `0.90` is loose for a 95% band. Say which.

**(c)** Real-panel placebo at `≤ 0.10`, secondary, with the dependence caveat,
is coherent.

**Holm family.** “H6, H7, H8 are the inferential family; the supported label
uses Holm across their primary tests.” H6 contributes four paired superiority
tests and H7 six more. H8’s primaries are a calibration bar and a coverage
bar. Those do not have the same p-value, and Holm does not apply to them
without an extra construction (the McNemar tests versus rivals are mentioned
and then not listed as primary). Either drop H8 from the Holm family or define
the p-values. Mixing them is incoherent. It is not rigging; it makes
“supported” either undefined or conservative in an unstated way.

Alpha `0.05` two-sided per comparison, then Holm, is otherwise standard.
Seeds, including the new streams, are frozen. That part is clean.

---

## 8. Severity-ranked issues

1. **High — uniqueness is relative to F.** The switch-game Shapley value (and
   its Owen variant) is exact, symmetric and efficient, and differs from TRACE
   on any table with a nonzero interaction. The title and Theorem 1 must say
   “unique among factor-through-coefficient rules,” not “the unique budget.”
   F is reverse-engineered from SHIFT_V2. This is not a proof error; it is an
   overclaim.

2. **High — `ED/8` is not the barycenter–marginal `W2²` gap.** Lemma 6 is
   correct for `λ₂` of a 50-50 mixture (`0.472217` on the stated Gaussians).
   `W2²(mixture, Vincent barycenter) = 0.6846`. Two Gaussian periods have gap
   `0`. §6.2 must not be citable as the `ED/8` statement. U7 does not test the
   constant `1/8`.

3. **Medium — per-feature structure is unique only after a centring
   convention.** Composition is invariant; structure moves between feature and
   intercept (R7). “Pooled-mean, same as the harness” identifies the convention
   for linear `f` only. Put the convention in the theorem.

4. **Medium — H5 item 1 is tautological.** It tests `Σ a = Δc̄`, not an
   independently computed barycenter `W2²`. Keep it as a software check; do
   not advertise it as evidence for Theorem 1 beyond implementation.

5. **Medium — H8(a) mean-channel FPR.** Union of two 95% bands is not a 5%
   test (simulated union `0.080` at `ρ=−0.75`). Two-sided CP upper at
   `10/200` is `0.09003`, which fails a `≤ 0.09` bar. Name the interval and
   raise the bound, or replace the object by one uniform band.

6. **Medium — H7 planted channels and the `sd` paired test.** Variance-fixed
   shape changes still move `λ₂` (`0.065` for the lognormal, `0.030` for
   `t_5`). The recovery table’s “`k` that move” is incomplete and the
   “(+ small `k ≥ 3`)” clause is not a frozen set. The primary rival
   comparison is on `sd`, which two of the six scenarios barely move. The
   `e_k ≤ 0.05` bar, after `/sd(y_A)`, is larger than several planted
   signals.

7. **Low — S1 “equivalently”.** Path-average is the solution, not the axiom
   (R17). Joint feature maps are outside Lemma 5(d) (R6g). Holm across H6–H8
   mixes superiority tests with calibration bars. Theorem 2’s “optionally”
   residual is two theorems in one sentence. Authors’ XF “max diff `0.92`” is
   one random draw, not a bound (a referee `p=2` draw gave `0.71`).

Nothing in the list is a hole in the algebra of Lemmas 1–4 or in the formula
of Theorems 1–2 as written.

---

## 9. Suggested fixes, and a cleaner theorem

**Edits to the note (for the authors; not made here).**

- Title: “A characterisation of the TRACE shift budget,” or “Uniqueness given
  factorisation through coefficient-scale Shapley pieces.”
- Delete “Equivalently” in S1.
- Lemma 6: state the hypotheses (two components, equal weights, `λ₂` only).
  Add one sentence: this is *not* `W2²(F, Q̄)` and *not*
  `W2²(F_A,F_B) − W2²(Q̄_A,Q̄_B)`. Put the `1/8` identity in U7, not only
  `gap > 0.1`.
- Theorem 1 box: “There is exactly one map of the form `Φ = ψ ∘ v_α ∘ α`
  with `α` satisfying E1,S1,N1,L1 and `ψ` satisfying E2,S2,N2,L2 on `Γ(N)`.
  It is `Φ_{kjr} = a_{kjr} Δc̄_k/(2k+1)`. The same four axioms applied to the
  switch game yield a different exact allocation whenever some `I_{kj} ≠ 0`.”
  Repeat: barycenter of the additive model; centring of each `f_{t,kj}` is
  part of the data; copula changes with fixed marginals are invisible.
- Theorem 2: split off the residual as a one-line corollary.
- Lemma 5(d): “bijection of a single coordinate, `f` re-expressed; statement
  about `Φ(G)`, not about an estimator.”

**Cleaner theorem.**

> **Theorem.** Let `G = (G_{kj}[s,t])` be the `2×2` tables of an additive
> coefficient model in the orthogonal shifted-Legendre basis, with each
> `f_{t,kj}` centred at a fixed reference measure. Let `α` be the unique
> 2-player value satisfying efficiency, symmetry, null-player and additivity,
> and write `a = α(G)`, `Δc̄_k = Σ_{j,r} a_{kjr}`. Let `v_a` be the game
> `v_a(S) = ‖Σ_{(k,j,r)∈S} a_{kjr} P*_k‖²` on `N = {(k,j,r)}`. Let `ψ` be
> the unique value on all TU games satisfying Shapley’s four axioms. Then
> `ψ(v_a)_{kjr} = a_{kjr} Δc̄_k/(2k+1)`, the pieces sum to
> `W2²(Q̄_A, Q̄_B)` of the period barycenters, and this is the unique map
> `G ↦ ψ(v_{α(G)})`. The same four axioms, applied instead to the switch
> game whose worth is the barycenter `W2²` after a subset of
> `(feature, role)` switches, give a different exact allocation unless every
> interaction `I_{kj}` vanishes. Per-feature exactness is for the barycenter
> of the additive model only; it depends on the centring of `f`; and it is
> blind to dependence among features with fixed marginals.

That is the theorem the proofs already establish.

**Pre-registration (also not edited here).**

- H5 item 1: keep, but describe it as “E1 / implementation identity.” Add an
  optional independent barycenter check `Σ Φ` versus `∫(Q̄_B−Q̄_A)²` from the
  fitted quantile functions.
- H7: freeze `{k}` per scenario; report `λ₂` leakage for the variance-fixed
  shape laws; either move the paired rival test onto the planted L-moment
  channels or stop calling H7 a shape-recovery comparison to rivals.
- H8(a): one uniform mean-channel band, or a union bar near `0.08–0.10`;
  specify one-sided versus two-sided CP and set a bound that `10/200`
  satisfies (`0.091` if two-sided, `0.09` if one-sided).
- Holm: H6 and H7 paired tests only.

---

## 10. What I could not break

I did not find a counter-example to Parseval, to 2-player or Hilbert-space
Shapley, to channel separation under orthogonality, to the closed form of
Theorem 1 inside the class F+E+S+N+L, to the residual extension in Theorem 2,
to translation invariance, to `σ²` scaling, to period-swap invariance, or to
the necessity of each named axiom in the authors’ table. The authors’
`uniqueness_checks.py` is a fair check of those identities (U1–U6, X-lines).
Its gaps are the ones above: U7 does not test `ED/8`; U6 does not test
coordinate reparametrisation or isolate translation from scaling; XF is one
draw; nothing in that script is a check of H5–H8.

---

## 11. Numerical record

`python3 docs/theory/verify/referee_checks.py` — all assertions PASS
(seed `20261013`, numpy only). Selected constants:

| id | quantity | value |
|---|---|---|
| R8 | `ED(N(0,1),N(3,1))` | `3.777733` |
| R8 | `ED/8` | `0.472217` |
| R8b | sample `λ₂` gap, `n=8×10^4` | `0.47231` |
| R8d | 5:1 mixture `λ₂` gap | `0.2634` |
| R8f | `W2²(mixture, N(1.5,1))` | `0.6846` |
| R9 | TRACE vs switch, `G=[[0,0],[1,3]]`, `w=1/3` | `(2,1)` vs `(5/3,4/3)` |
| R9e | `p=2` random max `‖Φ_sw−Φ_tr‖_∞` | `0.7127` |
| R9h | max `‖Owen−TRACE‖_∞`, same draw | `0.4226` |
| R13 | copula `W2²` marginal / barycenter | `0.4758` / `~0` |
| R14 | skew vs Gaussian `Δλ₂`, `Δλ₃` | `0.0652`, `0.1430` |
| R14 | `t_5` vs Gaussian `Δλ₂`, `Δλ₄` | `0.0295`, `0.0340` |
| R15 | CP upper at `10/200`, two- / one-sided | `0.09003` / `0.08333` |
| R15c | union FPR of two 95% bands, `ρ=−0.75` | `0.080` |
