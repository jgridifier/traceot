# Research board v4: root causes of TRACE's measured shortcomings, cross-field fixes, and a v4 architecture

Scope: classical statistics and classical ML only (no neural networks, no large pretrained models). Interpretability must survive every change. Inputs read on `main` on 2026-10-14: `docs/theory/THEORY.md`, `SHIFT_V2.md`, `UNIQUENESS.md` (v2, incl. Theorem 4), `REFEREE_REPORT.md`, `FIXES2_PLAN.md`, `docs/benchmark/TRACE_CLAIMS_RESULTS.md`, `docs/research/LANDSCAPE.md`. Every citation is listed in [REFS_V4.md](REFS_V4.md) with the URL where it was checked; inline references are `[R#]`.

Evidence labels (same as LANDSCAPE): **[P]** proven (theorem in a cited paper, or derived here with stated assumptions); **[E]** empirical in a cited paper; **[T]** measured in this repo; **[C]** conjecture, to be tested by the stated bar.

Teaching version: [`docs/learn/research_board.html`](../learn/research_board.html).

---

## 0. Executive summary

The eight shortcomings (a)–(h) come from **five root causes**, not eight:

| # | Root cause | Shortcomings it drives |
|---|---|---|
| RC1 | **Wrong projection.** Pinball/CRPS fitting gives a density-weighted projection of Q(u\|x) onto the Legendre basis, while the reported target (mean, L-moments, barycenter W2²) is the unweighted L²[0,1] projection. | (a), (d) shrinkage, part of (b) |
| RC2 | **Barycenter ≠ mixture.** TRACE decomposes the W2 barycenter `E_x Q(u\|x)` (averaging quantile functions), but a composition change also moves the *marginal* law through the spread of conditional locations (averaging CDFs). That "mixing" effect has no player in the game. | (c), (f), the l3/l4 coverage cells in (d) |
| RC3 | **Inference on a quadratic functional by refitting.** W2² is quadratic in an estimated difference; its plug-in is biased upward by tr Var, the refit bootstrap re-creates RC1's bias, percentile bands are not studentized, and every replicate refits the booster. | (d), (e), (h) |
| RC4 | **Reweighting balances the wrong thing.** The composition counterfactual uses a cross-fitted logistic period-odds model (linear index). The additive barycenter needs balance of each feature's *bin marginals*; the mixture target needs joint balance. The logistic index gives neither exactly. | (g), part of (d) |
| RC5 | **Non-identified, path-dependent per-feature split.** Cyclic greedy boosting with correlated features splits shared signal arbitrarily (concurvity); per-feature pieces are products `w_k Δc̄_k a_kj` whose rank is noise-dominated when `Δc̄_k` is small; one bag. | (b) |

**Top 5 modifications (v4 core)**, ranked by (expected gain × chance of best-in-class) / cost:

1. **M1 Mixture-lift game**: add a *mixing* player per feature (exact numerical version plus a closed-form second-order "covariance attribution"), so composition-driven shape shifts are visible. Targets (c), (f). Bar: H7 comp cells ≥ 0.6 / 0.8 recovery and paired error vs CFM with CI ≤ 0 in ≥ 4/5 cells.
2. **M2 L²-projection fitting**: squared-loss channel 0 plus PIT orthogonal-series pseudo-outcomes for k ≥ 1 with a Sobolev ridge across k, then PIT-isotonic recalibration. Targets (a), (d). Bar: MSE ratio vs EBM upper CI ≤ 1.00; channel ratios fitted/true in [0.95, 1.05] for k ≤ 3; CRPS ≤ EBM+conformal in ≥ 2/3 cases.
3. **M3 Orthogonal inference without refits**: split-product unbiased W2², one-step influence-function linearisation, multiplier bootstrap at bin level, studentized randomization null; cheap-bootstrap fallback. Targets (d), (e), (h). Bar: pooled coverage ≥ 0.88, every cell ≥ 0.85; null FPR CP-upper ≤ 0.09 at 200 reps; band time ≤ 2× one fit at 1e6×100.
4. **M4 Bin-marginal balancing weights** (raking / entropy balancing on TRACE's own bins; Riesz-regression for the mixture target). Targets (g), (d). Bar: composition-piece error ≤ 0.5× current on H7 comp cells and on frozen ShiftBench per-feature error.
5. **M5 Stability layer**: bin-ridge + bagging + purification + Owen blocks for correlated features + stability-selection reporting. Targets (b). Bar: bootstrap rank stability ≥ 0.77 (OB) and split-half ≥ 0.46; per-feature error ≤ 0.09.

Proposed v4 architecture in §4.

---

## 1. TRACE decomposed into components, and the root cause per shortcoming

### 1.1 Components

| Component | Current choice | Where |
|---|---|---|
| C1 Basis | shifted Legendre `P*_k`, K = 6, channels = L-moment scaled coefficients; W2² = Σ_k w_k Δc_k² (Parseval, `w_k = 1/(2k+1)`) | THEORY §1.1 |
| C2 Fitting loss | integrated pinball (≈ CRPS/2) on GL nodes; optional `pit_rounds` refit (F1, pending v3) | THEORY §1.5, FIXES2 F1 |
| C3 Booster / regularization | cyclic binned first-order boosting, lr 1.0, 40 rounds, patience 8, one bag, centring, gain screen, PAV for monotone Q | TRACE_SIMS |
| C4 Reweighting / counterfactual | cross-fitted logistic period odds (DFL), self-normalized, trim | SHIFT_V2 §3, FIXES2 F2 |
| C5 Attribution game | reference-symmetric switch game on the barycenter, Φ_kjr = w_k Δc̄_k a_kjr (Theorem 4) | UNIQUENESS |
| C6 Inference / bands | refit bootstrap (B_refit = 100) and marginal bootstrap, percentile bands | SHIFT_V2 §5 |
| C7 Null test | permutation floor on total W2²; band-based joint test | SHIFT_V2, H8 |
| C8 Entry / exit | unbalanced: entry, exit, stay pieces summing to Δc̄ | SHIFT_V2 §7 |

### 1.2 Root cause per shortcoming

**(a) Mean MSE 1.01–1.05× EBM; CRPS behind EBM+conformal.** [T] for the numbers.
*Root cause RC1 (C2, C3).* The pinball first-order condition is `∫ (F(Q(u|x)) − u) e_k(u) du = 0`, i.e. to first order `∫ f(Q*(u|x)) (Q − Q*) e_k du = 0` (FIXES2 §1). The fitted channel 0 is a density-weighted average of quantiles, not `E[Y|x] = ∫ Q du`. Squared error is minimized only by the L² projection, so any skew or tail noise leaks into the mean [P: the minimizer of E(Y−m)² is E[Y|x]]. Second, CRPS is a *marginal-calibration plus sharpness* score [R11]; raw TRACE is over-dispersed (0.952 at nominal 80%, TRACE_SIMS) because centring removes the shared channel step until late rounds, and global conformal shifting cannot fix dispersion. EBM+conformal wins because it is calibrated by construction [R26].

**(b) Per-feature stability poor (bootstrap 0.50 vs OB 0.77; split-half 0.19 vs 0.46; per-feature error 0.21 vs 0.04–0.09).** [T]
*Root cause RC5 (C3, C5), plus RC1.*
1. *Non-identification under concurvity.* With correlated x_j, x_l, the additive model is identified only up to functions in `span(f(x_j)) ∩ span(g(x_l))` restricted by the bin grid; cyclic greedy boosting with early stopping picks one split by path, and a bootstrap resample changes the path [P: non-uniqueness of backfitting fixed points under concurvity; purification gives a canonical representative, R33].
2. *Product noise.* `Φ_kj = w_k Δc̄_k a_kj`: when `Δc̄_k ≈ 0` for a channel, a relative error ε in `Δc̄_k` multiplies every feature's piece; ranks across features *within* a channel are preserved but ranks over (k, j) are not. Shape channels (k ≥ 2) have the largest relative noise (RC1 shrinks them by 0.5–0.85 and their variance is high).
3. *Instability of greedy selection* is a known property [R31 Yu 2013, "Stability"]; EBM averages 8+ outer bags, TRACE uses one.
4. *Per-feature error 0.21* is largely target mismatch: the truth is defined on the marginal (CFM-type) composition, but TRACE's per-feature pieces are barycenter pieces (RC2) built from logistic-index weights (RC4).

**(c) Shape-shift recovery 1/10 cells; CFM much better on composition-driven shape shifts.** [T]
*Root cause RC2 (C5).* Worked statement, location–scale conditional law `Y = μ(X) + σ ε`, ε with density f, quantile q(u):
- Barycenter: `Q̄(u) = E μ(X) + σ q(u)`. Its channels k ≥ 1 do not depend on the law of X at all. A composition change moves only channel 0.
- Mixture (marginal law): `F_M(y) = E_X F((y − μ(X))/σ)`. With `δ = μ(X) − Eμ(X)`, `v = Var δ` small relative to σ², a second-order expansion gives [P, Taylor with bounded f''; derived in §3 M1]
  `Q_M(u) = Q̄(u) − (v / 2σ) · f'(q(u)) / f(q(u)) + O(E|δ|³/σ²)`.
  The correction is a fixed function of u times `Var_x μ(X)`. Its Legendre coefficients are non-zero in every channel whose basis function correlates with the location score `−f'/f` evaluated at q(u) (for skewed ε, channels 2 and 3 especially).
So when composition changes the *spread* of conditional locations, the marginal law changes shape, but the barycenter (and hence every TRACE player) books nothing in k ≥ 1. The H7 numbers match exactly this: skew_comp error 0.078–0.083 for both TRACE and DFL-on-TRACE, 0.012–0.022 for CFM which averages CDFs; skew_comp l3 band coverage 0.087. This is a **target** gap, not variance: it does not shrink with n (0.078 → 0.083 from n = 2e3 to 2e4).

**(d) Band coverage 0.81 vs 0.90; tail channels worst; channels k ≥ 1 shrunk to ≈ 0.6.** [T]
*Root causes RC1 + RC2 + RC3 (C2, C6).* (i) The refit bootstrap reproduces the RC1 shrinkage, so it is centred on the wrong value (FIXES2 table: k = 4,5 ratios 0.5–0.66 at n = 2e4–8e4). (ii) Cells with coverage 0.09–0.40 are exactly the composition-driven shape cells (RC2): no band centred on the barycenter covers a mixture truth. (iii) Squared terms: for `D̂ = D + e`, `E‖D̂‖² = ‖D‖² + tr Var(e)` [P], so plug-in W2² pieces are biased up by the noise trace and the percentile bootstrap doubles that bias rather than removing it [P: bootstrap of a smooth function at a boundary / degenerate point is inconsistent, R39 Fang & Santos]. (iv) Tail channels have heavy-tailed estimators (Legendre P_k weights the extreme u most), so non-studentized percentile intervals under-cover [P: Edgeworth, first-order error of percentile intervals O(n^{-1/2}) for skewed statistics]. (v) entry_exit|sd coverage 0.500: the sd-type functional is non-linear in the entry/exit weights, linearised nowhere.

**(e) Null size ≈ 6.8% vs 5% (rivals similar).** [T]
*Root cause RC3 (C7), harness-level.* The failing statistics are the band-based mean-channel tests, and every rival using the same union/band rule shows 6.8–7.8%, while the permutation floor itself was calibrated at 21/400 = 5.25% (FIXES2 B5). So the excess is from deciding "shift" by "percentile band excludes 0", which is anti-conservative for small n and for non-pivotal statistics [P: percentile intervals are not exact; R38 Chung & Romano for the studentized fix]. Note 13/192 is not significantly above 5% (Holm p = 0.17); the frozen bar is strict. [C] for the attribution of cause to the band rule; the permutation-floor evidence is [T].

**(f) Marginal vs model W2² gap up to 3%.** [T]
*Root cause RC2 (C5) plus additive-misspecification.* The model barycenter and the marginal law differ by the mixing term above, whose size is ∝ `Var_x μ(X)/σ²` in each period; plus whatever the additive model misses (interactions). This is predicted, not a bug: Lemma "marginal shift is not an average" (THEORY §2.2).

**(g) Reweighting weak for many features.** [T]
*Root cause RC4 (C4).* Under the additive barycenter, `G_kj[s,t] = E_{x~F_s} f_{t,kj}(x_j)` depends on F_s only through the **bin marginal** of x_j (UNIQUENESS §1). A logistic model with a linear index in raw x balances the first moments of x, not the bin marginals; any nonlinear composition change inside a feature's range (e.g. a change in skew of x_j at fixed mean) is mis-weighted. For the mixture target the needed object is the full density ratio, which a linear index also misses. Trimming adds first-order bias in the trimmed mass (FIXES2 §1).

**(h) Bootstrap refits costly.** [T]
*Root cause RC3 (C6).* B refits × 17 s at 1e6×100 (B = 100 → ~30 min). Refits are needed only because inference is not linearised.

---

## 2. Cross-field survey: what each field offers TRACE, with an honest verdict

Verdict scale: **Adopt** (core of v4), **Use** (supporting role), **Note** (correct but no gain for 1-D outcome), **Drop**.

### 2.1 Physics
- **JKO gradient flows / Fokker–Planck** [R1] and **Langevin as composite optimization in measure space** [R2]. A JKO step is a proximal step in W2. In 1-D, W2 is flat: the quantile map is an isometry onto a convex cone of L²[0,1], so a JKO step on a convex energy is a *proximal-gradient step in quantile space*. For TRACE's booster this means: a ridge-type proximal step per round (= Sobolev ridge of M2) is the W2-natural regulariser, and its fixed point is unique (strong convexity) → helps (b). **Use** (as the justification of M2's ridge, not as a new solver).
- **Entropy-regularized OT / statistical mechanics** [R3 Cuturi; R5 Genevay; R24 Mena & Niles-Weed]. Gives smooth, fast transport and CLTs, but in 1-D exact W2 is already O(n log n). The useful idea is **debiasing**: the Sinkhorn divergence subtracts self-transport terms to remove the entropic bias [R4 Feydy]. The analogue for TRACE's quadratic functional is the split-product estimator (M3). **Use** (idea only).
- **Renormalization / multiscale** [R22 Schmitzer, multiscale scaling; R30 wavelet shrinkage]. Coarse-to-fine bins with shrinkage of fine-scale contrasts (Haar) give stable shape functions where data are thin. **Use** in M5 (optional multiscale bins).

### 2.2 Fluids
- **Benamou–Brenier dynamic OT** [R6]. In 1-D the geodesic between two quantile functions is the straight line `Q_τ = (1−τ)Q_A + τQ_B`. Result (derived §3 M8): the Aumann–Shapley path attribution along this geodesic in the space of G-tables equals Φ, giving a continuous-time characterization of TRACE's budget. **Use** (theory, cheap; strengthens uniqueness).
- **Helmholtz–Hodge decomposition** [R7]. Splits a velocity field into gradient, divergence-free and harmonic parts. For a 1-D outcome there is no rotational part (every 1-D map is a gradient). It becomes relevant only for multi-output TRACE or for explaining a multivariate covariate drift (gradient part = pure density reweighting; rotational part = measure-preserving rearrangement invisible to marginals). **Note** (future multi-output).
- **Lagrangian vs Eulerian.** TRACE's quantile view is Lagrangian (follow the u-th unit), CFM/DFL is Eulerian (CDF at fixed y). RC2 is exactly the Lagrangian/Eulerian gap: averaging Lagrangian positions vs averaging Eulerian densities. **Adopt** as the conceptual frame of M1.

### 2.3 Computer vision
- **Linear OT (LOT) embeddings** [R8 Wang et al.; R9 Moosmüller & Cloninger] and **CDT / Radon-CDT** [R10 Park et al.; R11b Kolouri et al.]. In 1-D the LOT/CDT embedding with uniform reference *is* the quantile function, and it is an exact isometry. TRACE already is a CDT regression. **Note** (confirms the basis; no gain for 1-D).
- **Sliced / max-sliced Wasserstein** [R12 Deshpande; R13 Nadjahi]. Only for multi-output. **Note**.
- **Sinkhorn divergences and debiasing** [R4]. See 2.1. **Use** (idea).
- **Multiscale / wavelet** [R30]. See M5. **Use**.

### 2.4 Biology
- **Waddington-OT** [R14 Schiebinger 2019], **trajectory-inference theory** [R15 Lavenant et al.], **moscot** [R16]. Their unbalanced OT handles growth/death with a prior on proliferation rates; the decomposition into "transport" vs "mass change" is the same as TRACE's stay vs entry/exit. Lesson: when unit identities are **observed** (TRACE's entry/exit setting), the mass change is *known* and a balanced decomposition with exact mixture weights is correct; unbalanced penalties (KL relaxation, [R17 Chizat et al.; R18 Liero–Mielke–Savaré; R19 Séjourné et al.]) are only needed when identities are latent, and their ρ makes the split a tuning choice. **Use** (M7: keep exact weights; add the mixture player so entry/exit sd terms are covered).

### 2.5 Econometrics
- **Distribution regression and counterfactual distributions** [R20 Chernozhukov, Fernández-Val & Melly]. Fits F(y|x) at thresholds; counterfactual marginal = average of conditional CDFs (Eulerian). Gives a *model-based mixture* rival and the functional-delta-method bands. **Adopt** in M1 (the mixture computed from TRACE's own Q(u|x)).
- **DFL reweighting** [R21] and **RIF regression** [R23 Firpo, Fortin & Lemieux]. Already in the benchmark as rivals.
- **Double/debiased ML** [R25 Chernozhukov et al.], **automatic debiasing / Riesz representers** [R27, R28], **localized DML for quantiles** [R29 Kallus, Mao & Uehara]. Composition pieces are linear functionals of the covariate law weighted by the regression; the Riesz representer is the density ratio, and Riesz regression learns it directly by minimizing `E[α² − 2 m(W; α)]` without a propensity model. With TRACE's bins as the dictionary this is a closed-form least-squares problem. **Adopt** (M3, M4).
- **Balancing weights** [R34 Zubizarreta; R35 Hainmueller entropy balancing; R36 Imai & Ratkovic CBPS]. Exact balance of chosen moments. **Adopt** in M4 with bin indicators as moments.

### 2.6 Statistics
- **L-moments** [R37 Hosking]: basis is right; the RC1 fix is to estimate L-moment coefficients by their sample (U-statistic / PIT) form, not by pinball.
- **CRPS / Cramér–von Mises / quantile-weighted scores** [R11 Gneiting & Raftery; R40 Gneiting & Ranjan]: CRPS = 2∫ pinball; quantile-weighted CRPS for tail channels.
- **Isotonic distributional regression** [R41 Henzi, Ziegel & Gneiting]: calibrated, CRPS-optimal under stochastic-order constraints; a classical recalibrator for M2.
- **Kernel embeddings / MMD** [R42 Gretton et al.; R43 Muandet et al.]: a kernel two-sample test is an alternative null for (e) with known calibration under permutation. **Use** as a secondary null.
- **Empirical likelihood** [R44 Owen]: EL intervals are range-respecting and adapt to skew; useful for tail channels in (d). **Use**.
- **Stability selection** [R45 Meinshausen & Bühlmann; R46 Shah & Samworth] and **knockoffs** [R47 Barber & Candès; R48 Candès et al.]: error-controlled feature-level claims. Stability selection is cheap with bagging (M5); model-X knockoffs need a covariate model and add cost. **Use** / **Note**.
- **Conformal** [R26 Lei et al.; R49 Romano et al. CQR; R50 Chernozhukov, Wüthrich & Zhu DCP; R51 Tibshirani et al. weighted]: PIT-scale DCP is the right calibrator for a full conditional distribution. **Adopt** in M2.
- **Inference for Wasserstein functionals** [R52 Panaretos & Zemel; R53 del Barrio & Loubes; R54 Sommerfeld & Munk; R39 Fang & Santos]: CLTs for empirical W2 and the failure of the naive bootstrap at the null; motivates split-product and studentized tests. **Adopt** in M3.
- **Fast resampling** [R55 Kleiner et al. BLB; R56 Lam cheap bootstrap; R57 Chernozhukov, Chetverikov & Kato multiplier bootstrap]. **Adopt** in M3.
- **Purification of additive models** [R33 Lengerich et al.] and **InterpretML/EBM** [R58]. **Adopt** in M5.
- **Shift explanation via mechanisms** [R59 Budhathoki et al.; R60 Kulinski & Inouye]: Shapley over causal mechanisms; confirms that composition vs structure is a 2-mechanism Shapley game. **Note** (consistent with Theorem 4).
- **Fréchet / Wasserstein regression** [R61 Petersen & Müller; R62 Chen, Lin & Müller]: background.
- **Quantile curves without crossing** [R63 Chernozhukov, Fernández-Val & Galichon]: rearrangement as alternative to PAV; rearrangement is a W2 projection in 1-D and never increases the L² error.
- **Trend filtering** [R64 Tibshirani]: piecewise-polynomial shape functions with locally adaptive smoothness; alternative to bin ridge in M5.
- **Calibrated additive quantile regression (qgam)** [R65 Fasiolo et al.]: smoothed pinball for Newton steps (LANDSCAPE #2).

---

## 3. Ranked theoretical modifications

Common protocol for bars: frozen generators of PREREG_TRACE_CLAIMS (H5–H8) and ShiftBench; new seeds; paired comparisons with one-sided 95% CIs; Holm across the family; failures stay on record. Compute is quoted for n = 1e6, p = 100, K = 6, 8 threads, relative to the current 17 s fit.

### M1. Mixture-lift game: barycenter players + mixing players (rank 1)
**Targets:** (c), (f); indirectly (d) l3/l4 cells, (b) per-feature error.
**Math.** For hybrid (s,t) define the model marginal `F_{[s,t]}(y) = E_{x~F_s} F_t(y|x)` with `F_t(·|x)` the inverse of TRACE's monotone `Q_t(·|x)`, and `Q_{[s,t]} = F_{[s,t]}^{-1}`. Write the exact identity
`Q_{[s,t]}(u) = Q̄[s,t](u) + M[s,t](u)`, where `M = Q_{[s,t]} − Q̄[s,t]` is the **mixing term** (Lagrangian–Eulerian gap).
*Game.* Players: for each channel k and feature j the existing comp/struct pair, plus one **mixing player per period role** (comp-mix, struct-mix). Using the reference-symmetric switch game of Theorem 4 on the marginal quantile functions, `V_sym(S) = ⟨D_M, Q_S − Q_A⟩` is linear in `Q_S`, so the barycenter players keep exactly their Φ values *projected on D_M instead of D*, and the mixing players receive `⟨D_M, ΔM⟩` split by the 2-player Shapley rule. Efficiency holds by construction: pieces sum to `‖Q_{[B,B]} − Q_{[A,A]}‖² = W2²(F_A, F_B)` of the *model marginals* [P: Theorem 4's proof uses only additivity of the hybrid map in S; adding M as a separate summand keeps additivity because the player set is a disjoint union].
*Per-feature mixing pieces, closed form (second order).* Assume location–scale within a period, `Q_t(u|x) = μ_t(x) + σ_t q_t(u)` with `μ_t = b + Σ_j f_{t,0j}`, finite `E|δ|³`, f_t twice differentiable with bounded `f'/f` and `f''/f` on the quantile range used. Then
`M[s,t](u) = −(v_{s,t}/(2σ_t)) · s_t(u) + O(E|δ|³/σ_t²)`, `s_t(u) = f_t'(q_t(u))/f_t(q_t(u))`, `v_{s,t} = Var_{x~F_s} μ_t(x)`.
Proof sketch: `F_M(y) = E F((y−μ̄−δ)/σ)`; expand to second order in δ, use `Eδ = 0`: `F_M(y) = F(z) + (v/2σ²) f'(z) + O(·)`, `z = (y−μ̄)/σ`; solve `F_M = u` at `z_M = q + η`: `f(q)η + (v/2σ²) f'(q) = 0`, so `z_M = q(u) − (v/2σ²) f'(q)/f(q) + O(·)`, and `y = μ̄ + σ z_M`. Check: Gaussian ε gives `−f'/f = q`, so `Q_M = Q̄ + (v/2σ) q`, the first-order expansion of the exact `√(σ²+v)`. ∎
The variance splits exactly and order-free by the **covariance attribution** `v = Σ_j Cov(f_{0j}(x_j), μ(x))` (each term is the feature's share of the variance; identity, no axioms beyond linearity, and it equals the Shapley value of the variance game for additive μ restricted to its covariance form; reduces to `Var f_{0j}` under independence). So the mixing piece per feature and channel is `m_{kj} ∝ w_k · Cov_s(f_{0j}, μ_t) · ⟨s_t, P*_k⟩`. Under a Gaussian ε, `−s_t(u) = q(u)`, so mixing only rescales the noise quantile function (L-moment ratios τ3, τ4 unchanged; correct: a small Gaussian spread of locations leaves the law ≈ Gaussian with larger variance); for skewed ε it moves channels 2–3, which is the H7 skew_comp mechanism.
*Exact version.* Compute `Q_{[s,t]}` numerically (grid inversion); assign the mixing players' total exactly; split it across features by the covariance weights (exact in total, approximate in split beyond second order; the residual is reported).
**Expected gain [C].** H7 comp cells from 0/6 to ≥ 4/6 passing; error close to CFM (which targets the same marginal) while keeping per-channel, per-feature pieces CFM lacks. (f) gap goes to the additive-misspecification part only.
**Falsify (prereg-able).** Recovery on skew_comp, tail_comp, shape_mixed ≥ 0.6 (n = 2e3) and ≥ 0.8 (n = 2e4); paired error vs CFM with upper CI ≤ +0.002 (non-inferiority) in all 3 comp cells and CI < 0 in ≥ 1; struct cells not worse than v3 by > 0.002. Falsified if comp error stays > 2× CFM at n = 2e4 (would mean the TRACE conditional law itself, not the game, is the bottleneck).
**Compute.** Grid inversion on a 1e5 subsample × 128 y-points × 4 hybrids: < 2 s. Covariance weights O(np): < 1 s.
**Practicality / risk.** High / medium. Risk: per-feature mixing split is approximate when `Var μ` is large relative to σ² (heavy composition) and the uniqueness theorem then holds only for the channel totals. Interpretation needs care ("spread of predicted locations changed").

### M2. L²-projection fitting with PIT pseudo-outcomes and Sobolev ridge; PIT-isotonic recalibration (rank 2)
**Targets:** (a), (d) shrinkage, (b) partially.
**Math.** (i) *Channel 0 by squared loss*: by orthogonality `∫ Q P*_0 du = E[Y|x]`, so least-squares boosting of y gives the L² channel 0 directly [P]. (ii) *Channels k ≥ 1*: F1's identity `c_k(x) = E[(2k+1) Y P*_k(U) | x]`, `U = F(Y|x)` (exact for continuous laws) [P, FIXES2]. Generalise the PIT estimate beyond location–scale: obtain Û from the current fitted model's `F̂(Y|x)` (cross-fitted, 2 folds), iterate twice. Error enters through `E[Y P*_k'(U)(Û−U)|x]`, which is second order if Û is root-n·rate consistent in sup norm on the bulk [C for the rate; first-order expansion is P]. (iii) *Sobolev ridge across k*: penalty `λ Σ_k k^{2s} ‖f_{kj}‖²` (a JKO-type proximal step in the flat W2 geometry, [R1]), λ by cross-validated CRPS; unique minimiser per round removes path dependence. (iv) *Recalibrate* by PIT-scale distributional conformal [R50] or isotonic distributional regression on the PIT [R41]; both preserve monotonicity and are marginally calibrated [P under exchangeability].
**Expected gain [C].** MSE equal to an EBM with the same bins (same loss, same booster); shrinkage ratios → 1 for k ≤ 3; CRPS ≤ EBM+conformal on ≥ 2/3 benchmark cases because the TRACE distribution is now calibrated *and* sharper than a global residual band.
**Falsify.** MSE ratio vs EBM: paired upper CI ≤ 1.00 on all 3 cases (TRACE_SIMS cases, 20 seeds); fitted/true `c_k` in [0.95, 1.05] for k = 1..3 and [0.85, 1.15] for k = 4,5 at n = 2e4 (FIXES2 no-signal design); CRPS ≤ EBM+conformal in ≥ 2/3 and 80/90% coverage within ±2 pp. Falsified if k = 4,5 ratios stay < 0.8 (would show PIT error dominates in the tails).
**Compute.** K−1 extra least-squares backfits on shared bins (histograms reused): ≈ +25–40 s at 1e6×100; recalibration O(n log n): < 2 s.
**Practicality / risk.** High / low–medium. Risk: pseudo-outcomes `Y P*_k(U)` are noisy for k ≥ 4 (variance grows with k); the ridge controls it at some bias.

### M3. Orthogonal inference without refits (rank 3)
**Targets:** (d), (e), (h).
**Math.** (i) *Unbiased quadratic.* Split each period into halves 1,2; `D̂^{(1)}, D̂^{(2)}` independent estimates of D. Then `E⟨D̂^{(1)}, D̂^{(2)}⟩ = ‖D‖²` exactly [P: independence], removing the `tr Var` bias of `‖D̂‖²`. Players `Φ_kjr = w_k Δc̄_k a_kjr` get the symmetrised cross-fit form `½ w_k (Δc̄_k^{(1)} a_kjr^{(2)} + Δc̄_k^{(2)} a_kjr^{(1)})`; efficiency is preserved because the budget identity is bilinear [P, same algebra as Prop. 1]. This is the cross-fitting analogue of Sinkhorn debiasing [R4]. (ii) *One-step linearisation.* Every piece is a smooth function of a finite vector θ = (bin means of TRACE pseudo-outcomes per period, bin masses, weights); its influence function is computed in closed form at bin level (chain rule through the additive backfit as a linear smoother, since M2 makes each round a linear least-squares map). (iii) *Multiplier bootstrap* [R57] on the bin-level influence contributions gives simultaneous bands for all players; studentized. (iv) *Null*: studentized randomization test of period labels [R38], exact under exchangeability and asymptotically valid when variances differ; report band and test separately (do not decide "shift" by band exclusion). (v) Fallback for components that are not smooth (PAV, trimming): cheap bootstrap with 3–5 refits and t quantiles [R56]. (vi) Tail channels: EL interval for the channel coefficient [R44] as a sensitivity check.
**Expected gain [C].** Coverage up to nominal in cells where the centre is right (with M1, M2 fixing the centre); FPR → 5% because the randomization test is exact; band time from minutes to seconds.
**Falsify.** H8(b): pooled ≥ 0.88 and every cell ≥ 0.85 (frozen bar is 0.90/0.85; we pre-register 0.88 as the M3-alone bar, 0.90 with M1+M2); H8(a): CP upper ≤ 0.09 at 200 reps; band + test ≤ 2× one fit at 1e6×100. Falsified if coverage in *struct* cells (where the centre is already right) stays < 0.85.
**Compute.** IF assembly O(np) + multiplier bootstrap at bin level (≤ 256 bins × 100 features × K players, B = 2000): ≈ 5–10 s. Split halves add one extra fit (or reuse the 2 cross-fit folds of M2): ≈ +17 s. Randomization null R = 200 on the linearised statistic: < 5 s.
**Practicality / risk.** Medium–high / medium. Risk: linearisation through early-stopped boosting is approximate; mitigated because M2 makes rounds linear smoothers and the cheap-bootstrap fallback checks it.

### M4. Bin-marginal balancing weights; Riesz regression for the mixture target (rank 4)
**Targets:** (g), (d), (b) per-feature error.
**Math.** *Proposition (sufficiency of bin-marginal balance).* For the additive barycenter, `G_kj[s,t] = E_{x~F_s} f_{t,kj}(x_j)` with f piecewise constant on bins. If weights w on period-t data satisfy `Σ_i w_i 1{x_ij ∈ b} = P_s(x_j ∈ b)` for every feature j and bin b, then `Σ_i w_i f_{t,kj}(x_ij) = G_kj[s,t]` exactly for every k, j and every fitted f [P: linearity; f is a linear combination of bin indicators]. So **raking** (iterative proportional fitting) / entropy balancing [R35] on bin marginals makes the barycenter composition counterfactual exact, with no propensity model and no trimming. Entropy balancing gives the minimum-KL weights satisfying the constraints (a Gibbs/exponential-family form: statistical-mechanics view); existence iff the target marginals are in the convex hull, with diagnostics when not. *Mixture target (M1)* needs joint balance; use Riesz regression [R28] with the dictionary = bin indicators + pairwise interactions of the top features, which minimises `E_t[α²] − 2 E_s[α]` in closed form (least squares), and gives the automatic-debiasing correction for the composition functional [R27].
**Expected gain [C].** Composition pieces exact up to sampling error on the barycenter; per-feature composition error falls toward OB's level; no trim bias.
**Falsify.** Per-feature composition-share error on H7 comp cells and frozen ShiftBench ≤ 0.5× v3's; S1/S2 share coverage ≥ 18/20 untrimmed; max weight ≤ 50× median in ≥ 95% of reps (else report overlap failure). Falsified if error falls < 20%.
**Compute.** Raking: p × bins per sweep over n (≈ 0.1 s/sweep at 1e6×100, 30–50 sweeps): ≈ 5 s, below the 10.9 s logistic. Riesz least squares with ~5k columns: ≈ 10 s.
**Practicality / risk.** High / low (raking) and medium (Riesz). Risk: sparse bins under strong composition shift (overlap); handled by bin merging.

### M5. Stability layer: bin ridge + bagging + purification + Owen blocks + stability reporting (rank 5)
**Targets:** (b).
**Math.** (i) ridge on bin contrasts with a fused/Haar penalty (multiscale, [R30], or trend filtering [R64]) gives a strictly convex per-round problem, hence a unique, Lipschitz-in-data solution [P]; (ii) bagging B_bag = 8 averages path noise; variance of the averaged shape falls as `ρ + (1−ρ)/B_bag` of the single-bag variance with ρ the between-bag correlation [P]; (iii) purification [R33] fixes the centring convention canonically (resolves Lemma 7's dependence); (iv) correlated features (|Spearman| > 0.7 or mutual information above a threshold) are grouped as Owen blocks; report group pieces (exact by Theorem 4's Owen = Shapley on V_sym) and within-group pieces flagged unstable; (v) stability-selection frequencies [R45, R46] from the bags give each feature a selection probability with a PFER bound.
**Expected gain [C].** Rank stability at or above OB; per-feature error to ≤ 0.09 when combined with M1/M4 (the remaining gap is target mismatch, not variance).
**Falsify.** H6 real: bootstrap ≥ 0.77, split-half ≥ 0.46 (OB level), both with 95% CI excluding a 0.05 deficit; per-feature error ≤ 0.09 on frozen ShiftBench n = 2e4. Falsified if bootstrap stability < 0.65 with 8 bags (would mean instability is in the target, i.e. RC2/RC4, not the fit).
**Compute.** 8 bags × 17 s on 8 threads ≈ 20–40 s wall with shared histograms; ridge free.
**Practicality / risk.** High / low. Risk: grouping changes what a "feature" is; must be reported transparently.

### Ranked list beyond the top 5

| Rank | Modification | Targets | Math basis | Gain [C] | Cost 1e6×100 | Practicality / risk | Falsification bar |
|---|---|---|---|---|---|---|---|
| 6 | **Exact entry/exit with mixture player** (M7): observed identities ⇒ balanced decomposition with known weights; mixing player covers sd-type terms | (d) entry_exit cells | M1 + exact mixture weights (biology lesson, [R14–R16]) | entry_exit|sd coverage 0.50 → ≥ 0.85 | +1 s | high / low | every entry_exit cell ≥ 0.85; pieces sum ≤ 1e-9 |
| 7 | **Newton/Fisher steps on smoothed CRPS** (LANDSCAPE #2) | (a), leakage | Hessian `∫e_j e_k f(Q)` [R65] | fewer rounds, less leakage | ≈ +30% | medium / medium | leakage ≤ 0.10; CRPS ≥ 0.5% better in 2/3 |
| 8 | **Benamou–Brenier path uniqueness** (M8): Aumann–Shapley along the 1-D geodesic equals Φ | theory | Q_τ linear ⇒ V_sym linear along path ⇒ AS = Φ [P sketch: AS of a game that is linear in the path coordinates equals the coordinate-wise contribution] | stronger uniqueness story | 0 | high / none | symbolic check in `verify/` to 1e-12 |
| 9 | **Kernel (MMD/energy) conditional two-sample test** as secondary null | (e) | [R42] permutation-calibrated | power on shape-only shifts | ≈ 10 s on 1e5 subsample | high / low | FPR CP-upper ≤ 0.09; power ≥ TRACE test on tail cells |
| 10 | **Knockoff filter for per-feature structure claims** | (b) | model-X [R48] | FDR control of "feature j drives structure" | ≥ 2× fit | low / high (needs X model) | FDR ≤ 0.1 on null features over 200 reps |
| 11 | **Multiscale bins** (renormalization) | (b), speed | coarse-to-fine Haar shrinkage [R30, R22] | stability in thin bins | ≈ +10% | medium / low | as M5 |
| 12 | **Hodge split of covariate drift** | explanation only | [R7] | none for 1-D outcome | — | low / — | — |
| 13 | **Sliced / LOT / Radon-CDT multi-output** | future | [R8–R13] | multi-output TRACE | — | low now | — |
| 14 | **Entropic OT for 1-D** | — | [R3, R24] | none (exact 1-D is cheaper) | — | Drop | — |

---

## 4. Proposed v4 architecture

```
                 ┌──────────────── FIT (per period, cross-fitted, 8 bags) ────────────────┐
 data ─► bins ─► │ ch0: squared-loss boosting (EBM loss)                                    │
                 │ ch1..K-1: PIT pseudo-outcome LS boosting, Sobolev ridge across k (M2)    │
                 │ bin ridge / Haar shrink, purification, Owen groups (M5)                  │
                 │ monotone rearrangement / PAV; PIT-isotonic or DCP recalibration (M2)     │
                 └──────────────────────────────────────────────────────────────────────────┘
                                  │ Q_t(u|x), bin-level sufficient statistics
                 ┌──────────────── COUNTERFACTUAL ────────────────┐
                 │ raking / entropy balancing on bin marginals (M4)│  ← exact for barycenter
                 │ Riesz regression weights for mixture target (M4)│  ← joint balance
                 └─────────────────────────────────────────────────┘
                                  │
                 ┌──────────────── GAME ───────────────────────────────────────────┐
                 │ reference-symmetric switch game on MARGINAL quantile functions  │
                 │ players: comp/struct × (channel, feature or Owen group)         │
                 │          + mixing players (covariance attribution) (M1)         │
                 │          + entry/exit/stay with exact weights (M7)              │
                 └─────────────────────────────────────────────────────────────────┘
                                  │
                 ┌──────────────── INFERENCE ──────────────────────────────────────┐
                 │ split-product unbiased W2² pieces; bin-level influence functions │
                 │ multiplier bootstrap (simultaneous, studentized) (M3)            │
                 │ studentized randomization null; cheap-bootstrap check (M3)       │
                 │ stability-selection frequencies per feature (M5)                 │
                 └──────────────────────────────────────────────────────────────────┘
```

Budget at 1e6×100, 8 threads [C, estimated from component costs above]: fit ≈ 17 s → ≈ 60–90 s with 8 bags and K−1 LS channels; counterfactual ≈ 5–15 s; game ≈ 3 s; inference ≈ 10–30 s. Total ≈ 1.5–2.5 min **including bands**, versus ≈ 30 min today for B_refit = 100.

Invariants kept: exact budget (all pieces sum to the model-marginal W2² to ≤ 1e-9), period-swap antisymmetry, order invariance (Theorem 4 extends since the game stays linear in the switched quantile function).

What v4 does not promise: per-feature mixing pieces are exact only in total (split is second-order); interactions are still outside the additive model (reported via the (f) residual).

Suggested order of work (each step pre-registered separately): M2 → M4 → M1 → M3 → M5 → M7. M2 first because M1's mixture is computed from TRACE's conditional law and inherits its shrinkage.
