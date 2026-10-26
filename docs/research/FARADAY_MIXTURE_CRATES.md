# Faraday survey: obscure OT / stats code for TRACE mixture redesign

**Date:** 2026-10-26 (America/New_York)  
**Pinned question:** What obscure but solid method/code could make TRACE book **mix-driven shape change** correctly and compete with CFM on composition cells, while staying fast and glass-box?  
**Constraints:** no neural nets; prefer classical OT / stats / physics-style numerics; age OK; 1–2 developers / few stars is a feature; Rust core preferred; do **not** merge foreign crates unless clearly MIT/Apache and tiny — prefer algorithm borrow notes; never mention banks/industry in repo artifacts.  
**Method:** Faraday crate survey (cast ≥15 → Faraday cards → shortlist with TRACE hooks). Every claim below was checked against clone / README / tests / license files on 2026-10-26. Ideas that started as brainstorm were labeled and verified.

Related: [BOARD_V4.md](BOARD_V4.md) (RC2, M1, M4), [LANDSCAPE.md](LANDSCAPE.md), [REFS_V4.md](REFS_V4.md). Teaching page: [faraday_mixture_crates.html](../learn/faraday_mixture_crates.html).

---

## 1. One-sentence diagnosis (RC2)

TRACE’s game is played on **barycenters of quantile curves** `E_x Q(u|x)`. A composition change that spreads conditional locations changes the **marginal** law `F_M(y) = E_x F(y|x)` (Eulerian / CFM target) without moving channels `k ≥ 1` of the barycenter (Lagrangian). v4’s mixing player is the right *game* idea; the survey asks which obscure code gives a sharp, verifiable **numerics** for that player.

---

## 2. Top 3 recommendations (sharp algorithmic borrows)

### R1 — Borrow QAG’s 1-D quantile chart as the glass-box identity for M1 (not the PyTorch package)

**Identity (verified):** [Sensifai-BV/QAG](https://github.com/Sensifai-BV/QAG) — *Quantile Affine Geometry: Exact Linearization of 1D Wasserstein Primitives* (0 stars, 1 team account, pushed 2026-09-13). Source `qag_core.py` states and implements: the 1-D W₂ space is flat in quantile coordinates; the Fréchet `p=2` barycenter is the **pointwise mean of quantile functions**; geodesics are straight lines in Q-space; PAVA / sort have exact almost-everywhere Jacobians. **No LICENSE file in the repo** (do not vendor; borrow math only).

**TRACE hook (plain math).** Keep TRACE’s Legendre channel basis on quantile curves. Define the model marginal quantile by Eulerian averaging, not by averaging Q:

\[
F_{[s,t]}(y) = \mathbb{E}_{x\sim F_s}\bigl[F_t(y\mid x)\bigr],\qquad
Q_{[s,t]}(u) = F_{[s,t]}^{-1}(u).
\]

The mixing term is exactly the Lagrangian–Eulerian gap that QAG’s docs name by omission (they optimize *inside* the quantile chart):

\[
M_{[s,t]}(u) = Q_{[s,t]}(u) - \bar Q_{[s,t]}(u),\qquad
\bar Q_{[s,t]}(u) = \mathbb{E}_{x\sim F_s} Q_t(u\mid x).
\]

On a uniform `u`-grid this is: evaluate conditional CDFs at grid `y`, average, invert (or use QAG’s `_merged_quantile` idea for discrete atoms). Then run the existing switch game on `Q_{[s,t]}` so mixing players receive `⟨D_M, ΔM⟩`. For small location spread, the second-order formula in BOARD_V4 §M1 remains the fast path; the grid inversion is the exact path and the falsifier.

**Why v4 may have missed it.** A “mixing player” that still averages *quantile bumps* (or only the covariance attribution) stays Lagrangian. QAG’s code makes the flat-chart identity impossible to miss: barycenter = mean of Q; the CFM target is *not* that object. v4 correctly diagnosed RC2 in words; this borrow pins the numerics to “invert the averaged CDF,” which is what CFM does and what TRACE must match on composition cells.

### R2 — Borrow wnet’s dual / mixture-proportion derivatives for the per-feature mixing split

**Identity (verified):** [michalsta/wnet](https://github.com/michalsta/wnet) — C++/Python Wasserstein via LEMON min-cost flow (MIT licence file; 2 stars; 1 primary developer + Copilot; active 2026-10-23). README + `wasserstein_network.py` expose `spectrum_proportion_derivatives()` and `signal_part_derivatives()`: gradients of transport cost w.r.t. **mixture weights** and peak intensities, with dual-cut certificates (`docs/dual_cuts.md`). Tests under `tests/` are extensive.

**TRACE hook (plain math).** Treat period-`t` conditional laws as a discrete mixture over TRACE bins (or over location clusters): `μ_t = ∑_j π_j ν_{t,j}`. The mixing player’s total mass is `W_2^2(F_{[A,B]}, F_{[A,A]})` attributable to `π` changing under composition. At an optimal plan, dual potentials `(φ,ψ)` give

\[
\frac{\partial}{\partial π_j} W_p^p(μ, ν) = \text{(wnet spectrum proportion derivative)}.
\]

Borrow the *idea*, not the LEMON dependency: in 1-D on a sorted grid, the dual is the cumulative cost of the monotone map; `∂/∂π_j` collapses to an integral of `φ` against component `j`. Use that to replace (or audit) BOARD_V4’s second-order covariance split `Cov(f_{0j}, μ)` when `Var μ / σ²` is not small.

**Why v4 may have missed it.** v4’s default split is a second-order Taylor identity. wnet is the rare small package that ships **exact dual derivatives w.r.t. mixture proportions** with certificates — exactly the object a glass-box mixing attribution wants when the Taylor regime fails.

### R3 — Entropy balancing on TRACE bins (Census EBW) + kernel Riesz (krrr) for M4 weights

**Identity (verified):**

- [uscensusbureau/entropy-balance-weighting](https://github.com/uscensusbureau/entropy-balance-weighting) — accelerated EBW (CC0 / US Gov public domain; 7 stars; tests for sparse + dense). Solves min KL weights subject to `X^T r = m` (moment matching), with elastic mode when constraints are infeasible.
- [rieszreg/krrr](https://github.com/rieszreg/krrr) (archived → monorepo `rieszreg/rieszreg`) — kernel ridge Riesz regression (0 stars on the archived repo; classical kernel ridge, no NN). Fits the Riesz representer `α` for linear estimands by minimizing `E[α² − 2 m(W;α)]`.

**TRACE hook (plain math).** For barycenter composition (BOARD_V4 Prop. on bin-marginal sufficiency): set `X` = TRACE bin indicators per feature; set target moments = period-`s` bin masses; EBW returns weights that make `∑_i w_i f_{t,kj}(x_{ij}) = G_{kj}[s,t]` exactly. For the **mixture** target, bin-marginals are not enough — use Riesz with dictionary = bin indicators (+ top pairwise) so `α` estimates the density ratio for `E_{x∼F_s} g(Q_t(·|x))` functionals. Both are closed-form / convex; no propensity logistic.

**Why v4 may have missed it.** RC4 already says logistic period-odds balances the wrong thing. These two small packages are the concrete solvers for the two balance problems M4 named (raking/EBW for barycenter; Riesz for mixture), ready to prototype in Python then reimplement thinly in Rust.

---

## 3. Full shortlist table

| # | Candidate | Lang | Stars† | Maintainers | License | Fit to RC2 / M1 / M4 | Keep? |
|---|-----------|------|--------|-------------|---------|----------------------|-------|
| 1 | [QAG](https://github.com/Sensifai-BV/QAG) | Py+Torch | 0 | 1 team | **none in repo** | Exact 1-D Q-chart; barycenter = mean Q | **Borrow math only** |
| 2 | [wnet](https://github.com/michalsta/wnet) | C++/Py | 2 | 1 | MIT | Mixture-proportion derivatives | **Borrow dual idea** |
| 3 | [wass](https://github.com/arclabs561/wass) | Rust | 0 | 1 | MIT OR Apache-2.0 | 1-D W, free-support bary, unbalanced Sinkhorn, Gaussian W₂ closed form; many e2e tests | **Borrow / optional tiny dep** |
| 4 | [MAM](https://github.com/dan-mim/Computing-Wasserstein-Barycenters-MAM) | Py+MPI | 3 | 1 | MIT | Exact (un)balanced W barycenter via Douglas–Rachford | Partial (high-D bary; TRACE is 1-D) |
| 5 | [Census EBW](https://github.com/uscensusbureau/entropy-balance-weighting) | Py | 7 | Census | CC0 | Bin-marginal balancing for M4 | **Adopt algorithm** |
| 6 | [krrr](https://github.com/rieszreg/krrr) → rieszreg | Py | 0 | 1 | (check monorepo) | Riesz weights for mixture target | **Adopt algorithm** |
| 7 | [fast_uot](https://github.com/thibsej/fast_uot) | Py | 9 | 1 | MIT | 1-D Frank–Wolfe UOT + 1-D UOT barycenters | Partial (entry/exit / unbalanced) |
| 8 | [free-support-barycenters](https://github.com/jvlindheim/free-support-barycenters) | Py | 8 | 1 | BSD-3 | Approximative free-support W₂ bary (POT backend) | Foil / algorithm note |
| 9 | [rust-optimal-transport](https://github.com/kachark/rust-optimal-transport) | Rust | 13 | 1 | MIT | EMD + Sinkhorn + unbalanced; POT-inspired | Solid Rust foil; no 1-D closed form |
| 10 | [sliced_OT_for_GMMs](https://github.com/MoePien/sliced_OT_for_GMMs) | Py+Torch | 4 | 2 | MIT | Mixture Wasserstein for GMMs | Partial (multi-D GMM; TRACE outcome is 1-D) |
| 11 | [numerical_ot](https://github.com/doudoualexou/numerical_ot) (adapted W bary) | Py | 0 | 1 | MIT | Adapted Wasserstein barycenters | Note for time-structured TRACE |
| 12 | [ddt-core](https://github.com/distributional-ml/ddt-core) | C++/Py | 1 | small | Apache-2.0 | Distributional trees via 1-D W splits | Note (rival architecture, not M1) |

† Stars as of 2026-10-26 via GitHub API.

---

## 4. Do **not** touch (and why)

| Item | Why |
|------|-----|
| **GeomLoss** (~714★), **POT** (~2858★) | Explicit foils — well-reviewed; use only as oracles in tests. |
| **SuperInstance/\*** (`wasserstein-agents`, `optimal-transport-rs`, …) | Near-duplicate AI-style repos, ~0–1★, agent-dynamics framing; not a careful numerics source. |
| **oxicuda-ot / Lattice inference stack** | Broad ML/CUDA or transformer-inference monorepos; OT is a side crate next to NN code — wrong dependency shape for TRACE. (`lattice-transport` alone is competent Sinkhorn/barycenter Rust, but pulling it means the Lattice ecosystem.) |
| **thibsej/unbalanced_gromov_wasserstein** | UGW for graphs / domain adaptation; PyTorch; wrong geometry for 1-D outcome TRACE. |
| **riesznet / BaryIR / BaryBind / CWB (NF)** | Neural. |
| **WassersteinDictionaryLearning** | LGPL imaging WDL; heavy, wrong license posture for merge. |
| **emd-ring** | Exact EMD but **hard-coded 9-bin circular HOG**; not general. |
| **QAG as a cargo/pip dependency** | No license file; Torch; algorithm borrow only. |
| **Merging any of the above into `jgridifier/traceot` wholesale** | Standing rule: Rust-core algorithms, thin Python bindings; prefer reimplementation of a page of math. |

---

## 5. Faraday cards (kept set)

### Card A — QAG (`Sensifai-BV/QAG`)

- **Identity:** Python reference for Quantile Affine Geometry; 0★; team `sensifai-team`; last push 2026-09-13; **no LICENSE file**.
- **Implements:** `quantile_projection` (stable sort / exact Jacobian); Fréchet `barycenter` for `p=1,2,>1` in Q-space (`p=2` = weighted mean of sorted samples — verified in `qag_core.py`); geodesics; Q-PCA; PAVA with block-average derivative; `wasserstein_p_pow`.
- **Math vs code:** Docs claim 1-D W₂ is flat in quantile coordinates; code matches (barycenter closed form is literally `(α · Q).sum(0)`). Does **not** implement the Eulerian mixture `E F(·|x)`; that gap is the TRACE lesson.
- **Quality:** Careful docstrings with lemmas; benchmark harness `run.py`; no formal `tests/` package but numerical checks in the runner.
- **Fit:** **Help** for M1 identity + exact barycenter-vs-mixture contrast.
- **Risk:** Unlicensed repo; Torch; do not vendor.

### Card B — wnet (`michalsta/wnet`)

- **Identity:** Py/C++ MCF Wasserstein; 2★; MIT; active Oct 2026; primary author Michał Startek.
- **Implements:** W_p and truncated W; L1/L2/L∞ ground metrics; **derivatives w.r.t. peak intensities and spectrum mixture proportions**; dual cuts; 1-D chain solvers (SlopeDP, ConvexSweep).
- **Math vs code:** Dual-based derivatives match standard OT sensitivity; README is explicit that network-level costs are in `W_p^p` units.
- **Quality:** Large `tests/`, Valgrind workflow, dual-cut docs, CHANGELOG.
- **Fit:** **Help** for mixing attribution gradients; **do not** add LEMON as a TRACE dep — reimplement 1-D dual form.
- **Risk:** Chemometrics-oriented API; C++ build; overkill as a dependency.

### Card C — wass (`arclabs561/wass`)

- **Identity:** Rust OT crate on crates.io; 0★ / ~2k downloads; MIT OR Apache-2.0; single maintainer; pushed 2026-10-26.
- **Implements (verified in `src/`):** `wasserstein_1d`, Sinkhorn (+ log, divergence), unbalanced Sinkhorn (+ divergence), Gromov-W, semidiscrete SGD potentials, sparse L2 plans, fixed-support entropic barycenter (log-domain IBP), **free-support barycenter**, **Gaussian W₂ closed forms** (`w2_gaussian_1d`, diagonal), WFR via unbalanced Sinkhorn.
- **Math vs code:** Module docs honestly mark entropic barycenters as blurred; free-support uses Sinkhorn plans in an outer loop (Cuturi–Doucet style), not exact LP.
- **Quality:** 19 e2e/rosetta tests under `tests/`; benches; examples (OCR unbalanced, barycenter morph).
- **Fit:** **Partial/help** — best Rust place to steal 1-D W and Gaussian closed form for TRACE sims; free-support bary is a foil to the *quantile* barycenter TRACE already has in 1-D.
- **Risk:** Young crate, f32-first API, 0★; still dual-licensed cleanly if a tiny dep is ever wanted.

### Card D — MAM (`dan-mim/Computing-Wasserstein-Barycenters-MAM`)

- **Identity:** Exact W barycenter via Method of Averaged Marginals / Douglas–Rachford; SIAM J. Math. Data Sci. 2024; 3★; MIT; 1 author.
- **Implements:** Balanced + unbalanced; free-support folder; MPI parallel marginal projections.
- **Math vs code:** Paper-backed exact (non-entropic) barycenter; code is research-style (pickle outputs, MPI required for main path).
- **Quality:** Published method; thin automated tests.
- **Fit:** **Partial** — TRACE’s 1-D barycenter is already closed-form in Q-space; MAM matters if TRACE ever leaves 1-D outcomes.
- **Risk:** MPI/ops weight; not the RC2 fix.

### Card E — Census entropy-balance-weighting

- **Identity:** US Census Bureau; 7★; CC0; sparse-capable EBW.
- **Implements:** `entropy_balance(x_sample, weights0, mean_population_moments)` with bounds + elastic infeasibility mode.
- **Math vs code:** Matches the standard Hainmueller EBW KL program; README states the Lagrangian clearly.
- **Quality:** pytest suite for sparse/dense; working paper linked.
- **Fit:** **Help** for M4 bin-marginal balancing (set `X` = TRACE bin dummies).
- **Risk:** Pardiso/MKL optional stack for scale; for TRACE bin counts a dense Newton is enough — borrow the problem statement.

### Card F — krrr / RieszReg

- **Identity:** Kernel ridge Riesz representer learner; archived solo repo → `rieszreg/rieszreg` monorepo; classical kernels (Gaussian, etc.), no NN.
- **Implements:** `KernelRieszRegressor` for ATE/ATT/shift estimands; direct / Nyström / RFF solvers; tests for estimands.
- **Fit:** **Help** for M4 mixture-target weights (Riesz `α` ≈ density ratio in a TRACE bin dictionary).
- **Risk:** Confirm license in the live monorepo before any copy; prefer a 20-line least-squares Riesz on bin indicators inside TRACE Rust.

### Card G — fast_uot (`thibsej/fast_uot`)

- **Identity:** AISTATS 2022; 9★; MIT; translation-invariant Sinkhorn + **1-D Frank–Wolfe UOT** + `uot1dbar` unbalanced barycenters.
- **Fit:** **Partial** — useful for unbalanced / entry–exit mass (BOARD M7), not the balanced composition mixing of RC2.
- **Risk:** Some example TODOs; PyTorch optional paths.

### Card H — free-support-barycenters (`jvlindheim`)

- **Identity:** von Lindheim 2022 approximative free-support W₂ bary; 8★; BSD-3; POT `emd` backend.
- **Fit:** Algorithm note / foil to entropic free-support in `wass`.
- **Risk:** Depends on POT; not 1-D-specialized.

### Card I — rust-optimal-transport (`kachark`)

- **Identity:** 13★; MIT; last meaningful push 2024-03; EMD (C++ network simplex via cxx), Sinkhorn, Greenkhorn, unbalanced Sinkhorn.
- **Fit:** Mature Rust OT foil; **none** for RC2 directly (no quantile/mixture abstraction).
- **Risk:** Bitrot risk; C++ build for EMD.

### Card J — sliced_OT_for_GMMs (`MoePien`)

- **Identity:** TMLR/arXiv 2025 “Slicing the Gaussian Mixture Wasserstein Distance”; 4★; MIT; 2 authors.
- **Implements:** MSW / SMSW between GMMs; GMM barycenter notebooks.
- **Fit:** **Partial** — “mixture Wasserstein” is the right *phrase* for RC2, but the code is multi-D GMM sliced OT. For TRACE’s 1-D outcome the exact object is the mixture of 1-D conditionals (no slicing needed).
- **Risk:** Torch; easy to over-import.

### Card K — numerical_ot (adapted W barycenters)

- **Identity:** ETH bachelor thesis code; 0★; MIT; fixed-point + GD for adapted Wasserstein barycenters.
- **Fit:** **Note** if TRACE adds time-adapted / nested distance structure; not needed for static composition cells.
- **Risk:** Thesis-grade; thin reviews.

### Card L — ddt-core

- **Identity:** Distributional Decision Trees; Apache-2.0; 1★; 1-D Wasserstein splits; predicts full ECDF per leaf.
- **Fit:** **Note** as an alternative architecture (tree of distributions) rather than an M1 patch to TRACE’s additive Q-booster.
- **Risk:** Different product shape.

---

## 6. Cast net (dropped after skim; ≥15 total)

Also inspected and dropped or foiled: `elli0t-yash/sliced-wasserstein` (SW₁ only), `JulianKnodt/wasserstein` (tiny 1-D, no license/tests), `URI-ABD/wasserstein` (LEMON MCF, 2021), `nanit/emd-ring` (9-bin ring), `wplohrmann/optimal-transport-rs` (2020, thin), `matthieuheitz/WassersteinDictionaryLearning` (LGPL imaging), `justinhou95/EntropicAdaptedWassersteinGaussian` (GP adapted W), `EddieYang211/ebal-py` (smaller EBW than Census), `thibsej/unbalanced_gromov_wasserstein` (UGW), SuperInstance OT agent repos, GeomLoss, POT, oxicuda-ot.

---

## 7. Concrete TRACE modification sketch (single sharp path)

**Do this first (one PR worth of math, no new deps):**

1. **Exact mixing term (QAG identity).** From fitted TRACE conditionals on a `y`-grid, form `F_{[s,t]}(y) = ∑_{bins} \hat P_s(\mathrm{bin}) F_t(y\mid\mathrm{bin})` (or row-level average of CDF values). Invert to `Q_{[s,t]}`. Set `M = Q_{[s,t]} - \bar Q_{[s,t]}`. Assert on a synthetic two-component location mixture that `‖M‖_{L^2}` matches the known mixture-minus-barycenter formula and that CFM’s composition cells see `M` in channels `k≥1`.
2. **Game.** Extend the switch game to marginal quantiles; mixing players get `⟨D_M, ΔM⟩` (BOARD_V4 M1). Keep barycenter players on `D` projected onto `D_M` as written there.
3. **Split audit (wnet idea).** Compare covariance split vs 1-D dual `∂W²/∂π_j` on the same synthetic; report residual when `Var μ / σ²` is large.
4. **Weights (EBW/Riesz).** Replace logistic composition weights with EBW on bin indicators for barycenter pieces; Riesz LS for mixture pieces.

Falsify with the existing H7 bars in BOARD_V4 (comp recovery ≥ 0.6 / 0.8; non-inferior to CFM).

---

## 8. Verification log

| Claim | How verified |
|-------|----------------|
| QAG `p=2` barycenter = mean of sorted Q | Read `qag_core.py` lines 334–338 |
| wnet mixture proportion derivatives | README + `spectrum_proportion_derivatives` in `wasserstein_network.py` |
| wass dual license + 1-D / free-support / Gaussian | `LICENSE*`, `src/barycenter.rs`, `src/gaussian.rs`, `tests/` listing |
| MAM = DR / averaged marginals, MIT, SIAM 2024 | README + `LICENCE` + `mam/solver.py` docstring |
| Census EBW problem statement | README math + `entropy_balance_weighting/` + `LICENSE.md` CC0 |
| fast_uot 1-D FW UOT + uot1dbar | `fastuot/uot1d.py`, `uot1dbar.py`, MIT LICENSE |
| No license in QAG | `ls` of clone — only README, `qag_core.py`, `run.py`, requirements |
| Stars / push dates | `gh api repos/...` on 2026-10-26 |

No LLM brainstorm numbers were used as facts; candidate names from the task brief were independently re-checked.

---

## 9. What this survey is not

- Not a benchmark bake-off.
- Not permission to merge foreign code into the Rust core.
- Not a replacement for BOARD_V4’s pre-registered bars — it supplies **which obscure numerics** to borrow so M1/M4 can hit those bars.
