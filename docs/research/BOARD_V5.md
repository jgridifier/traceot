# Research board v5 — after v4 FAIL

**Date:** 2026-10-26 (America/New_York).  
**Freeze input:** [`V4_RESULTS.md`](../benchmark/V4_RESULTS.md) @ prereg `d50732f`, branch `quant/v4-spec`.  
**Rule:** no new full scored suite here. Failures stay failures. No banks / industry framing.

---

## 1. Honest status

TRACE v4 **failed** Jared’s SOTA / speed bar.

| Bar (PREREG_V4) | Verdict | Source numbers |
|---|---|---|
| Overall accuracy | **FAIL** | Every H7 cell FAIL; ShiftBench mean FAIL on `mixed`/`hetero` |
| H7 shape (10 cells) | **FAIL** all 10 | e.g. `skew_comp` n=20k: mean err v4 **0.05193**, best rival CFM, upper 95% CI of \(d\) **0.04074** ≫ δ **0.004025** |
| ShiftBench mean-channel | **BEST** on `tail`; **FAIL** on `mixed`, `hetero` | `tail` mean \(d\) upper CI **−0.009**; `mixed` mean err v4 **0.467** vs rival ~0.018 |
| Coverage | **INCOMPLETE** | Budget CI only; no per-cell lsplit / mean bands |
| Speed ≥5× CFM | **FAIL** | CFM/v4 = **1.02×** at \(n=10^5\), **2.94×** at \(n=10^6\) |

v4 shipped the planned pillars (M1–M4 skeleton). Shipping ≠ clearing the bar.

---

## 2. What moved the needle vs what did not

Numbers below are from `benchmarks/results/v4_*.jsonl` (same seeds as the freeze), not invented.

### Did **not** move accuracy enough (primary FAIL)

1. **M1 mixing player did not recover composition-shaped L-moments.**  
   On `skew_comp` at \(n=20000\): planted Shapley graded channel has \(|\mathrm{truth}_c|\approx 0.040\), \(|\mathrm{truth}_s|\approx 0.0001\), but TRACE_V4 estimates average \(|\widehat{c}|\approx 0.0027\), \(|\widehat{s}|\approx 0.013\) — mass lands in the **wrong role**. Mean graded error **0.0519** vs CFM **0.0124** / DFL **0.077**. Same pattern on `tail_comp` (v4 **0.060** vs CFM **0.0047**). Bolting a mixing term onto the old curve-average game did not make TRACE match the mixture counterfactual CFM already computes.

2. **Structure cells leak into composition.**  
   On `skew_struct` \(n=20000\): truth is almost pure structure (\(|\mathrm{truth}_c|\approx 0.0004\), \(|\mathrm{truth}_s|\approx 0.050\)), yet v4 shows \(|\widehat{c}|\approx 0.039\) ≈ \(|\widehat{s}|\). Error **0.0505** vs DFL **0.0037**. The game is not pinning roles.

3. **Mean-channel SplitBench estimand is broken on large location shifts.**  
   TRACE_V4 `split` exposes only `mean` (rivals expose mean/sd/q’s). On `mixed`, one draw: v4 mean split \(\approx(0.238, 0.238)\) vs truth \((\mathrm{comp},\mathrm{struct})\approx(0.660,-0.181)\) and CFM \(\approx(0.646,-0.174)\). Symmetric 50/50 split → scaled mean-channel error **0.467** vs CFM/DFL **~0.015**. On `tail` (near-null mean) v4 is **BEST** — so the mean game works when there is little to split, and fails when composition dominates the mean.

### Partial / infrastructure

4. **Speed improved vs historical booster-refit bands, but not vs CFM’s threshold grid.**  
   At \(n=10^6\), wall **46.5 s** (v4) vs **137 s** (CFM) → **2.94×**, short of 5×. At shape \(n=2000\), DFL is **faster** than v4 (~0.03 s vs ~0.10 s). Conditional speed gate cannot rescue accuracy FAIL.

5. **M3 bands incomplete for the coverage bar.**  
   Cover track wrote 800 records with `budget_ci` only. Oracle coverage vs Shapley / ShiftBench truth was never scorable → **INCOMPLETE**, not a pass.

### Relative bright spot (do not over-claim)

6. **`tail` mean-channel BEST** (upper CI of \(d\) negative). Structure-ish / small-mean regimes are where the current game is least wrong — consistent with v3’s relative competence on structure-only cells, not a SOTA claim.

---

## 3. Root-cause read (five bullets)

1. **Wrong primary object:** v4 still centers the feature game on a barycenter / curve-average world and patches mixture with M1; CFM’s object *is* the mixture counterfactual, and that is what the H7 composition cells demand.
2. **Role leakage:** graded estimates put comparable mass on the silent role (comp↔struct), so L¹ error vs Shapley stays ~0.05 while rivals sit at ~0.004–0.015.
3. **Mean path attribution fails under large composition means:** symmetric splits on `mixed`/`hetero` are a smoking gun for a game identity / path bug or a wrong mean estimand, not a “need more reps” issue.
4. **Coverage bar never instrumented:** multiplier bands were not persisted per cell → cannot claim M3 worked empirically.
5. **Speed story is “faster than refitting,” not “5× CFM”:** 1–3× at the prereg sizes; DFL remains cheaper on small \(n\).

---

## 4. V5 directions (1–2 novel moves — no NN, no retest of the same FAIL)

Do **not** re-prereg another mixing-player / PIT / raking incremental pass against the same bars. Prefer a change of estimand or geometry.

### Direction A — **Mixture-first game (CFM object + TRACE feature algebra)**

Make the **mixture counterfactual distribution** the primitive (as in Chernozhukov–Fernández-Val–Melly distribution regression / counterfactual distributions), then run TRACE’s order-free feature budget on **1D OT maps between those mixture laws** (quantile / CDT maps), not on barycenters of conditional quantile curves.

- Borrow: [Chernozhukov, Fernández-Val, Melly (Econometrica 2013)](https://doi.org/10.3982/ECTA10582) — verified DOI; also [Econometric Society page](https://www.econometricsociety.org/publications/econometrica/2013/11/01/inference-counterfactual-distributions).
- Borrow (CV / harmonic analysis): 1D [Cumulative Distribution Transform](https://arxiv.org/abs/1511.03206) / Radon-CDT (Kolouri–Park–Rohde) — OT maps that are closed-form in 1D; use as the linear space where feature shares are computed ([arXiv:1511.03206](https://arxiv.org/abs/1511.03206), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4871726/)).
- Why this is not “run CFM”: CFM stops at the distribution effect; TRACE’s differentiator is **order-free per-feature shares** of that effect. V4 failed because it attributed the wrong object.
- Success sketch (for a future prereg, not now): H7 graded error NI/BEST vs CFM on composition cells **and** mean-channel NI on `mixed`/`hetero`, with wall time ≤ CFM (CDT/quantile maps beat threshold grids).

### Direction B — **Hellinger–Kantorovich composition (transport + reaction)**

Treat covariate-mix change as **mass creation/destruction** in feature space, not only reweighting + a mixing patch. Use the Hellinger–Kantorovich / entropy-transport geometry (reaction + Wasserstein) so composition→shape is a geodesic term, then attribute along HK geodesics with classical dual potentials (no nets).

- Borrow (physics / fluids / metric analysis): Liero–Mielke–Savaré, *Optimal Entropy-Transport problems and a new Hellinger–Kantorovich distance*, Inventiones 2018 — [Springer](https://link.springer.com/article/10.1007/s00222-017-0759-8), OA PDF [FU Berlin](http://publications.imp.fu-berlin.de/2179/1/Liero2018_Article_OptimalEntropy-TransportProble.pdf); dynamic form SIAM J. Math. Anal. 2016 [doi:10.1137/15M1041420](https://doi.org/10.1137/15M1041420).
- Why novel vs v4 M1: M1 was a second-order expansion glued onto W₂ barycenters; HK **is** the metric that jointly prices transport and growth/death — the natural language for entry/exit and mix-driven shape.
- Keep: transparent classical OT; Rust dual / Newton solvers already in-repo for entropy balance.

**Preference for next build:** **A first** (smaller jump, directly attacks the CFM gap and the mixed/hetero mean FAIL); keep B as the composition/entry-exit geometry once A’s object is correct.

---

## 5. Explicit non-goals

- No neural nets.
- No full scored suite in this note.
- No silent edit of `PREREG_V4.md`.
- No re-running H7 hoping M1+PIT+raking will suddenly NI.

---

## 6. Pointers

- Results: [`docs/benchmark/V4_RESULTS.md`](../benchmark/V4_RESULTS.md)
- Applied: [`docs/learn/research_board.html`](../learn/research_board.html), [`docs/learn/nightly.html`](../learn/nightly.html)
- Theory freeze: [`docs/theory/V4.md`](../theory/V4.md), [`docs/benchmark/PREREG_V4.md`](../benchmark/PREREG_V4.md)
