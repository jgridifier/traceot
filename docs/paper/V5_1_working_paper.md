# TRACE v5.1: splitting a shift in distribution shape into composition and response on a mixture counterfactual

**Working paper, skeleton.** Status: **SHIPPED (V5.1 only)**, opened 2026-11-09 (America/New_York).
Approved by the owner on 2026-11-05: *"Yes, ship V5.1: open the working-paper skeleton and app on
the passed V5.1 gate, sealed holdout untouched."* Decision record:
[DECISION_2026-11-04.md](../research/DECISION_2026-11-04.md) (PR #37).

**Ground rules for this document.** Docs only. Every number below is quoted from an existing
file in this repository, with a link to it; no new run was made for this paper. Indices
9600–9619 stay sealed and no holdout was run (PREREG_V5_1 §3, §6). Sections marked **TODO**
are not written yet.

Readable version: [docs/learn/v5_1_paper.html](../learn/v5_1_paper.html) ·
How to use it: [docs/learn/v5_1_app.html](../learn/v5_1_app.html).

---

## Abstract

When the distribution of an outcome changes between two periods A and B, part of the change
comes from the inputs moving (*composition*) and part from the way the outcome responds to the
inputs (*response*, also called *structure*). TRACE v5.1 splits the change in each L-moment of
the outcome into these two pieces. It builds the four mixture counterfactual laws
\(F_{[s,t]}(y)=E_{x\sim F_s}F_t(y\mid x)\) for \(s,t\in\{A,B\}\), with each period's
conditional law given by an ordinary-least-squares location fit plus that period's empirical
residual law, and takes the two-player Shapley split of the L-moment change. The fit is closed
form and has no tunable parameters
([PREREG_V5_1 §2](../research/PREREG_V5_1.md)). In a frozen, pre-registered gate on fresh
seeds, v5.1 was **BEST** against the better of two paired rivals (DFL reweighting and CFM) in
all four pure-composition cells, and passed the role check in all four
([V5_1_GATE_RESULTS.md](../benchmark/V5_1_GATE_RESULTS.md)). The simulator's mean is linear,
so this result says nothing about non-linear means; a follow-up that tried curved means was
pre-registered, run once, and parked ([PARK_V5_2.md](../research/PARK_V5_2.md)).

**TODO:** final abstract wording once §1 and §6 are written.

## 1. Problem

Decomposing a distribution shift's shape change (L-moments) into composition vs structure via a
four-hybrid mixture and a two-player Shapley split
([DECISION_2026-11-04.md, plan outline](../research/DECISION_2026-11-04.md)).

**TODO:** motivation, notation, relation to DFL (1996) reweighting and the counterfactual
distributions of Chernozhukov, Fernández-Val and Melly (2013) (cited in
`crates/core/src/trace/v5/mod.rs`), and why L-moments.

## 2. Method (TRACE_V5_1)

Quoted from [PREREG_V5_1 §2](../research/PREREG_V5_1.md) (the frozen method):

* Per period \(t\in\{A,B\}\): \(\hat\mu_t(x)=\hat\beta_{0,t}+x^\top\hat\beta_t\), the
  least-squares fit of \(y\) on an intercept and all \(p\) columns of \(x\) (no penalty, no
  feature selection, no transforms). Residuals \(r_t=y-\hat\mu_t(x)\); residual law = empirical
  law of \(r_t\).
* Unchanged from v5 @ `3d695bc`: four hybrids \([A,A],[A,B],[B,A],[B,B]\) via
  `mixture_cdf_location`, 256-point support grid, `n_quad = 64`, exact L-moments of the gridded
  law, two-player Shapley split.
* Implemented in the Rust core with a thin Python switch (no neural nets). The fit is closed
  form; there are **no tunable parameters**.

Implementation: `TRACE_V5_1` = `explain_v5(..., mean_model="ols")` (PR #29), implementation
merge SHA `665939bd4aa78ffc7474b91d482d18ac0191d595`
([V5_1_GATE_RESULTS.md](../benchmark/V5_1_GATE_RESULTS.md), note 1). Code:
`crates/core/src/trace/v5/explain.rs` (`explain_v5`, `fit_ols`, `LocationModel::Ols`),
Python wrapper `python/traceot/v5.py` (`explain_v5_1`).

**TODO:** write the Shapley split out in full; state the budget identity
(composition + response = total change, tested in `tests/python/test_v5.py`); grid and
quadrature details.

## 3. Pre-registration history (failures kept)

| step | freeze | verdict | record |
|---|---|---|---|
| v4 | `d50732f` | Overall accuracy **FAIL**; coverage **INCOMPLETE** | [V4_RESULTS.md](../benchmark/V4_RESULTS.md) |
| v5 first gate (mixture, binned mean booster) | `bce021e` | **FAIL** (n = 20 000 cells INCOMPLETE); kill rule applies, v5 first gate closed | [V5_GATE1_RESULTS.md](../benchmark/V5_GATE1_RESULTS.md), [PREREG_V5.md](../research/PREREG_V5.md) |
| v5 failure diagnosis | — | **EXPLORATORY**, not a gate, no claim; spent dev seeds 9500–9549 only | [V5_DIAGNOSIS.md](../research/V5_DIAGNOSIS.md) |
| **v5.1** (OLS location fit) | `b33f507` | **PASS** — all four cells BEST, all role checks PASS | [V5_1_GATE_RESULTS.md](../benchmark/V5_1_GATE_RESULTS.md), [PREREG_V5_1.md](../research/PREREG_V5_1.md) |
| v5.2 (curved means, additive spline) | `df5aff8` | **FAIL** — curved-mean claim parked | [V5_2_GATE_RESULTS.md](../benchmark/V5_2_GATE_RESULTS.md), [PARK_V5_2.md](../research/PARK_V5_2.md) |

Notes, quoted:

* v5 park: the v5 first gate closed under its kill rule ("its kill rule stands and the v5 first
  gate is closed", [PREREG_V5_1.md](../research/PREREG_V5_1.md), header). There is no separate
  `PARK_V5.md`; the record is V5_GATE1_RESULTS.md.
* v5.2 park: "V5.1's linear-mean claim stands exactly as stated in V5_1_GATE_RESULTS.md. This
  failure neither strengthens nor weakens it." ([PARK_V5_2.md](../research/PARK_V5_2.md))
* Holdout: "9600–9619 belong to the closed PREREG_V5 gate; they stay sealed and are not reused"
  (PREREG_V5_1 §3); "There is no separate holdout in this freeze; the gate seeds above are the
  test." (PREREG_V5_1 §6)

**TODO:** one paragraph per step on what was learned; list of v1–v3 records.

## 4. Gate design

Cells, seeds and bars are as frozen in [PREREG_V5_1 §3–§5](../research/PREREG_V5_1.md):
`skew_comp`, `tail_comp` at n = 2000 (p = 5, indices 9700–9749, 50 reps) and n = 20 000
(p = 20, indices 9800–9819, 20 reps); paired d = err_v5.1 − err_best_rival; BEST if the
one-sided 95% upper bound is below 0, NON-INFERIOR if below δ = 0.10 × planted magnitude;
role check; Holm across the four tests; 30-minute wall cap on 8 cores.

**TODO:** describe the simulator families and the truth file in prose.

## 5. Results

Quoted as is from [V5_1_GATE_RESULTS.md](../benchmark/V5_1_GATE_RESULTS.md)
(prereg `b33f507`, implementation merge `665939bd4aa78ffc7474b91d482d18ac0191d595`,
gate verdict **PASS**, wall time 15.0 s on 8 workers (cap 1800 s), 140/140 tasks finished):

| cell | n | p | reps | err v5.1 | err DFL | err CFM | mean d | upper t | upper boot | δ | verdict | \|ĉ\| | \|ŝ\| | role | p NI (Holm) | p BEST (Holm) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|---|
| skew_comp | 2000 | 5 | 50 | 0.008351 | 0.07792 | 0.0231 | -0.01475 | -0.01206 | -0.01218 | 0.004025 | **BEST** | 0.0379 | 0.001774 | **PASS** | 4.21e-16 (1.68e-15) | 1.51e-12 (6.05e-12) |
| tail_comp | 2000 | 5 | 50 | 0.005568 | 0.01348 | 0.01138 | -0.003761 | -0.002339 | -0.00237 | 0.0004586 | **BEST** | 0.004493 | 0.001809 | **PASS** | 4.23e-06 (8.47e-06) | 2.62e-05 (5.23e-05) |
| skew_comp | 20000 | 20 | 20 | 0.002875 | 0.08113 | 0.01113 | -0.00825 | -0.006373 | -0.006447 | 0.004025 | **BEST** | 0.04097 | 0.0007853 | **PASS** | 3.5e-10 (1.05e-09) | 1.77e-07 (5.32e-07) |
| tail_comp | 20000 | 20 | 20 | 0.001933 | 0.009159 | 0.005063 | -0.002988 | -0.001949 | -0.002048 | 0.0004586 | **BEST** | 0.003909 | 0.0007253 | **PASS** | 7.98e-06 (8.47e-06) | 4.26e-05 (5.23e-05) |

Truth (Shapley, graded channel): skew_comp λ3: comp 0.04011, struct -0.0001375; tail_comp λ4: comp 0.004298, struct 0.000288.

p values are unadjusted with Holm (across the completed cells) in brackets.
Full numbers: [`benchmarks/results/v5_1_gate_summary.json`](../../benchmarks/results/v5_1_gate_summary.json);
per-replication records: [`benchmarks/results/v5_1_gate_shape.jsonl`](../../benchmarks/results/v5_1_gate_shape.jsonl).

**TODO:** figure of paired d per cell (from the existing jsonl only); discussion.

## 6. Limitations

Quoted verbatim from [PREREG_V5_1 §7](../research/PREREG_V5_1.md) (Known limitation, stated in
advance):

> The simulator's mean is linear, so a linear location fit is the correct family by
> construction. A PASS shows that v5's mixture object works once the location model is
> right; it does **not** show robustness to non-linear means. That needs its own
> pre-registration.

Out of scope, quoted verbatim from [PREREG_V5_1 §6](../research/PREREG_V5_1.md):

> * Out of scope (not run under this SHA): `skew_struct`, `tail_struct`, `shape_mixed`,
>   coverage, ShiftBench, speed, per-feature shares, non-linear-mean scenarios.

The one attempt to go past the linear-mean limit (v5.2) failed and is parked
([PARK_V5_2.md](../research/PARK_V5_2.md)). The `mean_model="spline"` option stays in the
library and "carries no validated claim" (PARK_V5_2.md).

## 7. Software

Rust core `traceot_core::trace::{explain_v5, ExplainV5Config, LocationModel}` with
`LocationModel::Ols`; thin Python binding `traceot.explain_v5_1(x_a, y_a, x_b, y_b)`, the same as
`traceot.explain_v5(..., mean_model="ols")`. Note the library default is still
`mean_model="booster"` (v5), so callers must ask for v5.1 by name. Usage:
[v5_1_app.html](../learn/v5_1_app.html).

## 8. Open items

* [ ] **TODO** write §1 (problem and related work), the full method in §2, and the discussion in §5.
* [ ] Linear-mean check in the application: residual-vs-fit check, report only, no gating claim
  (decision plan item; not implemented in the library yet).
* [ ] Release tag against the gate's implementation merge SHA
  `665939bd4aa78ffc7474b91d482d18ac0191d595` (decision plan item; not tagged yet).
* [ ] Decide whether the library default should become `"ols"`; today it is `"booster"`. A
  change is a code change, not part of this docs-only ship.
* [ ] Non-linear means: need their own pre-registration on fresh seeds whose hypothesis is not
  chosen by looking at the v5.2 run (PARK_V5_2.md).
* [ ] Out-of-scope suites in PREREG_V5_1 §6 (struct cells, coverage, ShiftBench, speed,
  per-feature shares) each need their own pre-registration before any claim.
* [ ] Indices 9600–9619 stay sealed; no holdout confirmation of v5.1 is permitted by its prereg.
