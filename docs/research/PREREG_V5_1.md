# Pre-registration: TRACE v5.1 — linear location model inside the v5 mixture (one hypothesis)

**Frozen:** 2026-10-30 (America/New_York), on branch `quant/v5-1-prereg`, **before any
v5.1 result exists on the seeds below**. The commit that adds this file is the freeze
point; record its SHA in the header of any later `V5_1_GATE_RESULTS.md`. Any change to
the hypothesis, method, cells, seeds, or bars after that SHA is a deviation.

Prior freeze: [PREREG_V5.md](PREREG_V5.md) @ `bce021e`, first gate **FAIL**
([V5_GATE1_RESULTS.md](../benchmark/V5_GATE1_RESULTS.md)); its kill rule stands and the
v5 first gate is closed. Evidence for this narrower test:
[V5_DIAGNOSIS.md](V5_DIAGNOSIS.md) (**exploratory**, spent dev seeds 9500–9549 only).

---

## 1. Hypothesis (the only one)

**H.** The v5 first-gate failure is caused by the **mean (location) model**, not by the
Eulerian mixture and not by a homoskedastic residual assumption. Replacing the binned
mean booster with an ordinary-least-squares linear location fit, keeping everything
else in v5 unchanged, makes both pure-composition cells (`skew_comp`, `tail_comp`) at
both sizes (n = 2000, n = 20 000) **BEST or NON-INFERIOR** against the best paired
rival **and** passes the role check.

Explicitly **not** tested: heteroskedastic or location-scale-shape conditionals. The
diagnosis found no heteroskedasticity to fix (the simulator is homoskedastic and the
true homoskedastic conditional is BEST), so that hypothesis is not pre-registered.

## 2. Method under test (fixed now)

`TRACE_V5_1` = `explain_v5` with exactly one change:

* Per period \(t\in\{A,B\}\): \(\hat\mu_t(x)=\hat\beta_{0,t}+x^\top\hat\beta_t\), the
  least-squares fit of \(y\) on an intercept and all \(p\) columns of \(x\) (no
  penalty, no feature selection, no transforms). Residuals \(r_t=y-\hat\mu_t(x)\);
  residual law = empirical law of \(r_t\) (as v5).
* Unchanged from v5 @ `3d695bc`: four hybrids \([A,A],[A,B],[B,A],[B,B]\) via
  `mixture_cdf_location`, 256-point support grid, `n_quad = 64`, exact L-moments of the
  gridded law, two-player Shapley split.
* Implemented in the Rust core with a thin Python switch (no neural nets). The fit is
  closed form; there are **no tunable parameters**.

## 3. Cells and seeds

Generators, truth file, `seed_of(...)` streams, estimand (graded-channel `lsplit` vs
Shapley truth), and rivals (DFL, CFM, re-run paired on the same seeds) are unchanged from
PREREG_V5 §3–4.

| cell | n | p | indices | reps |
|---|---:|---:|---|---:|
| `skew_comp`, `tail_comp` | 2000 | 5 | **9700–9749** | 50 |
| `skew_comp`, `tail_comp` | 20 000 | 20 | **9800–9819** | 20 |

**Why these seeds.** 9500–9549 are spent: they decided the failed v5 gate and the
hypothesis above was chosen by looking at them. 9600–9619 belong to the closed
PREREG_V5 gate; they stay sealed and are not reused, so no index is shared between the
failed freeze and this one. 9100–9429 are v4's. 9700–9749 and 9800–9819 appear in no
runner, results file, or pre-registration in the repo at this freeze. The n = 20 000
cells are **inside** this gate; they are not optional.

## 4. Bars — inherited unchanged from PREREG_V4 §5

Per cell, paired \(d_r=\mathrm{err}_{\mathrm{v5.1},r}-\mathrm{err}_{\mathrm{best\,rival},r}\)
(best rival = min of DFL, CFM per replication); one-sided 95% upper bound by t-interval
**and** 10 000-draw percentile bootstrap (seed 2026); both must agree, the larger bound is
used.

* **BEST:** upper 95% bound of \(d\) \(<0\).
* **NON-INFERIOR:** upper 95% bound \(<\delta\), \(\delta=0.10\times\) planted magnitude
  of the graded channel (\(|{\rm truth}_c|+|{\rm truth}_s|\) from the frozen truth file:
  0.004025 on `skew_comp`, 0.0004586 on `tail_comp`).
* **Role check (per cell):** mean \(|\hat c|>\) mean \(|\hat s|\) on the graded channel,
  and swapping the two roles does not lower the error (operationalisation of
  V5_GATE1_RESULTS note 2, unchanged).
* **Gate PASS:** all four cells BEST or NON-INFERIOR **and** all four role checks pass.
  Holm across the four paired tests; unadjusted values also reported.

No new bars.

## 5. Kill rule

If **any** cell fails accuracy, fails the role check, or is INCOMPLETE (wall cap
30 minutes on 8 cores, PREREG_V5 §8; incomplete cells are never imputed), H is
rejected: record the failure, write `PARK_V5.md`, and stop the v5 line. No second
amendment on these cells, no seed changes, no re-runs to chase a bar.

## 6. No tuning, no holdout games

* No change to the method in §2 after this freeze. No tuning on any seed.
* No peeking: nothing is run on 9700–9749 or 9800–9819 before the implementation is
  merged and this SHA is recorded.
* There is no separate holdout in this freeze; the gate seeds above are the test.
* Out of scope (not run under this SHA): `skew_struct`, `tail_struct`, `shape_mixed`,
  coverage, ShiftBench, speed, per-feature shares, non-linear-mean scenarios.

## 7. Known limitation (stated in advance)

The simulator's mean is linear, so a linear location fit is the correct family by
construction. A PASS shows that v5's mixture object works once the location model is
right; it does **not** show robustness to non-linear means. That needs its own
pre-registration.

## 8. Report

`docs/benchmark/V5_1_GATE_RESULTS.md` with this file's SHA, the four-cell table (err,
paired d, both upper bounds, δ, verdict, \(|\hat c|\), \(|\hat s|\), role check), Holm
values, wall time, and every failure kept. No result numbers appear in this file.

Learn page: [v5_1_prereg.html](../learn/v5_1_prereg.html).
