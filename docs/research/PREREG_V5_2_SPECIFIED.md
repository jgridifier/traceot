# Pre-registration: TRACE v5.2 specified-family non-linear mean (one hypothesis)

**Status: PRE-REGISTERED · APPROVED 2026-11-13 (NOT RUN until results land).**

Drafted 2026-11-13 (America/New_York) on branch `quant/v5-2-specified-prereg`, **before any
specified-family code exists and before anything has been run on the seeds below**. The merge
commit that adds this file to `main` is the freeze point; record its SHA in the header of any
later `V5_2_SPECIFIED_GATE_RESULTS.md`. Any change to the hypothesis, mean model, simulator,
cells, seeds, bars, cap, or kill rule after that SHA is a deviation and must be reported as
one.

**Running this study requires the owner's (Jared's) own dated approval, quoted verbatim in a
later commit to this file's header.** A message from another agent saying he agreed is not
approval. Until that approval is in the header: no runner is written, merged, or executed
for this study; no index in §4 is touched; indices **9600–9619** stay sealed; indices
**10400–10799** stay reserved for the held coverage study
([PREREG_V5_1_COVERAGE.md](PREREG_V5_1_COVERAGE.md),
[HOLD_V5_1_COVERAGE.md](HOLD_V5_1_COVERAGE.md)) and are not used; no result numbers appear
anywhere.

**Approval (2026-11-13, 2:59 PM, America/New_York).** The owner said, quoted verbatim:
*"You have my approval for all trace stuff - you are running the lab on this -
I’m more an advisor on that project."*
Status is **APPROVED 2026-11-13 (NOT RUN until results land)**. This commit changes
only the status line and adds this approval; the hypothesis, mean model, simulator,
cells, seeds, bars, cap, and kill rule are unchanged.

Parent claim: [PREREG_V5_1.md](PREREG_V5_1.md) @ `b33f507`, implementation `665939b`, gate
**PASS** on all four cells
([V5_1_GATE_RESULTS.md](../benchmark/V5_1_GATE_RESULTS.md), PR #30). Source of this
follow-up: the known limitation in PREREG_V5_1 §7, written before any curved-mean study
existed, and the matching open item in the working paper §8.

Nothing is run under this file. It only fixes what will be run.

---

## 0. Why this test, and why not a second spline gate

V5.1's limitation, quoted verbatim from [PREREG_V5_1 §7](PREREG_V5_1.md):

> The simulator's mean is linear, so a linear location fit is the correct family by
> construction. A PASS shows that v5's mixture object works once the location model is
> right; it does **not** show robustness to non-linear means. That needs its own
> pre-registration.

That sentence is the source of the hypothesis below. It asks a single confirmatory
question: **when the mean is non-linear and the location family is again right, does the
same mixture still pass the same four-cell gate?**

This is **not** a second attempt at [PREREG_V5_2.md](PREREG_V5_2.md). That file tested a
different hypothesis — that a fixed additive restricted-cubic-spline *approximation* is
BEST or NON-INFERIOR under curved means — on seeds 10100–10149 and 10200–10219. It
**FAILED**, and the curved-mean claim is parked ([PARK_V5_2.md](PARK_V5_2.md)). That park
forbids amendments on those cells, a new knot rule, seed changes, and re-runs to chase a
bar. It also requires that any further curved-mean work have a new pre-registration on
fresh seeds, "and its hypothesis must not be chosen by looking at this run beyond what is
written here." A second spline gate would either repeat a closed test or pick a new
hypothesis from that run. Neither is done here.

The method and the simulator both differ from v5.2, and both differences are stated from
the V5.1 limitation, not from v5.2's cell-level outcomes:

* **Method.** Ordinary least squares on a correctly specified expanded basis: the V5.1
  linear terms plus the exact two non-linear terms that define the true mean (§2). No
  spline, no knot rule, no smoothing parameter.
* **Simulator.** The curved part of the mean depends only on columns that the gated
  composition scenarios do not shift (§3). That isolates a non-linear mean from the
  separate question of extrapolating a curved term along a covariate whose law changes.

**Alternatives considered and not chosen.** The held V5.1 coverage study is a different
item on the PREREG_V5_1 §6 out-of-scope list; it stays held, and this file does not
touch it. The √n sampling-variability check proposed in
[HOLD_V5_1_COVERAGE.md](HOLD_V5_1_COVERAGE.md) is, by that note, not to be registered
unless the owner asks for it in his own words; he has not. The report-only
`linear_mean_check` diagnostic (PR #47) has no threshold and no claim; validating it
would also be a useful study, but it would not answer the location-model question that
PREREG_V5_1 §7 left open.

---

## 1. Hypothesis (the only one)

**H.** When the true conditional mean is a smooth **non-linear additive** function of the
inputs and the location model is the **correctly specified parametric family** containing
that mean (§2–§3), `TRACE_V5_2_SPECIFIED` — the v5.1 mixture with only the per-period
location fit changed — is **BEST or NON-INFERIOR** against the best paired rival (DFL,
CFM) on the graded channel **and** passes the role check, in **every** gate cell of §4:
both pure-composition shapes (`skew_comp`, `tail_comp`) at both sizes (n = 2000, p = 5
and n = 20 000, p = 20).

Explicitly **not** tested: an approximating location model (spline, penalised smoother,
or any basis that does not contain the true mean); interactions between inputs; non-smooth
means; heteroskedastic or shape-varying conditionals; structure-only cells; linear-control
cells; per-feature shares; ShiftBench; speed; coverage or intervals. One hypothesis, one
method, one four-cell gate.

---

## 2. Method under test (the location-model change)

`TRACE_V5_2_SPECIFIED` = `explain_v5(..., mean_model="specified")` with exactly one change
from `TRACE_V5_1` @ `665939b`: the per-period location fit.

V5.1 fits, for each period \(t\in\{A,B\}\),

\[
\hat\mu_t(x) \;=\; \hat\beta_{0,t} + x^\top\hat\beta_t.
\]

This follow-up replaces that with ordinary least squares on the same intercept and the
same \(p\) columns, plus the two known non-linear terms that define the true mean in §3:

\[
\hat\mu_t(x) \;=\; \hat\beta_{0,t} + x^\top\hat\beta_t + \hat\gamma_{1,t}\, x_3^2 + \hat\gamma_{2,t}\,\sin(x_5).
\]

Here \(x_3\) and \(x_5\) are the raw (uncentred) third and fifth input columns
(1-indexed; code columns 2 and 4). \(\sin\) is the standard radian sine (`f64::sin` in
the Rust core). The constant \(-\kappa/\sqrt{2}\) in the true mean is absorbed by the
intercept, and \(\kappa\) is absorbed by \((\hat\gamma_{1,t},\hat\gamma_{2,t})\), so the
family contains the true mean exactly. There is **no penalty, no smoothing parameter, no
feature selection, no data-driven choice of terms**, and **no column centering**: the
design is computed on the raw draws, the same scale the simulator uses. Solved by a
pivoted QR in the Rust core; if a column is numerically rank-deficient, drop it by the
fixed rule \(|R_{kk}|\le 10^{-10}\,|R_{00}|\) and record the drop — a fixed rule, not a
tuning step. Design width: intercept + \(p\) + 2 (8 columns at p = 5, 23 at p = 20).

**Residual law and everything downstream** are unchanged from v5.1: empirical law of
\(r_t=y-\hat\mu_t(x)\), four hybrids \([A,A],[A,B],[B,A],[B,B]\) via
`mixture_cdf_location`, 256-point grid, `n_quad = 64`, exact L-moments of the gridded
law, two-player Shapley split. Rust core, thin Python switch, no neural nets. The fit is
closed form; there are **no tunable parameters**.

---

## 3. Simulator (curved mean, curvature on unshifted columns)

Based on the frozen H7 shape generator `benchmarks/shiftbench/sb_dgp_shape.py`
(`ShapeScenario`), which must not be modified; the extension is a new module that
imports it. Only the mean changes. The frozen linear part is kept:

\[
\mu_\kappa(X) \;=\; 1 + X_{1:5}^\top\beta \;+\; \kappa\, g(X),\qquad
g(X) \;=\; \tfrac{1}{\sqrt{2}}\,(X_3^2-1) \;+\; \sqrt{2}\,\sin(X_5),
\]

with \(\beta=(1,-0.8,0.6,0.4,-0.3)\) as frozen, columns 1-indexed (\(X_3\) is code column
2; \(X_5\) is code column 4). \(g\) is additive and smooth, and has mean 0 under the
period-A law \(X\sim N(0,I)\): \(E[X_3^2-1]=0\), \(E[\sin X_5]=0\). Under that law,
\(\mathrm{Var}(g)=1+(1-e^{-2})\approx 1.86\), next to \(\mathrm{Var}(X_{1:5}^\top\beta)=2.25\),
so at \(\kappa=1\) the curve is not a small perturbation.

**One curvature level, pre-stated:** \(\kappa=1\). No linear-control cell is inside this
gate (that would be a different hypothesis, about variance cost on a linear mean).

**Why these two columns.** In the frozen `SPEC`, `skew_comp` replaces only column 0 and
`tail_comp` replaces only column 1; columns 2 and 4 keep the same law in both periods.
Putting \(g\) on those unshifted columns means the only law that changes in the mean
between periods is the *linear* term on the shifted column, as in the already-calibrated
V5.1 simulator; the curved contribution has the same law in both periods. The test is
therefore a test of a non-linear mean, not a test of tail-extrapolation of a quadratic
along a shifted covariate.

**Noise and calibration.** Unchanged laws per scenario (`SPEC`). The period-B noise scale
\(\sigma_B\) is re-calibrated per scenario with the frozen procedure (`calibrate`, oracle
\(2\cdot 10^6\) draws, oracle seed 99, common random numbers) so that
\(\lambda_2(Y_B)=\lambda_2(Y_A)\). If calibration is infeasible for a gated scenario (no
\(\sigma_B>0\) solves it), that cell is **INCOMPLETE**: stop before any gate index is
touched, record the INCOMPLETE in the results file, and **do not decide H** — do not
redesign \(g\), \(\kappa\), or the cells. A new pre-registration on fresh seeds would be
required to test a different DGP.

**Truth.** Shapley composition/structure split of the graded channel, computed by the
same oracle code into a new file
`benchmarks/results/shiftbench_truth_shape_v5_2_specified.json`. Truth is computed from
the oracle seed only, never from gate seeds, and is committed after implementation and
**before** any gate run. The planted magnitude \(|{\rm truth}_c|+|{\rm truth}_s|\) for
each cell is read from that file and is the basis of \(\delta\) (§5). No truth or
result numbers appear in this file.

---

## 4. Gate cells and seeds

Generators other than the mean, `seed_of(...)` streams, estimand (graded-channel `lsplit`
vs Shapley truth), and rivals (DFL, CFM, re-run paired on the same seeds) are unchanged
from PREREG_V5 §3–4 / PREREG_V5_1 §3. The scenario key is extended by κ (e.g.
`seed_of("data", "skew_comp|k1", n, p, rep)`), so these streams are disjoint from every
linear-mean stream that shares an index-looking number in a different key.

| cell | n | p | indices | reps |
|---|---:|---:|---|---:|
| `skew_comp`, `tail_comp` | 2000 | 5 | **11000–11049** | 50 |
| `skew_comp`, `tail_comp` | 20 000 | 20 | **11100–11119** | 20 |

Four cells: 2 × 50 + 2 × 20 = **140 tasks**. All four are inside the gate; none is
optional. The same index range is reused across the two cells of one size, but the data
stream differs because the scenario key (including κ) enters `seed_of`, exactly as v5.1
did across its two scenarios.

**Why this sample size.** It mirrors the V5.1 gate that this test is the non-linear-mean
analogue of, and it inherits the same paired bars (PREREG_V4 §5 / PREREG_V5_1 §4): 50
replications at n = 2000 give a Student-t interval with 49 df and a 10 000-draw
percentile bootstrap of the mean; 20 replications at n = 20 000 give 19 df. Those sizes
were frozen for this protocol in PREREG_V4. The n = 20 000 cells are **inside** this
gate; they are not optional.

### Seed inventory (every spent or reserved range) and the fresh range

Searched on `main` @ `24edd96` (2026-11-13 ET): every pre-registration and results doc
under `docs/benchmark/`, `docs/research/`, `docs/theory/`, `docs/TRACE_SIMS.md`, every
runner under `benchmarks/`, and the `rep` field of every `benchmarks/results/*.jsonl`.
Index ranges below are replication indices fed through `seed_of(...)` unless marked as a
raw RNG seed. To stay safe across both namespaces, the fresh range is disjoint from
**all** of them, from the sealed holdout, and from the held coverage reservation.

| range | used by | record |
|---|---|---|
| 0–199 | PR #12 / TRACE_CLAIMS / ShiftBench: accuracy 0–49, coverage 0–39, invariance 0–4, null200 0–199, real-panel boot/placebo 0–99 (own streams) | `run_synthetic.py`, `run_trace_claims.py` |
| 0–1999 | toy-null calibration smoke loop (single raw seed 0, 2000 draws) | `benchmarks/shiftbench/sb_h8.py` |
| 1000–1019 | shift v2 / FIXES2 baseline | `docs/theory/PREREG_SHIFT_v2.md`, `FIXES2_PLAN.md` |
| 1000–1199 | trace_sims v1 period-2 draws (raw seeds `1000 + r`, r < 200) | `benchmarks/trace_sims/run_v1.py` |
| 2101–2119, 3101–3120, 6101–6119 (+ raw 6, 11) | trace_sims v2 / tail v2 | `docs/theory/PREREG_TAIL_v2.md` |
| 5000–6499 (incl. 5000–5019, 5200–5599, 6000–6019, 6100–6499), 7007–7012 | FIXES2 | `FIXES2_PLAN.md`, PREREG_TRACE_V3 §3 |
| 7900–7909 | v3 timing pilot | `docs/benchmark/PREREG_TRACE_V3.md` |
| 8000–8499 | v3 (shape 8000–8499, coverage 8000–8299) | `PREREG_TRACE_V3.md` |
| 9100–9149, 9200–9219, 9300–9379, 9400–9429 | v4 (H7 shape, H8b coverage, ShiftBench, speed 9400–9402) | `PREREG_V4.md`, `run_v4.py` |
| 9500–9549 | v5 first gate, then v5 diagnosis (spent) | `PREREG_V5.md`, `V5_DIAGNOSIS.md` |
| **9600–9619** | **sealed** v5 n = 20 000 indices; never run, never reused | `PREREG_V5.md`, `PREREG_V5_1.md`, `run_v5_gate1.py` (`HELD_BACK`) |
| 9700–9749, 9800–9819 | v5.1 gate | `PREREG_V5_1.md`, `v5_1_gate_shape.jsonl` |
| 10100–10149, 10200–10219 | v5.2 gate (spent; not reused) | `PREREG_V5_2.md`, `v5_2_gate_shape.jsonl` |
| 10220–10399 | explicit buffer after v5.2 (PREREG_V5_1_COVERAGE §4); left unused | `PREREG_V5_1_COVERAGE.md` |
| **10400–10799** | **reserved** for the held V5.1 coverage study; not used, not approved, not run | `PREREG_V5_1_COVERAGE.md`, `HOLD_V5_1_COVERAGE.md` |
| 10800–10999 | remainder of the 10300–10999 block surveyed as unused at the coverage freeze; left unused | `PREREG_V5_1_COVERAGE.md` |
| 81001–81020, 83001–83020, 84001–84050 | trace_sims v2 items (raw seeds) | `docs/TRACE_SIMS.md` |
| fixed constants | oracle truth seed 99; paired-bootstrap seed 2026; ADOPT `SeedSequence` (own namespace) | `sb_dgp_shape.py`, `summarize_*.py` |

Numbers that look like seeds but are not: `10001` in `run_v1.py` (a monotonicity-check
grid size), `10007` / `20011` in TRACE_DESIGN (shuffle-seed offsets inside the booster),
and arXiv/DOI fragments in the literature notes.

**Fresh range: 11000–11049 and 11100–11119.** At this freeze, on `main` @ `24edd96`, from
the repo root:

```
git grep -E '(^|[^0-9])11(00[0-9]|0[1-4][0-9]|10[0-9]|11[0-9])([^0-9]|$)'
```

returns no hits in any runner, results file, test, pre-registration, or learn page.
(After this commit the only hits are this file, its learn page, the index, the
changelog, and the paper §8 pointer.) The range is disjoint from every row of the
table, including 9600–9619 and 10400–10799. The runner,
when written after approval, must assert every index lies in 11000–11049 or 11100–11119
and none lies in any range above (a `FORBIDDEN` set, as in `run_v5_1_gate.py` and
`run_v5_2_gate.py`), and must also forbid 9600–9619 and 10400–10799 by name.

---

## 5. Bars — inherited unchanged from PREREG_V4 §5 (as scored for V5.1)

Per cell, paired \(d_r=\mathrm{err}_{\mathrm{specified},r}-\mathrm{err}_{\mathrm{best\,rival},r}\)
(best rival = min of DFL, CFM per replication). One-sided 95% upper bound of mean \(d\)
by **Student-t with n−1 df** (as the V5.1 scorer, V5_1_GATE_RESULTS note 2) **and** a
10 000-draw percentile bootstrap (seed 2026); both must agree, the larger bound is used.

* **BEST:** upper 95% bound \(<0\).
* **NON-INFERIOR:** upper 95% bound \(<\delta\), \(\delta=0.10\times\) planted magnitude
  of the graded channel for that cell (from the specified-family truth file, §3).
* **Role check**, as defined in V5_GATE1_RESULTS note 2 and applied in V5.1: mean
  \(|\hat c|>\) mean \(|\hat s|\) on the graded channel, **and** swapping the two roles
  does not lower the error.
* **Multiplicity:** Holm across the **four** paired accuracy tests. Unadjusted values
  always reported.
* **Gate PASS:** all four cells BEST or NON-INFERIOR after Holm, all four role checks
  pass, and the run finishes within the cap. Anything else is **FAIL**, except the
  calibration-INCOMPLETE stop of §3, which leaves H undecided.

**Scoring convention, fixed now to avoid the V5.2 note-3 ambiguity.** "After Holm" means:
the CI verdict (BEST or NON-INFERIOR, using the larger of the t and bootstrap bounds)
**and** the matching Holm-adjusted one-sided t p-value (p BEST for a BEST claim, p NI
for a NON-INFERIOR claim) is \(<0.05\). Both are required. Unadjusted p-values and the
CI-only verdict are also reported and cannot rescue a cell that fails this rule.

No intervals or bands are part of this gate. Coverage is a different, held study.

---

## 6. Runtime cap, stopping rules, and kill rule

* **Cap:** single pass over all 140 tasks (n = 20 000 cells scheduled first) on 8
  workers, wall cap **30 minutes** (as PREREG_V5 §8 / V5.1; the V5.1 gate finished in
  15 s, the 560-task V5.2 pass in 62 s, so 140 similar tasks are expected well inside
  the cap). No task dropped, retried, or re-run; unfinished cells are INCOMPLETE and
  never imputed.
* **Single pass.** Gate seeds are run once, after the order of work in §9. No interim
  look, no early stop on a passing prefix, no extension of the seed range if a cell is
  close to a bar.
* **Kill rule.** If **any** cell fails accuracy, fails the role check, or is INCOMPLETE
  because of the wall cap, **H is rejected**. Then: record every failure in
  `docs/benchmark/V5_2_SPECIFIED_GATE_RESULTS.md`, write
  `docs/research/PARK_V5_2_SPECIFIED.md`, and **keep V5.1's linear-mean claim exactly as
  stated** in V5_1_GATE_RESULTS (a FAIL here neither strengthens nor weakens it). The
  V5.2 park also stands unchanged. No amendment on these cells, no new basis term, no
  seed changes, no re-runs to chase a bar. The calibration-INCOMPLETE stop of §3 is the
  one exception: H is not decided, gate seeds are not touched, and the same "no
  redesign" rule applies.

---

## 7. Anti-job (what this study will not do or claim)

* **Not a general non-linear location model.** A PASS does not validate a spline, a
  GAM, a polynomial of other degree, or any approximating basis. It shows only that the
  v5.1 mixture, with a correctly specified location family, meets the inherited bars
  under this one non-linear mean. Claiming robustness to unknown curvature would be a
  different study, on different seeds.
* **Not a re-opening of V5.2.** This file does not un-park, amend, re-run, or
  re-interpret [PARK_V5_2.md](PARK_V5_2.md) or seeds 10100–10149 and 10200–10219. A PASS
  here does not revive the parked spline claim; a FAIL here does not add a new charge
  against it.
* **Not the held coverage study, and not the √n alternative.** Indices 10400–10799 are
  not used. No coverage runner is written. The √n check in
  [HOLD_V5_1_COVERAGE.md](HOLD_V5_1_COVERAGE.md) is not registered and not run.
* **Not a validation of `linear_mean_check`.** No threshold is attached to it; it gates
  nothing in this study.
* **Not a V5.1 confirmation and not an unsealing.** Indices 9600–9619 stay sealed and
  are not read, run, or tuned on. This is not a holdout confirmation of V5.1
  ([DECISION_2026-11-04.md](DECISION_2026-11-04.md)).
* **Not in scope:** `skew_struct`, `tail_struct`, `shape_mixed`, ShiftBench, speed,
  per-feature shares, bands or intervals, real data, any neural net, any change to the
  Rust-core / thin-Python split, any change to the downstream mixture, grid, quadrature,
  L-moments, or Shapley split.
* **No result numbers in this file**, and none anywhere until a results document is
  written after one approved single pass.
* **This pull request runs nothing.** Implementation, the truth file, and the gate each
  come later, in the order of §9, and only after the approval line in the header.

The most that may be claimed if the gate passes is the sentence in §1, restricted to the
four cells, this DGP, and this correctly specified location family.

---

## 8. Limitations (stated in advance)

1. **The family is right by construction.** That is the point of the test, and it is
   also its limit: a PASS does not show that a user who does not know \(g\) can recover
   the split with an approximating model.
2. **One curve shape, one curvature level, two unshifted columns.** The result does not
   generalise to other \(g\), to curvature on a shifted column, to interactions, or to
   kinks and jumps, without a new pre-registration on fresh seeds.
3. **Same simulator family as v4–v5.1** (Gaussian inputs in A, five active columns,
   homoskedastic noise), with only the mean changed. Real-data behaviour is out of
   scope.
4. **No intervals.** Coverage is not assessed.
5. **Rivals are unchanged** (DFL, CFM with their existing fixed settings). A stronger
   or weaker rival under curvature would be a different test.
6. **Structure-only cells are out of scope.** A later study, on fresh seeds, could ask
   whether a correctly specified location model attributes a structure-only change to
   composition; that question is not asked here.

---

## 9. Order of work (nothing before step 1)

1. Owner approval quoted verbatim and dated in this file's header (a docs-only commit).
2. Implement `mean_model="specified"` (`LocationModel::Specified` in the Rust core, thin
   Python switch) and the simulator module, with unit tests on **non-gate** seeds only
   (small fixed seeds below 1000, as the fixtures in `tests/`). The implementation PR
   runs nothing on 11000–11049 or 11100–11119. The `FORBIDDEN` guard and the "results
   file must not exist" single-pass check as in `run_v5_1_gate.py`.
3. Compute and commit the truth file (oracle seed 99 only). If calibration is
   infeasible, stop as in §3; do not run gate seeds.
4. Merge; record the implementation merge SHA in the results file's header, which does
   not yet exist.
5. Only then, one single pass on 11000–11049 and 11100–11119.

Indices 9600–9619 and 10400–10799 are not touched by any step.

---

## 10. Report

`docs/benchmark/V5_2_SPECIFIED_GATE_RESULTS.md` with this file's freeze SHA, the
implementation SHA, the four-cell table (err specified, err DFL, err CFM, mean d, upper
t, upper boot, δ, verdict, \(|\hat c|\), \(|\hat s|\), role, p NI / p BEST with Holm),
wall time, and every failure kept. No result numbers appear in this file.

Learn page: [v5_2_specified_prereg.html](../learn/v5_2_specified_prereg.html).
