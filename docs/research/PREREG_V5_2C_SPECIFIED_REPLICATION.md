# Pre-registration: TRACE v5.2c specified-family replication (same hypothesis, fresh seeds, spawn-safe harness, rehearsal, watchdog)

**Status: PRE-REGISTERED · APPROVED 2026-11-13 · NOT RUN.**

Drafted 2026-11-13 (America/New_York) on branch `quant/v5-2c-specified-prereg`, **after**
the V5.2b specified-family pass aborted with zero results and spent seeds
12000–12049 and 12100–12119
([V5_2B_ABORT.md](../benchmark/V5_2B_ABORT.md)), and **before anything has been
run on the seeds below**. The merge commit that adds this file to `main` is the
freeze point; record its SHA in the header of any later
`V5_2C_SPECIFIED_GATE_RESULTS.md`. Any change to the hypothesis, mean model,
simulator, cells, seeds, bars, cap, kill rule, pre-flight, spawn-safe harness,
rehearsal, or watchdog after that SHA is a deviation and must be reported as one.

**This file is a replication of
[PREREG_V5_2_SPECIFIED.md](PREREG_V5_2_SPECIFIED.md), frozen at
`cc8c1d7afc9c58a8b97b8b3b28b6ad6fd1704920` (PR #54), and of the mandatory
pre-flight rule of
[PREREG_V5_2B_SPECIFIED_REPLICATION.md](PREREG_V5_2B_SPECIFIED_REPLICATION.md)
§2, frozen at `19ed3202c701a5ee924399885265b1996d4a13cc` (PR #61).** The
hypothesis, method, simulator, rivals, Holm scoring, BEST / NON-INFERIOR rule,
\(\delta\), 30-minute cap, role check, anti-job, and limitations are those of
the specified-family parent, incorporated here **by reference**, and are not
restated so they cannot drift. They are, by this sentence, byte-identical to
the corresponding sections of that freeze. The scientific body of that file
after the line beginning "Parent claim:" was kept byte-identical through later
header-only and pointer-only updates (approval
[PR #56](https://github.com/jgridifier/traceot/pull/56), results
[PR #60](https://github.com/jgridifier/traceot/pull/60)). The pre-flight *rule*
(what must be imported, what smoke fits must succeed, what versions are
recorded, that any failure aborts before any gate seed is touched and leaves H
undecided) is that of V5.2B §2, incorporated **by reference**, and is not
restated so it cannot drift. The *way* pre-flight and the gate pool are
executed is the spawn-safe harness of §2 below; that is a change of process
isolation, not a change of what is checked.

The descriptive `tail_comp` composition under-estimate, a secondary
report-only item, is that of V5.2B §3, also incorporated by reference. It is
not a scored bar and does not enter the pass/fail rule.

This file changes **only** the following, each stated in full below:

(a) Fresh seeds, disjoint from every used or reserved range, including the
spent V5.2 specified seeds, the spent V5.2b seeds, the spent coverage seeds,
and the sealed holdout.

(b) A spawn-safe harness: the pre-flight runs in a spawned child process, and
the gate pool is created with multiprocessing's `'spawn'` start method from a
parent that never calls TRACE.

(c) A mandatory REHEARSAL of the full parallel pipeline, on throwaway non-gate
seeds, before any gate seed is touched. Rehearsal outputs never enter scoring.
Any rehearsal failure aborts before gate seeds are touched.

(d) A watchdog: if no task completes within 5 minutes, the pass is aborted and
recorded as ABORTED (not FAIL). H is left undecided.

**Approval (2026-11-13, 2:59 PM, America/New_York), already recorded.** The owner
said, quoted verbatim:
*"You have my approval for all trace stuff - you are running the lab on this -
I’m more an advisor on that project."*
That standing approval is already in the header of
[PREREG_V5_2_SPECIFIED.md](PREREG_V5_2_SPECIFIED.md)
([PR #56](https://github.com/jgridifier/traceot/pull/56), merge
`05375628e44b1b203c0fb6c92a1c20d37e067742`) and of
[PREREG_V5_2B_SPECIFIED_REPLICATION.md](PREREG_V5_2B_SPECIFIED_REPLICATION.md).
This file is opened under that same approval, dated 2026-11-13, quoted here.
Status is **APPROVED 2026-11-13 (NOT RUN until results land)**. A message from
another agent saying he agreed is not an additional approval and is not
needed; the quote above is the approval.

**Why a new pre-registration, not a retry.**
[PREREG_V5_2B_SPECIFIED_REPLICATION.md](PREREG_V5_2B_SPECIFIED_REPLICATION.md)
§2, frozen at `19ed320`, says that once a successful pre-flight is followed by
the start of the gate pass, the single-pass rule of the parent applies and
those seeds are spent, including if a later task then fails. The V5.2b pass
([V5_2B_ABORT.md](../benchmark/V5_2B_ABORT.md)) had a successful pre-flight
and then started the gate: `multiprocessing.Pool(8)` was created and the 140
tasks were submitted. The pre-flight TRACE smoke had already run in the parent
process and initialized Rayon; the pool used the default `'fork'` start
method; every worker deadlocked; the process was SIGKILLed at 16:05 ET,
America/New_York, with an empty results file (0 bytes, 0 tasks). Verdict:
**ABORTED (no verdict)**. H is not decided. It is not a FAIL and not a park.
Seeds 12000–12049 and 12100–12119 are spent and will not be reused. A complete
accuracy test against DFL and CFM needs a new pre-registration on fresh seeds,
with a harness that does not fork after Rayon has been initialized in the
parent, and with a rehearsal that would have caught that deadlock before any
gate seed was submitted. That is this file. It is not a re-run, not an
amendment of the frozen V5.2 specified or V5.2b protocols, and not a second
hypothesis.

Parent claim: the same as [PREREG_V5_2_SPECIFIED.md](PREREG_V5_2_SPECIFIED.md)
([PREREG_V5_1.md](PREREG_V5_1.md) @ `b33f507`, implementation `665939b`, the known
limitation in PREREG_V5_1 §7). The method and simulator already exist at
implementation merge `1aab4778abf8f84c8c8f7d9603eb8c322f1ba3c9`
([PR #58](https://github.com/jgridifier/traceot/pull/58)) and are **reused**; this
file does not redesign them. The truth file
`benchmarks/results/shiftbench_truth_shape_v5_2_specified.json` (oracle seed 99
only, committed with that implementation) is **reused**, because the simulator is
unchanged. It is not recomputed.

Nothing is run under this file. It only freezes what will be run.

---

## 0. What is identical (by reference)

The following sections of [PREREG_V5_2_SPECIFIED.md](PREREG_V5_2_SPECIFIED.md) at
`cc8c1d7afc9c58a8b97b8b3b28b6ad6fd1704920` are incorporated here in full, and are
not rewritten. The summaries that follow are a reading aid. If any summary
wording were to differ from the parent file, the parent file at that freeze
SHA governs.

* **§1 Hypothesis.** The only hypothesis (H): when the true conditional mean is a
  smooth non-linear additive function of the inputs and the location model is the
  correctly specified parametric family containing that mean,
  `TRACE_V5_2_SPECIFIED` is BEST or NON-INFERIOR against the best paired rival
  (DFL, CFM) on the graded channel **and** passes the role check, in **every**
  gate cell: both pure-composition shapes (`skew_comp`, `tail_comp`) at both
  sizes (n = 2000, p = 5 and n = 20 000, p = 20).
* **§2 Method under test.** `explain_v5(..., mean_model="specified")`: per-period
  OLS on the intercept, the raw \(p\) columns, \(x_3^2\), and \(\sin(x_5)\), on
  uncentred draws; residual law and everything downstream unchanged from V5.1.
  Rivals receive the same centred arrays and the same fixed settings as in the
  V5.1 / V5.2 gates.
* **§3 Simulator.** Curved mean \(\mu_\kappa(X)=1+X_{1:5}^\top\beta+\kappa\,g(X)\)
  with \(g\) on unshifted columns, \(\kappa=1\). Calibration of \(\sigma_B\) by
  the frozen procedure (oracle seed 99). If calibration is infeasible, that cell
  is INCOMPLETE: stop before any gate index is touched, do not decide H, and do
  not redesign. (Calibration was already feasible at the implementation merge;
  the committed truth file is reused. The runner still applies this stop if a
  calibrated \(\sigma_B\) is missing.)
* **§5 Bars.** Paired \(d_r=\mathrm{err}_{\mathrm{specified},r}-\mathrm{err}_{\mathrm{best\,rival},r}\);
  one-sided 95% upper bound by Student-t with n−1 df **and** a 10 000-draw
  percentile bootstrap (seed 2026); both must agree, the larger bound is used.
  **BEST** if that bound \(<0\); **NON-INFERIOR** if that bound \(<\delta\),
  \(\delta=0.10\times\) planted magnitude of the graded channel for that cell
  (from the reused truth file). Role check as in V5.1. Holm across the four
  paired accuracy tests. "After Holm" means the CI verdict **and** the matching
  Holm-adjusted one-sided t p-value (p BEST for a BEST claim, p NI for a
  NON-INFERIOR claim) is \(<0.05\); both are required. **Gate PASS:** all four
  cells BEST or NON-INFERIOR after Holm, all four role checks pass, and the run
  finishes within the cap. Anything else is **FAIL**, except the
  calibration-INCOMPLETE stop of parent §3, the pre-flight abort of V5.2B §2
  (incorporated below), the rehearsal abort of §3 below, and the watchdog abort
  of §4 below, each of which leaves H undecided.
* **§6 Runtime cap, single pass, and kill rule**, with only the file-name
  substitutions in §6 of this file. Cap: single pass over all 140 tasks
  (n = 20 000 cells scheduled first) on 8 workers, wall cap **30 minutes**. No
  task dropped, retried, or re-run; unfinished cells are INCOMPLETE and never
  imputed. The 30-minute clock starts after a successful pre-flight **and** a
  successful rehearsal, at the first gate task; pre-flight time and rehearsal
  time do not count against the cap. **Kill rule:** if any cell fails accuracy,
  fails the role check, or is INCOMPLETE because of the wall cap, H is rejected;
  then record every failure in `PARK_V5_2C_SPECIFIED.md`, and do not amend, do
  not change seeds, and do not re-run to chase a bar. The calibration-INCOMPLETE
  stop, the pre-flight abort, the rehearsal abort, and the watchdog abort are
  each a stop that leaves H undecided: they are not a FAIL and they do not write
  a park file.

The following sections of
[PREREG_V5_2B_SPECIFIED_REPLICATION.md](PREREG_V5_2B_SPECIFIED_REPLICATION.md) at
`19ed3202c701a5ee924399885265b1996d4a13cc` are also incorporated here in full,
and are not rewritten, except that every mention of the V5.2b gate index ranges
(12000–12049, 12100–12119) is replaced by the fresh ranges of §1 of this file,
and that pre-flight is executed under the spawn-safe harness of §2:

* **§2 Mandatory pre-flight (the rule, not the process isolation).** Before any
  index in §1 is touched, and after calibration of \(\sigma_B\) is confirmed
  feasible: in the exact Python virtual environment that will run the gate,
  import TRACE (`import traceot`, and `explain_v5_2_specified`); import both
  rivals (the same DFL and CFM classes the runner will call); run one smoke fit
  of each of TRACE, DFL, and CFM on a dataset drawn with non-gate seed **99**
  (the oracle-truth seed; not a gate index; not in 13000–13049 or 13100–13119;
  not in 9600–9619, 10400–10799, 11000–11119, or 12000–12119), at n = 2000,
  p = 5, `skew_comp`, \(\kappa=1\), with the same raw-vs-centred split as the
  gate; record package versions of at least Python, `traceot`, numpy, scipy,
  and scikit-learn, together with the smoke-fit outcome of each method. Any
  failed import, failed smoke fit, or missing required package is a
  **pre-flight abort**: write the failure and the version record, do not touch
  any gate index, do not decide H, and do not redesign. It is not a FAIL and
  not a park. Because no gate seed was used, the seeds in §1 remain unspent,
  and a later attempt on those same pre-registered seeds is allowed only after
  a successful pre-flight (and, under this file, a successful rehearsal) in a
  repaired environment. Once a successful pre-flight **and** a successful
  rehearsal are followed by the start of the gate pass, the single-pass rule
  applies and those seeds are spent, including if a later task then fails or
  the watchdog of §4 then trips. The official results file must not already
  exist when the pre-flight starts. Pre-flight itself does not create that
  file. The smoke numbers are not a result and are not reported as one.
* **§3 Descriptive `tail_comp` composition under-estimate (secondary, report
  only).** The observation from the V5.2 specified pass is acknowledged and
  will be reported for every cell as a secondary descriptive note. It is not
  a scored bar: it does not enter Holm, BEST, NON-INFERIOR, the role check, or
  the gate PASS / FAIL / park decision, and it is not used to change the
  hypothesis.

---

## 1. (a) Fresh seeds

Generators other than the mean, `seed_of(...)` streams, estimand, and rivals are
unchanged from [PREREG_V5_2_SPECIFIED.md](PREREG_V5_2_SPECIFIED.md) §4 (and so
from PREREG_V5 §3–4 / PREREG_V5_1 §3). The scenario key is still extended by κ
(e.g. `seed_of("data", "skew_comp|k1", n, p, rep)`). The cells, sizes,
replication counts, and sample-size justification are those of parent §4. Only
the index ranges change:

| cell | n | p | indices | reps |
|---|---:|---:|---|---:|
| `skew_comp`, `tail_comp` | 2000 | 5 | **13000–13049** | 50 |
| `skew_comp`, `tail_comp` | 20 000 | 20 | **13100–13119** | 20 |

Four cells: 2 × 50 + 2 × 20 = **140 tasks**. All four are inside the gate; none is
optional. The same index range is reused across the two cells of one size, but
the data stream differs because the scenario key (including κ) enters `seed_of`,
exactly as in the parent.

### Seed inventory (every spent or reserved range) and the fresh range

Searched on `main` @ `b030932` (the merge of
[PR #63](https://github.com/jgridifier/traceot/pull/63), the V5.2b abort
record): every
pre-registration and results doc under `docs/benchmark/`, `docs/research/`,
`docs/theory/`, `docs/TRACE_SIMS.md`, every runner under `benchmarks/`, and the
`rep` field of every `benchmarks/results/*.jsonl`. The table below is the
inventory of [PREREG_V5_2B_SPECIFIED_REPLICATION.md](PREREG_V5_2B_SPECIFIED_REPLICATION.md)
§1, updated for the V5.2b abort. To stay safe across both namespaces, the
fresh range is disjoint from **all** of them, from the sealed holdout, and
from every spent specified, specified-replication, and coverage index.

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
| **10400–10799** | V5.1 coverage study; **spent** (PASS, [PR #59](https://github.com/jgridifier/traceot/pull/59)); not reused | `PREREG_V5_1_COVERAGE.md`, `v5_1_coverage.jsonl` |
| 10800–10999 | remainder of the 10300–10999 block surveyed as unused at the coverage freeze; left unused | `PREREG_V5_1_COVERAGE.md` |
| **11000–11049**, **11100–11119** | V5.2 specified-family gate; **spent** (INCOMPLETE, rivals unavailable, [PR #60](https://github.com/jgridifier/traceot/pull/60)); not reused | `PREREG_V5_2_SPECIFIED.md`, `v5_2_specified_gate_shape.jsonl` |
| 11120–11999 | buffer after the spent specified range; left unused | `PREREG_V5_2B_SPECIFIED_REPLICATION.md` |
| **12000–12049**, **12100–12119** | V5.2b specified-family replication; **spent** (ABORTED, no verdict, zero results); not reused | [V5_2B_ABORT.md](../benchmark/V5_2B_ABORT.md), this file |
| 12120–12999 | buffer after the spent V5.2b range; left unused | this file |
| 81001–81020, 83001–83020, 84001–84050 | trace_sims v2 items (raw seeds) | `docs/TRACE_SIMS.md` |
| fixed constants | oracle truth seed 99; paired-bootstrap seed 2026; ADOPT `SeedSequence` (own namespace); pre-flight smoke seed 99; rehearsal seeds 97 and 98 (this file, §3) | `sb_dgp_shape.py`, `summarize_*.py`, this file |

Numbers that look like seeds but are not: `10001` in `run_v1.py` (a monotonicity-check
grid size), `10007` / `20011` in TRACE_DESIGN (shuffle-seed offsets inside the
booster), and arXiv/DOI fragments in the literature notes (including 12004,
12017, 12027, 12945, 12958, which are not replication indices). SVG path
coordinates in `docs/learn/adopt.html` (for example 12937, 12953) are also not
indices.

**Fresh range: 13000–13049 and 13100–13119.** At this draft, on `main` @
`b030932`, from the repo root:

```
git grep -E '(^|[^0-9])13(00[0-9]|0[1-4][0-9]|10[0-9]|11[0-9])([^0-9]|$)'
```

returns no hits in any runner, results file, test, or pre-registration. No
`rep` field in `benchmarks/results/` equals any index in 13000–13049 or
13100–13119. The range is disjoint from every row of the table, including
9600–9619, 10400–10799, 11000–11119, and 12000–12119. The gap 13050–13099
between the two fresh blocks is left unused and is included in the runner's
`FORBIDDEN` set. The runner, when written after this freeze, must assert every
gate index lies in 13000–13049 or 13100–13119 and none lies in any range above
(a `FORBIDDEN` set, as in `run_v5_2b_specified_gate.py`), and must also forbid
9600–9619, 10400–10799, 11000–11049 / 11100–11119, and 12000–12049 /
12100–12119 **by name**.

---

## 2. (b) Spawn-safe harness

The V5.2b abort was caused by running the pre-flight TRACE smoke in the parent
process (which initialized Rayon) and then forking `multiprocessing.Pool(8)`.
This section is the protocol addition that addresses that, and it is frozen
before any new seed is used. It does not change the hypothesis, the method, the
cells, the bars, or what the pre-flight of V5.2B §2 checks.

**Pre-flight runs in a spawned child process.** The parent process that drives
`main()` does not `import traceot`, does not call `explain_v5_2_specified` or
any other TRACE entry point, and does not otherwise initialize a Rayon runtime.
The pre-flight of V5.2B §2 (imports, one smoke fit per method on seed 99,
version record) is executed in a child created with multiprocessing's
`'spawn'` start method. The parent waits for that child to exit and reads the
pre-flight record from a file the child wrote. The child's success or failure
is the pre-flight outcome of V5.2B §2, with the same abort rule. After the
child has exited, the parent still has not called TRACE.

**The gate pool (and the rehearsal pool of §3) is created with
multiprocessing's `'spawn'` start method, from that same parent, which has
never called TRACE.** Workers are fresh Python interpreters. Each worker may
import TRACE and call the method under test; that is expected and is how the
tasks run. The parent itself still does not call TRACE. The default `'fork'`
start method is not used for pre-flight, rehearsal, or the gate.

A regression test, in the runner pull request, runs this sequence on a tiny
non-gate seed (not a gate index, not in 9600–9619 or any spent range) and
asserts that the pool completes within a timeout. That test is not a gate
result and does not touch any index in §1.

---

## 3. (c) Mandatory REHEARSAL

The V5.2b pre-flight checked that each method can fit once, in a single
process. It did not check that the parallel pipeline can complete a task after
that. The deadlock appeared only when the pool started. This section is the
protocol addition that addresses that, and it is frozen before any new gate
seed is used.

**Before any index in §1 is touched**, after a successful pre-flight of V5.2B
§2 executed under §2 of this file:

The runner runs the **full parallel pipeline**, all three methods (TRACE, DFL,
CFM), through the **same runner** (the same task function, the same raw-vs-
centred split, the same spawn-safe pool, the same watchdog of §4), on **two
throwaway non-gate seeds per cell**. The rehearsal seeds are **98** and
**97**. They are not gate indices; they are not in 13000–13049 or 13100–13119;
they are not in 9600–9619, 10400–10799, 11000–11119, or 12000–12119. Each of
the four gate cells (`skew_comp` and `tail_comp`, at n = 2000, p = 5 and at
n = 20 000, p = 20, \(\kappa=1\)) is rehearsed on both seeds. That is
2 × 2 × 2 = **8 rehearsal tasks**. Rehearsal uses the same n and p as the cell
it rehearses, so the expensive size is exercised.

**Rehearsal outputs go to a separate file** (not the official gate results
file) and **never enter scoring**. They are not a result, are not reported as
a scored number, and are not used to decide H. They may be committed with the
results as a record that the rehearsal ran.

**Any rehearsal failure aborts before any gate seed is touched.** A failure
is any of: a task exception; a method (TRACE, DFL, or CFM) recording an
error; a missing or incomplete rehearsal record; a watchdog trip of §4 during
rehearsal; or the 8 tasks not all completing. Then: write the rehearsal
record, **do not touch any gate index**, **do not decide H**, and do not
redesign. This stop is the same kind of stop as a pre-flight abort: H is left
undecided; it is not a FAIL and not a park. Because no gate seed was used,
the seeds in §1 remain unspent, and a later attempt on those same
pre-registered seeds is allowed only after a successful pre-flight and a
successful rehearsal. The official gate results file is not created by a
rehearsal abort.

The 30-minute gate cap does not include rehearsal time. Rehearsal is not
itself a gate pass.

---

## 4. (d) Watchdog

**If no task completes within 5 minutes, abort and record ABORTED (not FAIL).**
H is left undecided. This is not a wall-cap INCOMPLETE, not an accuracy FAIL,
and not a park. It exists to stop a deadlock or a silent hang of the kind that
produced the V5.2b abort, rather than waiting for the 30-minute cap and
scoring a FAIL against an empty field.

The watchdog is measured from the start of the current pool (rehearsal pool or
gate pool, each on its own clock) to the first task completion, and, after
that, from each completion to the next: if 5 minutes elapse with no additional
completion, the watchdog trips. "Completes" means the runner receives a
finished task record from the pool and writes it to the appropriate file
(rehearsal file or official gate results file).

* **During rehearsal.** A watchdog trip is a rehearsal failure of §3: gate
  seeds are not touched, they remain unspent, H is not decided, the record is
  ABORTED (rehearsal watchdog), and it is not a FAIL.
* **During the gate pass.** A watchdog trip is **ABORTED (not FAIL)**. H is
  not decided and no park file is written. Because the gate pass has started,
  the seeds in §1 are spent, including if zero gate tasks had completed, the
  same way V5.2b's seeds were spent. They are not reused. The official results
  file, the wall file (with `stop` equal to a watchdog-abort marker), and
  every record written so far are kept. Unfinished tasks are not imputed and
  are not retried. If at least one task had already completed, those records
  are kept as a descriptive record; they still do not produce a PASS or a
  FAIL, because the pass was aborted and H is not decided.

The 30-minute cap of parent §6 still applies to the gate, starting at the
first gate task, and is a separate rule: unfinished cells at the cap are
INCOMPLETE and, under the inherited kill rule, H is rejected. The watchdog is
the earlier stop for a hang in which *nothing is completing*. It does not
shorten the cap when tasks are completing on time.

---

## 5. Order of work (nothing on gate seeds before step 5)

1. This file, with the approval already in the header, is merged to `main`.
   That merge SHA is the freeze. This pull request runs nothing on 13000–13049
   or 13100–13119. Indices **9600–9619** stay sealed. Indices 12000–12049 and
   12100–12119 are not re-run.
2. A runner pull request, after the freeze, implements the spawn-safe harness
   of §2, the rehearsal of §3, the watchdog of §4, the pre-flight of V5.2B §2
   (by reference, executed in a spawned child), and the fresh-seed ranges of
   §1. It records the freeze SHA. The `FORBIDDEN` guard includes every range
   in the inventory, and names 9600–9619, 10400–10799, 11000–11049 /
   11100–11119, and 12000–12049 / 12100–12119 by name. The "results file must
   not exist" single-pass check is as in `run_v5_2b_specified_gate.py`. The
   method, simulator, and truth file already committed at `1aab477` are reused.
   The runner pull request runs nothing on the gate seeds. Unit tests use only
   non-gate seeds (small fixed seeds below 1000, including the smoke seed 99
   and the rehearsal seeds 97 and 98). The regression test of §2 uses a tiny
   non-gate seed and a timeout; it is not a gate run.
3. Merge the runner. Record the runner merge SHA in the results file's header,
   which does not yet exist.
4. Only then, in a virtual environment that includes scikit-learn: run the
   pre-flight of V5.2B §2, in a spawned child, as in §2 of this file. If it
   fails, stop as in V5.2B §2; do not touch gate seeds.
5. If the pre-flight passes: run the rehearsal of §3. If it fails, stop as in
   §3; do not touch gate seeds.
6. If the rehearsal passes: one single pass on 13000–13049 and 13100–13119,
   with the 30-minute cap starting at this step, under the spawn-safe pool and
   the watchdog of §4.

Indices 9600–9619 are not touched by any step. Indices 10400–10799,
11000–11049 / 11100–11119, and 12000–12049 / 12100–12119 are not re-run.

---

## 6. Report

`docs/benchmark/V5_2C_SPECIFIED_GATE_RESULTS.md` with this file's freeze SHA,
the implementation SHA (`1aab4778abf8f84c8c8f7d9603eb8c322f1ba3c9`, reused),
the runner SHA, the pre-flight record (package versions and smoke outcomes),
the rehearsal record (completed or aborted; not scored), the four-cell table
(err specified, err DFL, err CFM, mean d, upper t, upper boot, \(\delta\),
verdict, \(|\hat c|\), \(|\hat s|\), role, p NI / p BEST with Holm) if the
gate produced records to score, wall time of the gate pass, every failure
kept, the watchdog outcome, and the secondary descriptive report of V5.2B §3
for every cell. No result numbers appear in this file. If the pass is
ABORTED, that file records ABORTED (no verdict), the reason, and that H is
not decided, instead of a PASS or FAIL.

Learn page of this pre-registration:
[v5_2c_specified_prereg.html](../learn/v5_2c_specified_prereg.html).
Results, after the single pass:
`docs/learn/v5_2c_specified.html`.
