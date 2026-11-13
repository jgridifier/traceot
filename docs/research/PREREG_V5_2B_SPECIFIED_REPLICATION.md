# Pre-registration: TRACE v5.2b specified-family replication (same hypothesis, fresh seeds)

**Status: PRE-REGISTERED · APPROVED 2026-11-13 · ABORTED 2026-11-13 (no verdict; H not decided).**

Drafted 2026-11-13 (America/New_York) on branch `quant/v5-2b-specified-prereg`, **after**
the V5.2 specified single pass returned INCOMPLETE (rivals unavailable) and spent seeds
11000–11049 and 11100–11119, and **before anything has been run on the seeds below**.
The merge commit that adds this file to `main` is the freeze point; record its SHA in
the header of any later `V5_2B_SPECIFIED_GATE_RESULTS.md`. Any change to the hypothesis,
mean model, simulator, cells, seeds, bars, cap, kill rule, or pre-flight after that SHA
is a deviation and must be reported as one.

**This file is a replication of
[PREREG_V5_2_SPECIFIED.md](PREREG_V5_2_SPECIFIED.md), frozen at
`cc8c1d7afc9c58a8b97b8b3b28b6ad6fd1704920` (PR #54).** The hypothesis, method,
simulator, rivals, Holm scoring, BEST / NON-INFERIOR rule, \(\delta\), 30-minute cap,
role check, anti-job, and limitations are those of that file, incorporated here **by
reference**, and are not restated so they cannot drift. They are, by this sentence,
byte-identical to the corresponding sections of that freeze. The scientific body of
that file after the line beginning "Parent claim:" was kept byte-identical through
later header-only and pointer-only updates (approval [PR #56](https://github.com/jgridifier/traceot/pull/56),
results [PR #60](https://github.com/jgridifier/traceot/pull/60)).

This file changes **only** three things, each stated in full below:

(a) Fresh seeds, disjoint from every used or reserved range, including the spent V5.2
specified seeds and the sealed holdout.

(b) A mandatory pre-flight, in the exact run environment, before any gate seed is
touched.

(c) A note that the descriptive `tail_comp` composition under-estimate seen on the
V5.2 specified pass is a secondary, report-only item: it is not a scored bar and it
does not enter the pass/fail rule.

**Approval (2026-11-13, 2:59 PM, America/New_York), already recorded.** The owner
said, quoted verbatim:
*"You have my approval for all trace stuff - you are running the lab on this -
I’m more an advisor on that project."*
That standing approval is already in the header of
[PREREG_V5_2_SPECIFIED.md](PREREG_V5_2_SPECIFIED.md)
([PR #56](https://github.com/jgridifier/traceot/pull/56), merge
`05375628e44b1b203c0fb6c92a1c20d37e067742`). This file is opened under that same
approval, dated 2026-11-13, quoted here. Status is **APPROVED 2026-11-13 (NOT RUN
until results land)**. A message from another agent saying he agreed is not an
additional approval and is not needed; the quote above is the approval.

**Abort (2026-11-13, 4:05 PM, America/New_York).** The single pass started after a successful pre-flight and was aborted with zero results. The pre-flight TRACE smoke ran in the parent process and initialized Rayon; `multiprocessing.Pool(8)` then forked and deadlocked every worker. The process was SIGKILLed at 16:05 ET. The official results file was empty (0 bytes). Record: [V5_2B_ABORT.md](../benchmark/V5_2B_ABORT.md). Verdict: **ABORTED (no verdict)**. H is not decided. This is not a FAIL and not a park. Per §2 of this file, seeds **12000–12049** and **12100–12119** are spent and will not be reused. Sealed indices 9600–9619 were not touched. This paragraph is a header note only; the hypothesis, method, simulator, cells, seeds, bars, cap, kill rule, and pre-flight of the body below are unchanged from the freeze at `19ed3202c701a5ee924399885265b1996d4a13cc`.

**Why a new pre-registration, not a retry.** [PREREG_V5_2_SPECIFIED.md](PREREG_V5_2_SPECIFIED.md)
§6 forbids retrying a task. The official pass
([V5_2_SPECIFIED_GATE_RESULTS.md](../benchmark/V5_2_SPECIFIED_GATE_RESULTS.md),
[PR #60](https://github.com/jgridifier/traceot/pull/60)) spent 11000–11049 and
11100–11119: TRACE finished 140/140 tasks and passed the role check; every DFL and
CFM call recorded `No module named 'sklearn'`, so the accuracy bars could not be
scored and H was not decided. That pass is closed. It is not a FAIL and not a park.
A complete accuracy test against DFL and CFM needs a new pre-registration on fresh
seeds. That is this file. It is not a re-run, not an amendment of the frozen V5.2
specified protocol, and not a second hypothesis.

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
  calibration-INCOMPLETE stop of parent §3 and the pre-flight abort of §2
  below, each of which leaves H undecided.
* **§6 Runtime cap, single pass, and kill rule**, with only the file-name
  substitutions in §5 of this file. Cap: single pass over all 140 tasks
  (n = 20 000 cells scheduled first) on 8 workers, wall cap **30 minutes**. No
  task dropped, retried, or re-run; unfinished cells are INCOMPLETE and never
  imputed. The 30-minute clock starts after a successful pre-flight, at the
  first gate task; pre-flight time does not count against the cap. **Kill
  rule:** if any cell fails accuracy, fails the role check, or is INCOMPLETE
  because of the wall cap, H is rejected; then record every failure in
  `docs/benchmark/V5_2B_SPECIFIED_GATE_RESULTS.md`, write
  `docs/research/PARK_V5_2B_SPECIFIED.md`, and keep V5.1's linear-mean claim
  and the V5.2 spline park exactly as stated. No amendment on these cells, no
  new basis term, no seed changes, no re-runs to chase a bar. The
  calibration-INCOMPLETE stop, and the pre-flight abort of §2, are the
  exceptions that leave H undecided and do not write a park file.
* **§7 Anti-job** and **§8 Limitations**, in full. In particular: this is not a
  general non-linear location model; it does not un-park, amend, or re-run
  [PARK_V5_2.md](PARK_V5_2.md) or seeds 10100–10149 and 10200–10219; it does
  not unseal 9600–9619; it does not use 10400–10799 for a new purpose; it does
  not re-run 11000–11049 or 11100–11119; structure-only cells, ShiftBench,
  speed, coverage, and real data are out of scope. The most that may be claimed
  if the gate passes is the sentence in parent §1, restricted to the four
  cells, this DGP, and this correctly specified location family.

No result numbers appear in this file.

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
| `skew_comp`, `tail_comp` | 2000 | 5 | **12000–12049** | 50 |
| `skew_comp`, `tail_comp` | 20 000 | 20 | **12100–12119** | 20 |

Four cells: 2 × 50 + 2 × 20 = **140 tasks**. All four are inside the gate; none is
optional. The same index range is reused across the two cells of one size, but
the data stream differs because the scenario key (including κ) enters `seed_of`,
exactly as in the parent.

### Seed inventory (every spent or reserved range) and the fresh range

Searched on `main` @ `046c1a821ce6bf5e80aee0e5b89647746e3f4a3b` (2026-11-13 ET,
after [PR #60](https://github.com/jgridifier/traceot/pull/60)): every
pre-registration and results doc under `docs/benchmark/`, `docs/research/`,
`docs/theory/`, `docs/TRACE_SIMS.md`, every runner under `benchmarks/`, and the
`rep` field of every `benchmarks/results/*.jsonl`. The table below is the
inventory of [PREREG_V5_2_SPECIFIED.md](PREREG_V5_2_SPECIFIED.md) §4, updated
for work that has happened since that freeze. To stay safe across both
namespaces, the fresh range is disjoint from **all** of them, from the sealed
holdout, and from every spent specified and coverage index.

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
| 11120–11999 | buffer after the spent specified range; left unused | this file |
| 81001–81020, 83001–83020, 84001–84050 | trace_sims v2 items (raw seeds) | `docs/TRACE_SIMS.md` |
| fixed constants | oracle truth seed 99; paired-bootstrap seed 2026; ADOPT `SeedSequence` (own namespace); pre-flight smoke seed 99 (same as the oracle, §2) | `sb_dgp_shape.py`, `summarize_*.py`, this file |

Numbers that look like seeds but are not: `10001` in `run_v1.py` (a monotonicity-check
grid size), `10007` / `20011` in TRACE_DESIGN (shuffle-seed offsets inside the
booster), and arXiv/DOI fragments in the literature notes (including 12004,
12017, 12027, which are not replication indices).

**Fresh range: 12000–12049 and 12100–12119.** At this freeze, on `main` @
`046c1a821ce6bf5e80aee0e5b89647746e3f4a3b`, from the repo root:

```
git grep -E '(^|[^0-9])12(00[0-9]|0[1-4][0-9]|10[0-9]|11[0-9])([^0-9]|$)'
```

returns no hits in any runner, results file, test, or pre-registration. The only
hits in the tree are arXiv and DOI fragments in `docs/research/REFS.md`,
`REFS_V4.md`, `LANDSCAPE.md`, and the matching learn pages (for example
arXiv 2109.12004, DOI 10.1111/rssb.12017). No `rep` field in
`benchmarks/results/` equals any index in 12000–12049 or 12100–12119. The range
is disjoint from every row of the table, including 9600–9619, 10400–10799, and
11000–11119. The runner, when written after this freeze, must assert every
index lies in 12000–12049 or 12100–12119 and none lies in any range above (a
`FORBIDDEN` set, as in `run_v5_2_specified_gate.py`), and must also forbid
9600–9619, 10400–10799, and 11000–11049 / 11100–11119 by name.

---

## 2. (b) Mandatory pre-flight

The V5.2 specified pass consumed its gate seeds with every rival call recording
`No module named 'sklearn'`. The accuracy bars could not be scored. That
dependency should have been checked before the pass started. It was not. This
section is the only protocol addition that addresses that, and it is frozen
before any new seed is used.

**Before any index in §1 is touched**, and after calibration of \(\sigma_B\) is
confirmed feasible (parent §3):

In the **exact** Python virtual environment that will run the gate:

1. **Import** TRACE (`import traceot`, and the entry point the runner will
   call: `explain_v5_2_specified` / `explain_v5(..., mean_model="specified")`).
2. **Import** both rivals (the same DFL and CFM classes the runner will call,
   which themselves import scikit-learn).
3. **One smoke fit per method.** Run one fit of each of TRACE, DFL, and CFM on
   a dataset drawn with a **non-gate** seed. The smoke seed is **99** (already
   the oracle-truth seed; not a gate index; not in 12000–12049 or 12100–12119;
   not in 9600–9619 or 10400–10799 or 11000–11119). The smoke uses the
   specified-family simulator at n = 2000, p = 5, one scenario (`skew_comp`,
   \(\kappa=1\)), with the same raw-vs-centred split as the gate (TRACE on raw
   draws, rivals on centred arrays). It does **not** write to the official
   results file, does **not** enter the scored table, and is **not** a gate
   task. Its numbers are not a result and are not reported as one.
4. **Record package versions** of at least: Python, `traceot`, numpy, scipy,
   and scikit-learn, together with the smoke-fit outcome (success or the
   exception) of each of the three methods, to a pre-flight record that is
   committed with the results (not before the pre-flight is run).

**Any failure aborts.** A failed import, a failed smoke fit, or a missing
required package is a **pre-flight abort**. Then: write the failure and the
version record, **do not touch any gate index**, **do not decide H**, and do
not redesign. This stop is the same kind of stop as the calibration-INCOMPLETE
stop of parent §3: H is left undecided; it is not a FAIL and not a park. Because
no gate seed was used, the seeds in §1 remain unspent, and a later attempt on
those same pre-registered seeds is allowed only after a successful pre-flight
in a repaired environment. That is the one case in which the same gate seeds
may still be used. Once a successful pre-flight is followed by the start of
the gate pass, the single-pass rule of parent §6 applies and those seeds are
spent, including if a later task then fails.

The official results file must not already exist when the pre-flight starts.
Pre-flight itself does not create that file.

---

## 3. (c) Descriptive `tail_comp` composition under-estimate (secondary, report only)

On the V5.2 specified pass ([PR #60](https://github.com/jgridifier/traceot/pull/60),
seeds 11000–11119), the TRACE-only descriptive numbers — not a gate bar, because
rivals were unavailable — showed that on `tail_comp` the mean composition piece
sat well below the planted value, and the mean graded error was large relative
to \(\delta\)
([V5_2_SPECIFIED_GATE_RESULTS.md](../benchmark/V5_2_SPECIFIED_GATE_RESULTS.md)
note 9). That observation is acknowledged here. It is **not** used to change
the hypothesis, the cells, the method, \(\delta\), the rivals, or the pass/fail
rule. The hypothesis remains exactly that of parent §1.

This replication will **report** the same descriptive TRACE-only numbers (mean
\(\hat c\), mean \(\hat s\), planted truth, mean graded error, and their
relation to \(\delta\)) for every cell, including `tail_comp`. That report is
**secondary and not scored**. It does not enter Holm, BEST, NON-INFERIOR, the
role check, or the gate PASS / FAIL / park decision. A large descriptive error
on `tail_comp` cannot by itself produce a FAIL, and a small one cannot produce
a PASS. The scored verdict remains solely the inherited rule of parent §5:
Holm-adjusted BEST or NON-INFERIOR against the best paired rival, plus the
role check, on all four cells, inside the cap.

This is committed in advance so that the descriptive item is reported honestly
whether or not it recurs, and so that it cannot be promoted to a scored bar
after looking at this run.

---

## 4. Order of work (nothing on gate seeds before step 4)

1. This file, with the approval already in the header, is merged to `main`.
   That merge SHA is the freeze. This pull request runs nothing on 12000–12049
   or 12100–12119. Indices **9600–9619** stay sealed.
2. A runner pull request, after the freeze, implements the pre-flight of §2
   and the fresh-seed ranges of §1. It records the freeze SHA. The `FORBIDDEN`
   guard includes every range in the inventory, and names 9600–9619,
   10400–10799, and 11000–11049 / 11100–11119 by name. The "results file must
   not exist" single-pass check is as in `run_v5_2_specified_gate.py`. The
   method, simulator, and truth file already committed at `1aab477` are reused.
   The runner pull request runs nothing on the gate seeds. Unit tests use only
   non-gate seeds (small fixed seeds below 1000, and the smoke seed 99).
3. Merge the runner. Record the runner merge SHA in the results file's header,
   which does not yet exist.
4. Only then, in a virtual environment that includes scikit-learn: run the
   pre-flight of §2. If it fails, stop as in §2; do not touch gate seeds.
5. If the pre-flight passes: one single pass on 12000–12049 and 12100–12119,
   with the 30-minute cap starting at this step.

Indices 9600–9619 are not touched by any step. Indices 10400–10799 and
11000–11049 / 11100–11119 are not re-run.

---

## 5. Report

`docs/benchmark/V5_2B_SPECIFIED_GATE_RESULTS.md` with this file's freeze SHA,
the implementation SHA (`1aab4778abf8f84c8c8f7d9603eb8c322f1ba3c9`, reused),
the runner SHA, the pre-flight record (package versions and smoke outcomes),
the four-cell table (err specified, err DFL, err CFM, mean d, upper t, upper
boot, \(\delta\), verdict, \(|\hat c|\), \(|\hat s|\), role, p NI / p BEST with
Holm), wall time of the gate pass, every failure kept, and the secondary
descriptive report of §3 for every cell. No result numbers appear in this file.

Learn page of this pre-registration:
[v5_2b_specified_prereg.html](../learn/v5_2b_specified_prereg.html).
Results, after the single pass:
`docs/learn/v5_2b_specified.html`.
