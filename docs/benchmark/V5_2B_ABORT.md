# TRACE v5.2b specified-family replication — abort (no verdict)

**Prereg SHA:** `19ed3202c701a5ee924399885265b1996d4a13cc`
([PREREG_V5_2B_SPECIFIED_REPLICATION.md](../research/PREREG_V5_2B_SPECIFIED_REPLICATION.md),
[PR #61](https://github.com/jgridifier/traceot/pull/61)).
**Implementation merge SHA:** `1aab4778abf8f84c8c8f7d9603eb8c322f1ba3c9`
([PR #58](https://github.com/jgridifier/traceot/pull/58)), reused, not redesigned.
**Runner merge SHA:** `45d0d74` — the merge of
[PR #62](https://github.com/jgridifier/traceot/pull/62)
(`f8ff4b2`, `run_v5_2b_specified_gate.py`).
**Approval:** 2026-11-13, 2:59 PM, America/New_York, already in the pre-registration
header under the owner's standing approval
([PR #56](https://github.com/jgridifier/traceot/pull/56), merge
`05375628e44b1b203c0fb6c92a1c20d37e067742`).
**Verdict:** **ABORTED (no verdict)**. H is not decided. This is not a FAIL and is
not a park. There is no scored four-cell table, because no gate task completed.
**Seeds 12000–12049 and 12100–12119 are spent** and will not be reused. Sealed
indices 9600–9619 were not touched. Indices 10400–10799 and 11000–11049 /
11100–11119 were not re-run.

This file is the official record of the abort. It is not a results file of a
completed pass, and it is not a park.

---

## Timeline (America/New_York, EST, UTC−5)

All times below are from the files in `/tmp/v52b-abort-evidence/`, which were
copied from the run directory at the moment of the kill. The official results
path was
`/workspace/traceot-v52b-run/benchmarks/results/v5_2b_specified_gate_shape.jsonl`.

1. **15:53:11.** The single pass started in `/workspace/v52b-venv` (Python 3.13.5,
   scikit-learn 1.9.1, TRACE 0.1.0 built from the runner merge), working directory
   `benchmarks/shiftbench`, command
   `/workspace/v52b-venv/bin/python run_v5_2b_specified_gate.py --workers 8`.
   The runner planned **140 tasks**, reported no infeasible-calibration cells, and
   began the mandatory pre-flight on non-gate seed 99.
2. **15:53.** Pre-flight finished successfully. Imports of TRACE, DFL, and CFM
   were ok; one smoke fit of each method on seed 99 (`skew_comp`, n = 2000, p = 5)
   was ok; package versions were recorded (see the JSON below). The runner printed
   `pre-flight OK` and wrote
   `v5_2b_specified_gate_preflight.json` (`ok: true`). Per
   [PREREG_V5_2B_SPECIFIED_REPLICATION.md](../research/PREREG_V5_2B_SPECIFIED_REPLICATION.md)
   §2 and §4, the 30-minute cap starts at this point, at the first gate task.
   Because the pre-flight passed, the runner then created `multiprocessing.Pool(8)`
   in the same process and submitted the 140 gate tasks. That is the start of the
   gate pass.
3. **15:53 to 16:05, about twelve minutes.** No task completed. The official
   results file was opened in append mode (so it existed) and remained
   **zero bytes** — no JSONL record was written. No wall file was written.
4. **16:05.** The process (PID 1118278) was **SIGKILLed**. The shell recorded
   `--: line 18: 1118278 Killed`. A `py-spy` attach attempted after the kill
   returned only `process already gone` (saved as `stack.txt`). The results
   directory was then cleaned of the empty official file; a 0-byte copy is kept
   in the evidence directory as `empty.jsonl`.

dmesg has no OOM-killer event at 16:05. Earlier, unrelated OOM kills at 14:33
and 14:38 ET targeted different PIDs and are not this run. The kill was a
deliberate SIGKILL after a hang with zero completions, not an out-of-memory
death and not the 30-minute cap (the cap had about 18 minutes left; SIGKILL
also skipped the runner's `finally` block, which is why no wall file was
written).

For comparison, earlier TRACE gates on this machine finished 140 tasks in
about 15 seconds (V5.1) to about a minute (V5.2 spline, 490 of 560 tasks).
Twelve minutes with **zero** completions is not a slow rival fit.

---

## Cause

The pre-flight TRACE smoke ran **in the parent process** and initialized
Rayon. The runner then built `multiprocessing.Pool(8)`, which on Linux
defaults to the `'fork'` start method. Every forked worker deadlocked.

This is read from the runner as merged, not from a live stack (py-spy could
not attach after the kill):

* `preflight()`, called in `main()` of the same process that later creates
  the pool, does `import traceot` and `traceot.explain_v5_2_specified(...)`.
  That loads the Rust extension and starts a Rayon thread pool in the parent.
* `collect_versions()`, also called from `preflight()`, does
  `__import__("traceot")` in the same process.
* After `pre-flight OK`, `main()` does `pool = Pool(args.workers)` with
  `from multiprocessing import Pool`, which uses the default start method.
  On this Linux host that default is `'fork'`.
* Forking a process that already has a Rayon runtime is unsafe: worker
  threads are not reproduced across `fork`, mutexes can be left locked, and
  the child deadlocks on the next Rayon call. Each of the eight workers has
  to call TRACE (and so Rayon) to do its first task. None returned.

The evidence is consistent with that mechanism and does not show a second
cause: pre-flight itself succeeded (TRACE ran to completion once, in the
parent); the hang begins exactly when the forked pool is created; zero of
140 tasks wrote a record; py-spy after the SIGKILL could not capture a
live stack (`process already gone`).

---

## What was seen (zero results)

* Official results file: **empty** (0 bytes, 0 records). Copied as
  `/tmp/v52b-abort-evidence/empty.jsonl`.
* Wall file: **not written** (SIGKILL skipped `finally`).
* Pre-flight record: **written, `ok: true`**, preserved below and at
  `benchmarks/results/v5_2b_specified_gate_preflight.json`.
* No scored table, no TRACE numbers, no rival numbers, no role-check
  numbers. Nothing from this pass enters a verdict.

The run log, verbatim:

```
=== START Fri, 13 Nov 2026 15:53:11 -0500 ===
python: /workspace/v52b-venv/bin/python
cwd: /workspace/traceot-v52b-run/benchmarks/shiftbench
sklearn 1.9.1
traceot 0.1.0
v5.2b specified gate: 140 tasks -> /workspace/traceot-v52b-run/benchmarks/results/v5_2b_specified_gate_shape.jsonl (cap 1800s); cells with infeasible calibration (INCOMPLETE, no tasks): []
pre-flight: importing TRACE and rivals, smoke fits on seed 99
pre-flight OK: traceot=0.1.0, numpy=2.5.4, scipy=1.18.1, scikit-learn=1.9.1
--: line 18: 1118278 Killed                  /workspace/v52b-venv/bin/python run_v5_2b_specified_gate.py --workers 8
```

---

## Seeds declared spent

[PREREG_V5_2B_SPECIFIED_REPLICATION.md](../research/PREREG_V5_2B_SPECIFIED_REPLICATION.md)
§2, frozen at `19ed320`, says:

> Once a successful pre-flight is followed by the start of the gate pass,
> the single-pass rule of parent §6 applies and those seeds are spent,
> including if a later task then fails.

The pre-flight succeeded. The gate pass started: the pool was created and
the 140 tasks were submitted. Under that letter, **12000–12049 and
12100–12119 are spent**, even though zero tasks completed. The lab
decision (Quant, 2026-11-13) is to follow that letter and **not reuse
them**. This is the one-pass rule, not a retry ban being applied to a
pre-flight abort: a pre-flight *failure* would have left the seeds
unspent (§2); that is not what happened.

H is not decided. The abort is a harness deadlock followed by a SIGKILL,
not an accuracy FAIL, not a role-check FAIL, and not a wall-cap
INCOMPLETE. The kill rule of parent §6 that writes a park file is not
triggered. The V5.1 linear-mean claim is unchanged. The V5.2 spline park
stays in force. The V5.2 specified pass remains INCOMPLETE (rivals
unavailable).

A complete accuracy test against DFL and CFM needs a new pre-registration
on fresh seeds, with a harness that does not fork after Rayon has been
initialized in the parent. This file does not open that pre-registration.

---

## Pre-flight JSON

The file written at 15:53 ET, copied from
`/tmp/v52b-abort-evidence/v5_2b_specified_gate_preflight.json`, and
committed at `benchmarks/results/v5_2b_specified_gate_preflight.json`:

```json
{
  "versions": {
    "python": "3.13.5 (main, Aug 10 2026, 12:06:59) [GCC 14.2.0]",
    "traceot": "0.1.0",
    "numpy": "2.5.4",
    "scipy": "1.18.1",
    "scikit-learn": "1.9.1"
  },
  "imports": {
    "sklearn": "ok",
    "traceot": "ok",
    "DFL": "ok",
    "CFM": "ok"
  },
  "smoke": {
    "TRACE": "ok",
    "DFL": "ok",
    "CFM": "ok"
  },
  "ok": true,
  "smoke_seed": 99,
  "smoke_scen": "skew_comp",
  "smoke_n": 2000,
  "smoke_p": 5,
  "prereg_sha": "19ed3202c701a5ee924399885265b1996d4a13cc"
}
```

The smoke seed is 99, a non-gate index, as frozen in §2. No index in
12000–12049 or 12100–12119 was used for a completed task. Sealed indices
9600–9619 were not used.

---

## Sealed and reserved seeds

Indices **9600–9619** were not touched. Indices 10400–10799 (spent V5.1
coverage) and 11000–11049 / 11100–11119 (spent V5.2 specified) were not
re-run.
