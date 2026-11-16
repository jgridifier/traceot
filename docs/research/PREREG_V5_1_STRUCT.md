# Pre-registration: TRACE v5.1 on the structure-only and mixed shape cells (one hypothesis)

**Status: PRE-REGISTERED — not run.**

Drafted 2026-11-16 (America/New_York) on branch `quant/prereg-v5-1-struct`, on
`main` @ `ad2b791` (merge of PR #69), **before anything has been run on the seeds
below**. The merge commit that adds this file to `main` is the freeze point; record
its SHA in the header of any later `V5_1_STRUCT_GATE_RESULTS.md`. Any change to the
hypothesis, method, cells, seeds, bars, cap, kill rule, or harness after that SHA is
a deviation and must be reported as one.

**Approval.** Opened under the owner's standing approval for this lab, dated
2026-11-13, quoted verbatim in the header of
[PREREG_V5_2C_SPECIFIED_REPLICATION.md](PREREG_V5_2C_SPECIFIED_REPLICATION.md):
*"You have my approval for all trace stuff - you are running the lab on this - I’m
more an advisor on that project."* A message from another agent saying he agreed is
not an additional approval and is not needed.

Nothing is run under this file. It only freezes what will be run.

---

## 0. Why this suite, and not the others

Since the V5.2c park ([PARK_V5_2C_SPECIFIED.md](PARK_V5_2C_SPECIFIED.md)) the only
validated claim is V5.1's: on a linear-mean simulator, `TRACE_V5_1` is BEST against
DFL and CFM on the two **pure-composition** shape cells (`skew_comp`, `tail_comp`) at
both sizes ([V5_1_GATE_RESULTS.md](../benchmark/V5_1_GATE_RESULTS.md)). Every later
freeze listed four out-of-scope suites: struct cells, ShiftBench, speed, and
per-feature shares. This file picks **struct cells** (plus the mixed cell that
belongs to the same H7 shape family) for these reasons:

* **It closes the obvious hole in the existing claim.** A method that labelled every
  shift "composition" would also pass the V5.1 gate. The structure-only cells
  (composition truth ≈ 0) and the mixed cell (both roles planted, opposite signs)
  are the direct falsification test of that. A PASS turns "BEST on composition
  cells" into "separates composition from structure on the full H7 shape family",
  which is the claim a working paper needs.
* **Nothing new has to be designed.** The simulator (`sb_dgp_shape.py`), its truth
  file (`shiftbench_truth_shape.json`, oracle seed 99, already committed, already
  holding `skew_struct`, `tail_struct`, `shape_mixed`), the method
  (`explain_v5_1`), the rivals, the scorer, the bars, and the spawn-safe harness all
  exist. The only freedoms left are the ones fixed in this file, so there are few
  researcher degrees of freedom.
* **The hypothesis is not chosen by looking at V5.1 on these cells.** `TRACE_V5_1`
  has never been run on `skew_struct`, `tail_struct`, or `shape_mixed` (every V5.x
  freeze excluded them). The only earlier numbers on these cells are v4's
  (a different method; all FAIL, [V4_RESULTS.md](../benchmark/V4_RESULTS.md)) and
  V5.2's curved-mean spline (a different simulator; [PARK_V5_2.md](PARK_V5_2.md)).
  Those are named here as the known risk, not as tuning input.
* **It is cheap.** 210 tasks of the same size as the 140-task V5.1 gate, which ran
  in 15.0 s.

Why not the others now:

* **ShiftBench.** Different estimand (mean / sd / quantile channels with a
  per-feature mean truth, `shiftbench_truth.json`) and it includes a `hetero`
  scenario that is outside V5.1's homoskedastic-residual design by construction.
  It needs a new adapter from `explain_v5` output to ShiftBench channels and new
  channel choices before any freeze, and v4 already failed `mixed` / `hetero`, so
  part of the hypothesis would be shaped by a known failure. Higher design risk,
  lower prior.
* **Speed.** The inherited bar is ≥ 5× CFM; v4 reached 2.94× at n = 10⁶ and DFL is
  faster at small n. A speed result alone is an engineering note, not a
  publishable claim, and it says nothing if the accuracy breadth is unknown.
  Per-method wall time is reported **descriptively** in this suite instead (§8).
* **Per-feature shares.** The eventual differentiator of TRACE, but there is no
  committed per-feature truth for the shape (L-moment) channel and no order-free
  per-feature rival built from DFL / CFM. It needs a new oracle and a new rival
  before a freeze — the largest design surface of the four. It is also only
  meaningful once the composition / structure split itself is shown to be right
  on structure and mixed shifts, which is what this suite tests. It is the
  recommended **next** suite after this one.

## 1. Hypothesis (the only one)

**H.** On the frozen linear-mean H7 shape simulator, `TRACE_V5_1`, unchanged, is
**BEST or NON-INFERIOR** against the best paired rival (DFL, CFM) after Holm on all
six cells below — the two structure-only scenarios (`skew_struct`, `tail_struct`)
and the mixed scenario (`shape_mixed`), each at n = 2000 and n = 20 000 — **and**
passes the role check in all six cells, **and** passes the false-attribution check
in all four structure-only cells.

## 2. Method under test (fixed; no change)

`TRACE_V5_1` = `traceot.explain_v5_1(XA, yA, XB, yB)` =
`explain_v5(..., mean_model="ols")`, exactly as gated by
[PREREG_V5_1.md](PREREG_V5_1.md) (freeze `b33f507`, implementation merge
`665939bd4aa78ffc7474b91d482d18ac0191d595`): OLS location fit per period, empirical
residual law, four hybrids, 256-point grid, `n_quad = 64`, exact L-moments,
two-player Shapley split. Rust core, thin Python binding, no neural nets, **no
tunable parameters**. The report-only residual-vs-fit check added later does not
enter the split and is not used.

The runner PR (§9 step 1) must state whether any commit after `665939b` touched the
`explain_v5` OLS code path in `crates/core`; if one did, it must show, on oracle
seed 99 only (never a gate seed), that `explain_v5_1` output on one `skew_struct`
draw is bit-identical between `665939b` and the runner SHA. If it is not, stop:
that is a method change and needs its own pre-registration.

## 3. Simulator, truth, rivals, estimand (all reused, none recomputed)

* **Simulator:** `benchmarks/shiftbench/sb_dgp_shape.py`, unchanged. In
  `skew_struct` and `tail_struct`, X has the same law in both periods; only the
  noise law changes (skew, t5). In `shape_mixed`, column 0 changes to chi3 **and**
  the noise law changes to negative skew.
* **Truth:** `benchmarks/results/shiftbench_truth_shape.json` as committed (oracle
  seed 99), **not recomputed**. Graded channel λ₃ for `skew_*` and `shape_mixed`,
  λ₄ for `tail_*`. Planted Shapley values from that file:

  | scenario | channel | truth comp | truth struct | δ = 0.10 × (\|c\|+\|s\|) |
  |---|---|---:|---:|---:|
  | `skew_struct` | λ₃ | −0.000351 | 0.050369 | 0.005072 |
  | `tail_struct` | λ₄ | 0.0000851 | 0.009742 | 0.0009827 |
  | `shape_mixed` | λ₃ | 0.039513 | −0.053466 | 0.009298 |

  (δ values match the v4 results table for these cells.)
* **Rivals:** DFL and CFM (`sb_shape_readout.DFL_L`, `CFM_L`), re-run paired on the
  same data and the same method seed, as in `run_v5_1_gate.py`.
* **Estimand and error:** graded-channel `lsplit` vs the Shapley truth, scored with
  `graded_err` exactly as in `summarize_v5_1_gate.py`.
* **Streams:** `seed_of("data", scen, n, p, rep)` and `seed_of("m", rep)`, as in
  `run_v5_1_gate.py`.

## 4. Cells, sample size, and seeds

| # | scenario | n | p | indices | reps |
|---|---|---:|---:|---|---:|
| 1–3 | `skew_struct`, `tail_struct`, `shape_mixed` | 2000 | 5 | **14000–14049** | 50 |
| 4–6 | `skew_struct`, `tail_struct`, `shape_mixed` | 20 000 | 20 | **14100–14119** | 20 |

Six cells: 3 × 50 + 3 × 20 = **210 tasks**. All six are inside the gate; none is
optional. The same index range is reused across the three scenarios of one size;
the data stream differs because the scenario name enters `seed_of`, exactly as in
V5.1.

**Sample size.** 50 and 20 replications are those of every V5.x gate (PREREG_V5 §3,
inherited by V5.1), so the result is directly comparable to the V5.1 table. At these
sizes the V5.1 gate resolved paired differences an order of magnitude smaller than
the δ values above (upper bounds around −0.002 to −0.012). No power claim beyond
that is made; no replication is added if a cell is close to a bar.

### Seed inventory (every spent or reserved range) and the fresh range

Searched on `main` @ `ad2b791`: every pre-registration and results doc under
`docs/benchmark/`, `docs/research/`, `docs/theory/`, `docs/TRACE_SIMS.md`, every
runner under `benchmarks/`, and the `rep` field of every
`benchmarks/results/*.jsonl` (largest `rep` in the repo: 13119). This is the
inventory of [PREREG_V5_2C_SPECIFIED_REPLICATION.md](PREREG_V5_2C_SPECIFIED_REPLICATION.md)
§1, updated for the V5.2c pass.

| range | used by | record |
|---|---|---|
| 0–199 | PR #12 / TRACE_CLAIMS / ShiftBench (own streams) | `run_synthetic.py`, `run_trace_claims.py` |
| 0–1999 | toy-null calibration smoke loop (raw seed 0, 2000 draws) | `sb_h8.py` |
| 1000–1019 | shift v2 / FIXES2 baseline | `docs/theory/PREREG_SHIFT_v2.md`, `FIXES2_PLAN.md` |
| 1000–1199 | trace_sims v1 period-2 draws | `benchmarks/trace_sims/run_v1.py` |
| 2101–2119, 3101–3120, 6101–6119 (+ raw 6, 11) | trace_sims v2 / tail v2 | `docs/theory/PREREG_TAIL_v2.md` |
| 5000–6499 (incl. 5000–5019, 5200–5599, 6000–6019, 6100–6499), 7007–7012 | FIXES2 | `FIXES2_PLAN.md`, PREREG_TRACE_V3 §3 |
| 7900–7909 | v3 timing pilot | `PREREG_TRACE_V3.md` |
| 8000–8499 | v3 (shape 8000–8499, coverage 8000–8299) | `PREREG_TRACE_V3.md` |
| 9100–9149, 9200–9219, 9300–9379, 9400–9429 | v4 (shape, coverage, ShiftBench, speed 9400–9402) | `PREREG_V4.md`, `run_v4.py` |
| 9500–9549 | v5 first gate, then v5 diagnosis (spent) | `PREREG_V5.md`, `V5_DIAGNOSIS.md` |
| **9600–9619** | **sealed holdout; never run, never touched** | `PREREG_V5.md`, `PREREG_V5_1.md` |
| 9700–9749, 9800–9819 | v5.1 gate (spent) | `PREREG_V5_1.md` |
| 10100–10149, 10200–10219 | v5.2 gate (spent) | `PREREG_V5_2.md` |
| 10220–10399 | buffer after v5.2; left unused | `PREREG_V5_1_COVERAGE.md` |
| **10400–10799** | V5.1 coverage study (PR #46); **not touched by this file** | `PREREG_V5_1_COVERAGE.md` |
| 10800–10999 | remainder of the coverage block; left unused | `PREREG_V5_1_COVERAGE.md` |
| 11000–11049, 11100–11119 | V5.2 specified gate (spent) | `PREREG_V5_2_SPECIFIED.md` |
| 11120–11999 | buffer; left unused | `PREREG_V5_2B_SPECIFIED_REPLICATION.md` |
| 12000–12049, 12100–12119 | V5.2b (spent, aborted) | `V5_2B_ABORT.md` |
| 12120–12999 | buffer; left unused | `PREREG_V5_2C_SPECIFIED_REPLICATION.md` |
| 13000–13049, 13100–13119 | V5.2c (spent, FAIL) | `PARK_V5_2C_SPECIFIED.md` |
| 13050–13099 | gap inside V5.2c; forbidden there; left unused | `PREREG_V5_2C_SPECIFIED_REPLICATION.md` |
| 13120–13999 | buffer after V5.2c; left unused | this file |
| 81001–81020, 83001–83020, 84001–84050 | trace_sims v2 items (raw seeds) | `docs/TRACE_SIMS.md` |
| fixed constants | oracle seed 99; bootstrap seed 2026; pre-flight smoke seed 99; rehearsal seeds 97, 98 | `sb_dgp_shape.py`, scorers, V5.2c §3 |

**Fresh range: 14000–14049 and 14100–14119.** At this draft, on `main` @ `ad2b791`:

```
git grep -nE '(^|[^0-9.])14(0[0-4][0-9]|1[01][0-9])([^0-9]|$)'
```

returns only `"n_train": 14000` in `trace_diagnosis.json` and `trace_ebm.json`
(a training-set size, not a replication index), and no `rep` field equals any index
in 13120–14999. The gap 14050–14099 is left unused. The runner must assert every
gate index lies in 14000–14049 or 14100–14119 and must carry a `FORBIDDEN` set
covering every row of the table above, naming **9600–9619** and **10400–10799**
explicitly.

## 5. Bars and scoring (inherited; two checks adapted to these cells)

Per cell, paired \(d_r=\mathrm{err}_{\mathrm{v5.1},r}-\mathrm{err}_{\mathrm{best\,rival},r}\)
(best rival = min of DFL, CFM per replication). One-sided 95% upper bound of mean
\(d\) by Student-t with n−1 df **and** a 10 000-draw percentile bootstrap (seed
2026); the larger bound is used.

* **Primary metric:** mean paired \(d\) on the graded channel, per cell.
* **Comparator:** best paired rival per replication (min of DFL, CFM).
* **Margin:** δ = 0.10 × planted magnitude of the graded channel (table in §3).
* **BEST:** upper bound < 0. **NON-INFERIOR:** upper bound < δ.
* **"After Holm"** (convention of PREREG_V5_2_SPECIFIED §5): the CI verdict **and**
  the matching Holm-adjusted one-sided t p-value (p BEST for BEST, p NI for
  NON-INFERIOR) is < 0.05. Both are required. Holm across the **six** accuracy tests.
* **Role check, structure-only cells (1, 2, 4, 5)** — mirrored V5.1 rule, as
  PREREG_V5_2 §5: mean \(|\hat s|>\) mean \(|\hat c|\) on the graded channel, and
  swapping the two roles does not lower the error.
* **Role check, mixed cells (3, 6)** — both roles are planted with opposite signs,
  so dominance is not the test. Pass if sign(mean \(\hat c\)) = sign(truth comp)
  (+), sign(mean \(\hat s\)) = sign(truth struct) (−), and swapping the two roles
  does not lower the error.
* **False-attribution check, structure-only cells (1, 2, 4, 5)** — as PREREG_V5_2 §5:
  one-sided 95% upper bound (t and bootstrap, larger used) of mean \(|\hat c_r|\) on
  the graded channel < δ, with the Holm-adjusted one-sided t p-value < 0.05, Holm
  across these four tests.
* **Gate PASS:** all six cells BEST or NON-INFERIOR after Holm, all six role checks
  pass, all four false-attribution checks pass, and the pass finishes within the cap.
  Anything else is **FAIL**, except ABORTED (§6), which leaves H undecided.

## 6. Harness, cap, and kill rule

* **Harness:** the spawn-safe harness, mandatory pre-flight (in a spawned child),
  mandatory rehearsal on throwaway seeds 97 and 98 (never scored), and the 5-minute
  no-progress watchdog of
  [PREREG_V5_2C_SPECIFIED_REPLICATION.md](PREREG_V5_2C_SPECIFIED_REPLICATION.md)
  §2–4, incorporated by reference, reusing `benchmarks/shiftbench/spawn_safe.py`.
* **Cap:** single pass over all 210 tasks (n = 20 000 cells first) on 8 workers,
  wall cap **30 minutes**. No task dropped, retried, or re-run; unfinished cells are
  INCOMPLETE and never imputed.
* **Single pass.** No interim look, no early stop on a passing prefix, no extension
  of the seed range.
* **ABORTED.** A pre-flight or rehearsal failure stops before any gate seed is
  touched; H is undecided and 14000–14119 stay unspent. A watchdog trip after the
  gate starts is recorded as ABORTED (no verdict); the gate seeds are then spent and
  are not reused.
* **Kill rule.** If **any** cell fails accuracy, fails its role check, fails the
  false-attribution check, or is INCOMPLETE, **H is rejected**. Then: record every
  failure in `docs/benchmark/V5_1_STRUCT_GATE_RESULTS.md`, write
  `docs/research/PARK_V5_1_STRUCT.md`, and restate V5.1's claim in every public
  place (README, paper, learn pages) as **composition-only cells**, with this
  failure named. No amendment on these cells, no seed changes, no re-runs to chase
  a bar, no new location model under this file. Any later attempt needs a new
  pre-registration on fresh seeds.

## 7. Anti-job and limitations (stated in advance)

* No curved mean, no heteroskedastic scenario, no real data, no coverage or bands,
  no ShiftBench channels, no per-feature shares are tested or claimed.
* The simulator's mean is linear, so OLS is the right family by construction
  (PREREG_V5_1 §7). A PASS says the V5.1 split is right on structure-only and mixed
  shape shifts **under a correct location model**; it says nothing about
  non-linear means (V5.2 and V5.2c remain parked).
* A PASS does not unseal 9600–9619 and does not touch the V5.1 coverage study.

## 8. Report (either way)

`docs/benchmark/V5_1_STRUCT_GATE_RESULTS.md` with this file's freeze SHA, the
implementation SHA, the runner SHA, the pre-flight and rehearsal records, and the
six-cell table: err v5.1, err DFL, err CFM, mean d, upper t, upper boot, δ, verdict,
mean \(\hat c\), mean \(\hat s\), \(|\hat c|\), \(|\hat s|\), role, false-attribution
bound (cells 1, 2, 4, 5), p NI / p BEST unadjusted and Holm; wall time; every
failure kept; the gate verdict (PASS, FAIL, or ABORTED). No result numbers appear in
this file.

**Secondary, descriptive, not scored** (cannot rescue or sink H):

1. The open `tail_comp` composition under-estimate (PREREG_V5_2B §3): for every cell,
   mean \(\hat c\) and \(\hat s\) against the planted truth, and their relation to δ.
   On `tail_struct` this shows whether the λ₄ composition piece is biased when its
   truth is ≈ 0.
2. Median per-task wall seconds of TRACE, DFL, and CFM per cell (no speed bar).

On PASS the README, paper draft, and learn pages may state the widened claim in
§7's words. On FAIL they state the narrowed claim of §6. Either way a learn page is
published from the results file.

## 9. Order of work (nothing on gate seeds before step 3)

1. Runner PR (runs nothing on gate seeds): `run_v5_1_struct_gate.py` and its
   summariser, reusing `run_v5_1_gate.py`, `summarize_v5_1_gate.py`, and
   `spawn_safe.py`; the `FORBIDDEN` set of §4; the method check of §2 (seed 99
   only); CI green.
2. Pre-flight, then rehearsal (seeds 97, 98), per §6.
3. One single pass on 14000–14049 and 14100–14119, then the report of §8.

**Planned cost.** 210 tasks; the 140-task V5.1 gate took 15.0 s and the 140-task
V5.2c pass 12.7 s on 8 workers, so about 20–40 s of wall time is expected (cap
30 minutes), plus about 3 s of rehearsal. Machine cost is negligible; the work is
the runner PR and the write-up.
