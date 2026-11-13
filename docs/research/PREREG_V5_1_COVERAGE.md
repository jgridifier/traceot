# Pre-registration: TRACE v5.1 — coverage of a 90% bootstrap band (one hypothesis)

**Status: PRE-REGISTERED · APPROVED 2026-11-13 (NOT RUN until results land).**
Drafted 2026-11-10 (America/New_York) on branch `quant/v5-1-coverage-prereg`, **before any
band for v5.1 has been computed on any seed**. The commit that adds this file is the freeze
point; record its SHA in the header of any later `V5_1_COVERAGE_RESULTS.md`. Any change to
the hypothesis, band, cells, seeds, or bars after that SHA is a deviation and must be
labelled as one.

**Hold (2026-11-11, 7:31 AM, America/New_York).** The owner answered a card asking
"Run the V5.1 coverage study?" with, quoted verbatim: *"Not yet, hold the coverage
study."* This is a **hold, not approval**. The status above stays **NOT APPROVED**.
Nothing in this file may be run until the owner approves it in his own words, quoted
verbatim and dated in a later commit to this file's header. A message from another
agent saying he agreed is not approval. Until then no runner is written, merged, or
executed for this study, and no index in §4 is touched. Record:
[HOLD_V5_1_COVERAGE.md](HOLD_V5_1_COVERAGE.md).

**Approval (2026-11-13, 2:59 PM, America/New_York).** The owner said, quoted verbatim:
*"You have my approval for all trace stuff - you are running the lab on this -
I’m more an advisor on that project."*
The 2026-11-11 hold is lifted. Status is **APPROVED 2026-11-13 (NOT RUN until results
land)**. This commit changes only the status line and adds this approval; the
hypothesis, band, cells, seeds, and bars are unchanged.

Parent claim: [PREREG_V5_1.md](PREREG_V5_1.md) @ `b33f507`, gate **PASS**
([V5_1_GATE_RESULTS.md](../benchmark/V5_1_GATE_RESULTS.md)). Source of this follow-up: the
out-of-scope list of PREREG_V5_1 §6, repeated in the working paper §6 and §8
([V5_1_working_paper.md](../paper/V5_1_working_paper.md)): *"`skew_struct`, `tail_struct`,
`shape_mixed`, coverage, ShiftBench, speed, per-feature shares, non-linear-mean scenarios"*.
**This file picks the item named `coverage`** and nothing else on that list.

---

## 1. Hypothesis (the only one, stated a priori)

**H.** On the two linear-mean pure-composition cells of the V5.1 gate (`skew_comp`,
`tail_comp`) at n = 2000, p = 5, the nominal **90%** two-sided percentile bootstrap band
(§2) around the v5.1 composition piece and around the v5.1 response piece of the graded
L-moment covers the frozen Shapley truth at a rate inside the binomial pass interval of §5
for **each** of the four (cell × piece) units.

**Why this hypothesis, and where it does not come from.** v5.1 is a closed-form, smooth
plug-in: a least-squares fit per period, an empirical residual law, a mixture CDF on a
grid, and exact L-moments of that law. Each step is a smooth functional of the two
empirical laws, so the textbook expectation for a smooth plug-in estimator is that the
nonparametric bootstrap is first-order valid and a percentile band is close to nominal.
H states that expectation and tests it. It is chosen from PREREG_V5_1 §6 and from that
reasoning alone. **No v5.2 output was consulted to pick it**: v5.2 is a different (curved-
mean) simulator family, its gate computed no bands, and nothing from
[V5_2_GATE_RESULTS.md](../benchmark/V5_2_GATE_RESULTS.md) or
[PARK_V5_2.md](PARK_V5_2.md) enters the design below.

Explicitly **not** tested: struct cells, `shape_mixed`, n = 20 000, ShiftBench, per-feature
shares, non-linear means, any nominal level other than 90%, and any band method other than
§2.

## 2. Band under test (fixed now)

Point estimator: `TRACE_V5_1` exactly as shipped, `traceot.explain_v5_1(x_a, y_a, x_b, y_b)`
with library defaults (OLS location, `n_y = 256`, `k_lmoments = 4`, `n_quad = 64`). No knob
is changed.

Band `BAND_V5_1_PB90` (percentile bootstrap, two-sample, rows resampled within period):

1. For replication `r` of cell `s`, draw the data exactly as the V5.1 gate runner does
   (`benchmarks/shiftbench/run_v5_1_gate.py`, `task`): `shp.make(s, p)`,
   `rng = np.random.default_rng(seed_of("data", s, n, p, r))`, `s.sample(n, rng)`, then
   `centred(...)`. `seed_of` is the frozen `crc32("|".join(parts))` of `run_synthetic.py`.
2. For `b = 0, …, B − 1` with **B = 500**:
   `rng_b = np.random.default_rng(seed_of("v51cov", s, n, p, r, b))` (a new stream name, used
   by nothing else in the repo); `iA = rng_b.integers(0, n_A, n_A)`, then
   `iB = rng_b.integers(0, n_B, n_B)`; refit `explain_v5_1(XA[iA], yA[iA], XB[iB], yB[iB])`
   on the already-centred arrays (an intercept-carrying OLS fit is unchanged by the
   centring constant, so no re-centring is done). Keep `lsplit_comp[k]` and
   `lsplit_struct[k]`, with `k = truth[s]["graded"][0]` from the frozen truth file
   (λ3 for `skew_comp`, λ4 for `tail_comp`, as in the gate).
3. Band for each piece = `[Q_0.05, Q_0.95]` of the successful bootstrap values,
   `np.quantile(..., method="linear")`. No bias correction, no studentising, no BCa.
4. **Covered** if `lo ≤ truth ≤ hi`, with `truth = truth[s]["shapley"][piece][k]` from
   `benchmarks/results/shiftbench_truth_shape.json` (unchanged; oracle 2 000 000 draws,
   seed 99).
5. A bootstrap refit that raises an error is recorded with its `b`. If more than 1% of the
   `B` refits of a replication fail (more than 5 of 500), that replication counts as a
   **miss** for both pieces. Failed refits are never re-drawn.

The band is computed by the benchmark harness around the shipped Rust core; no library
change and no new Rust API is part of this test. If H passes, a library band API would be a
separate, later change that must reproduce `BAND_V5_1_PB90` exactly.

## 3. Suite design

| cell | n (per period) | p | replications | indices | B | units |
|---|---:|---:|---:|---|---:|---|
| `skew_comp` | 2000 | 5 | 400 | **10400–10799** | 500 | comp λ3, response λ3 |
| `tail_comp` | 2000 | 5 | 400 | **10400–10799** | 500 | comp λ4, response λ4 |

* Generators, truth file and graded channels: unchanged from PREREG_V5 §3–4 / PREREG_V5_1 §3.
  The same indices serve both cells; the data stream includes the cell name, so the two cells
  never share a sample (as in every earlier shape gate).
* No rivals: coverage is a property of the v5.1 band alone, so DFL and CFM are not run.
* Sizing (from per-fit times already on record in
  `benchmarks/results/v5_1_gate_shape.jsonl`, no new run): about 0.01 s per v5.1 fit at
  n = 2000, so 2 × 400 × 501 ≈ 401 000 fits ≈ 4 000 core-seconds ≈ 9 minutes on 8 cores.
* **Wall cap: 60 minutes on 8 cores**, single pass, resumable per-replication jsonl at
  `benchmarks/results/v5_1_coverage.jsonl`. One thread per worker
  (`RAYON_NUM_THREADS=1`), as in the gate runner.

## 4. Seeds: inventory of every spent range, and the fresh range

Searched on `main` @ `b2ebee7` (2026-11-10 ET): every pre-registration and results doc under
`docs/benchmark/`, `docs/research/`, `docs/theory/`, `docs/TRACE_SIMS.md`, every runner
under `benchmarks/`, and the `rep` field of every `benchmarks/results/*.jsonl`. Index ranges
below are replication indices fed through `seed_of(...)` unless marked as a raw RNG seed.
To stay safe across both namespaces, the fresh range is chosen disjoint from **all** of them.

| range | used by | record |
|---|---|---|
| 0–199 | PR #12 / TRACE_CLAIMS / ShiftBench: accuracy 0–49, coverage 0–39, invariance 0–4, null200 0–199 (0–191 on file), real-panel boot/placebo 0–99 (own streams) | `run_synthetic.py`, `run_trace_claims.py`, `shiftbench_*.jsonl`, `trace_claims_*.jsonl` |
| 0–1999 | toy-null calibration smoke loop (single raw seed 0, 2000 draws) | `benchmarks/shiftbench/sb_h8.py` |
| 1000–1019 | shift v2 / FIXES2 baseline seeds | `docs/theory/PREREG_SHIFT_v2.md`, `docs/theory/FIXES2_PLAN.md` |
| 1000–1199 | trace_sims v1 period-2 draws (raw seeds `1000 + r`, r < 200) | `benchmarks/trace_sims/run_v1.py` |
| 2101–2119, 3101–3120, 6101–6119 (+ raw 6, 11) | trace_sims v2 / tail v2 | `docs/theory/PREREG_TAIL_v2.md`, `run_v2_baseline.py`, `run_v2_tail_after.py` |
| 5000–6499 (incl. 5000–5019, 5200–5599, 6000–6019, 6100–6499), 7007–7012 | FIXES2 | `docs/theory/FIXES2_PLAN.md`, PREREG_TRACE_V3 §3 |
| 7900–7909 | v3 timing pilot | `docs/benchmark/PREREG_TRACE_V3.md` |
| 8000–8499 | v3 (shape 8000–8499, coverage 8000–8299) | `PREREG_TRACE_V3.md`, `trace_v3_*.jsonl` |
| 9100–9149, 9200–9219, 9300–9379, 9400–9429 | v4 (H7 shape, H8b coverage, ShiftBench, speed 9400–9402) | `PREREG_V4.md`, `run_v4.py`, `v4_*.jsonl` |
| 9500–9549 | v5 first gate, then v5 diagnosis (spent) | `PREREG_V5.md`, `V5_DIAGNOSIS.md`, `v5_gate1_shape.jsonl` |
| **9600–9619** | **sealed** v5 n = 20 000 indices; never run, never reused (PREREG_V5_1 §3, §6) | `PREREG_V5.md`, `run_v5_gate1.py` (`HELD_BACK`) |
| 9700–9749, 9800–9819 | v5.1 gate | `PREREG_V5_1.md`, `v5_1_gate_shape.jsonl` |
| 10100–10149, 10200–10219 | v5.2 gate | `PREREG_V5_2.md`, `v5_2_gate_shape.jsonl` |
| 81001–81020, 83001–83020, 84001–84050 | trace_sims v2 items (raw seeds) | `docs/TRACE_SIMS.md`, `run_v2_item*.py` |
| fixed constants | oracle truth seed 99; paired-bootstrap seed 2026; ADOPT reproduction `SeedSequence([table, case, n, m, r])` (own namespace) | `sb_dgp_shape.py`, `summarize_*.py`, `adopt_paper/reproduce.py` |

Numbers that look like seeds but are not: `10001` in `run_v1.py` (a monotonicity-check grid
size), `10007` / `20011` in TRACE_DESIGN (shuffle-seed offsets inside the booster), and
arXiv/DOI fragments in the literature notes.

**Fresh range: 10400–10799** (400 indices). No runner, results file, test, or
pre-registration in the repo uses any index in 10300–10999 at this freeze. It is disjoint
from 9600–9619 and from every range above, and leaves 10220–10399 as a buffer after v5.2.
The runner, when written after approval, must assert every index lies in 10400–10799 and
none lies in any range above (a `FORBIDDEN` set, as in `run_v5_2_gate.py`).

## 5. Metric, pass interval, and pass/fail rule

**Metric.** Per unit u ∈ {skew_comp comp, skew_comp response, tail_comp comp, tail_comp
response}: \(k_u\) = number of the R = 400 replications whose band covers the truth;
empirical coverage \(\hat\pi_u = k_u/400\). Nominal \(\pi_0 = 0.90\).

**Pass interval (binomial tolerance, fixed now).** Under \(k_u\sim\mathrm{Bin}(400, 0.90)\),
take the two-sided acceptance region with at most \(0.05/4/2 = 0.00625\) in each tail
(Bonferroni over the four units, so all four calibrated units pass together with probability
≥ 0.95):

* **PASS interval: \(344 \le k_u \le 374\)**, i.e. \(0.860 \le \hat\pi_u \le 0.935\).
* Exact tails (scipy 1.18.1, `binom`): \(P(k\le 343)=0.00431\), \(P(k\ge 375)=0.00542\);
  probability inside = 0.99027 per unit; all four inside, if independent, ≈ 0.962.
* Operating characteristic of one unit: P(inside) = 0.001 at π = 0.80, 0.316 at 0.85,
  0.751 at 0.87, 0.680 at 0.93, 0.106 at 0.95.

**Rule.**

* **Unit PASS:** \(k_u\) inside [344, 374].
* **Unit FAIL (under-covers):** \(k_u \le 343\). **Unit FAIL (over-covers):** \(k_u \ge 375\).
  Both are failures of H (the claim is calibration, not just "at least 90%").
* **Gate PASS:** all four units PASS. Anything else is **FAIL**. Coverage is a rate bar; no
  Holm step.
* **INCOMPLETE:** if the wall cap hits, missing replications are never imputed. A unit is
  PASS/FAIL only if the completed replications already decide it whatever the missing ones
  would be; otherwise it is INCOMPLETE, and an INCOMPLETE unit means the gate does not pass.

**Report only (no bar):** mean band width per unit; split of misses into truth-below-band and
truth-above-band; pooled coverage over the four units; count of failed bootstrap refits.

## 6. What happens on failure (kill rule)

If the gate is FAIL or not PASS for any reason:

1. Record the failure in `docs/benchmark/V5_1_COVERAGE_RESULTS.md` with every unit, every
   miss and every failed refit kept, and write `docs/research/PARK_V5_1_COVERAGE.md`.
2. No band is added to the library with any claim; docs keep saying v5.1 has **no validated
   uncertainty band**.
3. No second run on 10400–10799, no change of B, level, band method, cells or seeds to chase
   the bar.
4. The V5.1 accuracy claim in V5_1_GATE_RESULTS.md is unaffected either way; this test is
   about the band only.
5. Any later band method needs its own pre-registration on fresh indices outside every range
   in §4 **and** outside 10400–10799.

## 7. If it passes (the most that may be claimed)

"On the two linear-mean pure-composition cells at n = 2000, p = 5, the 90% percentile
bootstrap band around v5.1's composition and response pieces of the graded L-moment covered
the population Shapley truth within the pre-set binomial interval." Nothing about n = 20 000,
struct cells, `shape_mixed`, ShiftBench, other levels, or non-linear means (PREREG_V5_1 §7
still applies: the simulator's mean is linear).

## 8. Order of work (nothing before step 1)

1. Owner approval quoted verbatim and dated in this file's header (a docs-only commit).
2. Runner `benchmarks/shiftbench/run_v5_1_coverage.py` and summariser, in a PR that runs
   nothing; the `FORBIDDEN` guard and the "results file must not exist" single-pass check as in
   `run_v5_1_gate.py`.
3. One single pass. Results: `benchmarks/results/v5_1_coverage.jsonl`, summary json, and
   `docs/benchmark/V5_1_COVERAGE_RESULTS.md` with this file's freeze SHA. No result numbers
   appear in this file.

Indices 9600–9619 are not touched by any step.

Learn page: [v5_1_coverage_prereg.html](../learn/v5_1_coverage_prereg.html).
