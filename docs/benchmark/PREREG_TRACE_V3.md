# Pre-registration: TRACE v3 (H7 and H8(b) with the PIT refit on; H5 restated)

Date: 2026-10-14. Builds on [PREREG_TRACE_CLAIMS.md](PREREG_TRACE_CLAIMS.md) and
[PREREG_TRACE_CLAIMS_AMENDMENT_1.md](PREREG_TRACE_CLAIMS_AMENDMENT_1.md) (bars, scenarios, truth and
estimands unchanged unless stated here). The baseline is the PR #12 result
([TRACE_CLAIMS_RESULTS.md](TRACE_CLAIMS_RESULTS.md), merge `3963374`), where H7 and H8(b) failed.

**State when this file was frozen.** No v3 replication has been run. A timing pilot (§4) was run
on throwaway replications 7900–7909; only wall-clock times were read, and its records were deleted
without being opened for accuracy. The PR #12 results are public and were seen; this pre-registration
is therefore a *second attempt* after a failure, and is labelled as such.

## 1. What changes from PR #12 (and only this)

1. **PIT pseudo-outcome refit on.** `traceot.shift_decompose(..., pit_rounds=20)` in both period fits
   and in every bootstrap refit. `20` is the value used in the FIXES2 validation (PR #11). To make this
   possible, this branch adds a `pit_rounds` argument (default `0`) to `shift_decompose` /
   `DecomposeConfig`, passed to the per-period quantile fit. With the default `0` the library
   behaves exactly as before; no other code path changes.
2. **Everything else as in PR #12 (Amendment 1, D5):** `K=4`, `folds=5`, `k_fit=4`, `n_quad=64`,
   one thread per worker. **Trim: `0.99`, which is the library default of `shift_decompose`** (and the
   PR #12 value), so no justification for a departure is needed.
3. **Fresh seeds.** Replication indices start at **8000** (`rep = 8000, 8001, …`); data, method and
   bootstrap seeds are the frozen `seed_of(...)` streams of PR #12 evaluated at these indices
   (`seed_of("data", scen, n, p, rep)`, `seed_of("m", rep)`, `seed_of("boot", scen, rep)`,
   `seed_of("mb", rep, b)`). Indices ≥ 8000 were never used by any earlier run (PR #12 used 0–199,
   FIXES2 used 5000–6499, the pilot 7900–7909).
4. **More replications** (§4), sized by the pilot. The bars are rates and do not change.

Code: `benchmarks/shiftbench/run_trace_v3.py` (runner), `benchmarks/shiftbench/summarize_trace_v3.py`
(scorer, written before any v3 record exists, committed with or before the results).

## 2. Hypotheses and bars

### H7 (shape shifts at fixed mean and L-scale): unchanged from Amendment 1 §A2 and D12
Five scenarios `skew_struct, tail_struct, skew_comp, tail_comp, shape_mixed`; oracle truth
`benchmarks/results/shiftbench_truth_shape.json` (unchanged). TRACE scored on `lsplit` against the
Shapley truth. Recovery per replication: `r_k ≤ 0.25` on the graded channel, every planted piece
with the right sign, `|comp̂_0| + |struct̂_0| ≤ 0.05·sd(y_A)`.

* **H7-recovery bar:** rate ≥ 0.80 at (20,000, 20) and ≥ 0.60 at (2,000, 5) in each of the 5 scenarios
  (10 cells, all must pass).
* **H7-rival bar:** supported iff TRACE beats the best rival in ≥ 4 of 5 scenarios at (20,000, 20).
  **Rival records are reused, not re-run:** the DFL and CFM shape read-outs from PR #12
  (`benchmarks/results/trace_claims_shape.jsonl`, replications 0–19 at (20,000, 20)). The best rival
  per scenario is the one fixed by PR #12 (lower mean error): DFL for `skew_struct, tail_struct`; CFM
  for `skew_comp, tail_comp, shape_mixed`. Because the rival data seeds differ from the v3 seeds, the
  paired test of Amendment 1 is impossible; it is replaced by an **unpaired** comparison:
  `d = mean(err_TRACE,v3) − mean(err_rival,PR12)`, Welch 95% interval and a 10,000-draw two-sample
  percentile bootstrap (seed 2026). "Beats" iff **both** upper ends are < 0. This deviation is forced
  by the reuse instruction and is stated before any result.
* **Holm** over the 5 Welch p-values (two-sided), reported as in PR #12. H6 and H8(a) are not part of v3.

### H8(b) (band coverage): unchanged bars
Coverage track design of PR #12 (n = 2,000, p = 5, B = 30 bootstrap refits, 95% percentile bands),
scenarios `mixed, tail, hetero, entry_exit` (A-reference `split` vs frozen truth, five stats × two
roles) and the five shape scenarios (`lsplit` vs Shapley truth, channels l1–l4 × two roles). `null`
is run and its joint-test rejection rate reported, not graded (as in PR #12).
* **Bar:** pooled coverage ≥ 0.90 **and** every scenario × stat/channel cell ≥ 0.85.

### H5 restated (correction made after seeing PR #12)
PR #12 graded two things under H5: the barycenter reconstruction (H5.1b, passed, worst 3.6e-15) and
the equality of TRACE's marginal W2² pieces with the **empirical** quantile W2² (H5.2 / `g52`, failed in
all 1,175 replications, up to 3% relative). The second comparison is not an identity of the method:
TRACE's marginal pieces are computed on its own fitted laws and quadrature, not on the raw sample
quantile function, so it should never have been a bar. **This restatement is made after seeing the
PR #12 failure and is therefore not a claim and not evidence.** The PR #12 H5 FAIL stays on record.
* **H5.1b (bar):** `|W_rec − Σφ| ≤ 1e-8 · max(1, W_rec)` in every v3 shape replication, with `W_rec`
  the 256-node Gauss–Legendre integral of the averaged fitted quantile functions (`predict_quantile`).
* **Marginal vs empirical (descriptive only):** `|Σ marg_phi + resid_phi − mean((sort y_B − sort y_A)²)|`
  absolute and relative to the empirical value: median and max reported, no bar.

## 3. Failure and stop rules
* Every miss is recorded with scenario, size, replication and value, and stays on record.
* **Hard wall-clock cap 2.5 h** for the whole run (both tracks), enforced by `timeout`. Records are
  appended per task (checkpoint). Tasks are ordered replication-major so a stop leaves balanced cells.
* If the cap hits: a bar is PASS/FAIL only if the completed replications already decide it whatever
  the missing ones would be (for a rate bar with n planned and k hits in m done: PASS iff
  `k ≥ bar·n`, FAIL iff `k + (n − m) < bar·n`); otherwise **INCOMPLETE**. The H7-rival test is
  computed on whatever v3 replications exist and labelled with its n.
* A crash in a task is a recorded failure of that replication for recovery and a missing cell for coverage.

## 4. Sizing (timing pilot, replications 7900–7909) and power
Pilot on this box, 8 workers, `pit_rounds=20` (times only):
* shape (2,000, 5): ~0.1–0.2 s per fit; (20,000, 20): 3.7–5.5 s per fit (+ the H5.1b quadrature check);
* coverage: 3.2–4.0 s per replication (31 fits at n = 2,000).

Planned counts (total ≈ 1,000×6 + 2,500×0.5 + 3,000×3.6 ≈ 18,000 core-seconds ≈ 40 min on 8 cores,
leaving room for other load on the box; target ≤ 2 h, cap 2.5 h):

| track | cell | replications (indices) | PR #12 |
|---|---|---|---|
| H7 shape | (20,000, 20) per scenario | 200 (8000–8199) | 20 |
| H7 shape | (2,000, 5) per scenario | 500 (8000–8499) | 50 |
| H8(b) coverage | per scenario (10) | 300 (8000–8299) | 40 |

**Power (exact two-sided 95% Clopper–Pearson at the bar):**
* Recovery at n = 200 with an observed rate of exactly 0.80: CI [0.738, 0.853]. A true rate of 0.80 passes
  about half the time; the bar is a point-estimate bar as frozen.
* Recovery at n = 500 with observed 0.60: CI [0.556, 0.643].
* Coverage cell: 300 replications × 2 roles = 600 indicators; at 0.85 the CI is about [0.819, 0.878],
  at 0.90 [0.873, 0.923] (treating indicators as independent; the two roles within a replication are
  correlated, so the true interval is somewhat wider). Pooled coverage (≈ 24,000 indicators) is pinned
  to about ±0.004.
* H7-rival: the rival side has only 20 replications (reused), so the Welch interval is dominated by
  the rival's spread; differences of the size seen in PR #12 for the composition scenarios (≥ 0.005
  absolute) are detectable, sub-0.001 differences on the `*_struct` scenarios are not.

## 5. Report
`docs/benchmark/TRACE_V3_RESULTS.md`: every bar with its value, k/n, CP interval, verdict, and the
PR #12 value side by side; the H5.1b check; the descriptive marginal-vs-empirical numbers; Holm; and
any INCOMPLETE cells with the reason.
