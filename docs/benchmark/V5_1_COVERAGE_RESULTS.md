# TRACE v5.1 coverage — results

**Prereg SHA:** `dfaaf72e775ad6065b88a1c7931033c8dedbdbd3` ([PREREG_V5_1_COVERAGE.md](../research/PREREG_V5_1_COVERAGE.md)).
**Approval:** 2026-11-13, 2:59 PM, America/New_York. The owner said, quoted verbatim: *"You have my approval for all trace stuff - you are running the lab on this - I’m more an advisor on that project."* Approval commit: [PR #56](https://github.com/jgridifier/traceot/pull/56), merge `05375628e44b1b203c0fb6c92a1c20d37e067742`.
**Runner merge SHA:** `14754e6120e34208bbcdaa26d7cf3173659e10c0` ([PR #57](https://github.com/jgridifier/traceot/pull/57)).
**Gate verdict:** **PASS**.
**Wall time:** 349.886 s on 8 workers (cap 3600 s); 800/800 tasks finished.
**Run window:** 2026-11-13, 3:16:34 PM to 3:22:24 PM, America/New_York.

Hypothesis H (PREREG_V5_1_COVERAGE §1): on the two linear-mean pure-composition
cells at n = 2000, p = 5, the nominal 90% two-sided percentile bootstrap band
(`BAND_V5_1_PB90`, B = 500) around the v5.1 composition piece and around the v5.1
response piece of the graded L-moment covers the frozen Shapley truth at a rate
inside the binomial pass interval for each of the four (cell × piece) units.

Pass interval (fixed in §5): 344 ≤ k ≤ 374 of 400
(0.860 ≤ π̂ ≤ 0.935). Unit FAIL (under-covers) if k ≤ 343; unit FAIL
(over-covers) if k ≥ 375. Gate PASS only if all four units PASS. Coverage is a
rate bar; no Holm step. Missing replications are never imputed.

| unit | n done | k covers | π̂ | pass interval | mean width | miss below | miss above | forced-miss (refit) | verdict |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| skew_comp comp | 400 | 365 | 0.9125 | [344, 374] | 0.02795 | 14 | 21 | 0 | **PASS** |
| skew_comp response | 400 | 361 | 0.9025 | [344, 374] | 0.009303 | 17 | 22 | 0 | **PASS** |
| tail_comp comp | 400 | 356 | 0.8900 | [344, 374] | 0.0154 | 15 | 29 | 0 | **PASS** |
| tail_comp response | 400 | 360 | 0.9000 | [344, 374] | 0.007292 | 23 | 17 | 0 | **PASS** |

**Pooled coverage** (report only, no bar): 1442/1600 (0.90125).
**Failed bootstrap refits** (report only): 0 across 800 completed replications.

Full numbers: `benchmarks/results/v5_1_coverage_summary.json`; per-replication
records: `benchmarks/results/v5_1_coverage.jsonl`; wall:
`benchmarks/results/v5_1_coverage_wall.json`. A plain-language page is at
[docs/learn/v5_1_coverage.html](../learn/v5_1_coverage.html).

Independently checked after the official summariser: every `rep` lies in
10400–10799; none lies in 9600–9619 or any other FORBIDDEN range; graded index
k = 2 on `skew_comp` and k = 3 on `tail_comp`; `n_success` = 500 on every
replication; the stored `covered` flag matches `lo ≤ truth ≤ hi` for both
pieces.

## Notes and deviations

1. **Band.** `BAND_V5_1_PB90` as frozen: two-sample percentile bootstrap, rows
   resampled within period, B = 500, stream `seed_of("v51cov", s, n, p, r, b)`,
   `[Q_0.05, Q_0.95]` via `np.quantile(..., method="linear")`, no bias correction,
   no studentising, no BCa. Covered if `lo ≤ truth ≤ hi`. More than 5 of 500
   failed refits makes that replication a miss for both pieces; failed refits
   were never re-drawn.
2. **Point estimator.** `TRACE_V5_1` = `explain_v5_1` with library defaults, on
   arrays centred exactly as the V5.1 gate runner (`centred` of `run_synthetic.py`).
   Bootstrap refits use the already-centred arrays, with no re-centring.
3. **Seeds.** Every index was asserted to lie in 10400–10799 and outside the
   FORBIDDEN set, including the sealed range 9600–9619, which was not touched.
4. **Single pass.** The results file was required not to exist at the start of
   the run. No second pass, no change of B, level, band method, cells, or seeds.
5. **Scope.** A PASS (or a FAIL) is about the band only. The V5.1 accuracy claim
   in V5_1_GATE_RESULTS.md is unaffected. Nothing is claimed about n = 20 000,
   struct cells, `shape_mixed`, ShiftBench, other levels, or non-linear means.
   No library band API is added here (the pre-registration leaves that as a
   separate later change that must reproduce `BAND_V5_1_PB90`).
6. **Executable (operational, not a design change).** The run used a Python 3.13
   wheel built from `4cf9754ff4deeb7ed1da9d2a0267529563ed87b3` (the then-unmerged
   specified-implementation branch, later merged to `main` as
   [PR #58](https://github.com/jgridifier/traceot/pull/58)). At the moment of the
   run, `main` was at the coverage-runner merge
   `14754e6120e34208bbcdaa26d7cf3173659e10c0`. The difference in `explain.rs`
   between those two commits is only additive: the `LocationModel::Specified`
   variant, its match arm, and a docstring change on `dropped_cols`. `fit_ols`,
   `predict_ols`, and the `Ols` match arm are unchanged. The coverage runner
   called `explain_v5_1` with library defaults (`mean_model="ols"`). This is
   recorded as an operational fact. It is not a reason to re-run.
7. **Monitoring.** A log was inspected while the run was in progress (about 51
   of 800 tasks then written, coverage near the eventual rate). No parameter was
   changed, the run was not stopped, and there was no second pass. The
   pre-registration forbids an interim look that changes the design; this was
   operational monitoring only.
8. **No park file.** All four units PASS, so `PARK_V5_1_COVERAGE.md` is not written.
