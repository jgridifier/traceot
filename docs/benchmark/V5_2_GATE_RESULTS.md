# TRACE v5.2 gate — results (curved means)

**Freeze SHA:** `df5aff8` ([PREREG_V5_2.md](../research/PREREG_V5_2.md)).
**Implementation merge SHA:** `ff0a8c659f5e7a3b68b424d354724635ef0c7c83` (PR #34).
**Gate verdict:** **FAIL** — H is rejected; the curved-mean claim is parked
([PARK_V5_2.md](../research/PARK_V5_2.md)). V5.1's linear-mean claim stands exactly as
stated in [V5_1_GATE_RESULTS.md](V5_1_GATE_RESULTS.md).
**Wall time:** 61.7 s on 8 workers (cap 1800 s); 490/490 runnable tasks finished (560 pre-registered; cells [3, 7] had no runnable tasks, see note 1).

Single pass on indices 10100–10149 (n = 2000, p = 5) and 10200–10219 (n = 20 000,
p = 20). 9500–9549, 9600–9619, 9700–9749 and 9800–9819 were not touched.

| # | family | scenario | κ | n | reps | err v5.2 | err DFL | err CFM | mean d | upper t | upper boot | δ | CI | p NI (Holm) | p BEST (Holm) | verdict | \|ĉ\| | \|ŝ\| | role | FA bound (13–16) | FA | err v5.1 (secondary) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|---:|---|---:|---|---:|
| 1 | curved | skew_comp | 0.5 | 2000 | 50 | 0.05065 | 0.2185 | 0.05939 | -0.008743 | -0.005503 | -0.005494 | 0.01225 | BEST | 6.03e-15 (9.64e-14) | 1.94e-05 (0.000271) | **BEST** | 0.08974 | 0.04342 | **PASS** | — | — | 0.06861 |
| 2 | curved | tail_comp | 0.5 | 2000 | 50 | 0.008025 | 0.01278 | 0.01225 | -0.001495 | 0.0001675 | 0.000118 | 0.0003424 | NON-INFERIOR | 0.035 (0.28) | 0.069 (0.621) | **FAIL** | 0.005702 | 0.003004 | **PASS** | — | — | 0.005898 |
| 3 | curved | skew_comp | 1 | 2000 | 0 | — | — | — | — | — | — | — | — | — | — | **INCOMPLETE** | — | — | **INCOMPLETE** | — | — | — |
| 4 | curved | tail_comp | 1 | 2000 | 50 | 0.01149 | 0.01578 | 0.01494 | -0.0004316 | 0.001861 | 0.001735 | 0.0002161 | FAIL | 0.319 (1) | 0.377 (1) | **FAIL** | 0.008286 | 0.003426 | **PASS** | — | — | 0.01008 |
| 5 | curved | skew_comp | 0.5 | 20000 | 20 | 0.0463 | 0.216 | 0.04811 | -0.001801 | -0.0005063 | -0.0006651 | 0.01225 | BEST | 5.07e-14 (7.61e-13) | 0.0133 (0.146) | **NON-INFERIOR** | 0.09121 | 0.0434 | **PASS** | — | — | 0.06991 |
| 6 | curved | tail_comp | 0.5 | 20000 | 20 | 0.002752 | 0.007235 | 0.003754 | -0.0006044 | 0.0002846 | 0.0001863 | 0.0003424 | NON-INFERIOR | 0.0406 (0.284) | 0.127 (1) | **FAIL** | 0.0037 | 0.0008093 | **PASS** | — | — | 0.001942 |
| 7 | curved | skew_comp | 1 | 20000 | 0 | — | — | — | — | — | — | — | — | — | — | **INCOMPLETE** | — | — | **INCOMPLETE** | — | — | — |
| 8 | curved | tail_comp | 1 | 20000 | 20 | 0.003507 | 0.005622 | 0.005211 | -0.001049 | -0.0002369 | -0.0002874 | 0.0002161 | BEST | 0.0072 (0.0648) | 0.0189 (0.189) | **FAIL** | 0.002477 | 0.0008682 | **PASS** | — | — | 0.0029 |
| 9 | control | skew_comp | 0 | 2000 | 50 | 0.0122 | 0.08035 | 0.01895 | -0.006751 | -0.004035 | -0.004095 | 0.004025 | BEST | 1.15e-08 (1.5e-07) | 6.25e-05 (0.000813) | **BEST** | 0.04016 | 0.004624 | **PASS** | — | — | — |
| 10 | control | tail_comp | 0 | 2000 | 50 | 0.005773 | 0.01158 | 0.009249 | -0.002217 | -0.001123 | -0.001168 | 0.0004586 | BEST | 7.68e-05 (0.000845) | 0.000675 (0.0081) | **BEST** | 0.004734 | 0.002387 | **PASS** | — | — | — |
| 11 | control | skew_comp | 0 | 20000 | 20 | 0.004091 | 0.07856 | 0.01234 | -0.008249 | -0.00661 | -0.006713 | 0.004025 | BEST | 3.57e-11 (4.99e-10) | 2.36e-08 (3.78e-07) | **BEST** | 0.04044 | 0.001656 | **PASS** | — | — | — |
| 12 | control | tail_comp | 0 | 20000 | 20 | 0.001419 | 0.008775 | 0.004906 | -0.003364 | -0.00251 | -0.0026 | 0.0004586 | BEST | 1.36e-07 (1.63e-06) | 8.36e-07 (1.25e-05) | **BEST** | 0.004333 | 0.00055 | **PASS** | — | — | — |
| 13 | struct | skew_struct | 1 | 2000 | 50 | 0.01954 | 0.02511 | 0.02302 | 0.001256 | 0.003863 | 0.003787 | 0.003153 | FAIL | 0.114 (0.685) | 0.788 (1) | **FAIL** | 0.012 | 0.02918 | **PASS** | 0.01386 | **FAIL** | 0.01466 |
| 14 | struct | tail_struct | 1 | 2000 | 50 | 0.01246 | 0.01381 | 0.01279 | 0.002554 | 0.004141 | 0.004081 | 0.0004502 | FAIL | 0.985 (1) | 0.995 (1) | **FAIL** | 0.007768 | 0.006406 | **FAIL** | 0.009109 | **FAIL** | 0.01038 |
| 15 | struct | skew_struct | 1 | 20000 | 20 | 0.006433 | 0.006945 | 0.0095 | 0.000652 | 0.001802 | 0.001743 | 0.003153 | NON-INFERIOR | 0.000661 (0.00661) | 0.83 (1) | **NON-INFERIOR** | 0.004416 | 0.03062 | **PASS** | 0.005842 | **FAIL** | 0.005554 |
| 16 | struct | tail_struct | 1 | 20000 | 20 | 0.00375 | 0.004 | 0.003867 | 0.00107 | 0.001461 | 0.001421 | 0.0004502 | FAIL | 0.994 (1) | 1 (1) | **FAIL** | 0.002491 | 0.004224 | **PASS** | 0.003245 | **FAIL** | 0.003277 |

CI = verdict of the one-sided 95% bound alone (larger of t and bootstrap); *verdict* =
after Holm across all 16 accuracy tests (note 3). FA = false-attribution check: upper
95% bound of mean |ĉ| on the graded channel must be < δ (Holm across the 4 is in the
JSON). err v5.1 is the secondary, descriptive OLS column (not gated).

## Failures (all kept)

* cell 2 (tail_comp, κ=0.5, n=2000): accuracy FAIL
* cell 3 (skew_comp, κ=1, n=2000): accuracy INCOMPLETE, role INCOMPLETE
* cell 4 (tail_comp, κ=1, n=2000): accuracy FAIL
* cell 6 (tail_comp, κ=0.5, n=20000): accuracy FAIL
* cell 7 (skew_comp, κ=1, n=20000): accuracy INCOMPLETE, role INCOMPLETE
* cell 8 (tail_comp, κ=1, n=20000): accuracy FAIL
* cell 13 (skew_struct, κ=1, n=2000): accuracy FAIL, false attribution FAIL
* cell 14 (tail_struct, κ=1, n=2000): accuracy FAIL, role FAIL, false attribution FAIL
* cell 15 (skew_struct, κ=1, n=20000): false attribution FAIL
* cell 16 (tail_struct, κ=1, n=20000): accuracy FAIL, false attribution FAIL

## Notes and deviations

1. **Cells 3 and 7 INCOMPLETE (not a code deviation; PREREG_V5_2 §3 rule).** For
   `skew_comp` at κ = 1.0 the σ_B calibration has no solution: with the χ²₃-type
   column 0 in period B, λ₂ of the period-B mean alone already exceeds λ₂(Y_A)
   (λ₂(Y_B) − λ₂(Y_A) = +0.0857 at σ_B → 0+, +0.0969 at 0.3, +0.912 at 3.0; oracle
   seed 99). Per §3 the cells are INCOMPLETE and were not re-designed; their 70 tasks
   were not run, so the pass covered 490 of the 560 pre-registered tasks. Recorded in
   `benchmarks/results/shiftbench_truth_shape_v5_2.json` before the gate ran.
2. **Seed key.** `seed_of("data", key, n, p, rep)` with key `scen|k{κ:g}`, i.e.
   `skew_comp|k0.5`, `skew_comp|k1`, `skew_comp|k0` (the prereg gives only the
   `|k0.5` example). Fixed in the runner merged in PR #34 before the run.
3. **"After Holm" (interpretation).** The prereg says cells must be BEST or
   NON-INFERIOR after Holm across all 16. The scorer (committed in PR #34, before the
   run) requires the CI verdict **and** the matching Holm-adjusted one-sided t p
   (p BEST for BEST, p NI for NON-INFERIOR) < 0.05, with INCOMPLETE cells entering
   the family as p = 1 so it stays 16. Under the looser reading (CI verdict alone)
   the gate still FAILS: cells 4, 13, 14, 16 fail accuracy on the CI bound, cell 14
   fails the role check, all four false-attribution checks fail, and cells 3 and 7
   are INCOMPLETE.
4. **Quantile rule.** Knots use linear-interpolation (type 7) quantiles of the
   pooled, centred covariates (the runner centres X as in V5.1). Pivoted-QR drop
   rule: |R_kk| ≤ 1e-10·|R_00|. No column was dropped in any of the 490 tasks.
5. **Report fields.** `explain_v5` now also returns `dropped_cols_a/b` (empty for the
   booster and OLS paths); the booster and OLS paths are otherwise unchanged from
   `665939b`.
6. **Secondary v5.1 column** was run inside the same pass (cells 1–8, 13–16), as
   pre-registered; it is descriptive only.
7. No tuning, no re-runs, no dropped or retried tasks.

Full numbers: `benchmarks/results/v5_2_gate_summary.json`; per-replication records:
`benchmarks/results/v5_2_gate_shape.jsonl`; wall: `v5_2_gate_wall.json`.
