# TRACE v5.1 structure/mixed-cell gate — results

**Prereg SHA:** `85a8fccfac44b6a1506037d785d4be6cf56c3bdb` ([PREREG_V5_1_STRUCT.md](../research/PREREG_V5_1_STRUCT.md)).
**Implementation merge SHA:** `665939bd4aa78ffc7474b91d482d18ac0191d595`.
**Runner SHA:** `5562b22140156b9cbec6aaa51f757cb1a803cd92`.
**Gate verdict:** **FAIL**. H is rejected.
**Park:** [PARK_V5_1_STRUCT.md](../research/PARK_V5_1_STRUCT.md). The kill rule requires every failure to be recorded there. This is not a re-run and not an amendment. The validated V5.1 claim is restated as **composition-only cells**.
**Wall time:** 19.3 s on 8 workers (cap 1800 s); 210/210 tasks finished.
**Pre-flight:** **OK**. Method identity (§2): explain_v5_1 is bit-identical to 665939b on the seed-99 skew_struct draw.
**Rehearsal:** **OK** (12/12 tasks, 2.9 s).

Bars: PREREG_V5_1_STRUCT.md §5 (paired d = err_v5.1 − err_best_rival, one-sided
95% upper bound, t-interval and 10 000-draw bootstrap (seed 2026), larger used;
BEST if < 0, NON-INFERIOR if < δ = 0.10 × planted magnitude). **After Holm:** the
CI verdict and the matching Holm-adjusted one-sided t p-value (p BEST for a BEST
claim, p NI for a NON-INFERIOR claim) must both hold (p < 0.05), Holm across the
six accuracy tests. Rivals DFL and CFM re-run on the same seeds (paired), on the
same centred arrays as the V5.1 gate. Role check: on structure-only cells,
mean |ŝ| > mean |ĉ| and swapping roles does not lower the error; on mixed cells,
sign(mean ĉ) = sign(truth comp) (+), sign(mean ŝ) = sign(truth struct) (−), and
swapping roles does not lower the error. False-attribution (structure-only):
upper bound of mean |ĉ| < δ and the Holm-adjusted one-sided t p-value < 0.05,
Holm across the four structure-only tests.

| cell | n | p | reps | err v5.1 | err DFL | err CFM | mean d | upper t | upper boot | δ | verdict | mean ĉ | mean ŝ | \|ĉ\| | \|ŝ\| | role | FA bound | p NI (Holm) | p BEST (Holm) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---|---|
| skew_struct | 2000 | 5 | 50 | 0.01176 | 0.01161 | 0.01995 | 0.002018 | 0.004366 | 0.004219 | 0.005072 | **NON-INFERIOR** | 0.0001158 | 0.04787 | 0.006704 | 0.04787 | **PASS** | 0.007726 (**FAIL**) | 0.017 (0.034) | 0.922 (1) |
| tail_struct | 2000 | 5 | 50 | 0.00679 | 0.00811 | 0.01138 | 0.0001755 | 0.001677 | 0.001642 | 0.0009827 | **FAIL** | 0.0008566 | 0.009367 | 0.003703 | 0.009405 | **PASS** | 0.004308 (**FAIL**) | 0.186 (0.186) | 0.577 (1) |
| shape_mixed | 2000 | 5 | 50 | 0.01302 | 0.08165 | 0.02754 | -0.01447 | -0.01129 | -0.01138 | 0.009298 | **BEST** | 0.04079 | -0.05086 | 0.04079 | 0.05086 | **PASS** | — | 3.59e-17 (2.15e-16) | 3.69e-10 (1.85e-09) |
| skew_struct | 20000 | 20 | 20 | 0.004001 | 0.005406 | 0.01706 | -0.001405 | -0.0003228 | -0.0004151 | 0.005072 | **NON-INFERIOR** | -6.785e-05 | 0.05001 | 0.002096 | 0.05001 | **PASS** | 0.00274 (**PASS**) | 1.52e-09 (6.08e-09) | 0.0184 (0.0738) |
| tail_struct | 20000 | 20 | 20 | 0.002037 | 0.002856 | 0.007474 | -0.0006787 | -5.891e-05 | -0.0001256 | 0.0009827 | **NON-INFERIOR** | 0.0002137 | 0.01029 | 0.001104 | 0.01029 | **PASS** | 0.001417 (**FAIL**) | 9.03e-05 (0.000271) | 0.0368 (0.11) |
| shape_mixed | 20000 | 20 | 20 | 0.004037 | 0.08059 | 0.02206 | -0.01803 | -0.01613 | -0.01619 | 0.009298 | **BEST** | 0.04027 | -0.05359 | 0.04027 | 0.05359 | **PASS** | — | 2.84e-16 (1.42e-15) | 5.46e-13 (3.27e-12) |

p values are unadjusted with Holm (across the six accuracy tests) in brackets.
False-attribution Holm is across the four structure-only tests.
Truth (committed shiftbench_truth_shape.json, oracle seed 99, not recomputed): skew_struct λ3: comp -0.000351, struct 0.05037; tail_struct λ4: comp 8.511e-05, struct 0.009742; shape_mixed λ3: comp 0.03951, struct -0.05347.

Full numbers: `benchmarks/results/v5_1_struct_gate_summary.json`;
per-replication records: `benchmarks/results/v5_1_struct_gate_shape.jsonl`.

## Failures kept

- skew_struct n=2000: false-attribution FAIL
- tail_struct n=2000: accuracy FAIL
- tail_struct n=2000: false-attribution FAIL
- tail_struct n=20000: false-attribution FAIL

## Secondary, descriptive, not scored

These readouts cannot rescue or sink H (PREREG_V5_1_STRUCT.md §8).

1. Mean ĉ and ŝ against the planted truth, and their relation to δ.
   On `tail_struct` this shows whether the λ₄ composition piece is biased
   when its truth is ≈ 0 (the open `tail_comp` composition under-estimate).

| cell | n | mean ĉ | truth c | mean ŝ | truth s | δ | ĉ − truth c | ŝ − truth s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| skew_struct | 2000 | 0.0001158 | -0.000351 | 0.04787 | 0.05037 | 0.005072 | 0.0004668 | -0.002503 |
| tail_struct | 2000 | 0.0008566 | 8.511e-05 | 0.009367 | 0.009742 | 0.0009827 | 0.0007715 | -0.0003746 |
| shape_mixed | 2000 | 0.04079 | 0.03951 | -0.05086 | -0.05347 | 0.009298 | 0.001275 | 0.00261 |
| skew_struct | 20000 | -6.785e-05 | -0.000351 | 0.05001 | 0.05037 | 0.005072 | 0.0002832 | -0.0003545 |
| tail_struct | 20000 | 0.0002137 | 8.511e-05 | 0.01029 | 0.009742 | 0.0009827 | 0.0001286 | 0.0005503 |
| shape_mixed | 20000 | 0.04027 | 0.03951 | -0.05359 | -0.05347 | 0.009298 | 0.0007592 | -0.0001207 |

2. Median per-task wall seconds of TRACE, DFL, and CFM per cell (no speed bar).

| cell | n | median TRACE s | median DFL s | median CFM s |
|---|---:|---:|---:|---:|
| skew_struct | 2000 | 0.006349 | 0.004104 | 0.1801 |
| tail_struct | 2000 | 0.0065 | 0.004071 | 0.1761 |
| shape_mixed | 2000 | 0.006418 | 0.004212 | 0.1894 |
| skew_struct | 20000 | 0.0683 | 0.06904 | 1.723 |
| tail_struct | 20000 | 0.06844 | 0.064 | 1.727 |
| shape_mixed | 20000 | 0.06733 | 0.08414 | 1.818 |

## Notes and deviations

1. **Method identity.** Four commits after `665939b` touched `crates/core`:
   `5db5806` (V5.2 spline), `3b8a1d4` (V5.1 made the default), `3b382f0`
   (report-only residual-vs-fit check), and `4cf9754` (V5.2 specified model).
   Pre-flight compared `explain_v5_1` on one centred `skew_struct` draw at
   oracle seed 99 to the frozen `665939b` record. The split, the four hybrid
   quantile / L-moment curves, and the mixing residuals were bit-identical.
   The later report-only `linear_mean_check` and `dropped_cols_*` fields are
   present only on the later SHA and do not enter the split.
2. **Harness.** Spawn-safe pool, mandatory pre-flight in a spawned child,
   rehearsal on seeds 97 and 98 (12 tasks, never scored), 5-minute no-progress
   watchdog, 30-minute cap. The confirmatory pass ran only with `--confirmatory`.
   Sealed seeds 9600–9619 and coverage seeds 10400–10799 were in FORBIDDEN
   and were not run.
3. **No deviations** from PREREG_V5_1_STRUCT.md are recorded in this file.
   Any later change would be a deviation and must be reported as one.

