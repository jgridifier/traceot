# TRACE v5.1 gate — results

**Prereg SHA:** `b33f507` ([PREREG_V5_1.md](../research/PREREG_V5_1.md)).
**Implementation merge SHA:** `665939bd4aa78ffc7474b91d482d18ac0191d595`.
**Gate verdict:** **PASS**.
**Wall time:** 15.0 s on 8 workers (cap 1800 s); 140/140 tasks finished.

Bars: PREREG_V4 §5 as inherited by PREREG_V5_1 §4 (paired d = err_v5.1 − err_best_rival,
one-sided 95% upper bound, t-interval and 10 000-draw bootstrap (seed 2026) must agree, larger used;
BEST if < 0, NON-INFERIOR if < δ = 0.10 × planted magnitude). Rivals DFL and CFM
re-run on the same seeds (paired). Role check: mean |ĉ| > mean |ŝ| and swapping
roles does not lower the error (V5_GATE1_RESULTS note 2).

| cell | n | p | reps | err v5.1 | err DFL | err CFM | mean d | upper t | upper boot | δ | verdict | \|ĉ\| | \|ŝ\| | role | p NI (Holm) | p BEST (Holm) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|---|
| skew_comp | 2000 | 5 | 50 | 0.008351 | 0.07792 | 0.0231 | -0.01475 | -0.01206 | -0.01218 | 0.004025 | **BEST** | 0.0379 | 0.001774 | **PASS** | 4.21e-16 (1.68e-15) | 1.51e-12 (6.05e-12) |
| tail_comp | 2000 | 5 | 50 | 0.005568 | 0.01348 | 0.01138 | -0.003761 | -0.002339 | -0.00237 | 0.0004586 | **BEST** | 0.004493 | 0.001809 | **PASS** | 4.23e-06 (8.47e-06) | 2.62e-05 (5.23e-05) |
| skew_comp | 20000 | 20 | 20 | 0.002875 | 0.08113 | 0.01113 | -0.00825 | -0.006373 | -0.006447 | 0.004025 | **BEST** | 0.04097 | 0.0007853 | **PASS** | 3.5e-10 (1.05e-09) | 1.77e-07 (5.32e-07) |
| tail_comp | 20000 | 20 | 20 | 0.001933 | 0.009159 | 0.005063 | -0.002988 | -0.001949 | -0.002048 | 0.0004586 | **BEST** | 0.003909 | 0.0007253 | **PASS** | 7.98e-06 (8.47e-06) | 4.26e-05 (5.23e-05) |

Truth (Shapley, graded channel): skew_comp λ3: comp 0.04011, struct -0.0001375; tail_comp λ4: comp 0.004298, struct 0.000288.

p values are unadjusted with Holm (across the completed cells) in brackets.
Full numbers: `benchmarks/results/v5_1_gate_summary.json`; per-replication
records: `benchmarks/results/v5_1_gate_shape.jsonl`.

## Notes and deviations

1. **Implementation.** `TRACE_V5_1` = `explain_v5(..., mean_model="ols")`
   (PR #29). Only the location fit changed; grid, quadrature, hybrids, L-moments and
   the Shapley split are as at `3d695bc`. Gate seeds were first run after the merge.
2. **t-interval.** PREREG_V4 §5 specifies a t-interval; the earlier v4/v5 scorers
   used a normal quantile. This scorer uses Student t with n−1 df (and t-based
   one-sided p values), as written in the pre-registration. The t bound is wider,
   so the change can only make a cell harder to pass.
3. **Wall cap.** Single pass over all 140 tasks (n=20 000 cells scheduled first)
   with a 30-minute cap; no task was dropped, retried or re-run.
4. **Role check** as in V5_GATE1_RESULTS note 2: mean |ĉ| > mean |ŝ| on the graded
   channel and the role-swapped error is not lower.
5. **Holm** across all four cells (family of four, as pre-registered).
6. **Scope (PREREG_V5_1 §7).** The simulator's mean is linear, so the linear
   location model is the right family by construction. This PASS says the v5
   mixture works once the location model is right; it says nothing about
   non-linear means, which need their own pre-registration. No other suite
   (struct cells, coverage, ShiftBench, speed) was run under this SHA.
