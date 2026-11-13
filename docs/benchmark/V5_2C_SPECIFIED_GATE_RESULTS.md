# TRACE v5.2c specified-family replication — results

**Prereg SHA:** `f022ba3e42ebe4b7cee1dbdf2bf1c254b248b1f6` ([PREREG_V5_2C_SPECIFIED_REPLICATION.md](../research/PREREG_V5_2C_SPECIFIED_REPLICATION.md)).
**Implementation merge SHA:** `1aab4778abf8f84c8c8f7d9603eb8c322f1ba3c9` (reused, [PR #58](https://github.com/jgridifier/traceot/pull/58)).
**Runner SHA:** `1048c78e986c005e771d18f8c8eb27a8f9b69870`.
**Gate verdict:** **FAIL**. H is rejected.
**Park:** [PARK_V5_2C_SPECIFIED.md](../research/PARK_V5_2C_SPECIFIED.md). The inherited kill rule requires every failure to be recorded there. This is not a re-run and not an amendment.
**Wall time:** 12.7 s on 8 workers (cap 1800 s); 140/140 tasks finished. The 30-minute clock started after a successful pre-flight and a successful rehearsal.
**Pre-flight:** **OK**. Smoke fits on seed 99: {'TRACE': 'ok', 'DFL': 'ok', 'CFM': 'ok'}. Versions: traceot=0.1.0, numpy=2.5.4, scipy=1.18.1, scikit-learn=1.9.1.

Bars: PREREG_V4 §5 as inherited by PREREG_V5_2_SPECIFIED §5 and incorporated by reference in PREREG_V5_2C_SPECIFIED_REPLICATION (paired d = err_specified − err_best_rival, one-sided 95% upper bound, t-interval and 10 000-draw bootstrap (seed 2026) must agree, larger used; BEST if < 0, NON-INFERIOR if < δ = 0.10 × planted magnitude). **After Holm:** the CI verdict and the matching Holm-adjusted one-sided t p-value (p BEST for a BEST claim, p NI for a NON-INFERIOR claim) must both hold (p < 0.05). Rivals DFL and CFM re-run on the same seeds (paired), on the same centred arrays as earlier gates. The method under test is fit on raw (uncentred) draws. Role check: mean |ĉ| > mean |ŝ| and swapping roles does not lower the error.

| cell | n | p | reps | err specified | err DFL | err CFM | mean d | upper t | upper boot | δ | verdict | \|ĉ\| | \|ŝ\| | role | p NI (Holm) | p BEST (Holm) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|---|
| skew_comp | 2000 | 5 | 50 | 0.01321 | 0.06854 | 0.01771 | -0.00451 | -0.002249 | -0.002327 | 0.003322 | **BEST** | 0.03482 | 0.003764 | **PASS** | 2.3e-07 (9.21e-07) | 0.000793 (0.00317) |
| tail_comp | 2000 | 5 | 50 | 0.01043 | 0.01509 | 0.01172 | 0.0007307 | 0.002399 | 0.002323 | 0.0003562 | **FAIL** | 0.008416 | 0.002106 | **PASS** | 0.646 (0.646) | 0.767 (0.767) |
| skew_comp | 20000 | 20 | 20 | 0.006248 | 0.06911 | 0.008546 | -0.002298 | -0.0007251 | -0.0008636 | 0.003322 | **BEST** | 0.03482 | 0.001265 | **PASS** | 3.09e-06 (9.26e-06) | 0.0103 (0.0309) |
| tail_comp | 20000 | 20 | 20 | 0.003511 | 0.009134 | 0.004315 | -0.0007185 | 0.0002914 | 0.0002064 | 0.0003562 | **FAIL** | 0.004748 | 0.0009487 | **PASS** | 0.0407 (0.0814) | 0.117 (0.234) |

p values are unadjusted with Holm (across the completed cells) in brackets.
Full numbers: `benchmarks/results/v5_2c_specified_gate_summary.json`;
per-replication records: `benchmarks/results/v5_2c_specified_gate_shape.jsonl`.
A plain-language page is at [docs/learn/v5_2c_specified.html](../learn/v5_2c_specified.html).

## Secondary descriptive report (not a gate bar)

PREREG_V5_2B_SPECIFIED_REPLICATION §3 (by reference in V5.2c): TRACE-only
numbers, reported for
every cell, including `tail_comp`. This table is **not scored**. It does
not enter Holm, BEST, NON-INFERIOR, the role check, or the gate PASS /
FAIL / park decision. The observation from the V5.2 specified pass that
the `tail_comp` mean composition piece sat well below the planted value
is acknowledged; it is not used to change the hypothesis.

| cell | n | mean err | mean ĉ | mean ŝ | truth c | truth s | δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| skew_comp | 2000 | 0.01321 | 0.03482 | -0.001227 | 0.03294 | -0.0002855 | 0.003322 |
| tail_comp | 2000 | 0.01043 | 0.003972 | 0.0004051 | 0.003385 | -0.0001776 | 0.0003562 |
| skew_comp | 20000 | 0.006248 | 0.03482 | -0.0004601 | 0.03294 | -0.0002855 | 0.003322 |
| tail_comp | 20000 | 0.003511 | 0.004333 | -0.0003052 | 0.003385 | -0.0001776 | 0.0003562 |

## Notes and deviations

1. **Implementation, reused.** `TRACE_V5_2_SPECIFIED` = `explain_v5(..., mean_model="specified")` at `1aab4778abf8f84c8c8f7d9603eb8c322f1ba3c9`. The simulator and the truth file (oracle seed 99) are
   those already committed with that implementation. They were not
   recomputed. Residual law, grid, quadrature, hybrids, L-moments and
   the Shapley split are as in V5.1.
2. **No column centering** for the method under test, as frozen in the
   parent §2: the design (`x₃²`, `sin(x₅)`) is computed on the raw draws.
   Rivals use the same `centred` arrays as the V5.1 and V5.2 gates.
3. **Holm scoring** is imported from `summarize_v5_2_specified_gate.py`
   and is the convention frozen in the parent §5: both the CI verdict
   and the matching Holm p-value are required.
4. **Pre-flight.** Ran in a spawned child, in the exact gate environment, before any gate
   seed was touched: imports of TRACE, DFL, CFM and scikit-learn; one
   smoke fit of each method on non-gate seed 99; package versions
   recorded. The parent process did not import TRACE. The 30-minute cap
   started after this succeeded and after a successful rehearsal.
5. **Wall cap.** Single pass over all 140 tasks (n = 20 000 cells
   scheduled first) with a 30-minute cap; no task dropped, retried, or
   re-run.
6. **Scope.** A PASS shows only that the v5.1 mixture, with a correctly
   specified location family, meets the inherited bars under this one
   non-linear mean, on these fresh seeds. It does not un-park the V5.2
   spline claim, it does not change V5.1's linear-mean claim, and it
   does not re-run seeds 11000–11119 or 12000–12119. Indices 9600–9619 were not
   touched; 10400–10799 and 12000–12119 were not re-run.
7. **Spawn-safe harness.** Pre-flight ran in a spawned child. The
   rehearsal and gate pools were created with multiprocessing's
   `'spawn'` start method from a parent that never called TRACE.
8. **Rehearsal.** The full parallel pipeline (TRACE, DFL, CFM) ran on
   throwaway seeds 98 and 97, two per cell (8 tasks), at the same n
   and p as the gate cells, before any gate seed was touched.
   Rehearsal outputs are in
   `benchmarks/results/v5_2c_specified_gate_rehearsal.jsonl` and do
   not enter scoring.
9. **Watchdog.** If no task completed within 5 minutes, the pass
   would have been recorded as ABORTED (not FAIL). The 30-minute cap
   is a separate rule and still applies when tasks complete on time.
