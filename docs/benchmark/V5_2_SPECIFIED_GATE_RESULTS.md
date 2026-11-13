# TRACE v5.2 specified-family gate — results

**Prereg SHA:** `cc8c1d7afc9c58a8b97b8b3b28b6ad6fd1704920` ([PREREG_V5_2_SPECIFIED.md](../research/PREREG_V5_2_SPECIFIED.md)).
**Implementation merge SHA:** `1aab4778abf8f84c8c8f7d9603eb8c322f1ba3c9` ([PR #58](https://github.com/jgridifier/traceot/pull/58)).
**Approval:** 2026-11-13, 2:59 PM, America/New_York. The owner said, quoted verbatim: *"You have my approval for all trace stuff - you are running the lab on this - I’m more an advisor on that project."* Approval commit: [PR #56](https://github.com/jgridifier/traceot/pull/56), merge `05375628e44b1b203c0fb6c92a1c20d37e067742`.
**Gate verdict:** **INCOMPLETE (rivals unavailable)**. H is not decided. This is not a FAIL and is not a park.
**Wall time:** 1.305 s on 8 workers (cap 1800 s); 140/140 TRACE tasks finished. Calibration of σ_B was feasible (no cell skipped).
**Run window:** 2026-11-13, 3:23:21 PM to 3:23:23 PM, America/New_York.

Bars: PREREG_V4 §5 as inherited by PREREG_V5_2_SPECIFIED §5 (paired d = err_specified − err_best_rival, one-sided 95% upper bound, t-interval
and 10 000-draw bootstrap (seed 2026) must agree, larger used; BEST if < 0,
NON-INFERIOR if < δ = 0.10 × planted magnitude). **After Holm:** the CI
verdict and the matching Holm-adjusted one-sided t p-value (p BEST for a
BEST claim, p NI for a NON-INFERIOR claim) must both hold (p < 0.05).
Rivals DFL and CFM re-run on the same seeds (paired), on the same centred
arrays as earlier gates. The method under test is fit on raw (uncentred)
draws (PREREG_V5_2_SPECIFIED §2). Role check: mean |ĉ| > mean |ŝ| and
swapping roles does not lower the error.

| cell | n | p | reps | err specified | err DFL | err CFM | mean d | upper t | upper boot | δ | verdict | \|ĉ\| | \|ŝ\| | role | p NI (Holm) | p BEST (Holm) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|---|
| skew_comp | 2000 | 5 | 50 | 0.01314 | — | — | — | — | — | 0.003322 | **INCOMPLETE (rivals unavailable)** | 0.03387 | 0.003177 | **PASS** | — | — |
| tail_comp | 2000 | 5 | 50 | 0.01136 | — | — | — | — | — | 0.0003562 | **INCOMPLETE (rivals unavailable)** | 0.009138 | 0.002055 | **PASS** | — | — |
| skew_comp | 20000 | 20 | 20 | 0.005438 | — | — | — | — | — | 0.003322 | **INCOMPLETE (rivals unavailable)** | 0.03296 | 0.001157 | **PASS** | — | — |
| tail_comp | 20000 | 20 | 20 | 0.004112 | — | — | — | — | — | 0.0003562 | **INCOMPLETE (rivals unavailable)** | 0.003839 | 0.0006189 | **PASS** | — | — |

p values are unadjusted with Holm (across the completed cells) in brackets.
Full numbers: `benchmarks/results/v5_2_specified_gate_summary.json`;
per-replication records: `benchmarks/results/v5_2_specified_gate_shape.jsonl`.
A plain-language page is at [docs/learn/v5_2_specified.html](../learn/v5_2_specified.html).

## Notes and deviations

1. **Implementation.** `TRACE_V5_2_SPECIFIED` = `explain_v5(..., mean_model="specified")`.
   Only the location fit changed; residual law, grid, quadrature, hybrids,
   L-moments and the Shapley split are as in V5.1. Gate seeds are run only after
   this implementation is merged, and only if calibration of σ_B was feasible.
2. **No column centering** for the method under test, as frozen in §2: the design
   (`x₃²`, `sin(x₅)`) is computed on the raw draws. Rivals use the same `centred`
   arrays as the V5.1 and V5.2 gates, so their settings are unchanged.
3. **Holm scoring** follows the convention fixed in §5 to avoid the V5.2 note-3
   ambiguity: both the CI verdict and the matching Holm p-value are required.
4. **Wall cap.** Single pass over all 140 tasks (n = 20 000 cells scheduled first)
   with a 30-minute cap; no task dropped, retried, or re-run.
5. **Scope (PREREG_V5_2_SPECIFIED §7–8).** A PASS shows only that the v5.1
   mixture, with a correctly specified location family, meets the inherited bars
   under this one non-linear mean. It does not un-park the V5.2 spline claim, and
   it does not change V5.1's linear-mean claim. Indices 9600–9619 were not
   touched; 10400–10799 were not used.
6. **Rivals unavailable (the reason H is not decided).** Every one of the
   140 DFL calls and every one of the 140 CFM calls recorded the error
   `No module named 'sklearn'`. The single pass ran in a fresh Python 3.13
   virtual environment that had the TRACE wheel, numpy and scipy, but not
   scikit-learn, which `sb_shape_readout.py` imports for both rivals. That
   dependency should have been checked before the pass started. It was not.
   A previously used environment on the same machine (`v51-venv`) has
   scikit-learn 1.9.1. This is an environment miss, not a method failure
   and not a wall-cap miss.

   Scoring those cells as BEST against an infinite rival error would have
   been a false PASS. The summariser was therefore patched, after the run
   and before this file was written, to treat "all rivals recorded an
   error" as **INCOMPLETE (rivals unavailable)** and to leave H undecided.
   That is a scoring-harness fix. It does not re-run any gate seed and it
   does not change any TRACE number in the jsonl.

7. **No second pass.** PREREG_V5_2_SPECIFIED §6: "No task dropped, retried,
   or re-run" and "no re-runs to chase a bar." The 140 TRACE fits and the
   280 rival error records are the official pass. Rivals were not retried
   after scikit-learn was noticed to be missing. Seeds 11000–11049 and
   11100–11119 are spent. A complete accuracy test against DFL and CFM
   would need a new pre-registration on fresh seeds. This file does not
   open that pre-registration.

8. **No park file.** The kill rule writes `PARK_V5_2_SPECIFIED.md` when H
   is rejected (accuracy FAIL, role-check FAIL, or wall-cap INCOMPLETE).
   None of those happened. H is not decided, the same way the
   calibration-INCOMPLETE stop of §3 leaves H undecided. The V5.1
   linear-mean claim is unchanged. The V5.2 spline park stays in force.

9. **TRACE-only numbers (descriptive, not a gate bar).** TRACE finished
   every task, with no dropped design columns. Role check (does not need
   rivals) is PASS on all four cells: mean |ĉ| > mean |ŝ|, and swapping
   roles does not lower the error. Mean graded errors and mean signed
   pieces, reported only:

   | cell | n | mean err | mean ĉ | mean ŝ | truth c | truth s | δ |
   |---|---:|---:|---:|---:|---:|---:|---:|
   | skew_comp | 2000 | 0.01314 | 0.03387 | −0.000780 | 0.03294 | −0.00029 | 0.00332 |
   | tail_comp | 2000 | 0.01136 | 0.000721 | 0.000338 | 0.00338 | −0.00018 | 0.000356 |
   | skew_comp | 20000 | 0.005438 | 0.03296 | −0.000590 | 0.03294 | −0.00029 | 0.00332 |
   | tail_comp | 20000 | 0.004112 | 0.002092 | 0.000101 | 0.00338 | −0.00018 | 0.000356 |

   On `tail_comp`, the mean composition piece is well below the planted
   value, and the mean error is large relative to δ. That is reported
   honestly. It is not a scored accuracy verdict, because the bars
   compare against rivals that this pass did not produce.

10. **Sealed and reserved seeds.** Indices 9600–9619 were not touched.
    Indices 10400–10799 were not used on this pass (they were already
    spent on the V5.1 coverage study).
