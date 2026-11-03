# PARK — TRACE v5.2 curved-mean claim

**Parked:** 2026-11-03 (America/New_York), by the kill rule of
[PREREG_V5_2.md](PREREG_V5_2.md) §6 (freeze `df5aff8`).
**Implementation:** `ff0a8c659f5e7a3b68b424d354724635ef0c7c83` (PR #34).
**Results:** [V5_2_GATE_RESULTS.md](../benchmark/V5_2_GATE_RESULTS.md).

## What is parked

H (PREREG_V5_2 §1): *with a smooth non-linear additive mean, `TRACE_V5_2` (v5.1 mixture
with a fixed additive restricted-cubic-spline location fit) is BEST or NON-INFERIOR
against DFL/CFM, passes the role check in all 16 cells, and does not attribute a
structure-only change to composition.* **Rejected.** No curved-mean claim is made.

## Why (every failure, from the single pass)

* **INCOMPLETE:** cells 3 and 7 (`skew_comp`, κ = 1.0) — σ_B calibration infeasible
  (the period-B mean alone is already wider in λ₂ than Y_A).
* **Accuracy FAIL after Holm:** cells 2, 4, 6, 8 (`tail_comp`, κ = 0.5 and 1.0, both
  sizes), 13, 14, 16 (structure only). Cells 4, 13, 14, 16 fail on the CI bound itself.
* **Role FAIL:** cell 14 (`tail_struct`, n = 2000).
* **False attribution FAIL:** all four structure-only cells (13–16): the upper bound
  of mean |ĉ| is above δ at both sizes.

## What stays

* **V5.1's linear-mean claim stands exactly as stated** in
  [V5_1_GATE_RESULTS.md](../benchmark/V5_1_GATE_RESULTS.md). This failure neither
  strengthens nor weakens it.
* The `mean_model="spline"` code stays in the library as an option; it carries no
  validated claim.
* 9600–9619 stays sealed. 10100–10149 and 10200–10219 are now spent.

## Not allowed under this park

No amendment on these cells, no new knot rule, no seed changes, no re-runs to chase a
bar. Any further curved-mean work needs a new pre-registration on fresh seeds, and its
hypothesis must not be chosen by looking at this run beyond what is written here.
