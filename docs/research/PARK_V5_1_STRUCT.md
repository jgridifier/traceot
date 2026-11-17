# PARK — TRACE v5.1 structure/mixed-cell claim

**Parked:** 2026-11-17, 8:09 AM, America/New_York, by the kill rule of
[PREREG_V5_1_STRUCT.md](PREREG_V5_1_STRUCT.md) §6 (freeze
`85a8fccfac44b6a1506037d785d4be6cf56c3bdb`, PR #70).
**Implementation:** `665939bd4aa78ffc7474b91d482d18ac0191d595` (V5.1 merge,
PR #29). Pre-flight confirmed that `explain_v5_1` on one centred
`skew_struct` draw at oracle seed 99 is bit-identical to this SHA.
**Runner:** `5562b22140156b9cbec6aaa51f757cb1a803cd92` (PR #72).
**Results:** [V5_1_STRUCT_GATE_RESULTS.md](../benchmark/V5_1_STRUCT_GATE_RESULTS.md).

## What is parked

H (PREREG_V5_1_STRUCT.md §1): *on the frozen linear-mean H7 shape simulator,
`TRACE_V5_1`, unchanged, is BEST or NON-INFERIOR against the best paired
rival (DFL, CFM) after Holm on all six structure-only and mixed cells, and
passes the role check in all six cells, and passes the false-attribution
check in all four structure-only cells.* **Rejected.** No widened claim
that V5.1 "separates composition from structure on the full H7 shape
family" is made.

## Why (every failure, from the single pass)

The pass finished 210/210 tasks in 19.3 s (cap 1 800 s), after a successful
pre-flight (method-identity bit-identical to `665939b`; smoke fits on seed
99) and a successful rehearsal (12/12 on throwaway seeds 98 and 97 in
2.9 s). All six role checks passed. Both mixed cells (`shape_mixed` at
n = 2 000 and n = 20 000) were BEST after Holm, with the correct signs.
The failures, and only these, are:

* **Accuracy FAIL after Holm:** `tail_struct`, n = 2 000, p = 5, 50 reps.
  TRACE mean error 0.00679, best rival (DFL) 0.00811, CFM 0.01138.
  Mean paired \(d\) = +0.0001755 (TRACE slightly worse on average than the
  best rival). One-sided 95% upper bound 0.001677 (t; bootstrap 0.001642;
  the larger is used) is above \(\delta\) = 0.0009827, so the CI verdict
  itself is FAIL. Holm-adjusted p NI = 0.186; p BEST = 1. Role check PASS
  (\(|\hat s|\) = 0.009405, \(|\hat c|\) = 0.003703).

* **False-attribution FAIL:** `skew_struct`, n = 2 000, p = 5, 50 reps.
  Accuracy was NON-INFERIOR after Holm (upper bound 0.004366 < \(\delta\)
  = 0.005072; Holm p NI = 0.034). Role check PASS. The one-sided 95%
  upper bound of mean \(|\hat c|\) is 0.007726, which is not \(< \delta\).
  Holm-adjusted p for this check is 1. False-attribution FAIL.

* **False-attribution FAIL:** `tail_struct`, n = 2 000, p = 5, 50 reps
  (in addition to the accuracy FAIL above). Upper bound of mean
  \(|\hat c|\) is 0.004308, above \(\delta\) = 0.0009827. Holm-adjusted
  p is 1.

* **False-attribution FAIL:** `tail_struct`, n = 20 000, p = 20, 20 reps.
  Accuracy was NON-INFERIOR after Holm (CI verdict BEST, upper bound
  −0.00005891; Holm p BEST = 0.11, so the cell is not BEST; Holm p NI
  = 0.000271). Role check PASS. Upper bound of mean \(|\hat c|\) is
  0.001417, above \(\delta\) = 0.0009827. Holm-adjusted p is 1.

The remaining two cells did not fail any scored check:
`shape_mixed` at both sizes (BEST, role PASS) and `skew_struct` at
n = 20 000 (NON-INFERIOR after Holm, role PASS, false-attribution PASS).
They cannot rescue H. No cell was INCOMPLETE. The watchdog did not trip.
There is no environment or harness failure on this pass.

A descriptive (unscored) note, which does not enter the verdict: on the
structure-only cells the *signed* mean \(\hat c\) stays close to the
planted near-zero composition truth, but the mean of \(|\hat c|\) is not
inside \(\delta\). The false-attribution bar is on the absolute value.

## What stays

* **V5.1's composition-cell claim stands exactly as stated** in
  [V5_1_GATE_RESULTS.md](../benchmark/V5_1_GATE_RESULTS.md): on the
  linear-mean simulator, `TRACE_V5_1` is BEST against DFL and CFM on
  `skew_comp` and `tail_comp` at both sizes, with the role check
  passing. In every public place the claim is now named as
  **composition-only cells**, with this failure recorded.
* The V5.1 coverage PASS stands unchanged
  ([V5_1_COVERAGE_RESULTS.md](../benchmark/V5_1_COVERAGE_RESULTS.md));
  it was not touched.
* The V5.2 spline park and the V5.2c specified-family park stand
  unchanged ([PARK_V5_2.md](PARK_V5_2.md),
  [PARK_V5_2C_SPECIFIED.md](PARK_V5_2C_SPECIFIED.md)).
* 9600–9619 stays sealed. 14000–14049 and 14100–14119 are now spent.
  10400–10799 was not touched.

## Not allowed under this park

No amendment on these cells, no seed changes, no re-runs to chase a bar,
no new location model under PREREG_V5_1_STRUCT.md. Any later attempt
to widen V5.1 beyond composition-only cells needs a new
pre-registration on fresh seeds, and its hypothesis must not be chosen
by looking at this run beyond what is written here.
