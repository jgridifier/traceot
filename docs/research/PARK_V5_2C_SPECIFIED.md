# PARK — TRACE v5.2c specified-family claim

**Parked:** 2026-11-13, 4:29 PM, America/New_York, by the kill rule of
[PREREG_V5_2C_SPECIFIED_REPLICATION.md](PREREG_V5_2C_SPECIFIED_REPLICATION.md)
(freeze `f022ba3e42ebe4b7cee1dbdf2bf1c254b248b1f6`, PR #64), which
incorporates the kill rule of
[PREREG_V5_2_SPECIFIED.md](PREREG_V5_2_SPECIFIED.md) §6 (freeze `cc8c1d7`)
by reference.
**Implementation:** `1aab4778abf8f84c8c8f7d9603eb8c322f1ba3c9` (reused, PR #58).
**Runner:** `1048c78e986c005e771d18f8c8eb27a8f9b69870` (PR #66).
**Results:** [V5_2C_SPECIFIED_GATE_RESULTS.md](../benchmark/V5_2C_SPECIFIED_GATE_RESULTS.md).

## What is parked

H (PREREG_V5_2_SPECIFIED §1, by reference in V5.2c): *a correctly specified
non-linear location model (ordinary least squares on the linear terms plus
the exact two curved terms, \(x_3^2\) and \(\sin x_5\)) still passes the
four-cell gate* — BEST or NON-INFERIOR against DFL/CFM after Holm, and the
role check, in every cell. **Rejected.** No specified-family curved-mean
claim is made.

## Why (every failure, from the single pass)

The pass finished 140/140 tasks in 12.7 s (cap 1 800 s), after a successful
pre-flight and a successful rehearsal (8/8 on throwaway seeds 98 and 97).
All four role checks passed. The two `skew_comp` cells were BEST after Holm.
The failures, and only these, are:

* **Accuracy FAIL after Holm:** `tail_comp`, n = 2 000, p = 5, 50 reps.
  TRACE mean error 0.01043, best rival (CFM) 0.01172, DFL 0.01509.
  Mean paired \(d\) = +0.0007307 (TRACE slightly worse on average).
  One-sided 95% upper bound 0.002399 (t; bootstrap 0.002323; the larger
  is used) is above \(\delta\) = 0.0003562, so the CI verdict itself is
  FAIL. Holm-adjusted p NI = 0.646; p BEST = 0.767. Role check PASS
  (\(|\hat c|\) = 0.008416, \(|\hat s|\) = 0.002106).

* **Accuracy FAIL after Holm:** `tail_comp`, n = 20 000, p = 20, 20 reps.
  TRACE mean error 0.003511, best rival (CFM) 0.004315, DFL 0.009134.
  Mean paired \(d\) = −0.0007185 (TRACE better on average).
  The CI upper bound 0.0002914 (t; bootstrap 0.0002064) is below
  \(\delta\) = 0.0003562, so the CI verdict is NON-INFERIOR. The
  matching Holm-adjusted p NI is 0.0814, which is not \(< 0.05\).
  Both the CI verdict and the matching Holm p-value are required.
  Verdict FAIL. Holm-adjusted p BEST = 0.234. Role check PASS
  (\(|\hat c|\) = 0.004748, \(|\hat s|\) = 0.0009487).

No cell failed the role check. No cell was INCOMPLETE. The watchdog
did not trip. There is no environment or harness failure on this pass.

## What stays

* **V5.1's linear-mean claim stands exactly as stated** in
  [V5_1_GATE_RESULTS.md](../benchmark/V5_1_GATE_RESULTS.md). This
  failure neither strengthens nor weakens it.
* The V5.2 spline park stands unchanged
  ([PARK_V5_2.md](PARK_V5_2.md)). This failure does not add a new
  charge against that parked claim, and a later specified-family
  PASS would not have revived it.
* The first specified-family pass remains
  **INCOMPLETE (rivals unavailable)**
  ([V5_2_SPECIFIED_GATE_RESULTS.md](../benchmark/V5_2_SPECIFIED_GATE_RESULTS.md));
  H was not decided there. The V5.2b pass remains
  **ABORTED (no verdict)**
  ([V5_2B_ABORT.md](../benchmark/V5_2B_ABORT.md)).
* The `mean_model="specified"` code stays in the library as an option;
  it carries no validated claim.
* 9600–9619 stays sealed. 13000–13049 and 13100–13119 are now spent,
  as are 11000–11049 / 11100–11119 and 12000–12049 / 12100–12119.

## Not allowed under this park

No amendment on these cells, no new basis term, no seed changes, no
re-runs to chase a bar. Any further specified-family or curved-mean
work needs a new pre-registration on fresh seeds, and its hypothesis
must not be chosen by looking at this run beyond what is written here.
