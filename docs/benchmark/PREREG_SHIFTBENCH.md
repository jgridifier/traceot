# Pre-registration: ShiftBench

Frozen before any method result was computed or looked at. The commit that adds this file
is the freeze point. Only smoke runs on a handful of tasks, written outside the
repository and not inspected for accuracy, were used to check the harness runs end to end
and to size the grid by run time.

## What is fixed

* Design, scenarios, planted-value definitions, sizes, seeds and metrics: exactly as in
  [SHIFTBENCH.md](SHIFTBENCH.md) and the code in `benchmarks/shiftbench/` at this commit.
* Rival specifications (section 4 of the design). No tuning on the suite.
* The real-data panel, windows, outcome, features and winsorising.

## Primary endpoints

1. Mean-channel split error at n = 20,000, p = 20, averaged over the 11 scenarios.
2. Per-feature error (mean channel) at the same size.
3. Band coverage (target 0.95) and null false-positive rate (target 0.05).

Everything else (other channels, other sizes, invariance, runtime, real-data stability)
is secondary.

## Hypotheses about the rivals

* H1. OB has the largest mean-channel error in `nonlinear`; in the linear scenarios
  (`loc_comp`, `loc_struct`, `mixed`, `offsetting`) OB is within 0.02 of the best method.
* H2. Every method's mean-channel null false-positive rate is at most 0.10.
* H3. In `correlated`, SHAP's per-feature error exceeds RIF's (path-dependent attribution
  spreads x0's composition onto correlated features).
* H4. In `tail`, RIF has a larger q10/q90 split error than CFM (first-order
  linearisation of the quantile functional).

A hypothesis is supported when the paired per-replication difference has a 95% interval
on the predicted side; otherwise it is reported as not supported.

## Rules for adding TRACE

TRACE is evaluated on the same grid, seeds and truth with no changes to metrics. A claim
that TRACE beats a rival on an endpoint requires the paired per-replication difference
to have a 95% interval excluding zero, at every accuracy-grid size where both report it.
TRACE results are reported in full whether favourable or not.

## Deviations

Any change after the freeze is listed here with its commit and reason, and the frozen
version is still reported.
