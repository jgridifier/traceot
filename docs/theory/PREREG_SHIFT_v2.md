# Pre-registration: shift decomposition v2

Frozen BEFORE any implementation or result exists. The commit containing this file is the freeze
point; its sha is recorded in the PR. Any change after freeze must be a new, dated amendment section
below; bars are never edited. Failed bars stay on record as failed.

## Common setup
* Seeds: 20 per scenario, seeds 1000..1019 (numpy PCG64 / Rust equivalent documented in code).
* n = 20,000 rows per period, p = 8 features, K = 6 channels, 5-fold cross-fit, B = 200 bootstrap
  replicates, R = 200 permutations, weight trimming at the 0.995 quantile.
* Covariates: X_A ~ N(0, Σ), Σ_ij = 0.3^|i−j|. Response: Y = m(X) + s(X)·ε, ε ~ skew-normal(α=3)
  standardized; m(x) = Σ_j β_j tanh(x_j), s(x) = exp(0.2 x_1 − 0.1 x_2).
* "Truth" for each planted scenario is computed by Monte Carlo with n = 2,000,000 per period using
  the known data-generating process (exact counterfactual laws, no reweighting).
* Band = 95% bootstrap band of §6 of SHIFT_V2.md.

## Scenarios
| ID | Scenario | Planted change A→B |
|----|----------|--------------------|
| S1 | pure composition | X_B ~ N(μ, Σ), μ = (0.5, −0.3, 0.2, 0,…); same m, s |
| S2 | pure structure | X_B ~ X_A; β_B = β_A + (0.4, 0, −0.3, 0,…), s_B(x)=1.2·s(x) |
| S3 | mixed, known split | S1 covariates + S2 structure |
| S4 | null | identical DGP in A and B |
| S5 | entry/exit | 15% of A units exit (x_1 > q_0.85), 25% new units enter from N((1,0,…),Σ); stayers unchanged |

## Pass bars (frozen)
| Bar | Statistic | Pass criterion |
|-----|-----------|----------------|
| B1 | S1: structure share of total W2² (marginal target) | true value (≈0) inside band in ≥18/20 seeds |
| B2 | S2: composition share of total W2² | true value (≈0) inside band in ≥18/20 seeds |
| B3 | S3: composition and structure W2² per channel k=0..2 | MC truth inside band for both, ≥18/20 seeds |
| B4 | S3: per-feature comp share for features 1–3 (barycenter target) | MC truth inside band ≥18/20 seeds |
| B5 | S4: total W2² exceeds permutation 95% floor | ≤1/20 seeds (false positive) |
| B6 | S4: any player flagged "resolved" at the uniform band | ≤1/20 seeds |
| B7 | All: order invariance (swap A/B roles in ordering) | |diff| ≤ 1e-9 every seed |
| B8 | All: Σ players = total W2² (each channel and overall) | |diff| ≤ 1e-9 every seed |
| B9 | S5: entry and exit terms vs MC truth | truth inside band ≥18/20 seeds; stay term band covers 0 ≥18/20 |
| B10 | Verify script `verify/shift_v2_checks.py` | all checks PASS |
| B11 | Speed: 1,000,000 rows × 100 features per period, decomposition only (fitted models given), 1 machine | reported honestly; target ≤ 60 s wall, not a pass/fail gate for correctness |

## Reporting
Results table with per-bar counts (k/20), the seeds that failed, and raw outputs committed under
`docs/theory/verify/PREREG_SHIFT_v2_RESULTS.txt`.
