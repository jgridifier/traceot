# TRACE claims (H5–H8): results

Scored exactly per [PREREG_TRACE_CLAIMS.md](PREREG_TRACE_CLAIMS.md) as amended by [AMENDMENT_1](PREREG_TRACE_CLAIMS_AMENDMENT_1.md) (deviations D1–D12 logged before results). Runs were stopped early at the user's request (D13): `placebo`, `h6syn` and `scale` not run; `null200` has 192 of 200 replications. Raw records: `benchmarks/results/trace_claims_*.jsonl`; summary: `trace_claims_summary.json` (produced by `summarize_trace_claims.py`, rendered by `render_trace_claims.py`). Failures stay on record.

## Verdicts

| hypothesis | bar | result | verdict |
|---|---|---|---|
| H5 exact budget + invariance | every gap ≤ 1e-9·max(1,W2²) in every replication; permutation ≤ 1e-6 | fail counts {'g51a': 0, 'g51b': 0, 'g52': 1175, 'g52_internal': 0, 'g53_chan': 0, 'g53_coef': 0, 'g54': 0}; invariance max 1.11e-16 | FAIL |
| H6 real bootstrap vs best rival (OB) | mean diff ≥ 0.05, both 95% CIs > 0 | -0.270 [-0.335, -0.206] (pct [-0.335, -0.209]), p=4.08e-13 | FAIL (TRACE less stable) |
| H6 real split_half vs best rival (OB) | mean diff ≥ 0.05, both 95% CIs > 0 | -0.269 [-0.377, -0.161] (pct [-0.373, -0.165]), p=7.54e-06 | FAIL (TRACE less stable) |
| H7 recovery skew_struct|n=2000 | ≥ 0.6 | 0.540 (50 reps) | FAIL |
| H7 recovery skew_struct|n=20000 | ≥ 0.8 | 1.000 (20 reps) | PASS |
| H7 recovery tail_struct|n=2000 | ≥ 0.6 | 0.120 (50 reps) | FAIL |
| H7 recovery tail_struct|n=20000 | ≥ 0.8 | 0.600 (20 reps) | FAIL |
| H7 recovery skew_comp|n=2000 | ≥ 0.6 | 0.000 (50 reps) | FAIL |
| H7 recovery skew_comp|n=20000 | ≥ 0.8 | 0.000 (20 reps) | FAIL |
| H7 recovery tail_comp|n=2000 | ≥ 0.6 | 0.000 (50 reps) | FAIL |
| H7 recovery tail_comp|n=20000 | ≥ 0.8 | 0.000 (20 reps) | FAIL |
| H7 recovery shape_mixed|n=2000 | ≥ 0.6 | 0.000 (50 reps) | FAIL |
| H7 recovery shape_mixed|n=20000 | ≥ 0.8 | 0.000 (20 reps) | FAIL |
| H7 rival test skew_struct vs DFL (err diff, n=20k) | CI < 0 | 0.000164 [-9.57e-05, 0.000424] (pct [-7.4e-05, 0.0004]), p=0.202 | no win |
| H7 rival test tail_struct vs DFL (err diff, n=20k) | CI < 0 | -0.000154 [-0.000399, 9.05e-05] (pct [-0.000368, 7.44e-05]), p=0.203 | no win |
| H7 rival test skew_comp vs CFM (err diff, n=20k) | CI < 0 | 0.071 [0.069, 0.074] (pct [0.069, 0.073]), p=4.24e-23 | no win |
| H7 rival test tail_comp vs CFM (err diff, n=20k) | CI < 0 | 0.005 [0.003, 0.007] (pct [0.003, 0.007]), p=2.22e-05 | no win |
| H7 rival test shape_mixed vs CFM (err diff, n=20k) | CI < 0 | 0.061 [0.058, 0.064] (pct [0.058, 0.064]), p=1.47e-20 | no win |
| H7 rival claim | CI < 0 in ≥ 4 of 5 | 0.0 of 5 | NOT SUPPORTED |
| H8(a) null FPR, joint test, mean channel (200 reps) | one-sided CP upper ≤ 0.09 (x ≤ 11) | x = 13/192, CP upper 0.105 | FAIL |
| H8(b) coverage | pooled ≥ 0.90, every cell ≥ 0.85 | pooled 0.807, min cell 0.087 | FAIL |
| H6 synthetic bootstrap / split-half | as above | not run (D13) | INCOMPLETE |
| **H6 overall** | all four pass | real arm: both intervals below 0 (recorded failures); synthetic arm not run | **FAIL** (synthetic arm INCOMPLETE) |
| **H7 overall** | every recovery bar and >= 4/5 rival wins | 1/10 recovery cells pass; 0.0/5 rival wins | **FAIL** |
| H8(a) partial-data note | frozen bar defined at n = 200 | x = 13 of 192; x already exceeds 11, so the n = 200 bar fails whatever the 8 missing replications show | FAIL (partial, outcome determined) |
| H8(c) placebo (secondary) | <= 0.10 | not run (D13) | INCOMPLETE |
| Scale (frozen track, TRACE) | runtime/memory caps | not run (D13) | INCOMPLETE |

## Holm (family of tests with p-values, D9)

| test | p | Holm threshold | Holm reject | bar met |
|---|---|---|---|---|
| H7 skew_comp | 4.24e-23 | 0.006 | yes | no |
| H7 shape_mixed | 1.47e-20 | 0.007 | yes | no |
| H6 real bootstrap | 4.08e-13 | 0.008 | yes | no |
| H6 real split_half | 7.54e-06 | 0.010 | yes | no |
| H7 tail_comp | 2.22e-05 | 0.013 | yes | no |
| H8a TRACE mean-channel FPR | 0.171 | 0.017 | no | no |
| H7 skew_struct | 0.202 | 0.025 | no | no |
| H7 tail_struct | 0.203 | 0.050 | no | no |

## Details

### H5

Replications: 1175. Worst relative gaps: g51a 1.36e-16, g51b 3.6e-15, g52 0.030, g52_internal 5.47e-16, g53_chan 3.51e-17, g53_coef 6.24e-17, g54 2.2e-17.

Failure examples (scenario, n, p, rep, gap): {"g52": [["null", 2000, 5, 0, 0.00045085791523342446], ["null", 2000, 5, 2, 0.0004050271838891978], ["null", 2000, 5, 1, 0.0008414928168688201], ["null", 2000, 5, 3, 0.0006760099006290106], ["null", 2000, 5, 6, 0.0004612317892436265], ["null", 2000, 5, 5, 0.0004344970423600606], ["null", 2000, 5, 4, 0.00016573979307668227], ["null", 2000, 5, 7, 3.082089348857824e-05], ["null", 2000, 5, 8, 0.0004903695813168029], ["null", 2000, 5, 10, 0.00028575828996852326]], "g51b": [], "g51a": [], "g52_internal": [], "g53_chan": [], "g53_coef": [], "g54": []}

Rival contrast (median / max, scaled by sd(y_A); order gap = A-reference vs B-reference):

| method | stat | additivity median | order median | order max |
|---|---|---|---|---|
| OB | mean | 5.24e-16 | 0.003 | 0.369 |
| DFL | mean | 0.000 | 0.009 | 0.234 |
| DFL | sd | 0.000 | 0.008 | 0.313 |
| DFL | q10 | 0.000 | 0.016 | 0.454 |
| DFL | q50 | 0.000 | 0.011 | 0.155 |
| DFL | q90 | 0.000 | 0.019 | 0.575 |
| CFM | mean | 0.002 | 0.005 | 0.079 |
| CFM | sd | 0.007 | 0.005 | 0.062 |
| CFM | q10 | 0.005 | 0.014 | 0.169 |
| CFM | q50 | 0.002 | 0.008 | 0.152 |
| CFM | q90 | 0.005 | 0.019 | 0.197 |
| RIF | mean | 4.13e-15 | 0.004 | 0.083 |
| RIF | sd | 2.9e-15 | 0.045 | 0.236 |
| RIF | q10 | 5.73e-15 | 0.038 | 0.383 |
| RIF | q50 | 5.99e-15 | 0.009 | 0.131 |
| RIF | q90 | 1.89e-14 | 0.047 | 0.400 |
| SHAP | mean | 0.000514 | 0.009 | 0.089 |

### H6 means

* real bootstrap: TRACE 0.504, OB 0.774, RIF 0.513, SHAP 0.564
* real split_half: TRACE 0.186, OB 0.455, RIF -0.003, SHAP 0.380
* real undefined Spearman counts (set to 0, D11): {'TRACE': 0, 'OB': 0, 'RIF': 0, 'SHAP': 0}

### H7 errors (graded channel, L-moment scale)

| scenario | TRACE | DFL | CFM | planted magnitude |
|---|---|---|---|---|
| skew_struct|n=2000 | 0.014 | 0.014 | 0.027 | 0.051 |
| skew_struct|n=20000 | 0.005 | 0.005 | 0.017 | 0.051 |
| tail_struct|n=2000 | 0.010 | 0.010 | 0.011 | 0.010 |
| tail_struct|n=20000 | 0.002 | 0.003 | 0.006 | 0.010 |
| skew_comp|n=2000 | 0.078 | 0.076 | 0.022 | 0.040 |
| skew_comp|n=20000 | 0.083 | 0.082 | 0.012 | 0.040 |
| tail_comp|n=2000 | 0.012 | 0.012 | 0.011 | 0.005 |
| tail_comp|n=20000 | 0.009 | 0.010 | 0.004 | 0.005 |
| shape_mixed|n=2000 | 0.082 | 0.083 | 0.028 | 0.093 |
| shape_mixed|n=20000 | 0.080 | 0.080 | 0.019 | 0.093 |

### H8

| statistic | rejections | FPR | one-sided CP upper | bar |
|---|---|---|---|---|
| TRACE_mean_aref | 13/192 | 0.068 | 0.105 | fail |
| TRACE_mean_orderfree | 15/192 | 0.078 | 0.118 | fail |
| TRACE_channels | 9/192 | 0.047 | 0.080 | pass |
| TRACE_features | 9/192 | 0.047 | 0.080 | pass |
| OB_mean | 14/192 | 0.073 | 0.112 | fail |
| OB_features | 1/192 | 0.005 | 0.024 | pass |
| DFL_mean | 13/192 | 0.068 | 0.105 | fail |
| CFM_mean | 15/192 | 0.078 | 0.118 | fail |
| RIF_mean | 15/192 | 0.078 | 0.118 | fail |
| RIF_features | 10/192 | 0.052 | 0.087 | pass |
| SHAP_mean | 14/192 | 0.073 | 0.112 | fail |
| SHAP_features | 4/192 | 0.021 | 0.047 | pass |

McNemar vs rivals (mean channel): {"OB": {"trace_only": 2, "rival_only": 3, "p_exact": 1.0}, "DFL": {"trace_only": 2, "rival_only": 2, "p_exact": 1.0}, "CFM": {"trace_only": 1, "rival_only": 3, "p_exact": 0.625}, "RIF": {"trace_only": 2, "rival_only": 4, "p_exact": 0.6875}, "SHAP": {"trace_only": 3, "rival_only": 4, "p_exact": 1.0}}

Frozen band design null (40 reps): TRACE union-rule FPR 0.175, joint-test FPR 0.100. Rivals (frozen union rule, existing records): {"mean_coverage": {"OB": 0.9, "DFL": 0.896875, "CFM": 0.903125, "RIF": 0.90625, "SHAP": 0.89375}, "null_union_fpr_mean": {"OB": 0.175, "DFL": 0.175, "CFM": 0.2, "RIF": 0.2, "SHAP": 0.1}}

Coverage cells: mixed|mean 0.863, mixed|sd 0.938, mixed|q10 0.925, mixed|q50 0.963, mixed|q90 0.863, tail|mean 0.900, tail|sd 0.887, tail|q10 0.887, tail|q50 0.812, tail|q90 0.787, hetero|mean 0.925, hetero|sd 0.925, hetero|q10 0.850, hetero|q50 0.900, hetero|q90 0.963, entry_exit|mean 0.887, entry_exit|sd 0.500, entry_exit|q10 0.850, entry_exit|q50 0.875, entry_exit|q90 0.725, skew_struct|l1 0.863, skew_struct|l2 0.925, skew_struct|l3 0.912, skew_struct|l4 0.925, tail_struct|l1 0.900, tail_struct|l2 0.838, tail_struct|l3 0.988, tail_struct|l4 0.900, skew_comp|l1 0.912, skew_comp|l2 0.850, skew_comp|l3 0.087, skew_comp|l4 0.400, tail_comp|l1 0.900, tail_comp|l2 0.875, tail_comp|l3 0.950, tail_comp|l4 0.613, shape_mixed|l1 0.900, shape_mixed|l2 0.900, shape_mixed|l3 0.113, shape_mixed|l4 0.300

## Frozen ShiftBench tracks with TRACE (A-reference split, D4)

**n=2000,p=5**: mean-channel error OB 0.029, DFL 0.023, CFM 0.019, RIF 0.019, SHAP 0.021, TRACE 0.022; per-feature error OB 0.073, RIF 0.110, SHAP 0.068, TRACE 0.256; TRACE order-free vs order-free truth (mean) 0.020.
  Paired TRACE − rival (mean channel): OB -0.006 [-0.008, -0.005] (pct [-0.007, -0.005]), p=3.53e-17; DFL -0.000315 [-0.001, 0.000835] (pct [-0.001, 0.000735]), p=0.584; CFM 0.003 [0.002, 0.004] (pct [0.002, 0.004]), p=7.43e-10; RIF 0.003 [0.002, 0.004] (pct [0.002, 0.004]), p=4.94e-10; SHAP 0.001 [0.000337, 0.002] (pct [0.000363, 0.002]), p=0.004
**n=20000,p=20**: mean-channel error OB 0.017, DFL 0.009, CFM 0.007, RIF 0.006, SHAP 0.010, TRACE 0.013; per-feature error OB 0.039, RIF 0.090, SHAP 0.047, TRACE 0.213; TRACE order-free vs order-free truth (mean) 0.010.
  Paired TRACE − rival (mean channel): OB -0.005 [-0.005, -0.004] (pct [-0.005, -0.004]), p=5.54e-12; DFL 0.004 [0.002, 0.006] (pct [0.001, 0.005]), p=0.003; CFM 0.006 [0.005, 0.007] (pct [0.006, 0.007]), p=1.03e-14; RIF 0.006 [0.006, 0.007] (pct [0.006, 0.007]), p=7.63e-15; SHAP 0.003 [0.003, 0.003] (pct [0.003, 0.003]), p=2.49e-12
**n=200000,p=20**: mean-channel error OB 0.014, DFL 0.003, CFM 0.003, RIF 0.002, SHAP 0.004, TRACE 0.011; per-feature error OB 0.029, RIF 0.034, SHAP 0.041, TRACE 0.197; TRACE order-free vs order-free truth (mean) 0.008.
  Paired TRACE − rival (mean channel): OB -0.003 [-0.003, -0.002] (pct [-0.003, -0.003]), p=2.6e-05; DFL 0.008 [0.007, 0.008] (pct [0.007, 0.008]), p=6.94e-06; CFM 0.008 [0.007, 0.008] (pct [0.008, 0.008]), p=6.13e-07; RIF 0.009 [0.008, 0.009] (pct [0.009, 0.009]), p=1.68e-07; SHAP 0.007 [0.006, 0.007] (pct [0.006, 0.007]), p=2.51e-06

## Scale (TRACE)

null

Rival scale cells are missing: the ShiftBench run script stopped before its scale track (D10).

## Reading (written after scoring; does not change any verdict)

* **H5.** Everything algebraic held to machine precision: the independent barycenter reconstruction
  (H5.1b, worst 3.6e-15), channel and coefficient identities, period swap, and row/column permutation
  (1.1e-16). The failure is H5.2 under D8: TRACE's marginal pieces sum exactly to its *own* marginal
  W2² (5e-16) but differ from the harness's empirical `mean((sort y_B − sort y_A)²)` by up to 3% relative,
  so TRACE's marginal quantile functions are not the empirical ones (most likely its 64-node
  quadrature and smoothing of Q). This is a real failure of the pre-registered claim, not a scoring quirk.
* **H6.** On the real panel TRACE's per-feature rankings are less stable than OB's (bootstrap 0.50 vs
  0.77; split-half 0.19 vs 0.46). Recorded failure. The synthetic arm was not run.
* **H7.** TRACE's marginal counterfactual tracks DFL almost exactly (both reweight). Both recover
  pure structure shape changes (`skew_struct` at n = 20k) but not composition-driven shape changes,
  where CFM is far more accurate (e.g. `skew_comp` error 0.083 vs 0.012). Note that TRACE is scored
  against the order-free truth and CFM against the A-reference truth (D12).
* **H8.** The joint max-|t| test with B = 100 over-rejects for *every* method at n = 2,000
  (6.8%–7.8% on the mean channel), so the failure is mostly the test's calibration, not TRACE in
  particular (McNemar vs each rival: no difference). Coverage fails mainly on composition-driven
  shape channels (`skew_comp` λ₃ 0.09, `shape_mixed` λ₃ 0.11) and `entry_exit` sd (0.50).
* **Frozen tracks.** TRACE's mean-channel split error is mid-pack (n = 20k: 0.0128 vs 0.0064–0.0175).
  Its per-feature error is the largest (0.21 vs 0.04–0.09). Part of that may be an estimand mismatch:
  TRACE's per-feature pieces are order-free barycenter pieces, the frozen per-feature truth is the
  A-reference marginal split, and the adapter has no A-reference per-feature output. This was not
  separated out and is not claimed as an excuse.
