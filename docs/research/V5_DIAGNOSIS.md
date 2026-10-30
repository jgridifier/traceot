# TRACE v5 first-gate failure — diagnosis (EXPLORATORY)

**Status: EXPLORATORY.** Not a gate, not a scored result, no claim. Run only on the
dev indices already spent by the failed first gate (9500–9549, n = 2000,
`skew_comp` / `tail_comp`). Indices 9600–9619 were **not** touched. Script:
[`benchmarks/shiftbench/diag_v5_gate1.py`](../../benchmarks/shiftbench/diag_v5_gate1.py);
output: [`benchmarks/results/v5_diag_gate1.json`](../../benchmarks/results/v5_diag_gate1.json).
Failed gate of record: [V5_GATE1_RESULTS.md](../benchmark/V5_GATE1_RESULTS.md) (prereg `bce021e`).

## Question

v5 = (fitted conditional \(F_t(y\mid x)\)) + (average CDFs on a 256-point grid, invert,
read L-moments). Which part produces the error that failed the gate?

## Method

Same seeds, same data, same paired best rival (min of DFL/CFM per replication), same
`cell_verdict` (PREREG_V4 §5 protocol). Only the conditional is swapped; the mixture is
always built with the Rust `mixture_cdf_location` on a 256-point grid and inverted with
`mixture_quantile`.

| variant | conditional used |
|---|---|
| V5 | recorded v5 output (binned mean booster + empirical residual law); re-run reproduces it exactly |
| ORACLE | true \(F_t(y\mid x)\): true mean \(\mu(x)\) + true noise law of period \(t\) |
| ORACLE_EXACT | as ORACLE, but all \(n^2\) location+residual pairs, no grid (isolates grid/inversion error) |
| ORACLE_MU | true \(\mu(x)\) + the period's empirical residuals |
| OLS | estimated linear least-squares mean + its empirical residuals (no oracle information) |

## Results (50 replications per cell)

`skew_comp` (truth comp 0.04011, struct −0.00014; δ = 0.004025; best rival err 0.02218)

| variant | err | mean d | upper 95% | verdict | \|ĉ\| | \|ŝ\| | bias ĉ | bias ŝ |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| V5 | 0.02699 | +0.00481 | 0.00711 | FAIL | 0.0262 | 0.0129 | −0.0139 | +0.0131 |
| ORACLE | 0.00681 | −0.01537 | −0.01304 | BEST | 0.0408 | 0.0003 | +0.0007 | −0.0001 |
| ORACLE_EXACT | 0.00681 | −0.01537 | −0.01303 | BEST | 0.0408 | 0.0003 | +0.0007 | −0.0001 |
| ORACLE_MU | 0.00840 | −0.01377 | −0.01147 | BEST | 0.0408 | 0.0018 | +0.0007 | −0.0003 |
| OLS | 0.00891 | −0.01326 | −0.01088 | BEST | 0.0413 | 0.0023 | +0.0012 | −0.0003 |

`tail_comp` (truth comp 0.004298, struct 0.000288; δ = 0.000459; best rival err 0.00972)

| variant | err | mean d | upper 95% | verdict | \|ĉ\| | \|ŝ\| | bias ĉ | bias ŝ |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| V5 | 0.00971 | −0.00001 | 0.00164 | FAIL (role FAIL) | 0.0031 | 0.0049 | −0.0046 | +0.0044 |
| ORACLE | 0.00341 | −0.00631 | −0.00487 | BEST | 0.0047 | 0.0005 | −0.0000 | +0.0002 |
| ORACLE_EXACT | 0.00342 | −0.00631 | −0.00486 | BEST | 0.0048 | 0.0005 | −0.0000 | +0.0002 |
| ORACLE_MU | 0.00430 | −0.00542 | −0.00404 | BEST | 0.0047 | 0.0012 | −0.0000 | +0.0004 |
| OLS | 0.00496 | −0.00476 | −0.00326 | BEST | 0.0048 | 0.0018 | −0.0000 | +0.0008 |

## Decomposition of v5 error

| piece | skew_comp | tail_comp |
|---|---:|---:|
| v5 total | 0.02699 | 0.00971 |
| grid / inversion (\|ORACLE − ORACLE_EXACT\|) | 0.000005 | 0.000001 |
| floor with the true conditional (ORACLE) | 0.00681 | 0.00341 |
| conditional-model error (V5 − ORACLE) | **0.02018 (75%)** | **0.00629 (65%)** |
| … of which residual-law sampling (ORACLE_MU − ORACLE) | 0.00160 | 0.00089 |
| … of which the mean model (V5 − OLS) | **0.01807** | **0.00475** |

## Reading

1. **The mixture object is not the problem.** With the true conditional, the exact same
   Eulerian construction is BEST in both cells with the role check passing. Grid and
   inversion error is ~10⁻⁶ — three orders below δ.
2. **The suspected cause (homoskedastic, shared residual shape) is not supported.** The
   simulator is itself homoskedastic with one noise law per period, and ORACLE uses
   exactly that assumption and is BEST. Rank correlation of |residual| with the fitted
   mean is 0.000 / −0.004 (`skew_comp` A/B) and 0.001 / 0.006 (`tail_comp`). A
   heteroskedastic or location-scale-shape model has nothing to fix here.
3. **The failure is the mean (location) model.** v5's bias is a near-equal and opposite
   transfer from composition to response (skew: −0.0139 / +0.0131; tail: −0.0046 /
   +0.0044). That is the pattern of a mean model that does not carry the covariate
   tails of period B: the hybrid "covariates B, conditional A" uses period A's fit on B's
   heavy-tailed / skewed covariate, where a binned booster trained on Gaussian A
   covariates is flat beyond its bins. Mass that belongs to composition is then booked
   to response. Replacing only the mean fit with ordinary least squares removes the
   bias (|bias ĉ| ≤ 0.0012) and both cells become BEST with the role check passing.
   The flat-extrapolation mechanism is a hypothesis consistent with these numbers; it
   was not isolated separately here.

## Limits

* Exploratory, on spent seeds; the OLS variant was chosen after seeing v5 fail. It is
  evidence for a narrower pre-registered test on fresh seeds, not a result.
* The simulator's mean is linear, so OLS is the correct family by construction. A pass
  with a linear location term would not show robustness to a non-linear mean.
* n = 20 000 was not examined.

Next step: [PREREG_V5_1.md](PREREG_V5_1.md).
