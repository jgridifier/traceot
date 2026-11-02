# Pre-registration: TRACE v5.2 — curved (non-linear) means (one hypothesis)

**Frozen:** 2026-11-02 (America/New_York), on branch `quant/v5-2-prereg`, **before any
v5.2 code exists and before anything has been run on the seeds below**. The merge commit
that adds this file to `main` is the freeze point; record its SHA in the header of any
later `V5_2_GATE_RESULTS.md`. Any change to the hypothesis, mean model, simulator, cells,
seeds, bars, cap, or kill rule after that SHA is a deviation and must be reported as one.

Prior freeze: [PREREG_V5_1.md](PREREG_V5_1.md) @ `b33f507`, implementation `665939b`,
gate **PASS** on all four cells
([V5_1_GATE_RESULTS.md](../benchmark/V5_1_GATE_RESULTS.md), PR #30). That PASS is on a
simulator whose mean is linear by construction (PREREG_V5_1 §7), so it says nothing
about curved means. This file pre-registers the test that would.

Nothing is run under this file. It only fixes what will be run.

---

## 1. Hypothesis (the only one)

**H.** When the true conditional mean is a smooth **non-linear additive** function of the
inputs, `TRACE_V5_2` — the v5.1 mixture with the OLS location fit replaced by a
**fixed additive natural cubic spline basis fitted by least squares** (§2) — is
**BEST or NON-INFERIOR** against the best paired rival (DFL, CFM) on the graded channel
and passes the role check in **every** gate cell of §4: both pure-composition shapes
(`skew_comp`, `tail_comp`) at both curvature levels and both sizes, the linear-control
cells, and the structure-only (zero-composition) cells, where it must also not
attribute the change to composition (§5).

Explicitly **not** tested: interactions between inputs in the mean, non-smooth means
(jumps, kinks), heteroskedastic or shape-varying conditionals, per-feature shares,
ShiftBench, speed. One hypothesis, one method, one gate.

## 2. Mean model under test (fixed now, no tuning)

`TRACE_V5_2` = `explain_v5(..., mean_model="spline")` with exactly one change from
`TRACE_V5_1` @ `665939b`: the per-period location fit.

* **Basis.** For each input column \(j=1..p\), a restricted (natural) cubic spline in
  the truncated-power form of Harrell (2015, §2.4.5) with **K = 5 knots** placed at the
  **5%, 27.5%, 50%, 72.5%, 95% quantiles** of column \(j\) computed on the **pooled**
  covariates \(x_A\cup x_B\) (so both periods share one basis). Each column contributes
  \(K-1=4\) basis functions (the linear term plus 3 non-linear terms); the model is
  additive (no cross terms). Design: intercept + \(4p\) columns (21 at p = 5, 81 at
  p = 20).
* **Fit.** Per period \(t\in\{A,B\}\), ordinary least squares of \(y\) on that design:
  no penalty, no smoothing parameter, no feature selection, no data-driven choice of K.
  Solved by a pivoted QR (or equivalent) in the Rust core; if a basis column is
  numerically rank-deficient (tied quantiles), drop it by the QR pivot rule and record
  it — that is a fixed rule, not a tuning step.
* **Residual law and everything downstream** are unchanged from v5.1: empirical law of
  \(r_t=y-\hat\mu_t(x)\), four hybrids \([A,A],[A,B],[B,A],[B,B]\) via
  `mixture_cdf_location`, 256-point grid, `n_quad = 64`, exact L-moments of the gridded
  law, two-player Shapley split.
* Rust core, thin Python switch, no neural nets.

**Why this model, and not the alternatives.**

1. *It is tuning-free and fully specified in advance.* The knot count and placement are
   a published fixed rule: Harrell recommends K = 5 as a good default for large samples
   with quantile placement (5, 27.5, 50, 72.5, 95)% and notes that knot *location* matters
   little once the count is fixed (Harrell 2015, *Regression Modeling Strategies*, 2nd ed.,
   §2.4.6; Stone & Koo 1985, *Proc. ASA Stat. Comput.*).
   Our smallest cell has
   n = 2000 rows per period for 21 parameters, so K = 5 is well inside its range.
2. *It nests v5.1.* Dropping the three non-linear terms per column gives exactly the
   v5.1 OLS fit, so on a linear mean the only cost is estimation variance from
   \(3p\) extra coefficients. The linear-control cells (§4) measure that cost.
3. *Additive regression is the standard low-dimensional answer to smooth curvature.*
   Additive models avoid the curse of dimensionality: the optimal rate for an additive
   smooth mean is the one-dimensional rate (Stone 1985, *Ann. Statist.* 13:689–705);
   regression splines are the classical estimator of each component (Hastie &
   Tibshirani 1990, *Generalized Additive Models*, ch. 2–4; Wood 2017, *Generalized
   Additive Models*, 2nd ed., §5.3). An unpenalised fixed-knot basis is chosen over a
   penalised smoother specifically so that no smoothing parameter (GCV/REML) has to be
   selected from the data.
4. *Not the v4/v5 booster with frozen defaults.* That booster was the v5 location model,
   and the v5 first gate failed because of it ([V5_DIAGNOSIS.md](V5_DIAGNOSIS.md),
   exploratory; [V5_GATE1_RESULTS.md](../benchmark/V5_GATE1_RESULTS.md)). Its binned,
   piecewise-constant fit is biased in exactly the low-density tails that composition
   shifts push mass into, and re-testing it would mostly re-test a known failure.
5. *Natural (linear-beyond-boundary) splines extrapolate linearly* outside the 5% and
   95% knots, which is the conservative choice when period B moves covariate mass into a
   tail (`skew_comp` shifts column 0 to a skewed law). This is also a known risk; see §9.

## 3. Simulator extension (curved means at pre-stated levels)

Based on the frozen H7 shape generator `benchmarks/shiftbench/sb_dgp_shape.py`
(`ShapeScenario`), which must not be modified; the extension is a new module that
imports it. Only the mean changes:

\[
\mu_\kappa(X) \;=\; 1 + X_{1:5}^\top\beta \;+\; \kappa\, g(X),\qquad
g(X) \;=\; \tfrac{1}{\sqrt2}\,(X_1^2-1) \;+\; \sqrt2\,\sin(X_3),
\]

with \(\beta=(1,-0.8,0.6,0.4,-0.3)\) as frozen, columns 1-indexed (so \(X_1\) is code
column 0, the column `skew_comp` replaces with a χ²₃-type law; \(X_3\) is code column 2,
which no scenario shifts). \(g\) is additive, smooth, and has mean 0 under the period-A
law (\(X\sim N(0,I)\)): \(E[X_1^2-1]=0\), \(E[\sin X_3]=0\).

* **Curvature levels (pre-stated):** \(\kappa\in\{0,\;0.5,\;1.0\}\). \(\kappa=0\) is
  the **linear control** (identical to the v5.1 simulator). \(\kappa=1.0\) puts
  \(\mathrm{Var}(g)= 1 + 2\cdot\tfrac{1}{2}(1-e^{-2}) \approx 1.86\) of curved signal
  next to \(\mathrm{Var}(X^\top\beta)=2.25\) under period A, i.e. the curve is not a
  small perturbation.
* **Noise and calibration:** unchanged laws per scenario (`SPEC`). The period-B noise
  scale \(\sigma_B\) is re-calibrated per (scenario, κ) with the frozen procedure
  (`calibrate`, oracle 2·10⁶ draws, oracle seed 99, common random numbers) so that
  \(\lambda_2(Y_B)=\lambda_2(Y_A)\). If calibration is infeasible for a (scenario, κ)
  (no \(\sigma_B>0\) solves it), that cell is **INCOMPLETE** — it is not re-designed.
* **Truth:** Shapley composition/structure split of the graded channel computed by the
  same oracle code into a new file
  `benchmarks/results/shiftbench_truth_shape_v5_2.json`. Truth is computed from the
  oracle seed only, never from gate seeds, and is committed before any gate run. The
  planted magnitude \(|{\rm truth}_c|+|{\rm truth}_s|\) for each cell is read from that
  file and is the basis of δ (§5).
* **Structure-only scenarios** `skew_struct`, `tail_struct`: \(X\) has the same law in
  both periods (the composition truth is exactly 0), only the noise law changes. With
  \(\kappa>0\) the curved mean is present in both periods, which is precisely where a
  mis-fit location model could leak structure into a spurious composition term.

## 4. Gate cells and seeds

Rivals DFL and CFM re-run paired on the same data; estimand, graded channel (λ₃ for
skew, λ₄ for tail), `seed_of(...)` streams as in `run_v5_1_gate.py`, with the scenario
key extended by κ (e.g. `seed_of("data", "skew_comp|k0.5", n, p, rep)`).

| # | family | scenario | κ | n | p | indices | reps |
|---|---|---|---|---:|---:|---|---:|
| 1–4 | curved composition | `skew_comp`, `tail_comp` | 0.5, 1.0 | 2000 | 5 | **10100–10149** | 50 |
| 5–8 | curved composition | `skew_comp`, `tail_comp` | 0.5, 1.0 | 20 000 | 20 | **10200–10219** | 20 |
| 9–10 | linear control | `skew_comp`, `tail_comp` | 0 | 2000 | 5 | **10100–10149** | 50 |
| 11–12 | linear control | `skew_comp`, `tail_comp` | 0 | 20 000 | 20 | **10200–10219** | 20 |
| 13–14 | structure only | `skew_struct`, `tail_struct` | 1.0 | 2000 | 5 | **10100–10149** | 50 |
| 15–16 | structure only | `skew_struct`, `tail_struct` | 1.0 | 20 000 | 20 | **10200–10219** | 20 |

16 cells: 8 cells × 50 + 8 cells × 20 = **560 tasks**. All cells
are inside the gate; none is optional. The same index range is reused across cells of
one size, but the data stream differs per cell because the scenario key (including κ)
enters `seed_of`, exactly as v5.1 did across its two scenarios.

**Seed freshness (proof).** At the freeze, on the commit before this file, from the repo
root:

```
git grep -E -c '(^|[^0-9.])101[0-4][0-9]([^0-9]|$)' | wc -l   # -> 0
git grep -E -c '(^|[^0-9.])102[01][0-9]([^0-9]|$)'  | wc -l   # -> 0
```

Both return 0: no runner, results file, JSONL record, pre-registration, or learn page
mentions any index in 10100–10149 or 10200–10219. (After this commit the only hits are
this file and its learn page.)

**Not used, and why.** 9100–9429: v4. 9500–9549: spent (V5 gate, and V5.1's hypothesis
was chosen by looking at them). 9600–9619: sealed (§7). 9700–9749, 9800–9819: used by
the V5.1 gate.

**Secondary, descriptive only (not gated, no Holm, cannot rescue or sink H):**
`TRACE_V5_1` (OLS) on cells 1–8 and 13–16, reported alongside, to show how much the
linear fit loses under curvature.

## 5. Bars — inherited unchanged from PREREG_V4 §5 (as scored for V5.1)

Per cell, paired \(d_r=\mathrm{err}_{\mathrm{v5.2},r}-\mathrm{err}_{\mathrm{best\,rival},r}\)
(best rival = min of DFL, CFM per replication). One-sided 95% upper bound of mean \(d\) by
**Student-t with n−1 df** (as the V5.1 scorer, V5_1_GATE_RESULTS note 2) **and** a
10 000-draw percentile bootstrap (seed 2026); both must agree, the larger bound is used.

* **BEST:** upper 95% bound \(<0\).
* **NON-INFERIOR:** upper 95% bound \(<\delta\), \(\delta=0.10\times\) planted magnitude
  of the graded channel for that cell (from the v5.2 truth file, §3).
* **Role check, composition and control cells (1–12)**, as defined in
  V5_GATE1_RESULTS note 2 and applied in V5.1: mean \(|\hat c|>\) mean \(|\hat s|\) on
  the graded channel, **and** swapping the two roles does not lower the error.
* **Role check, structure-only cells (13–16)** — the same definition with the roles
  mirrored (the true dominant role is structure): mean \(|\hat s|>\) mean \(|\hat c|\),
  and swapping does not lower the error.
* **False-attribution check, structure-only cells (13–16):** the one-sided 95% upper
  bound (t and bootstrap, larger used) of mean \(|\hat c_r|\) on the graded channel is
  \(<\delta\). This is not a new error metric; it is the NON-INFERIOR margin applied to
  the composition piece whose truth is 0.
* **Multiplicity:** Holm across **all 16** paired accuracy tests (and, separately,
  across the 4 false-attribution tests, which are also listed with unadjusted values).
  Unadjusted values always reported.
* **Gate PASS:** all 16 cells BEST or NON-INFERIOR after Holm, all 16 role checks pass,
  all 4 false-attribution checks pass, and the run finishes within the cap.

**Coverage.** `explain_v5` returns point splits only; **no intervals or bands exist** for
v5.1 or v5.2 at this freeze. Coverage is therefore **secondary and not part of the
gate**. If bands are added later, their coverage needs its own pre-registration (with the
PREREG_V4 coverage bar: pooled ≥ 0.90, every cell ≥ 0.85).

## 6. Runtime cap and kill rule

* **Cap:** single pass over all 560 tasks (n = 20 000 cells scheduled first) on 8
  workers, wall cap **30 minutes** (as PREREG_V5 §8 / V5.1). No task dropped, retried or
  re-run; unfinished cells are INCOMPLETE and never imputed.
* **Kill rule:** if **any** cell fails accuracy, fails a role check, fails the
  false-attribution check, or is INCOMPLETE, H is rejected. Then: record every failure
  in `V5_2_GATE_RESULTS.md`, **park the curved-mean claim** (write `PARK_V5_2.md`), and
  **keep V5.1's linear-mean claim exactly as stated** in V5_1_GATE_RESULTS (a v5.2 FAIL
  neither strengthens nor weakens it). No amendment on these cells, no new knot rule, no
  seed changes, no re-runs to chase a bar.

## 7. V5.1 confirmation and the sealed holdout

* **9500–9549 cannot confirm anything.** They decided the failed V5 gate and V5.1's
  hypothesis was chosen by looking at them; any result on them is exploratory by
  construction. They are not used here.
* The V5.1 claim is already confirmed once, on fresh seeds (9700–9749, 9800–9819). The
  linear-control cells 9–12 here are an incidental second look at a linear mean, but with
  a different estimator (spline, not OLS) and are not a V5.1 replication.
* **Decision: 9600–9619 stays sealed.** It is the final holdout for the eventual **full
  claim** (the method frozen after the curved-mean question is settled, run once across
  the full scored suite under its own pre-registration). Nothing in this freeze, or in
  any amendment of it, may read, run, or tune on 9600–9619. Recommendation: keep it
  sealed until that full-claim pre-registration exists and names it; spending it now on a
  partial question would leave no untouched seeds for the claim that matters.

## 8. No tuning, no peeking

* No change to §2 after this freeze; no tuning on any seed (gate, spent, or sealed).
* Order of work: (1) implement `mean_model="spline"` and the simulator module, with unit
  tests on **non-gate** seeds only (small fixed seeds below 1000, as the fixtures in
  `tests/`); (2) compute and commit the v5.2 truth file (oracle seed 99); (3) merge;
  (4) record the implementation merge SHA; (5) only then run 10100–10149 and
  10200–10219 once.

## 9. Limitations (stated in advance)

1. **The family is right by construction again, partly.** \(g\) is additive and smooth,
   so an additive spline is close to the correct family. A PASS shows robustness to
   smooth additive curvature only — not to interactions, kinks, or jumps.
2. **\(X_1^2\) is not in the span of a natural spline.** Beyond the 95% knot the basis is
   linear while the truth is quadratic; under `skew_comp` period B pushes column 0 into
   a long right tail. Some extrapolation bias is expected there; this is a genuine part
   of the test, not a defect to be fixed after the fact.
3. **Variance cost.** 4p coefficients instead of p+1; at p = 20 and n = 20 000 this is
   small, at p = 5 and n = 2000 it is the main risk on the linear-control cells.
4. **Two curvature levels, one curve shape.** κ ∈ {0.5, 1.0} and one \(g\); the result
   does not generalise to other curve shapes without new seeds.
5. **No intervals.** Coverage is not assessed (§5).
6. **Same simulator family as v4–v5.1** (Gaussian inputs in A, five active columns,
   homoskedastic noise). Real-data behaviour is out of scope.
7. **Rivals are unchanged** (DFL, CFM with their existing fixed settings); a stronger
   rival under curvature would be a different test.

## 10. Report

`docs/benchmark/V5_2_GATE_RESULTS.md` with this file's freeze SHA, the implementation
SHA, the 16-cell table (err v5.2, err DFL, err CFM, mean d, upper t, upper boot, δ,
verdict, |ĉ|, |ŝ|, role, false-attribution bound for 13–16, p NI / p BEST with Holm),
the secondary v5.1 column, wall time, and every failure kept. No result numbers appear
in this file.

Learn page: [v5_2_prereg.html](../learn/v5_2_prereg.html).

### References

* Harrell, F. E. (2015). *Regression Modeling Strategies*, 2nd ed. Springer. §2.4.5–2.4.6
  (restricted cubic splines; fixed knot-count and quantile placement rule).
* Stone, C. J., & Koo, C.-Y. (1985). Additive splines in statistics. *Proc. Statistical
  Computing Section, ASA*, 45–48.
* Stone, C. J. (1985). Additive regression and other nonparametric models. *Annals of
  Statistics* 13(2), 689–705.
* Hastie, T., & Tibshirani, R. (1990). *Generalized Additive Models*. Chapman & Hall.
* Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*, 2nd ed. CRC.
* Hosking, J. R. M. (1990). L-moments. *JRSS B* 52(1), 105–124.
