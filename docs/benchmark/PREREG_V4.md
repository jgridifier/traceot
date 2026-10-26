# Pre-registration: TRACE v4

**Frozen:** 2026-10-26 (America/New_York), on branch `quant/v4-spec`, **before any v4
empirical result exists**. Theory: [V4.md](../theory/V4.md). Derivations:
[V4_DERIVATIONS.txt](../theory/verify/V4_DERIVATIONS.txt) (10 PASS). Implementation
spec: [V4_IMPL_SPEC.md](../theory/V4_IMPL_SPEC.md). Builds on
[PREREG_TRACE_CLAIMS.md](PREREG_TRACE_CLAIMS.md) / Amendment 1 and
[PREREG_TRACE_V3.md](PREREG_TRACE_V3.md) for generators and estimands; bars and seeds
below supersede those documents for the v4 run only.

**State at freeze.** No `explain_v4` / `TRACE_V4` adapter output has been scored. No
file under `benchmarks/results/v4_*` exists at this commit. The commit SHA that adds
*this* file is the freeze point; record that SHA in the header of `V4_RESULTS.md`
when results are written. Any change to scenarios, seeds, margins, or bars after that
SHA is a deviation and must be labelled as such.

Success standard (overall): on accuracy, TRACE v4 is **BEST**, or statistically
**NON-INFERIOR** to the best rival and **≥ 5× faster** than CFM at the speed sizes.

---

## 1. Method under test (fixed now)

* **TRACE v4** as specified in V4.md / V4_IMPL_SPEC: model-mixture target, mixing-player
  game (M1), L² / PIT fitting (M2), split-product + multiplier bootstrap inference (M3,
  \(B=2000\) Rademacher, studentized), bin-marginal raking / Riesz weights (M4). No
  neural nets. Rust core, thin Python bindings.
* Adapter name: `TRACE_V4` (see TRACE_ADAPTER_SPEC.md); entry point
  `trace.explain_v4(...)` returning pieces, bands, timings.
* Rivals: **CFM** and **DFL** only for the paired accuracy and speed comparisons below,
  re-run on the **same seeds** as TRACE v4 (not reused from PR #12 / v3). Other ShiftBench
  rivals may be reported descriptively; they are not in the inferential family.

Any change to the above after this freeze is a deviation (§7).

---

## 2. Scenarios and sizes

Generators unchanged from `benchmarks/` (H7/H8 runner and ShiftBench).

| Track | Scenarios | Size | Notes |
|---|---|---|---|
| H7 shape | `skew_struct`, `tail_struct`, `skew_comp`, `tail_comp`, `shape_mixed` | \(n=2000\) and \(n=20000\) (p as in the frozen H7 grid: 5 at 2e3, 20 at 2e4) | Graded-channel L-moment error vs Shapley truth |
| H8b coverage | Same generators as TRACE_CLAIMS coverage track (`mixed`, `tail`, `hetero`, `entry_exit`, plus the five shape scenarios) | as frozen H8b design (\(n=2000\), \(p=5\)) | Split-product / multiplier bands, not refit percentile |
| ShiftBench accuracy | Main synthetic accuracy track | \(n=20000\), \(p=20\) | Mean-channel and per-feature error |

Oracle truth files unchanged (`benchmarks/results/shiftbench_truth_shape.json`, ShiftBench
truth). Estimands: H7 `lsplit` vs Shapley; coverage as in TRACE_CLAIMS H8(b) but centred
on the v4 mixture pieces; ShiftBench as in PREREG_SHIFTBENCH.

---

## 3. Fresh seeds (exclusive to v4)

Replication indices (never used by PR #12, FIXES2, v3, or the v3 pilot):

| Track | Indices | Count per cell |
|---|---|---|
| H7 shape, \(n=2000\) | **9100–9149** | 50 |
| H7 shape, \(n=20000\) | **9200–9219** | 20 |
| H8b coverage | **9300–9379** | 80 |
| ShiftBench accuracy | **9400–9429** | 30 |

Data / method / bootstrap streams use the frozen `seed_of(...)` of `run_synthetic.py`
evaluated at these indices. CFM and DFL are re-run on the **same** indices; comparisons
are **paired** per replication.

---

## 4. Metrics

* **H7 accuracy:** existing graded-channel error on the L-moment scale (`lsplit` vs
  Shapley truth), as in Amendment 1 / v3.
* **ShiftBench accuracy:** mean-channel error and per-feature error (frozen ShiftBench
  metrics, scaled by \(\mathrm{sd}(y_A)\)).
* **Coverage:** indicator that the v4 band for a cell contains the frozen truth; pooled
  over all coverage indicators; per cell = scenario × stat/channel × role.
* **Speed:** wall time of `explain+bands` vs CFM on the same machine, 8 threads, CFM with
  its default threshold grid, at \(n\in\{10^5,10^6\}\), \(p=20\). CFM bands for timing use
  its own bootstrap \(B=100\) on 3 seeds only.

---

## 5. Bars (Holm across the family, \(\alpha=0.05\))

Per accuracy cell, compute the paired mean error difference
\(d = \mathrm{err}_{\mathrm{v4}} - \mathrm{err}_{\mathrm{best\,rival}}\) and its one-sided
95% interval over replications (t-interval and 10 000-draw percentile bootstrap of the
\(d_r\), seed 2026; both must agree for a claim).

* **BEST** in a cell: upper 95% CI of \(d\) \(< 0\).
* **NON-INFERIOR** in a cell: upper 95% CI of \(d\) \(< \delta\), where
  * H7 / shape cells: \(\delta = 0.10 \times\) (planted magnitude of the graded channel),
  * ShiftBench: \(\delta = 0.002\cdot\mathrm{sd}(y_A)\).
* **Overall accuracy PASS:** every cell is BEST or NON-INFERIOR.
* **Coverage PASS:** pooled coverage \(\ge 0.90\) **and** every cell \(\ge 0.85\).
* **Speed (conditional):** if **any** accuracy cell is only NON-INFERIOR (none of the
  failed-BEST cells is BEST), require wall-time ratio
  \(\mathrm{time}_{\mathrm{CFM}} / \mathrm{time}_{\mathrm{v4}} \ge 5\) at **both**
  \(n=10^5\) and \(n=10^6\). If every accuracy cell is BEST, speed is reported but not a
  gate.

Holm adjustment across the primary paired tests in the accuracy family; unadjusted
results also reported. Coverage is a rate bar (no Holm).

---

## 6. Budget and early stop

* **Wall-clock target:** whole run (all tracks + speed) \(\le\) **2 hours** on 8 cores.
* Multiplier bootstrap \(B=2000\), no booster refits for v4 bands. CFM timing as above.
* Resumable per-seed jsonl cache under `benchmarks/results/v4_*.jsonl`.
* **Early-stop rule:** if the wall cap hits, cells not completed are **INCOMPLETE**,
  never imputed. A rate bar is PASS/FAIL only if the completed replications already
  decide it whatever the missing ones would be; otherwise INCOMPLETE.
* Every miss stays on record with scenario, size, replication, and value.

---

## 7. Deviations

Any post-freeze change to scenarios, seeds, margins, bands, rival set, or bars is a
deviation: name it in `V4_RESULTS.md`, do not silently edit this file. Amending this
file requires a new dated subsection and a new freeze SHA.

---

## 8. Report

`docs/benchmark/V4_RESULTS.md`: header with this file's freeze SHA; verdict table
(TRACE v4 vs CFM vs DFL, paired CIs, coverage, speed ratios); every INCOMPLETE cell;
failures kept. No result numbers appear in *this* file.
