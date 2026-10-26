# TRACE v4 results

**Prereg SHA:** `d50732f` (frozen in `docs/benchmark/PREREG_V4.md`).
**Overall accuracy (prereg bar):** **FAIL**
**Coverage bar:** **INCOMPLETE**
**Speed (conditional):** FAIL conditional (≥5× both sizes); ratios={100000: 1.020376728676559, 1000000: 2.9400156180169517} (accuracy already FAIL — speed does not rescue)
**Scored run status:** shape=350 cover=800 shiftbench=90 speed=6 records.

Failures stay failures. Incomplete cells are labelled INCOMPLETE, never imputed.

## Deviations

1. **Missing sklearn on first scored pass.** The initial `run_v4.py all --workers 8` completed shape/cover/shiftbench/speed, but every CFM/DFL call raised `No module named 'sklearn'`. TRACE_V4 rows and seeds are unchanged. After installing `scikit-learn` in the box venv, CFM/DFL were **backfilled on the same seeds** into `v4_shape.jsonl` and `v4_shiftbench.jsonl`, and `v4_speed.jsonl` was re-run once. No TRACE_V4 recomputation; no seed change.
2. **ShiftBench per-feature vs rivals.** CFM/DFL adapters in this repo do not return `feat_comp`/`feat_struct`, so the three per-feature paired cells stay INCOMPLETE (TRACE_V4 alone has feature pieces).
3. **Coverage bands.** `explain_v4` / cover track persist a budget CI but not per-cell lsplit / mean-channel multiplier bands, so the oracle coverage bar cannot be evaluated and stays INCOMPLETE.

## H7 shape (graded-channel error vs best rival)

| cell | n | reps | mean err v4 | best rival | mean d | upper 95% CI | δ | verdict |
|---|---:|---:|---:|---|---:|---:|---:|---|
| skew_struct | 2000 | 50 | 0.05817 | DFL | 0.04669 | 0.05058 | 0.005072 | **FAIL** |
| skew_struct | 20000 | 20 | 0.0505 | DFL | 0.04687 | 0.04778 | 0.005072 | **FAIL** |
| tail_struct | 2000 | 50 | 0.0511 | CFM | 0.04398 | 0.04711 | 0.0009827 | **FAIL** |
| tail_struct | 20000 | 20 | 0.04772 | DFL | 0.04518 | 0.04701 | 0.0009827 | **FAIL** |
| skew_comp | 2000 | 50 | 0.05426 | CFM | 0.03429 | 0.03736 | 0.004025 | **FAIL** |
| skew_comp | 20000 | 20 | 0.05193 | CFM | 0.03951 | 0.04074 | 0.004025 | **FAIL** |
| tail_comp | 2000 | 50 | 0.06689 | CFM | 0.05922 | 0.06256 | 0.0004586 | **FAIL** |
| tail_comp | 20000 | 20 | 0.06035 | CFM | 0.05574 | 0.05662 | 0.0004586 | **FAIL** |
| shape_mixed | 2000 | 50 | 0.2357 | CFM | 0.2092 | 0.2195 | 0.009298 | **FAIL** |
| shape_mixed | 20000 | 20 | 0.1947 | CFM | 0.1727 | 0.1762 | 0.009298 | **FAIL** |

## ShiftBench accuracy (mean-channel + per-feature vs best rival)

| cell | metric | reps | mean err v4 | best rival | mean d | upper 95% CI | δ | verdict |
|---|---|---:|---:|---|---:|---:|---:|---|
| mixed | mean_channel | 30 | 0.4671 | DFL | 0.4493 | 0.451 | 0.002 | **FAIL** |
| mixed | per_feature | 30 | 0.1033 | n/a | — | — | 0.002 | INCOMPLETE |
| tail | mean_channel | 30 | 0.006963 | CFM | -0.01088 | -0.009273 | 0.002 | **BEST** |
| tail | per_feature | 30 | 0.001402 | n/a | — | — | 0.002 | INCOMPLETE |
| hetero | mean_channel | 30 | 0.2653 | DFL | 0.2482 | 0.2501 | 0.002 | **FAIL** |
| hetero | per_feature | 30 | 0.08141 | n/a | — | — | 0.002 | INCOMPLETE |

## H8b coverage

Records: 800. Cover track wrote point estimates and a budget CI, but not per-cell split-product / multiplier bands for lsplit or mean-channel pieces.
Oracle coverage vs planted truth (PREREG_V4 §5: pooled ≥ 0.90 and every cell ≥ 0.85) requires those bands. Marked **INCOMPLETE** — not imputed, not passed.
Reps by scenario: entry_exit=80, hetero=80, mixed=80, null=80, shape_mixed=80, skew_comp=80, skew_struct=80, tail=80, tail_comp=80, tail_struct=80.

## Speed vs CFM

| n | reps | mean sec v4 | mean sec CFM | ratio CFM/v4 |
|---:|---:|---:|---:|---:|
| 100000 | 3 | 8.62 | 8.8 | 1.02 |
| 1000000 | 3 | 46.5 | 137 | 2.94 |
