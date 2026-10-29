# TRACE v5 first gate — results

**Prereg SHA:** `bce021e` ([PREREG_V5.md](../research/PREREG_V5.md)).
**First-gate verdict:** **FAIL**.
**Full scored suite: not started.**

Bars: PREREG_V4 §5 as inherited by PREREG_V5 §5 (paired d = err_v5 − err_best_rival,
one-sided 95% upper bound, normal and 10 000-draw bootstrap (seed 2026) must agree;
BEST if < 0, NON-INFERIOR if < δ = 0.10 × planted magnitude). Rivals DFL and CFM
re-run on the same seeds (paired).

| cell | n | reps | err v5 | err DFL | err CFM | mean d | upper 95% | δ | verdict | \|ĉ\| | \|ŝ\| | falsifier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| skew_comp | 2000 | 50 | 0.02699 | 0.07988 | 0.02218 | 0.004808 | 0.00711 | 0.004025 | **FAIL** | 0.0262 | 0.01294 | **PASS** |
| tail_comp | 2000 | 50 | 0.009708 | 0.01427 | 0.0106 | -1.311e-05 | 0.001644 | 0.0004586 | **FAIL** | 0.003115 | 0.004922 | **FAIL** |
| skew_comp | 20000 | 0 | — | — | — | — | — | 0.004025 | INCOMPLETE | — | — | INCOMPLETE |
| tail_comp | 20000 | 0 | — | — | — | — | — | 0.0004586 | INCOMPLETE | — | — | INCOMPLETE |

Truth (Shapley, graded channel): skew_comp λ3: comp 0.04011, struct -0.0001375; tail_comp λ4: comp 0.004298, struct 0.000288.

Holm-adjusted p-values (completed cells only) are in `benchmarks/results/v5_gate1_summary.json`.

## Deviations and notes

1. **n = 20 000 cells not run.** PREREG_V5 §3 assigns indices 9600–9619 to the
   n = 20 000 cells of this gate. The owner instructed that these indices are
   held back and must not be touched in this run, so both n = 20 000 cells are
   **INCOMPLETE**. Under §8 an INCOMPLETE required cell blocks a first-gate PASS.
2. **Falsifier operationalisation.** §5 states the rule in words. Applied as: PASS
   iff mean |ĉ| > mean |ŝ| on the graded channel (truth puts the mass in
   composition) **and** swapping the two roles would not lower the error.
3. **Conditional model.** PREREG_V5 §2.1 leaves the conditional family to in-repo
   defaults. v5 uses the v4 binned mean booster (`FitL2Config::default().mean`,
   unchanged) plus the period's empirical residual law, a 256-point grid on the
   exact support (v4 default size), and exact L-moments of the gridded law. Fixed
   before any gate output was seen; no tuning.
4. **Per-feature game not scored.** The scored estimand is the role split of the
   graded L-moment; per-feature shares on the mixture law are not implemented in
   this PR.
5. **Holm.** Applied across the completed cells' one-sided tests; with two
   INCOMPLETE cells the family has two members instead of four.
