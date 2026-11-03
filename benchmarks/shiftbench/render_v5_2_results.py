# ruff: noqa: E501
"""Render docs/benchmark/V5_2_GATE_RESULTS.md from v5_2_gate_summary.json (no scoring here)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "benchmarks" / "results"
OUT_MD = ROOT / "docs" / "benchmark" / "V5_2_GATE_RESULTS.md"
IMPL_SHA = "ff0a8c659f5e7a3b68b424d354724635ef0c7c83"
FREEZE_SHA = "df5aff8"


def f(x, nd=4):
    return "—" if x is None else f"{x:.{nd}g}"


def main():
    s = json.loads((RES / "v5_2_gate_summary.json").read_text())
    w = s["wall"]
    L = [
        "# TRACE v5.2 gate — results (curved means)",
        "",
        f"**Freeze SHA:** `{FREEZE_SHA}` ([PREREG_V5_2.md](../research/PREREG_V5_2.md)).",
        f"**Implementation merge SHA:** `{IMPL_SHA}` (PR #34).",
        f"**Gate verdict:** **{s['overall']}** — H is rejected; the curved-mean claim is parked",
        "([PARK_V5_2.md](../research/PARK_V5_2.md)). V5.1's linear-mean claim stands exactly as",
        "stated in [V5_1_GATE_RESULTS.md](V5_1_GATE_RESULTS.md).",
        f"**Wall time:** {w['wall_sec']:.1f} s on {w['workers']} workers (cap {w['cap_sec']} s); "
        f"{w['tasks_done']}/{w['tasks_total']} runnable tasks finished "
        f"({w['tasks_preregistered']} pre-registered; cells {w['cells_infeasible']} had no "
        "runnable tasks, see note 1).",
        "",
        "Single pass on indices 10100–10149 (n = 2000, p = 5) and 10200–10219 (n = 20 000,",
        "p = 20). 9500–9549, 9600–9619, 9700–9749 and 9800–9819 were not touched.",
        "",
        "| # | family | scenario | κ | n | reps | err v5.2 | err DFL | err CFM | mean d | upper t "
        "| upper boot | δ | CI | p NI (Holm) | p BEST (Holm) | verdict | \\|ĉ\\| | \\|ŝ\\| "
        "| role | FA bound (13–16) | FA | err v5.1 (secondary) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|---:|---|---:|---|---:|",
    ]
    for c in s["cells"]:
        if "ci_verdict" not in c:
            L.append(
                f"| {c['cell']} | {c['family']} | {c['scen']} | {c['kappa']:g} | {c['n']} | 0 | — | — | — "
                "| — | — | — | — | — | — | — | **INCOMPLETE** | — | — | **INCOMPLETE** | — | "
                f"{'**INCOMPLETE**' if c['family'] == 'struct' else '—'} | — |"
            )
            continue
        r = c["mean_err_rivals"]
        L.append(
            f"| {c['cell']} | {c['family']} | {c['scen']} | {c['kappa']:g} | {c['n']} | {c['reps']} | "
            f"{f(c['mean_err_v5_2'])} | {f(r['DFL'])} | {f(r['CFM'])} | {f(c['mean_d'])} | "
            f"{f(c['upper_t'])} | {f(c['upper_boot'])} | {f(c['delta'])} | {c['ci_verdict']} | "
            f"{c['p_ni_unadj']:.3g} ({c['p_ni_holm']:.3g}) | {c['p_best_unadj']:.3g} ({c['p_best_holm']:.3g}) | "
            f"**{c['verdict']}** | {f(c['mean_abs_comp_hat'])} | {f(c['mean_abs_struct_hat'])} | "
            f"**{c['role']}** | {f(c.get('false_attr_upper'))} | "
            f"{('**' + c['false_attr'] + '**') if 'false_attr' in c else '—'} | "
            f"{f(c.get('secondary_mean_err_v5_1'))} |"
        )
    fails = []
    for c in s["cells"]:
        why = []
        if c["verdict"] != "BEST" and c["verdict"] != "NON-INFERIOR":
            why.append(f"accuracy {c['verdict']}")
        if c["role"] != "PASS":
            why.append(f"role {c['role']}")
        if c.get("false_attr") not in (None, "PASS"):
            why.append(f"false attribution {c['false_attr']}")
        if why:
            fails.append(
                f"* cell {c['cell']} ({c['scen']}, κ={c['kappa']:g}, n={c['n']}): " + ", ".join(why)
            )
    L += [
        "",
        "CI = verdict of the one-sided 95% bound alone (larger of t and bootstrap); *verdict* =",
        "after Holm across all 16 accuracy tests (note 3). FA = false-attribution check: upper",
        "95% bound of mean |ĉ| on the graded channel must be < δ (Holm across the 4 is in the",
        "JSON). err v5.1 is the secondary, descriptive OLS column (not gated).",
        "",
        "## Failures (all kept)",
        "",
        *fails,
        "",
        "## Notes and deviations",
        "",
        "1. **Cells 3 and 7 INCOMPLETE (not a code deviation; PREREG_V5_2 §3 rule).** For",
        "   `skew_comp` at κ = 1.0 the σ_B calibration has no solution: with the χ²₃-type",
        "   column 0 in period B, λ₂ of the period-B mean alone already exceeds λ₂(Y_A)",
        "   (λ₂(Y_B) − λ₂(Y_A) = +0.0857 at σ_B → 0+, +0.0969 at 0.3, +0.912 at 3.0; oracle",
        "   seed 99). Per §3 the cells are INCOMPLETE and were not re-designed; their 70 tasks",
        "   were not run, so the pass covered 490 of the 560 pre-registered tasks. Recorded in",
        "   `benchmarks/results/shiftbench_truth_shape_v5_2.json` before the gate ran.",
        '2. **Seed key.** `seed_of("data", key, n, p, rep)` with key `scen|k{κ:g}`, i.e.',
        "   `skew_comp|k0.5`, `skew_comp|k1`, `skew_comp|k0` (the prereg gives only the",
        "   `|k0.5` example). Fixed in the runner merged in PR #34 before the run.",
        '3. **"After Holm" (interpretation).** The prereg says cells must be BEST or',
        "   NON-INFERIOR after Holm across all 16. The scorer (committed in PR #34, before the",
        "   run) requires the CI verdict **and** the matching Holm-adjusted one-sided t p",
        "   (p BEST for BEST, p NI for NON-INFERIOR) < 0.05, with INCOMPLETE cells entering",
        "   the family as p = 1 so it stays 16. Under the looser reading (CI verdict alone)",
        "   the gate still FAILS: cells 4, 13, 14, 16 fail accuracy on the CI bound, cell 14",
        "   fails the role check, all four false-attribution checks fail, and cells 3 and 7",
        "   are INCOMPLETE.",
        "4. **Quantile rule.** Knots use linear-interpolation (type 7) quantiles of the",
        "   pooled, centred covariates (the runner centres X as in V5.1). Pivoted-QR drop",
        "   rule: |R_kk| ≤ 1e-10·|R_00|. No column was dropped in any of the 490 tasks.",
        "5. **Report fields.** `explain_v5` now also returns `dropped_cols_a/b` (empty for the",
        "   booster and OLS paths); the booster and OLS paths are otherwise unchanged from",
        "   `665939b`.",
        "6. **Secondary v5.1 column** was run inside the same pass (cells 1–8, 13–16), as",
        "   pre-registered; it is descriptive only.",
        "7. No tuning, no re-runs, no dropped or retried tasks.",
        "",
        "Full numbers: `benchmarks/results/v5_2_gate_summary.json`; per-replication records:",
        "`benchmarks/results/v5_2_gate_shape.jsonl`; wall: `v5_2_gate_wall.json`.",
    ]
    OUT_MD.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
