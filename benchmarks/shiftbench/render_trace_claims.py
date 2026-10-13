# ruff: noqa: E501, E731, B007, B023, F841
"""Render docs/benchmark/TRACE_CLAIMS_RESULTS.md from trace_claims_summary.json (mechanical; no edits to numbers)."""

import json
from pathlib import Path

from run_synthetic import RES

S = json.loads((RES / "trace_claims_summary.json").read_text())
OUT = Path(__file__).resolve().parents[2] / "docs" / "benchmark" / "TRACE_CLAIMS_RESULTS.md"


def f(x, d=3):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{d}g}" if abs(x) < 1e-3 and x != 0 else f"{x:.{d}f}"
    return str(x)


def ci(pt):
    return f"{f(pt['mean'])} [{f(pt['t_ci'][0])}, {f(pt['t_ci'][1])}] (pct [{f(pt['pct_ci'][0])}, {f(pt['pct_ci'][1])}]), p={f(pt['p'])}"


L = [
    "# TRACE claims (H5–H8): results",
    "",
    "Scored exactly per [PREREG_TRACE_CLAIMS.md](PREREG_TRACE_CLAIMS.md) as amended by "
    "[AMENDMENT_1](PREREG_TRACE_CLAIMS_AMENDMENT_1.md) (deviations D1–D12 logged before results). "
    "Raw records: `benchmarks/results/trace_claims_*.jsonl`; summary: `trace_claims_summary.json` "
    "(produced by `summarize_trace_claims.py`, rendered by `render_trace_claims.py`). Failures stay on record.",
    "",
]

h5, h6, h7, h8 = S["h5"], S["h6"], S["h7"], S["h8"]
cov = h8["coverage"]
a = h8.get("h8a_primary", {})
L += ["## Verdicts", "", "| hypothesis | bar | result | verdict |", "|---|---|---|---|"]
L.append(
    f"| H5 exact budget + invariance | every gap ≤ 1e-9·max(1,W2²) in every replication; permutation ≤ 1e-6 | "
    f"fail counts {h5['fail_counts']}; invariance max {f((h5['invariance'] or {}).get('max_change_abs'))} | "
    f"{'PASS' if h5['pass'] else 'FAIL'} |"
)
for dom in ("real", "synthetic"):
    for nm in ("bootstrap", "split_half"):
        o = h6.get(dom, {}).get(nm)
        if o:
            L.append(
                f"| H6 {dom} {nm} vs best rival ({o['best_rival']}) | mean diff ≥ 0.05, both 95% CIs > 0 | {ci(o['paired'])} | {o['verdict']} |"
            )
for k, v in h7["recovery"].items():
    L.append(
        f"| H7 recovery {k} | ≥ {v['bar']} | {f(v['rate'])} ({v['reps']} reps) | {'PASS' if v['pass'] else 'FAIL'} |"
    )
for k, v in h7["rival_test"].items():
    L.append(
        f"| H7 rival test {k} vs {v['best_rival']} (err diff, n=20k) | CI < 0 | {ci(v['paired'])} | {'win' if v['below_zero'] else 'no win'} |"
    )
L.append(
    f"| H7 rival claim | CI < 0 in ≥ 4 of 5 | {h7['rival_wins']} of 5 | {'SUPPORTED' if h7['rival_supported'] else 'NOT SUPPORTED'} |"
)
if a:
    L.append(
        f"| H8(a) null FPR, joint test, mean channel (200 reps) | one-sided CP upper ≤ 0.09 (x ≤ 11) | x = {a['x']}/{a['n']}, "
        f"CP upper {f(h8['null200']['TRACE_mean_aref']['cp_upper_1s'])} | {'PASS' if a['pass'] else 'FAIL'} |"
    )
L.append(
    f"| H8(b) coverage | pooled ≥ 0.90, every cell ≥ 0.85 | pooled {f(cov['pooled'])}, min cell {f(cov['min_cell'])} | {'PASS' if cov['pass'] else 'FAIL'} |"
)
if "placebo" in h8:
    L.append(
        f"| H8(c) placebo (secondary) | ≤ 0.10 | {f(h8['placebo']['TRACE'])} | {'PASS' if h8['placebo_pass'] else 'FAIL'} |"
    )
L += [
    "",
    "## Holm (family of tests with p-values, D9)",
    "",
    "| test | p | Holm threshold | Holm reject | bar met |",
    "|---|---|---|---|---|",
]
for t in S["holm"]:
    L.append(
        f"| {t['test']} | {f(t['p'])} | {f(t['holm_threshold'])} | {f(t['holm_reject'])} | {f(t['bar_ok'])} |"
    )
L += [
    "",
    "## Details",
    "",
    "### H5",
    "",
    f"Replications: {h5['n_replications']}. Worst relative gaps: "
    + ", ".join(f"{k} {f(v)}" for k, v in h5["worst_relative"].items())
    + ".",
    "",
]
if any(h5["fail_examples"].values()):
    L.append("Failure examples (scenario, n, p, rep, gap): " + json.dumps(h5["fail_examples"]))
L += [
    "",
    "Rival contrast (median / max, scaled by sd(y_A); order gap = A-reference vs B-reference):",
    "",
    "| method | stat | additivity median | order median | order max |",
    "|---|---|---|---|---|",
]
for m, d in h5["rival_contrast"].items():
    for s in ("mean", "sd", "q10", "q50", "q90"):
        if f"{s}|add" in d:
            L.append(
                f"| {m} | {s} | {f(d[s + '|add']['median'])} | {f(d[s + '|order']['median'])} | {f(d[s + '|order']['max'])} |"
            )
L += ["", "### H6 means", ""]
for dom in ("real", "synthetic"):
    for nm in ("bootstrap", "split_half"):
        o = h6.get(dom, {}).get(nm)
        if o:
            L.append(f"* {dom} {nm}: " + ", ".join(f"{m} {f(v)}" for m, v in o["means"].items()))
    if dom in h6:
        L.append(f"* {dom} undefined Spearman counts (set to 0, D11): {h6[dom].get('nan_counts')}")
L += [
    "",
    "### H7 errors (graded channel, L-moment scale)",
    "",
    "| scenario | TRACE | DFL | CFM | planted magnitude |",
    "|---|---|---|---|---|",
]
for k, v in h7["errors"].items():
    L.append(
        f"| {k} | {f(v['TRACE'])} | {f(v.get('DFL'))} | {f(v.get('CFM'))} | {f(v['planted_magnitude'])} |"
    )
L += ["", "### H8", ""]
if "null200" in h8:
    L += ["| statistic | rejections | FPR | one-sided CP upper | bar |", "|---|---|---|---|---|"]
    for k, v in h8["null200"].items():
        L.append(
            f"| {k} | {v['x']}/{v['n']} | {f(v['fpr'])} | {f(v['cp_upper_1s'])} | {'pass' if v['pass_bar'] else 'fail'} |"
        )
    L += ["", "McNemar vs rivals (mean channel): " + json.dumps(h8["mcnemar_vs_rivals"])]
L += [
    "",
    f"Frozen band design null (40 reps): TRACE union-rule FPR {f(cov['frozen_null_union_fpr'])}, joint-test FPR {f(cov['frozen_null_joint_fpr'])}. "
    f"Rivals (frozen union rule, existing records): {json.dumps(S['frozen_rival_coverage'])}",
    "",
    "Coverage cells: " + ", ".join(f"{k} {f(v)}" for k, v in cov["cells"].items()),
]
if "placebo" in h8:
    L.append("Placebo FPR by method: " + ", ".join(f"{m} {f(v)}" for m, v in h8["placebo"].items()))
L += ["", "## Frozen ShiftBench tracks with TRACE (A-reference split, D4)", ""]
for size, o in S["frozen_accuracy"].items():
    L.append(
        f"**{size}**: mean-channel error "
        + ", ".join(
            f"{m} {f(v)}" for m, v in o["channel"].get("mean", {}).items() if m != "paired_vs"
        )
        + "; per-feature error "
        + ", ".join(f"{m} {f(v)}" for m, v in o["feature"].items())
        + f"; TRACE order-free vs order-free truth (mean) {f(o['trace_orderfree_channel'].get('mean'))}."
    )
    pv = o["channel"].get("mean", {}).get("paired_vs", {})
    if pv:
        L.append(
            "  Paired TRACE − rival (mean channel): "
            + "; ".join(f"{m} {ci(p)}" for m, p in pv.items())
        )
L += [
    "",
    "## Scale (TRACE)",
    "",
    json.dumps(S["scale"]),
    "",
    "Rival scale cells are missing: the ShiftBench run script stopped before its scale track (D10).",
]
OUT.write_text("\n".join(L) + "\n")
print("wrote", OUT)
