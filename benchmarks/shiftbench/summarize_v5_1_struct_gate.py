"""Score the TRACE v5.1 structure/mixed-cell gate (PREREG_V5_1_STRUCT.md §5, §8).

Reads benchmarks/results/v5_1_struct_gate_shape.jsonl (and the pre-flight /
wall records), writes v5_1_struct_gate_summary.json and
docs/benchmark/V5_1_STRUCT_GATE_RESULTS.md.

Helpers are reused: paired_ci / holm from summarize_v5_1_gate.py, after_holm
from summarize_v5_2_specified_gate.py, graded_err / planted_mag from
summarize_v4.py. Truth is the committed shiftbench_truth_shape.json (oracle
seed 99), never recomputed. Failures stay failures; cells without all
replications are INCOMPLETE, never imputed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from run_v5_1_struct_gate import IMPL_SHA, N_TASKS, PREREG_SHA
from scipy import stats
from summarize_v4 import graded_err, planted_mag
from summarize_v5_1_gate import holm, paired_ci
from summarize_v5_2_specified_gate import after_holm

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "benchmarks" / "results"
OUT_MD = ROOT / "docs" / "benchmark" / "V5_1_STRUCT_GATE_RESULTS.md"
OUT_JSON = RES / "v5_1_struct_gate_summary.json"
JSONL = RES / "v5_1_struct_gate_shape.jsonl"
WALL_PATH = RES / "v5_1_struct_gate_wall.json"
PREFLIGHT_PATH = RES / "v5_1_struct_gate_preflight.json"
TRUTH_FILE = RES / "shiftbench_truth_shape.json"
REPS = {2000: 50, 20000: 20}
CELLS = [
    ("skew_struct", 2000),
    ("tail_struct", 2000),
    ("shape_mixed", 2000),
    ("skew_struct", 20000),
    ("tail_struct", 20000),
    ("shape_mixed", 20000),
]
STRUCT_CELLS = {
    ("skew_struct", 2000),
    ("tail_struct", 2000),
    ("skew_struct", 20000),
    ("tail_struct", 20000),
}
ALPHA = 0.05
# Prereg §3 table, used as a freeze check (computed δ must match these).
FROZEN_DELTA = {
    "skew_struct": 0.005072,
    "tail_struct": 0.0009827,
    "shape_mixed": 0.009298,
}


def false_attr_verdict(upper, delta, p_holm) -> str:
    """PREREG_V5_1_STRUCT.md §5: bound < δ and Holm p < 0.05, both required."""
    if upper < delta and p_holm < ALPHA:
        return "PASS"
    return "FAIL"


def p_upper(v, margin):
    """One-sided t p value for H0: mean(v) >= margin."""
    v = np.asarray(v, float)
    se = v.std(ddof=1) / np.sqrt(len(v))
    return float(stats.t.sf((margin - v.mean()) / se, len(v) - 1))


def _load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else {}


def _abort_summary(overall: str, wall: dict, preflight: dict, note: str, seeds_spent: bool):
    summary = {
        "prereg_sha": PREREG_SHA,
        "impl_sha": IMPL_SHA,
        "overall": overall,
        "wall": wall,
        "preflight": {
            "ok": preflight.get("ok"),
            "error": preflight.get("error"),
            "method_identity_ok": preflight.get("method_identity", {}).get("ok"),
        },
        "cells": [],
        "seeds_spent": seeds_spent,
        "note": note,
    }
    return summary


def _write_abort_md(summary: dict, wall: dict, preflight: dict) -> None:
    overall = summary["overall"]
    spent = summary["seeds_spent"]
    spent_txt = (
        "The gate seeds have been spent and are not reused."
        if spent
        else "Gate seeds 14000–14049 and 14100–14119 were not touched and remain unspent."
    )
    ident = preflight.get("method_identity", {})
    L = [
        "# TRACE v5.1 structure/mixed-cell gate — results",
        "",
        (
            f"**Prereg SHA:** `{PREREG_SHA}` "
            "([PREREG_V5_1_STRUCT.md](../research/PREREG_V5_1_STRUCT.md))."
        ),
        f"**Implementation merge SHA:** `{IMPL_SHA}`.",
        f"**Gate verdict:** **{overall}**. H is not decided.",
        f"**Pre-flight:** {'OK' if preflight.get('ok') else 'not OK'}"
        + (f" ({preflight.get('error')})" if preflight.get("error") else "")
        + ".",
        "**Method identity (§2):** "
        + (
            "bit-identical to 665939b on the seed-99 skew_struct draw."
            if ident.get("ok")
            else f"not confirmed ({ident.get('error', preflight.get('error', 'n/a'))})."
        ),
        f"**Wall record:** stop={wall.get('stop')!r}, "
        f"tasks_done={wall.get('tasks_done', 0)}/"
        f"{wall.get('tasks_total', wall.get('tasks_preregistered', N_TASKS))}.",
        "",
        summary["note"],
        spent_txt,
        "",
        "This is not a PASS and not a FAIL. No cell table is scored.",
        "",
    ]
    OUT_MD.write_text("\n".join(L) + "\n")


def score_cell(scen, n, rows, truth):
    t = truth[scen]
    sh, k = t["shapley"], t["graded"][0]
    delta = 0.10 * planted_mag(sh, k)
    # Freeze check: agree with the prereg table to 4 significant figures.
    frozen = FROZEN_DELTA[scen]
    if abs(delta - frozen) / frozen > 5e-4:
        raise RuntimeError(f"delta for {scen} is {delta}, expected about {frozen}")
    c = {
        "scen": scen,
        "n": n,
        "p": 5 if n == 2000 else 20,
        "graded_lmoment": k + 1,
        "delta": delta,
        "truth": {"comp": sh["comp"][k], "struct": sh["struct"][k]},
        "family": "struct" if (scen, n) in STRUCT_CELLS else "mixed",
    }
    rs = sorted([r for r in rows if r["scen"] == scen and r["n"] == n], key=lambda r: r["rep"])
    c["reps"] = len(rs)
    if len(rs) < REPS[n]:
        c.update(
            verdict="INCOMPLETE",
            role="INCOMPLETE",
            note=f"{len(rs)}/{REPS[n]} reps finished inside the 30-minute cap",
        )
        if c["family"] == "struct":
            c["false_attr"] = "INCOMPLETE"
        return c
    e5, eriv, names, ch, sh_hat, eswap = [], [], [], [], [], []
    sec_v5, sec_dfl, sec_cfm = [], [], []
    for r in rs:
        ls = r["TRACE_V5_1"]["lsplit"]
        e5.append(graded_err(ls, sh, k))
        ch.append(ls["comp"][k])
        sh_hat.append(ls["struct"][k])
        eswap.append(graded_err({"comp": ls["struct"], "struct": ls["comp"]}, sh, k))
        best, bn = np.inf, None
        for nm, rv in r["rivals"].items():
            if "error" in rv:
                continue
            e = graded_err(rv["lsplit"], sh, k)
            if np.isfinite(e) and e < best:
                best, bn = e, nm
        eriv.append(best)
        names.append(bn)
        sec_v5.append(r["TRACE_V5_1"]["sec"])
        sec_dfl.append(r["rivals"]["DFL"]["sec"])
        sec_cfm.append(r["rivals"]["CFM"]["sec"])
    e5, eriv = np.array(e5), np.array(eriv)
    d = e5 - eriv
    ci = paired_ci(d)
    hi = max(ci["hi"], ci["boot_hi"])
    if not np.isfinite(hi):
        ci_verdict = "INCOMPLETE"
    elif hi < 0:
        ci_verdict = "BEST"
    elif hi < delta:
        ci_verdict = "NON-INFERIOR"
    else:
        ci_verdict = "FAIL"
    mc, ms = float(np.mean(np.abs(ch))), float(np.mean(np.abs(sh_hat)))
    mean_c, mean_s = float(np.mean(ch)), float(np.mean(sh_hat))
    swap_lower = float(np.mean(eswap)) < float(np.mean(e5))
    if c["family"] == "struct":
        dominant_ok = ms > mc
        role_ok = dominant_ok and not swap_lower
    else:
        sign_ok = np.sign(mean_c) == np.sign(c["truth"]["comp"]) and np.sign(mean_s) == np.sign(
            c["truth"]["struct"]
        )
        role_ok = bool(sign_ok) and not swap_lower
    rival_err = {
        nm: float(np.nanmean([graded_err(r["rivals"][nm].get("lsplit"), sh, k) for r in rs]))
        for nm in ("DFL", "CFM")
    }
    c.update(
        mean_err_v5=float(e5.mean()),
        mean_err_rivals=rival_err,
        best_rival_mode=max(set(names), key=names.count) if names else None,
        mean_d=ci["mean"],
        upper_t=ci["hi"],
        upper_boot=ci["boot_hi"],
        upper_used=hi,
        ci_verdict=ci_verdict,
        p_ni_unadj=p_upper(d, delta),
        p_best_unadj=p_upper(d, 0.0),
        mean_abs_comp_hat=mc,
        mean_abs_struct_hat=ms,
        mean_comp_hat=mean_c,
        mean_struct_hat=mean_s,
        mean_err_roles_swapped=float(np.mean(eswap)),
        role="PASS" if role_ok else "FAIL",
        sec_v5_mean=float(np.mean(sec_v5)),
        sec_v5_median=float(np.median(sec_v5)),
        sec_dfl_mean=float(np.mean(sec_dfl)),
        sec_dfl_median=float(np.median(sec_dfl)),
        sec_cfm_mean=float(np.mean(sec_cfm)),
        sec_cfm_median=float(np.median(sec_cfm)),
    )
    if c["family"] == "struct":
        a = np.abs(ch)
        fa = paired_ci(a)
        c["false_attr_upper_t"] = fa["hi"]
        c["false_attr_upper_boot"] = fa["boot_hi"]
        c["false_attr_upper"] = max(fa["hi"], fa["boot_hi"])
        c["false_attr_p_unadj"] = p_upper(a, delta)
    return c


def _fmt(x, digits=4):
    if x is None:
        return "—"
    if isinstance(x, str):
        return x
    return f"{x:.{digits}g}"


def write_results_md(summary: dict, wall: dict, preflight: dict, runner_sha: str | None) -> None:
    cells = summary["cells"]
    overall = summary["overall"]
    ident = preflight.get("method_identity", {})
    L = [
        "# TRACE v5.1 structure/mixed-cell gate — results",
        "",
        (
            f"**Prereg SHA:** `{PREREG_SHA}` "
            "([PREREG_V5_1_STRUCT.md](../research/PREREG_V5_1_STRUCT.md))."
        ),
        f"**Implementation merge SHA:** `{IMPL_SHA}`.",
        f"**Runner SHA:** `{runner_sha or wall.get('runner_sha') or 'not recorded'}`.",
        f"**Gate verdict:** **{overall}**.",
        (
            f"**Wall time:** {wall.get('wall_sec', 0):.1f} s on {wall.get('workers', '?')} workers "
            f"(cap {wall.get('cap_sec', 1800)} s); "
            f"{wall.get('tasks_done', 0)}/{wall.get('tasks_total', N_TASKS)} tasks finished."
        ),
        (
            f"**Pre-flight:** **{'OK' if preflight.get('ok') else 'not OK'}**. "
            f"Method identity (§2): "
            + (
                "explain_v5_1 is bit-identical to 665939b on the seed-99 skew_struct draw."
                if ident.get("ok")
                else f"failed ({ident.get('error', 'n/a')})."
            )
        ),
        (
            f"**Rehearsal:** **{'OK' if wall.get('rehearsal_ok') else wall.get('stop', 'n/a')}** "
            f"({wall.get('rehearsal_tasks_done', '?')}/"
            f"{wall.get('rehearsal_tasks_total', 12)} tasks"
            + (
                f", {wall.get('rehearsal_wall_sec', 0):.1f} s"
                if wall.get("rehearsal_wall_sec")
                else ""
            )
            + ")."
        ),
        "",
        "Bars: PREREG_V5_1_STRUCT.md §5 (paired d = err_v5.1 − err_best_rival, one-sided",
        "95% upper bound, t-interval and 10 000-draw bootstrap (seed 2026), larger used;",
        "BEST if < 0, NON-INFERIOR if < δ = 0.10 × planted magnitude). **After Holm:** the",
        "CI verdict and the matching Holm-adjusted one-sided t p-value (p BEST for a BEST",
        "claim, p NI for a NON-INFERIOR claim) must both hold (p < 0.05), Holm across the",
        "six accuracy tests. Rivals DFL and CFM re-run on the same seeds (paired), on the",
        "same centred arrays as the V5.1 gate. Role check: on structure-only cells,",
        "mean |ŝ| > mean |ĉ| and swapping roles does not lower the error; on mixed cells,",
        "sign(mean ĉ) = sign(truth comp) (+), sign(mean ŝ) = sign(truth struct) (−), and",
        "swapping roles does not lower the error. False-attribution (structure-only):",
        "upper bound of mean |ĉ| < δ and the Holm-adjusted one-sided t p-value < 0.05,",
        "Holm across the four structure-only tests.",
        "",
        "| cell | n | p | reps | err v5.1 | err DFL | err CFM | mean d | upper t | "
        "upper boot | δ | verdict | mean ĉ | mean ŝ | \\|ĉ\\| | \\|ŝ\\| | role | "
        "FA bound | p NI (Holm) | p BEST (Holm) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|"
        "---:|---|---:|---|---|",
    ]
    failures = []
    for c in cells:
        if c["verdict"] == "INCOMPLETE" or c.get("ci_verdict") is None and "mean_err_v5" not in c:
            L.append(
                f"| {c['scen']} | {c['n']} | {c['p']} | {c['reps']} | — | — | — | — | — | — | "
                f"{_fmt(c['delta'])} | INCOMPLETE | — | — | — | — | INCOMPLETE | "
                f"{'INCOMPLETE' if c.get('family') == 'struct' else '—'} | — | — |"
            )
            failures.append(f"{c['scen']} n={c['n']}: INCOMPLETE ({c.get('note', '')})")
            continue
        fa = _fmt(c.get("false_attr_upper")) if c.get("family") == "struct" else "—"
        if c.get("family") == "struct" and "false_attr" in c:
            fa = f"{_fmt(c.get('false_attr_upper'))} (**{c['false_attr']}**)"
        L.append(
            f"| {c['scen']} | {c['n']} | {c['p']} | {c['reps']} | "
            f"{_fmt(c['mean_err_v5'])} | {_fmt(c['mean_err_rivals']['DFL'])} | "
            f"{_fmt(c['mean_err_rivals']['CFM'])} | {_fmt(c['mean_d'])} | "
            f"{_fmt(c['upper_t'])} | {_fmt(c['upper_boot'])} | {_fmt(c['delta'])} | "
            f"**{c['verdict']}** | {_fmt(c['mean_comp_hat'])} | "
            f"{_fmt(c['mean_struct_hat'])} | {_fmt(c['mean_abs_comp_hat'])} | "
            f"{_fmt(c['mean_abs_struct_hat'])} | **{c['role']}** | {fa} | "
            f"{_fmt(c['p_ni_unadj'], 3)} ({_fmt(c.get('p_ni_holm'), 3)}) | "
            f"{_fmt(c['p_best_unadj'], 3)} ({_fmt(c.get('p_best_holm'), 3)}) |"
        )
        if c["verdict"] != "BEST" and c["verdict"] != "NON-INFERIOR":
            failures.append(f"{c['scen']} n={c['n']}: accuracy {c['verdict']}")
        if c["role"] != "PASS":
            failures.append(f"{c['scen']} n={c['n']}: role {c['role']}")
        if c.get("family") == "struct" and c.get("false_attr") != "PASS":
            failures.append(f"{c['scen']} n={c['n']}: false-attribution {c.get('false_attr')}")
    L += [
        "",
        "p values are unadjusted with Holm (across the six accuracy tests) in brackets.",
        "False-attribution Holm is across the four structure-only tests.",
        "Truth (committed shiftbench_truth_shape.json, oracle seed 99, not recomputed): "
        + "; ".join(
            f"{s} λ{next(c['graded_lmoment'] for c in cells if c['scen'] == s)}: "
            f"comp {next(c['truth']['comp'] for c in cells if c['scen'] == s):.4g}, "
            f"struct {next(c['truth']['struct'] for c in cells if c['scen'] == s):.4g}"
            for s in ("skew_struct", "tail_struct", "shape_mixed")
        )
        + ".",
        "",
        "Full numbers: `benchmarks/results/v5_1_struct_gate_summary.json`;",
        "per-replication records: `benchmarks/results/v5_1_struct_gate_shape.jsonl`.",
        "",
        "## Failures kept",
        "",
    ]
    if failures:
        L += [f"- {f}" for f in failures]
    else:
        L.append("None. Every accuracy, role, and false-attribution check passed.")
    L += [
        "",
        "## Secondary, descriptive, not scored",
        "",
        "These readouts cannot rescue or sink H (PREREG_V5_1_STRUCT.md §8).",
        "",
        "1. Mean ĉ and ŝ against the planted truth, and their relation to δ.",
        "   On `tail_struct` this shows whether the λ₄ composition piece is biased",
        "   when its truth is ≈ 0 (the open `tail_comp` composition under-estimate).",
        "",
        "| cell | n | mean ĉ | truth c | mean ŝ | truth s | δ | ĉ − truth c | ŝ − truth s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in cells:
        if "mean_comp_hat" not in c:
            L.append(
                f"| {c['scen']} | {c['n']} | — | {_fmt(c['truth']['comp'])} | — | "
                f"{_fmt(c['truth']['struct'])} | {_fmt(c['delta'])} | — | — |"
            )
            continue
        L.append(
            f"| {c['scen']} | {c['n']} | {_fmt(c['mean_comp_hat'])} | "
            f"{_fmt(c['truth']['comp'])} | {_fmt(c['mean_struct_hat'])} | "
            f"{_fmt(c['truth']['struct'])} | {_fmt(c['delta'])} | "
            f"{_fmt(c['mean_comp_hat'] - c['truth']['comp'])} | "
            f"{_fmt(c['mean_struct_hat'] - c['truth']['struct'])} |"
        )
    L += [
        "",
        "2. Median per-task wall seconds of TRACE, DFL, and CFM per cell (no speed bar).",
        "",
        "| cell | n | median TRACE s | median DFL s | median CFM s |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in cells:
        if "sec_v5_median" not in c:
            L.append(f"| {c['scen']} | {c['n']} | — | — | — |")
            continue
        L.append(
            f"| {c['scen']} | {c['n']} | {_fmt(c['sec_v5_median'])} | "
            f"{_fmt(c['sec_dfl_median'])} | {_fmt(c['sec_cfm_median'])} |"
        )
    L += [
        "",
        "## Notes and deviations",
        "",
        "1. **Method identity.** Four commits after `665939b` touched `crates/core`:",
        "   `5db5806` (V5.2 spline), `3b8a1d4` (V5.1 made the default), `3b382f0`",
        "   (report-only residual-vs-fit check), and `4cf9754` (V5.2 specified model).",
        "   Pre-flight compared `explain_v5_1` on one centred `skew_struct` draw at",
        "   oracle seed 99 to the frozen `665939b` record. The split, the four hybrid",
        "   quantile / L-moment curves, and the mixing residuals were bit-identical.",
        "   The later report-only `linear_mean_check` and `dropped_cols_*` fields are",
        "   present only on the later SHA and do not enter the split.",
        "2. **Harness.** Spawn-safe pool, mandatory pre-flight in a spawned child,",
        "   rehearsal on seeds 97 and 98 (12 tasks, never scored), 5-minute no-progress",
        "   watchdog, 30-minute cap. The confirmatory pass ran only with `--confirmatory`.",
        "   Sealed seeds 9600–9619 and coverage seeds 10400–10799 were in FORBIDDEN",
        "   and were not run.",
        "3. **No deviations** from PREREG_V5_1_STRUCT.md are recorded in this file.",
        "   Any later change would be a deviation and must be reported as one.",
        "",
    ]
    OUT_MD.write_text("\n".join(L) + "\n")


def _is_rehearsal_stage_abort(wall: dict) -> bool:
    stop = wall.get("stop")
    if stop in ("rehearsal_abort",):
        return True
    # A watchdog during rehearsal is recorded without confirmatory=True.
    if stop == "watchdog_abort" and not wall.get("confirmatory"):
        return True
    return False


def main():
    wall = _load_json(WALL_PATH)
    preflight = _load_json(PREFLIGHT_PATH)
    stop = wall.get("stop")
    if stop == "awaiting_confirmatory":
        summary = _abort_summary(
            "NOT RUN (awaiting --confirmatory)",
            wall,
            preflight,
            "Pre-flight and rehearsal finished; the confirmatory pass has not been started.",
            seeds_spent=False,
        )
        OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")
        _write_abort_md(summary, wall, preflight)
        print(json.dumps({"overall": summary["overall"], "wall": wall.get("stop")}, indent=1))
        return
    if stop == "preflight_abort":
        summary = _abort_summary(
            "ABORTED (pre-flight)",
            wall,
            preflight,
            "Pre-flight failed. H is not decided.",
            seeds_spent=False,
        )
        OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")
        _write_abort_md(summary, wall, preflight)
        print(json.dumps({"overall": summary["overall"]}, indent=1))
        return
    if _is_rehearsal_stage_abort(wall):
        label = "ABORTED (watchdog)" if stop == "watchdog_abort" else "ABORTED (rehearsal)"
        summary = _abort_summary(
            label,
            wall,
            preflight,
            "Rehearsal did not complete successfully. H is not decided.",
            seeds_spent=False,
        )
        OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")
        _write_abort_md(summary, wall, preflight)
        print(json.dumps({"overall": summary["overall"]}, indent=1))
        return
    if stop == "watchdog_abort" and wall.get("confirmatory"):
        summary = _abort_summary(
            "ABORTED (watchdog)",
            wall,
            preflight,
            "The 5-minute no-progress watchdog tripped after the gate started. H is not decided.",
            seeds_spent=True,
        )
        OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")
        _write_abort_md(summary, wall, preflight)
        print(json.dumps({"overall": summary["overall"]}, indent=1))
        return

    if not JSONL.exists():
        summary = _abort_summary(
            "ABORTED (no gate file)",
            wall,
            preflight,
            "No gate results file is present. H is not decided.",
            seeds_spent=False,
        )
        OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")
        _write_abort_md(summary, wall, preflight)
        print(json.dumps({"overall": summary["overall"]}, indent=1))
        return

    rows = [json.loads(x) for x in JSONL.read_text().splitlines() if x]
    truth = json.loads(TRUTH_FILE.read_text())
    cells = [score_cell(scen, n, rows, truth) for scen, n in CELLS]
    for c, a, b in zip(
        cells,
        holm([c.get("p_ni_unadj", 1.0) for c in cells]),
        holm([c.get("p_best_unadj", 1.0) for c in cells]),
        strict=True,
    ):
        if "ci_verdict" not in c:
            continue
        c["p_ni_holm"], c["p_best_holm"] = a, b
        c["verdict"] = after_holm(c["ci_verdict"], c["upper_used"], c["delta"], b, a)
    st = [c for c in cells if c["family"] == "struct"]
    for c, a in zip(st, holm([c.get("false_attr_p_unadj", 1.0) for c in st]), strict=True):
        if "false_attr_upper" not in c:
            continue
        c["false_attr_p_holm"] = a
        c["false_attr"] = false_attr_verdict(c["false_attr_upper"], c["delta"], a)
    checks = (
        [c["verdict"] for c in cells]
        + [c["role"] for c in cells]
        + [c.get("false_attr") for c in st]
    )
    if any(v == "FAIL" for v in checks) or any(v == "INCOMPLETE" for v in checks):
        overall = "FAIL"
    else:
        overall = "PASS"
    unfinished = wall.get("tasks_done", 0) < wall.get("tasks_total", N_TASKS)
    over_cap = wall.get("wall_sec", 0) > wall.get("cap_sec", 30 * 60)
    if unfinished or over_cap:
        # Unfinished cells are already INCOMPLETE; the kill rule rejects H.
        overall = "FAIL"
    summary = {
        "prereg_sha": PREREG_SHA,
        "impl_sha": IMPL_SHA,
        "overall": overall,
        "wall": wall,
        "preflight": {
            "ok": preflight.get("ok"),
            "method_identity_ok": preflight.get("method_identity", {}).get("ok"),
            "versions": preflight.get("versions"),
        },
        "cells": cells,
        "seeds_spent": True,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    write_results_md(summary, wall, preflight, wall.get("runner_sha"))
    print(json.dumps({"overall": overall, "wall": wall}, indent=1))
    for c in cells:
        print(
            c["scen"],
            c["n"],
            c.get("verdict"),
            c.get("role"),
            c.get("false_attr", ""),
            {k: round(c[k], 6) for k in ("mean_err_v5", "mean_d", "upper_used", "delta") if k in c},
        )


if __name__ == "__main__":
    main()
