"""Score the TRACE v5.2c specified-family replication gate.

Reads benchmarks/results/v5_2c_specified_gate_shape.jsonl, writes
v5_2c_specified_gate_summary.json and docs/benchmark/V5_2C_SPECIFIED_GATE_RESULTS.md.

Bars, Holm, BEST / NON-INFERIOR, δ, and the role check are those of
PREREG_V5_2_SPECIFIED §5, incorporated by reference in
PREREG_V5_2C_SPECIFIED_REPLICATION.md. Scoring helpers are imported from
summarize_v5_2_specified_gate so the rule cannot drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from summarize_v4 import graded_err, normalize_lsplit, planted_mag
from summarize_v5_1_gate import holm
from summarize_v5_2_specified_gate import after_holm, ci_verdict, truth_key

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "benchmarks" / "results"
OUT_MD = ROOT / "docs" / "benchmark" / "V5_2C_SPECIFIED_GATE_RESULTS.md"
OUT_JSON = RES / "v5_2c_specified_gate_summary.json"
JSONL = RES / "v5_2c_specified_gate_shape.jsonl"
WALL_PATH = RES / "v5_2c_specified_gate_wall.json"
PREFLIGHT_PATH = RES / "v5_2c_specified_gate_preflight.json"
PREREG_SHA = "f022ba3e42ebe4b7cee1dbdf2bf1c254b248b1f6"
IMPL_SHA = "1aab4778abf8f84c8c8f7d9603eb8c322f1ba3c9"
# Filled after the runner PR is merged (order of work §4 step 3).
RUNNER_SHA = ""
REPS = {2000: 50, 20000: 20}
CELLS = [("skew_comp", 2000), ("tail_comp", 2000), ("skew_comp", 20000), ("tail_comp", 20000)]
TRUTH_FILE = RES / "shiftbench_truth_shape_v5_2_specified.json"
METHOD = "TRACE_V5_2_SPECIFIED"
PREREG_LINK = (
    "[PREREG_V5_2C_SPECIFIED_REPLICATION.md](../research/PREREG_V5_2C_SPECIFIED_REPLICATION.md)"
)


def _load_preflight():
    if PREFLIGHT_PATH.exists():
        return json.loads(PREFLIGHT_PATH.read_text())
    return {}


def _write_incomplete(overall: str, wall: dict, note: str, extra_lines: list[str] | None = None):
    summary = {
        "prereg_sha": PREREG_SHA,
        "impl_sha": IMPL_SHA,
        "runner_sha": RUNNER_SHA,
        "overall": overall,
        "wall": wall,
        "preflight": _load_preflight(),
        "note": note,
    }
    if "cells_infeasible" in wall:
        summary["cells_infeasible"] = wall["cells_infeasible"]
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    L = [
        "# TRACE v5.2c specified-family replication — results",
        "",
        f"**Prereg SHA:** `{PREREG_SHA}` ({PREREG_LINK}).",
        f"**Implementation merge SHA:** `{IMPL_SHA}` (reused, PR #58).",
        f"**Runner SHA:** `{RUNNER_SHA or wall.get('runner_sha', 'not yet recorded')}`.",
        f"**Gate verdict:** **{overall}**. H is not decided.",
        "",
        note,
    ]
    if extra_lines:
        L += extra_lines
    OUT_MD.write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    wall = json.loads(WALL_PATH.read_text()) if WALL_PATH.exists() else {}
    preflight = _load_preflight()

    if wall.get("stop") == "preflight_abort":
        _write_incomplete(
            "INCOMPLETE (pre-flight abort)",
            wall,
            "Pre-flight failed. H is not decided. Gate seeds were not touched "
            "and remain unspent. This is not a FAIL and is not a park. "
            f"Error: {wall.get('preflight_error', preflight.get('error', 'unknown'))}.",
            extra_lines=[
                "",
                f"Pre-flight record: `benchmarks/results/{PREFLIGHT_PATH.name}`.",
                "Package versions recorded: "
                + ", ".join(
                    f"{k}={v}" for k, v in preflight.get("versions", {}).items() if k != "python"
                )
                + ".",
            ],
        )
        return

    if wall.get("stop") == "calibration_incomplete":
        infeas = wall.get("cells_infeasible", [])
        _write_incomplete(
            "INCOMPLETE (calibration)",
            wall,
            "Calibration was infeasible for at least one gated scenario. "
            "H is not decided. Gate seeds were not touched.",
            extra_lines=["", f"Infeasible scenarios: {infeas}."],
        )
        return

    if (
        wall.get("stop") in ("rehearsal_abort", "watchdog_abort")
        and wall.get("tasks_done", 0) == 0
        and not JSONL.exists()
    ):
        # Rehearsal abort, or a watchdog that tripped during rehearsal
        # (stop is watchdog_abort and no gate task started). Gate seeds unspent.
        during = (
            "rehearsal"
            if wall.get("stop") == "rehearsal_abort" or wall.get("rehearsal_ok") is not True
            else "gate"
        )
        if wall.get("stop") == "rehearsal_abort" or (
            wall.get("stop") == "watchdog_abort" and during == "rehearsal"
        ):
            overall = (
                "ABORTED (rehearsal watchdog)"
                if wall.get("stop") == "watchdog_abort"
                else "ABORTED (rehearsal)"
            )
            note = (
                "Rehearsal failed. H is not decided. Gate seeds were not touched "
                "and remain unspent. This is not a FAIL and is not a park. "
                f"Reason: {wall.get('rehearsal_error', 'unknown')}."
            )
            extra = [
                "",
                "Rehearsal record: `benchmarks/results/v5_2c_specified_gate_rehearsal.jsonl`.",
                f"Rehearsal tasks: {wall.get('rehearsal_tasks_done', 0)}/"
                f"{wall.get('rehearsal_tasks_total', 8)}.",
                f"Pre-flight record: `benchmarks/results/{PREFLIGHT_PATH.name}`.",
            ]
            _write_incomplete(overall, wall, note, extra_lines=extra)
            return

    if wall.get("stop") == "watchdog_abort":
        # Gate watchdog: seeds spent, H not decided, not a FAIL.
        n_done = wall.get("tasks_done", 0)
        note = (
            "The gate watchdog tripped: no task completed within 5 minutes "
            "(or 5 minutes elapsed between completions). Verdict: **ABORTED "
            "(watchdog)**. H is not decided. This is not a FAIL and is not a "
            "park. Because the gate pass had started, the seeds are spent and "
            "will not be reused. Unfinished tasks were not imputed and were "
            "not retried. "
            f"{n_done}/{wall.get('tasks_total', 140)} tasks had completed."
        )
        extra = [
            "",
            f"Pre-flight record: `benchmarks/results/{PREFLIGHT_PATH.name}`.",
            "Rehearsal record: `benchmarks/results/v5_2c_specified_gate_rehearsal.jsonl`.",
            "Any completed gate records are a descriptive record only; they "
            "do not produce a PASS or a FAIL.",
        ]
        if JSONL.exists():
            n_lines = len([x for x in JSONL.read_text().splitlines() if x])
            extra.append(f"Gate jsonl records written: {n_lines}.")
        _write_incomplete("ABORTED (watchdog)", wall, note, extra_lines=extra)
        return

    if not JSONL.exists():
        raise SystemExit(
            "results jsonl is missing and wall stop is not a "
            "pre-flight abort, calibration-INCOMPLETE, rehearsal abort, "
            "or watchdog abort"
        )

    truth = json.loads(TRUTH_FILE.read_text())
    rows = [json.loads(x) for x in JSONL.read_text().splitlines() if x]
    cells = []
    for scen, n in CELLS:
        t = truth[truth_key(scen)]
        if not t.get("feasible", True):
            cells.append(
                {
                    "scen": scen,
                    "n": n,
                    "verdict": "INCOMPLETE",
                    "falsifier": "INCOMPLETE",
                    "note": "calibration infeasible",
                }
            )
            continue
        sh = t["shapley"]
        k = t["graded"][0]
        delta = 0.10 * planted_mag(sh, k)
        rs = sorted([r for r in rows if r["scen"] == scen and r["n"] == n], key=lambda r: r["rep"])
        cell = {
            "scen": scen,
            "n": n,
            "graded_lmoment": k + 1,
            "delta": delta,
            "reps": len(rs),
            "truth": {"comp": sh["comp"][k], "struct": sh["struct"][k]},
            "skey": truth_key(scen),
        }
        if len(rs) < REPS[n]:
            cell["verdict"] = "INCOMPLETE"
            cell["verdict_ci"] = "INCOMPLETE"
            cell["falsifier"] = "INCOMPLETE"
            cell["incomplete_reason"] = "wall_cap"
            cell["note"] = f"{len(rs)}/{REPS[n]} reps finished inside the 30-minute cap"
            cells.append(cell)
            continue
        e5, eriv, names, c_hat, s_hat, e_swap = [], [], [], [], [], []
        for r in rs:
            ls = r[METHOD]["lsplit"]
            e5.append(graded_err(ls, sh, k))
            c_hat.append(ls["comp"][k])
            s_hat.append(ls["struct"][k])
            e_swap.append(graded_err({"comp": ls["struct"], "struct": ls["comp"]}, sh, k))
            best, bn = np.inf, None
            for nm, rv in r["rivals"].items():
                if "error" in rv:
                    continue
                e = graded_err(rv["lsplit"], sh, k)
                if np.isfinite(e) and e < best:
                    best, bn = e, nm
            eriv.append(best)
            names.append(bn)
        e5, eriv = np.array(e5), np.array(eriv)
        mc, ms = float(np.mean(np.abs(c_hat))), float(np.mean(np.abs(s_hat)))
        swap_rescue = float(np.mean(e_swap)) <= float(np.mean(e5))
        fals = "PASS" if (mc > ms and not swap_rescue) else "FAIL"
        n_rival = int(np.isfinite(eriv).sum())
        rival_msgs = sorted(
            {
                f"{nm}: {r['rivals'][nm]['error']}"
                for r in rs
                for nm in ("DFL", "CFM")
                if "error" in r["rivals"].get(nm, {})
            }
        )
        cell.update(
            {
                "mean_err_specified": float(e5.mean()),
                "mean_abs_comp_hat": mc,
                "mean_abs_struct_hat": ms,
                "mean_comp_hat": float(np.mean(c_hat)),
                "mean_struct_hat": float(np.mean(s_hat)),
                "mean_err_roles_swapped": float(np.mean(e_swap)),
                "falsifier": fals,
                "n_rivals_ok": n_rival,
                "rival_errors": rival_msgs,
                "sec_specified_mean": float(np.mean([r[METHOD]["sec"] for r in rs])),
                "sec_cfm_mean": float(np.mean([r["rivals"]["CFM"]["sec"] for r in rs])),
                "sec_dfl_mean": float(np.mean([r["rivals"]["DFL"]["sec"] for r in rs])),
            }
        )
        if n_rival == 0:
            cell["verdict"] = "INCOMPLETE (rivals unavailable)"
            cell["verdict_ci"] = "INCOMPLETE"
            cell["incomplete_reason"] = "rivals"
            cell["note"] = (
                "every DFL and CFM call in this cell recorded an error; "
                "accuracy bars cannot be scored"
            )
            cells.append(cell)
            continue
        d = e5 - eriv
        verd_ci, ci, hi = ci_verdict(d, delta)
        se = d.std(ddof=1) / np.sqrt(len(d))
        dof = len(d) - 1
        p_ni = float(stats.t.sf((delta - d.mean()) / se, dof))
        p_best = float(stats.t.sf((0 - d.mean()) / se, dof))
        rival_err = {}
        rival_c, rival_s = {}, {}
        for nm in ("DFL", "CFM"):
            ev, ac, as_ = [], [], []
            for r in rs:
                rv = r["rivals"][nm]
                if "error" in rv or "lsplit" not in rv:
                    ev.append(np.nan)
                    continue
                ev.append(graded_err(rv["lsplit"], sh, k))
                ls = normalize_lsplit(rv["lsplit"])
                ac.append(abs(ls["comp"][k]))
                as_.append(abs(ls["struct"][k]))
            rival_err[nm] = float(np.nanmean(ev))
            rival_c[nm] = float(np.mean(ac)) if ac else float("nan")
            rival_s[nm] = float(np.mean(as_)) if as_ else float("nan")
        cell.update(
            {
                "mean_err_rivals": rival_err,
                "best_rival_mode": max(set(names), key=names.count) if names else None,
                "mean_d": ci["mean"],
                "upper_t": ci["hi"],
                "upper_boot": ci["boot_hi"],
                "upper_used": hi,
                "verdict_ci": verd_ci,
                "p_ni_unadj": p_ni,
                "p_best_unadj": p_best,
                "rival_mean_abs_comp": rival_c,
                "rival_mean_abs_struct": rival_s,
            }
        )
        cells.append(cell)

    done = [
        c for c in cells if c.get("verdict_ci") not in (None, "INCOMPLETE") and "p_ni_unadj" in c
    ]
    for c, a in zip(done, holm([c["p_ni_unadj"] for c in done]), strict=True):
        c["p_ni_holm"] = a
    for c, a in zip(done, holm([c["p_best_unadj"] for c in done]), strict=True):
        c["p_best_holm"] = a
    for c in cells:
        if "p_ni_holm" in c:
            c["verdict"] = after_holm(
                c["verdict_ci"], c["upper_used"], c["delta"], c["p_best_holm"], c["p_ni_holm"]
            )
        elif "verdict" not in c:
            c["verdict"] = "INCOMPLETE"

    acc = [c["verdict"] for c in cells]
    fal = [c.get("falsifier", "INCOMPLETE") for c in cells]
    wall_incomplete = any(
        c.get("incomplete_reason") == "wall_cap"
        or (c.get("verdict") == "INCOMPLETE" and "cap" in c.get("note", ""))
        for c in cells
    )
    rivals_incomplete = any(
        c.get("incomplete_reason") == "rivals" or "rivals unavailable" in str(c.get("verdict", ""))
        for c in cells
    )
    if wall.get("stop") == "calibration_incomplete":
        overall = "INCOMPLETE (calibration)"
    elif wall.get("stop") == "preflight_abort":
        overall = "INCOMPLETE (pre-flight abort)"
    elif wall.get("stop") == "rehearsal_abort":
        overall = "ABORTED (rehearsal)"
    elif wall.get("stop") == "watchdog_abort":
        overall = "ABORTED (watchdog)"
    elif any(v == "FAIL" for v in acc + fal) or wall_incomplete:
        overall = "FAIL"
    elif rivals_incomplete:
        overall = "INCOMPLETE (rivals unavailable)"
    elif any(v == "INCOMPLETE" for v in acc + fal):
        overall = "FAIL"
    else:
        overall = "PASS"

    summary = {
        "prereg_sha": PREREG_SHA,
        "impl_sha": IMPL_SHA,
        "runner_sha": RUNNER_SHA,
        "overall": overall,
        "cells": cells,
        "wall": wall,
        "preflight": preflight,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    versions = preflight.get("versions", {})
    version_line = ", ".join(f"{k}={v}" for k, v in versions.items() if k != "python")
    L = [
        "# TRACE v5.2c specified-family replication — results",
        "",
        f"**Prereg SHA:** `{PREREG_SHA}` ({PREREG_LINK}).",
        f"**Implementation merge SHA:** `{IMPL_SHA}` (reused, "
        "[PR #58](https://github.com/jgridifier/traceot/pull/58)).",
        f"**Runner SHA:** `{RUNNER_SHA or 'set in the results PR after the runner merge'}`.",
        f"**Gate verdict:** **{overall}**."
        + (" H is not decided." if ("INCOMPLETE" in overall or "ABORTED" in overall) else ""),
        f"**Wall time:** {wall.get('wall_sec', float('nan')):.1f} s on "
        f"{wall.get('workers', '?')} workers (cap {wall.get('cap_sec', 1800)} s); "
        f"{wall.get('tasks_done', 0)}/{wall.get('tasks_total', 140)} tasks finished. "
        "The 30-minute clock started after a successful pre-flight and a successful rehearsal.",
        f"**Pre-flight:** **{'OK' if preflight.get('ok') else 'see record'}**. "
        f"Smoke fits on seed 99: {preflight.get('smoke', {})}. "
        f"Versions: {version_line or 'see pre-flight record'}.",
        "",
        "Bars: PREREG_V4 §5 as inherited by PREREG_V5_2_SPECIFIED §5 and "
        "incorporated by reference in PREREG_V5_2C_SPECIFIED_REPLICATION "
        "(paired d = err_specified − err_best_rival, one-sided 95% upper bound, "
        "t-interval and 10 000-draw bootstrap (seed 2026) must agree, larger used; "
        "BEST if < 0, NON-INFERIOR if < δ = 0.10 × planted magnitude). **After Holm:** "
        "the CI verdict and the matching Holm-adjusted one-sided t p-value "
        "(p BEST for a BEST claim, p NI for a NON-INFERIOR claim) must both hold "
        "(p < 0.05). Rivals DFL and CFM re-run on the same seeds (paired), on the "
        "same centred arrays as earlier gates. The method under test is fit on raw "
        "(uncentred) draws. Role check: mean |ĉ| > mean |ŝ| and swapping roles "
        "does not lower the error.",
        "",
        "| cell | n | p | reps | err specified | err DFL | err CFM | mean d | upper t | "
        "upper boot | δ | verdict | \\|ĉ\\| | \\|ŝ\\| | role | p NI (Holm) | p BEST (Holm) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|---|",
    ]
    for c in cells:
        pp = 5 if c["n"] == 2000 else 20
        if "mean_d" not in c:
            err5 = c.get("mean_err_specified")
            err5s = "—" if err5 is None else f"{err5:.4g}"
            mc = c.get("mean_abs_comp_hat")
            ms = c.get("mean_abs_struct_hat")
            mcs = "—" if mc is None else f"{mc:.4g}"
            mss = "—" if ms is None else f"{ms:.4g}"
            role = c.get("falsifier", "INCOMPLETE")
            L.append(
                f"| {c['scen']} | {c['n']} | {pp} | {c.get('reps', 0)} | {err5s} | "
                f"— | — | — | — | — | {c.get('delta', float('nan')):.4g} | "
                f"**{c['verdict']}** | {mcs} | {mss} | **{role}** | — | — |"
            )
            continue
        L.append(
            f"| {c['scen']} | {c['n']} | {pp} | {c['reps']} | {c['mean_err_specified']:.4g} | "
            f"{c['mean_err_rivals']['DFL']:.4g} | {c['mean_err_rivals']['CFM']:.4g} | "
            f"{c['mean_d']:.4g} | {c['upper_t']:.4g} | {c['upper_boot']:.4g} | "
            f"{c['delta']:.4g} | **{c['verdict']}** | {c['mean_abs_comp_hat']:.4g} | "
            f"{c['mean_abs_struct_hat']:.4g} | **{c['falsifier']}** | "
            f"{c['p_ni_unadj']:.3g} ({c['p_ni_holm']:.3g}) | "
            f"{c['p_best_unadj']:.3g} ({c['p_best_holm']:.3g}) |"
        )
    L += [
        "",
        "p values are unadjusted with Holm (across the completed cells) in brackets.",
        "Full numbers: `benchmarks/results/v5_2c_specified_gate_summary.json`;",
        "per-replication records: `benchmarks/results/v5_2c_specified_gate_shape.jsonl`.",
        "A plain-language page is at"
        " [docs/learn/v5_2c_specified.html](../learn/v5_2c_specified.html).",
        "",
        "## Secondary descriptive report (not a gate bar)",
        "",
        "PREREG_V5_2B_SPECIFIED_REPLICATION §3 (by reference in V5.2c): TRACE-only",
        "numbers, reported for",
        "every cell, including `tail_comp`. This table is **not scored**. It does",
        "not enter Holm, BEST, NON-INFERIOR, the role check, or the gate PASS /",
        "FAIL / park decision. The observation from the V5.2 specified pass that",
        "the `tail_comp` mean composition piece sat well below the planted value",
        "is acknowledged; it is not used to change the hypothesis.",
        "",
        "| cell | n | mean err | mean ĉ | mean ŝ | truth c | truth s | δ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in cells:
        if "mean_err_specified" not in c:
            L.append(
                f"| {c['scen']} | {c['n']} | — | — | — | "
                f"{c.get('truth', {}).get('comp', float('nan')):.4g} | "
                f"{c.get('truth', {}).get('struct', float('nan')):.4g} | "
                f"{c.get('delta', float('nan')):.4g} |"
            )
            continue
        L.append(
            f"| {c['scen']} | {c['n']} | {c['mean_err_specified']:.4g} | "
            f"{c['mean_comp_hat']:.4g} | {c['mean_struct_hat']:.4g} | "
            f"{c['truth']['comp']:.4g} | {c['truth']['struct']:.4g} | "
            f"{c['delta']:.4g} |"
        )
    L += [
        "",
        "## Notes and deviations",
        "",
        "1. **Implementation, reused.** `TRACE_V5_2_SPECIFIED` = "
        '`explain_v5(..., mean_model="specified")` at '
        f"`{IMPL_SHA}`. The simulator and the truth file (oracle seed 99) are",
        "   those already committed with that implementation. They were not",
        "   recomputed. Residual law, grid, quadrature, hybrids, L-moments and",
        "   the Shapley split are as in V5.1.",
        "2. **No column centering** for the method under test, as frozen in the",
        "   parent §2: the design (`x₃²`, `sin(x₅)`) is computed on the raw draws.",
        "   Rivals use the same `centred` arrays as the V5.1 and V5.2 gates.",
        "3. **Holm scoring** is imported from `summarize_v5_2_specified_gate.py`",
        "   and is the convention frozen in the parent §5: both the CI verdict",
        "   and the matching Holm p-value are required.",
        "4. **Pre-flight.** Ran in a spawned child, in the exact gate environment, before any gate",
        "   seed was touched: imports of TRACE, DFL, CFM and scikit-learn; one",
        "   smoke fit of each method on non-gate seed 99; package versions",
        "   recorded. The parent process did not import TRACE. The 30-minute cap",
        "   started after this succeeded and after a successful rehearsal.",
        "5. **Wall cap.** Single pass over all 140 tasks (n = 20 000 cells",
        "   scheduled first) with a 30-minute cap; no task dropped, retried, or",
        "   re-run.",
        "6. **Scope.** A PASS shows only that the v5.1 mixture, with a correctly",
        "   specified location family, meets the inherited bars under this one",
        "   non-linear mean, on these fresh seeds. It does not un-park the V5.2",
        "   spline claim, it does not change V5.1's linear-mean claim, and it",
        "   does not re-run seeds 11000–11119 or 12000–12119. Indices 9600–9619 were not",
        "   touched; 10400–10799 and 12000–12119 were not re-run.",
        "7. **Spawn-safe harness.** Pre-flight ran in a spawned child. The",
        "   rehearsal and gate pools were created with multiprocessing's",
        "   `'spawn'` start method from a parent that never called TRACE.",
        "8. **Rehearsal.** The full parallel pipeline (TRACE, DFL, CFM) ran on",
        "   throwaway seeds 98 and 97, two per cell (8 tasks), at the same n",
        "   and p as the gate cells, before any gate seed was touched.",
        "   Rehearsal outputs are in",
        "   `benchmarks/results/v5_2c_specified_gate_rehearsal.jsonl` and do",
        "   not enter scoring.",
        "9. **Watchdog.** If no task completed within 5 minutes, the pass",
        "   would have been recorded as ABORTED (not FAIL). The 30-minute cap",
        "   is a separate rule and still applies when tasks complete on time.",
    ]
    if overall.startswith("INCOMPLETE (rivals"):
        L += [
            "10. **Rivals unavailable.** Every DFL and CFM call on this pass recorded an",
            "   error. Accuracy bars cannot be scored, and must not be scored as BEST",
            "   against a missing rival. This should not happen after a successful",
            "   pre-flight; if it did, that is a deviation. H is not decided. Gate",
            "   seeds were not retried.",
        ]
    OUT_MD.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
