"""Score the TRACE v5.2 specified-family gate (PREREG_V5_2_SPECIFIED §5).

Reads benchmarks/results/v5_2_specified_gate_shape.jsonl, writes
v5_2_specified_gate_summary.json and docs/benchmark/V5_2_SPECIFIED_GATE_RESULTS.md.
Bars are inherited from PREREG_V4 §5 as scored for V5.1, with the Holm scoring
rule fixed in PREREG_V5_2_SPECIFIED §5: a cell is BEST or NON-INFERIOR only if
both the CI verdict (larger of t and bootstrap bounds) and the matching
Holm-adjusted one-sided t p-value are satisfied (p < 0.05). Failures stay
failures; cells without runs are INCOMPLETE, never imputed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from summarize_v4 import graded_err, normalize_lsplit, planted_mag
from summarize_v5_1_gate import holm, paired_ci

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "benchmarks" / "results"
OUT_MD = ROOT / "docs" / "benchmark" / "V5_2_SPECIFIED_GATE_RESULTS.md"
OUT_JSON = RES / "v5_2_specified_gate_summary.json"
PREREG_SHA = "cc8c1d7afc9c58a8b97b8b3b28b6ad6fd1704920"
# Filled after the implementation PR is merged (order of work §9 step 4).
IMPL_SHA = "1aab4778abf8f84c8c8f7d9603eb8c322f1ba3c9"
REPS = {2000: 50, 20000: 20}
CELLS = [("skew_comp", 2000), ("tail_comp", 2000), ("skew_comp", 20000), ("tail_comp", 20000)]
TRUTH_FILE = RES / "shiftbench_truth_shape_v5_2_specified.json"
METHOD = "TRACE_V5_2_SPECIFIED"


def truth_key(scen: str, kappa: float = 1.0) -> str:
    return f"{scen}|k{kappa:g}"


def ci_verdict(d, delta):
    """CI-only verdict (report also); the larger bound is used."""
    ci = paired_ci(d)
    hi = max(ci["hi"], ci["boot_hi"])
    if not np.isfinite(hi):
        return "INCOMPLETE", ci, hi
    if hi < 0:
        return "BEST", ci, hi
    if hi < delta:
        return "NON-INFERIOR", ci, hi
    return "FAIL", ci, hi


def after_holm(ci_verd: str, hi: float, delta: float, p_best_holm: float, p_ni_holm: float) -> str:
    """PREREG_V5_2_SPECIFIED §5 scoring convention.

    A BEST (or NON-INFERIOR) claim requires both the CI verdict and the matching
    Holm-adjusted one-sided t p-value < 0.05. The CI-only verdict cannot rescue a
    cell that fails this rule. If BEST fails Holm but NON-INFERIOR holds on both
    the bound and p_NI, the cell is NON-INFERIOR.
    """
    if ci_verd == "INCOMPLETE":
        return "INCOMPLETE"
    if ci_verd == "BEST" and p_best_holm < 0.05:
        return "BEST"
    if hi < delta and p_ni_holm < 0.05 and ci_verd in ("BEST", "NON-INFERIOR"):
        return "NON-INFERIOR"
    return "FAIL"


def main():
    wall_path = RES / "v5_2_specified_gate_wall.json"
    wall = json.loads(wall_path.read_text()) if wall_path.exists() else {}
    truth = json.loads(TRUTH_FILE.read_text())
    jsonl = RES / "v5_2_specified_gate_shape.jsonl"
    if wall.get("stop") == "calibration_incomplete" or not jsonl.exists():
        infeas = wall.get("cells_infeasible", [])
        summary = {
            "prereg_sha": PREREG_SHA,
            "impl_sha": IMPL_SHA,
            "overall": "INCOMPLETE (calibration)",
            "cells_infeasible": infeas,
            "wall": wall,
            "note": (
                "Calibration was infeasible for at least one gated scenario. "
                "H is not decided. Gate seeds were not touched."
            ),
        }
        OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")
        L = [
            "# TRACE v5.2 specified-family gate — results",
            "",
            f"**Prereg SHA:** `{PREREG_SHA}` "
            "([PREREG_V5_2_SPECIFIED.md](../research/PREREG_V5_2_SPECIFIED.md)).",
            f"**Implementation merge SHA:** `{IMPL_SHA}`.",
            "**Gate verdict:** **INCOMPLETE (calibration)**. H is not decided.",
            "",
            f"Infeasible scenarios: {infeas}. Per PREREG_V5_2_SPECIFIED §3, no gate",
            "index was touched, and the DGP was not redesigned.",
        ]
        OUT_MD.write_text("\n".join(L) + "\n")
        print("\n".join(L))
        return

    rows = [json.loads(x) for x in jsonl.read_text().splitlines() if x]
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
        rival_msgs = sorted({
            f"{nm}: {r['rivals'][nm]['error']}"
            for r in rs for nm in ("DFL", "CFM")
            if "error" in r["rivals"].get(nm, {})
        })
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
            # TRACE finished; rivals recorded only errors. Accuracy bars cannot
            # be scored (and must not be scored as BEST against an infinite
            # rival error). Role check is still scored. H is not decided on
            # the accuracy bars. This is not a wall-cap INCOMPLETE.
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
        c.get("incomplete_reason") == "rivals"
        or "rivals unavailable" in str(c.get("verdict", ""))
        for c in cells
    )
    if wall.get("stop") == "calibration_incomplete":
        overall = "INCOMPLETE (calibration)"
    elif any(v == "FAIL" for v in acc + fal) or wall_incomplete:
        # Wall-cap INCOMPLETE, accuracy FAIL, or role-check FAIL: H is rejected.
        overall = "FAIL"
    elif rivals_incomplete:
        # Harness did not produce rival numbers. H is not decided. Not a park.
        overall = "INCOMPLETE (rivals unavailable)"
    elif any(v == "INCOMPLETE" for v in acc + fal):
        overall = "FAIL"
    else:
        overall = "PASS"

    summary = {
        "prereg_sha": PREREG_SHA,
        "impl_sha": IMPL_SHA,
        "overall": overall,
        "cells": cells,
        "wall": wall,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    L = [
        "# TRACE v5.2 specified-family gate — results",
        "",
        f"**Prereg SHA:** `{PREREG_SHA}` "
        "([PREREG_V5_2_SPECIFIED.md](../research/PREREG_V5_2_SPECIFIED.md)).",
        f"**Implementation merge SHA:** `{IMPL_SHA}`.",
        f"**Gate verdict:** **{overall}**."
        + (
            " H is not decided."
            if "INCOMPLETE" in overall
            else ""
        ),
        f"**Wall time:** {wall.get('wall_sec', float('nan')):.1f} s on "
        f"{wall.get('workers', '?')} workers (cap {wall.get('cap_sec', 1800)} s); "
        f"{wall.get('tasks_done', 0)}/{wall.get('tasks_total', 140)} tasks finished.",
        "",
        "Bars: PREREG_V4 §5 as inherited by PREREG_V5_2_SPECIFIED §5 (paired "
        "d = err_specified − err_best_rival, one-sided 95% upper bound, t-interval",
        "and 10 000-draw bootstrap (seed 2026) must agree, larger used; BEST if < 0,",
        "NON-INFERIOR if < δ = 0.10 × planted magnitude). **After Holm:** the CI",
        "verdict and the matching Holm-adjusted one-sided t p-value (p BEST for a",
        "BEST claim, p NI for a NON-INFERIOR claim) must both hold (p < 0.05).",
        "Rivals DFL and CFM re-run on the same seeds (paired), on the same centred",
        "arrays as earlier gates. The method under test is fit on raw (uncentred)",
        "draws (PREREG_V5_2_SPECIFIED §2). Role check: mean |ĉ| > mean |ŝ| and",
        "swapping roles does not lower the error.",
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
        "Full numbers: `benchmarks/results/v5_2_specified_gate_summary.json`;",
        "per-replication records: `benchmarks/results/v5_2_specified_gate_shape.jsonl`.",
        "",
        "## Notes and deviations",
        "",
        "1. **Implementation.** `TRACE_V5_2_SPECIFIED` = "
        '`explain_v5(..., mean_model="specified")`.',
        "   Only the location fit changed; residual law, grid, quadrature, hybrids,",
        "   L-moments and the Shapley split are as in V5.1. Gate seeds are run only after",
        "   this implementation is merged, and only if calibration of σ_B was feasible.",
        "2. **No column centering** for the method under test, as frozen in §2: the design",
        "   (`x₃²`, `sin(x₅)`) is computed on the raw draws. Rivals use the same `centred`",
        "   arrays as the V5.1 and V5.2 gates, so their settings are unchanged.",
        "3. **Holm scoring** follows the convention fixed in §5 to avoid the V5.2 note-3",
        "   ambiguity: both the CI verdict and the matching Holm p-value are required.",
        "4. **Wall cap.** Single pass over all 140 tasks (n = 20 000 cells scheduled first)",
        "   with a 30-minute cap; no task dropped, retried, or re-run.",
        "5. **Scope (PREREG_V5_2_SPECIFIED §7–8).** A PASS shows only that the v5.1",
        "   mixture, with a correctly specified location family, meets the inherited bars",
        "   under this one non-linear mean. It does not un-park the V5.2 spline claim, and",
        "   it does not change V5.1's linear-mean claim. Indices 9600–9619 were not",
        "   touched; 10400–10799 were not used.",
    ]
    if overall.startswith("INCOMPLETE (rivals"):
        L += [
            "6. **Rivals unavailable.** Every DFL and CFM call on this pass recorded an",
            "   error. Accuracy bars (BEST / NON-INFERIOR after Holm) cannot be scored,",
            "   and must not be scored as BEST against a missing rival. Role checks are",
            "   scored from TRACE alone and are reported. H is not decided. This is not",
            "   a wall-cap INCOMPLETE and is not a park. Gate seeds were not retried.",
        ]
    OUT_MD.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
