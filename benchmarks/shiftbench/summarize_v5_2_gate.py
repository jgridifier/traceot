"""Score the TRACE v5.2 gate (PREREG_V5_2 §5, bars inherited from PREREG_V4 §5).

Reads benchmarks/results/v5_2_gate_shape.jsonl, writes v5_2_gate_summary.json. The
CI / Holm helpers are imported unchanged from the V5.1 scorer. Failures stay failures;
cells without (all) runs are INCOMPLETE, never imputed.

Rules as written:
* paired d = err_v5.2 - err_best_rival (best = min of DFL, CFM per rep); one-sided 95%
  upper bound, t (n-1 df) and 10 000-draw bootstrap (seed 2026), larger used;
  BEST < 0, NON-INFERIOR < delta = 0.10 x planted magnitude (v5.2 truth file).
* Holm across all 16 accuracy tests (INCOMPLETE cells enter the family with p = 1, so
  the family size stays 16); a cell passes accuracy only if its CI verdict is BEST or
  NON-INFERIOR **and** the matching Holm-adjusted one-sided p (p_best for BEST, p_ni
  for NON-INFERIOR) is < 0.05.
* Role check (1-12): mean |c| > mean |s| and the role-swapped error is not lower;
  (13-16) mirrored: mean |s| > mean |c| and swapping does not lower the error.
* False attribution (13-16): upper 95% bound (t and bootstrap, larger) of mean |c_r|
  < delta; one-sided t p values, Holm across the 4.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from run_v5_2_gate import CELLS, PREREG_SHA, SIZES
from sb_dgp_shape_v5_2 import key
from scipy import stats
from summarize_v4 import graded_err, normalize_lsplit, planted_mag
from summarize_v5_1_gate import holm, paired_ci

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "benchmarks" / "results"
OUT_JSON = RES / "v5_2_gate_summary.json"
ALPHA = 0.05


def p_upper(v, margin):
    """One-sided t p value for H0: mean(v) >= margin."""
    v = np.asarray(v, float)
    se = v.std(ddof=1) / np.sqrt(len(v))
    return float(stats.t.sf((margin - v.mean()) / se, len(v) - 1))


def score_cell(cell, fam, scen, kappa, n, rows, truth):
    t = truth[key(scen, kappa)]
    reps = len(SIZES[n][1])
    c = {"cell": cell, "family": fam, "scen": scen, "kappa": kappa, "n": n, "p": SIZES[n][0]}
    if not t.get("feasible", False):
        c.update(
            verdict="INCOMPLETE",
            role="INCOMPLETE",
            reps=0,
            note="sigma_B calibration infeasible (PREREG_V5_2 §3): lambda_2(Y_B) > lambda_2(Y_A) "
            "already at sigma_B -> 0+",
        )
        if fam == "struct":
            c["false_attr"] = "INCOMPLETE"
        return c
    sh, k = t["shapley"], t["graded"][0]
    delta = 0.10 * planted_mag(sh, k)
    c.update(
        graded_lmoment=k + 1, delta=delta, truth={"comp": sh["comp"][k], "struct": sh["struct"][k]}
    )
    rs = sorted([r for r in rows if r["cell"] == cell], key=lambda r: r["rep"])
    c["reps"] = len(rs)
    if len(rs) < reps:
        c.update(
            verdict="INCOMPLETE", role="INCOMPLETE", note=f"{len(rs)}/{reps} reps inside the cap"
        )
        if fam == "struct":
            c["false_attr"] = "INCOMPLETE"
        return c
    e2, eriv, names, ch, sh_hat, eswap, e1 = [], [], [], [], [], [], []
    for r in rs:
        ls = r["TRACE_V5_2"]["lsplit"]
        e2.append(graded_err(ls, sh, k))
        ch.append(ls["comp"][k])
        sh_hat.append(ls["struct"][k])
        eswap.append(graded_err({"comp": ls["struct"], "struct": ls["comp"]}, sh, k))
        if "TRACE_V5_1" in r:
            e1.append(graded_err(r["TRACE_V5_1"]["lsplit"], sh, k))
        best, bn = np.inf, None
        for nm, rv in r["rivals"].items():
            if "error" in rv:
                continue
            e = graded_err(rv["lsplit"], sh, k)
            if np.isfinite(e) and e < best:
                best, bn = e, nm
        eriv.append(best)
        names.append(bn)
    e2, eriv = np.array(e2), np.array(eriv)
    d = e2 - eriv
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
    swap_lower = float(np.mean(eswap)) < float(np.mean(e2))
    dominant_ok = (mc > ms) if fam != "struct" else (ms > mc)
    c.update(
        mean_err_v5_2=float(e2.mean()),
        mean_err_rivals={
            nm: float(np.nanmean([graded_err(r["rivals"][nm].get("lsplit"), sh, k) for r in rs]))
            for nm in ("DFL", "CFM")
        },
        best_rival_mode=max(set(names), key=names.count),
        mean_d=ci["mean"],
        upper_t=ci["hi"],
        upper_boot=ci["boot_hi"],
        upper_used=hi,
        ci_verdict=ci_verdict,
        p_ni_unadj=p_upper(d, delta),
        p_best_unadj=p_upper(d, 0.0),
        mean_abs_comp_hat=mc,
        mean_abs_struct_hat=ms,
        mean_comp_hat=float(np.mean(ch)),
        mean_struct_hat=float(np.mean(sh_hat)),
        mean_err_roles_swapped=float(np.mean(eswap)),
        role="PASS" if (dominant_ok and not swap_lower) else "FAIL",
        rival_mean_abs_comp={
            nm: float(
                np.mean([abs(normalize_lsplit(r["rivals"][nm]["lsplit"])["comp"][k]) for r in rs])
            )
            for nm in ("DFL", "CFM")
        },
        dropped_cols_any=int(
            sum(
                bool(r["TRACE_V5_2"]["dropped_cols_a"] or r["TRACE_V5_2"]["dropped_cols_b"])
                for r in rs
            )
        ),
        sec_v5_2_mean=float(np.mean([r["TRACE_V5_2"]["sec"] for r in rs])),
        sec_dfl_mean=float(np.mean([r["rivals"]["DFL"]["sec"] for r in rs])),
        sec_cfm_mean=float(np.mean([r["rivals"]["CFM"]["sec"] for r in rs])),
    )
    if e1:
        c["secondary_mean_err_v5_1"] = float(np.mean(e1))
    if fam == "struct":
        a = np.abs(ch)
        fa = paired_ci(a)
        c["false_attr_upper_t"] = fa["hi"]
        c["false_attr_upper_boot"] = fa["boot_hi"]
        c["false_attr_upper"] = max(fa["hi"], fa["boot_hi"])
        c["false_attr_p_unadj"] = p_upper(a, delta)
    return c


def main():
    rows = [json.loads(x) for x in (RES / "v5_2_gate_shape.jsonl").read_text().splitlines() if x]
    truth = json.loads((RES / "shiftbench_truth_shape_v5_2.json").read_text())
    cells = [score_cell(*cl, rows, truth) for cl in CELLS]
    # Holm across all 16 accuracy tests; INCOMPLETE enters with p = 1
    for c, a, b in zip(
        cells,
        holm([c.get("p_ni_unadj", 1.0) for c in cells]),
        holm([c.get("p_best_unadj", 1.0) for c in cells]),
        strict=True,
    ):
        if "ci_verdict" not in c:
            continue
        c["p_ni_holm"], c["p_best_holm"] = a, b
        if c["ci_verdict"] == "BEST" and b < ALPHA:
            c["verdict"] = "BEST"
        elif c["ci_verdict"] in ("BEST", "NON-INFERIOR") and a < ALPHA:
            c["verdict"] = "NON-INFERIOR"
        elif c["ci_verdict"] == "INCOMPLETE":
            c["verdict"] = "INCOMPLETE"
        else:
            c["verdict"] = "FAIL"
    st = [c for c in cells if c["family"] == "struct"]
    for c, a in zip(st, holm([c.get("false_attr_p_unadj", 1.0) for c in st]), strict=True):
        if "false_attr_upper" not in c:
            continue
        c["false_attr_p_holm"] = a
        c["false_attr"] = "PASS" if c["false_attr_upper"] < c["delta"] else "FAIL"
    checks = (
        [c["verdict"] for c in cells] + [c["role"] for c in cells] + [c["false_attr"] for c in st]
    )
    if any(v == "FAIL" for v in checks):
        overall = "FAIL"
    elif any(v == "INCOMPLETE" for v in checks):
        overall = "FAIL (INCOMPLETE cells; kill rule PREREG_V5_2 §6)"
    else:
        overall = "PASS"
    wall = json.loads((RES / "v5_2_gate_wall.json").read_text())
    if wall["wall_sec"] > wall["cap_sec"] or wall["tasks_done"] < wall["tasks_total"]:
        overall = "FAIL" if overall == "PASS" else overall
    summary = {"prereg_sha": PREREG_SHA, "overall": overall, "wall": wall, "cells": cells}
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"overall": overall, "wall": wall}, indent=1))
    for c in cells:
        print(
            c["cell"],
            c["scen"],
            c["kappa"],
            c["n"],
            c.get("verdict"),
            c.get("role"),
            c.get("false_attr", ""),
            {
                k: round(c[k], 5)
                for k in ("mean_err_v5_2", "mean_d", "upper_used", "delta")
                if k in c
            },
        )


if __name__ == "__main__":
    main()
