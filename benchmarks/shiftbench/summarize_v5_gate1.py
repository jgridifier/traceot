"""Score the TRACE v5 first gate (PREREG_V5 §5, bars inherited from PREREG_V4 §5).

Reads benchmarks/results/v5_gate1_shape.jsonl, writes v5_gate1_summary.json and
docs/benchmark/V5_GATE1_RESULTS.md. The CI / verdict code is imported unchanged from
summarize_v4.py (identical protocol). Failures stay failures; cells without runs are
INCOMPLETE, never imputed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from summarize_v4 import cell_verdict, graded_err, normalize_lsplit, planted_mag

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "benchmarks" / "results"
OUT_MD = ROOT / "docs" / "benchmark" / "V5_GATE1_RESULTS.md"
OUT_JSON = RES / "v5_gate1_summary.json"
PREREG_SHA = "bce021e"
CELLS = [("skew_comp", 2000), ("tail_comp", 2000), ("skew_comp", 20000), ("tail_comp", 20000)]


def holm(pvals):
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def main():
    rows = [json.loads(x) for x in (RES / "v5_gate1_shape.jsonl").read_text().splitlines() if x]
    truth = json.loads((RES / "shiftbench_truth_shape.json").read_text())
    cells = []
    for scen, n in CELLS:
        t = truth[scen]
        sh = t["shapley"]
        k = t["graded"][0]
        delta = 0.10 * planted_mag(sh, k)
        rs = sorted([r for r in rows if r["scen"] == scen and r["n"] == n], key=lambda r: r["rep"])
        cell = {"scen": scen, "n": n, "graded_lmoment": k + 1, "delta": delta, "reps": len(rs)}
        cell["truth"] = {"comp": sh["comp"][k], "struct": sh["struct"][k]}
        if len(rs) < 2:
            cell["verdict"] = "INCOMPLETE"
            cell["falsifier"] = "INCOMPLETE"
            cell["note"] = "indices 9600-9619 held back by the owner; not run"
            cells.append(cell)
            continue
        e5, eriv, names, c_hat, s_hat, e_swap = [], [], [], [], [], []
        for r in rs:
            ls = r["TRACE_V5"]["lsplit"]
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
        d = e5 - eriv
        verd, ci, hi = cell_verdict(d, delta)
        se = d.std(ddof=1) / np.sqrt(len(d))
        p_ni = float(stats.norm.sf((delta - d.mean()) / se))  # H0: mean d >= delta
        p_best = float(stats.norm.sf((0 - d.mean()) / se))  # H0: mean d >= 0
        rival_err = {}
        for nm in ("DFL", "CFM"):
            ev = [graded_err(r["rivals"][nm].get("lsplit"), sh, k) for r in rs]
            rival_err[nm] = float(np.nanmean(ev))
        rival_c = {
            nm: float(
                np.mean([abs(normalize_lsplit(r["rivals"][nm]["lsplit"])["comp"][k]) for r in rs])
            )
            for nm in ("DFL", "CFM")
        }
        rival_s = {
            nm: float(
                np.mean([abs(normalize_lsplit(r["rivals"][nm]["lsplit"])["struct"][k]) for r in rs])
            )
            for nm in ("DFL", "CFM")
        }
        mc, ms = float(np.mean(np.abs(c_hat))), float(np.mean(np.abs(s_hat)))
        swap_rescue = float(np.mean(e_swap)) <= float(np.mean(e5))
        fals = "PASS" if (mc > ms and not swap_rescue) else "FAIL"
        cell.update(
            {
                "mean_err_v5": float(e5.mean()),
                "mean_err_rivals": rival_err,
                "best_rival_mode": max(set(names), key=names.count),
                "mean_d": ci["mean"],
                "upper_t": ci["hi"],
                "upper_boot": ci["boot_hi"],
                "upper_used": hi,
                "verdict": verd,
                "p_ni_unadj": p_ni,
                "p_best_unadj": p_best,
                "mean_abs_comp_hat": mc,
                "mean_abs_struct_hat": ms,
                "mean_comp_hat": float(np.mean(c_hat)),
                "mean_struct_hat": float(np.mean(s_hat)),
                "rival_mean_abs_comp": rival_c,
                "rival_mean_abs_struct": rival_s,
                "mean_err_v5_roles_swapped": float(np.mean(e_swap)),
                "falsifier": fals,
                "sec_v5_mean": float(np.mean([r["TRACE_V5"]["sec"] for r in rs])),
                "sec_cfm_mean": float(np.mean([r["rivals"]["CFM"]["sec"] for r in rs])),
                "sec_dfl_mean": float(np.mean([r["rivals"]["DFL"]["sec"] for r in rs])),
            }
        )
        cells.append(cell)
    done = [c for c in cells if c["verdict"] != "INCOMPLETE"]
    for c, a in zip(done, holm([c["p_ni_unadj"] for c in done]), strict=True):
        c["p_ni_holm"] = a
    for c, a in zip(done, holm([c["p_best_unadj"] for c in done]), strict=True):
        c["p_best_holm"] = a
    acc = [c["verdict"] for c in cells]
    fal = [c["falsifier"] for c in cells]
    if any(v == "FAIL" for v in acc + fal):
        overall = "FAIL"
    elif any(v == "INCOMPLETE" for v in acc + fal):
        overall = "INCOMPLETE"
    else:
        overall = "PASS"
    summary = {"prereg_sha": PREREG_SHA, "overall": overall, "cells": cells}
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    L = [
        "# TRACE v5 first gate — results",
        "",
        f"**Prereg SHA:** `{PREREG_SHA}` ([PREREG_V5.md](../research/PREREG_V5.md)).",
        f"**First-gate verdict:** **{overall}**.",
        "**Full scored suite: not started.**",
        "",
        "Bars: PREREG_V4 §5 as inherited by PREREG_V5 §5 (paired d = err_v5 − err_best_rival,",
        "one-sided 95% upper bound, normal and 10 000-draw bootstrap (seed 2026) must agree;",
        "BEST if < 0, NON-INFERIOR if < δ = 0.10 × planted magnitude). Rivals DFL and CFM",
        "re-run on the same seeds (paired).",
        "",
        "| cell | n | reps | err v5 | err DFL | err CFM | mean d | upper 95% | δ | verdict "
        "| \\|ĉ\\| | \\|ŝ\\| | falsifier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for c in cells:
        if c["verdict"] == "INCOMPLETE":
            L.append(
                f"| {c['scen']} | {c['n']} | 0 | — | — | — | — | — | {c['delta']:.4g} | "
                "INCOMPLETE | — | — | INCOMPLETE |"
            )
            continue
        L.append(
            f"| {c['scen']} | {c['n']} | {c['reps']} | {c['mean_err_v5']:.4g} | "
            f"{c['mean_err_rivals']['DFL']:.4g} | {c['mean_err_rivals']['CFM']:.4g} | "
            f"{c['mean_d']:.4g} | {c['upper_used']:.4g} | {c['delta']:.4g} | "
            f"**{c['verdict']}** | {c['mean_abs_comp_hat']:.4g} | "
            f"{c['mean_abs_struct_hat']:.4g} | **{c['falsifier']}** |"
        )
    L += [
        "",
        "Truth (Shapley, graded channel): "
        + "; ".join(
            f"{c['scen']} λ{c['graded_lmoment']}: comp {c['truth']['comp']:.4g}, "
            f"struct {c['truth']['struct']:.4g}"
            for c in cells[:2]
        )
        + ".",
        "",
        "Holm-adjusted p-values (completed cells only) are in "
        "`benchmarks/results/v5_gate1_summary.json`.",
        "",
        "## Deviations and notes",
        "",
        "1. **n = 20 000 cells not run.** PREREG_V5 §3 assigns indices 9600–9619 to the",
        "   n = 20 000 cells of this gate. The owner instructed that these indices are",
        "   held back and must not be touched in this run, so both n = 20 000 cells are",
        "   **INCOMPLETE**. Under §8 an INCOMPLETE required cell blocks a first-gate PASS.",
        "2. **Falsifier operationalisation.** §5 states the rule in words. Applied as: PASS",
        "   iff mean |ĉ| > mean |ŝ| on the graded channel (truth puts the mass in",
        "   composition) **and** swapping the two roles would not lower the error.",
        "3. **Conditional model.** PREREG_V5 §2.1 leaves the conditional family to in-repo",
        "   defaults. v5 uses the v4 binned mean booster (`FitL2Config::default().mean`,",
        "   unchanged) plus the period's empirical residual law, a 256-point grid on the",
        "   exact support (v4 default size), and exact L-moments of the gridded law. Fixed",
        "   before any gate output was seen; no tuning.",
        "4. **Per-feature game not scored.** The scored estimand is the role split of the",
        "   graded L-moment; per-feature shares on the mixture law are not implemented in",
        "   this PR.",
        "5. **Holm.** Applied across the completed cells' one-sided tests; with two",
        "   INCOMPLETE cells the family has two members instead of four.",
    ]
    OUT_MD.write_text("\n".join(L) + "\n")
    print("\n".join(L[:20]))


if __name__ == "__main__":
    main()
