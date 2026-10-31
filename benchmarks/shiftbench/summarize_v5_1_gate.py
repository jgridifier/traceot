"""Score the TRACE v5.1 gate (PREREG_V5_1 §4, bars inherited from PREREG_V4 §5).

Reads benchmarks/results/v5_1_gate_shape.jsonl, writes v5_1_gate_summary.json and
docs/benchmark/V5_1_GATE_RESULTS.md. The CI / verdict code is imported unchanged from
summarize_v4.py (identical protocol). Failures stay failures; cells without runs are
INCOMPLETE, never imputed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from summarize_v4 import graded_err, normalize_lsplit, planted_mag

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "benchmarks" / "results"
OUT_MD = ROOT / "docs" / "benchmark" / "V5_1_GATE_RESULTS.md"
OUT_JSON = RES / "v5_1_gate_summary.json"
PREREG_SHA = "b33f507"
IMPL_SHA = "665939bd4aa78ffc7474b91d482d18ac0191d595"
REPS = {2000: 50, 20000: 20}
CELLS = [("skew_comp", 2000), ("tail_comp", 2000), ("skew_comp", 20000), ("tail_comp", 20000)]


def paired_ci(diff, alpha=0.05):
    """One-sided 95% upper bound: Student t (n-1 df) and 10 000-draw percentile
    bootstrap (seed 2026), as written in PREREG_V4 §5."""
    diff = np.asarray(diff, float)
    n = len(diff)
    m = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(n))
    hi = m + float(stats.t.ppf(1 - alpha, n - 1)) * se
    rng = np.random.default_rng(2026)
    boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(10_000)])
    return {"mean": m, "hi": hi, "boot_hi": float(np.quantile(boots, 1 - alpha)), "n": n}


def cell_verdict(d, delta):
    """Both bounds must agree; the larger is used (PREREG_V5_1 §4)."""
    ci = paired_ci(d)
    hi = max(ci["hi"], ci["boot_hi"])
    if not np.isfinite(hi):
        return "INCOMPLETE", ci, hi
    if hi < 0:
        return "BEST", ci, hi
    if hi < delta:
        return "NON-INFERIOR", ci, hi
    return "FAIL", ci, hi


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
    rows = [json.loads(x) for x in (RES / "v5_1_gate_shape.jsonl").read_text().splitlines() if x]
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
        if len(rs) < REPS[n]:
            cell["verdict"] = "INCOMPLETE"
            cell["falsifier"] = "INCOMPLETE"
            cell["note"] = f"{len(rs)}/{REPS[n]} reps finished inside the 30-minute cap"
            cells.append(cell)
            continue
        e5, eriv, names, c_hat, s_hat, e_swap = [], [], [], [], [], []
        for r in rs:
            ls = r["TRACE_V5_1"]["lsplit"]
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
        dof = len(d) - 1
        p_ni = float(stats.t.sf((delta - d.mean()) / se, dof))  # H0: mean d >= delta
        p_best = float(stats.t.sf((0 - d.mean()) / se, dof))  # H0: mean d >= 0
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
                "sec_v5_mean": float(np.mean([r["TRACE_V5_1"]["sec"] for r in rs])),
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

    wall = json.loads((RES / "v5_1_gate_wall.json").read_text())
    summary["wall"] = wall
    summary["impl_sha"] = IMPL_SHA
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    L = [
        "# TRACE v5.1 gate — results",
        "",
        f"**Prereg SHA:** `{PREREG_SHA}` ([PREREG_V5_1.md](../research/PREREG_V5_1.md)).",
        f"**Implementation merge SHA:** `{IMPL_SHA}`.",
        f"**Gate verdict:** **{overall}**.",
        f"**Wall time:** {wall['wall_sec']:.1f} s on {wall['workers']} workers "
        f"(cap {wall['cap_sec']} s); {wall['tasks_done']}/{wall['tasks_total']} tasks finished.",
        "",
        "Bars: PREREG_V4 §5 as inherited by PREREG_V5_1 §4 (paired d = err_v5.1 − err_best_rival,",
        "one-sided 95% upper bound, t-interval and 10 000-draw bootstrap (seed 2026) must "
        "agree, larger used;",
        "BEST if < 0, NON-INFERIOR if < δ = 0.10 × planted magnitude). Rivals DFL and CFM",
        "re-run on the same seeds (paired). Role check: mean |ĉ| > mean |ŝ| and swapping",
        "roles does not lower the error (V5_GATE1_RESULTS note 2).",
        "",
        "| cell | n | p | reps | err v5.1 | err DFL | err CFM | mean d | upper t | upper boot "
        "| δ | verdict | \\|ĉ\\| | \\|ŝ\\| | role | p NI (Holm) | p BEST (Holm) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|---|",
    ]
    for c in cells:
        pp = 5 if c["n"] == 2000 else 20
        if c["verdict"] == "INCOMPLETE":
            L.append(
                f"| {c['scen']} | {c['n']} | {pp} | {c['reps']} | — | — | — | — | — | — | "
                f"{c['delta']:.4g} | INCOMPLETE | — | — | INCOMPLETE | — | — |"
            )
            continue
        L.append(
            f"| {c['scen']} | {c['n']} | {pp} | {c['reps']} | {c['mean_err_v5']:.4g} | "
            f"{c['mean_err_rivals']['DFL']:.4g} | {c['mean_err_rivals']['CFM']:.4g} | "
            f"{c['mean_d']:.4g} | {c['upper_t']:.4g} | {c['upper_boot']:.4g} | "
            f"{c['delta']:.4g} | **{c['verdict']}** | {c['mean_abs_comp_hat']:.4g} | "
            f"{c['mean_abs_struct_hat']:.4g} | **{c['falsifier']}** | "
            f"{c['p_ni_unadj']:.3g} ({c['p_ni_holm']:.3g}) | "
            f"{c['p_best_unadj']:.3g} ({c['p_best_holm']:.3g}) |"
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
        "p values are unadjusted with Holm (across the completed cells) in brackets.",
        "Full numbers: `benchmarks/results/v5_1_gate_summary.json`; per-replication",
        "records: `benchmarks/results/v5_1_gate_shape.jsonl`.",
        "",
        "## Notes and deviations",
        "",
        '1. **Implementation.** `TRACE_V5_1` = `explain_v5(..., mean_model="ols")`',
        "   (PR #29). Only the location fit changed; grid, quadrature, hybrids, L-moments and",
        "   the Shapley split are as at `3d695bc`. Gate seeds were first run after the merge.",
        "2. **t-interval.** PREREG_V4 §5 specifies a t-interval; the earlier v4/v5 scorers",
        "   used a normal quantile. This scorer uses Student t with n−1 df (and t-based",
        "   one-sided p values), as written in the pre-registration. The t bound is wider,",
        "   so the change can only make a cell harder to pass.",
        "3. **Wall cap.** Single pass over all 140 tasks (n=20 000 cells scheduled first)",
        "   with a 30-minute cap; no task was dropped, retried or re-run.",
        "4. **Role check** as in V5_GATE1_RESULTS note 2: mean |ĉ| > mean |ŝ| on the graded",
        "   channel and the role-swapped error is not lower.",
        "5. **Holm** across all four cells (family of four, as pre-registered).",
        "6. **Scope (PREREG_V5_1 §7).** The simulator's mean is linear, so the linear",
        "   location model is the right family by construction. This PASS says the v5",
        "   mixture works once the location model is right; it says nothing about",
        "   non-linear means, which need their own pre-registration. No other suite",
        "   (struct cells, coverage, ShiftBench, speed) was run under this SHA.",
    ]
    OUT_MD.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
