# ruff: noqa: E501, E731, B007, B023, F841
"""Score the TRACE v3 runs (PREREG_TRACE_V3.md). Writes benchmarks/results/trace_v3_summary.json
and docs/benchmark/TRACE_V3_RESULTS.md (tables; a hand-written reading is appended after)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import sb_dgp as dgp
import sb_dgp_shape as shp
from run_synthetic import COVER_SCEN, RES
from run_trace_v3 import COVER_REPS, SHAPE_GRID
from sb_h8 import joint_reject
from scipy.stats import beta, ttest_ind

TRUTH = json.loads((RES / "shiftbench_truth.json").read_text())
TSHAPE = json.loads((RES / "shiftbench_truth_shape.json").read_text())
BASE = json.loads((RES / "trace_claims_summary.json").read_text())  # PR #12
BEST = {
    "skew_struct": "DFL",
    "tail_struct": "DFL",
    "skew_comp": "CFM",
    "tail_comp": "CFM",
    "shape_mixed": "CFM",
}
PLANTED = {
    "skew_struct": ("struct",),
    "tail_struct": ("struct",),
    "skew_comp": ("comp",),
    "tail_comp": ("comp",),
    "shape_mixed": ("comp", "struct"),
}
DOC = Path(__file__).resolve().parents[2] / "docs/benchmark/TRACE_V3_RESULTS.md"


def jl(p):
    p = Path(p)
    out = []
    if p.exists():
        for x in p.read_text().splitlines():
            try:
                out.append(json.loads(x))
            except json.JSONDecodeError:
                pass
    return out


def cp(k, n):
    if n == 0:
        return [0.0, 1.0]
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return [float(lo), float(hi)]


def rate_verdict(k, m, n_plan, bar):
    if m >= n_plan:
        return "PASS" if k / m >= bar else "FAIL"
    if k >= bar * n_plan:
        return "PASS"
    if k + (n_plan - m) < bar * n_plan:
        return "FAIL"
    return "INCOMPLETE"


def h7():
    recs = jl(RES / "trace_v3_shape.jsonl")
    old = jl(RES / "trace_claims_shape.jsonl")
    by = defaultdict(list)
    for r in recs:
        by[(r["scen"], r["n"])].append(r)
    out = {"recovery": {}, "rival": {}, "errors": {}}
    tests = []
    for sc in shp.SHAPE_SCENARIOS:
        T = TSHAPE[sc]
        k = T["graded"][0]
        tsh, tar = T["shapley"], T["aref"]
        mag = abs(tsh["comp"][k]) + abs(tsh["struct"][k])
        for n, p, reps in SHAPE_GRID:
            hits, errs = [], []
            for r in by[(sc, n)]:
                ls = r["TRACE"]["x"]["lsplit"]
                err = abs(ls["comp"][k] - tsh["comp"][k]) + abs(ls["struct"][k] - tsh["struct"][k])
                ok = err / mag <= 0.25
                for role in PLANTED[sc]:
                    ok &= np.sign(ls[role][k]) == np.sign(tsh[role][k])
                ok &= abs(ls["comp"][0]) + abs(ls["struct"][0]) <= 0.05 * r["sdA"]
                hits.append(int(ok))
                errs.append(err)
            bar = 0.80 if n == 20000 else 0.60
            kk, m = sum(hits), len(hits)
            out["recovery"][f"{sc}|n={n}"] = {
                "k": kk,
                "m": m,
                "planned": reps,
                "rate": kk / m if m else None,
                "cp95": cp(kk, m),
                "bar": bar,
                "verdict": rate_verdict(kk, m, reps, bar),
                "pr12_rate": BASE["h7"]["recovery"][f"{sc}|n={n}"]["rate"],
            }
            out["errors"][f"{sc}|n={n}"] = {
                "TRACE_v3": float(np.mean(errs)) if errs else None,
                "TRACE_pr12": BASE["h7"]["errors"][f"{sc}|n={n}"]["TRACE"],
                "planted": mag,
            }
            if n == 20000 and errs:
                b = BEST[sc]
                rv = []
                for r in old:
                    if r["scen"] == sc and r["n"] == 20000:
                        c, s = r[b]["lsplit"][f"l{k + 1}"]
                        rv.append(abs(c - tar["comp"][k]) + abs(s - tar["struct"][k]))
                eT, eR = np.asarray(errs), np.asarray(rv)
                d = float(eT.mean() - eR.mean())
                tt = ttest_ind(eT, eR, equal_var=False)
                ci = tt.confidence_interval(0.95)
                rng = np.random.default_rng(2026)
                bm = eT[rng.integers(0, len(eT), (10000, len(eT)))].mean(1) - eR[
                    rng.integers(0, len(eR), (10000, len(eR)))
                ].mean(1)
                pc = np.quantile(bm, [0.025, 0.975]).tolist()
                below = bool(ci.high < 0 and pc[1] < 0)
                out["rival"][sc] = {
                    "best_rival": b,
                    "n_trace": len(eT),
                    "n_rival": len(eR),
                    "mean_trace": float(eT.mean()),
                    "mean_rival": float(eR.mean()),
                    "d": d,
                    "welch_ci": [float(ci.low), float(ci.high)],
                    "pct_ci": pc,
                    "p": float(tt.pvalue),
                    "beats": below,
                    "pr12_d": BASE["h7"]["rival_test"][sc]["paired"]["mean"],
                }
                tests.append((f"H7 {sc}", float(tt.pvalue), below))
    rv = [v["verdict"] for v in out["recovery"].values()]
    out["recovery_verdict"] = (
        "FAIL" if "FAIL" in rv else ("INCOMPLETE" if "INCOMPLETE" in rv else "PASS")
    )
    wins = sum(v["beats"] for v in out["rival"].values())
    out["rival_wins"] = wins
    out["rival_verdict"] = "PASS" if wins >= 4 else "FAIL"
    return out, tests, recs


def h5(recs):
    g, gr, desc, rel = [], [], [], []
    fails = []
    for r in recs:
        h = r["h5"]
        ok = h["g51b"] <= 1e-8 * max(1.0, h["W_rec"])
        g.append(h["g51b"])
        gr.append(h["g51b"] / max(1.0, h["W_rec"]))
        if not ok:
            fails.append([r["scen"], r["n"], r["rep"], h["g51b"]])
        desc.append(h["g52_desc"])
        rel.append(h["g52_desc"] / h["W2_emp"] if h["W2_emp"] > 0 else np.nan)
    return {
        "n": len(recs),
        "fails": len(fails),
        "fail_examples": fails[:10],
        "worst_abs": max(g) if g else None,
        "worst_scaled": max(gr) if gr else None,
        "verdict": ("PASS" if not fails else "FAIL") if recs else "INCOMPLETE",
        "marg_vs_emp_abs": {"median": float(np.median(desc)), "max": float(np.max(desc))}
        if desc
        else None,
        "marg_vs_emp_rel": {"median": float(np.nanmedian(rel)), "max": float(np.nanmax(rel))}
        if rel
        else None,
    }


def band(v):
    return np.quantile(np.asarray(v, float), [0.025, 0.975], axis=0)


def h8b():
    recs = jl(RES / "trace_v3_cover.jsonl")
    cov = defaultdict(list)
    reps = defaultdict(int)
    joint_null = []
    for r in recs:
        sc, bs = r["scen"], r["boots"]
        reps[sc] += 1
        if sc in shp.SHAPE_SCENARIOS:
            tt = TSHAPE[sc]["shapley"]
            for k in range(4):
                for role in ("comp", "struct"):
                    lo, hi = band([b["lsplit"][role][k] for b in bs])
                    cov[(sc, f"l{k + 1}")].append(float(lo <= tt[role][k] <= hi))
            continue
        if sc == "null":
            joint_null.append(
                float(
                    joint_reject(
                        r["point"]["split"]["mean"], np.array([b["split"]["mean"] for b in bs])
                    )[0]
                )
            )
            continue
        t = TRUTH[sc]
        for s in dgp.STATS:
            (lc, ls), (hc, hs) = band(np.array([b["split"][s] for b in bs]))
            cov[(sc, s)].append(float(lc <= t[s]["comp"] <= hc))
            cov[(sc, s)].append(float(ls <= t[s]["struct"] <= hs))
    base = BASE["h8"]["coverage"]["cells"]
    cells = {}
    for (a, b), v in cov.items():
        n_plan = 2 * COVER_REPS
        k, m = int(sum(v)), len(v)
        cells[f"{a}|{b}"] = {
            "cov": k / m,
            "k": k,
            "m": m,
            "cp95": cp(k, m),
            "verdict": rate_verdict(k, m, n_plan, 0.85),
            "pr12": base.get(f"{a}|{b}"),
        }
    allv = [x for v in cov.values() for x in v]
    graded = [s for s in COVER_SCEN if s != "null"] + list(shp.SHAPE_SCENARIOS)
    complete = all(reps[s] >= COVER_REPS for s in graded)
    pooled = float(np.mean(allv)) if allv else None
    if not allv:
        verdict = "INCOMPLETE"
    elif complete:
        verdict = (
            "PASS" if pooled >= 0.90 and min(c["cov"] for c in cells.values()) >= 0.85 else "FAIL"
        )
    else:
        cv = [c["verdict"] for c in cells.values()]
        verdict = "FAIL" if "FAIL" in cv else "INCOMPLETE"
    return {
        "reps": dict(reps),
        "complete": complete,
        "pooled": pooled,
        "pooled_cp95": cp(int(sum(allv)), len(allv)) if allv else None,
        "pr12_pooled": BASE["h8"]["coverage"]["pooled"],
        "min_cell": min((c["cov"] for c in cells.values()), default=None),
        "cells": cells,
        "verdict": verdict,
        "null_joint_fpr": float(np.mean(joint_null)) if joint_null else None,
        "null_n": len(joint_null),
    }


def holm(tests):
    tests = sorted(tests, key=lambda t: t[1])
    m, out, stop = len(tests), [], False
    for i, (name, p, ok) in enumerate(tests):
        rej = (not stop) and min(1.0, p * (m - i)) <= 0.05
        stop |= not rej
        out.append({"test": name, "p": p, "holm_reject": bool(rej), "trace_better": bool(ok)})
    return out


def f(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def render(S):
    L = [
        "# TRACE v3 results (H7, H8(b) with `pit_rounds=20`; H5.1b)",
        "",
        "Pre-registration: [PREREG_TRACE_V3.md](PREREG_TRACE_V3.md). Baseline: PR #12 ([TRACE_CLAIMS_RESULTS.md](TRACE_CLAIMS_RESULTS.md)).",
        "Generated by `benchmarks/shiftbench/summarize_trace_v3.py`; raw records `benchmarks/results/trace_v3_*.jsonl`.",
        "",
        "## Verdicts",
        "",
        "| claim | bar | v3 | PR #12 |",
        "|---|---|---|---|",
    ]
    h = S["h7"]
    L.append(
        f"| H7 recovery (10 cells) | every cell ≥ bar | **{h['recovery_verdict']}** ({sum(v['verdict'] == 'PASS' for v in h['recovery'].values())}/10 cells pass) | FAIL (1/10) |"
    )
    L.append(
        f"| H7 rival | beats best rival in ≥ 4/5 | **{h['rival_verdict']}** ({h['rival_wins']}/5) | FAIL (0/5) |"
    )
    c = S["h8b"]
    L.append(
        f"| H8(b) coverage | pooled ≥ 0.90 and every cell ≥ 0.85 | **{c['verdict']}** (pooled {f(c['pooled'])}, min cell {f(c['min_cell'])}) | FAIL (pooled 0.807, min 0.0875) |"
    )
    L.append(
        f"| H5.1b barycenter reconstruction | gap ≤ 1e-8·max(1,W) every rep | **{S['h5']['verdict']}** ({S['h5']['fails']}/{S['h5']['n']} misses, worst {S['h5']['worst_scaled']:.1e}) | PASS (worst 3.6e-15) |"
    )
    L += [
        "",
        "## H7 recovery",
        "",
        "| scenario | n | k/m (planned) | rate | CP 95% | bar | verdict | PR #12 rate | mean err v3 | mean err PR #12 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, v in h["recovery"].items():
        sc, n = key.split("|")
        e = h["errors"][key]
        L.append(
            f"| {sc} | {n[2:]} | {v['k']}/{v['m']} ({v['planned']}) | {f(v['rate'])} | [{f(v['cp95'][0])}, {f(v['cp95'][1])}] | {v['bar']} | {v['verdict']} | {f(v['pr12_rate'])} | {f(e['TRACE_v3'], 4)} | {f(e['TRACE_pr12'], 4)} |"
        )
    L += [
        "",
        "## H7 rival (unpaired; rival records reused from PR #12, n = 20)",
        "",
        "| scenario | best rival | TRACE v3 err (n) | rival err | d | Welch 95% | bootstrap 95% | p | beats | PR #12 paired d |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for sc, v in h["rival"].items():
        L.append(
            f"| {sc} | {v['best_rival']} | {v['mean_trace']:.4f} ({v['n_trace']}) | {v['mean_rival']:.4f} | {v['d']:+.4f} | [{v['welch_ci'][0]:+.4f}, {v['welch_ci'][1]:+.4f}] | [{v['pct_ci'][0]:+.4f}, {v['pct_ci'][1]:+.4f}] | {v['p']:.2g} | {'yes' if v['beats'] else 'no'} | {v['pr12_d']:+.4f} |"
        )
    L += [
        "",
        "Holm (5 H7 tests): "
        + "; ".join(
            f"{t['test']} p={t['p']:.2g} reject={t['holm_reject']} TRACE better={t['trace_better']}"
            for t in S["holm"]
        )
        + ".",
    ]
    L += [
        "",
        "## H8(b) coverage",
        "",
        f"Replications per scenario: {c['reps']}. Pooled {f(c['pooled'])} (CP {[round(x, 4) for x in c['pooled_cp95']] if c['pooled_cp95'] else '—'}; PR #12 {f(c['pr12_pooled'])}). Null joint-test rejection rate (not graded): {f(c['null_joint_fpr'])} over {c['null_n']}.",
        "",
        "| cell | coverage | k/m | CP 95% | ≥ 0.85 | PR #12 |",
        "|---|---|---|---|---|---|",
    ]
    for key, v in c["cells"].items():
        L.append(
            f"| {key} | {v['cov']:.3f} | {v['k']}/{v['m']} | [{v['cp95'][0]:.3f}, {v['cp95'][1]:.3f}] | {v['verdict']} | {f(v['pr12'])} |"
        )
    s5 = S["h5"]
    L += [
        "",
        "## H5",
        "",
        f"H5.1b: {s5['fails']} misses in {s5['n']} shape replications; worst |W_rec − Σφ| = {s5['worst_abs']:.2e} (scaled {s5['worst_scaled']:.2e}).",
        f"Marginal vs empirical W2² (descriptive, **not a claim**; see PREREG_TRACE_V3 §2): absolute median {s5['marg_vs_emp_abs']['median']:.2e}, max {s5['marg_vs_emp_abs']['max']:.2e}; relative median {s5['marg_vs_emp_rel']['median']:.2%}, max {s5['marg_vs_emp_rel']['max']:.2%}. The PR #12 H5 FAIL stays on record.",
        "",
    ]
    return "\n".join(L)


def main():
    S = {}
    S["h7"], tests, recs = h7()
    S["h5"] = h5(recs)
    S["h8b"] = h8b()
    S["holm"] = holm(tests)
    (RES / "trace_v3_summary.json").write_text(json.dumps(S, indent=1, default=float))
    DOC.write_text(render(S))
    print("wrote", DOC)


if __name__ == "__main__":
    main()
