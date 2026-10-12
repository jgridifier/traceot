"""Summarise the TRACE-claims runs into benchmarks/results/trace_claims_summary.json.

Implements the scoring of PREREG_TRACE_CLAIMS.md as amended (AMENDMENT_1, D4-D12).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, spearmanr, t as tdist

import sb_dgp as dgp
import sb_dgp_shape as shp
from run_synthetic import ACC_GRID, RES
from sb_h8 import cp_upper_one_sided, h8a_pass, joint_reject

RIV = ["OB", "DFL", "CFM", "RIF", "SHAP"]
TRUTH = json.loads((RES / "shiftbench_truth.json").read_text())
TRUTH_OF = json.loads((RES / "trace_claims_truth_orderfree.json").read_text())
TSHAPE = json.loads((RES / "shiftbench_truth_shape.json").read_text())
FROZEN_COVER = Path("/workspace/wt-shiftbench/benchmarks/results/shiftbench_coverage.jsonl")


def jl(p):
    p = Path(p)
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def paired(d, seed=2026):
    d = np.asarray(d, float)
    n = len(d)
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    q = tdist.ppf(0.975, n - 1)
    p = float(2 * tdist.sf(abs(m / se), n - 1)) if se > 0 else (0.0 if m != 0 else 1.0)
    rng = np.random.default_rng(seed)
    bm = d[rng.integers(0, n, (10000, n))].mean(1)
    return {"n": n, "mean": m, "t_ci": [m - q * se, m + q * se], "pct_ci": np.quantile(bm, [0.025, 0.975]).tolist(), "p": p}


def sp(a, b, cnt):
    r = spearmanr(a, b)[0]
    if not np.isfinite(r):
        cnt[0] += 1
        return 0.0
    return float(r)


# ------------------------------------------------------------ frozen tracks
def frozen_accuracy():
    riv = {(r["scen"], r["n"], r["p"], r["rep"]): r["methods"] for r in jl(RES / "shiftbench_accuracy.jsonl")}
    tr = {(r["scen"], r["n"], r["p"], r["rep"]): r for r in jl(RES / "trace_claims_acc.jsonl")}
    out = {}
    for n, p, reps in ACC_GRID:
        ch = defaultdict(lambda: defaultdict(list))  # stat -> method -> per-rep mean over scenarios
        fe = defaultdict(list)
        ch_of = defaultdict(list)
        for rep in range(reps):
            acc = defaultdict(lambda: defaultdict(list))
            facc = defaultdict(list)
            of = defaultdict(list)
            for sc in dgp.SCENARIOS:
                k = (sc, n, p, rep)
                if k not in tr or k not in riv:
                    continue
                t = TRUTH[sc]
                sd = t["sd_A"]
                ms = dict(riv[k])
                ms["TRACE"] = tr[k]["TRACE"]
                for m, o in ms.items():
                    for s, (c, st) in o["split"].items():
                        acc[s][m].append(0.5 * (abs(c - t[s]["comp"]) + abs(st - t[s]["struct"])) / sd)
                    if o.get("feat_comp") is not None:
                        tc, ts = np.zeros(p), np.zeros(p)
                        tc[:5], ts[:5] = t["feat_comp"][:5], t["feat_struct"][:5]
                        facc[m].append(float((np.abs(np.asarray(o["feat_comp"]) - tc) + np.abs(np.asarray(o["feat_struct"]) - ts)).sum() / sd))
                so = tr[k]["TRACE"]["x"]["split_of"]
                for s in dgp.STATS:
                    of[s].append(0.5 * (abs(so[s][0] - TRUTH_OF[sc][s]["comp"]) + abs(so[s][1] - TRUTH_OF[sc][s]["struct"])) / sd)
            for s in acc:
                for m, v in acc[s].items():
                    if len(v) == len(dgp.SCENARIOS):
                        ch[s][m].append(np.mean(v))
            for m, v in facc.items():
                if len(v) == len(dgp.SCENARIOS):
                    fe[m].append(np.mean(v))
            for s, v in of.items():
                if len(v) == len(dgp.SCENARIOS):
                    ch_of[s].append(np.mean(v))
        size = f"n={n},p={p}"
        o = {"channel": {}, "feature": {}, "trace_orderfree_channel": {s: float(np.mean(v)) for s, v in ch_of.items() if v}}
        for s in ch:
            o["channel"][s] = {m: float(np.mean(v)) for m, v in ch[s].items()}
            if ch[s].get("TRACE"):
                o["channel"][s]["paired_vs"] = {m: paired(np.subtract(ch[s]["TRACE"], ch[s][m])) for m in RIV
                                                if m in ch[s] and len(ch[s][m]) == len(ch[s]["TRACE"]) and len(ch[s][m]) > 1}
        o["feature"] = {m: float(np.mean(v)) for m, v in fe.items()}
        if fe.get("TRACE"):
            o["feature_paired_vs"] = {m: paired(np.subtract(fe["TRACE"], fe[m])) for m in ("OB", "RIF", "SHAP")
                                      if len(fe.get(m, [])) == len(fe["TRACE"]) and len(fe[m]) > 1}
        out[size] = o
    return out


def frozen_invariance():
    recs = jl(RES / "trace_claims_inv.jsonl")
    if not recs:
        return None
    return {"max_change_over_sdA": max(r["max_change"] / TRUTH[r["scen"]]["sd_A"] for r in recs),
            "max_change_abs": max(r["max_change"] for r in recs), "n": len(recs)}


def band(vals):
    return np.quantile(np.asarray(vals, float), [0.025, 0.975], axis=0)


def frozen_and_h8b_coverage():
    recs = jl(RES / "trace_claims_cover.jsonl")
    cov = defaultdict(list)
    union_null = []
    joint_null = []
    for r in recs:
        sc = r["scen"]
        bs = r["boots"]
        if sc in shp.SHAPE_SCENARIOS:
            tt = TSHAPE[sc]["shapley"]
            for k in range(4):
                for i, role in enumerate(("comp", "struct")):
                    lo, hi = band([b["lsplit"][role][k] for b in bs])
                    cov[(sc, f"l{k + 1}")].append(float(lo <= tt[role][k] <= hi))
            continue
        t = TRUTH[sc]
        if sc == "null":
            pt = r["point"]["split"]["mean"]
            bv = np.array([b["split"]["mean"] for b in bs])
            (lc, ls), (hc, hs) = band(bv)
            union_null.append(float(not (lc <= 0 <= hc) or not (ls <= 0 <= hs)))
            joint_null.append(float(joint_reject(pt, bv)[0]))
            continue
        for s in dgp.STATS:
            bv = np.array([b["split"][s] for b in bs])
            (lc, ls), (hc, hs) = band(bv)
            cov[(sc, s)].append(float(lc <= t[s]["comp"] <= hc))
            cov[(sc, s)].append(float(ls <= t[s]["struct"] <= hs))
    cells = {f"{a}|{b}": float(np.mean(v)) for (a, b), v in cov.items()}
    pooled = float(np.mean([x for v in cov.values() for x in v])) if cov else None
    return {"cells": cells, "pooled": pooled, "min_cell": min(cells.values()) if cells else None,
            "pass": bool(pooled is not None and pooled >= 0.90 and min(cells.values()) >= 0.85),
            "frozen_null_union_fpr": float(np.mean(union_null)) if union_null else None,
            "frozen_null_joint_fpr": float(np.mean(joint_null)) if joint_null else None}


def rival_frozen_coverage():
    """Frozen-rule numbers for rivals (existing ShiftBench coverage records), for comparability."""
    cov, fpr = defaultdict(list), defaultdict(list)
    for line in FROZEN_COVER.read_text().splitlines() if FROZEN_COVER.exists() else []:
        r = json.loads(line)
        t = TRUTH[r["scen"]]
        for m in RIV:
            bv = np.array([b[m]["split"]["mean"] for b in r["boots"]])
            (lc, ls), (hc, hs) = band(bv)
            if r["scen"] == "null":
                fpr[m].append(float(not (lc <= 0 <= hc) or not (ls <= 0 <= hs)))
            else:
                cov[m] += [float(lc <= t["mean"]["comp"] <= hc), float(ls <= t["mean"]["struct"] <= hs)]
    return {"mean_coverage": {m: float(np.mean(v)) for m, v in cov.items()},
            "null_union_fpr_mean": {m: float(np.mean(v)) for m, v in fpr.items()}}


# ------------------------------------------------------------ H5
def h5():
    recs = jl(RES / "trace_claims_acc.jsonl") + jl(RES / "trace_claims_shape.jsonl")
    fails = defaultdict(list)
    worst = defaultdict(float)
    for r in recs:
        h = r["h5"]
        tol = h["tol"]
        for g in ("g51a", "g51b", "g52", "g52_internal", "g53_chan", "g53_coef", "g54"):
            rel = h[g] / tol * 1e-9  # in units of the bar's relative scale
            worst[g] = max(worst[g], h[g] / (h["sdA2"] * max(1, h["W2_emp"] / h["sdA2"])))
            if h[g] > tol:
                fails[g].append([r["scen"], r["n"], r["p"], r["rep"], h[g]])
    inv = frozen_invariance()
    contrast = defaultdict(lambda: defaultdict(list))
    for r in jl(RES / "trace_claims_acc.jsonl"):
        sd = TRUTH[r["scen"]]["sd_A"]
        for m, o in r["contrast"].items():
            for s, v in o.items():
                contrast[m][s + "|add"].append(v["add_gap"] / sd)
                contrast[m][s + "|order"].append(v["order_gap"] / sd)
    con = {m: {k: {"median": float(np.median(v)), "max": float(np.max(v)), "share_gt_1e-9": float(np.mean(np.asarray(v) > 1e-9))}
               for k, v in d.items()} for m, d in contrast.items()}
    n_tot = len(recs)
    primary = ("g51b", "g52", "g53_chan", "g53_coef", "g54")
    ok = all(len(fails[g]) == 0 for g in primary) and inv is not None and inv["max_change_abs"] <= 1e-6
    return {"n_replications": n_tot, "fail_counts": {g: len(fails[g]) for g in worst}, "worst_relative": dict(worst),
            "fail_examples": {g: v[:10] for g, v in fails.items()}, "invariance": inv,
            "pass": ok, "rival_contrast": con}


# ------------------------------------------------------------ H6
def h6():
    out = {}
    # real
    riv = jl(RES / "shiftbench_real.jsonl")
    tr = jl(RES / "trace_claims_real.jsonl")
    if riv and tr:
        def vec(o):
            return np.abs(np.asarray(o["feat_comp"])) + np.abs(np.asarray(o["feat_struct"]))
        full = next(r for r in riv if r["kind"] == "full")["res"]
        tfull = next(r for r in tr if r["kind"] == "full")["TRACE"]
        rb = {r["k"]: r["res"] for r in riv if r["kind"] == "boot"}
        tb = {r["k"]: r["TRACE"] for r in tr if r["kind"] == "boot"}
        rs = {r["k"]: r for r in riv if r["kind"] == "split"}
        ts = {r["k"]: r for r in tr if r["kind"] == "split"}
        nan = defaultdict(lambda: [0])
        boot = defaultdict(list)
        half = defaultdict(list)
        for k in sorted(set(rb) & set(tb)):
            boot["TRACE"].append(sp(vec(tfull), vec(tb[k]), nan["TRACE"]))
            for m in ("OB", "RIF", "SHAP"):
                boot[m].append(sp(vec(full[m]), vec(rb[k][m]), nan[m]))
        for k in sorted(set(rs) & set(ts)):
            half["TRACE"].append(sp(vec(ts[k]["h1"]["TRACE"]), vec(ts[k]["h2"]["TRACE"]), nan["TRACE"]))
            for m in ("OB", "RIF", "SHAP"):
                half[m].append(sp(vec(rs[k]["h1"][m]), vec(rs[k]["h2"][m]), nan[m]))
        out["real"] = {}
        for name, d in (("bootstrap", boot), ("split_half", half)):
            best = max(("OB", "RIF", "SHAP"), key=lambda m: np.mean(d[m]))
            pt = paired(np.subtract(d["TRACE"], d[best]))
            out["real"][name] = {"means": {m: float(np.mean(v)) for m, v in d.items()}, "best_rival": best, "paired": pt}
        out["real"]["nan_counts"] = {m: v[0] for m, v in nan.items()}
    syn = jl(RES / "trace_claims_h6syn.jsonl")
    if syn:
        nan = defaultdict(lambda: [0])
        boot, half = defaultdict(list), defaultdict(list)
        for r in syn:
            for m in ("TRACE", "OB", "RIF", "SHAP", "TRACE_w2"):
                half[m].append(sp(r["halves"][0][m], r["halves"][1][m], nan[m]))
                boot[m].append(float(np.mean([sp(r["full"][m], b[m], nan[m]) for b in r["boots"]])))
        out["synthetic"] = {}
        for name, d in (("bootstrap", boot), ("split_half", half)):
            best = max(("OB", "RIF", "SHAP"), key=lambda m: np.mean(d[m]))
            out["synthetic"][name] = {"means": {m: float(np.mean(v)) for m, v in d.items()}, "best_rival": best,
                                      "paired": paired(np.subtract(d["TRACE"], d[best]))}
        out["synthetic"]["nan_counts"] = {m: v[0] for m, v in nan.items()}
    tests = []
    for dom in ("real", "synthetic"):
        for name in ("bootstrap", "split_half"):
            if dom in out and name in out[dom]:
                pt = out[dom][name]["paired"]
                ok = pt["mean"] >= 0.05 and pt["t_ci"][0] > 0 and pt["pct_ci"][0] > 0
                bad = pt["t_ci"][1] < 0
                out[dom][name]["verdict"] = "pass" if ok else ("FAIL (TRACE less stable)" if bad else ("partial" if pt["t_ci"][0] > 0 else "not supported"))
                tests.append((f"H6 {dom} {name}", pt["p"], ok and True))
    out["pass"] = bool(len(tests) == 4 and all(t[2] for t in tests))
    return out, tests


# ------------------------------------------------------------ H7
PLANTED = {"skew_struct": ("struct",), "tail_struct": ("struct",), "skew_comp": ("comp",), "tail_comp": ("comp",),
           "shape_mixed": ("comp", "struct")}


def h7():
    recs = jl(RES / "trace_claims_shape.jsonl")
    out = {"recovery": {}, "rival_test": {}, "errors": {}}
    tests = []
    by = defaultdict(list)
    for r in recs:
        by[(r["scen"], r["n"], r["p"])].append(r)
    all_ok = True
    for sc in shp.SHAPE_SCENARIOS:
        T = TSHAPE[sc]
        k = T["graded"][0]
        tsh, tar = T["shapley"], T["aref"]
        mag = abs(tsh["comp"][k]) + abs(tsh["struct"][k])
        for n, p, reps in [(2000, 5, 50), (20000, 20, 20)]:
            rr = sorted(by[(sc, n, p)], key=lambda r: r["rep"])
            if not rr:
                continue
            rec, eT, eR = [], [], defaultdict(list)
            for r in rr:
                ls = r["TRACE"]["x"]["lsplit"]
                err = abs(ls["comp"][k] - tsh["comp"][k]) + abs(ls["struct"][k] - tsh["struct"][k])
                ok = err / mag <= 0.25
                for role in PLANTED[sc]:
                    ok &= np.sign(ls[role][k]) == np.sign(tsh[role][k])
                ok &= abs(ls["comp"][0]) + abs(ls["struct"][0]) <= 0.05 * r["sdA"]
                rec.append(float(ok))
                eT.append(err)
                for m in ("DFL", "CFM"):
                    c, s = r[m]["lsplit"][f"l{k + 1}"]
                    eR[m].append(abs(c - tar["comp"][k]) + abs(s - tar["struct"][k]))
            bar = 0.80 if n == 20000 else 0.60
            rate = float(np.mean(rec))
            out["recovery"][f"{sc}|n={n}"] = {"rate": rate, "bar": bar, "pass": rate >= bar, "reps": len(rec)}
            all_ok &= rate >= bar
            out["errors"][f"{sc}|n={n}"] = {"TRACE": float(np.mean(eT)), **{m: float(np.mean(v)) for m, v in eR.items()},
                                            "planted_magnitude": mag}
            if n == 20000:
                best = min(("DFL", "CFM"), key=lambda m: np.mean(eR[m]))
                pt = paired(np.subtract(eT, eR[best]))
                out["rival_test"][sc] = {"best_rival": best, "paired": pt, "below_zero": pt["t_ci"][1] < 0 and pt["pct_ci"][1] < 0}
                tests.append((f"H7 {sc}", pt["p"], out["rival_test"][sc]["below_zero"]))
    wins = sum(v["below_zero"] for v in out["rival_test"].values())
    out["rival_wins"] = wins
    out["recovery_pass"] = bool(all_ok)
    out["rival_supported"] = wins >= 4
    return out, tests


# ------------------------------------------------------------ H8
def h8():
    out = {}
    recs = jl(RES / "trace_claims_null200.jsonl")
    tests = []
    if recs:
        rej = defaultdict(list)
        for r in recs:
            P, B = r["point"], r["boots"]
            rej["TRACE_mean_aref"].append(joint_reject(P["TRACE"]["split"]["mean"], [b["TRACE"]["split"]["mean"] for b in B])[0])
            rej["TRACE_mean_orderfree"].append(joint_reject(P["TRACE"]["split_of"]["mean"], [b["TRACE"]["split_of"]["mean"] for b in B])[0])
            v = lambda o: np.r_[o["lsplit"]["comp"], o["lsplit"]["struct"]]
            rej["TRACE_channels"].append(joint_reject(v(P["TRACE"]), [v(b["TRACE"]) for b in B])[0])
            f = lambda o: np.r_[o["feat_comp"], o["feat_struct"]]
            rej["TRACE_features"].append(joint_reject(f(P["TRACE"]), [f(b["TRACE"]) for b in B])[0])
            for m in RIV:
                rej[f"{m}_mean"].append(joint_reject(P[m]["mean"], [b[m]["mean"] for b in B])[0])
                if P[m]["feat"] is not None:
                    rej[f"{m}_features"].append(joint_reject(P[m]["feat"], [b[m]["feat"] for b in B])[0])
        n = len(recs)
        out["null200"] = {k: {"x": int(sum(v)), "n": len(v), "fpr": float(np.mean(v)), "cp_upper_1s": cp_upper_one_sided(int(sum(v)), len(v)),
                              "pass_bar": h8a_pass(int(sum(v)), len(v))} for k, v in rej.items()}
        x = int(sum(rej["TRACE_mean_aref"]))
        p = binomtest(x, n, 0.09, alternative="less").pvalue
        out["h8a_primary"] = {"x": x, "n": n, "pass": h8a_pass(x, n), "p_H0_fpr_ge_0.09": float(p)}
        tests.append(("H8a TRACE mean-channel FPR", float(p), h8a_pass(x, n)))
        mc = {}
        a = np.asarray(rej["TRACE_mean_aref"], bool)
        for m in RIV:
            b = np.asarray(rej[f"{m}_mean"], bool)
            n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
            mc[m] = {"trace_only": n10, "rival_only": n01,
                     "p_exact": float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0}
        out["mcnemar_vs_rivals"] = mc
    pl = jl(RES / "trace_claims_placebo.jsonl")
    if pl:
        rej = defaultdict(list)
        for r in pl:
            P, B = r["point"], r["boots"]
            rej["TRACE"].append(joint_reject(P["TRACE"]["split"]["mean"], [b["TRACE"]["split"]["mean"] for b in B])[0])
            for m in RIV:
                rej[m].append(joint_reject(P[m]["mean"], [b[m]["mean"] for b in B])[0])
        out["placebo"] = {m: float(np.mean(v)) for m, v in rej.items()}
        out["placebo_pass"] = out["placebo"]["TRACE"] <= 0.10
    out["coverage"] = frozen_and_h8b_coverage()
    return out, tests


def holm(tests):
    tests = sorted(tests, key=lambda t: t[1])
    m = len(tests)
    out, stop = [], False
    for i, (name, p, ok) in enumerate(tests):
        adj = min(1.0, p * (m - i))
        reject = (not stop) and adj <= 0.05
        stop |= not reject
        out.append({"test": name, "p": p, "holm_threshold": 0.05 / (m - i), "holm_reject": bool(reject), "bar_ok": bool(ok)})
    return out


def main():
    S = {"frozen_accuracy": frozen_accuracy(), "frozen_rival_coverage": rival_frozen_coverage(), "h5": h5()}
    S["h6"], t6 = h6()
    S["h7"], t7 = h7()
    S["h8"], t8 = h8()
    S["holm"] = holm(t6 + t7 + t8)
    sc = RES / "trace_claims_scale.json"
    S["scale"] = json.loads(sc.read_text()) if sc.exists() else None
    (RES / "trace_claims_summary.json").write_text(json.dumps(S, indent=1, default=float))
    print("wrote", RES / "trace_claims_summary.json")


if __name__ == "__main__":
    main()
