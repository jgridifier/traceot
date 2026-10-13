"""Turn raw ShiftBench records into the pre-registered metrics.

python summarize.py   ->  benchmarks/results/shiftbench_summary.json
                          docs/benchmark/SHIFTBENCH_RESULTS.md
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from run_synthetic import ACC_GRID, COVER_SCEN, RES, TRUTH
from sb_dgp import N_ACTIVE, SCENARIOS, STATS
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
METHODS = ["OB", "DFL", "CFM", "RIF", "SHAP"]


def jl(name):
    p = RES / name
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def mean_ci(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return None
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
    return [float(v.mean()), float(1.96 * se)]


def feat_truth(t, p):
    c, s = np.zeros(p), np.zeros(p)
    c[:N_ACTIVE], s[:N_ACTIVE] = t["feat_comp"][:N_ACTIVE], t["feat_struct"][:N_ACTIVE]
    return c, s


def accuracy(truth):
    recs = jl("shiftbench_accuracy.jsonl")
    ch = defaultdict(list)  # (n,p,method,stat,scen) -> scaled abs errors (comp and struct)
    fe = defaultdict(list)
    inert = defaultdict(list)
    sec = defaultdict(list)
    for r in recs:
        t = truth[r["scen"]]
        sd = t["sd_A"]
        for m, o in r["methods"].items():
            key = (r["n"], r["p"], m)
            sec[key].append(o["sec"])
            for s, (c, st) in o["split"].items():
                err = 0.5 * (abs(c - t[s]["comp"]) + abs(st - t[s]["struct"])) / sd
                ch[key + (s, r["scen"])].append(err)
            if o["feat_comp"] is not None:
                tc, ts = feat_truth(t, r["p"])
                ec = np.abs(np.asarray(o["feat_comp"]) - tc)
                es = np.abs(np.asarray(o["feat_struct"]) - ts)
                fe[key + (r["scen"],)].append(float((ec + es).sum() / sd))
                inert[key].append(float((ec[N_ACTIVE:] + es[N_ACTIVE:]).sum() / sd))
    out = {
        "channel": {},
        "channel_by_scen": {},
        "feature": {},
        "feature_by_scen": {},
        "inert": {},
        "sec": {},
    }
    for n, p, _ in ACC_GRID:
        for m in METHODS:
            k = f"n={n},p={p}|{m}"
            out["sec"][k] = float(np.median(sec[(n, p, m)])) if sec[(n, p, m)] else None
            for s in STATS:
                per = [np.mean(ch[(n, p, m, s, sc)]) for sc in SCENARIOS if ch[(n, p, m, s, sc)]]
                out["channel"][f"{k}|{s}"] = float(np.mean(per)) if per else None
                for sc in SCENARIOS:
                    v = ch[(n, p, m, s, sc)]
                    out["channel_by_scen"][f"{k}|{s}|{sc}"] = mean_ci(v) if v else None
            per = [np.mean(fe[(n, p, m, sc)]) for sc in SCENARIOS if fe[(n, p, m, sc)]]
            out["feature"][k] = float(np.mean(per)) if per else None
            for sc in SCENARIOS:
                v = fe[(n, p, m, sc)]
                out["feature_by_scen"][f"{k}|{sc}"] = mean_ci(v) if v else None
            out["inert"][k] = float(np.mean(inert[(n, p, m)])) if inert[(n, p, m)] else None
    return out


def coverage(truth):
    recs = jl("shiftbench_coverage.jsonl")
    cov = defaultdict(list)
    fpr = defaultdict(list)
    ffpr = defaultdict(list)
    width = defaultdict(list)
    for r in recs:
        t = truth[r["scen"]]
        for m in METHODS:
            if m not in r["point"]:
                continue
            for s in r["point"][m]["split"]:
                bc = np.array([b[m]["split"][s][0] for b in r["boots"]])
                bs = np.array([b[m]["split"][s][1] for b in r["boots"]])
                lc, hc = np.quantile(bc, [0.025, 0.975])
                ls, hs = np.quantile(bs, [0.025, 0.975])
                if r["scen"] == "null":
                    fpr[(m, s)].append(float(not (lc <= 0 <= hc) or not (ls <= 0 <= hs)))
                else:
                    cov[(m, s)].append(float(lc <= t[s]["comp"] <= hc))
                    cov[(m, s)].append(float(ls <= t[s]["struct"] <= hs))
                    width[(m, s)].append(0.5 * ((hc - lc) + (hs - ls)) / t["sd_A"])
            if r["scen"] == "null" and r["point"][m]["feat_comp"] is not None:
                fc = np.array([b[m]["feat_comp"] for b in r["boots"]])
                fs = np.array([b[m]["feat_struct"] for b in r["boots"]])
                for F in (fc, fs):
                    lo, hi = np.quantile(F, [0.025, 0.975], axis=0)
                    ffpr[m].extend(((lo > 0) | (hi < 0)).astype(float).tolist())
    return {
        "coverage": {f"{m}|{s}": float(np.mean(v)) for (m, s), v in cov.items()},
        "width": {f"{m}|{s}": float(np.mean(v)) for (m, s), v in width.items()},
        "null_fpr": {f"{m}|{s}": float(np.mean(v)) for (m, s), v in fpr.items()},
        "null_feature_fpr": {m: float(np.mean(v)) for m, v in ffpr.items()},
        "scenarios": list(COVER_SCEN),
    }


def invariance(truth):
    recs = jl("shiftbench_invariance.jsonl")
    d = defaultdict(float)
    for r in recs:
        sd = truth[r["scen"]]["sd_A"]
        for m in METHODS:
            a, b = r["base"][m], r["perm"][m]
            diffs = [
                abs(x - y)
                for s in a["split"]
                for x, y in zip(a["split"][s], b["split"][s], strict=True)
            ]
            for k in ("feat_comp", "feat_struct"):
                if a[k] is not None:
                    diffs += np.abs(np.subtract(a[k], b[k])).tolist()
            d[m] = max(d[m], max(diffs) / sd)
    return dict(d)


def real():
    recs = jl("shiftbench_real.jsonl")
    if not recs:
        return {}
    meta = next(r for r in recs if r["kind"] == "meta")
    sd = meta["sd_A"]
    full = next(r for r in recs if r["kind"] == "full")["res"]
    boots = [r["res"] for r in recs if r["kind"] == "boot"]
    splits = [r for r in recs if r["kind"] == "split"]
    out = {"meta": meta, "full": {}, "boot_sd": {}, "rank_stability": {}, "split_half": {}}
    for m in METHODS:
        f = full[m]
        out["full"][m] = {
            "split": f["split"],
            "feat_comp": f["feat_comp"],
            "feat_struct": f["feat_struct"],
        }
        out["boot_sd"][m] = {
            s: [float(np.std([b[m]["split"][s][i] for b in boots]) / sd) for i in (0, 1)]
            for s in f["split"]
        }
        if f["feat_comp"] is not None:
            rs = []
            for b in boots:
                for k in ("feat_comp", "feat_struct"):
                    rs.append(spearmanr(np.abs(f[k]), np.abs(b[m][k]))[0])
            out["rank_stability"][m] = float(np.nanmean(rs))
        agree = []
        tot = []
        for r in splits:
            a, b = r["h1"][m], r["h2"][m]
            tot.append(
                np.mean([abs(a["split"]["mean"][i] - b["split"]["mean"][i]) for i in (0, 1)]) / sd
            )
            if a["feat_comp"] is not None:
                va = np.r_[a["feat_comp"], a["feat_struct"]]
                vb = np.r_[b["feat_comp"], b["feat_struct"]]
                agree.append(np.corrcoef(va, vb)[0, 1])
        out["split_half"][m] = {
            "mean_total_gap": float(np.mean(tot)),
            "feature_corr": float(np.mean(agree)) if agree else None,
        }
    return out


def fmt(v, d=3):
    return "n/a" if v is None else f"{v:.{d}f}"


def markdown(S):
    L = [
        "# ShiftBench results (rivals)",
        "",
        "Generated by `benchmarks/shiftbench/summarize.py` from the raw records in "
        "`benchmarks/results/shiftbench_*`. Design and metrics: "
        "[SHIFTBENCH.md](SHIFTBENCH.md), frozen in [PREREG_SHIFTBENCH.md](PREREG_SHIFTBENCH.md).",
        "All errors are in units of the period-A outcome standard deviation. Lower is better "
        "except coverage (target 0.95) and real-data agreement (higher is better).",
        "",
    ]
    A = S["accuracy"]
    for n, p, reps in ACC_GRID:
        L += [
            f"## Split error, n = {n:,} per period, p = {p} ({reps} reps x 11 scenarios)",
            "",
            "| method | " + " | ".join(STATS) + " | per-feature (mean) | inert mass | median sec |",
            "|---" * (len(STATS) + 4) + "|",
        ]
        for m in METHODS:
            k = f"n={n},p={p}|{m}"
            L.append(
                f"| {m} | "
                + " | ".join(fmt(A["channel"][f"{k}|{s}"]) for s in STATS)
                + f" | {fmt(A['feature'][k])} | {fmt(A['inert'][k])} | "
                f"{fmt(A['sec'][k], 2)} |"
            )
        L.append("")
    n, p, _ = ACC_GRID[1]
    L += [
        f"## Mean-channel split error by scenario, n = {n:,}, p = {p} (mean ± 95% CI)",
        "",
        "| scenario | " + " | ".join(METHODS) + " |",
        "|---" * (len(METHODS) + 1) + "|",
    ]
    for sc in SCENARIOS:
        row = []
        for m in METHODS:
            v = A["channel_by_scen"][f"n={n},p={p}|{m}|mean|{sc}"]
            row.append("n/a" if v is None else f"{v[0]:.3f} ± {v[1]:.3f}")
        L.append(f"| {sc} | " + " | ".join(row) + " |")
    L.append("")
    C = S["coverage"]
    L += [
        "## Bootstrap bands (n = 2,000, p = 5)",
        "",
        "Coverage of the planted composition and structure values by 95% percentile bands, "
        f"pooled over {', '.join(x for x in C['scenarios'] if x != 'null')}. "
        "Null false-positive rate: share of null replications where either band excludes 0.",
        "",
        "| method | stat | coverage | mean width | null FPR |",
        "|---|---|---|---|---|",
    ]
    for m in METHODS:
        for s in STATS:
            k = f"{m}|{s}"
            if k in C["coverage"]:
                L.append(
                    f"| {m} | {s} | {fmt(C['coverage'][k])} | {fmt(C['width'][k])} | "
                    f"{fmt(C['null_fpr'].get(k))} |"
                )
    L += [
        "",
        "Per-feature null false-positive rate (mean channel, any feature band excluding 0, "
        "per feature and side): "
        + ", ".join(f"{m} {v:.3f}" for m, v in C["null_feature_fpr"].items()),
        "",
    ]
    L += [
        "## Order invariance",
        "",
        "Largest change in any output when rows and columns are "
        "permuted (scaled): " + ", ".join(f"{m} {v:.2e}" for m, v in S["invariance"].items()),
        "",
    ]
    sc = S.get("scale", [])
    if sc:
        L += [
            "## Runtime and memory (scenario `mixed`, one run, 4 threads)",
            "",
            "| method | n | p | seconds | peak RSS MB | data MB |",
            "|---|---|---|---|---|---|",
        ]
        for r in sc:
            if "sec" in r:
                L.append(
                    f"| {r['method']} | {r['n']:,} | {r['p']} | {r['sec']:.1f} | "
                    f"{r['peak_rss_mb']:.0f} | {r['data_mb']:.0f} |"
                )
            else:
                L.append(f"| {r['method']} | {r['n']:,} | {r['p']} | {r['status']} | | |")
        L.append("")
    R = S.get("real", {})
    if R:
        M = R["meta"]
        L += [
            "## Real-data track",
            "",
            f"Public quarterly financial panel, {M['units']} units, {M['n_A']} rows in "
            f"period A and {M['n_B']} in period B, 8 features. No ground truth.",
            "",
            "| method | mean comp | mean struct | boot sd (comp, struct) | rank stability | "
            "split-half total gap | split-half feature corr |",
            "|---|---|---|---|---|---|---|",
        ]
        for m in METHODS:
            f = R["full"][m]["split"]["mean"]
            b = R["boot_sd"][m]["mean"]
            sh = R["split_half"][m]
            L.append(
                f"| {m} | {f[0]:.3f} | {f[1]:.3f} | {b[0]:.3f}, {b[1]:.3f} | "
                f"{fmt(R['rank_stability'].get(m))} | {sh['mean_total_gap']:.3f} | "
                f"{fmt(sh['feature_corr'])} |"
            )
        L.append("")
    return "\n".join(L)


def main():
    truth = json.loads(TRUTH.read_text())
    S = {
        "accuracy": accuracy(truth),
        "coverage": coverage(truth),
        "invariance": invariance(truth),
        "real": real(),
    }
    sp = RES / "shiftbench_scale.json"
    S["scale"] = json.loads(sp.read_text()) if sp.exists() else []
    (RES / "shiftbench_summary.json").write_text(json.dumps(S, indent=1))
    (ROOT / "docs" / "benchmark" / "SHIFTBENCH_RESULTS.md").write_text(markdown(S))


if __name__ == "__main__":
    main()
