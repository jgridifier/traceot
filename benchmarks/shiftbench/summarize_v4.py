# ruff: noqa: E501
"""Summarize PREREG_V4 jsonl into docs/benchmark/V4_RESULTS.md.

Reads benchmarks/results/v4_*.jsonl. Failures stay failures. Records prereg SHA.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "benchmarks" / "results"
OUT = ROOT / "docs" / "benchmark" / "V4_RESULTS.md"
PREREG_SHA = "d50732f"
TRUTH_SHAPE = RES / "shiftbench_truth_shape.json"
TRUTH_SB = RES / "shiftbench_truth.json"

SHAPE_SCEN = ("skew_struct", "tail_struct", "skew_comp", "tail_comp", "shape_mixed")
SB_SCEN = ("mixed", "tail", "hetero")
SB_STATS = ("mean", "sd", "q10", "q50", "q90")


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def paired_ci(diff: np.ndarray, alpha=0.05):
    """One-sided 95% upper bound via normal SE and percentile bootstrap (seed 2026)."""
    diff = np.asarray(diff, float)
    n = len(diff)
    if n < 2:
        return {
            "mean": float(diff.mean()) if n else float("nan"),
            "lo": float("nan"),
            "hi": float("nan"),
            "boot_hi": float("nan"),
            "n": n,
        }
    m = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(n))
    z = 1.64485  # one-sided 95%
    hi = m + z * se
    lo = m - z * se
    rng = np.random.default_rng(2026)
    boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(10_000)])
    b_hi = float(np.quantile(boots, 1 - alpha))
    return {"mean": m, "lo": float(lo), "hi": float(hi), "boot_hi": b_hi, "n": n}


def normalize_lsplit(lsplit):
    """TRACE {comp:[],struct:[]} or rival {l1:(c,s),...} → {comp,struct} lists."""
    if lsplit is None:
        return None
    if "comp" in lsplit and "struct" in lsplit:
        return {
            "comp": [float(x) for x in lsplit["comp"]],
            "struct": [float(x) for x in lsplit["struct"]],
        }
    # rival keyed by l1..l4
    keys = sorted(
        [k for k in lsplit if str(k).startswith("l") and str(k)[1:].isdigit()],
        key=lambda k: int(str(k)[1:]),
    )
    if not keys:
        return None
    comp, struct = [], []
    for k in keys:
        pair = lsplit[k]
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            comp.append(float(pair[0]))
            struct.append(float(pair[1]))
        else:
            return None
    return {"comp": comp, "struct": struct}


def graded_err(lsplit, truth_shapley, k_graded):
    """Abs error on graded L-moment channel vs Shapley truth (v3/Amendment-1 style)."""
    ls = normalize_lsplit(lsplit)
    if ls is None:
        return float("nan")
    k = int(k_graded)
    c = ls["comp"]
    s = ls["struct"]
    if k >= len(c) or k >= len(s):
        return float("nan")
    tc = float(truth_shapley["comp"][k])
    ts = float(truth_shapley["struct"][k])
    return abs(c[k] - tc) + abs(s[k] - ts)


def planted_mag(truth_shapley, k_graded):
    k = int(k_graded)
    return abs(float(truth_shapley["comp"][k])) + abs(float(truth_shapley["struct"][k]))


def mean_channel_err(split, truth_scen, sdA):
    """Mean |est - truth| over stats, scaled by sd(y_A)."""
    if not split:
        return float("nan")
    errs = []
    for st in SB_STATS:
        if st not in split or st not in truth_scen:
            continue
        pair = split[st]
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            ec, es = float(pair[0]), float(pair[1])
        else:
            continue
        tc = float(truth_scen[st]["comp"])
        ts = float(truth_scen[st]["struct"])
        errs.append(abs(ec - tc) + abs(es - ts))
    if not errs:
        return float("nan")
    return float(np.mean(errs)) / max(float(sdA), 1e-12)


def feat_err(feat_comp, feat_struct, truth_scen, sdA):
    if feat_comp is None:
        return float("nan")
    tc = np.asarray(truth_scen.get("feat_comp", []), float)
    ts = np.asarray(truth_scen.get("feat_struct", []), float)
    fc = np.asarray(feat_comp, float)
    fs = np.asarray(feat_struct if feat_struct is not None else np.zeros_like(fc), float)
    k = min(len(fc), len(tc), len(fs), len(ts))
    if k == 0:
        return float("nan")
    err = float(np.mean(np.abs(fc[:k] - tc[:k]) + np.abs(fs[:k] - ts[:k])))
    return err / max(float(sdA), 1e-12)


def cell_verdict(d_arr, delta):
    ci = paired_ci(d_arr)
    # both normal and bootstrap upper bounds must agree for a claim
    hi = max(ci["hi"], ci["boot_hi"])
    if not np.isfinite(hi):
        return "INCOMPLETE", ci, hi
    if hi < 0:
        return "BEST", ci, hi
    if hi < delta:
        return "NON-INFERIOR", ci, hi
    return "FAIL", ci, hi


def main():
    shape = load_jsonl(RES / "v4_shape.jsonl")
    cover = load_jsonl(RES / "v4_cover.jsonl")
    sb = load_jsonl(RES / "v4_shiftbench.jsonl")
    speed = load_jsonl(RES / "v4_speed.jsonl")
    truth_shape = json.loads(TRUTH_SHAPE.read_text()) if TRUTH_SHAPE.exists() else {}
    truth_sb = json.loads(TRUTH_SB.read_text()) if TRUTH_SB.exists() else {}

    lines_body = [
        "## H7 shape (graded-channel error vs best rival)",
        "",
        "| cell | n | reps | mean err v4 | best rival | mean d | upper 95% CI | δ | verdict |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|",
    ]

    cells = defaultdict(list)
    for r in shape:
        cells[(r["scen"], r["n"])].append(r)

    accuracy_verdicts = []

    for scen in SHAPE_SCEN:
        t_scen = truth_shape.get(scen, {})
        shapley = t_scen.get("shapley")
        graded = t_scen.get("graded", [0])
        k = graded[0] if graded else 0
        mag = planted_mag(shapley, k) if shapley else float("nan")
        delta = 0.10 * mag if np.isfinite(mag) else float("nan")
        for n in (2000, 20000):
            rows = cells.get((scen, n), [])
            if not rows or shapley is None:
                lines_body.append(
                    f"| {scen} | {n} | {len(rows)} | — | — | — | — | — | INCOMPLETE |"
                )
                accuracy_verdicts.append("INCOMPLETE")
                continue
            ev4, e_riv, names = [], [], []
            for r in rows:
                e0 = graded_err(r["TRACE_V4"].get("lsplit"), shapley, k)
                ev4.append(e0)
                best_e, best_n = float("inf"), None
                for nm, rv in r.get("rivals", {}).items():
                    if "error" in rv:
                        continue
                    e = graded_err(rv.get("lsplit"), shapley, k)
                    if np.isfinite(e) and e < best_e:
                        best_e, best_n = e, nm
                if best_n is None:
                    e_riv.append(float("nan"))
                    names.append("n/a")
                else:
                    e_riv.append(best_e)
                    names.append(best_n)
            ev4 = np.asarray(ev4, float)
            e_riv = np.asarray(e_riv, float)
            mask = np.isfinite(ev4) & np.isfinite(e_riv)
            if mask.sum() < 2 or not np.isfinite(delta):
                lines_body.append(
                    f"| {scen} | {n} | {len(rows)} | {np.nanmean(ev4):.4g} | "
                    f"{names[0] if names else 'n/a'} | — | — | {delta:.4g} | INCOMPLETE |"
                )
                accuracy_verdicts.append("INCOMPLETE")
                continue
            d = ev4[mask] - e_riv[mask]
            verd, ci, hi = cell_verdict(d, delta)
            accuracy_verdicts.append(verd)
            rival = max(set(names), key=names.count) if names else "n/a"
            lines_body.append(
                f"| {scen} | {n} | {int(mask.sum())} | {float(np.mean(ev4[mask])):.4g} | "
                f"{rival} | {ci['mean']:.4g} | {hi:.4g} | {delta:.4g} | **{verd}** |"
            )

    # ShiftBench accuracy
    lines_body += [
        "",
        "## ShiftBench accuracy (mean-channel + per-feature vs best rival)",
        "",
        "| cell | metric | reps | mean err v4 | best rival | mean d | upper 95% CI | δ | verdict |",
        "|---|---|---:|---:|---|---:|---:|---:|---|",
    ]
    sb_by = defaultdict(list)
    for r in sb:
        sb_by[r["scen"]].append(r)

    for scen in SB_SCEN:
        t_scen = truth_sb.get(scen, {})
        rows = sb_by.get(scen, [])
        for metric, kind in (("mean_channel", "mean"), ("per_feature", "feat")):
            if not rows or not t_scen:
                lines_body.append(
                    f"| {scen} | {metric} | {len(rows)} | — | — | — | — | — | INCOMPLETE |"
                )
                accuracy_verdicts.append("INCOMPLETE")
                continue
            ev4, e_riv, names = [], [], []
            deltas = []
            for r in rows:
                sdA = float(r.get("sdA", t_scen.get("sd_A", 1.0)))
                delta = 0.002  # already in sd units after scaling
                deltas.append(delta)
                e0 = _sb_err(r, "TRACE_V4", t_scen, kind)
                ev4.append(e0)
                best_e, best_n = float("inf"), None
                for nm, rv in r.get("rivals", {}).items():
                    if "error" in rv:
                        continue
                    e = _sb_err({"TRACE_V4": rv, "sdA": sdA}, "TRACE_V4", t_scen, kind)
                    if np.isfinite(e) and e < best_e:
                        best_e, best_n = e, nm
                if best_n is None:
                    e_riv.append(float("nan"))
                    names.append("n/a")
                else:
                    e_riv.append(best_e)
                    names.append(best_n)
            ev4 = np.asarray(ev4, float)
            e_riv = np.asarray(e_riv, float)
            mask = np.isfinite(ev4) & np.isfinite(e_riv)
            delta = float(np.mean(deltas)) if deltas else 0.002
            if mask.sum() < 2:
                lines_body.append(
                    f"| {scen} | {metric} | {len(rows)} | {np.nanmean(ev4):.4g} | "
                    f"{names[0] if names else 'n/a'} | — | — | {delta:.4g} | INCOMPLETE |"
                )
                accuracy_verdicts.append("INCOMPLETE")
                continue
            d = ev4[mask] - e_riv[mask]
            verd, ci, hi = cell_verdict(d, delta)
            accuracy_verdicts.append(verd)
            rival = max(set(names), key=names.count) if names else "n/a"
            lines_body.append(
                f"| {scen} | {metric} | {int(mask.sum())} | {float(np.mean(ev4[mask])):.4g} | "
                f"{rival} | {ci['mean']:.4g} | {hi:.4g} | {delta:.4g} | **{verd}** |"
            )

    # Coverage
    lines_body += ["", "## H8b coverage", ""]
    if not cover:
        lines_body.append("INCOMPLETE — no coverage records.")
        cov_verdict = "INCOMPLETE"
    else:
        # v4 cover jsonl stores point + budget_ci only (no bootstrap lsplit bands).
        # Oracle coverage vs planted Shapley / ShiftBench truth therefore cannot be scored.
        lines_body.append(
            f"Records: {len(cover)}. Cover track wrote point estimates and a budget CI, "
            "but not per-cell split-product / multiplier bands for lsplit or mean-channel pieces."
        )
        lines_body.append(
            "Oracle coverage vs planted truth (PREREG_V4 §5: pooled ≥ 0.90 and every cell ≥ 0.85) "
            "requires those bands. Marked **INCOMPLETE** — not imputed, not passed."
        )
        by_scen = defaultdict(int)
        for r in cover:
            by_scen[r["scen"]] += 1
        lines_body.append(
            "Reps by scenario: " + ", ".join(f"{k}={v}" for k, v in sorted(by_scen.items())) + "."
        )
        cov_verdict = "INCOMPLETE"

    # Speed
    lines_body += ["", "## Speed vs CFM", ""]
    speed_pass = None
    ratios = {}
    if not speed:
        lines_body.append("INCOMPLETE — no speed records.")
        speed_note = "INCOMPLETE"
    else:
        lines_body.append("| n | reps | mean sec v4 | mean sec CFM | ratio CFM/v4 |")
        lines_body.append("|---:|---:|---:|---:|---:|")
        by_n = defaultdict(list)
        for r in speed:
            by_n[r["n"]].append(r)
        for n, rows in sorted(by_n.items()):
            sv = float(np.mean([r["sec_v4"] for r in rows]))
            cfm_ok = [r["sec_cfm"] for r in rows if r.get("cfm_error") is None]
            sc = float(np.mean(cfm_ok)) if cfm_ok else float("nan")
            ratio = sc / sv if sv > 0 and np.isfinite(sc) else float("nan")
            ratios[n] = ratio
            lines_body.append(f"| {n} | {len(rows)} | {sv:.3g} | {sc:.3g} | {ratio:.3g} |")
        # Conditional gate
        # PREREG: if ANY cell is only NI (and not all BEST), require ≥5× at both sizes
        all_acc_best = accuracy_verdicts and all(v == "BEST" for v in accuracy_verdicts)
        if all_acc_best:
            speed_pass = "N/A (all accuracy BEST; speed reported, not a gate)"
        elif "INCOMPLETE" in accuracy_verdicts or "FAIL" in accuracy_verdicts:
            # still report whether ratio would pass
            r1, r2 = ratios.get(100_000), ratios.get(1_000_000)
            ok = np.isfinite(r1) and np.isfinite(r2) and r1 >= 5 and r2 >= 5
            speed_pass = f"{'PASS' if ok else 'FAIL'} conditional (≥5× both sizes); ratios={ratios}"
            if "FAIL" in accuracy_verdicts:
                speed_pass += " (accuracy already FAIL — speed does not rescue)"
        else:
            # some NI, no FAIL/INCOMPLETE
            r1, r2 = ratios.get(100_000), ratios.get(1_000_000)
            ok = np.isfinite(r1) and np.isfinite(r2) and r1 >= 5 and r2 >= 5
            speed_pass = "PASS" if ok else "FAIL"
        speed_note = speed_pass

    # Overall accuracy (PREREG early-stop: FAIL if completed cells already decide against PASS)
    if not accuracy_verdicts:
        overall = "INCOMPLETE"
    elif "FAIL" in accuracy_verdicts:
        overall = "FAIL"
    elif "INCOMPLETE" in accuracy_verdicts:
        overall = "INCOMPLETE"
    elif all(v == "BEST" for v in accuracy_verdicts):
        overall = "PASS"
    else:
        # some NON-INFERIOR, rest BEST — speed gate
        r1, r2 = ratios.get(100_000), ratios.get(1_000_000)
        speed_ok = np.isfinite(r1) and np.isfinite(r2) and r1 >= 5 and r2 >= 5
        overall = "PASS" if speed_ok else "FAIL"

    lines = [
        "# TRACE v4 results",
        "",
        f"**Prereg SHA:** `{PREREG_SHA}` (frozen in `docs/benchmark/PREREG_V4.md`).",
        f"**Overall accuracy (prereg bar):** **{overall}**",
        f"**Coverage bar:** **{cov_verdict}**",
        f"**Speed (conditional):** {speed_note}",
        f"**Scored run status:** shape={len(shape)} cover={len(cover)} shiftbench={len(sb)} speed={len(speed)} records.",
        "",
        "Failures stay failures. Incomplete cells are labelled INCOMPLETE, never imputed.",
        "",
        "## Deviations",
        "",
        "1. **Missing sklearn on first scored pass.** The initial `run_v4.py all --workers 8` "
        "completed shape/cover/shiftbench/speed, but every CFM/DFL call raised "
        "`No module named 'sklearn'`. TRACE_V4 rows and seeds are unchanged. After installing "
        "`scikit-learn` in the box venv, CFM/DFL were **backfilled on the same seeds** into "
        "`v4_shape.jsonl` and `v4_shiftbench.jsonl`, and `v4_speed.jsonl` was re-run once. "
        "No TRACE_V4 recomputation; no seed change.",
        "2. **ShiftBench per-feature vs rivals.** CFM/DFL adapters in this repo do not "
        "return `feat_comp`/`feat_struct`, so the three per-feature paired cells stay "
        "INCOMPLETE (TRACE_V4 alone has feature pieces).",
        "3. **Coverage bands.** `explain_v4` / cover track persist a budget CI but not "
        "per-cell lsplit / mean-channel multiplier bands, so the oracle coverage bar cannot "
        "be evaluated and stays INCOMPLETE.",
        "",
    ] + lines_body

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} overall={overall} cov={cov_verdict} speed={speed_note}")
    print("verdicts:", accuracy_verdicts)


def _sb_err(r, key, t_scen, kind):
    block = r.get(key) or {}
    sdA = float(r.get("sdA", t_scen.get("sd_A", 1.0)))
    if kind == "mean":
        return mean_channel_err(block.get("split"), t_scen, sdA)
    return feat_err(block.get("feat_comp"), block.get("feat_struct"), t_scen, sdA)


if __name__ == "__main__":
    main()
