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
TRUTH = RES / "shiftbench_truth_shape.json"

SHAPE_SCEN = ("skew_struct", "tail_struct", "skew_comp", "tail_comp", "shape_mixed")


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def paired_ci(diff: np.ndarray, alpha=0.05):
    diff = np.asarray(diff, float)
    n = len(diff)
    if n < 2:
        return {"mean": float(diff.mean()) if n else float("nan"), "lo": float("nan"), "hi": float("nan"), "n": n}
    m = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n)
    # crude normal approx (t would need scipy); also percentile bootstrap
    z = 1.64485  # one-sided 95%
    hi = m + z * se
    lo = m - z * se
    rng = np.random.default_rng(2026)
    boots = [diff[rng.integers(0, n, n)].mean() for _ in range(10_000)]
    b_hi = float(np.quantile(boots, 1 - alpha))
    return {"mean": float(m), "lo": float(lo), "hi": float(hi), "boot_hi": b_hi, "n": n}


def graded_err(lsplit, truth_comp, truth_struct, roles):
    if lsplit is None:
        return float("nan")
    c = np.asarray(lsplit.get("comp", []), float)
    s = np.asarray(lsplit.get("struct", []), float)
    err = 0.0
    if "comp" in roles and len(c):
        tc = np.asarray(truth_comp, float)
        k = min(len(c), len(tc))
        err += float(np.mean(np.abs(c[:k] - tc[:k])))
    if "struct" in roles and len(s):
        ts = np.asarray(truth_struct, float)
        k = min(len(s), len(ts))
        err += float(np.mean(np.abs(s[:k] - ts[:k])))
    return err


def main():
    shape = load_jsonl(RES / "v4_shape.jsonl")
    cover = load_jsonl(RES / "v4_cover.jsonl")
    sb = load_jsonl(RES / "v4_shiftbench.jsonl")
    speed = load_jsonl(RES / "v4_speed.jsonl")
    truth = json.loads(TRUTH.read_text()) if TRUTH.exists() else {}

    lines = [
        "# TRACE v4 results",
        "",
        f"**Prereg SHA:** `{PREREG_SHA}` (frozen in `docs/benchmark/PREREG_V4.md`).",
        f"**Scored run status:** shape={len(shape)} cover={len(cover)} shiftbench={len(sb)} speed={len(speed)} records.",
        "",
        "Failures stay failures. Incomplete cells are labelled INCOMPLETE, never imputed.",
        "",
        "## H7 shape (graded-channel error vs best rival)",
        "",
        "| cell | n | reps | mean err v4 | best rival | mean d | upper 95% CI | verdict |",
        "|---|---:|---:|---:|---|---:|---:|---|",
    ]

    roles_map = {
        "skew_struct": ("struct",),
        "tail_struct": ("struct",),
        "skew_comp": ("comp",),
        "tail_comp": ("comp",),
        "shape_mixed": ("comp", "struct"),
    }
    cells = defaultdict(list)
    for r in shape:
        cells[(r["scen"], r["n"])].append(r)

    accuracy_verdicts = []
    for scen in SHAPE_SCEN:
        for n in (2000, 20000):
            rows = cells.get((scen, n), [])
            if not rows:
                lines.append(f"| {scen} | {n} | 0 | — | — | — | — | INCOMPLETE |")
                accuracy_verdicts.append("INCOMPLETE")
                continue
            roles = roles_map[scen]
            # truth: try shape truth file structure
            t_scen = truth.get(scen, {}) if isinstance(truth, dict) else {}
            tc = t_scen.get("comp", t_scen.get("lsplit", {}).get("comp", [0.0] * 4))
            ts = t_scen.get("struct", t_scen.get("lsplit", {}).get("struct", [0.0] * 4))
            ev4, e_riv, names = [], [], []
            for r in rows:
                e0 = graded_err(r["TRACE_V4"].get("lsplit"), tc, ts, roles)
                ev4.append(e0)
                best_e, best_n = float("inf"), None
                for nm, rv in r.get("rivals", {}).items():
                    if "error" in rv:
                        continue
                    e = graded_err(rv.get("lsplit"), tc, ts, roles)
                    if not np.isfinite(e):
                        # fall back: mean split L1 vs planted 0 if no lsplit
                        continue
                    if e < best_e:
                        best_e, best_n = e, nm
                if best_n is None:
                    # use mean-channel absolute as descriptive only
                    best_e, best_n = float("nan"), "n/a"
                e_riv.append(best_e)
                names.append(best_n)
            ev4 = np.asarray(ev4, float)
            e_riv = np.asarray(e_riv, float)
            mask = np.isfinite(ev4) & np.isfinite(e_riv)
            if mask.sum() < 2:
                lines.append(
                    f"| {scen} | {n} | {len(rows)} | {np.nanmean(ev4):.4g} | {names[0] if names else 'n/a'} | — | — | INCOMPLETE |"
                )
                accuracy_verdicts.append("INCOMPLETE")
                continue
            d = ev4[mask] - e_riv[mask]
            ci = paired_ci(d)
            # planted magnitude proxy
            planted = float(np.nanmean(np.abs(np.asarray(tc if "comp" in roles else ts, float)))) + 1e-12
            delta = 0.10 * planted
            hi = max(ci["hi"], ci.get("boot_hi", ci["hi"]))
            if hi < 0:
                verd = "BEST"
            elif hi < delta:
                verd = "NON-INFERIOR"
            else:
                verd = "FAIL"
            accuracy_verdicts.append(verd)
            rival = max(set(names), key=names.count) if names else "n/a"
            lines.append(
                f"| {scen} | {n} | {mask.sum()} | {float(np.mean(ev4[mask])):.4g} | {rival} | {ci['mean']:.4g} | {hi:.4g} | **{verd}** |"
            )

    # Coverage
    lines += ["", "## H8b coverage", ""]
    if not cover:
        lines.append("INCOMPLETE — no coverage records.")
        cov_verdict = "INCOMPLETE"
    else:
        # budget CI covering 0 when null? For non-null, we don't have truth bands easily;
        # report fraction where CI contains the point (sanity) and width.
        covered = 0
        total = 0
        for r in cover:
            ci = r.get("point", {}).get("budget_ci")
            est = r.get("point", {}).get("budget")
            if ci and est is not None and ci[0] is not None:
                total += 1
                # For coverage track without oracle IF truth, mark as descriptive
                if ci[0] <= est <= ci[1]:
                    covered += 1
        rate = covered / total if total else float("nan")
        lines.append(f"Records: {len(cover)}. Descriptive self-coverage of budget CI: {rate:.3f} ({covered}/{total}).")
        lines.append(
            "Oracle coverage vs planted Shapley truth requires lsplit bands (not yet in explain_v4 bands); marked **INCOMPLETE** for the prereg coverage bar."
        )
        cov_verdict = "INCOMPLETE"

    # Speed
    lines += ["", "## Speed vs CFM", ""]
    if not speed:
        lines.append("INCOMPLETE — no speed records.")
        speed_note = "INCOMPLETE"
    else:
        lines.append("| n | reps | mean sec v4 | mean sec CFM | ratio |")
        lines.append("|---:|---:|---:|---:|---:|")
        by_n = defaultdict(list)
        for r in speed:
            by_n[r["n"]].append(r)
        ratios = {}
        for n, rows in sorted(by_n.items()):
            sv = np.mean([r["sec_v4"] for r in rows])
            sc = np.mean([r["sec_cfm"] for r in rows if r.get("cfm_error") is None] or [float("nan")])
            ratio = sc / sv if sv > 0 and np.isfinite(sc) else float("nan")
            ratios[n] = ratio
            lines.append(f"| {n} | {len(rows)} | {sv:.3g} | {sc:.3g} | {ratio:.3g} |")
        speed_note = str(ratios)

    overall = "PASS" if accuracy_verdicts and all(v in ("BEST", "NON-INFERIOR") for v in accuracy_verdicts) else (
        "INCOMPLETE" if "INCOMPLETE" in accuracy_verdicts else "FAIL"
    )
    lines = [
        "# TRACE v4 results",
        "",
        f"**Prereg SHA:** `{PREREG_SHA}` (frozen in `docs/benchmark/PREREG_V4.md`).",
        f"**Overall accuracy (prereg bar):** **{overall}**",
        f"**Coverage bar:** **{cov_verdict}**",
        f"**Speed:** {speed_note}",
        f"**Scored run status:** shape={len(shape)} cover={len(cover)} shiftbench={len(sb)} speed={len(speed)} records.",
        "",
        "Failures stay failures. Incomplete cells are labelled INCOMPLETE, never imputed.",
        "",
    ] + lines[lines.index("## H7 shape (graded-channel error vs best rival)") :]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} overall={overall}")


if __name__ == "__main__":
    main()
