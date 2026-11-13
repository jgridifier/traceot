"""Score the TRACE v5.1 coverage study (PREREG_V5_1_COVERAGE §5).

Reads benchmarks/results/v5_1_coverage.jsonl, writes v5_1_coverage_summary.json and
docs/benchmark/V5_1_COVERAGE_RESULTS.md. Pass interval per unit: 344 <= k <= 374 of
400 (nominal 90%, Bonferroni over four units). Failures stay failures. Missing
replications are never imputed; a unit is PASS/FAIL only if the completed
replications already decide it whatever the missing ones would be.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "benchmarks" / "results"
OUT_MD = ROOT / "docs" / "benchmark" / "V5_1_COVERAGE_RESULTS.md"
OUT_JSON = RES / "v5_1_coverage_summary.json"
PREREG_SHA = "dfaaf72e775ad6065b88a1c7931033c8dedbdbd3"
R = 400
PASS_LO, PASS_HI = 344, 374  # inclusive; 0.860 <= pi_hat <= 0.935
UNITS = (
    ("skew_comp", "comp"),
    ("skew_comp", "struct"),
    ("tail_comp", "comp"),
    ("tail_comp", "struct"),
)
PIECE_LABEL = {"comp": "comp", "struct": "response"}


def decide(k_cover: int, n_done: int, n_total: int = R) -> tuple[str, dict]:
    """Unit verdict given k covers among n_done completed reps (of n_total)."""
    missing = n_total - n_done
    k_min = k_cover  # all missing miss
    k_max = k_cover + missing  # all missing cover
    info = {
        "k_cover": k_cover,
        "n_done": n_done,
        "n_missing": missing,
        "k_min": k_min,
        "k_max": k_max,
    }
    if n_done == n_total:
        if PASS_LO <= k_cover <= PASS_HI:
            return "PASS", info
        if k_cover <= 343:
            return "FAIL (under-covers)", info
        return "FAIL (over-covers)", info
    # Incomplete: decide only if already determined.
    if k_max <= 343:
        return "FAIL (under-covers)", info
    if k_min >= 375:
        return "FAIL (over-covers)", info
    if PASS_LO <= k_min and k_max <= PASS_HI:
        return "PASS", info
    return "INCOMPLETE", info


def main():
    path = RES / "v5_1_coverage.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines() if x]
    wall = json.loads((RES / "v5_1_coverage_wall.json").read_text())
    units = []
    for scen, piece in UNITS:
        rs = sorted(
            [r for r in rows if r["scen"] == scen],
            key=lambda r: r["rep"],
        )
        covers = [r for r in rs if r["covered"][piece]]
        misses = [r for r in rs if not r["covered"][piece]]
        k = len(covers)
        verd, info = decide(k, len(rs))
        widths = [
            r["band"][piece][1] - r["band"][piece][0]
            for r in rs
            if all(x == x for x in r["band"][piece])  # not NaN
        ]
        # Miss split: truth below the band vs truth above (among misses that have a band
        # and were not forced-miss by refit failures).
        below = above = forced = no_band = 0
        for r in misses:
            if r.get("miss_due_to_refit_failures"):
                forced += 1
                continue
            lo, hi = r["band"][piece]
            if lo != lo or hi != hi:  # NaN
                no_band += 1
                continue
            t = r["truth"][piece]
            if t < lo:
                below += 1
            elif t > hi:
                above += 1
        unit = {
            "scen": scen,
            "piece": piece,
            "label": f"{scen} {PIECE_LABEL[piece]}",
            "verdict": verd,
            "k_cover": k,
            "n_done": len(rs),
            "pi_hat": (k / len(rs)) if rs else None,
            "pi_hat_of_400": k / R,
            "mean_width": (sum(widths) / len(widths)) if widths else None,
            "n_width": len(widths),
            "miss_truth_below_band": below,
            "miss_truth_above_band": above,
            "miss_forced_refit": forced,
            "miss_no_band": no_band,
            "n_failed_refits": sum(r["n_fail"] for r in rs),
            "n_reps_with_any_failed_refit": sum(1 for r in rs if r["n_fail"] > 0),
            **info,
        }
        units.append(unit)

    # Failed refits are per-replication (same for both pieces); count from rows.
    n_failed_refits = sum(r["n_fail"] for r in rows)
    k_pool = sum(u["k_cover"] for u in units)
    n_pool = sum(u["n_done"] for u in units)
    verdicts = [u["verdict"] for u in units]
    if any(v.startswith("FAIL") for v in verdicts):
        overall = "FAIL"
    elif any(v == "INCOMPLETE" for v in verdicts):
        overall = "INCOMPLETE"
    else:
        overall = "PASS"

    summary = {
        "prereg_sha": PREREG_SHA,
        "overall": overall,
        "units": units,
        "pooled_k": k_pool,
        "pooled_n": n_pool,
        "pooled_coverage": (k_pool / n_pool) if n_pool else None,
        "n_failed_refits": n_failed_refits,
        "wall": wall,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    def fmt_w(u):
        return "—" if u["mean_width"] is None else f"{u['mean_width']:.4g}"

    L = [
        "# TRACE v5.1 coverage — results",
        "",
        f"**Prereg SHA:** `{PREREG_SHA}` "
        "([PREREG_V5_1_COVERAGE.md](../research/PREREG_V5_1_COVERAGE.md)).",
        f"**Gate verdict:** **{overall}**.",
        f"**Wall time:** {wall['wall_sec']:.1f} s on {wall['workers']} workers "
        f"(cap {wall['cap_sec']} s); {wall['tasks_done']}/{wall['tasks_total']} tasks finished.",
        "",
        "Hypothesis H (PREREG_V5_1_COVERAGE §1): on the two linear-mean pure-composition",
        "cells at n = 2000, p = 5, the nominal 90% two-sided percentile bootstrap band",
        "(`BAND_V5_1_PB90`, B = 500) around the v5.1 composition piece and around the v5.1",
        "response piece of the graded L-moment covers the frozen Shapley truth at a rate",
        "inside the binomial pass interval for each of the four (cell × piece) units.",
        "",
        f"Pass interval (fixed in §5): {PASS_LO} ≤ k ≤ {PASS_HI} of {R}",
        "(0.860 ≤ π̂ ≤ 0.935). Unit FAIL (under-covers) if k ≤ 343; unit FAIL",
        "(over-covers) if k ≥ 375. Gate PASS only if all four units PASS. Coverage is a",
        "rate bar; no Holm step. Missing replications are never imputed.",
        "",
        "| unit | n done | k covers | π̂ | pass interval | mean width | "
        "miss below | miss above | forced-miss (refit) | verdict |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for u in units:
        pi = "—" if u["n_done"] == 0 else f"{u['k_cover'] / u['n_done']:.4f}"
        L.append(
            f"| {u['label']} | {u['n_done']} | {u['k_cover']} | {pi} | "
            f"[{PASS_LO}, {PASS_HI}] | {fmt_w(u)} | {u['miss_truth_below_band']} | "
            f"{u['miss_truth_above_band']} | {u['miss_forced_refit']} | **{u['verdict']}** |"
        )
    pool = "—" if n_pool == 0 else f"{k_pool}/{n_pool} ({k_pool / n_pool:.4f})"
    L += [
        "",
        f"**Pooled coverage** (report only, no bar): {pool}.",
        f"**Failed bootstrap refits** (report only): {n_failed_refits} across {len(rows)} "
        "completed replications.",
        "",
        "Full numbers: `benchmarks/results/v5_1_coverage_summary.json`; per-replication",
        "records: `benchmarks/results/v5_1_coverage.jsonl`.",
        "",
        "## Notes and deviations",
        "",
        "1. **Band.** `BAND_V5_1_PB90` as frozen: two-sample percentile bootstrap, rows",
        '   resampled within period, B = 500, stream `seed_of("v51cov", s, n, p, r, b)`,',
        '   `[Q_0.05, Q_0.95]` via `np.quantile(..., method="linear")`, no bias correction,',
        "   no studentising, no BCa. Covered if `lo ≤ truth ≤ hi`. More than 5 of 500",
        "   failed refits makes that replication a miss for both pieces; failed refits",
        "   were never re-drawn.",
        "2. **Point estimator.** `TRACE_V5_1` = `explain_v5_1` with library defaults, on",
        "   arrays centred exactly as the V5.1 gate runner (`centred` of `run_synthetic.py`).",
        "   Bootstrap refits use the already-centred arrays, with no re-centring.",
        "3. **Seeds.** Every index was asserted to lie in 10400–10799 and outside the",
        "   FORBIDDEN set, including the sealed range 9600–9619, which was not touched.",
        "4. **Single pass.** The results file was required not to exist at the start of",
        "   the run. No second pass, no change of B, level, band method, cells, or seeds.",
        "5. **Scope.** A PASS (or a FAIL) is about the band only. The V5.1 accuracy claim",
        "   in V5_1_GATE_RESULTS.md is unaffected. Nothing is claimed about n = 20 000,",
        "   struct cells, `shape_mixed`, ShiftBench, other levels, or non-linear means.",
    ]
    OUT_MD.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
