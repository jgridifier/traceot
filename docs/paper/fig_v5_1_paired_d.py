#!/usr/bin/env python3
# ruff: noqa: E501
"""Draw the §5 paired-d figure from already-committed V5.1 gate outputs.

Reads only:
  benchmarks/results/v5_1_gate_shape.jsonl
  benchmarks/results/shiftbench_truth_shape.json
  benchmarks/results/v5_1_gate_summary.json   (self-check against published means)

Writes only:
  docs/learn/figures/v5_1_paired_d.svg

No TRACE fit is run. No seed stream is drawn. Indices 9600–9619 are not read
or written. The error formula is the same `graded_err` / `normalize_lsplit`
as `benchmarks/shiftbench/summarize_v4.py`, used by `summarize_v5_1_gate.py`.

Usage (from the repository root):

    python docs/paper/fig_v5_1_paired_d.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "benchmarks" / "results"
OUT = ROOT / "docs" / "learn" / "figures" / "v5_1_paired_d.svg"

CELLS = (
    ("skew_comp", 2000, 5, 50),
    ("tail_comp", 2000, 5, 50),
    ("skew_comp", 20000, 20, 20),
    ("tail_comp", 20000, 20, 20),
)


# --- scoring, copied from summarize_v4.py (do not change) -----------------


def normalize_lsplit(lsplit):
    """TRACE {comp:[],struct:[]} or rival {l1:(c,s),...} → {comp,struct} lists."""
    if lsplit is None:
        return None
    if "comp" in lsplit and "struct" in lsplit:
        return {
            "comp": [float(x) for x in lsplit["comp"]],
            "struct": [float(x) for x in lsplit["struct"]],
        }
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


# --- data -----------------------------------------------------------------


def paired_d(rows, truth, scen, n):
    """d_r = err_v5.1 − err_best_rival, same loop as summarize_v5_1_gate.py."""
    t = truth[scen]
    sh = t["shapley"]
    k = t["graded"][0]
    rs = sorted(
        [r for r in rows if r["scen"] == scen and r["n"] == n],
        key=lambda r: r["rep"],
    )
    diffs = []
    for r in rs:
        e5 = graded_err(r["TRACE_V5_1"]["lsplit"], sh, k)
        best = float("inf")
        for _nm, rv in r["rivals"].items():
            if "error" in rv:
                continue
            e = graded_err(rv["lsplit"], sh, k)
            if math.isfinite(e) and e < best:
                best = e
        diffs.append(e5 - best)
    return diffs


def load_and_check():
    rows = [json.loads(x) for x in (RES / "v5_1_gate_shape.jsonl").read_text().splitlines() if x]
    truth = json.loads((RES / "shiftbench_truth_shape.json").read_text())
    summary = json.loads((RES / "v5_1_gate_summary.json").read_text())
    cells = []
    for scen, n, p, reps in CELLS:
        d = paired_d(rows, truth, scen, n)
        pub = next(c for c in summary["cells"] if c["scen"] == scen and c["n"] == n)
        if len(d) != reps or len(d) != pub["reps"]:
            raise SystemExit(f"rep count mismatch {scen} n={n}: got {len(d)}, expected {reps}")
        mean = sum(d) / len(d)
        if not math.isclose(mean, pub["mean_d"], rel_tol=0, abs_tol=1e-15):
            raise SystemExit(
                f"mean d mismatch {scen} n={n}: computed {mean!r} vs published {pub['mean_d']!r}"
            )
        cells.append(
            {
                "scen": scen,
                "n": n,
                "p": p,
                "reps": reps,
                "d": d,
                "mean_d": pub["mean_d"],
                "upper_t": pub["upper_t"],
                "upper_boot": pub["upper_boot"],
                "delta": pub["delta"],
                "verdict": pub["verdict"],
            }
        )
    return cells


# --- SVG ------------------------------------------------------------------


def fmt(x, sig=4):
    """Match the gate-results table's :.4g style for annotations."""
    return f"{x:.4g}"


def nice_ticks(lo, hi, n=5):
    """About n ticks on [lo, hi], on a 1-2-5 grid. Purely deterministic."""
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / (n - 1)
    exp = math.floor(math.log10(raw))
    frac = raw / (10**exp)
    if frac <= 1:
        step = 10**exp
    elif frac <= 2:
        step = 2 * 10**exp
    elif frac <= 5:
        step = 5 * 10**exp
    else:
        step = 10 ** (exp + 1)
    # start at a multiple of step at or below lo
    k0 = math.floor(lo / step - 1e-12)
    k1 = math.ceil(hi / step + 1e-12)
    ticks = []
    k = k0
    while k <= k1:
        v = k * step
        if lo - 1e-12 * span <= v <= hi + 1e-12 * span:
            ticks.append(v)
        k += 1
    return ticks or [lo, hi]


def beeswarm(values, y_of, y_top, y_bot, radius):
    """Deterministic 1-d packing: sorted by value, first x-slot with no overlap.

    y_of maps a data value to an SVG y (pixels). Returns (x_offset, y_svg) in
    panel coordinates, x_offset in pixels from the mid-line, alternating sides.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = []  # (x_off, y, i)
    out = [None] * len(values)
    min_d2 = (2.0 * radius) ** 2
    for i in order:
        y = y_of(values[i])
        # clamp into the panel so a point is never drawn on the frame
        y = min(max(y, y_top + radius), y_bot - radius)
        slot = 0
        while True:
            # slot 0 at x=0; then +1, -1, +2, -2, ...
            if slot == 0:
                x = 0.0
            else:
                side = 1 if slot % 2 == 1 else -1
                k = (slot + 1) // 2
                x = side * k * (2.0 * radius)
            ok = True
            for px, py, _ in placed:
                dy = y - py
                dx = x - px
                if dx * dx + dy * dy < min_d2 - 1e-9:
                    ok = False
                    break
            if ok:
                placed.append((x, y, i))
                out[i] = (x, y)
                break
            slot += 1
            if slot > 40:
                # should never happen at these n; park the point on the mid-line
                out[i] = (0.0, y)
                placed.append((0.0, y, i))
                break
    return out


def svg_escape(s):
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render(cells):
    # layout: 2x2 panels, light paper background. Room above each panel for a
    # two-line heading (title + published numbers) so nothing sits on the points.
    W, H = 900, 760
    margin_l, margin_r, margin_t, margin_b = 66, 28, 86, 48
    gap_x, gap_y = 62, 86
    inner_w = W - margin_l - margin_r
    inner_h = H - margin_t - margin_b
    pw = (inner_w - gap_x) / 2
    ph = (inner_h - gap_y) / 2

    bg = "#fdfdfc"
    ink = "#1d1d1f"
    muted = "#5b5b60"
    rule = "#c8c8cc"
    point = "#1b6ca8"
    point_edge = "#0e4a75"
    zero = "#c0392b"
    mean_c = "#0b6e4f"

    font = "system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif"
    radius = 3.5

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
        '<title id="title">Paired d of TRACE v5.1 versus the best rival, four gate cells</title>',
        (
            '<desc id="desc">One point per already-committed gate replication. '
            "Solid red line at 0 (the BEST bar). Dashed green line at the published mean d "
            "from V5_1_GATE_RESULTS.md. Drawn from v5_1_gate_shape.jsonl; no new fit.</desc>"
        ),
        f'<rect width="{W}" height="{H}" fill="{bg}"/>',
        f'<text x="{W / 2}" y="26" text-anchor="middle" font-family="{font}" font-size="15.5" font-weight="600" fill="{ink}">Paired d = err(v5.1) − err(best rival), graded channel</text>',
    ]

    def n_label(n):
        return "20 000" if n == 20000 else str(n)

    for idx, cell in enumerate(cells):
        col, row = idx % 2, idx // 2
        x0 = margin_l + col * (pw + gap_x)
        y0 = margin_t + row * (ph + gap_y)
        dvals = cell["d"]
        ymin = min(min(dvals), 0.0, cell["mean_d"], cell["upper_t"], cell["upper_boot"])
        ymax = max(max(dvals), 0.0, cell["mean_d"], cell["upper_t"], cell["upper_boot"])
        pad = 0.12 * (ymax - ymin if ymax > ymin else 1.0)
        ymin -= pad
        ymax += pad

        def y_of(v, _y0=y0, _ph=ph, _ymin=ymin, _ymax=ymax):
            return _y0 + (_ymax - v) / (_ymax - _ymin) * _ph

        parts.append(
            f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{pw:.2f}" height="{ph:.2f}" fill="#ffffff" stroke="{rule}" stroke-width="1"/>'
        )

        title = (
            f"{cell['scen']}   ·   n = {n_label(cell['n'])},  p = {cell['p']},  {cell['reps']} reps"
        )
        stats = (
            f"mean d = {fmt(cell['mean_d'])}"
            f"    upper t = {fmt(cell['upper_t'])}"
            f"    upper boot = {fmt(cell['upper_boot'])}"
            f"    δ = {fmt(cell['delta'])}"
            f"    {cell['verdict']}"
        )
        parts.append(
            f'<text x="{x0 + pw / 2:.2f}" y="{y0 - 22:.2f}" text-anchor="middle" font-family="{font}" font-size="12.5" font-weight="600" fill="{ink}">{svg_escape(title)}</text>'
        )
        parts.append(
            f'<text x="{x0 + pw / 2:.2f}" y="{y0 - 7:.2f}" text-anchor="middle" font-family="{font}" font-size="10" fill="{muted}">{svg_escape(stats)}</text>'
        )

        ticks = nice_ticks(ymin, ymax, n=6)
        for t in ticks:
            yy = y_of(t)
            if yy < y0 - 0.5 or yy > y0 + ph + 0.5:
                continue
            parts.append(
                f'<line x1="{x0:.2f}" x2="{x0 + pw:.2f}" y1="{yy:.2f}" y2="{yy:.2f}" stroke="{rule}" stroke-width="0.6" stroke-dasharray="2,3"/>'
            )
            lab = fmt(t if abs(t) > 1e-14 else 0.0)
            parts.append(
                f'<text x="{x0 - 8:.2f}" y="{yy + 4:.2f}" text-anchor="end" font-family="{font}" font-size="10" fill="{muted}">{lab}</text>'
            )

        # zero line (BEST bar) — label just inside, left
        yz = y_of(0.0)
        parts.append(
            f'<line x1="{x0:.2f}" x2="{x0 + pw:.2f}" y1="{yz:.2f}" y2="{yz:.2f}" stroke="{zero}" stroke-width="1.5"/>'
        )
        # put the "0 (BEST bar)" label on the right, slightly above or below the
        # line depending on which side has more free space
        zero_dy = -6 if (yz - y0) > 18 else 12
        parts.append(
            f'<text x="{x0 + pw - 6:.2f}" y="{yz + zero_dy:.2f}" text-anchor="end" font-family="{font}" font-size="9.5" fill="{zero}">0  (BEST bar)</text>'
        )

        # published mean d — label on the left, inside, opposite the zero label side
        ym = y_of(cell["mean_d"])
        parts.append(
            f'<line x1="{x0:.2f}" x2="{x0 + pw:.2f}" y1="{ym:.2f}" y2="{ym:.2f}" stroke="{mean_c}" stroke-width="1.4" stroke-dasharray="5,3.5"/>'
        )
        mean_dy = 13 if ym < yz else -5
        # if mean is very close to a panel edge, flip
        if ym + mean_dy > y0 + ph - 2:
            mean_dy = -5
        if ym + mean_dy < y0 + 10:
            mean_dy = 13
        parts.append(
            f'<text x="{x0 + 6:.2f}" y="{ym + mean_dy:.2f}" text-anchor="start" font-family="{font}" font-size="9.5" fill="{mean_c}">mean d</text>'
        )

        # points, centred
        mid_x = x0 + pw * 0.5
        positions = beeswarm(dvals, y_of, y0, y0 + ph, radius)
        for xoff, yy in positions:
            parts.append(
                f'<circle cx="{mid_x + xoff:.2f}" cy="{yy:.2f}" r="{radius}" fill="{point}" fill-opacity="0.84" stroke="{point_edge}" stroke-width="0.5"/>'
            )

        if col == 0:
            cx, cy = x0 - 52, y0 + ph / 2
            parts.append(
                f'<text x="{cx:.2f}" y="{cy:.2f}" text-anchor="middle" font-family="{font}" font-size="11" fill="{ink}" transform="rotate(-90 {cx:.2f} {cy:.2f})">paired d</text>'
            )

    parts.append(
        f'<text x="{W / 2}" y="{H - 26}" text-anchor="middle" font-family="{font}" font-size="10.5" fill="{muted}">Solid red line: 0, the BEST bar (one-sided 95% upper bound of the mean must be below 0).  Dashed green line: published mean d.</text>'
    )
    parts.append(
        f'<text x="{W / 2}" y="{H - 12}" text-anchor="middle" font-family="{font}" font-size="10" fill="{muted}">Points: committed replications in v5_1_gate_shape.jsonl, scored with the same graded_err as summarize_v5_1_gate.py.  Numbers quoted from V5_1_GATE_RESULTS.md.  No new run.</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main():
    cells = load_and_check()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    svg = render(cells)
    OUT.write_text(svg)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} bytes)")
    for c in cells:
        print(
            f"  {c['scen']:10s} n={c['n']:5d}  reps={c['reps']:2d}  "
            f"mean_d={c['mean_d']:.6g}  [{min(c['d']):.6g}, {max(c['d']):.6g}]  {c['verdict']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
