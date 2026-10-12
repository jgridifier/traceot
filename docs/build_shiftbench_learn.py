"""Write docs/learn/shift_benchmark.html from benchmarks/results/shiftbench_summary.json.

No CDN; the figure is inline SVG built here.

    python docs/build_shiftbench_learn.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "learn" / "shift_benchmark.html"
SUMMARY = ROOT / "benchmarks" / "results" / "shiftbench_summary.json"
FINDINGS = ROOT / "docs" / "benchmark" / "findings.html"  # hand-written paragraphs
METHODS = ["OB", "DFL", "CFM", "RIF", "SHAP"]
STATS = ["mean", "sd", "q10", "q50", "q90"]
SCEN = [
    "null",
    "loc_comp",
    "loc_struct",
    "mixed",
    "scale",
    "tail",
    "correlated",
    "offsetting",
    "hetero",
    "nonlinear",
    "entry_exit",
]
COLORS = {"OB": "#7a7a80", "DFL": "#b85c38", "CFM": "#2f7d4a", "RIF": "#1b6ca8", "SHAP": "#8e44ad"}
SIZE = "n=20000,p=20"

CSS = """
:root { color-scheme: light dark; --fg:#1d1d1f; --bg:#fdfdfc; --muted:#5b5b60; --rule:#ddd;
        --accent:#1b6ca8; --note:#f3f6fa; }
@media (prefers-color-scheme: dark) { :root { --fg:#e9e9ec; --bg:#141416; --muted:#a3a3aa;
  --rule:#333; --accent:#6fb3e6; --note:#1d2229; } svg { background:#fff; border-radius:6px; } }
html { -webkit-text-size-adjust:100%; }
body { margin:0; background:var(--bg); color:var(--fg); font-family: system-ui, -apple-system,
  "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; font-size:1.0625rem; line-height:1.6; }
main { max-width:40rem; margin:0 auto; padding:1.25rem 1rem 4rem; }
h1 { font-size:1.7rem; line-height:1.2; margin:1.2rem 0 .4rem; }
h2 { font-size:1.3rem; margin:2.4rem 0 .6rem; padding-top:.6rem; border-top:1px solid var(--rule); }
a { color:var(--accent); }
.lede { color:var(--muted); }
.status { display:inline-block; font-size:.8rem; font-weight:600; padding:.15rem .5rem;
  border-radius:999px; background:#fff3cd; color:#664d03; }
.note { background:var(--note); border-left:4px solid var(--accent); padding:.6rem .9rem;
  border-radius:4px; margin:1rem 0; }
figure { margin:1.2rem 0; } figure svg { width:100%; height:auto; display:block; }
figcaption { color:var(--muted); font-size:.9rem; margin-top:.35rem; }
.scroll { overflow-x:auto; -webkit-overflow-scrolling:touch; margin:1rem 0; }
table { border-collapse:collapse; font-size:.82rem; font-variant-numeric:tabular-nums;
  min-width:100%; }
caption { text-align:left; font-weight:600; padding:.3rem 0; }
th, td { border-bottom:1px solid var(--rule); padding:.3rem .45rem; text-align:right;
  white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:.9em; }
footer { color:var(--muted); font-size:.85rem; margin-top:3rem; border-top:1px solid var(--rule);
  padding-top:1rem; }
"""


def f3(v, d=3):
    return "n/a" if v is None else f"{v:.{d}f}"


def table(caption, head, rows):
    h = "".join(f"<th>{html.escape(x)}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (
        f'<div class="scroll"><table><caption>{html.escape(caption)}</caption>'
        f"<thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>"
    )


def fig_split():
    """Schematic: A -> counterfactual -> B along one axis."""
    t = 'font-family="system-ui,sans-serif" font-size="13" fill="#1d1d1f"'
    return (
        '<svg role="img" aria-label="Composition then structure" viewBox="0 0 640 120" '
        'xmlns="http://www.w3.org/2000/svg"><rect width="640" height="120" fill="#f7f5f2"/>'
        '<circle cx="80" cy="60" r="26" fill="#1b6ca8"/><circle cx="320" cy="60" r="26" '
        'fill="#9bbfdc"/><circle cx="560" cy="60" r="26" fill="#b85c38"/>'
        '<line x1="108" y1="60" x2="290" y2="60" stroke="#1d1d1f" stroke-width="2"/>'
        '<line x1="350" y1="60" x2="530" y2="60" stroke="#1d1d1f" stroke-width="2"/>'
        f'<text x="58" y="104" {t}>period A</text>'
        f'<text x="268" y="104" {t}>counterfactual</text>'
        f'<text x="536" y="104" {t}>period B</text>'
        f'<text x="140" y="48" {t}>composition</text>'
        f'<text x="400" y="48" {t}>structure</text>'
        "</svg>"
    )


def fig_bars(S):
    """Mean-channel split error by scenario at the primary size."""
    by = S["accuracy"]["channel_by_scen"]
    w, h, left, top = 640, 360, 92, 24
    rowh = (h - top - 30) / len(SCEN)
    vals = [by.get(f"{SIZE}|{m}|mean|{s}") for s in SCEN for m in METHODS]
    vmax = max((v[0] for v in vals if v), default=1) * 1.05
    t = 'font-family="system-ui,sans-serif" font-size="11" fill="#1d1d1f"'
    body = []
    for i, s in enumerate(SCEN):
        y0 = top + i * rowh
        body.append(f'<text x="4" y="{y0 + rowh * 0.6:.1f}" {t}>{s}</text>')
        bh = rowh * 0.8 / len(METHODS)
        for k, m in enumerate(METHODS):
            v = by.get(f"{SIZE}|{m}|mean|{s}")
            if not v:
                continue
            bw = (w - left - 20) * v[0] / vmax
            body.append(
                f'<rect x="{left}" y="{y0 + k * bh:.1f}" width="{bw:.1f}" '
                f'height="{bh - 1:.1f}" fill="{COLORS[m]}"/>'
            )
    for k, m in enumerate(METHODS):
        body.append(
            f'<rect x="{left + k * 100}" y="4" width="10" height="10" '
            f'fill="{COLORS[m]}"/><text x="{left + 14 + k * 100}" y="13" {t}>{m}</text>'
        )
    body.append(f'<text x="{left}" y="{h - 8}" {t}>split error / sd(y_A), 0 to {vmax:.2f}</text>')
    return (
        f'<svg role="img" aria-label="Mean-channel split error by scenario" '
        f'viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="{w}" height="{h}" fill="#f7f5f2"/>{"".join(body)}</svg>'
    )


def results_html(S):
    A, C = S["accuracy"], S["coverage"]
    rows = []
    for m in METHODS:
        k = f"{SIZE}|{m}"
        rows.append(
            [m]
            + [f3(A["channel"][f"{k}|{s}"]) for s in STATS]
            + [f3(A["feature"][k]), f3(A["inert"][k])]
        )
    out = [
        table(
            "Split error at n = 20,000 per period, p = 20 (lower is better)",
            ["method", *STATS, "per-feature", "inert mass"],
            rows,
        )
    ]
    out.append(
        f"<figure>{fig_bars(S)}<figcaption>Mean-channel split error by scenario "
        "(n = 20,000, p = 20). Missing bars: the method does not report that "
        "quantity.</figcaption></figure>"
    )
    rows = []
    for m in METHODS:
        for s in STATS:
            k = f"{m}|{s}"
            if k in C["coverage"]:
                rows.append(
                    [m, s, f3(C["coverage"][k]), f3(C["width"][k]), f3(C["null_fpr"].get(k))]
                )
    out.append(
        table(
            "95% bootstrap bands at n = 2,000 (coverage target 0.95, FPR target 0.05)",
            ["method", "stat", "coverage", "width", "null FPR"],
            rows,
        )
    )
    R = S.get("real")
    if R:
        rows = []
        for m in METHODS:
            f, b, sh = R["full"][m]["split"]["mean"], R["boot_sd"][m]["mean"], R["split_half"][m]
            rows.append(
                [
                    m,
                    f"{f[0]:.3f}",
                    f"{f[1]:.3f}",
                    f"{b[0]:.3f} / {b[1]:.3f}",
                    f3(R["rank_stability"].get(m)),
                    f3(sh["mean_total_gap"]),
                    f3(sh["feature_corr"]),
                ]
            )
        out.append(
            table(
                "Real panel, mean channel (no ground truth)",
                ["method", "comp", "struct", "boot sd", "rank stab.", "half gap", "half corr"],
                rows,
            )
        )
    sc = [r for r in S.get("scale", []) if r.get("n") == 1_000_000]
    if sc:
        rows = [
            [
                r["method"],
                str(r["p"]),
                f3(r.get("sec"), 1) if "sec" in r else r["status"],
                f3(r.get("peak_rss_mb"), 0) if "sec" in r else "",
            ]
            for r in sc
        ]
        out.append(
            table("One million rows per period", ["method", "p", "seconds", "peak MB"], rows)
        )
    return "\n".join(out)


def main():
    S = json.loads(SUMMARY.read_text())
    findings = FINDINGS.read_text() if FINDINGS.exists() else ""
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ShiftBench: how well do methods explain a distribution shift? · traceot</title>
<meta name="description" content="A plain-language guide to the ShiftBench benchmark for
explaining distribution shift, and how five standard methods did.">
<style>{CSS}</style></head><body><main>
<p><a href="../index.html">traceot</a> · <span class="status">pre-registered benchmark</span></p>
<h1>ShiftBench: how well do methods explain a distribution shift?</h1>
<p class="lede">When an outcome's distribution moves between two periods, you want to know
why. This page explains the question, how we built a test with known answers, and how five
standard methods did. TRACE joins the same test when its shift decomposition lands.</p>

<h2 id="question">The question</h2>
<p>Compare an outcome in two periods, A and B. Part of the change happens because the
<em>mix</em> of units changed: their features moved. That part is called
<strong>composition</strong>. The rest happens because the <em>same</em> features now lead
to different outcomes. That part is called <strong>structure</strong>.</p>
<p>To separate them, imagine a third world: period-B features, but period-A rules. Going from
A to that world is composition. Going from that world to B is structure.</p>
<figure>{fig_split()}<figcaption>The counterfactual world sits between the two periods.
</figcaption></figure>
<p>The split can be asked of any summary of the outcome: its mean, its spread, or its tails
(here the 10th and 90th percentiles). Many methods only answer it for the mean.</p>

<h2 id="truth">A test with known answers</h2>
<p>On real data nobody knows the true split, so a method can't be graded. ShiftBench
generates data where we planted the split, and computes the true values from two million
draws. Its eleven scenarios each stress one thing: no shift at all, features moving, rules
changing, both at once, spread changes, tail-only changes with the mean held fixed,
correlated features, a feature shift cancelled by a rule change, noise that depends on a
feature, curved effects, and units entering the panel. Most scenarios also add inert
features that do nothing, to catch methods that blame the wrong feature.</p>
<p>Everything (scenarios, sizes, seeds, metrics, and our predictions about the rivals) was
written down and committed before any method ran. That record is the
<a href="https://github.com/jgridifier/traceot/blob/main/docs/benchmark/PREREG_SHIFTBENCH.md">pre-registration</a>.</p>

<h2 id="methods">The five rivals</h2>
<ul>
<li><strong>Oaxaca-Blinder (OB)</strong>: fit a straight-line model in each period and
split the mean change. The textbook baseline.</li>
<li><strong>DiNardo-Fortin-Lemieux (DFL)</strong>: reweight period-A units so their
features look like period B, then read off any summary of the reweighted outcomes.</li>
<li><strong>Chernozhukov-Fernandez-Val-Melly (CFM)</strong>: model the whole conditional
distribution in period A, then average it over period-B features.</li>
<li><strong>Firpo-Fortin-Lemieux RIF regression (RIF)</strong>: turn each summary into a
per-unit quantity whose average is that summary, then split it like Oaxaca-Blinder.</li>
<li><strong>Shapley drift (SHAP)</strong>: fit a boosted-tree model per period and split the
mean change with Shapley values. A strong, popular choice.</li>
</ul>

<h2 id="results">Results</h2>
{results_html(S)}
{findings}

<h2 id="trace">Where TRACE fits</h2>
<p>TRACE plugs in through one adapter. It runs on the same frozen data, seeds and metrics.
A claim that it beats a rival needs a paired 95% interval that excludes zero at every size.</p>

<footer>Design: <a href="https://github.com/jgridifier/traceot/blob/main/docs/benchmark/SHIFTBENCH.md">SHIFTBENCH.md</a>.
Full tables: <a href="https://github.com/jgridifier/traceot/blob/main/docs/benchmark/SHIFTBENCH_RESULTS.md">SHIFTBENCH_RESULTS.md</a>.
Built by <code>docs/build_shiftbench_learn.py</code>.</footer>
</main></body></html>
"""
    OUT.write_text(page)


if __name__ == "__main__":
    main()
