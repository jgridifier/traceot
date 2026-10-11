"""Write docs/learn/trace.html. No CDN. Figures are inline SVG from this script.

python docs/build_trace_learn.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "learn" / "trace.html"
SIMS = ROOT / "benchmarks" / "results" / "trace_sims.json"
EBM = ROOT / "benchmarks" / "results" / "trace_ebm.json"


def _poly(pts, stroke, width=2):
    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return f'<polyline fill="none" stroke="{stroke}" stroke-width="{width}" points="{d}"/>'


def _frame(w, h, body, label):
    return (
        f'<svg role="img" aria-label="{label}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="{w}" height="{h}" fill="#f7f5f2"/>'
        f"{body}</svg>"
    )


def fig_ebm():
    w, h = 640, 180
    panels = []
    colors = ("#1b6ca8", "#b85c38", "#2f7d4a", "#1d1d1f")
    titles = ("f1(x1)", "f2(x2)", "f3(x3)", "sum")
    for i, (title, color) in enumerate(zip(titles, colors, strict=True)):
        x0 = 16 + i * 156
        pts = []
        for t in range(40):
            u = t / 39
            if i == 0:
                y = math.sin(2 * math.pi * u)
            elif i == 1:
                y = 0.3 + 1.4 * (u - 0.5) ** 2
            elif i == 2:
                y = 0.8 * math.tanh(4 * (u - 0.4))
            else:
                y = (
                    math.sin(2 * math.pi * u)
                    + 0.3
                    + 1.4 * (u - 0.5) ** 2
                    + 0.8 * math.tanh(4 * (u - 0.4))
                ) / 3
            px = x0 + 8 + u * 120
            py = 100 - y * 28
            pts.append((px, py))
        panels.append(
            f'<text x="{x0 + 8}" y="22" font-size="13" font-family="system-ui,sans-serif" fill="#1d1d1f">{title}</text>'
            f'<line x1="{x0 + 8}" y1="100" x2="{x0 + 128}" y2="100" stroke="#ccc"/>'
            + _poly(pts, color)
        )
    return _frame(w, h, "".join(panels), "Four panels: three feature shape functions and their sum")


def fig_compare():
    w, h = 640, 200
    body = []
    body.append(
        '<text x="20" y="22" font-size="13" font-family="system-ui,sans-serif" fill="#1d1d1f">EBM: one mean</text>'
    )
    body.append(
        '<text x="340" y="22" font-size="13" font-family="system-ui,sans-serif" fill="#1d1d1f">TRACE: a quantile fan</text>'
    )
    mean, fans = [], []
    for t in range(50):
        u = t / 49
        x = 24 + u * 250
        m = 110 - 40 * math.sin(math.pi * u)
        mean.append((x, m))
        for k, level in enumerate((0.1, 0.3, 0.5, 0.7, 0.9)):
            spread = (0.55 + 0.9 * u) * (level - 0.5)
            tail = 0.15 * u * ((level - 0.5) ** 3) * 8
            y = 110 - 40 * math.sin(math.pi * u) - 70 * (spread + tail)
            fans.append((k, 340 + u * 270, y))
    body.append(_poly(mean, "#1d1d1f", 2.4))
    for k, color in enumerate(("#9bb8d3", "#5d8eb8", "#1b6ca8", "#5d8eb8", "#9bb8d3")):
        pts = [(x, y) for kk, x, y in fans if kk == k]
        body.append(_poly(pts, color, 1.6 if k != 2 else 2.4))
    return _frame(w, h, "".join(body), "A single mean curve beside a fan of conditional quantiles")


def fig_channels():
    w, h = 640, 200
    # z from the normal quantile function, drawn against u in (0, 1).
    series = [[], [], []]
    norm = NormalDist()
    for t in range(60):
        u = (t + 0.5) / 60
        z = norm.inv_cdf(u)
        xs = 40 + u * 560
        vals = (1.0, z, z**3 - 3 * z)
        for m, v in enumerate(vals):
            series[m].append((xs, v))
    colors = ("#1b6ca8", "#b85c38", "#2f7d4a")
    names = ("location: 1", "scale: z(u)", "shape: z³ − 3z")
    body = ['<line x1="40" y1="110" x2="600" y2="110" stroke="#ccc"/>']
    for m, (pts, color, name) in enumerate(zip(series, colors, names, strict=True)):
        lo = min(y for _x, y in pts)
        hi = max(y for _x, y in pts)
        span = hi - lo or 1
        drawn = [(x, 110 - (y - (lo + hi) / 2) / span * 70) for x, y in pts]
        body.append(_poly(drawn, color, 2))
        body.append(
            f'<text x="{40 + m * 190}" y="28" font-size="13" font-family="system-ui,sans-serif" fill="{color}">{name}</text>'
        )
    body.append(
        '<text x="40" y="188" font-size="12" font-family="system-ui,sans-serif" fill="#5b5b60">quantile level u, from 0 to 1</text>'
    )
    return _frame(
        w, h, "".join(body), "Location, scale and shape basis functions of the quantile level"
    )


def fig_pvalues():
    # A stylised histogram: most mass spread out, a pile near zero.
    counts = [18, 4, 3, 3, 3, 3, 3, 3, 3, 4]
    w, h = 640, 180
    body = [
        '<text x="20" y="22" font-size="13" font-family="system-ui,sans-serif" fill="#1d1d1f">conformal p-values on the test rows</text>'
    ]
    bw = 56
    for i, c in enumerate(counts):
        bh = c * 6
        x = 30 + i * (bw + 6)
        y = 150 - bh
        color = "#b85c38" if i == 0 else "#1b6ca8"
        body.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" fill="{color}"/>')
    body.append(
        '<text x="30" y="170" font-size="12" font-family="system-ui,sans-serif" fill="#5b5b60">0</text>'
    )
    body.append(
        '<text x="560" y="170" font-size="12" font-family="system-ui,sans-serif" fill="#5b5b60">1</text>'
    )
    return _frame(w, h, "".join(body), "Histogram of conformal p-values with extra mass near zero")


def _fmt(v, digits=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _results_html():
    if not SIMS.exists():
        return "<p>The suite has not been run in this copy yet. Criteria and, once run, the numbers are in <code>docs/TRACE_SIMS.md</code>.</p>"
    sims = json.loads(SIMS.read_text())
    rows = []
    for key in ("s1", "s2", "s3", "s4"):
        block = sims.get(key)
        if not block:
            continue
        sec = block.get("seconds")
        sec_txt = f"{sec:.2f}s" if isinstance(sec, float) else "—"
        rows.append(
            f"<tr><td>{key.upper()}</td><td>{'pass' if block.get('pass') else 'fail'}</td>"
            f"<td>{sec_txt}</td></tr>"
        )
    table = (
        '<div class="scroll"><table><caption>Suite checks</caption>'
        "<thead><tr><th>Design</th><th>Result</th><th>Wall</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )
    extra = ""
    if EBM.exists():
        ebm = json.loads(EBM.read_text())
        erows = []
        for key in ("s1_period2", "vendor_spend", "scale_small"):
            case = ebm.get(key)
            if not case:
                continue
            tr = case["trace"]
            eb = case["ebm"]
            if not eb.get("available"):
                erows.append(
                    f"<tr><td>{case['name']}</td><td>{_fmt(tr['crps'])}</td><td>{_fmt(tr['cover80'])}</td>"
                    f"<td colspan='2'>EBM not installed ({eb.get('reason', '')})</td></tr>"
                )
                continue
            conf = eb["conformal"]
            erows.append(
                f"<tr><td>{case['name']} TRACE</td><td>{_fmt(tr['crps'])}</td><td>{_fmt(tr['cover80'])}</td>"
                f"<td>{_fmt(tr['mse_mean'])}</td><td>{_fmt(tr['fit_seconds'], 2)}</td></tr>"
                f"<tr><td>{case['name']} EBM + conformal</td><td>{_fmt(conf['crps'])}</td>"
                f"<td>{_fmt(conf['cover80'])}</td><td>{_fmt(conf['mse_mean'])}</td>"
                f"<td>{_fmt(eb['fit_seconds'], 2)}</td></tr>"
            )
        extra = (
            '<div class="scroll"><table><caption>TRACE and EBM plus residual quantiles</caption>'
            "<thead><tr><th></th><th>CRPS</th><th>80% cover</th><th>MSE of mean</th><th>Fit s</th></tr></thead><tbody>"
            + "".join(erows)
            + "</tbody></table></div>"
            "<p>Full tables, including where a mean EBM wins, are in <code>docs/TRACE_SIMS.md</code>.</p>"
        )
    return table + extra


def main():
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Learn TRACE · traceot</title>
<meta name="description" content="What explainable boosting machines are, how TRACE models a whole conditional distribution, and the checks around it.">
<style>
  :root {{ color-scheme: light dark; --fg:#1d1d1f; --bg:#fdfdfc; --muted:#5b5b60; --rule:#ddd; --accent:#1b6ca8; --note:#f3f6fa; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --fg:#e9e9ec; --bg:#141416; --muted:#a3a3aa; --rule:#333; --accent:#6fb3e6; --note:#1d2229; }}
    svg {{ background:#fff; border-radius:6px; }}
  }}
  html {{ -webkit-text-size-adjust:100%; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
         font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif;
         font-size:1.0625rem; line-height:1.6; }}
  main {{ max-width:40rem; margin:0 auto; padding:1.25rem 1rem 4rem; }}
  h1 {{ font-size:1.7rem; line-height:1.2; margin:1.2rem 0 .4rem; }}
  h2 {{ font-size:1.25rem; margin:1.6rem 0 .5rem; }}
  h3 {{ font-size:1.05rem; margin:1.3rem 0 .3rem; }}
  a {{ color:var(--accent); }}
  .lede {{ color:var(--muted); }}
  .status {{ display:inline-block; font-size:.8rem; font-weight:600; padding:.15rem .5rem; border-radius:999px; background:#fff3cd; color:#664d03; }}
  .note {{ background:var(--note); border-left:4px solid var(--accent); padding:.6rem .9rem; border-radius:4px; margin:1rem 0; }}
  figure {{ margin:1rem 0; }}
  figure svg {{ width:100%; height:auto; display:block; }}
  figcaption {{ color:var(--muted); font-size:.9rem; margin-top:.35rem; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:.9em; }}
  .scroll {{ overflow-x:auto; -webkit-overflow-scrolling:touch; margin:1rem 0; }}
  table {{ border-collapse:collapse; font-size:.85rem; font-variant-numeric:tabular-nums; min-width:100%; }}
  caption {{ text-align:left; font-weight:600; padding:.3rem 0; }}
  th, td {{ border-bottom:1px solid var(--rule); padding:.3rem .45rem; text-align:right; white-space:nowrap; }}
  th:first-child, td:first-child {{ text-align:left; }}
  .tabs {{ display:flex; flex-wrap:wrap; gap:.4rem; margin:1rem 0 .2rem; }}
  .tabs label {{
    border:1px solid var(--rule); border-radius:999px; padding:.25rem .7rem; cursor:pointer; font-size:.92rem;
  }}
  .panel {{ display:none; }}
  input.tab {{ position:absolute; opacity:0; pointer-events:none; }}
  #t1:checked ~ .tabs label[for="t1"],
  #t2:checked ~ .tabs label[for="t2"],
  #t3:checked ~ .tabs label[for="t3"],
  #t4:checked ~ .tabs label[for="t4"] {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
  #t1:checked ~ .p1, #t2:checked ~ .p2, #t3:checked ~ .p3, #t4:checked ~ .p4 {{ display:block; }}
  footer {{ color:var(--muted); font-size:.85rem; margin-top:2rem; border-top:1px solid var(--rule); padding-top:1rem; }}
</style>
</head>
<body>
<main>
<p><a href="../index.html">traceot</a> · <span class="status">research, unvalidated</span></p>
<h1>TRACE, beside explainable boosting</h1>
<p class="lede">A glass-box model of a whole conditional distribution, and the checks that sit next to it. TRACE is our own unpublished method. The boosting ideas it borrows are older, and they are cited below.</p>

<input class="tab" type="radio" name="tab" id="t1" checked>
<input class="tab" type="radio" name="tab" id="t2">
<input class="tab" type="radio" name="tab" id="t3">
<input class="tab" type="radio" name="tab" id="t4">
<div class="tabs" role="tablist">
  <label for="t1">EBM refresher</label>
  <label for="t2">TRACE vs EBM</label>
  <label for="t3">How TRACE works</label>
  <label for="t4">Checks and oddness</label>
</div>

<section class="panel p1">
<h2>What an explainable boosting machine is</h2>
<p>An explainable boosting machine (EBM) predicts a single number, usually a mean, as a sum of small functions of one feature at a time:</p>
<p><code>g(x) = g0 + f1(x1) + f2(x2) + … + a few pairwise terms.</code></p>
<p>Each <code>fj</code> is a shape function. In the usual implementation the feature is cut into bins, and the fit cycles through the features: hold the rest of the sum fixed, update this feature's bins from the residual gradient, then move on. Many rounds, a small learning rate. Because the pieces are added, you can plot each <code>fj</code> on its own and read off what that feature does. That is the glass box.</p>
<figure>{fig_ebm()}
<figcaption>Cyclic boosting learns one shape per feature. The prediction is their sum. The picture is a sketch, not a fitted model.</figcaption></figure>
<p>Binning is what makes this cheap. A column becomes a bin code, and the update is a histogram of gradients. The cost of a round grows with the number of rows, not with the square of the number of rows.</p>
<p>Pairs are optional. Looking at every pair in full is expensive, so GA2M / FAST first screens candidate pairs from the residuals and only then fits the ones that look useful (Lou, Caruana, Gehrke and Hooker, <em>Accurate Intelligible Models with Pairwise Interactions</em>, KDD 2013). InterpretML packages this whole pattern: cyclic boosting, binning, the FAST-style interaction pass, and the plots (Nori, Jenkins, Koch and Caruana, <em>InterpretML: A Unified Framework for Machine Learning Interpretability</em>, arXiv:1909.09223, 2019).</p>
<h3>Where EBMs are strong</h3>
<ul>
<li>They are built for the thing most regression losses ask for: a conditional mean (or a similar one-number summary). On mean squared error they are often the right baseline, and they can beat a distributional model that spends its capacity on scale and tails.</li>
<li>The shape plots are the explanation. There is no second model that “explains” the first.</li>
<li>The public implementation in InterpretML is careful, fast, and widely exercised. TRACE is a research prototype. When the comparison below says EBM wins, that is the result.</li>
</ul>
<p>What a mean model does not show is a change in width or in the tail when the centre stays put. That is the gap TRACE is aimed at.</p>
</section>

<section class="panel p2">
<h2>Mean versus the whole distribution</h2>
<figure>{fig_compare()}
<figcaption>Left: one curve, the sort of summary an EBM regression returns. Right: several quantile levels of one conditional distribution. A feature can move the middle, the width, or the tails.</figcaption></figure>
<div class="scroll"><table>
<caption>What each model is willing to explain</caption>
<thead><tr><th></th><th>EBM regression</th><th>EBM + residual quantiles</th><th>TRACE</th></tr></thead>
<tbody>
<tr><td>Target</td><td>a mean</td><td>a mean, plus a residual band</td><td>Q(u | x) on a grid of u</td></tr>
<tr><td>Form</td><td>sum of shape functions</td><td>that sum, shifted by calibration residuals</td><td>Q0(u) + sum of location, scale and shape channels</td></tr>
<tr><td>Plot</td><td>fj(xj)</td><td>the same fj, and one global band</td><td>per feature: a step in location, in scale, and in the tail</td></tr>
<tr><td>Interactions</td><td>FAST screen, then a pair table</td><td>the same</td><td>a residual-gradient screen, then a pair table per channel</td></tr>
<tr><td>Order of columns</td><td>the cycle order is part of the path</td><td>same</td><td>the sum is order-invariant by construction</td></tr>
</tbody>
</table></div>
<p>The fair distributional baseline is not “EBM has no intervals”. Fit the EBM, take residuals on a calibration split, and add those residual quantiles to the prediction. The band then has a width, but the width does not depend on x, and it does not have a separate tail shape per feature. TRACE's scale channel is allowed to depend on x. Its shape channel is allowed to bend the tails.</p>
<p>The residual-quantile baseline can still win on coverage, because split-conformal residuals are built to hit a coverage target when the residuals are exchangeable. TRACE's grid pinball loss does not promise 80% or 90% coverage. Both numbers are reported.</p>
{_results_html()}
<p>There is no winner rule on the vendor-spend draw. It is a simulated table of units, price, deliveries, distance, quality and season, with a planted location, a planted scale and a small tail. The metrics sit next to S1–S4 in the simulation note.</p>
</section>

<section class="panel p3">
<h2>How TRACE fits</h2>
<p>The model, written on the quantile function:</p>
<p><code>Q(u | x) = Q0(u) + Σj fj(xj, u) + Σpairs fjk(xj, xk, u)</code></p>
<p>then a monotone projection in <code>u</code>. <code>Q0</code> is the marginal quantile of the training outcomes. Each feature function uses a short basis in the level <code>u</code>.</p>
<figure>{fig_channels()}
<figcaption>The three channels, drawn against the quantile level. Location is flat. Scale follows the normal quantile z(u). Shape follows z³ − 3z, which is small in the middle and large in the tails. Each column is rescaled to unit root-mean-square before fitting.</figcaption></figure>
<ol>
<li>Cut every feature into histogram bins. A missing value gets its own bin. Codes are one byte when there are at most 255 bins.</li>
<li>Each round, compute the pinball subgradient of the current quantile fit once, on a fixed grid (19 levels by default). Average pinball on that grid is the training loss, and twice that average is the reported CRPS approximation.</li>
<li>Update every feature from that same gradient. Do not refresh the gradient between features. Refreshing it would make the answer depend on which column was visited first, which is the failure TRACE is written to avoid.</li>
<li>Add the updates in a canonical order of the raw columns, so a permutation of the columns is the same sum.</li>
<li>Stop early when a held-out slice of the pinball loss stops improving. Drop a feature whose held-out gain stays below a threshold, and, with several bags, keep a feature only if it is selected in at least 70% of the bags.</li>
<li>Screen pairs by a residual-gradient energy on a two-dimensional histogram, fit the selected pairs, then project each predicted curve so it increases in <code>u</code>.</li>
</ol>
<p>The projection is what makes the delivered quantiles monotone. It is also what blurs a shape interaction: a bend that would have decreased in <code>u</code> gets flattened, and the pure interaction contrast shrinks. A pure location interaction, constant in <code>u</code>, is left alone. The measured ratios are in the simulation note. Pair tables in v0 are centered by their grand mean only, so a pair can still carry a main-effect margin. That is an open gap, not a hidden feature.</p>
<p>A round is one pass over the rows for the gradient and one pass per feature for the histogram. Nothing in that path builds an <code>n × n</code> matrix. The design note has the pseudocode and the complexity table: <a href="https://github.com/jgridifier/traceot/blob/main/docs/TRACE_DESIGN.md">TRACE_DESIGN.md</a>.</p>
</section>

<section class="panel p4">
<h2>Checks and oddness</h2>
<p>The booster will happily use a column that is a copy of the outcome, a copy of another column, or a column that only lines up with the outcome inside a random split. The checks are part of the method, and they are ordinary statistics with fixed seeds, not a second model.</p>
<ul>
<li><strong>Target proxy.</strong> Correlation of each column with <code>y</code>, and with <code>y</code> shifted by a row. A column that leads <code>y</code> (today's feature is tomorrow's outcome) shows up at a positive shift.</li>
<li><strong>Duplicate.</strong> Bit-identical columns, and pairs that are almost collinear.</li>
<li><strong>Placebo target.</strong> Shuffle <code>y</code> and look at the largest correlation any feature manages. A real proxy sits far above that maximum.</li>
<li><strong>Adversarial validation.</strong> Wasserstein distance, per feature, between the training window and the test window. A time-only column moves a lot under a time split and little under a random split.</li>
<li><strong>Time-respecting importance.</strong> Fit on the past, shuffle a column on the future, and see whether the pinball loss actually gets worse. A random split can still reward a column that will not travel forward.</li>
</ul>
<h3>Oddness</h3>
<p>Hold out a calibration set. Score every row by its mean pinball loss under the fitted quantiles. The conformal p-value of a new score <code>s</code> is <code>(1 + number of calibration scores ≥ s) / (n_cal + 1)</code>. Under exchangeability, those p-values stay above <code>α</code> except for about a fraction <code>α</code> of ordinary rows. A planted spike in <code>y</code> should land in the tail.</p>
<figure>{fig_pvalues()}
<figcaption>Sketch of the intended picture: ordinary rows spread across the p-value range, anomalous rows pile up near zero. The real false-alarm rate and power are measured in S3, not in this drawing.</figcaption></figure>
<p>The planted designs, the thresholds that were written down first, and the numbers that came back are in <a href="https://github.com/jgridifier/traceot/blob/main/docs/TRACE_SIMS.md">TRACE_SIMS.md</a>.</p>
</section>

<footer>
<p>Personal research project. Simulated examples only. TRACE is unpublished and unvalidated.</p>
<p>Background: Nori, Jenkins, Koch, Caruana, arXiv:1909.09223 (2019). Lou, Caruana, Gehrke, Hooker, KDD (2013). No page numbers are stated here beyond what those records themselves print in the design note.</p>
</footer>
</main>
</body>
</html>
"""
    OUT.write_text(html)
    print(OUT, "bytes", OUT.stat().st_size)


if __name__ == "__main__":
    main()
