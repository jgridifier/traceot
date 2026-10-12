# ruff: noqa: E501, B905, F841
"""Build docs/learn/trace_theory.html from computations, not from hand-drawn numbers.

The page teaches the results in docs/theory/THEORY.md. Check IDs point at
docs/theory/verify/RESULTS.txt. This script does not change those criteria.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "learn" / "trace_theory.html"


def pstar_matrix(k_max: int, u: np.ndarray) -> np.ndarray:
    """Shifted Legendre P*_0 .. P*_{k_max-1} on u, Hosking's convention."""
    u = np.asarray(u, dtype=np.float64)
    cols = [np.ones_like(u), 2.0 * u - 1.0]
    x = 2.0 * u - 1.0
    for k in range(1, k_max - 1):
        nxt = ((2 * k + 1) * x * cols[k] - k * cols[k - 1]) / (k + 1)
        cols.append(nxt)
    return np.stack(cols[:k_max], axis=1)


def gl01(n: int):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def sample_lmoments(x: np.ndarray, r_max: int) -> np.ndarray:
    """Hosking unbiased L-moments. x shape (reps, m)."""
    xs = np.sort(np.asarray(x, dtype=np.float64), axis=-1)
    m = xs.shape[-1]
    j = np.arange(1, m + 1, dtype=np.float64)
    b = []
    for r in range(r_max):
        wts = np.ones(m)
        for s in range(1, r + 1):
            wts = wts * (j - s) / (m - s)
        wts = wts.copy()
        wts[:r] = 0.0
        b.append((xs * wts).sum(-1) / m)
    b = np.stack(b, axis=-1)
    out = []
    for r in range(r_max):
        acc = np.zeros(xs.shape[0])
        for k in range(r + 1):
            coef = ((-1) ** (r - k)) * math.comb(r, k) * math.comb(r + k, k)
            acc = acc + coef * b[..., k]
        out.append(acc)
    return np.stack(out, axis=-1)


def plug_in_lmoments(x: np.ndarray, r_max: int, n_u: int = 64) -> np.ndarray:
    """∫ Q̂(u) P*_{r-1}(u) du with the lower pinball endpoint as Q̂."""
    xs = np.sort(np.asarray(x, dtype=np.float64), axis=-1)
    m = xs.shape[-1]
    u, w = gl01(n_u)
    idx = np.clip(np.ceil(u * m) - 1.0, 0, m - 1).astype(int)
    q = xs[:, idx]
    p = pstar_matrix(r_max, u)
    return (q * w) @ p


def svg_bars(labels, series, colors, ylabel, title, width=640, height=230):
    """Grouped or stacked bars. series is a list of (name, values) stacked from zero if one group per label.

    Here each label is one stack: series[i][1][j] is the piece of stack j.
    """
    n = len(labels)
    totals = [sum(s[1][j] for s in series) for j in range(n)]
    top = max(totals) * 1.18 if totals else 1.0
    left, right, top_pad, bot = 48, 16, 28, 46
    plot_w = width - left - right
    plot_h = height - top_pad - bot
    gap = 18
    bw = (plot_w - gap * (n + 1)) / n
    parts = [
        f'<svg role="img" aria-label="{title}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
        '<rect width="100%" height="100%" fill="#f7f5f2"/>',
        f'<text x="{left}" y="18" font-size="13" font-family="system-ui,sans-serif" fill="#1d1d1f">{title}</text>',
    ]
    for tick in range(5):
        val = top * tick / 4
        y = top_pad + plot_h - (val / top) * plot_h
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e4e0da"/>'
        )
        parts.append(
            f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
            f'font-family="system-ui,sans-serif" fill="#5b5b60">{val:.2f}</text>'
        )
    legend_x = left
    for name, _vals, color in [(s[0], s[1], colors[i]) for i, s in enumerate(series)]:
        parts.append(
            f'<rect x="{legend_x}" y="{height - 16}" width="10" height="10" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{legend_x + 14}" y="{height - 7}" font-size="11" '
            f'font-family="system-ui,sans-serif" fill="#1d1d1f">{name}</text>'
        )
        legend_x += 14 + 7 * len(name) + 16
    for j, label in enumerate(labels):
        x = left + gap + j * (bw + gap)
        y_cursor = top_pad + plot_h
        for i, (_name, vals) in enumerate(series):
            h = (vals[j] / top) * plot_h
            y_cursor -= h
            parts.append(
                f'<rect x="{x:.1f}" y="{y_cursor:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{colors[i]}"/>'
            )
        parts.append(
            f'<text x="{x + bw / 2:.1f}" y="{height - 28}" text-anchor="middle" font-size="11" '
            f'font-family="system-ui,sans-serif" fill="#1d1d1f">{label}</text>'
        )
    parts.append(
        f'<text x="14" y="{top_pad + 12}" font-size="11" font-family="system-ui,sans-serif" fill="#5b5b60" transform="rotate(-90 14 {top_pad + 40})">{ylabel}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def compute():
    rng = np.random.default_rng(0)
    k = 6
    u, w = gl01(k + 2)
    p = pstar_matrix(k, u)
    # Two polynomial quantile functions. Seed 0, same construction as a teaching pair.
    c_a = np.array([0.2, 0.8, 0.05, -0.02, 0.01, 0.0])
    c_b = np.array([-0.1, 0.35, -0.12, 0.04, -0.01, 0.02])
    dc = c_a - c_b
    curve = p @ dc
    w2 = float(np.sum(w * curve**2))
    via_c = float(np.sum(dc**2 / (2 * np.arange(k) + 1)))
    via_l = float(np.sum((2 * np.arange(k) + 1) * (dc / (2 * np.arange(k) + 1)) ** 2))
    via_a = float(np.sum((dc / np.sqrt(2 * np.arange(k) + 1)) ** 2))
    wrong = float(np.sum((2 * np.arange(k) + 1) * dc**2))
    channel = dc**2 / (2 * np.arange(k) + 1)
    # Truncation at K=3
    resid = float(np.sum(channel[3:]))
    kept = float(np.sum(channel[:3]))

    # Shapley players: intercept and three features, expansion units, channel k=1 so w=1/3.
    a_players = np.array([0.30, 0.55, -0.20, 0.15])
    weight = 1.0 / 3.0
    phi = weight * a_players * a_players.sum()
    names = ["intercept", "x1", "x2", "x3"]

    # Monotone region for K=4, c1 = 1.
    n_t3, n_t4 = 81, 61
    t3 = np.linspace(-0.5, 0.5, n_t3)
    t4 = np.linspace(-0.25, 0.45, n_t4)
    ug = np.linspace(0, 1, 401)
    mono = np.zeros((n_t4, n_t3), dtype=bool)
    for i, tau4 in enumerate(t4):
        for j, tau3 in enumerate(t3):
            c1 = 1.0
            c2 = tau3 * 5.0 * c1 / 3.0
            c3 = tau4 * 7.0 * c1 / 3.0
            # Q' = 2 c1 + c2 (12u-6) + c3 (60u^2 - 60u + 12)
            qp = 2 * c1 + c2 * (12 * ug - 6) + c3 * (60 * ug**2 - 60 * ug + 12)
            mono[i, j] = bool(np.min(qp) >= -1e-9)

    # Midpoint Gram correlation of Legendre K=4.
    def max_corr(n_grid: int) -> float:
        uu = (np.arange(n_grid) + 0.5) / n_grid
        ww = np.full(n_grid, 1.0 / n_grid)
        b = pstar_matrix(4, uu)
        # discrete correlation of columns after removing the quadrature weight
        z = b * np.sqrt(ww)[:, None]
        g = z.T @ z
        d = np.sqrt(np.diag(g))
        corr = g / np.outer(d, d)
        off = np.abs(corr - np.eye(4))
        return float(off.max())

    corrs = {n: max_corr(n) for n in (19, 31, 101)}
    ugl, wgl = gl01(4)
    bgl = pstar_matrix(4, ugl) * np.sqrt(wgl)[:, None]
    ggl = bgl.T @ bgl
    # orthonormal columns would be I; raw P* Gram is diagonal with 1/(2k+1)
    gram = (pstar_matrix(4, ugl) * wgl[:, None]).T @ pstar_matrix(4, ugl)
    target = np.diag(1.0 / (2 * np.arange(4) + 1))
    gl_err = float(np.max(np.abs(gram - target)))

    # Shift teaching example: two groups, equal weight.
    # Period A: both groups N(0,1) in quantile = z.
    # Period B: group 0 shifts by +1, group 1 by -0.4, and the mixture weights change.
    # Computed on a GL grid of the conditional quantiles, Gaussian via a rational approximation is unnecessary:
    # use Q = c0 + c1 P*_1, a location-scale stand-in that is closed form.
    # Group quantiles are linear in u so the mixture quantile is found by sorting a fine grid of samples.
    def gauss_mix_quantile(weights, means, scales, u):
        # Inverse CDF of a two-point Gaussian mixture by bisection on the CDF.
        from math import erf

        def cdf(t, m, s):
            return 0.5 * (1.0 + erf((t - m) / (s * math.sqrt(2.0))))

        out = np.empty_like(u)
        for i, ui in enumerate(u):
            lo, hi = -12.0, 12.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                f = sum(wj * cdf(mid, mj, sj) for wj, mj, sj in zip(weights, means, scales))
                if f < ui:
                    lo = mid
                else:
                    hi = mid
            out[i] = 0.5 * (lo + hi)
        return out

    ufine, wfine = gl01(256)
    # P_A = (0.7, 0.3), P_B = (0.3, 0.7). Same conditionals would be pure composition.
    # Add a structure change on group 0 only: mean 0 -> 0.8.
    qa = gauss_mix_quantile([0.7, 0.3], [0.0, 0.0], [1.0, 1.0], ufine)
    qb = gauss_mix_quantile([0.3, 0.7], [0.8, 0.0], [1.0, 1.0], ufine)
    # Counterfactual: model A (means 0,0) with P_B weights.
    qc = gauss_mix_quantile([0.3, 0.7], [0.0, 0.0], [1.0, 1.0], ufine)
    # Other order: model B with P_A.
    qc2 = gauss_mix_quantile([0.7, 0.3], [0.8, 0.0], [1.0, 1.0], ufine)
    total = float(np.sum(wfine * (qb - qa) ** 2))
    structure = float(np.sum(wfine * (qb - qc) ** 2))
    composition = float(np.sum(wfine * (qc - qa) ** 2))
    cross = float(2 * np.sum(wfine * (qb - qc) * (qc - qa)))
    # structure-first composition share
    structure_first_comp = total - float(np.sum(wfine * (qc2 - qa) ** 2))

    # Hosking vs plug-in, exponential rate 1, lambda_2 = 0.5.
    reps = 4000
    bias = {}
    for m in (5, 20):
        samples = rng.exponential(size=(reps, m))
        hos = sample_lmoments(samples, 4)
        plug = plug_in_lmoments(samples, 4)
        bias[m] = {
            "hosking": (hos.mean(0) - np.array([1.0, 0.5, 1.0 / 6.0, 1.0 / 12.0])).tolist(),
            "plugin": (plug.mean(0) - np.array([1.0, 0.5, 1.0 / 6.0, 1.0 / 12.0])).tolist(),
        }

    # DRO teaching instance: one feature, 4 bins, pinball at u=0.8.
    # Shape values and a few rows. Exact dual vs max-jump bound.
    u_level = 0.8
    rho_bar = max(u_level, 1.0 - u_level)
    f = np.array([0.0, 0.4, 0.1, 0.9])
    bins = np.array([0, 1, 2, 3, 1, 0, 3, 2])
    y = np.array([0.2, 1.1, 0.0, 1.4, -0.3, 0.5, 2.0, 0.4])
    kappa = 1.0
    n = len(y)

    def loss(c, yi):
        r = yi - f[c]
        return max(u_level * r, (u_level - 1) * r)

    def dual(eps):
        # breakpoints of the convex piecewise-linear dual
        cands = [0.0]
        for i in range(n):
            for c in range(4):
                for c2 in range(4):
                    dd = kappa * (abs(c - bins[i]) - abs(c2 - bins[i]))
                    if dd > 1e-15:
                        lv = (loss(c, y[i]) - loss(c2, y[i])) / dd
                        if lv >= 0:
                            cands.append(lv)
        best = min(
            lam * eps
            + np.mean(
                [
                    max(loss(c, y[i]) - lam * kappa * abs(c - bins[i]) for c in range(4))
                    for i in range(n)
                ]
            )
            for lam in cands
        )
        return best

    emp = float(np.mean([loss(int(b), float(yi)) for b, yi in zip(bins, y)]))
    jumps = np.abs(np.diff(f))
    g_max = float(jumps.max())
    eps = 0.25
    exact = dual(eps)
    bound = emp + eps * rho_bar * g_max / kappa
    ratio = (exact - emp) / (bound - emp)

    return {
        "w2": w2,
        "via_c": via_c,
        "via_l": via_l,
        "via_a": via_a,
        "wrong": wrong,
        "channel": channel,
        "kept": kept,
        "resid": resid,
        "phi": phi,
        "phi_names": names,
        "a_players": a_players,
        "mono": mono,
        "t3": t3,
        "t4": t4,
        "corrs": corrs,
        "gl_err": gl_err,
        "total": total,
        "structure": structure,
        "composition": composition,
        "cross": cross,
        "structure_first_comp": structure_first_comp,
        "bias": bias,
        "dro_emp": emp,
        "dro_exact": exact,
        "dro_bound": bound,
        "dro_ratio": ratio,
        "ggl_off": gl_err,
    }


def heatmap_svg(mono, t3, t4):
    h, w = mono.shape
    width, height = 640, 280
    left, top = 48, 24
    pw, ph = 520, 200
    cells = []
    for i in range(h):
        for j in range(w):
            x = left + j * pw / w
            y = top + (h - 1 - i) * ph / h
            fill = "#2f7d4a" if mono[i, j] else "#b85c38"
            cells.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{pw / w + 0.4:.2f}" height="{ph / h + 0.4:.2f}" fill="{fill}"/>'
            )

    # |tau3|=1/5 guides, and the tau4 band at tau3=0 is a horizontal annotation.
    def x_of(v):
        return left + (v - t3[0]) / (t3[-1] - t3[0]) * pw

    def y_of(v):
        return top + (t4[-1] - v) / (t4[-1] - t4[0]) * ph

    guides = []
    for v in (-0.2, 0.2):
        guides.append(
            f'<line x1="{x_of(v):.1f}" y1="{top}" x2="{x_of(v):.1f}" y2="{top + ph}" stroke="#1d1d1f" stroke-dasharray="3 3" stroke-width="1"/>'
        )
    for v in (-1 / 14, 2 / 7):
        guides.append(
            f'<line x1="{left}" y1="{y_of(v):.1f}" x2="{left + pw}" y2="{y_of(v):.1f}" stroke="#1d1d1f" stroke-dasharray="2 3" stroke-width="1"/>'
        )
    return (
        f'<svg role="img" aria-label="Region of monotone four-channel quantile polynomials" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
        '<rect width="100%" height="100%" fill="#f7f5f2"/>'
        + "".join(cells)
        + "".join(guides)
        + f'<text x="{left}" y="{height - 18}" font-size="12" font-family="system-ui,sans-serif" fill="#1d1d1f">τ₃</text>'
        + f'<text x="16" y="{top + 14}" font-size="12" font-family="system-ui,sans-serif" fill="#1d1d1f">τ₄</text>'
        + f'<text x="{left + pw + 8}" y="{top + 14}" font-size="11" font-family="system-ui,sans-serif" fill="#2f7d4a">monotone</text>'
        + f'<text x="{left + pw + 8}" y="{top + 30}" font-size="11" font-family="system-ui,sans-serif" fill="#b85c38">not</text>'
        + "</svg>"
    )


def corr_svg(corrs, gl_err):
    labels = [f"mid {n}" for n in (19, 31, 101)] + ["GL n=4"]
    vals = [corrs[19], corrs[31], corrs[101], gl_err]
    # log-ish display: plot the values themselves; GL is ~0 so give it a visible minimum stroke via a note.
    width, height = 640, 200
    left, top, bot = 56, 28, 36
    topv = max(vals) * 1.25
    pw = width - left - 20
    ph = height - top - bot
    bw = pw / (len(vals) * 1.6)
    parts = [
        f'<svg role="img" aria-label="Off-diagonal correlation of the first four shifted Legendre polynomials" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
        '<rect width="100%" height="100%" fill="#f7f5f2"/>',
        '<text x="56" y="18" font-size="13" font-family="system-ui,sans-serif" fill="#1d1d1f">Largest |correlation| off the diagonal</text>',
    ]
    for i, (lab, val) in enumerate(zip(labels, vals)):
        h = 0 if topv == 0 else (val / topv) * ph
        x = left + (i + 0.4) * (pw / len(vals))
        y = top + ph - h
        color = "#1b6ca8" if i < 3 else "#2f7d4a"
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{max(h, 1):.1f}" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{x + bw / 2:.1f}" y="{height - 16}" text-anchor="middle" font-size="11" font-family="system-ui,sans-serif" fill="#1d1d1f">{lab}</text>'
        )
        parts.append(
            f'<text x="{x + bw / 2:.1f}" y="{y - 4:.1f}" text-anchor="middle" font-size="11" font-family="system-ui,sans-serif" fill="#1d1d1f">{val:.1e}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def page(fig):
    ch = fig["channel"]
    channel_svg = svg_bars(
        [f"k={i}" for i in range(6)] + ["sum"],
        [("1/(2k+1) · Δc²", list(ch) + [float(ch.sum())])],
        ["#1b6ca8"],
        "energy",
        "W2² split of one polynomial pair, seed 0",
    )
    # two stacks: correct vs the refuted weight, normalised to show the bookkeeping
    book_svg = svg_bars(
        ["true W2²", "with weight 1/(2k+1)", "with weight (2k+1) on c"],
        [("value", [fig["w2"], fig["via_c"], fig["wrong"]])],
        ["#1b6ca8"],
        "value",
        "Same pair: the weight has to match the coefficient",
    )
    phi = fig["phi"]
    # split positive and negative for a diverging bar by shifting
    shap_svg = svg_bars(
        fig["phi_names"],
        [
            ("positive share", [max(v, 0.0) for v in phi]),
            ("negative share", [max(-v, 0.0) for v in phi]),
        ],
        ["#1b6ca8", "#b85c38"],
        "|share|",
        "Shapley shares of one channel (x2 pushes the other way)",
    )
    shift_svg = svg_bars(
        ["total W2²", "structure", "composition", "cross term"],
        [("piece", [fig["total"], fig["structure"], fig["composition"], max(fig["cross"], 0.0)])],
        ["#1b6ca8"],
        "W2²",
        "Teaching mixture: structure, composition, and the cross term",
    )
    b5 = fig["bias"][5]["plugin"]
    b20 = fig["bias"][20]["plugin"]
    h5 = fig["bias"][5]["hosking"]
    bias_svg = svg_bars(
        ["λ2 plug-in m=5", "λ2 plug-in m=20", "λ2 Hosking m=5", "λ3 plug-in m=5"],
        [("absolute bias", [abs(b5[1]), abs(b20[1]), abs(h5[1]), abs(b5[2])])],
        ["#b85c38"],
        "|bias|",
        "Exponential rows: plug-in L-moments stay biased, Hosking does not",
    )
    dro_svg = svg_bars(
        ["empirical pinball", "exact worst case", "max-jump upper bound"],
        [("risk", [fig["dro_emp"], fig["dro_exact"], fig["dro_bound"]])],
        ["#1b6ca8"],
        "pinball",
        "One feature, eight rows: the jump penalty sits above the exact risk",
    )
    heat = heatmap_svg(fig["mono"], fig["t3"], fig["t4"])
    corr = corr_svg(fig["corrs"], fig["gl_err"])
    rel = abs(fig["w2"] - fig["via_c"]) / fig["w2"]
    phi_sum = float(np.sum(phi))
    energy = float((fig["a_players"].sum() ** 2) / 3.0)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TRACE theory · traceot</title>
<meta name="description" content="The mathematics behind TRACE v1: L-moments, attribution, shifts, distribution-valued rows, and Wasserstein-robust fits.">
<script>
window.MathJax = {{
  tex: {{ inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']] }},
  options: {{ skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
<style>
  :root {{ color-scheme: light dark; --fg:#1d1d1f; --bg:#fdfdfc; --muted:#5b5b60; --rule:#ddd; --accent:#1b6ca8; --note:#f3f6fa; --break:#f8ece6; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --fg:#e9e9ec; --bg:#141416; --muted:#a3a3aa; --rule:#333; --accent:#6fb3e6; --note:#1d2229; --break:#2a211c; }}
    svg {{ background:#fff; border-radius:6px; }}
  }}
  html {{ -webkit-text-size-adjust:100%; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
         font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif;
         font-size:1.0625rem; line-height:1.6; }}
  main {{ max-width:42rem; margin:0 auto; padding:1.25rem 1rem 4rem; }}
  h1 {{ font-size:1.7rem; line-height:1.2; margin:1.2rem 0 .4rem; }}
  h2 {{ font-size:1.25rem; margin:1.6rem 0 .5rem; }}
  h3 {{ font-size:1.05rem; margin:1.3rem 0 .3rem; }}
  a {{ color:var(--accent); }}
  .lede {{ color:var(--muted); }}
  .status {{ display:inline-block; font-size:.8rem; font-weight:600; padding:.15rem .5rem; border-radius:999px; background:#fff3cd; color:#664d03; }}
  .unv {{ display:inline-block; font-size:.75rem; font-weight:700; letter-spacing:.02em; padding:.05rem .4rem; border-radius:4px; background:#f8d7da; color:#842029; }}
  .note {{ background:var(--note); border-left:4px solid var(--accent); padding:.6rem .9rem; border-radius:4px; margin:1rem 0; }}
  .breaks {{ background:var(--break); border-left:4px solid #b85c38; padding:.6rem .9rem; border-radius:4px; margin:1rem 0; }}
  figure {{ margin:1rem 0; }}
  figure svg {{ width:100%; height:auto; display:block; }}
  figcaption {{ color:var(--muted); font-size:.9rem; margin-top:.35rem; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:.9em; }}
  .tabs {{ display:flex; flex-wrap:wrap; gap:.4rem; margin:1rem 0 .2rem; }}
  .tabs label {{ border:1px solid var(--rule); border-radius:999px; padding:.25rem .7rem; cursor:pointer; font-size:.92rem; }}
  .panel {{ display:none; }}
  input.tab {{ position:absolute; opacity:0; pointer-events:none; }}
  #t1:checked ~ .tabs label[for="t1"],
  #t2:checked ~ .tabs label[for="t2"],
  #t3:checked ~ .tabs label[for="t3"],
  #t4:checked ~ .tabs label[for="t4"],
  #t5:checked ~ .tabs label[for="t5"],
  #t6:checked ~ .tabs label[for="t6"],
  #t7:checked ~ .tabs label[for="t7"] {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
  #t1:checked ~ .p1, #t2:checked ~ .p2, #t3:checked ~ .p3, #t4:checked ~ .p4,
  #t5:checked ~ .p5, #t6:checked ~ .p6, #t7:checked ~ .p7 {{ display:block; }}
  label.slider {{ display:block; margin:.4rem 0; font-size:.95rem; }}
  input[type="range"] {{ width:100%; }}
  #qplot {{ width:100%; height:auto; background:#f7f5f2; border-radius:6px; }}
  footer {{ color:var(--muted); font-size:.85rem; margin-top:2rem; border-top:1px solid var(--rule); padding-top:1rem; }}
  mjx-container {{ overflow-x:auto; overflow-y:hidden; max-width:100%; }}
</style>
</head>
<body>
<main>
<p><a href="../index.html">traceot</a> · <a href="trace.html">TRACE</a> · <span class="status">research, unvalidated</span></p>
<h1>The theory behind TRACE</h1>
<p class="lede">Seven results, each with the plain idea, the statement, a sketch of the proof, the numerical check, and where it stops being true. The proofs live in <a href="https://github.com/jgridifier/traceot/blob/main/docs/theory/THEORY.md">THEORY.md</a>. The checks are <code>docs/theory/verify/</code>, archived in <code>RESULTS.txt</code> (46 pass, 0 fail, 11 October 2026). Figures on this page were recomputed by <code>docs/build_trace_theory.py</code>.</p>
<p>A citation marked <span class="unv">UNVERIFIED</span> is a detail the notes could not confirm against a primary source. Those flags stay visible on purpose.</p>

<input class="tab" type="radio" name="tab" id="t1" checked>
<input class="tab" type="radio" name="tab" id="t2">
<input class="tab" type="radio" name="tab" id="t3">
<input class="tab" type="radio" name="tab" id="t4">
<input class="tab" type="radio" name="tab" id="t5">
<input class="tab" type="radio" name="tab" id="t6">
<input class="tab" type="radio" name="tab" id="t7">
<div class="tabs" role="tablist">
  <label for="t1">L-moments</label>
  <label for="t2">Attribution</label>
  <label for="t3">Monotonicity</label>
  <label for="t4">Why tails mix</label>
  <label for="t5">Explaining a shift</label>
  <label for="t6">Distribution rows</label>
  <label for="t7">Robust fit</label>
</div>

<section class="panel p1">
<h2>L-moment and Legendre channels</h2>
<p>The shifted Legendre polynomials are perpendicular axes for quantile functions. The squared Wasserstein-2 distance between two scalar laws is the squared length of the difference of their quantile functions. Pythagoras then splits that length into one term per axis. The axes are not unit length, so the weight on each term depends on which coefficients you wrote down.</p>
<h3>Statement</h3>
<p>Write \\(P^*_k(u) = P_k(2u-1)\\), so \\(P^*_k(1) = 1\\). Then \\(\\langle P^*_j, P^*_k\\rangle = \\delta_{{jk}}/(2k+1)\\). L-moments are \\(\\lambda_r = \\int_0^1 Q(u)\\, P^*_{{r-1}}(u)\\, du\\). Three ways to write the same function \\(Q = \\sum_k c_k P^*_k\\):</p>
<ul>
<li>expansion coefficients \\(c_k = (2k+1)\\,\\lambda_{{k+1}}\\);</li>
<li>L-moments \\(\\lambda_{{k+1}} = c_k/(2k+1)\\);</li>
<li>orthonormal coefficients \\(a_k = c_k/\\sqrt{{2k+1}}\\), on the unit axes \\(e_k = \\sqrt{{2k+1}}\\, P^*_k\\).</li>
</ul>
<p><strong>Theorem 1.1.</strong> For two laws with finite second moment,</p>
\\[ W_2^2(F_A, F_B) = \\sum_{{k\\ge 0}} \\frac{{\\Delta c_k^2}}{{2k+1}} = \\sum_{{k\\ge 0}} (2k+1)\\,\\Delta\\lambda_{{k+1}}^2 = \\sum_{{k\\ge 0}} \\Delta a_k^2. \\]
<p>Keeping only the first \\(K\\) channels drops a residual \\(\\|R_K\\|^2 = \\sum_{{k\\ge K}} (2k+1)\\Delta\\lambda_{{k+1}}^2\\). That residual is nonnegative, shrinks as \\(K\\) grows, and goes to zero. There is no universal speed.</p>
<h3>Proof sketch</h3>
<p>\\(\\{{e_k\\}}\\) is an orthonormal basis of \\(L^2(0,1)\\): the inner-product identity is the orthogonality above, and polynomials are dense. Parseval says the squared length is the sum of squared coordinates on those unit axes, which is \\(\\sum \\Delta a_k^2\\). Substitute \\(a_k = \\sqrt{{2k+1}}\\,\\lambda_{{k+1}}\\) and \\(\\lambda_{{k+1}} = c_k/(2k+1)\\). The truncation identity is Pythagoras for the split into the first \\(K\\) axes and the rest.</p>
<h3>Check</h3>
<p>On this page's seed-0 polynomial pair, the quadrature \\(W_2^2\\) is {fig["w2"]:.6f}. The three formulas give {fig["via_c"]:.6f}, {fig["via_l"]:.6f} and {fig["via_a"]:.6f} (relative gap {rel:.1e}). Putting the L-moment weight \\((2k+1)\\) on the expansion coefficients \\(c_k\\) instead gives {fig["wrong"]:.4f}, which is the wrong number. The archived check <code>v1:1.4</code> reports relative error \\(1.8\\times 10^{{-16}}\\) on its pair (\\(W_2^2 = 1.238758\\)); <code>v1:1.5</code> reproduces the false formula as 1.5103 against 1.2388. <code>v1:1.6</code>: normal against exponential, residual 0.194, 0.0424, 0.0185, 0.0062 at \\(K = 1, 3, 5, 10\\). On the pair drawn here, the first three channels hold {fig["kept"]:.4f} and the tail holds {fig["resid"]:.4f}.</p>
<figure>{channel_svg}<figcaption>Each bar is one channel's share \\(\\Delta c_k^2/(2k+1)\\). They add to the squared distance. Seed 0, degree 5.</figcaption></figure>
<figure>{book_svg}<figcaption>Left two bars match. The right bar uses \\((2k+1)\\) on \\(c_k\\), which is the L-moment weight on the wrong coefficient.</figcaption></figure>
<div class="breaks"><strong>Where this breaks.</strong> No rate without smoothness. Quantile functions of unbounded laws blow up at 0 or 1, so the tail can still be large at \\(K = 3\\) or \\(4\\). The residual is a population quantity; sample L-moments of high order are noisy. The identity is for a scalar outcome. It is exact for the curves before the isotonic projection, not after.</div>
</section>

<section class="panel p2">
<h2>Per-feature Shapley attribution within a channel</h2>
<p>Inside one channel, the change at a row is a sum of feature contributions, including the intercept. The energy is the square of that sum, so it contains products of two features. Shapley splits each product equally. The shares add up exactly. A feature that pushes against the total change gets a negative share, and that negative share is real.</p>
<h3>Statement</h3>
<p><strong>Theorem 1.3.</strong> For the game \\(v(S) = w\\,(\\sum_{{j\\in S}} a_j)^2\\) with \\(w&gt;0\\) fixed, the Shapley value is \\(\\varphi_j = w\\, a_j \\sum_i a_i\\). In particular \\(\\sum_j \\varphi_j = v(N)\\). In expansion units, \\(w = 1/(2k+1)\\), so \\(\\varphi_{{jk}} = \\Delta f_{{jk}}\\,(\\sum_i \\Delta f_{{ik}})/(2k+1)\\). In orthonormal units the weight is 1 and \\(\\varphi_{{jk}} = a_{{jk}}\\sum_i a_{{ik}}\\).</p>
<p>If the features are independent and every non-intercept shape is centred under the evaluation law, the cross terms vanish and each feature's expected share is \\(w\\) times its variance. That is the functional-ANOVA main effect (Lengerich, Tan, Chang, Hooker and Caruana, AISTATS 2020, PMLR 108:2402–2412).</p>
<h3>Proof sketch</h3>
<p>Expand the square into squares and pairwise products. A square \\(a_i^2\\) belongs entirely to player \\(i\\). A product \\(2 a_i a_l\\) is a unanimity game on the pair \\(\\{{i,l\\}}\\), so each of those two players gets \\(a_i a_l\\). Collecting terms gives \\(\\varphi_j = w a_j \\sum_l a_l\\). Summing over \\(j\\) recovers \\(w(\\sum a)^2\\).</p>
<h3>Check</h3>
<p>Players here: {", ".join(f"{n}={a:.2f}" for n, a in zip(fig["phi_names"], fig["a_players"]))}, weight \\(1/3\\). Shares {", ".join(f"{n}={v:.4f}" for n, v in zip(fig["phi_names"], phi))}. They sum to {phi_sum:.6f}, and the channel energy is {energy:.6f}. Archived <code>v1:1.7</code>: 50 random games, brute force over every coalition, tolerance \\(10^{{-12}}\\), and negative values occur. <code>v1:1.8</code>: under independence and centring the covariance term is zero. <code>v1:1.8b</code>: with dependence the covariance is nonzero (the check found \\(-0.0168\\)) and the shares still add up.</p>
<figure>{shap_svg}<figcaption>x2's contribution has the opposite sign from the total, so its Shapley share is negative. The absolute values are stacked only so the negative piece is visible; the signed shares are the ones that add to the energy.</figcaption></figure>
<div class="breaks"><strong>Where this breaks.</strong> Dependence leaves covariance terms, split equally by convention, not by a causal claim. The intercept is a player: recentring moves mass between the intercept and the features, so the evaluation law has to be stated. Do not clip negative shares. The identity is exact only before the isotonic projection. The monotonisation adjustment is reported as its own line and is not spread back across features.</div>
</section>

<section class="panel p3">
<h2>Monotonicity</h2>
<p>A quantile function has to increase. A short polynomial does not, unless its coefficients satisfy an inequality. With three channels that inequality is a bound on the skewness. The exponential law is too skewed for three channels. Fixing the curve afterwards with the pool-adjacent-violators map makes it monotone, and also stops it being a sum of per-feature pieces. That gap is reported on its own line, with a bound.</p>
<h3>Statement</h3>
<p><strong>Proposition 1.5.</strong> \\(Q = c_0 + c_1 P^*_1 + c_2 P^*_2\\) is nondecreasing on \\([0,1]\\) if and only if \\(|c_2| \\le c_1/3\\). When \\(\\lambda_2 &gt; 0\\) this is \\(|\\tau_3| \\le 1/5\\), with \\(\\tau_3 = \\lambda_3/\\lambda_2\\).</p>
<p><strong>Proposition 1.6.</strong> With a fourth channel, check \\(Q'\\) at the endpoints, and at the interior vertex if the second derivative says the parabola opens upwards and the vertex sits in \\((0,1)\\). At \\(\\tau_3 = 0\\) the kurtosis band is \\(-1/14 \\le \\tau_4 \\le 2/7\\).</p>
<p>For a general degree, nonnegativity of \\(Q'\\) is a sums-of-squares condition (Markov–Lukács; Powers and Reznick, Trans. AMS, 2000). A cheaper sufficient test: write \\(Q'\\) in the Bernstein basis and ask for nonnegative coefficients. That test can refuse a curve that is actually monotone.</p>
<p><strong>Proposition 1.8.</strong> Let \\(\\Pi\\) be the \\(L^2\\) projection onto nondecreasing functions, and \\(d_i = \\|\\tilde Q_i - \\Pi\\tilde Q_i\\|\\). Then \\(\\|\\Pi\\tilde Q_1 - \\Pi\\tilde Q_2\\| \\le \\|\\tilde Q_1 - \\tilde Q_2\\|\\), and \\(\\|\\Pi\\tilde Q_1 - \\Pi\\tilde Q_2\\| \\ge \\|\\tilde Q_1 - \\tilde Q_2\\| - d_1 - d_2\\). The squared-distance adjustment after projection lies in \\([-2D(d_A+d_B),\\, 0]\\).</p>
<h3>Proof sketch</h3>
<p>For three channels, \\(Q'\\) is a straight line, so it stays nonnegative exactly when both endpoints do. The ratio \\(\\tau_3 = 3c_2/(5c_1)\\) turns \\(|c_2|\\le c_1/3\\) into \\(|\\tau_3|\\le 1/5\\). For four channels, \\(Q'\\) is a parabola; the minimum on the interval is at an endpoint or at the vertex when the parabola is convex. The projection bounds are the usual Hilbert-space facts for a closed convex cone, plus the triangle inequality. The adjustment bound is what you get by expanding \\(D^2 - (D - d_A - d_B)_+^2\\).</p>
<h3>Check</h3>
<p><code>v1:1.9</code> and <code>v1:1.10</code>: 20000/20000 random coefficient vectors agree with a dense grid. <code>v1:1.11b</code>: the exponential law has \\(\\tau_3 = 1/3\\). Its 3-term truncation has \\(\\min Q' = -2\\). Its 4-term truncation has \\(\\min Q' = 0.536\\). <code>v1:1.11</code>: Bernstein certificates had 0 false passes; of 817 monotone draws, 255, 618 and 759 were certified at degree elevation 0, 4 and 16. <code>v1:1.12</code>: 300 random pairs, the projection bounds hold.</p>
<figure>{heat}<figcaption>Green: the four-channel polynomial with \\(c_1 = 1\\) is nondecreasing. Vertical dashes are \\(|\\tau_3| = 1/5\\), the three-channel limit (the picture is the four-channel region, which is wider). Horizontal dashes are \\(\\tau_4 = -1/14\\) and \\(2/7\\), the band at \\(\\tau_3 = 0\\). Recomputed on a \\(81\\times 61\\) grid.</figcaption></figure>
<figure>
<svg id="qplot" viewBox="0 0 640 240" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Quantile curve for the chosen skewness and kurtosis"></svg>
<figcaption>Drag \\(\\tau_3\\) and \\(\\tau_4\\). The curve is \\(c_1 P^*_1 + c_2 P^*_2 + c_3 P^*_3\\) with \\(c_1 = 1\\). The status line says whether \\(Q'\\) stays nonnegative. This is the same derivative as Proposition 1.6, evaluated in the page.</figcaption>
</figure>
<label class="slider">τ₃ <output id="t3o"></output><input id="t3" type="range" min="-0.5" max="0.5" step="0.005" value="0"></label>
<label class="slider">τ₄ <output id="t4o"></output><input id="t4" type="range" min="-0.25" max="0.45" step="0.005" value="0"></label>
<p id="monoStatus"></p>
<div class="breaks"><strong>Where this breaks.</strong> Three channels cannot represent a monotone exponential quantile. The Bernstein test is one-sided: a monotone curve can fail it. The sums-of-squares test is exact and needs a semidefinite solver, which the core does not carry. After PAV, \\(\\Pi(\\sum_j f_j) \\ne \\sum_j \\Pi(f_j)\\). Report the pre-projection attribution and one monotonisation line. Do not split that line across features.</div>
</section>

<section class="panel p4">
<h2>Why orthogonality alone does not fix the tail feature</h2>
<p>A feature that only moves the tails should show up only in the tail channels, if the channels are perpendicular and the fit measures distance the same way. TRACE v0's channels \\(1\\), \\(z\\) and \\(\\mathrm{{He}}_3(z)\\) already are perpendicular in \\(L^2(0,1)\\), with \\(\\|\\mathrm{{He}}_m\\|^2 = m!\\). What mixed them was a coarse grid, the pinball loss, early stopping, and a generator that does not point along the reporting axes.</p>
<h3>Statement</h3>
<p><strong>Finding 1.</strong> On the continuous interval, \\(\\langle z, \\mathrm{{He}}_3(z)\\rangle\\) is zero (<code>v1:1.13</code>, about \\(-6\\times 10^{{-15}}\\)). On the 19-point midpoint grid the same correlation is \\(-0.455\\). Gauss–Legendre nodes with at least \\(K\\) nodes make the discrete Gram matrix of \\(P^*_0,\\ldots,P^*_{{K-1}}\\) exactly diagonal, because that rule integrates polynomials up to degree \\(2n-1\\) exactly. No polynomial rule is exact for \\(z\\), which is unbounded.</p>
<p>Integrated pinball loss has curvature \\(H_{{jk}} = \\int e_j(u) e_k(u)\\, f(Q(u))\\, du\\). That weights the level by the density at the quantile, so it is not the Wasserstein inner product. For a Gaussian outcome the v0 channels have correlation \\(-\\sqrt{{0.6}}\\) under that curvature (<code>v1:1.14</code>). A gradient step along that metric leaks a pure scale effect into the shape channel. The leak peaks near 0.408 for a scale of 1.2, and a Newton step sends it to zero (<code>v1:1.15</code>).</p>
<p><strong>Finding 2.</strong> In the plain Legendre basis the simulated scale feature \\((0.4+1.2 x)\\, z\\) itself has L-kurtosis. The population energy ratio of the planted tail feature to that scale feature, in channel \\(k = 3\\), is <strong>0.753</strong> (<code>v1:1.16</code>). No noise-free fit in that basis can pass a 3× rule. A reference-adapted basis with \\(e_1 = z\\) isolates the scale feature (load about \\(10^{{-16}}\\), <code>v1:1.16b</code>). A hypothesis that a mean-standardised conditional reference would reach 3× was tested and is false: the ratio was 2.00 (<code>v1:1.16d</code>).</p>
<p><strong>Proposition 1.9.</strong> With an orthonormal basis and the population \\(L^2\\) projection, feature \\(j\\) has zero energy in channel \\(k\\) exactly when its true effect is perpendicular to that channel. Identification is a property of the pair (truth, basis).</p>
<h3>Proof sketch</h3>
<p>Coefficients of an orthogonal projection are inner products. If the true effect of a feature lies in a subset of the axes, every other coefficient is zero. The pinball curvature is a different inner product, \\(f(Q(u))\\,du\\) instead of \\(du\\), so its eigen-axes are not the Legendre axes. Early-stopped gradient descent walks the gradient of that other metric and therefore mixes coordinates that Newton, which multiplies by the inverse curvature, separates in one step.</p>
<h3>Check</h3>
<p>Recomputed here: largest off-diagonal correlation of \\(P^*_0,\\ldots,P^*_3\\) on midpoint grids is {fig["corrs"][19]:.3e} (19), {fig["corrs"][31]:.3e} (31), {fig["corrs"][101]:.3e} (101). On 4 Gauss–Legendre nodes the Gram error against \\(\\mathrm{{diag}}\\,1/(2k+1)\\) is {fig["gl_err"]:.1e}. Archived <code>v1:1.13</code> reports midpoint correlations \\(-0.4546\\), \\(-0.2942\\), \\(-0.1008\\) for the v0 pair \\((z,\\mathrm{{He}}_3)\\), and a Gauss–Legendre correlation of \\(7.8\\times 10^{{-16}}\\). <code>v1:1.17</code>: fitting 8 channels and reporting 3 cuts a lognormal coefficient bias from 0.284 to 0.057.</p>
<figure>{corr}<figcaption>Midpoint grids leave a visible correlation. Four Gauss–Legendre nodes integrate these polynomials exactly, so the off-diagonal is numerical noise.</figcaption></figure>
<div class="breaks"><strong>Where this breaks.</strong> Switching the basis to Legendre does not make the old tail test pass: the population ratio is 0.753. The reporting basis has to contain the generator's directions, and a pass rule has to be written as an oracle projection in that basis. The estimator has to be the Wasserstein projection, or close to it: pinball plus early stopping is not. Quadrature has to be exact for the basis you fit. A weak signal still loses to noise.</div>
</section>

<section class="panel p5">
<h2>Explaining a shift between two periods</h2>
<p>For a fixed covariate profile, compare the two periods' predicted distributions and average over a stated population of profiles. That conditional shift splits exactly by channel and by feature. The overall, marginal shift is a mixture, and mixing can cancel. Part of a change is the covariates moving (composition) and part is the relationship moving (structure). The squared distance has a cross term, and the order of the two switches matters. Squared distances from noisy fits are biased upward; a product of two independent halves is not, when each half is unbiased.</p>
<h3>Statement</h3>
<p><strong>Theorem 2.1.</strong> For a stated covariate law \\(P\\) (default: period B), \\(D_{{\\mathrm{{cond}}}}(P) = \\mathbb{{E}}_{{x\\sim P}} W_2^2(Q_A(\\cdot|x), Q_B(\\cdot|x)) = \\sum_{{k,j}} \\mathbb{{E}}_P \\varphi_{{jk}}(x)\\).</p>
<p><strong>Proposition 2.2.</strong> With the same covariates, the marginal \\(W_2^2\\) is at most the average conditional \\(W_2^2\\). It can be much smaller. Two groups that swap places have marginal shift 0 and conditional shift 4.</p>
<p><strong>Proposition 2.3.</strong> The counterfactual "period A's model, period B's covariates" splits the quantile change into structure plus composition in every L-moment. The squared distance is those two squared lengths plus twice their inner product. The two-player Shapley split needs the other counterfactual as well, and it is efficient. Composition-first and structure-first need not agree.</p>
<p><strong>Proposition 2.4.</strong> If \\(\\hat\\Delta = \\Delta + \\varepsilon\\) with mean-zero noise, \\(\\mathbb{{E}}\\|\\hat\\Delta\\|^2 = \\|\\Delta\\|^2 + \\mathrm{{tr}}\\,\\Sigma\\). Two independent copies satisfy \\(\\mathbb{{E}}\\langle\\hat\\Delta_1,\\hat\\Delta_2\\rangle = \\|\\Delta\\|^2\\). Report the product unclipped: it can be negative.</p>
<p><strong>Proposition 2.5.</strong> If the rows are exchangeable across the period label, a permutation p-value \\((1 + \\#\\{{T_\\pi \\ge T_{{\\mathrm{{obs}}}}\\}})/(1+R)\\) has level \\(\\alpha\\). Permuting inside covariate cells tests "no conditional shift" while allowing the covariates to differ. The bootstrap is for the size of a shift that is clearly not zero. It is a poor null floor for a squared statistic near zero.</p>
<h3>Proof sketch</h3>
<p>Conditional exactness is Theorem 1.1 and Theorem 1.3, then linearity of expectation. The marginal inequality is a coupling: draw a profile from \\(P\\) and a uniform level, and read both quantile functions. That coupling costs the conditional distance, and \\(W_2\\) is the cheapest coupling. The composition split is a telescoping sum. The cross term is the expansion of a square. Cross-fitting kills the noise-by-noise product because the two halves are independent and mean zero. The permutation level is the usual randomisation argument: under exchangeability the observed statistic is uniform among the permutations.</p>
<h3>Check</h3>
<p>The mixture drawn for this page (weights \\((0.7,0.3)\\) against \\((0.3,0.7)\\), and a mean shift of 0.8 on the first group) has total {fig["total"]:.4f} = structure {fig["structure"]:.4f} + composition {fig["composition"]:.4f} + cross {fig["cross"]:.4f}. Structure-first composition is {fig["structure_first_comp"]:.4f}, which is not the composition-first number. Archived <code>v2:2.1</code>: absolute gap \\(4.4\\times 10^{{-16}}\\). <code>v2:2.2</code>: marginal 1.099 ≤ conditional 3.583. <code>v2:2.3</code>: total 1.599 = 1.099 + 0.049 + 0.451, and the Shapley composition share is \\(-0.015\\). Path dependence: composition-first 0.049, structure-first \\(-0.079\\). <code>v2:2.4</code>: plug-in bias 0.1238 against theory 0.1250; cross-fit 0.2839 ± 0.0022 against truth 0.2843. <code>v2:2.5</code>: null rejection rate 0.103 at \\(\\alpha = 0.10\\), inside \\([0.048, 0.152]\\). <code>v2:2.6</code>: bootstrap SD over Monte Carlo SD has median 0.71.</p>
<figure>{shift_svg}<figcaption>A two-group Gaussian mixture recomputed on 256 Gauss–Legendre nodes. The cross term is part of the total. It is not optional bookkeeping.</figcaption></figure>
<div class="breaks"><strong>Where this breaks.</strong> The conditional split depends on the evaluation law and on the centring. It is exact only before projection. Bins that period B uses and period A never saw are extrapolations. There is no closed-form feature split of the marginal distance; a feature Shapley there means a new mixture per coalition. The cross term can be large (28% of the archived total). Cross-fitting removes noise-by-noise, not the bias of a regularised fit. Serial dependence breaks row exchangeability; block permutation is then approximate. <span class="unv">UNVERIFIED</span> Chernozhukov, Fernández-Val and Melly, Econometrica 2013: the journal volume and pages were not confirmed in the notes. <span class="unv">UNVERIFIED</span> Lehmann and Romano: the chapter number for the randomisation test.</div>
</section>

<section class="panel p6">
<h2>Distribution-valued rows</h2>
<p>Sometimes a row is a whole sample, not one number. The fit that minimises the average squared Wasserstein distance, inside a Legendre model, is ordinary least squares done separately in each channel. The targets are the row's L-moments. Hosking's unbiased sample L-moments stay useful when each row's sample is small. Plugging the empirical quantile function into the same formula does not: it is biased in the shape channels, and the bias does not go away just by collecting more rows.</p>
<h3>Statement</h3>
<p><strong>Theorem 3.1.</strong> With targets \\(t_{{ik}} = (2k+1)\\lambda_{{k+1}}(F_i)\\),</p>
\\[ \\sum_i w_i W_2^2(F_i, Q_\\theta(\\cdot|x_i)) = \\sum_k \\frac{{1}}{{2k+1}} \\sum_i w_i (t_{{ik}} - D(x_i)\\theta_k)^2 + \\sum_i w_i \\|(I-\\Pi_K)Q_i\\|^2. \\]
<p>Without a constraint that couples channels, each channel is its own weighted least squares. The weight \\(1/(2k+1)\\) does not change those separate answers. It matters when monotonicity constraints tie the channels together.</p>
<p><strong>Proposition 3.2.</strong> Hosking's \\(\\ell_r\\) is unbiased for \\(\\lambda_r\\) when the row has at least \\(r\\) observations (Hosking, JRSS-B 52(1):105–124, 1990). The plug-in \\(\\int \\hat Q_m P^*_k\\) is biased at order \\(1/m\\) for \\(k \\ge 1\\).</p>
<p><strong>Proposition 3.3.</strong> With unbiased targets, fixed row size, and a nonsingular design, weighted least squares is consistent and \\(\\sqrt{{n}}\\)-normal for the best linear approximation of the conditional L-moment, as the number of rows grows.</p>
<p><strong>Proposition 3.5.</strong> If a channel has between-row variance \\(\\tau_k^2\\) and sampling variance about \\(v_k/m_i\\), the Gauss–Markov weights are \\(1/(\\tau_k^2 + v_k/m_i)\\). They can differ by channel.</p>
<h3>Proof sketch</h3>
<p>Apply the truncation identity to each row. The piece inside the first \\(K\\) channels is a weighted sum of squared coordinate errors, and the tail does not depend on the coefficients. Unbiasedness of \\(\\ell_r\\) is Hosking's probability-weighted-moment argument. Consistency is then ordinary least squares with mean-zero row noise. The weights are Aitken's theorem for heteroskedastic linear models.</p>
<h3>Check</h3>
<p>Recomputed here, 4000 exponential samples. Plug-in bias of \\(\\lambda_2\\) is {b5[1]:+.4f} at \\(m = 5\\) and {b20[1]:+.4f} at \\(m = 20\\). Hosking's bias of \\(\\lambda_2\\) at \\(m = 5\\) is {h5[1]:+.4f}, sampling noise around zero. Archived <code>v3:3.1</code>: the two objectives agree at 13.69944898, coefficients to \\(8.5\\times 10^{{-14}}\\). <code>v3:3.2</code>: plug-in bias of \\(\\lambda_2\\) is \\(-0.100\\) at \\(m = 5\\) and \\(-0.025\\) at \\(m = 20\\). <code>v3:3.3</code>: unbiased-target RMSE ratio from \\(n = 200\\) to \\(n = 3200\\) is 0.260 (\\(\\sqrt{{n}}\\) says 0.25); plug-in targets plateau at 0.152. <code>v3:3.5</code>: slope variance 3.97e-3 (inverse variance) against 8.26e-3 (equal) and 5.35e-3 (proportional to \\(m\\)).</p>
<figure>{bias_svg}<figcaption>Absolute bias against the exponential law's known L-moments. Hosking at \\(m = 5\\) is a flat bar at the left of the noise. The plug-in \\(\\lambda_3\\) bar is the shape-channel problem.</figcaption></figure>
<div class="breaks"><strong>Where this breaks.</strong> Plug-in targets need the per-row sample size to grow, not just the number of rows. The \\(1/m\\) variance formula is a leading term and depends on the row's own law. For a Gaussian row, \\(m\\,\\mathbb{{E}} W_2^2\\) of the empirical measure grows with \\(m\\) (2.61, 3.03, 3.33 at \\(m = 10, 100, 1000\\), <code>v3:3.4b</code>), so the \\(1/m\\) rate fails. <span class="unv">UNVERIFIED</span> Bobkov and Ledoux, Memoirs AMS 261 no. 1259 (2019): both the constant in \\(\\mathbb{{E}} W_2^2 \\le 2 J_2(F)/(m+1)\\) and the theorem number. The uniform-law check still holds numerically (<code>v3:3.4</code>). Projecting a fitted curve onto monotone functions cannot increase the distance to a monotone target (<code>v3:3.6</code>). Imposing monotonicity during the fit couples the channels, and then the \\(1/(2k+1)\\) weights matter.</div>
</section>

<section class="panel p7">
<h2>Wasserstein-robust fitting on the bin grid</h2>
<p>An adversary may move a fraction of the rows onto other bins, paying a transport cost, to make the pinball loss as large as possible. Because the model is constant on bins, the worst case is a linear programme, and the inner move separates across features. A penalty on the largest jump between neighbouring bins is the honest Lipschitz-style stand-in, and it is only an upper bound. Moving the outcome alone, under a type-1 Wasserstein ball, adds the same constant to every model, so it does not regularise anything.</p>
<h3>Statement</h3>
<p>Fix a level \\(u\\). The model is \\(q(b) = \\sum_j f_j(b_j)\\). The cost is \\(d = \\sum_j \\kappa_j |b_j - b'_j|\\), plus an optional outcome term. \\(\\bar\\rho = \\max(u, 1-u)\\).</p>
<p><strong>Theorem 4.1.</strong> With covariates only,</p>
\\[ R(\\varepsilon) = \\min_{{\\lambda\\ge 0}}\\left\\{{ \\lambda\\varepsilon + \\frac{{1}}{{n}}\\sum_i \\max_c\\big[\\ell(c, y_i) - \\lambda\\, d(c, b_i)\\big] \\right\\}}. \\]
<p>The inner maximum splits over features and over the two slopes of the pinball. The map \\(\\theta \\mapsto R(\\varepsilon;\\theta)\\) is convex.</p>
<p><strong>Proposition 4.3.</strong> If \\(G_j\\) is the largest adjacent jump of feature \\(j\\), then \\(R(\\varepsilon) \\le \\mathrm{{empirical}} + \\varepsilon\\,\\bar\\rho\\,\\max_j G_j/\\kappa_j\\). The ratio of the exact gain to this bound, in the archived check, runs from 0.22 to 1.00.</p>
<p><strong>Proposition 4.5.</strong> Outcome-only type-1 cost \\(\\kappa_y|\\Delta y|\\) gives \\(R(\\varepsilon) = \\mathrm{{empirical}} + \\varepsilon\\,\\bar\\rho/\\kappa_y\\), the same for every model that has a residual of the right sign.</p>
<p><strong>Proposition 4.8.</strong> A local (type-∞) ball, at most \\(r\\) bins and \\(|\\Delta y|\\le \\delta\\), has an exact window formula. Total variation of each shape function over that window is an upper bound, not the type-1 penalty.</p>
<p>If the deployment law is within \\(\\varepsilon\\) of the training law in the stated cost, its expected loss is at most \\(R(\\varepsilon)\\), deterministically. Turning that into a probability statement about the unknown population law needs a concentration theorem.</p>
<h3>Proof sketch</h3>
<p>The primal is a transport linear programme with first marginal fixed at the empirical measure. It is feasible and bounded, so strong duality applies. Pinball is the maximum of two linear pieces, and \\(y - q(c)\\) separates across features, so the inner supremum separates. The jump bound is Kantorovich–Rubinstein: the loss is Lipschitz on the grid with constant \\(\\bar\\rho \\max_j G_j/\\kappa_j\\). Outcome-only equality is the same Lipschitz constant, attained by pushing one residual in the expensive direction.</p>
<h3>Check</h3>
<p>The eight-row, one-feature example on this page, level 0.8, \\(\\varepsilon = 0.25\\), \\(\\kappa = 1\\): empirical pinball {fig["dro_emp"]:.4f}, exact dual {fig["dro_exact"]:.4f}, max-jump bound {fig["dro_bound"]:.4f}. The exact gain is {fig["dro_ratio"]:.2f} of the bound's gain. Archived <code>v4:4.1</code>: primal, dual and the separable form agree to \\(10^{{-8}}\\). <code>v4:4.2</code>: gain ratios 0.22 to 1.00. <code>v4:4.4</code>: outcome-only type 1 matches \\(\\varepsilon\\,\\bar\\rho/\\kappa_y\\). <code>v4:4.4b</code>: joint cost equals the covariate-only value when the dual multiplier already exceeds \\(\\bar\\rho/\\kappa_y\\), and is larger otherwise (6.79 against 2.14 at \\(\\kappa_y = 0.05\\), \\(\\varepsilon = 0.4\\)). <code>v4:4.5</code>: window exact 0.891 ≤ total-variation bound 1.911. <code>v4:4.7</code>: the sum of per-feature marginal distances is at most the joint distance, with ratios up to 1.64, so those marginals are not a valid radius. <code>v4:4.10</code>: outcome-only type 2 depends on the residuals (gain 0.0150 against 0.0269 at \\(\\varepsilon = 0.05\\)).</p>
<figure>{dro_svg}<figcaption>The middle bar is the linear programme. The right bar is the largest-jump penalty. It is above the truth. It is not a substitute that happens to be equal.</figcaption></figure>
<div class="breaks"><strong>Where this breaks.</strong> The jump penalty is an upper bound. It is tight only when some residual sits next to the biggest jump with the right sign. A sum of adjacent jumps is the local, type-∞ penalty, not the type-1 penalty. Raw covariate units have no finite Lipschitz constant: the initial slope grows like one over the distance to a bin edge (<code>v4:4.3</code>: 77.7, 77.7, 297, 1487 as that distance falls). Outcome-only type 1 must not be exposed as a regulariser. Summing per-level worst cases bounds a shared adversary; it is not equal to one (<code>v4:4.8</code>: 0.570 ≤ 0.642). Do not set \\(\\varepsilon\\) from per-feature marginal distances. <span class="unv">UNVERIFIED</span> Mohajerin Esfahani and Kuhn, Math. Program. 171 (2018): the numbering of the Lipschitz-loss results, and the recalled exponent \\(n^{{-1/\\max(d,2)}}\\). <span class="unv">UNVERIFIED</span> Gao and Kleywegt: journal volume and pages. <span class="unv">UNVERIFIED</span> Blanchet and Murthy: page range of Theorem 1. <span class="unv">UNVERIFIED</span> Gao, Chen and Kleywegt (Oper. Res. 2024): theorem-level statements, confirmed at abstract level only. <span class="unv">UNVERIFIED</span> Shafieezadeh-Abadeh, Kuhn and Mohajerin Esfahani (JMLR 2019): theorem number and the exact equality conditions. Selection shifts that reweight rows without moving them are not small in this metric.</div>
</section>

<footer>
<p>Personal research project. Simulated examples only. TRACE is unpublished and unvalidated. The pass criteria for the v1 simulations are fixed in <code>docs/theory/DESIGN_DELTA.md</code> and are not revised from results.</p>
<p>Hosking, J. R. M. (1990). L-moments. JRSS-B 52(1):105–124. Petersen and Müller, Ann. Statist. 47(2):691–719 (2019), for global Fréchet regression in Wasserstein space. Other citations, including the ones flagged above, are as stated in THEORY.md.</p>
</footer>
</main>
<script>
(function () {{
  const svg = document.getElementById("qplot");
  const t3 = document.getElementById("t3");
  const t4 = document.getElementById("t4");
  const o3 = document.getElementById("t3o");
  const o4 = document.getElementById("t4o");
  const status = document.getElementById("monoStatus");
  function draw() {{
    const tau3 = parseFloat(t3.value);
    const tau4 = parseFloat(t4.value);
    o3.textContent = tau3.toFixed(3);
    o4.textContent = tau4.toFixed(3);
    const c1 = 1, c2 = tau3 * 5 * c1 / 3, c3 = tau4 * 7 * c1 / 3;
    const N = 161;
    let minQ = Infinity, maxQ = -Infinity, minD = Infinity;
    const pts = [];
    for (let i = 0; i < N; i++) {{
      const u = i / (N - 1);
      const p1 = 2 * u - 1;
      const p2 = 6 * u * u - 6 * u + 1;
      const p3 = 20 * u * u * u - 30 * u * u + 12 * u - 1;
      const q = c1 * p1 + c2 * p2 + c3 * p3;
      const qp = 2 * c1 + c2 * (12 * u - 6) + c3 * (60 * u * u - 60 * u + 12);
      if (q < minQ) minQ = q;
      if (q > maxQ) maxQ = q;
      if (qp < minD) minD = qp;
      pts.push([u, q, qp]);
    }}
    const pad = 0.15 * (maxQ - minQ + 1e-6);
    const y0 = minQ - pad, y1 = maxQ + pad;
    const X = u => 40 + u * 560;
    const Y = q => 20 + (y1 - q) / (y1 - y0) * 180;
    let d = "";
    pts.forEach((p, i) => {{ d += (i ? "L" : "M") + X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1); }});
    const ok = minD >= -1e-8;
    const three = Math.abs(tau3) <= 0.2 + 1e-9;
    svg.innerHTML =
      '<rect width="640" height="240" fill="#f7f5f2"/>' +
      '<line x1="40" y1="' + Y(0).toFixed(1) + '" x2="600" y2="' + Y(0).toFixed(1) + '" stroke="#ccc"/>' +
      '<path d="' + d + '" fill="none" stroke="' + (ok ? "#2f7d4a" : "#b85c38") + '" stroke-width="2"/>' +
      '<text x="40" y="16" font-size="12" font-family="system-ui,sans-serif" fill="#1d1d1f">Q(u), c1 = 1</text>';
    status.textContent = (ok ? "Q' stays nonnegative. " : "Q' goes negative (minimum " + minD.toFixed(3) + "). ")
      + (three ? "Three channels would still allow this τ₃. " : "Three channels cannot: |τ₃| > 1/5. ")
      + "Four-channel minimum of Q' is " + minD.toFixed(3) + ".";
  }}
  t3.addEventListener("input", draw);
  t4.addEventListener("input", draw);
  draw();
}})();
</script>
</body>
</html>
"""
    return html


def main():
    fig = compute()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page(fig))
    print(OUT, "bytes", OUT.stat().st_size)


if __name__ == "__main__":
    main()
