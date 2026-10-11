"""Build the self-contained learning page docs/learn/adopt.html and docs/ADOPT_RESULTS.md.

All figures are generated here with matplotlib and inlined as SVG (text converted to
paths), so the page needs no external assets or CDNs. Inputs:

* benchmarks/adopt_paper/paper_values.json   (verbatim paper numbers)
* benchmarks/results/adopt_reproduction.json (our run; see reproduce.py)
* benchmarks/results/scale_bench.json        (optional)
* benchmarks/results/criterion_adopt.json    (optional; ADOPT fit times)

    python docs/build_docs.py
"""

from __future__ import annotations

import html
import io
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from traceot import (
    ADOPTRegressor,  # noqa: E402
    quantile_grid,  # noqa: E402
)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "benchmarks" / "results"
PAPER = json.loads((ROOT / "benchmarks" / "adopt_paper" / "paper_values.json").read_text())

plt.rcParams.update(
    {
        "svg.fonttype": "path",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10,
        "svg.hashsalt": "traceot",
    }
)
C = ["#1b6ca8", "#d1495b", "#2e8b57", "#7a5195", "#666666"]


def svg(fig, alt: str) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    s = buf.getvalue()
    s = s[s.index("<svg") :]
    s = re.sub(r'width="[^"]+pt"', 'width="100%"', s, count=1)
    s = re.sub(r'height="[^"]+pt"', 'height="auto"', s, count=1)
    s = s.replace("<svg ", f'<svg role="img" aria-label="{html.escape(alt)}" ', 1)
    return s


# --------------------------------------------------------------------------- figures
def fig_ot_1d() -> str:
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 40000)
    b = rng.gamma(2.0, 1.0, 40000) - 1.0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    bins = np.linspace(-4, 8, 90)
    ax1.hist(a, bins, density=True, histtype="stepfilled", alpha=0.35, color=C[0], label="before")
    ax1.hist(b, bins, density=True, histtype="stepfilled", alpha=0.35, color=C[1], label="after")
    for lv in (0.1, 0.25, 0.5, 0.75, 0.9):
        qa, qb = np.quantile(a, lv), np.quantile(b, lv)
        ax1.annotate(
            "",
            xy=(qb, 0.03),
            xytext=(qa, 0.03),
            arrowprops=dict(arrowstyle="->", color="#333", lw=1),
        )
    ax1.set_title("Move each percentile to the same percentile")
    ax1.set_yticks([])
    ax1.legend(frameon=False)
    u = np.linspace(0.005, 0.995, 200)
    ax2.plot(np.quantile(a, u), np.quantile(b, u), color=C[3])
    ax2.plot([-3, 3], [-3, 3], ls=":", color="#999")
    ax2.set_xlabel("outcome before")
    ax2.set_ylabel("outcome after")
    ax2.set_title("The transport map T (monotone)")
    return svg(fig, "Two distributions and the monotone optimal transport map between them")


def quickstart_data(seed=0, n=300, q=60):
    rng = np.random.default_rng(seed)
    u = quantile_grid(q)
    X = rng.uniform(size=(n, 3))
    base = np.log(u / (1 - u))
    Y = (
        np.sin(2 * np.pi * X[:, [0]])
        + (1.0 + X[:, [1]]) * base
        + 0.8 * X[:, [2]] * np.maximum(base, 0.0)
        + 0.1 * rng.normal(size=(n, 1))
    )
    return X, Y, u


def samples_from_quantiles(qf, u, m=40000, seed=0):
    v = np.random.default_rng(seed).uniform(u[0], u[-1], m)
    return np.interp(v, u, qf)


def fig_components() -> str:
    X, Y, u = quickstart_data()
    est = ADOPTRegressor().fit(X, Y)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), sharey=True)
    titles = ["x1: moves it (shift)", "x2: widens it (scale)", "x3: fattens right tail"]
    bins = np.linspace(-7, 9, 80)
    for j, ax in enumerate(axes):
        for c, xv in zip(C, (0.1, 0.5, 0.9), strict=False):
            qf = est.transport(j, xv, est.mean_)
            ax.hist(
                samples_from_quantiles(qf, u),
                bins,
                density=True,
                histtype="step",
                color=c,
                lw=1.4,
                label=f"x{j + 1} = {xv}",
            )
        ax.set_title(titles[j])
        ax.set_yticks([])
        ax.legend(frameon=False, fontsize=7)
    return svg(fig, "Estimated ADOPT transports for each feature applied to the mean distribution")


def load_ours():
    p = RES / "adopt_reproduction.json"
    return json.loads(p.read_text()) if p.exists() else None


def fig_repro(ours) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.2))
    for ax, t, methods in (
        (axes[0], "table1", ["ADOPT", "GF", "FM"]),
        (axes[1], "table2", ["ADOPT", "GF", "FM"]),
    ):
        keys = list(PAPER[t]["rows"].keys())
        xs = np.arange(len(keys))
        w = 0.13
        k = 0
        for mi, m in enumerate(methods):
            pv = [float(PAPER[t]["rows"][r][m][0]) for r in keys]
            ov = [ours["results"][t][r][m]["mean"] for r in keys]
            ax.bar(xs + (k - 2.5) * w, pv, w, color=C[mi], alpha=0.45, label=f"{m} paper")
            ax.bar(xs + (k - 1.5) * w, ov, w, color=C[mi], label=f"{m} ours")
            k += 2
        ax.set_yscale("log")
        ax.set_xticks(xs, [r.replace("/", " ") for r in keys], fontsize=8)
        unit = "×10⁻³" if t == "table1" else "×10⁻²"
        name = "Table 1 (distributions, W2)" if t == "table1" else "Table 2 (SPD, log-Cholesky)"
        ax.set_title(f"{name}: MISE {unit}, log scale")
    axes[0].legend(frameon=False, fontsize=7, ncol=3)
    axes[1].set_xlabel("case  [m]  n")
    fig.tight_layout()
    return svg(fig, "Paper-reported versus reproduced MISE for each simulation setting")


def fig_scaling():
    sb = RES / "scale_bench.json"
    cr = RES / "criterion_adopt.json"
    if not (sb.exists() and cr.exists()):
        return None
    s = json.loads(sb.read_text())["results"]
    c = json.loads(cr.read_text())
    fig, ax = plt.subplots(figsize=(6, 3))
    n_ad = np.array(c["n"], float)
    ax.loglog(n_ad, np.array(c["seconds"]), "o-", color=C[1], label="ADOPT fit (p=3, q=100)")
    for p, col in ((10, C[0]), (100, C[2])):
        rr = [r for r in s if r["features"] == p]
        ax.loglog(
            [r["rows"] for r in rr],
            [r["round_total"] for r in rr],
            "s-",
            color=col,
            label=f"TRACE primitives: one round, p={p}",
        )
    ax.set_xlabel("rows")
    ax.set_ylabel("seconds")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("ADOPT is quadratic in n; the TRACE core is linear")
    return svg(fig, "Run time versus number of rows for ADOPT and the TRACE primitives")


def scale_table_md() -> str | None:
    sb = RES / "scale_bench.json"
    if not sb.exists():
        return None
    d = json.loads(sb.read_text())
    out = [
        "| rows | features | setup: sketch | setup: bin | round: grad | round: hist "
        "| round total | peak RSS (MB) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in d["results"]:
        out.append(
            f"| {r['rows']:.0e} | {r['features']} | {r['sketch']:.3f} | {r['bin']:.3f} "
            f"| {r['grad']:.3f} | {r['hist']:.3f} | {r['round_total']:.3f} "
            f"| {r['peak_rss_mb']:.0f} |"
        )
    m = d.get("meta", {})
    out.append("")
    out.append(
        f"Machine: {m.get('cpu_count')} logical CPUs, {m.get('machine')}, "
        f"Python {m.get('python')}, rayon threads: {m.get('rayon_threads')}, run {m.get('date')}."
    )
    return "\n".join(out)


def update_benchmarks_md():
    p = ROOT / "docs" / "BENCHMARKS.md"
    tab = scale_table_md()
    if tab is None or not p.exists():
        return
    s = p.read_text()
    start, end = "<!-- SCALE_TABLE -->", "<!-- /SCALE_TABLE -->"
    if end in s:
        pre, rest = s.split(start, 1)
        post = rest.split(end, 1)[1]
    else:
        pre, post = s.split(start, 1)
    p.write_text(pre + start + "\n" + tab + "\n" + end + post)


# --------------------------------------------------------------------------- tables
def fmt(m, se):
    return f"{m:.3f} ({se:.3f})"


def table_rows(ours, t):
    rows = []
    for key, vals in PAPER[t]["rows"].items():
        o = ours["results"][t][key]
        r = {"setting": key}
        for m in PAPER[t]["methods"]:
            r[f"paper {m}"] = f"{vals[m][0]} ({vals[m][1]})"
            r[f"ours {m}"] = fmt(o[m]["mean"], o[m]["se"]) if m in o else "not run"
        if "L2ADD" in o:
            r["ours L2-additive*"] = fmt(o["L2ADD"]["mean"], o["L2ADD"]["se"])
        rows.append(r)
    return rows


def md_table(rows):
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(r[c] for c in cols) + " |" for r in rows]
    return "\n".join(out)


def html_table(rows, caption):
    cols = list(rows[0].keys())
    h = [f"<table><caption>{html.escape(caption)}</caption><thead><tr>"]
    h += [f"<th scope='col'>{html.escape(c)}</th>" for c in cols]
    h.append("</tr></thead><tbody>")
    for r in rows:
        h.append("<tr>" + "".join(f"<td>{html.escape(r[c])}</td>" for c in cols) + "</tr>")
    h.append("</tbody></table>")
    return "".join(h)


def main():
    ours = load_ours()
    if ours is None:
        raise SystemExit("run benchmarks/adopt_paper/reproduce.py first")
    t1, t2 = table_rows(ours, "table1"), table_rows(ours, "table2")
    meta = ours["meta"]
    md = [
        "# ADOPT reproduction: paper vs ours",
        "",
        "Generated by `docs/build_docs.py` from `benchmarks/adopt_paper/paper_values.json` "
        "(verbatim from the PMLR PDF) and `benchmarks/results/adopt_reproduction.json`.",
        f"Our run: B = {meta['reps']} replicates, q = {meta['q']}, MISE integrated on a "
        f"{meta['mise_grid_per_axis']}³ midpoint grid, backend `{meta['backend']}`.",
        "Entries are mean (standard error). *L2-additive is our extra comparator, not in "
        "the paper. ADR (Han et al. 2020) was not re-implemented.",
        "",
        "## Table 1: distributional responses (W2), MISE ×10⁻³",
        "",
        md_table(t1),
        "",
        "## Table 2: SPD responses (log-Cholesky), MISE ×10⁻²",
        "",
        md_table(t2),
        "",
    ]
    (ROOT / "docs" / "ADOPT_RESULTS.md").write_text("\n".join(md))
    figs = {
        "ot": fig_ot_1d(),
        "comp": fig_components(),
        "repro": fig_repro(ours),
        "scale": fig_scaling(),
    }
    tpl = (ROOT / "docs" / "learn" / "adopt.template.html").read_text()
    page = (
        tpl.replace("{{FIG_OT}}", figs["ot"])
        .replace("{{FIG_COMP}}", figs["comp"])
        .replace("{{FIG_REPRO}}", figs["repro"])
        .replace("{{FIG_SCALE}}", figs["scale"] or "<p><em>Scaling figure not built yet.</em></p>")
        .replace("{{TABLE1}}", html_table(t1, "Table 1: distributional responses, MISE ×10⁻³"))
        .replace("{{TABLE2}}", html_table(t2, "Table 2: SPD responses, MISE ×10⁻²"))
        .replace("{{REPS}}", str(meta["reps"]))
    )
    (ROOT / "docs" / "learn" / "adopt.html").write_text(page)
    update_benchmarks_md()
    print("wrote docs/learn/adopt.html, docs/ADOPT_RESULTS.md, docs/BENCHMARKS.md table")


if __name__ == "__main__":
    main()
