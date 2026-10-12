# ruff: noqa: E501
"""Landscape item 3, scored before any result is used to change a bar.

Protocol, fixed from docs/research/LANDSCAPE.md §4 #3 before this run:
- Candidate span: k_fit=8, k_report=4, reporting="hermite" so channel 3 is He3, max_bins=16, learning_rate=1.0, max_rounds=40, tol=1e-8, mono="none".
- Penalty λ Σ_k k^{2s} a_k² and tail weight 1 + γ(u−½)² on channels k ≥ 2.
- The triple is chosen by validation CRPS from this grid, and from nowhere else: s ∈ {1, 2}, λ ∈ {1e-4, 1e-3, 1e-2, 1e-1}, γ ∈ {0, 1, 4}. Fit on the first 80% of the train block, score the last 20%, keep the lowest CRPS, then refit that triple on the whole train block.
- Baseline: k_fit=3, k_report=3, reporting="hermite", same bins, rounds and learning rate, λ=0, γ=0.
- Retention: S1 period 2, n=2000, 50 seeds 84001..84050. No test split. The grid is chosen on the last 20%, and the screened model is the fit on the first 80% (the model the screen sees). A feature is retained when its pinball gain or its shape-channel gain is > 0. Pass if x_shape, column 2, is retained on ≥ 40 of 50 seeds.
- Energy ratio: on those same 50 fits, hermite channel-3 energy of x_shape divided by the largest other feature. A denominator below 1e-15 makes the ratio missing and drops that seed from the mean. Pass if the mean of the finite ratios is ≥ 2 and a one-sided t of (ratio − 2) has p < 0.05, with at least 40 finite ratios.
- CRPS: the three cases, seeds 81001..81020, first 70% train, last 30% test. Pass if seed-mean CRPS ≤ 1.005 × baseline on all three cases.

    python benchmarks/trace_sims/run_v2_item3.py
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks" / "results" / "trace_v2_item3.json"
_NORM = NormalDist()
FRESH = list(range(81001, 81021))
SCREEN_SEEDS = list(range(84001, 84051))
GRID = [
    (s, lam, gamma)
    for s in (1.0, 2.0)
    for lam in (1e-4, 1e-3, 1e-2, 1e-1)
    for gamma in (0.0, 1.0, 4.0)
]
CAND = dict(
    k_fit=8,
    k_report=4,
    n_u=0,
    max_bins=16,
    learning_rate=1.0,
    max_rounds=40,
    tol=1e-8,
    mono="none",
    reporting="hermite",
)
BASE = dict(
    k_fit=3,
    k_report=3,
    n_u=0,
    max_bins=16,
    learning_rate=1.0,
    max_rounds=40,
    tol=1e-8,
    mono="none",
    reporting="hermite",
    sobolev_lambda=0.0,
    tail_gamma=0.0,
)


def _z(u: np.ndarray) -> np.ndarray:
    return np.array([_NORM.inv_cdf(float(v)) for v in np.clip(u, 1e-12, 1.0 - 1e-12)])


def _vendor(n: int, seed: int):
    rng = np.random.default_rng(seed)
    log_units = rng.uniform(0, 2, size=n)
    unit_price = rng.uniform(0.5, 5, size=n)
    deliveries = rng.integers(1, 12, size=n).astype(np.float64)
    distance = rng.uniform(0, 1, size=n)
    quality = rng.uniform(0, 1, size=n)
    season = rng.uniform(0, 1, size=n)
    noise = rng.normal(size=(n, 3))
    x = np.column_stack([log_units, unit_price, deliveries, distance, quality, season, noise])
    z = _z(rng.random(n))
    he3 = z**3 - 3 * z
    y = 10 + 4 * log_units + 2 * unit_price + (0.8 + 0.5 * log_units) * z + 0.05 * quality * he3
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def _s1(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(n, 8))
    z = _z(rng.random(n))
    y = 1.2 * x[:, 0] + (0.4 + 1.2 * x[:, 1]) * z + 0.12 * x[:, 2] * (z**3 - 3 * z)
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def _scale(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.random((n, 10))
    y = x[:, 0] + 0.1 * rng.normal(size=n)
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def _draw(case: str, seed: int):
    if case == "s1_period2":
        return _s1(2000, seed)
    if case == "vendor_spend":
        return _vendor(4000, seed)
    if case == "scale_small":
        return _scale(20_000, seed)
    raise ValueError(case)


def _crps(model, x, y, nodes, weights, core):
    q = model.predict(x, nodes, project=False)
    return float(
        core.v1_crps_quadrature(np.ascontiguousarray(y), np.ascontiguousarray(q), nodes, weights)
    )


def _choose(x, y, core, TraceV1):
    cut = max(int(0.8 * len(y)), len(y) - max(len(y) // 5, 1))
    best = None
    best_crps = math.inf
    nodes = None
    weights = None
    for s, lam, gamma in GRID:
        model = TraceV1(**CAND, sobolev_s=s, sobolev_lambda=lam, tail_gamma=gamma).fit(
            x[:cut], y[:cut]
        )
        if nodes is None:
            nodes = np.asarray(model.model_.u_nodes(), dtype=np.float64)
            weights = np.asarray(core.v1_gauss_legendre(int(nodes.shape[0]))[1], dtype=np.float64)
        score = _crps(model, x[cut:], y[cut:], nodes, weights, core)
        if score < best_crps:
            best_crps = score
            best = (s, lam, gamma, score)
    return best


def _one(spec: dict) -> dict:
    from traceot import TraceV1, _core

    kind = spec["kind"]
    if kind == "screen":
        x, y = _s1(2000, int(spec["seed"]))
        chosen = _choose(x, y, _core, TraceV1)
        s, lam, gamma, val_crps = chosen
        cut = max(int(0.8 * len(y)), len(y) - max(len(y) // 5, 1))
        model = TraceV1(**CAND, sobolev_s=s, sobolev_lambda=lam, tail_gamma=gamma).fit(
            x[:cut], y[:cut]
        )
        rows = np.arange(cut, len(y), dtype=np.int64)
        # The screen scores the rows the fit did not see. Predict needs the full X
        # only to bin them; gains are computed on `rows`.
        screen = model.feature_gains(x, y, rows)
        energies = [float(model.channel_energy(j, 3)) for j in range(x.shape[1])]
        shape = energies[2]
        others = max(energies[j] for j in range(len(energies)) if j != 2)
        ratio = None if others < 1e-15 else shape / others
        row = {
            "kind": "screen",
            "seed": int(spec["seed"]),
            "s": s,
            "lambda": lam,
            "gamma": gamma,
            "val_crps": val_crps,
            "retained": bool(screen["selected"][2]),
            "gain": float(screen["gains"][2]),
            "shape_gain": float(screen["shape_gains"][2]),
            "energies": energies,
            "ratio": ratio,
        }
        print(
            f"screen seed {spec['seed']} retained {row['retained']} ratio {ratio} s {s} lam {lam} g {gamma}",
            flush=True,
        )
        return row

    case, seed = spec["case"], int(spec["seed"])
    x, y = _draw(case, seed)
    ntr = int(0.7 * len(y))
    xtr, ytr, xte, yte = x[:ntr], y[:ntr], x[ntr:], y[ntr:]
    chosen = _choose(xtr, ytr, _core, TraceV1)
    s, lam, gamma, val_crps = chosen
    cand = TraceV1(**CAND, sobolev_s=s, sobolev_lambda=lam, tail_gamma=gamma).fit(xtr, ytr)
    base = TraceV1(**BASE).fit(xtr, ytr)
    nodes = np.asarray(cand.model_.u_nodes(), dtype=np.float64)
    weights = np.asarray(_core.v1_gauss_legendre(int(nodes.shape[0]))[1], dtype=np.float64)
    # Baseline has K_fit=3, so its own nodes differ. Score both curves on the candidate nodes.
    crps_c = _crps(cand, xte, yte, nodes, weights, _core)
    crps_b = _crps(base, xte, yte, nodes, weights, _core)
    row = {
        "kind": "crps",
        "case": case,
        "seed": seed,
        "s": s,
        "lambda": lam,
        "gamma": gamma,
        "val_crps": val_crps,
        "crps_candidate": crps_c,
        "crps_baseline": crps_b,
    }
    print(
        f"{case} seed {seed} crps {crps_c:.4f}/{crps_b:.4f} s {s} lam {lam} g {gamma}", flush=True
    )
    return row


def _ttest_less(diff: list[float]) -> dict:
    from scipy import stats

    a = np.asarray(diff, dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) < 3:
        return {"p": None, "mean": None, "n": int(len(a))}
    if float(np.std(a)) == 0.0:
        return {"p": 1.0 if float(a[0]) >= 0 else 0.0, "mean": float(a.mean()), "n": int(len(a))}
    res = stats.ttest_rel(a, np.zeros_like(a), alternative="less")
    return {"p": float(res.pvalue), "mean": float(a.mean()), "n": int(len(a))}


def _summarise(rows: list[dict]) -> dict:
    screen = [r for r in rows if r["kind"] == "screen"]
    n_ret = sum(bool(r["retained"]) for r in screen)
    ratios = [r["ratio"] for r in screen if r["ratio"] is not None]
    t_ratio = _ttest_less([2.0 - r for r in ratios])
    # ttest_less on (2 - ratio) is the test that ratio > 2.
    ratio_pass = bool(
        len(ratios) >= 40
        and float(np.mean(ratios)) >= 2.0
        and t_ratio["p"] is not None
        and t_ratio["p"] < 0.05
    )
    retain_pass = n_ret >= 40
    by = {}
    for case in ("s1_period2", "vendor_spend", "scale_small"):
        block = [r for r in rows if r["kind"] == "crps" and r["case"] == case]
        c = [r["crps_candidate"] for r in block]
        b = [r["crps_baseline"] for r in block]
        ratio = float(np.mean(c) / np.mean(b))
        by[case] = {
            "n": len(block),
            "crps_candidate": float(np.mean(c)),
            "crps_baseline": float(np.mean(b)),
            "ratio": ratio,
            "magnitude": ratio <= 1.005,
        }
    crps_pass = all(by[c]["magnitude"] for c in by)
    return {
        "retention": {
            "n": len(screen),
            "n_retained": n_ret,
            "pass": retain_pass,
            "bar": ">= 40 of 50",
        },
        "energy": {
            "n_finite": len(ratios),
            "mean_ratio": float(np.mean(ratios)) if ratios else None,
            "median_ratio": float(np.median(ratios)) if ratios else None,
            "ttest_ratio_gt_2": t_ratio,
            "pass": ratio_pass,
        },
        "cases": by,
        "crps_pass": crps_pass,
        "pass": bool(retain_pass and ratio_pass and crps_pass),
    }


def _finite(obj):
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError("non-finite result")
        return obj
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite(v) for v in obj]
    raise TypeError(type(obj))


def main() -> None:
    specs = [{"kind": "screen", "seed": s} for s in SCREEN_SEEDS]
    for case in ("s1_period2", "vendor_spend", "scale_small"):
        for seed in FRESH:
            specs.append({"kind": "crps", "case": case, "seed": seed})
    rows = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_one, spec) for spec in specs]
        for fut in as_completed(futures):
            rows.append(fut.result())
            payload = {
                "completed": len(rows),
                "of": len(specs),
                "seconds": time.perf_counter() - t0,
                "rows": rows,
            }
            if len(rows) == len(specs):
                payload["summary"] = _summarise(rows)
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(_finite(payload), indent=2, allow_nan=False) + "\n")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
