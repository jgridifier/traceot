# ruff: noqa: E501
"""Landscape item 1, scored before any result is used to change a bar.

Protocol, fixed from docs/research/LANDSCAPE.md §4 before this run:
- Cases: S1 period 2, vendor spend, n=2e4 p=10. Generators match compare_ebm.py.
- Fresh seeds 81001..81020. Historical seeds 11, 44 and 0 are reported and are not in the pass set.
- Train/test is the first 70% / last 30%. The mean and the EBM both fit on the first 75% of that train block, the same cut the existing conformal comparison uses.
- Mean booster: max_bins 1024, learning rate 0.04, 50000 rounds, patience 100, tolerance 1e-5, validation fraction 0.15, min leaf 4. Bins thinner than the leaf size are pooled with their neighbours so the leaf size does not freeze the fit. No bags.
- Residual channels: k_fit 6, k_report 4, 12 Gauss–Legendre nodes, 16 bins, 40 rounds, tol 1e-8, mean channel locked.
- CRPS is the Gauss quadrature 2 Σ w_t ρ_u on those nodes, for the hybrid, for EBM+conformal, and for TRACE raw interpolated onto the same nodes.
- Pass: seed-mean MSE ≤ 1.01× EBM on all 3 cases; seed-mean CRPS ≤ EBM+conformal on ≥ 2 of 3; seed-mean CRPS ≤ 1.01× TRACE raw on all 3.
- Also, a one-sided paired t on those same contrasts, p < 0.05. A CRPS case counts toward the 2-of-3 only if both the mean and the t-test agree.

    python benchmarks/trace_sims/run_v2_item1.py
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
OUT = ROOT / "benchmarks" / "results" / "trace_v2_item1.json"
_NORM = NormalDist()
FRESH = list(range(81001, 81021))
HISTORICAL = {"s1_period2": 11, "vendor_spend": 44, "scale_small": 0}


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


def _split(x, y, frac=0.7):
    n = int(frac * len(y))
    return x[:n], y[:n], x[n:], y[n:]


def _cut(n: int) -> int:
    cut = max(int(0.75 * n), n - max(n // 4, 1))
    if cut >= n - 5:
        cut = n // 2
    return cut


def _interp_quantiles(q, u_src, u_dst):
    out = np.empty((q.shape[0], len(u_dst)))
    for i in range(q.shape[0]):
        out[i] = np.interp(u_dst, u_src, q[i])
    return out


def _conformal(pred, resid, u):
    order = np.argsort(resid)
    ranked = np.asarray(resid, dtype=np.float64)[order]
    n = ranked.shape[0]
    idx = np.minimum(np.floor(np.asarray(u) * n).astype(np.int64), n - 1)
    return pred[:, None] + ranked[idx][None, :]


def _cover(y, q, u):
    # Linear interpolation of the curve, same rule as the Rust score.
    def at(level):
        return np.array([np.interp(level, u, row) for row in q])

    c80 = np.mean((y >= at(0.1)) & (y <= at(0.9)))
    c90 = np.mean((y >= at(0.05)) & (y <= at(0.95)))
    return float(c80), float(c90)


def _one(spec: dict) -> dict:
    from interpret.glassbox import ExplainableBoostingRegressor

    from traceot import TraceHybrid, TraceRegressor, _core

    case, seed = spec["case"], int(spec["seed"])
    x, y = _draw(case, seed)
    xtr, ytr, xte, yte = _split(x, y)
    cut = _cut(len(ytr))
    t0 = time.perf_counter()
    hybrid = TraceHybrid(
        k_fit=6,
        k_report=4,
        n_u=0,
        max_bins=16,
        max_rounds=40,
        tol=1e-8,
        mono="none",
        reporting="legendre",
    ).fit(xtr[:cut], ytr[:cut])
    hybrid_s = time.perf_counter() - t0
    nodes = np.asarray(hybrid.model_.u_nodes, dtype=np.float64)
    weights = np.asarray(hybrid.model_.u_weights, dtype=np.float64)
    q_h = hybrid.predict(xte, nodes, project=True)
    crps_h = float(
        _core.v1_crps_quadrature(
            np.ascontiguousarray(yte), np.ascontiguousarray(q_h), nodes, weights
        )
    )
    mse_h = float(np.mean((hybrid.mean(xte) - yte) ** 2))
    c80_h, c90_h = _cover(yte, q_h, nodes)

    t0 = time.perf_counter()
    ebm = ExplainableBoostingRegressor(interactions=0, random_state=0)
    ebm.fit(xtr[:cut], ytr[:cut])
    ebm_s = time.perf_counter() - t0
    pred = np.asarray(ebm.predict(xte), dtype=np.float64)
    pred_cal = np.asarray(ebm.predict(xtr[cut:]), dtype=np.float64)
    band = _conformal(pred, ytr[cut:] - pred_cal, nodes)
    crps_e = float(
        _core.v1_crps_quadrature(
            np.ascontiguousarray(yte), np.ascontiguousarray(band), nodes, weights
        )
    )
    mse_e = float(np.mean((pred - yte) ** 2))
    c80_e, c90_e = _cover(yte, band, nodes)

    kw = dict(
        n_channels=3,
        max_bins=16 if case != "scale_small" else 32,
        learning_rate=1.0,
        max_rounds=40,
        patience=8,
        min_rounds=20,
        val_frac=0.2,
        n_bags=1,
        max_interactions=0,
        u_levels=31 if case == "s1_period2" else 19,
        seed=0,
    )
    t0 = time.perf_counter()
    raw = TraceRegressor(**kw).fit(xtr, ytr)
    raw_s = time.perf_counter() - t0
    q_raw = _interp_quantiles(np.asarray(raw.predict(xte)), np.asarray(raw.u_), nodes)
    crps_r = float(
        _core.v1_crps_quadrature(
            np.ascontiguousarray(yte), np.ascontiguousarray(q_raw), nodes, weights
        )
    )
    row = {
        "case": case,
        "seed": seed,
        "historical": bool(spec["historical"]),
        "mse_hybrid": mse_h,
        "mse_ebm": mse_e,
        "crps_hybrid": crps_h,
        "crps_ebm": crps_e,
        "crps_raw": crps_r,
        "cover80_hybrid": c80_h,
        "cover90_hybrid": c90_h,
        "cover80_ebm": c80_e,
        "cover90_ebm": c90_e,
        "seconds_hybrid": hybrid_s,
        "seconds_ebm": ebm_s,
        "seconds_raw": raw_s,
        "mean_rounds": int(hybrid.model_.mean_rounds),
        "residual_rounds": int(hybrid.model_.rounds),
    }
    print(
        f"{case} seed {seed} mse {mse_h:.4f}/{mse_e:.4f} crps {crps_h:.4f}/{crps_e:.4f}/{crps_r:.4f} {hybrid_s:.1f}s",
        flush=True,
    )
    return row


def _ttest_less(diff: list[float]) -> dict:
    from scipy import stats

    a = np.asarray(diff, dtype=np.float64)
    if len(a) < 3 or not np.all(np.isfinite(a)):
        return {"p": None, "mean": None}
    if float(np.std(a)) == 0.0:
        return {"p": 1.0 if float(a[0]) >= 0 else 0.0, "mean": float(a.mean())}
    res = stats.ttest_rel(a, np.zeros_like(a), alternative="less")
    return {"p": float(res.pvalue), "mean": float(a.mean())}


def _summarise(rows: list[dict]) -> dict:
    fresh = [r for r in rows if not r["historical"]]
    by = {}
    for case in ("s1_period2", "vendor_spend", "scale_small"):
        block = [r for r in fresh if r["case"] == case]
        mse_h = [r["mse_hybrid"] for r in block]
        mse_e = [r["mse_ebm"] for r in block]
        crps_h = [r["crps_hybrid"] for r in block]
        crps_e = [r["crps_ebm"] for r in block]
        crps_r = [r["crps_raw"] for r in block]
        mse_ratio = float(np.mean(mse_h) / np.mean(mse_e))
        crps_vs_ebm = float(np.mean(crps_h) - np.mean(crps_e))
        crps_vs_raw = float(np.mean(crps_h) / np.mean(crps_r))
        t_mse = _ttest_less([h - 1.01 * e for h, e in zip(mse_h, mse_e, strict=True)])
        t_ebm = _ttest_less([h - e for h, e in zip(crps_h, crps_e, strict=True)])
        t_raw = _ttest_less([h - 1.01 * r for h, r in zip(crps_h, crps_r, strict=True)])
        by[case] = {
            "n": len(block),
            "mse_hybrid": float(np.mean(mse_h)),
            "mse_ebm": float(np.mean(mse_e)),
            "mse_ratio": mse_ratio,
            "mse_magnitude": mse_ratio <= 1.01,
            "mse_ttest": t_mse,
            "crps_hybrid": float(np.mean(crps_h)),
            "crps_ebm": float(np.mean(crps_e)),
            "crps_raw": float(np.mean(crps_r)),
            "crps_beats_ebm_mean": crps_vs_ebm <= 0.0,
            "crps_beats_ebm_ttest": t_ebm["p"] is not None and t_ebm["p"] < 0.05,
            "crps_vs_raw_ratio": crps_vs_raw,
            "raw_magnitude": crps_vs_raw <= 1.01,
            "raw_ttest": t_raw,
            "cover80_hybrid": float(np.mean([r["cover80_hybrid"] for r in block])),
            "cover90_hybrid": float(np.mean([r["cover90_hybrid"] for r in block])),
            "cover80_ebm": float(np.mean([r["cover80_ebm"] for r in block])),
            "cover90_ebm": float(np.mean([r["cover90_ebm"] for r in block])),
        }

    def _p_ok(test: dict) -> bool:
        p = test["p"]
        return p is not None and p < 0.05

    mse_ok = all(by[c]["mse_magnitude"] and _p_ok(by[c]["mse_ttest"]) for c in by)
    raw_ok = all(by[c]["raw_magnitude"] and _p_ok(by[c]["raw_ttest"]) for c in by)
    crps_wins = sum(1 for c in by if by[c]["crps_beats_ebm_mean"] and by[c]["crps_beats_ebm_ttest"])
    return {
        "cases": by,
        "mse_pass": mse_ok,
        "crps_wins": crps_wins,
        "crps_pass": crps_wins >= 2,
        "raw_pass": raw_ok,
        "pass": bool(mse_ok and crps_wins >= 2 and raw_ok),
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
    specs = []
    for case in ("s1_period2", "vendor_spend", "scale_small"):
        for seed in FRESH:
            specs.append({"case": case, "seed": seed, "historical": False})
        specs.append({"case": case, "seed": HISTORICAL[case], "historical": True})
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
    print("pass", OUT.read_text().count('"pass"'), flush=True)


if __name__ == "__main__":
    main()
