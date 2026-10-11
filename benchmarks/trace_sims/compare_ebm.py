"""TRACE against an explainable boosting machine, benchmark only.

``interpret`` is imported here and nowhere in the runtime package. If it is not
installed, the script records that and still scores TRACE on the vendor-spend draw.

    python benchmarks/trace_sims/compare_ebm.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np

from traceot import TraceRegressor, checks

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks" / "results" / "trace_ebm.json"
_NORM = NormalDist()


def _z(u: np.ndarray) -> np.ndarray:
    return np.array([_NORM.inv_cdf(float(v)) for v in np.clip(u, 1e-12, 1 - 1e-12)])


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
    names = [
        "log_units",
        "unit_price",
        "deliveries",
        "distance",
        "quality",
        "season",
        "noise0",
        "noise1",
        "noise2",
    ]
    return np.ascontiguousarray(x), np.ascontiguousarray(y), names


def _trace_row(x_tr, y_tr, x_te, y_te, **kw) -> dict:
    t0 = time.perf_counter()
    model = TraceRegressor(**kw).fit(x_tr, y_tr)
    fit_s = time.perf_counter() - t0
    q = model.predict(x_te)
    score = checks.score_quantiles(y_te, q, model.u_)
    score = {k: float(v) for k, v in score.items()}
    score["fit_seconds"] = fit_s
    c0, _ = model.effect(0)
    idx = np.arange(1, np.asarray(c0).shape[0])
    sl = np.asarray(c0)[1:, 0]
    sl = sl - sl.mean()
    idx = idx - idx.mean()
    denom = float(np.sqrt(np.dot(sl, sl) * np.dot(idx, idx)))
    score["location_bin_correlation"] = float(np.dot(sl, idx) / denom) if denom else float("nan")
    return score


def _conformal_quantiles(pred_te, resid, u):
    u = np.asarray(u, dtype=np.float64)
    order = np.argsort(resid)
    ranked = np.asarray(resid, dtype=np.float64)[order]
    n = ranked.shape[0]
    # Same rank rule as the Rust empirical quantile: floor(u * n), clipped.
    idx = np.minimum(np.floor(u * n).astype(np.int64), n - 1)
    q_resid = ranked[idx]
    return pred_te[:, None] + q_resid[None, :]


def _ebm_rows(x_tr, y_tr, x_te, y_te, u):
    try:
        from interpret.glassbox import ExplainableBoostingRegressor
    except Exception as exc:  # noqa: BLE001 — optional benchmark dependency
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    # Hold out the last 25% of the training rows as calibration for the residual quantiles.
    cut = max(int(0.75 * len(y_tr)), len(y_tr) - max(len(y_tr) // 4, 1))
    if cut >= len(y_tr) - 5:
        cut = len(y_tr) // 2
    t0 = time.perf_counter()
    ebm = ExplainableBoostingRegressor(interactions=0, random_state=0)
    ebm.fit(x_tr[:cut], y_tr[:cut])
    fit_s = time.perf_counter() - t0
    pred_te = np.asarray(ebm.predict(x_te), dtype=np.float64)
    pred_cal = np.asarray(ebm.predict(x_tr[cut:]), dtype=np.float64)
    resid = y_tr[cut:] - pred_cal
    point = np.repeat(pred_te[:, None], len(u), axis=1)
    point_score = checks.score_quantiles(y_te, point, u)
    band = _conformal_quantiles(pred_te, resid, u)
    band_score = checks.score_quantiles(y_te, band, u)
    quantile = {
        "available": False,
        "reason": "no quantile objective on this ExplainableBoostingRegressor",
    }
    # Some releases accept an objective string. Try once; a failure is recorded, not invented.
    try:
        qebm = ExplainableBoostingRegressor(interactions=0, random_state=0, objective="quantile")
        qebm.fit(x_tr[:cut], y_tr[:cut])
        qpred = np.asarray(qebm.predict(x_te), dtype=np.float64)
        quantile = {
            "available": True,
            "mse_of_prediction": float(np.mean((qpred - y_te) ** 2)),
            "note": "objective='quantile' was accepted; predict() is whatever that build returns",
        }
    except Exception as exc:  # noqa: BLE001
        quantile = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    out = {
        "available": True,
        "fit_seconds": fit_s,
        "point": {k: float(v) for k, v in point_score.items()},
        "conformal": {k: float(v) for k, v in band_score.items()},
        "quantile_objective": quantile,
    }
    # Shape: does the EBM term for feature 0 move with the feature?
    try:
        scores = np.asarray(ebm.term_scores_[0], dtype=np.float64).reshape(-1)
        out["term0_range"] = float(scores.max() - scores.min())
    except Exception as exc:  # noqa: BLE001
        out["term0_range"] = None
        out["term0_reason"] = f"{type(exc).__name__}: {exc}"
    return out


def _split(x, y, frac=0.7):
    n = int(frac * len(y))
    return x[:n], y[:n], x[n:], y[n:]


def run_case(name, x, y, trace_kw):
    xtr, ytr, xte, yte = _split(x, y)
    u_levels = trace_kw.get("u_levels", 19)
    u = (np.arange(u_levels) + 0.5) / u_levels
    return {
        "name": name,
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "trace": _trace_row(xtr, ytr, xte, yte, **trace_kw),
        "ebm": _ebm_rows(xtr, ytr, xte, yte, u),
    }


def main():
    kw = dict(
        n_channels=3,
        max_bins=16,
        learning_rate=0.35,
        max_rounds=40,
        patience=8,
        val_frac=0.2,
        n_bags=1,
        max_interactions=0,
        u_levels=19,
        seed=0,
    )
    x, y, names = _vendor(4000, 44)
    vendor = run_case("vendor_spend", x, y, kw)
    vendor["features"] = names
    # S1 period 2, the period that actually has location, scale and shape.
    rng = np.random.default_rng(11)
    n = 2000
    xs = rng.uniform(size=(n, 8))
    z = _z(rng.random(n))
    ys = 1.2 * xs[:, 0] + (0.4 + 1.2 * xs[:, 1]) * z + 0.12 * xs[:, 2] * (z**3 - 3 * z)
    s1 = run_case("s1_period2", xs, ys, {**kw, "u_levels": 31, "max_rounds": 60, "seed": 11})
    # A modest scale cell. 1e7 is the TRACE timing suite, not an EBM fit.
    rng = np.random.default_rng(0)
    n, p = 20_000, 10
    x4 = rng.random((n, p))
    y4 = x4[:, 0] + 0.1 * rng.normal(size=n)
    scale = run_case(
        "s4_style_n2e4_p10",
        x4,
        y4,
        {**kw, "max_rounds": 3, "patience": 0, "val_frac": 0.0, "max_bins": 32},
    )
    payload = {"vendor_spend": vendor, "s1_period2": s1, "scale_small": scale}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(OUT)


if __name__ == "__main__":
    main()
