"""TRACE simulation suite S1–S4.

Criteria are stated in docs/TRACE_SIMS.md and are not revised from these results.
S4 runs each cell in a child process so peak RSS belongs to that cell.

    python benchmarks/trace_sims/run_sims.py
    python benchmarks/trace_sims/run_sims.py --only s1
    python benchmarks/trace_sims/run_sims.py --child 1000000 100
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import subprocess
import sys
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np

from traceot import TraceRegressor, checks, primitives

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks" / "results" / "trace_sims.json"
_NORM = NormalDist()

S1_KW = dict(
    n_channels=3,
    max_bins=16,
    learning_rate=0.4,
    max_rounds=80,
    patience=12,
    val_frac=0.2,
    n_bags=1,
    max_interactions=0,
    min_count=5,
    u_levels=31,
    seed=11,
)
S2_KW = dict(
    n_channels=3,
    max_bins=16,
    learning_rate=0.35,
    max_rounds=25,
    patience=6,
    val_frac=0.2,
    n_bags=1,
    max_interactions=0,
    u_levels=19,
    seed=0,
)
S3_KW = dict(
    n_channels=3,
    max_bins=12,
    learning_rate=0.35,
    max_rounds=30,
    patience=8,
    val_frac=0.15,
    n_bags=1,
    max_interactions=0,
    u_levels=19,
    seed=33,
)
S4_KW = dict(
    n_channels=3,
    max_bins=32,
    learning_rate=0.3,
    max_rounds=3,
    patience=0,
    val_frac=0.0,
    n_bags=1,
    bag_frac=1.0,
    max_interactions=0,
    min_count=1,
    u_levels=19,
    seed=0,
)


def _z(u: np.ndarray) -> np.ndarray:
    return np.array([_NORM.inv_cdf(float(v)) for v in np.clip(u, 1e-12, 1 - 1e-12)])


def _energy(coef: np.ndarray, channel: int) -> float:
    sl = np.asarray(coef)[1:, channel]
    return float(np.dot(sl, sl) / sl.size)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _shared_codes(periods: list[np.ndarray], max_bins: int):
    p = periods[0].shape[1]
    edges = []
    coded = []
    for j in range(p):
        pooled = np.concatenate([block[:, j] for block in periods])
        pooled = np.ascontiguousarray(np.sort(pooled[np.isfinite(pooled)]))
        edge = primitives.bin_edges_from_sorted(pooled, max_bins)
        edges.append(np.asarray(edge, dtype=np.float64))
    for block in periods:
        codes = np.empty((p, block.shape[0]), dtype=np.uint8)
        for j in range(p):
            codes[j] = primitives.bin_column(np.ascontiguousarray(block[:, j]), edges[j])
        coded.append(codes)
    return coded, edges


def run_s1() -> dict:
    rng = np.random.default_rng(11)
    n = 2000
    x1 = rng.uniform(size=(n, 8))
    x2 = rng.uniform(size=(n, 8))
    z1 = _z(rng.random(n))
    z2 = _z(rng.random(n))
    he3 = z2**3 - 3 * z2
    y1 = z1
    y2 = 1.2 * x2[:, 0] + (0.4 + 1.2 * x2[:, 1]) * z2 + 0.12 * x2[:, 2] * he3
    (c1, c2), edges = _shared_codes([x1, x2], S1_KW["max_bins"])
    m1 = TraceRegressor(**S1_KW).fit_codes(c1, edges, y1)
    m2 = TraceRegressor(**S1_KW).fit_codes(c2, edges, y2)
    names = ["x_loc", "x_scale", "x_shape"] + [f"noise{i}" for i in range(5)]
    diff = []
    for j in range(8):
        a, _ = m1.effect(j)
        b, _ = m2.effect(j)
        diff.append(np.asarray(b) - np.asarray(a))
    channels = {"location": 0, "scale": 1, "shape": 2}
    energy = {
        names[j]: {name: _energy(diff[j], ch) for name, ch in channels.items()} for j in range(8)
    }

    def dominate(feature: str, channel: str) -> bool:
        own = energy[feature][channel]
        others = max(energy[n][channel] for n in names if n != feature)
        return own >= 3 * others

    idx = np.arange(1, diff[0].shape[0])
    corrs = {
        "x_loc_location": _corr(idx, diff[0][1:, 0]),
        "x_scale_scale": _corr(idx, diff[1][1:, 1]),
        "x_shape_shape": _corr(idx, diff[2][1:, 2]),
    }
    checks_out = {
        "location_dominates": dominate("x_loc", "location"),
        "scale_dominates": dominate("x_scale", "scale"),
        "shape_dominates": dominate("x_shape", "shape"),
        "location_correlation": corrs["x_loc_location"] > 0.85,
        "scale_correlation": corrs["x_scale_scale"] > 0.7,
        "shape_correlation": corrs["x_shape_shape"] > 0.7,
    }
    return {
        "energy": energy,
        "correlation": corrs,
        "checks": checks_out,
        "pass": all(checks_out.values()),
    }


def run_s2() -> dict:
    rng = np.random.default_rng(22)
    n = 1800
    t = np.arange(n)
    x_signal = rng.normal(size=n)
    seasonal = np.sin(2 * np.pi * t / n)
    y = 0.8 * x_signal + seasonal + 0.05 * rng.normal(size=n)
    x_future = np.empty(n)
    x_future[:-1] = y[1:]
    x_future[-1] = y[-1]
    x = np.column_stack(
        [
            x_signal,
            seasonal,
            y + 0.01 * rng.normal(size=n),
            x_future,
            x_signal.copy(),
            rng.normal(size=n),
        ]
    )
    names = ["x_signal", "x_leak", "x_proxy", "x_future", "x_dup", "x_noise"]
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    _corr_now, flags = checks.target_proxy_screen(x, y, 0.9, 1)
    flagged = [(int(f[0]), int(f[1]), float(f[2])) for f in flags]
    by_feature = {}
    for j, shift, _corr in flagged:
        by_feature.setdefault(j, []).append(shift)
    proxy_ok = 0 in by_feature.get(2, [])
    future_ok = any(s != 0 for s in by_feature.get(3, []))
    quiet = all(j not in by_feature for j in (0, 1, 5))
    dups = [
        (int(a), int(b), bool(ex), float(c)) for a, b, ex, c in checks.duplicate_screen(x, 0.999)
    ]
    exact = [(a, b) for a, b, ex, _c in dups if ex]
    dup_ok = exact == [(0, 4)] and all(ex for _a, _b, ex, _c in dups)
    real, placebo_max, placebo_mean = checks.placebo_screen(x, y, 20, 22)
    real = [float(v) for v in real]
    placebo_ok = float(placebo_max) < 0.25 and real[2] > 0.95

    n_tr = int(0.7 * n)
    w_time = np.asarray(checks.feature_w2(x[:n_tr], x[n_tr:], 64), dtype=np.float64)
    perm = np.random.default_rng(23).permutation(n)
    half = n // 2
    w_rand = np.asarray(checks.feature_w2(x[perm[:half]], x[perm[half:]], 64), dtype=np.float64)
    ratio = w_time / np.maximum(w_rand, 1e-15)
    leak_ratio_ok = float(ratio[1]) > 3
    signal_ratio_ok = float(ratio[0]) < 2
    leak_is_largest = int(np.argmax(ratio)) == 1

    m_time = TraceRegressor(**S2_KW).fit(x[:n_tr], y[:n_tr])
    imp_time = float(m_time.permutation_importance(x[n_tr:], y[n_tr:], repeats=5, seed=0)[1])
    tr, te = perm[:half], perm[half:]
    m_rand = TraceRegressor(**S2_KW).fit(x[tr], y[tr])
    imp_rand = float(m_rand.permutation_importance(x[te], y[te], repeats=5, seed=0)[1])
    importance_ok = imp_rand > 2 * imp_time

    out_checks = {
        "proxy_flagged": proxy_ok,
        "future_flagged": future_ok,
        "signal_leak_noise_quiet": quiet,
        "duplicate_exact_only": dup_ok,
        "placebo": placebo_ok,
        "leak_w2_ratio": leak_ratio_ok,
        "signal_w2_ratio": signal_ratio_ok,
        "leak_largest_w2_ratio": leak_is_largest,
        "time_split_importance": importance_ok,
    }
    return {
        "names": names,
        "proxy_flags": flagged,
        "duplicates": dups,
        "real_abs_corr": real,
        "placebo_max": float(placebo_max),
        "placebo_mean_max": float(placebo_mean),
        "w2_time": w_time.tolist(),
        "w2_random": w_rand.tolist(),
        "w2_ratio": ratio.tolist(),
        "importance_leak_random": imp_rand,
        "importance_leak_time": imp_time,
        "checks": out_checks,
        "pass": all(out_checks.values()),
    }


def run_s3() -> dict:
    rng = np.random.default_rng(33)
    n = 2400
    x = rng.uniform(size=(n, 4))
    y = x[:, 0] + 0.5 * x[:, 1] + rng.normal(scale=0.5, size=n)
    model = TraceRegressor(**S3_KW).fit(x[:1200], y[:1200])
    y_test = y[1800:].copy()
    y_test[:30] += 6.0
    y_test[200:300] += 2.5
    u = model.u_
    s_cal = checks.pinball_scores(y[1200:1800], model.predict(x[1200:1800]), u)
    s_te = checks.pinball_scores(y_test, model.predict(x[1800:]), u)
    p = np.asarray(checks.conformal_pvalues(s_cal, s_te))
    clean = np.ones(600, dtype=bool)
    clean[:30] = False
    clean[200:300] = False
    far = float(np.mean(p[clean] < 0.1))
    power = float(np.mean(p[:30] < 0.1))
    segment = float(np.mean(p[200:300] < 0.1))
    out_checks = {"far": far <= 0.20, "power": power >= 0.80, "segment": segment >= 0.50}
    return {
        "far": far,
        "power": power,
        "segment": segment,
        "checks": out_checks,
        "pass": all(out_checks.values()),
    }


def _s4_cell(n: int, p: int) -> dict:
    rng = np.random.default_rng(0)
    codes = np.empty((p, n), dtype=np.uint8)
    edges = []
    y = rng.normal(size=n)
    t_bin = 0.0
    for j in range(p):
        col = rng.random(n)
        if j == 0:
            y = y + 0.3 * col
        t0 = time.perf_counter()
        finite = np.ascontiguousarray(np.sort(col))
        edge = np.asarray(primitives.bin_edges_from_sorted(finite, 32), dtype=np.float64)
        codes[j] = primitives.bin_column(np.ascontiguousarray(col), edge)
        t_bin += time.perf_counter() - t0
        edges.append(edge)
        del col, finite
    t0 = time.perf_counter()
    TraceRegressor(**S4_KW).fit_codes(codes, edges, y)
    t_fit = time.perf_counter() - t0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return {
        "rows": n,
        "features": p,
        "bin_seconds": t_bin,
        "fit_seconds": t_fit,
        "wall_seconds": t_bin + t_fit,
        "peak_rss_mb": rss_mb,
    }


def run_s4() -> dict:
    cells = []
    for n in (10_000, 100_000, 1_000_000, 10_000_000):
        for p in (10, 100):
            cp = subprocess.run(
                [sys.executable, __file__, "--child", str(n), str(p)],
                capture_output=True,
                text=True,
                check=False,
            )
            if cp.returncode != 0:
                cells.append(
                    {
                        "rows": n,
                        "features": p,
                        "error": cp.stderr[-2000:],
                        "pass_cell": False,
                    }
                )
                continue
            row = json.loads(cp.stdout.strip().splitlines()[-1])
            row["pass_cell"] = True
            cells.append(row)
    by = {(c["rows"], c["features"]): c for c in cells if c.get("pass_cell")}

    def growth(p: int, big: int, small: int) -> float | None:
        a, b = by.get((big, p)), by.get((small, p))
        if not a or not b or b["wall_seconds"] <= 0:
            return None
        return a["wall_seconds"] / b["wall_seconds"]

    ratios = {f"p{p}_1e6_over_1e5": growth(p, 1_000_000, 100_000) for p in (10, 100)}
    ratios.update({f"p{p}_1e5_over_1e4": growth(p, 100_000, 10_000) for p in (10, 100)})
    finished = len(by) == 8
    growth_ok = finished and all(v is not None and v < 25 for v in ratios.values())
    return {
        "cells": cells,
        "ratios": ratios,
        "checks": {"all_finished": finished, "linear_growth": growth_ok},
        "pass": finished and growth_ok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=("s1", "s2", "s3", "s4"), default=None)
    ap.add_argument("--child", nargs=2, type=int, default=None)
    args = ap.parse_args()
    if args.child:
        print(json.dumps(_s4_cell(*args.child)))
        return
    runners = {"s1": run_s1, "s2": run_s2, "s3": run_s3, "s4": run_s4}
    keys = [args.only] if args.only else ["s1", "s2", "s3", "s4"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    if OUT.exists() and args.only:
        payload = json.loads(OUT.read_text())
    for key in keys:
        t0 = time.perf_counter()
        payload[key] = runners[key]()
        payload[key]["seconds"] = time.perf_counter() - t0
        print(key, "pass" if payload[key]["pass"] else "FAIL", f"{payload[key]['seconds']:.1f}s")
    payload["criteria"] = "docs/TRACE_SIMS.md"
    OUT.write_text(json.dumps(_finite(payload), indent=2, allow_nan=False))
    print(OUT)


def _finite(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite(v) for v in obj]
    return obj


if __name__ == "__main__":
    main()
