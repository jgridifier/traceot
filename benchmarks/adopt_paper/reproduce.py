"""Reproduce the simulation study of the ADOPT paper (Section 5, Tables 1 and 2).

Usage::

    python benchmarks/adopt_paper/reproduce.py --reps 200 --workers 6

Writes ``benchmarks/results/adopt_reproduction.json`` (summary) and
``benchmarks/results/adopt_reproduction_reps.csv`` (one row per replicate and method).
Fixed seeds: replicate r of setting s uses ``SeedSequence([TABLE, CASE, n, m, r])``.
Choices not fixed by the paper are documented in docs/REPRODUCTION_NOTES.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

# Several worker processes run in parallel; keep each one single-threaded.
os.environ.setdefault("RAYON_NUM_THREADS", "1")

import dgp  # noqa: E402  (benchmarks/adopt_paper/dgp.py, the paper's designs)
import traceot  # noqa: E402
from traceot import (  # noqa: E402
    ADOPTRegressor,
    FrechetMeanRegressor,
    GlobalFrechetRegressor,
    L2AdditiveQuantileRegressor,
)
from traceot.primitives import coords_to_spd, quantile_grid  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks" / "results"
Q = 100  # quantile grid size (CHOICE)
GRID_PER_AXIS = 10  # MISE integration grid on [0,1]^3 (CHOICE)
CASE_ID = {"I": 1, "II": 2}


def settings():
    for case in ("I", "II"):
        for n in (100, 300):
            yield ("table1", case, n, 0)
    for case in ("I", "II"):
        for m in (10, 20):
            for n in (100, 300):
                yield ("table2", case, n, m)


_CACHE: dict = {}


def _truth(table, case, m):
    key = (table, case, m)
    if key not in _CACHE:
        xg = dgp.mise_grid(GRID_PER_AXIS)
        if table == "table1":
            tr = dgp.W2Truth.build(case)
            _CACHE[key] = (tr, xg, tr.true_quantiles(xg, quantile_grid(Q)))
        else:
            means = dgp.spd_population_means(case, m)
            _CACHE[key] = (means, xg, dgp.spd_true_coords(case, xg, m, means))
    return _CACHE[key]


def run_one(task):
    table, case, n, m, rep = task
    ss = np.random.SeedSequence([1 if table == "table1" else 2, CASE_ID[case], n, m, rep])
    rng = np.random.default_rng(ss)
    obj, xg, truth = _truth(table, case, m)
    out = []
    if table == "table1":
        X, Y = dgp.simulate_w2(case, n, rng, q=Q, truth=obj)
        models = {
            "ADOPT": ADOPTRegressor(random_state=rep),
            "GF": GlobalFrechetRegressor(),
            "FM": FrechetMeanRegressor(),
            "L2ADD": L2AdditiveQuantileRegressor(random_state=rep),
        }
        for name, est in models.items():
            t0 = time.perf_counter()
            est.fit(X, Y)
            pred = est.predict(xg)
            dt = time.perf_counter() - t0
            mise = float(np.mean(np.mean((pred - truth) ** 2, axis=1)))
            bw = getattr(est, "cv_results_", None)
            out.append((name, mise, dt, None if not bw else float(bw["best_fraction"])))
    else:
        X, Z = dgp.simulate_spd(case, n, m, rng, means=obj)
        S = dgp_spd_matrices(Z, m)
        models = {
            "ADOPT": ADOPTRegressor(space="logcholesky", random_state=rep),
            "GF": GlobalFrechetRegressor(space="logcholesky"),
            "FM": FrechetMeanRegressor(space="logcholesky"),
        }
        for name, est in models.items():
            t0 = time.perf_counter()
            est.fit(X, S)
            pred = est.predict_encoded(xg)
            dt = time.perf_counter() - t0
            mise = float(np.mean(np.sum((pred - truth) ** 2, axis=1)))
            bw = getattr(est, "cv_results_", None)
            out.append((name, mise, dt, None if not bw else float(bw["best_fraction"])))
    return task, out


def dgp_spd_matrices(Z, m):
    return coords_to_spd(Z, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default=None, help="table1 or table2")
    args = ap.parse_args()
    tasks = [
        (t, c, n, m, r)
        for (t, c, n, m) in settings()
        if args.only in (None, t)
        for r in range(args.reps)
    ]
    t0 = time.time()
    with get_context("fork").Pool(args.workers) as pool:
        results = pool.map(run_one, tasks, chunksize=4)
    wall = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    agg: dict = {}
    for (t, c, n, m, r), out in results:
        key = f"{c}/{n}" if t == "table1" else f"{c}/{m}/{n}"
        for name, mise, dt, bw in out:
            rows.append([t, c, n, m, r, name, mise, dt, bw])
            agg.setdefault(t, {}).setdefault(key, {}).setdefault(name, []).append((mise, dt, bw))
    with open(OUT / "adopt_reproduction_reps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["table", "case", "n", "m", "rep", "method", "mise", "seconds", "bw_fraction"])
        w.writerows(rows)
    summary = {
        "meta": {
            "reps": args.reps,
            "q": Q,
            "mise_grid_per_axis": GRID_PER_AXIS,
            "traceot_version": traceot.__version__,
            "backend": "rust",
            "python": platform.python_version(),
            "wall_seconds": wall,
            "workers": args.workers,
        },
        "results": {},
    }
    for t, d in agg.items():
        scale = 1e3 if t == "table1" else 1e2
        summary["results"][t] = {"scale": 1 / scale}
        for key, methods in d.items():
            summary["results"][t][key] = {}
            for name, vals in methods.items():
                arr = np.array([v[0] for v in vals]) * scale
                bws = [v[2] for v in vals if v[2] is not None]
                summary["results"][t][key][name] = {
                    "mean": float(arr.mean()),
                    "se": float(arr.std(ddof=1) / np.sqrt(arr.size)),
                    "median_seconds": float(np.median([v[1] for v in vals])),
                    "bw_fraction_counts": (
                        {str(b): bws.count(b) for b in sorted(set(bws))} if bws else None
                    ),
                }
    with open(OUT / "adopt_reproduction.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["meta"], indent=2))


if __name__ == "__main__":
    main()
