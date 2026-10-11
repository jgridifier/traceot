"""Scaling benchmark of the TRACE primitives (Rust core via Python).

For rows in {1e4, 1e5, 1e6, 1e7} and features in {10, 100}, each configuration runs
in a fresh subprocess so that peak memory (ru_maxrss) is attributable. One "round"
is what a cyclic booster needs per pass over the data:

1. ``sketch``  - streaming quantile sketch per feature (chunk-parallel, mergeable);
2. ``bin``     - bin every feature to uint8 codes, columnar (p, n) layout;
3. ``grad``    - pinball subgradients on K = 19 levels, M = 2 channels (loc, scale);
4. ``hist``    - per-feature histograms of both channels (one O(n p) pass).

Data are neutral simulated values generated column by column (never an n x p float
matrix). Usage::

    python benchmarks/scale_bench.py               # full grid
    python benchmarks/scale_bench.py --max-rows 1e6
Writes benchmarks/results/scale_bench.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "results" / "scale_bench.json"


def one(n: int, p: int) -> dict:
    from statistics import NormalDist

    from traceot import get_backend

    be = get_backend("rust")
    rng = np.random.default_rng(12345)
    t = {}
    codes = np.empty((p, n), dtype=np.uint8)
    y = rng.normal(size=n)
    t_sketch = t_bin = t_gen = 0.0
    for j in range(p):
        t0 = time.perf_counter()
        x = rng.normal(size=n) if j % 2 == 0 else rng.lognormal(size=n)
        if j < 3:
            y += (0.5 if j % 2 == 0 else 0.1) * x
        t_gen += time.perf_counter() - t0
        t0 = time.perf_counter()
        sk = be.QuantileSketch(2048)
        sk.update_chunked(x)
        edges = sk.bin_edges(255)
        t_sketch += time.perf_counter() - t0
        t0 = time.perf_counter()
        codes[j] = be.bin_column(x, edges)
        t_bin += time.perf_counter() - t0
        del x
    t["generate"], t["sketch"], t["bin"] = t_gen, t_sketch, t_bin
    K = 19
    u = np.arange(1, K + 1) / (K + 1)
    z = np.array([NormalDist().inv_cdf(v) for v in u])
    basis = np.column_stack([np.ones(K), z])
    loc, scale = np.zeros(n), np.ones(n)
    t0 = time.perf_counter()
    G, loss = be.pinball_grad_location_scale(y, loc, scale, z, u, basis)
    t["grad"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    sums, counts = be.histograms(codes, G, 256)
    t["hist"] = time.perf_counter() - t0
    assert int(counts[0].sum()) == n
    t["round_total"] = t["grad"] + t["hist"]
    t["setup_total"] = t["sketch"] + t["bin"]
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=float, default=1e7)
    ap.add_argument("--child", nargs=2, type=int, default=None)
    args = ap.parse_args()
    if args.child:
        n, p = args.child
        res = one(n, p)
        res["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(json.dumps(res))
        return
    results = []
    for n in (10_000, 100_000, 1_000_000, 10_000_000):
        if n > args.max_rows:
            continue
        for p in (10, 100):
            cp = subprocess.run(
                [sys.executable, __file__, "--child", str(n), str(p)],
                capture_output=True,
                text=True,
                check=True,
            )
            r = json.loads(cp.stdout.strip().splitlines()[-1])
            r.update(rows=n, features=p)
            results.append(r)
            print(
                f"n={n:>9,} p={p:>3}  setup {r['setup_total']:7.2f}s  "
                f"round {r['round_total']:7.3f}s  peak {r['peak_rss_mb']:8.0f} MB",
                flush=True,
            )
    meta = {
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "rayon_threads": os.environ.get("RAYON_NUM_THREADS", "default (all cores)"),
        "date": time.strftime("%Y-%m-%d"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"meta": meta, "results": results}, indent=2))


if __name__ == "__main__":
    main()
