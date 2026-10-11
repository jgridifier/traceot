"""Diagnostic for the Table 2 mismatch: how do the SPD MISEs react to the noise level?

Not part of the reproduction table (which uses the paper's stated sigma = 0.01). This
only helps explain *why* our Table 2 levels differ from the paper's; see
docs/REPRODUCTION_NOTES.md, Section 4. Writes benchmarks/results/sensitivity_spd.json.

    python benchmarks/adopt_paper/sensitivity_spd.py --reps 20
"""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import get_context
from pathlib import Path

import numpy as np

os.environ.setdefault("RAYON_NUM_THREADS", "1")

import dgp  # noqa: E402
from traceot import ADOPTRegressor, FrechetMeanRegressor, GlobalFrechetRegressor  # noqa: E402
from traceot.primitives import coords_to_spd  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "sensitivity_spd.json"
CASE_ID = {"I": 1, "II": 2}


def run(task):
    case, m, n, sigma, rep = task
    rng = np.random.default_rng(np.random.SeedSequence([9, CASE_ID[case], n, m, rep]))
    means = dgp.spd_population_means(case, m)
    xg = dgp.mise_grid(10)
    truth = dgp.spd_true_coords(case, xg, m, means)
    X, Z = dgp.simulate_spd(case, n, m, rng, sigma=sigma, means=means)
    S = coords_to_spd(Z, m)
    out = {}
    for name, est in {
        "ADOPT": ADOPTRegressor(space="logcholesky", random_state=rep),
        "GF": GlobalFrechetRegressor(space="logcholesky"),
        "FM": FrechetMeanRegressor(space="logcholesky"),
    }.items():
        pred = est.fit(X, S).predict_encoded(xg)
        out[name] = float(np.mean(np.sum((pred - truth) ** 2, axis=1)))
    return task, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    tasks = [
        (c, m, n, s, r)
        for c in ("I", "II")
        for m in (10, 20)
        for n in (100, 300)
        for s in (0.01, 0.05, 0.1)
        for r in range(a.reps)
    ]
    with get_context("fork").Pool(a.workers) as pool:
        res = pool.map(run, tasks)
    agg: dict = {}
    for (c, m, n, s, _), out in res:
        for k, v in out.items():
            agg.setdefault(f"{c}/m={m}/n={n}/sigma={s}", {}).setdefault(k, []).append(v)
    summary = {
        key: {k: round(100 * float(np.mean(v)), 4) for k, v in d.items()} for key, d in agg.items()
    }
    OUT.write_text(json.dumps({"reps": a.reps, "scale": "x1e-2", "mise": summary}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
