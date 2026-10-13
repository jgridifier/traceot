"""ShiftBench real-data track (no ground truth): stability and split-half agreement.

    python run_real.py [--workers 6]

Input: data/real_panel.csv.gz (columns unit, period in {A, B}, y, x1..xk), a derived,
anonymised public quarterly financial panel; see data/README.md.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from run_synthetic import RES, centred, run_all, seed_of

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "real_panel.csv.gz"
N_BOOT = 100
N_SPLIT = 50


def load():
    d = pd.read_csv(DATA)
    xs = [c for c in d.columns if c.startswith("x")]
    return d, xs


def arrays(d, xs, units=None):
    if units is not None:  # cluster resample: each unit keeps all its rows
        d = pd.concat([d[d.unit == u] for u in units], ignore_index=True)
    A, B = d[d.period == "A"], d[d.period == "B"]
    XA, XB = centred(A[xs].to_numpy(float), B[xs].to_numpy(float))
    return XA, A.y.to_numpy(float), XB, B.y.to_numpy(float)


def task(args):
    kind, k = args
    d, xs = load()
    units = np.sort(d.unit.unique())
    rng = np.random.default_rng(seed_of("real", kind, k))
    if kind == "full":
        return {"kind": kind, "k": k, "res": run_all(*arrays(d, xs), seed_of("m", 0))}
    if kind == "boot":
        u = rng.choice(units, len(units), replace=True)
        return {"kind": kind, "k": k, "res": run_all(*arrays(d, xs, u), seed_of("m", k))}
    perm = rng.permutation(units)
    h1, h2 = perm[: len(units) // 2], perm[len(units) // 2 :]
    r1 = run_all(*arrays(d[d.unit.isin(h1)], xs), seed_of("m", k))
    r2 = run_all(*arrays(d[d.unit.isin(h2)], xs), seed_of("m", k))
    return {"kind": kind, "k": k, "h1": r1, "h2": r2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    tasks = [("full", 0)] + [("boot", k) for k in range(N_BOOT)]
    tasks += [("split", k) for k in range(N_SPLIT)]
    d, _ = load()
    meta = {
        "sd_A": float(d[d.period == "A"].y.std(ddof=0)),
        "n_A": int((d.period == "A").sum()),
        "n_B": int((d.period == "B").sum()),
        "units": int(d.unit.nunique()),
    }
    path = RES / "shiftbench_real.jsonl"
    with Pool(a.workers) as pool, open(path, "w") as f:
        f.write(json.dumps({"kind": "meta"} | meta) + "\n")
        for i, rec in enumerate(pool.imap_unordered(task, tasks)):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"real {i + 1}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
