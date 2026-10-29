"""TRACE v5 first gate (PREREG_V5.md, frozen at bce021e). Resumable jsonl.

    python run_v5_gate1.py [--workers 8]

Runs ONLY the n=2000 dev indices 9500-9549 on skew_comp / tail_comp with paired
DFL / CFM rivals. The n=20000 indices 9600-9619 are deliberately not run here
(held back by the owner; those cells are reported INCOMPLETE).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import sb_dgp_shape as shp
from run_synthetic import RES, centred, seed_of
from sb_shape_readout import CFM_L, DFL_L

PREREG_SHA = "bce021e"
SCEN = ("skew_comp", "tail_comp")
DEV_N, DEV_P = 2000, 5
DEV_REPS = list(range(9500, 9550))  # PREREG_V5 §3, n=2000
HELD_BACK = list(range(9600, 9620))  # PREREG_V5 §3, n=20000 — NOT RUN
OUT = RES / "v5_gate1_shape.jsonl"


def task(a):
    import traceot

    scen, rep = a
    n, p = DEV_N, DEV_P
    assert rep not in HELD_BACK
    s = shp.make(scen, p)
    rng = np.random.default_rng(seed_of("data", scen, n, p, rep))
    XA0, yA, XB0, yB = s.sample(n, rng)
    XA, XB = centred(XA0, XB0)
    seed = seed_of("m", rep)
    t0 = time.perf_counter()
    r = traceot.explain_v5(XA, yA, XB, yB)
    sec = time.perf_counter() - t0
    rivals = {}
    for name, meth in (("DFL", DFL_L()), ("CFM", CFM_L())):
        t1 = time.perf_counter()
        try:
            rd = meth.decompose(XA, yA, XB, yB, seed=seed)
            rivals[name] = {"lsplit": rd.extra.get("lsplit"), "sec": time.perf_counter() - t1}
        except Exception as e:  # noqa: BLE001 — keep failure on record
            rivals[name] = {"error": str(e), "sec": time.perf_counter() - t1}
    return {
        "scen": scen,
        "n": n,
        "p": p,
        "rep": rep,
        "TRACE_V5": {
            "lsplit": {"comp": list(r["lsplit_comp"]), "struct": list(r["lsplit_struct"])},
            "mixing_residual": {h: r[h]["mixing_residual"] for h in ("aa", "ab", "ba", "bb")},
            "sec": sec,
            "sec_core": r["seconds_total"],
        },
        "rivals": rivals,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    have = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                have.add((rec["scen"], rec["rep"]))
    todo = [(s, r) for r in DEV_REPS for s in SCEN if (s, r) not in have]
    print(f"gate1: {len(todo)} tasks -> {OUT}", flush=True)
    t0 = time.perf_counter()
    with OUT.open("a") as fh, Pool(args.workers) as pool:
        for rec in pool.imap_unordered(task, todo):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
    print(f"wall {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
