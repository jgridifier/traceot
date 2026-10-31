"""TRACE v5.1 gate (PREREG_V5_1.md, frozen at b33f507). Single pass, 30-minute wall cap.

    python run_v5_1_gate.py [--workers 8]

Four cells: skew_comp / tail_comp at n=2000, p=5 (indices 9700-9749) and n=20000,
p=20 (indices 9800-9819), with paired DFL / CFM rivals. Spent / sealed indices
9500-9549 and 9600-9619 are never run. Tasks not finished inside the cap are not
imputed; their cell is INCOMPLETE.
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

PREREG_SHA = "b33f507"
SCEN = ("skew_comp", "tail_comp")
SIZES = {2000: (5, list(range(9700, 9750))), 20000: (20, list(range(9800, 9820)))}
FORBIDDEN = set(range(9500, 9550)) | set(range(9600, 9620))
CAP_SEC = 30 * 60
OUT = RES / "v5_1_gate_shape.jsonl"


def task(a):
    import traceot

    scen, n, rep = a
    p = SIZES[n][0]
    assert rep not in FORBIDDEN
    s = shp.make(scen, p)
    rng = np.random.default_rng(seed_of("data", scen, n, p, rep))
    XA0, yA, XB0, yB = s.sample(n, rng)
    XA, XB = centred(XA0, XB0)
    seed = seed_of("m", rep)
    t0 = time.perf_counter()
    r = traceot.explain_v5_1(XA, yA, XB, yB)
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
        "TRACE_V5_1": {
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
    assert not OUT.exists(), "single pass: results file already exists"
    # large cells first so the cap binds on the expensive work, not on stragglers
    todo = [(s, n, r) for n in (20000, 2000) for r in SIZES[n][1] for s in SCEN]
    print(f"v5.1 gate: {len(todo)} tasks -> {OUT} (cap {CAP_SEC}s)", flush=True)
    t0 = time.perf_counter()
    done = 0
    pool = Pool(args.workers)
    try:
        with OUT.open("a") as fh:
            it = pool.imap_unordered(task, todo)
            while done < len(todo):
                left = CAP_SEC - (time.perf_counter() - t0)
                if left <= 0:
                    break
                try:
                    rec = it.next(timeout=left)
                except Exception as e:  # noqa: BLE001
                    if type(e).__name__ == "TimeoutError":
                        break
                    raise
                rec["wall_at_finish"] = time.perf_counter() - t0
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                done += 1
    finally:
        pool.terminate()
    wall = time.perf_counter() - t0
    (RES / "v5_1_gate_wall.json").write_text(
        json.dumps(
            {
                "wall_sec": wall,
                "tasks_done": done,
                "tasks_total": len(todo),
                "cap_sec": CAP_SEC,
                "workers": args.workers,
            }
        )
        + "\n"
    )
    print(f"wall {wall:.1f}s, {done}/{len(todo)} tasks", flush=True)


if __name__ == "__main__":
    main()
