"""TRACE v5.2 gate (PREREG_V5_2.md, frozen at df5aff8). Single pass, 30-minute wall cap.

    python run_v5_2_gate.py [--workers 8]

16 cells (PREREG_V5_2 §4): curved composition (kappa 0.5, 1.0), linear control
(kappa 0) and structure only (kappa 1.0), at n=2000, p=5 (indices 10100-10149) and
n=20000, p=20 (indices 10200-10219), with paired DFL / CFM rivals and the secondary
TRACE_V5_1 (OLS) on the curved and structure cells. Spent / sealed / used indices
9500-9549, 9600-9619, 9700-9749, 9800-9819 are never run. A (scenario, kappa) whose
sigma_B calibration is infeasible has no tasks and its cells are INCOMPLETE. Tasks
not finished inside the cap are not imputed.
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
import sb_dgp_shape_v5_2 as cv
from run_synthetic import RES, centred, seed_of
from sb_shape_readout import CFM_L, DFL_L

PREREG_SHA = "df5aff8"
SIZES = {2000: (5, list(range(10100, 10150))), 20000: (20, list(range(10200, 10220)))}
FORBIDDEN = (
    set(range(9500, 9550))
    | set(range(9600, 9620))
    | set(range(9700, 9750))
    | set(range(9800, 9820))
)
# (cell #, family, scenario, kappa, n) exactly as the table in PREREG_V5_2 §4
CELLS = []
for _n, _base in ((2000, 1), (20000, 5)):
    for _i, (_s, _k) in enumerate(
        (("skew_comp", 0.5), ("tail_comp", 0.5), ("skew_comp", 1.0), ("tail_comp", 1.0))
    ):
        CELLS.append((_base + _i, "curved", _s, _k, _n))
for _n, _base in ((2000, 9), (20000, 11)):
    for _i, _s in enumerate(("skew_comp", "tail_comp")):
        CELLS.append((_base + _i, "control", _s, 0.0, _n))
for _n, _base in ((2000, 13), (20000, 15)):
    for _i, _s in enumerate(("skew_struct", "tail_struct")):
        CELLS.append((_base + _i, "struct", _s, 1.0, _n))
CELLS.sort()
CAP_SEC = 30 * 60
OUT = RES / "v5_2_gate_shape.jsonl"
WALL = RES / "v5_2_gate_wall.json"


def task(a):
    import traceot

    cell, fam, scen, kappa, n, rep = a
    p = SIZES[n][0]
    assert rep not in FORBIDDEN and rep in SIZES[n][1]
    s = cv.make(scen, kappa, p)
    skey = cv.key(scen, kappa)
    rng = np.random.default_rng(seed_of("data", skey, n, p, rep))
    XA0, yA, XB0, yB = s.sample(n, rng)
    XA, XB = centred(XA0, XB0)
    seed = seed_of("m", rep)
    out = {"cell": cell, "family": fam, "scen": scen, "kappa": kappa, "n": n, "p": p, "rep": rep}
    t0 = time.perf_counter()
    r = traceot.explain_v5_2(XA, yA, XB, yB)
    out["TRACE_V5_2"] = {
        "lsplit": {"comp": list(r["lsplit_comp"]), "struct": list(r["lsplit_struct"])},
        "dropped_cols_a": r["dropped_cols_a"],
        "dropped_cols_b": r["dropped_cols_b"],
        "sec": time.perf_counter() - t0,
        "sec_core": r["seconds_total"],
    }
    if fam != "control":  # secondary, descriptive only (PREREG_V5_2 §4)
        t0 = time.perf_counter()
        r1 = traceot.explain_v5_1(XA, yA, XB, yB)
        out["TRACE_V5_1"] = {
            "lsplit": {"comp": list(r1["lsplit_comp"]), "struct": list(r1["lsplit_struct"])},
            "sec": time.perf_counter() - t0,
        }
    rivals = {}
    for name, meth in (("DFL", DFL_L()), ("CFM", CFM_L())):
        t1 = time.perf_counter()
        try:
            rd = meth.decompose(XA, yA, XB, yB, seed=seed)
            rivals[name] = {"lsplit": rd.extra.get("lsplit"), "sec": time.perf_counter() - t1}
        except Exception as e:  # noqa: BLE001 — keep failure on record
            rivals[name] = {"error": str(e), "sec": time.perf_counter() - t1}
    out["rivals"] = rivals
    return out


def plan():
    todo, skipped = [], []
    for n in (20000, 2000):  # large cells first so the cap binds on the expensive work
        for rep in SIZES[n][1]:
            for cell, fam, scen, kappa, nn in CELLS:
                if nn != n:
                    continue
                if cv.load_sigma(scen, kappa) is None:
                    skipped.append(cell)
                    continue
                todo.append((cell, fam, scen, kappa, n, rep))
    return todo, sorted(set(skipped))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    assert not OUT.exists(), "single pass: results file already exists"
    todo, skipped = plan()
    print(
        f"v5.2 gate: {len(todo)} tasks -> {OUT} (cap {CAP_SEC}s); "
        f"cells with infeasible calibration (INCOMPLETE, no tasks): {skipped}",
        flush=True,
    )
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
    WALL.write_text(
        json.dumps(
            {
                "wall_sec": wall,
                "tasks_done": done,
                "tasks_total": len(todo),
                "tasks_preregistered": 560,
                "cells_infeasible": skipped,
                "cap_sec": CAP_SEC,
                "workers": args.workers,
            }
        )
        + "\n"
    )
    print(f"wall {wall:.1f}s, {done}/{len(todo)} tasks", flush=True)


if __name__ == "__main__":
    main()
