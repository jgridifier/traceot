"""TRACE v5.2 specified-family gate (PREREG_V5_2_SPECIFIED.md). Single pass, 30-minute cap.

    python run_v5_2_specified_gate.py [--workers 8]

Four cells: skew_comp / tail_comp at n=2000, p=5 (indices 11000-11049) and n=20000,
p=20 (indices 11100-11119), kappa=1, with paired DFL / CFM rivals. This module is
the runner: invoking it executes the study; importing it does not. This file's
presence in a PR is not a run.

Sealed indices 9600-9619 and reserved coverage indices 10400-10799 are never run.
A scenario whose sigma_B calibration is infeasible has no tasks and its cells are
INCOMPLETE; H is then not decided and gate seeds are not touched. Tasks not
finished inside the cap are not imputed.

The method under test receives raw (uncentred) draws: PREREG_V5_2_SPECIFIED §2
requires the design, including x_3^2 and sin(x_5), to be computed on the raw scale
of the simulator (sin is not shift-invariant). Rivals receive the same centred
arrays as in the V5.1 / V5.2 gates, so their settings are unchanged.
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
import sb_dgp_shape_v5_2_specified as sp
from run_synthetic import RES, centred, seed_of
from sb_shape_readout import CFM_L, DFL_L

# Freeze point: the merge commit that added PREREG_V5_2_SPECIFIED.md to main (PR #54).
PREREG_SHA = "cc8c1d7afc9c58a8b97b8b3b28b6ad6fd1704920"
SCEN = ("skew_comp", "tail_comp")
KAPPA = 1.0
SIZES = {2000: (5, list(range(11000, 11050))), 20000: (20, list(range(11100, 11120)))}
ALLOWED = set(range(11000, 11050)) | set(range(11100, 11120))
FORBIDDEN = (
    set(range(0, 2000))
    | set(range(2101, 2120))
    | set(range(3101, 3121))
    | set(range(5000, 6500))
    | set(range(7007, 7013))
    | set(range(7900, 7910))
    | set(range(8000, 8500))
    | set(range(9100, 9150))
    | set(range(9200, 9220))
    | set(range(9300, 9380))
    | set(range(9400, 9430))
    | set(range(9500, 9550))
    | set(range(9600, 9620))  # sealed
    | set(range(9700, 9750))
    | set(range(9800, 9820))
    | set(range(10100, 10150))
    | set(range(10200, 10220))
    | set(range(10220, 10400))  # buffer after v5.2
    | set(range(10400, 10800))  # reserved for V5.1 coverage; not used
    | set(range(10800, 11000))
    | set(range(81001, 81021))
    | set(range(83001, 83021))
    | set(range(84001, 84051))
)
SEALED = set(range(9600, 9620))
COVERAGE_RESERVED = set(range(10400, 10800))
CAP_SEC = 30 * 60
OUT = RES / "v5_2_specified_gate_shape.jsonl"
WALL = RES / "v5_2_specified_gate_wall.json"


def assert_rep(rep: int, n: int) -> None:
    assert rep in ALLOWED, f"index {rep} is not in 11000-11049 or 11100-11119"
    assert rep in SIZES[n][1], f"index {rep} is not in the range for n={n}"
    assert rep not in FORBIDDEN, f"index {rep} is in the FORBIDDEN set"
    assert rep not in SEALED, f"index {rep} is in the sealed range 9600-9619"
    assert rep not in COVERAGE_RESERVED, f"index {rep} is reserved for V5.1 coverage"


def task(a):
    import traceot

    scen, kappa, n, rep = a
    p = SIZES[n][0]
    assert_rep(rep, n)
    s = sp.make(scen, kappa, p)
    skey = sp.key(scen, kappa)
    rng = np.random.default_rng(seed_of("data", skey, n, p, rep))
    XA0, yA, XB0, yB = s.sample(n, rng)
    # Method under test: raw draws (PREREG_V5_2_SPECIFIED §2, no column centering).
    seed = seed_of("m", rep)
    out = {"scen": scen, "kappa": kappa, "n": n, "p": p, "rep": rep, "skey": skey}
    t0 = time.perf_counter()
    r = traceot.explain_v5_2_specified(XA0, yA, XB0, yB)
    out["TRACE_V5_2_SPECIFIED"] = {
        "lsplit": {"comp": list(r["lsplit_comp"]), "struct": list(r["lsplit_struct"])},
        "dropped_cols_a": r["dropped_cols_a"],
        "dropped_cols_b": r["dropped_cols_b"],
        "sec": time.perf_counter() - t0,
        "sec_core": r["seconds_total"],
    }
    # Rivals: same centred arrays and settings as the V5.1 / V5.2 gates.
    XA, XB = centred(XA0, XB0)
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
    assert ALLOWED.isdisjoint(FORBIDDEN)
    assert SEALED <= FORBIDDEN
    assert COVERAGE_RESERVED <= FORBIDDEN
    for n in (20000, 2000):  # large cells first so the cap binds on the expensive work
        for rep in SIZES[n][1]:
            assert_rep(rep, n)
            for scen in SCEN:
                if sp.load_sigma(scen, KAPPA) is None:
                    skipped.append(sp.key(scen, KAPPA))
                    continue
                todo.append((scen, KAPPA, n, rep))
    return todo, sorted(set(skipped))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    assert not OUT.exists(), "single pass: results file already exists"
    todo, skipped = plan()
    print(
        f"v5.2 specified gate: {len(todo)} tasks -> {OUT} (cap {CAP_SEC}s); "
        f"cells with infeasible calibration (INCOMPLETE, no tasks): {skipped}",
        flush=True,
    )
    if skipped:
        # Calibration-INCOMPLETE stop: H is not decided; gate seeds are not touched.
        WALL.write_text(
            json.dumps(
                {
                    "wall_sec": 0.0,
                    "tasks_done": 0,
                    "tasks_total": 0,
                    "tasks_preregistered": 140,
                    "cells_infeasible": skipped,
                    "stop": "calibration_incomplete",
                    "cap_sec": CAP_SEC,
                    "workers": args.workers,
                    "prereg_sha": PREREG_SHA,
                }
            )
            + "\n"
        )
        print(
            "calibration INCOMPLETE: not running any gate index; H is not decided",
            flush=True,
        )
        return
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
                "tasks_preregistered": 140,
                "cells_infeasible": skipped,
                "cap_sec": CAP_SEC,
                "workers": args.workers,
                "prereg_sha": PREREG_SHA,
            }
        )
        + "\n"
    )
    print(f"wall {wall:.1f}s, {done}/{len(todo)} tasks", flush=True)


if __name__ == "__main__":
    main()
