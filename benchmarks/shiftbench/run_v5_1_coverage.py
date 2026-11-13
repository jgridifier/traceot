"""TRACE v5.1 coverage of a 90% bootstrap band (PREREG_V5_1_COVERAGE.md).

    python run_v5_1_coverage.py [--workers 8]

Single pass, 60-minute wall cap, B = 500, cells skew_comp / tail_comp at n = 2000,
p = 5, indices 10400-10799 (400 reps each). This module is the runner: invoking it
executes the study; importing it does not. This file's presence in a PR is not a run.

Sealed indices 9600-9619 are never run. Spent ranges listed in PREREG_V5_1_COVERAGE
§4 are in FORBIDDEN. The results file must not already exist (single-pass check).
Tasks not finished inside the cap are not imputed; a unit is decided only if the
completed replications already determine it.
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

# Freeze point: the commit that added PREREG_V5_1_COVERAGE.md (PR #46).
PREREG_SHA = "dfaaf72e775ad6065b88a1c7931033c8dedbdbd3"
SCEN = ("skew_comp", "tail_comp")
N, P, B, R = 2000, 5, 500, 400
ALLOWED = set(range(10400, 10800))  # 10400-10799 inclusive
# Every spent or reserved range in PREREG_V5_1_COVERAGE §4, plus the later
# V5.2-specified holdout, and the sealed range named explicitly.
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
    | set(range(9600, 9620))  # sealed; never run, never reused
    | set(range(9700, 9750))
    | set(range(9800, 9820))
    | set(range(10100, 10150))
    | set(range(10200, 10220))
    | set(range(11000, 11050))
    | set(range(11100, 11120))
    | set(range(81001, 81021))
    | set(range(83001, 83021))
    | set(range(84001, 84051))
)
SEALED = set(range(9600, 9620))
CAP_SEC = 60 * 60
OUT = RES / "v5_1_coverage.jsonl"
WALL = RES / "v5_1_coverage_wall.json"
TRUTH_FILE = RES / "shiftbench_truth_shape.json"


def assert_rep(rep: int) -> None:
    assert rep in ALLOWED, f"index {rep} is not in the fresh range 10400-10799"
    assert rep not in FORBIDDEN, f"index {rep} is in the FORBIDDEN set"
    assert rep not in SEALED, f"index {rep} is in the sealed range 9600-9619"


def _truth_pieces(truth: dict, scen: str) -> tuple[int, float, float]:
    t = truth[scen]
    k = t["graded"][0]
    return k, t["shapley"]["comp"][k], t["shapley"]["struct"][k]


def task(a):
    import traceot

    scen, n, p, rep, k, truth_c, truth_s = a
    assert_rep(rep)
    s = shp.make(scen, p)
    rng = np.random.default_rng(seed_of("data", scen, n, p, rep))
    XA0, yA, XB0, yB = s.sample(n, rng)
    XA, XB = centred(XA0, XB0)
    n_a, n_b = len(yA), len(yB)
    t0 = time.perf_counter()
    point = traceot.explain_v5_1(XA, yA, XB, yB)
    boots_c, boots_s = [], []
    failed_b = []
    for b in range(B):
        rng_b = np.random.default_rng(seed_of("v51cov", scen, n, p, rep, b))
        i_a = rng_b.integers(0, n_a, n_a)
        i_b = rng_b.integers(0, n_b, n_b)
        try:
            r = traceot.explain_v5_1(XA[i_a], yA[i_a], XB[i_b], yB[i_b])
        except Exception as e:  # noqa: BLE001 — failed refits are recorded, never redrawn
            failed_b.append({"b": b, "error": str(e)})
            continue
        boots_c.append(r["lsplit_comp"][k])
        boots_s.append(r["lsplit_struct"][k])
    n_fail = len(failed_b)
    n_ok = len(boots_c)
    miss_refit = n_fail > 5  # more than 1% of B=500
    band = {}
    covered = {}
    for piece, boots, truth in (
        ("comp", boots_c, truth_c),
        ("struct", boots_s, truth_s),
    ):
        if n_ok >= 1:
            lo, hi = (float(x) for x in np.quantile(boots, [0.05, 0.95], method="linear"))
        else:
            lo, hi = float("nan"), float("nan")
        band[piece] = [lo, hi]
        if miss_refit or not np.isfinite(lo):
            covered[piece] = False
        else:
            covered[piece] = bool(lo <= truth <= hi)
    sec = time.perf_counter() - t0
    return {
        "scen": scen,
        "n": n,
        "p": p,
        "rep": rep,
        "k": k,
        "truth": {"comp": truth_c, "struct": truth_s},
        "point": {
            "comp": point["lsplit_comp"][k],
            "struct": point["lsplit_struct"][k],
        },
        "band": band,
        "covered": covered,
        "n_success": n_ok,
        "n_fail": n_fail,
        "failed_b": failed_b,
        "miss_due_to_refit_failures": miss_refit,
        "sec": sec,
        "sec_core_point": point["seconds_total"],
    }


def plan():
    truth = json.loads(TRUTH_FILE.read_text())
    todo = []
    for scen in SCEN:
        k, tc, ts = _truth_pieces(truth, scen)
        for rep in range(10400, 10800):
            assert_rep(rep)
            todo.append((scen, N, P, rep, k, tc, ts))
    assert len(todo) == 2 * R
    assert ALLOWED.isdisjoint(FORBIDDEN)
    assert SEALED <= FORBIDDEN
    assert ALLOWED == set(range(10400, 10800))
    return todo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    assert not OUT.exists(), "single pass: results file already exists"
    todo = plan()
    print(
        f"v5.1 coverage: {len(todo)} tasks -> {OUT} "
        f"(cap {CAP_SEC}s, B={B}, workers={args.workers})",
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
                "tasks_preregistered": 2 * R,
                "cap_sec": CAP_SEC,
                "workers": args.workers,
                "B": B,
                "prereg_sha": PREREG_SHA,
            }
        )
        + "\n"
    )
    print(f"wall {wall:.1f}s, {done}/{len(todo)} tasks", flush=True)


if __name__ == "__main__":
    main()
