"""Run the ShiftBench synthetic suite. Raw records go to benchmarks/results/.

    python run_synthetic.py truth|accuracy|coverage|invariance|scale [--workers 6]

The design (grid, replications, seeds, metrics) is fixed by PREREG_SHIFTBENCH.md.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import sb_dgp as dgp
from sb_methods import RIVALS

HERE = Path(__file__).resolve().parent
RES = Path(os.environ.get("SHIFTBENCH_RES", HERE.parents[0] / "results"))
TRUTH = RES / "shiftbench_truth.json"

ACC_GRID = [(2000, 5, 50), (20000, 20, 20), (200000, 20, 5)]  # (n per period, p, reps)
COVER_SCEN = ("null", "mixed", "tail", "hetero", "entry_exit")
COVER_N, COVER_P, COVER_REPS, COVER_B = 2000, 5, 40, 30
INV_GRID = [(2000, 5), (20000, 20)]
INV_REPS = 5
SCALE_GRID = [
    (10_000, 10),
    (10_000, 100),
    (100_000, 10),
    (100_000, 100),
    (1_000_000, 10),
    (1_000_000, 100),
]
SCALE_TIMEOUT = 1200
SCALE_MEM_BYTES = 8 * 2**30  # address-space cap per child; exceeding it is reported


def seed_of(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode())


def centred(XA, XB):
    m = 0.5 * (XA.mean(0) + XB.mean(0))
    return XA - m, XB - m


def draw(scen, n, p, rep):
    s = dgp.make(scen, p)
    rng = np.random.default_rng(seed_of("data", scen, n, p, rep))
    XA, yA, XB, yB = s.sample(n, rng)
    XA, XB = centred(XA, XB)
    return XA, yA, XB, yB


def record(res):
    return {
        "split": res.split,
        "feat_comp": None if res.feat_comp is None else np.asarray(res.feat_comp).tolist(),
        "feat_struct": None if res.feat_struct is None else np.asarray(res.feat_struct).tolist(),
        "intercept": res.intercept,
    }


def run_all(XA, yA, XB, yB, seed):
    out = {}
    for m in RIVALS:
        t0 = time.perf_counter()
        r = m.decompose(XA, yA, XB, yB, seed=seed)
        out[m.name] = record(r) | {"sec": time.perf_counter() - t0}
    return out


# ---------------------------------------------------------------- tracks
def task_truth(_):
    return {s: dgp.make(s, 5).truth() for s in dgp.SCENARIOS}


def task_acc(args):
    scen, n, p, rep = args
    XA, yA, XB, yB = draw(scen, n, p, rep)
    return {
        "scen": scen,
        "n": n,
        "p": p,
        "rep": rep,
        "methods": run_all(XA, yA, XB, yB, seed_of("m", rep)),
    }


def task_cover(args):
    scen, rep = args
    n, p = COVER_N, COVER_P
    s = dgp.make(scen, p)
    rng = np.random.default_rng(seed_of("data", scen, n, p, rep))
    XA, yA, XB, yB = s.sample(n, rng)
    a, b = centred(XA, XB)
    point = run_all(a, yA, b, yB, seed_of("m", rep))
    brng = np.random.default_rng(seed_of("boot", scen, rep))
    boots = []
    for k in range(COVER_B):
        ia, ib = brng.integers(0, n, n), brng.integers(0, n, n)
        a, b = centred(XA[ia], XB[ib])
        boots.append(run_all(a, yA[ia], b, yB[ib], seed_of("mb", rep, k)))
    return {"scen": scen, "rep": rep, "point": point, "boots": boots}


def task_inv(args):
    scen, n, p, rep = args
    XA, yA, XB, yB = draw(scen, n, p, rep)
    seed = seed_of("m", rep)
    base = run_all(XA, yA, XB, yB, seed)
    rng = np.random.default_rng(seed_of("perm", scen, n, p, rep))
    ra, rb, cp = rng.permutation(n), rng.permutation(n), rng.permutation(p)
    perm = run_all(XA[ra][:, cp], yA[ra], XB[rb][:, cp], yB[rb], seed)
    inv = np.argsort(cp)
    for m in perm.values():  # map per-feature results back to the original columns
        for k in ("feat_comp", "feat_struct"):
            if m[k] is not None:
                m[k] = np.asarray(m[k])[inv].tolist()
    return {"scen": scen, "n": n, "p": p, "rep": rep, "base": base, "perm": perm}


def scale_child(name, n, p):
    """Runs in a fresh process: one method, one size; prints a JSON line."""
    XA, yA, XB, yB = draw("mixed", n, p, 0)
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    m = next(r for r in RIVALS if r.name == name)
    if hasattr(m, "n_jobs"):
        m.n_jobs = 4
    t0 = time.perf_counter()
    m.decompose(XA, yA, XB, yB, seed=0)
    sec = time.perf_counter() - t0
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    data_mb = 2 * (XA.nbytes + yA.nbytes) / 2**20
    print(
        json.dumps(
            {
                "method": name,
                "n": n,
                "p": p,
                "sec": sec,
                "peak_rss_mb": rss1 / 1024,
                "data_mb": data_mb,
                "extra_mb": max(rss1 - rss0, 0) / 1024,
            }
        )
    )


def _cap_memory():
    resource.setrlimit(resource.RLIMIT_AS, (SCALE_MEM_BYTES, SCALE_MEM_BYTES))


def run_scale():
    out = []
    for n, p in SCALE_GRID:
        for m in RIVALS:
            cmd = [sys.executable, __file__, "scale-child", m.name, str(n), str(p)]
            env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
            try:
                r = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=SCALE_TIMEOUT,
                    env=env,
                    cwd=HERE,
                    preexec_fn=_cap_memory,
                )
                if r.returncode == 0:
                    out.append(json.loads(r.stdout.strip().splitlines()[-1]))
                else:
                    out.append(
                        {
                            "method": m.name,
                            "n": n,
                            "p": p,
                            "status": "error",
                            "stderr": r.stderr[-400:],
                        }
                    )
            except subprocess.TimeoutExpired:
                out.append({"method": m.name, "n": n, "p": p, "status": "timeout"})
            print(out[-1], flush=True)
            (RES / "shiftbench_scale.json").write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("track")
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if a.track == "scale-child":
        scale_child(a.rest[0], int(a.rest[1]), int(a.rest[2]))
        return
    RES.mkdir(exist_ok=True)
    if a.track == "truth":
        TRUTH.write_text(json.dumps(task_truth(None), indent=1))
        return
    if a.track == "scale":
        run_scale()
        return
    if a.track == "accuracy":
        tasks = [
            (s, n, p, r) for n, p, reps in ACC_GRID for s in dgp.SCENARIOS for r in range(reps)
        ]
        fn, path = task_acc, RES / "shiftbench_accuracy.jsonl"
    elif a.track == "coverage":
        tasks = [(s, r) for s in COVER_SCEN for r in range(COVER_REPS)]
        fn, path = task_cover, RES / "shiftbench_coverage.jsonl"
    elif a.track == "invariance":
        tasks = [(s, n, p, r) for n, p in INV_GRID for s in dgp.SCENARIOS for r in range(INV_REPS)]
        fn, path = task_inv, RES / "shiftbench_invariance.jsonl"
    else:
        raise SystemExit(a.track)
    if a.rest:  # optional cap for smoke tests: first k tasks only
        tasks = tasks[: int(a.rest[0])]
    with Pool(a.workers) as pool, open(path, "w") as f:
        for i, rec in enumerate(pool.imap_unordered(fn, tasks)):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{a.track} {i + 1}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
