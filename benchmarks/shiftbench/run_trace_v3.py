# ruff: noqa: E501, E731, B007, B023, F841
"""TRACE v3 runs (PREREG_TRACE_V3.md): H7 shape track and H8(b) coverage track, pit_rounds ON.

    python run_trace_v3.py TRACK [--workers 8] [--pilot]
TRACKs: shape, cover. Records: benchmarks/results/trace_v3_<track>.jsonl (append-only checkpoint;
a restart skips tasks already written). --pilot runs replications 7900-7909 into
/workspace/v3_logs/pilot_<track>.jsonl (timing only, discarded).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import sb_adapter_trace_claims as adp
import sb_dgp as dgp
import sb_dgp_shape as shp
from numpy.polynomial import legendre as L
from run_synthetic import COVER_B, COVER_N, COVER_P, COVER_SCEN, RES, centred, record, seed_of

PIT_ROUNDS = 20  # PREREG_TRACE_V3 §2
CFG = dict(adp.CFG, pit_rounds=PIT_ROUNDS)  # K=4, folds=5, trim=0.99 (library default), pit 20
SEED0 = 8000  # fresh replication indices start here (§3)
PILOT0 = 7900
GL_U, GL_W = L.leggauss(256)
GL_U, GL_W = (GL_U + 1) / 2, GL_W / 2

# replication counts; set in PREREG_TRACE_V3 §4 from the timing pilot
SHAPE_GRID = [(2000, 5, 500), (20000, 20, 200)]
COVER_REPS = 300


class TraceV3(adp.TraceFull):
    def raw(self, XA, yA, XB, yB, seed=0):
        import traceot

        return traceot.shift_decompose(
            XA, yA, XB, yB, seed=seed, return_model=self.return_model, **CFG
        )


TR, TRM = TraceV3(), TraceV3(return_model=True)


def run(XA, yA, XB, yB, seed, model=False):
    t0 = time.perf_counter()
    m = TRM if model else TR
    raw = m.raw(XA, yA, XB, yB, seed)
    res = m.decompose(XA, yA, XB, yB, seed, r=raw)
    return res, raw, time.perf_counter() - t0


def h5_v3(XA, yA, XB, yB, res, raw):
    """H5.1b (bar) and the marginal-vs-empirical gap (descriptive only)."""
    x = res.extra
    phi = np.asarray(x["bary_phi"])
    m = raw["model"]
    qa = m.predict_quantile(XA, GL_U, "A").mean(0)
    qb = m.predict_quantile(XB, GL_U, "B").mean(0)
    wrec = float(np.sum(GL_W * (qb - qa) ** 2))
    w2emp = float(np.mean((np.sort(yB) - np.sort(yA)) ** 2))
    mp, rp = x["marg_phi"], x["resid_phi"]
    msum = float(np.sum(mp["comp"]) + np.sum(mp["struct"]) + rp["comp"] + rp["struct"])
    return {
        "W_rec": wrec,
        "sum_phi": float(phi.sum()),
        "g51b": abs(wrec - phi.sum()),
        "W2_emp": w2emp,
        "marg_sum": msum,
        "g52_desc": abs(msum - w2emp),
    }


def t_shape(a):
    scen, n, p, rep = a
    s = shp.make(scen, p)
    rng = np.random.default_rng(seed_of("data", scen, n, p, rep))
    XA0, yA, XB0, yB = s.sample(n, rng)
    XA, XB = centred(XA0, XB0)
    res, raw, sec = run(XA, yA, XB, yB, seed_of("m", rep), model=True)
    rec = record(res)
    ex = {k: v for k, v in res.extra.items() if not k.startswith("_")}
    return {
        "scen": scen,
        "n": n,
        "p": p,
        "rep": rep,
        "TRACE": rec
        | {
            "sec": sec,
            "x": {"lsplit": ex["lsplit"], "w2": ex["w2"], "trim_mass": ex.get("trim_mass")},
        },
        "h5": h5_v3(XA, yA, XB, yB, res, raw),
        "sdA": float(np.std(yA)),
    }


def _vecs(res):
    return {"split": res.split, "lsplit": res.extra["lsplit"]}


def t_cover(a):
    scen, rep = a
    n, p = COVER_N, COVER_P
    s = shp.make(scen, p) if scen in shp.SHAPE_SCENARIOS else dgp.make(scen, p)
    rng = np.random.default_rng(seed_of("data", scen, n, p, rep))
    XA, yA, XB, yB = s.sample(n, rng)
    t0 = time.perf_counter()
    a_, b_ = centred(XA, XB)
    point, _, _ = run(a_, yA, b_, yB, seed_of("m", rep))
    brng = np.random.default_rng(seed_of("boot", scen, rep))
    boots = []
    for k in range(COVER_B):
        ia, ib = brng.integers(0, n, n), brng.integers(0, n, n)
        a_, b_ = centred(XA[ia], XB[ib])
        r, _, _ = run(a_, yA[ia], b_, yB[ib], seed_of("mb", rep, k))
        boots.append(_vecs(r))
    return {
        "scen": scen,
        "rep": rep,
        "point": _vecs(point),
        "boots": boots,
        "sec": time.perf_counter() - t0,
    }


def tasks(track, pilot):
    if track == "shape":
        if pilot:
            return [
                (s, n, p, PILOT0 + r)
                for n, p, _ in SHAPE_GRID
                for s in shp.SHAPE_SCENARIOS
                for r in range(2)
            ]
        # interleave replications so a stop leaves balanced cells
        out = []
        for r in range(max(x[2] for x in SHAPE_GRID)):
            for n, p, reps in SHAPE_GRID:
                if r < reps:
                    out += [(s, n, p, SEED0 + r) for s in shp.SHAPE_SCENARIOS]
        return out
    if track == "cover":
        sc = list(COVER_SCEN) + list(shp.SHAPE_SCENARIOS)
        if pilot:
            return [(s, PILOT0 + r) for s in sc for r in range(1)]
        return [(s, SEED0 + r) for r in range(COVER_REPS) for s in sc]
    raise SystemExit(track)


def key(rec):
    return json.dumps([rec["scen"], rec.get("n"), rec.get("p"), rec["rep"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("track")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    fn = {"shape": t_shape, "cover": t_cover}[a.track]
    todo = tasks(a.track, a.pilot)
    path = (
        Path(f"/workspace/v3_logs/pilot_{a.track}.jsonl")
        if a.pilot
        else RES / f"trace_v3_{a.track}.jsonl"
    )
    done = set()
    if path.exists() and not a.pilot:
        for line in path.read_text().splitlines():
            try:
                done.add(key(json.loads(line)))
            except json.JSONDecodeError:
                pass
    mk = lambda t: json.dumps(
        [t[0], t[1] if len(t) == 4 else None, t[2] if len(t) == 4 else None, t[-1]]
    )
    todo = [t for t in todo if mk(t) not in done]
    t0 = time.time()
    with Pool(a.workers, maxtasksperchild=10) as pool, open(path, "w" if a.pilot else "a") as f:
        for i, rec in enumerate(pool.imap(fn, todo)):
            f.write(json.dumps(adp._tolist(rec)) + "\n")
            f.flush()
            print(f"{a.track} {i + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
