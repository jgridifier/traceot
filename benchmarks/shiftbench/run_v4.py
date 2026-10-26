# ruff: noqa: E501
"""TRACE v4 scored suite (PREREG_V4.md). Resumable jsonl under benchmarks/results/v4_*.jsonl.

    python run_v4.py TRACK [--workers 8] [--pilot]
TRACKs: shape, cover, shiftbench, speed, all
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
import sb_dgp as dgp
import sb_dgp_shape as shp
from run_synthetic import COVER_N, COVER_P, COVER_SCEN, RES, centred, seed_of
from sb_adapter_trace_v4 import TraceV4Method
from sb_shape_readout import CFM_L, DFL_L

SHAPE_SCEN = shp.SHAPE_SCENARIOS
# PREREG_V4 §3
H7_N2K = list(range(9100, 9150))  # 50
H7_N20K = list(range(9200, 9220))  # 20
COVER_REPS = list(range(9300, 9380))  # 80
SB_REPS = list(range(9400, 9430))  # 30
SHAPE_GRID = [(2000, 5, H7_N2K), (20000, 20, H7_N20K)]

V4 = TraceV4Method(k_channels=4, n_boot=2000)


def _jsonify(obj):
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def run_v4(XA, yA, XB, yB, seed):
    t0 = time.perf_counter()
    res = V4.decompose(XA, yA, XB, yB, seed=seed)
    return res, time.perf_counter() - t0


def t_shape(a):
    scen, n, p, rep = a
    s = shp.make(scen, p)
    rng = np.random.default_rng(seed_of("data", scen, n, p, rep))
    XA0, yA, XB0, yB = s.sample(n, rng)
    XA, XB = centred(XA0, XB0)
    seed = seed_of("m", rep)
    res, sec = run_v4(XA, yA, XB, yB, seed)
    rivals = {}
    for name, meth in (("DFL", DFL_L()), ("CFM", CFM_L())):
        t1 = time.perf_counter()
        try:
            rd = meth.decompose(XA, yA, XB, yB, seed=seed)
            rivals[name] = {
                "split": {k: (float(a), float(b)) for k, (a, b) in rd.split.items()},
                "sec": time.perf_counter() - t1,
            }
            if rd.extra:
                rivals[name]["lsplit"] = rd.extra.get("lsplit")
        except Exception as e:  # noqa: BLE001 — keep failure on record
            rivals[name] = {"error": str(e), "sec": time.perf_counter() - t1}
    return _jsonify(
        {
            "scen": scen,
            "n": n,
            "p": p,
            "rep": rep,
            "TRACE_V4": {
                "split": {k: (float(a), float(b)) for k, (a, b) in res.split.items()},
                "feat_comp": None if res.feat_comp is None else res.feat_comp.tolist(),
                "feat_struct": None if res.feat_struct is None else res.feat_struct.tolist(),
                "lsplit": res.extra.get("lsplit"),
                "budget": res.extra.get("budget"),
                "budget_ci": res.extra.get("budget_ci"),
                "sec": sec,
            },
            "rivals": rivals,
            "sdA": float(np.std(yA)),
        }
    )


def t_cover(a):
    scen, rep = a
    n, p = COVER_N, COVER_P
    s = shp.make(scen, p) if scen in shp.SHAPE_SCENARIOS else dgp.make(scen, p)
    rng = np.random.default_rng(seed_of("data", scen, n, p, rep))
    XA, yA, XB, yB = s.sample(n, rng)
    a_, b_ = centred(XA, XB)
    t0 = time.perf_counter()
    point, _ = run_v4(a_, yA, b_, yB, seed_of("m", rep))
    return _jsonify(
        {
            "scen": scen,
            "rep": rep,
            "point": {
                "split": {k: (float(a), float(b)) for k, (a, b) in point.split.items()},
                "lsplit": point.extra.get("lsplit"),
                "budget": point.extra.get("budget"),
                "budget_ci": point.extra.get("budget_ci"),
            },
            "sec": time.perf_counter() - t0,
        }
    )


def t_shiftbench(a):
    scen, rep = a
    n, p = 20000, 20
    s = dgp.make(scen, p)
    rng = np.random.default_rng(seed_of("data", scen, n, p, rep))
    XA0, yA, XB0, yB = s.sample(n, rng)
    XA, XB = centred(XA0, XB0)
    seed = seed_of("m", rep)
    res, sec = run_v4(XA, yA, XB, yB, seed)
    return {
        "scen": scen,
        "n": n,
        "p": p,
        "rep": rep,
        "TRACE_V4": {
            "split": {k: (float(a), float(b)) for k, (a, b) in res.split.items()},
            "feat_comp": None if res.feat_comp is None else res.feat_comp.tolist(),
            "feat_struct": None if res.feat_struct is None else res.feat_struct.tolist(),
            "sec": sec,
        },
        "sdA": float(np.std(yA)),
    }


def t_speed(a):
    n, rep = a
    p = 20
    s = dgp.make("mixed", p)
    rng = np.random.default_rng(seed_of("data", "speed", n, p, rep))
    XA0, yA, XB0, yB = s.sample(n, rng)
    XA, XB = centred(XA0, XB0)
    seed = seed_of("m", rep)
    _, sec_v4 = run_v4(XA, yA, XB, yB, seed)
    t1 = time.perf_counter()
    try:
        CFM_L().decompose(XA, yA, XB, yB, seed=seed)
        sec_cfm = time.perf_counter() - t1
        err = None
    except Exception as e:  # noqa: BLE001
        sec_cfm = time.perf_counter() - t1
        err = str(e)
    return {"n": n, "p": p, "rep": rep, "sec_v4": sec_v4, "sec_cfm": sec_cfm, "cfm_error": err}


def tasks(track, pilot):
    if track == "shape":
        out = []
        if pilot:
            for n, p, reps in SHAPE_GRID:
                for s in SHAPE_SCEN:
                    out.append((s, n, p, reps[0]))
            return out
        # interleave
        max_r = max(len(r) for _, _, r in SHAPE_GRID)
        for i in range(max_r):
            for n, p, reps in SHAPE_GRID:
                if i < len(reps):
                    out += [(s, n, p, reps[i]) for s in SHAPE_SCEN]
        return out
    if track == "cover":
        scens = list(COVER_SCEN) + [s for s in SHAPE_SCEN if s not in COVER_SCEN]
        reps = COVER_REPS[:2] if pilot else COVER_REPS
        return [(s, r) for r in reps for s in scens]
    if track == "shiftbench":
        scens = ["mixed", "tail", "hetero"]
        reps = SB_REPS[:2] if pilot else SB_REPS
        return [(s, r) for r in reps for s in scens]
    if track == "speed":
        reps = [9400, 9401, 9402] if not pilot else [9400]
        return [(n, r) for n in (100_000, 1_000_000) for r in reps]
    raise SystemExit(f"unknown track {track}")


def out_path(track, pilot):
    RES.mkdir(parents=True, exist_ok=True)
    if pilot:
        return RES / f"v4_pilot_{track}.jsonl"
    return RES / f"v4_{track}.jsonl"


def done_keys(path: Path, track: str):
    keys = set()
    if not path.exists():
        return keys
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if track == "shape":
            keys.add((r["scen"], r["n"], r["p"], r["rep"]))
        elif track in ("cover", "shiftbench"):
            keys.add((r["scen"], r["rep"]))
        elif track == "speed":
            keys.add((r["n"], r["rep"]))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("track", choices=["shape", "cover", "shiftbench", "speed", "all"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    tracks = ["shape", "cover", "shiftbench", "speed"] if args.track == "all" else [args.track]
    fn = {"shape": t_shape, "cover": t_cover, "shiftbench": t_shiftbench, "speed": t_speed}
    for track in tracks:
        path = out_path(track, args.pilot)
        todo = tasks(track, args.pilot)
        have = done_keys(path, track)
        if track == "shape":
            todo = [t for t in todo if (t[0], t[1], t[2], t[3]) not in have]
        elif track in ("cover", "shiftbench"):
            todo = [t for t in todo if (t[0], t[1]) not in have]
        else:
            todo = [t for t in todo if (t[0], t[1]) not in have]
        print(f"{track}: {len(todo)} tasks -> {path}", flush=True)
        if not todo:
            continue
        with path.open("a") as fh, Pool(args.workers) as pool:
            for rec in pool.imap_unordered(fn[track], todo):
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                tag = rec.get("scen", rec.get("n"))
                print(f"  done {tag} rep={rec.get('rep')}", flush=True)


if __name__ == "__main__":
    main()
