# ruff: noqa: E501, E731, B007, B023, F841
"""TRACE-claims runs: frozen ShiftBench tracks for TRACE + H5-H8 (PREREG_TRACE_CLAIMS.md + AMENDMENT_1).

    python run_trace_claims.py TRACK [--workers 7] [--limit k]
TRACKs: truth_of, acc, shape, inv, cover, null200, real, placebo, h6syn, scale, scale-child
Raw records: benchmarks/results/trace_claims_<track>.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import sb_dgp as dgp
import sb_dgp_shape as shp
from numpy.polynomial import legendre as L
from run_synthetic import (
    ACC_GRID,
    COVER_B,
    COVER_N,
    COVER_P,
    COVER_REPS,
    COVER_SCEN,
    INV_GRID,
    INV_REPS,
    RES,
    SCALE_GRID,
    SCALE_MEM_BYTES,
    SCALE_TIMEOUT,
    centred,
    draw,
    record,
    seed_of,
)
from sb_adapter_trace_claims import TraceFull, _tolist
from sb_methods import RIVALS
from sb_shape_readout import CFM_L, DFL_L

TR = TraceFull()
TRM = TraceFull(return_model=True)
SHAPE_GRID = [(2000, 5, 50), (20000, 20, 20)]
GL_U, GL_W = L.leggauss(256)
GL_U, GL_W = (GL_U + 1) / 2, GL_W / 2
PERFEAT = [m for m in RIVALS if m.per_feature]


def trace_rec(res, sec):
    rec = record(res)
    ex = {k: v for k, v in res.extra.items() if not k.startswith("_")}
    return rec | {"sec": sec, "x": ex}


def run_trace(XA, yA, XB, yB, seed, model=False):
    t0 = time.perf_counter()
    m = TRM if model else TR
    raw = m.raw(XA, yA, XB, yB, seed)
    res = m.decompose(XA, yA, XB, yB, seed, r=raw)
    return res, raw, time.perf_counter() - t0


def h5_checks(XA, yA, XB, yB, seed, res, raw):
    sdA2 = float(np.var(yA))
    x = res.extra
    K = len(x["bary_dc"])
    w = 1 / (2 * np.arange(K) + 1)
    phi = np.asarray(x["bary_phi"])
    dc = np.asarray(x["bary_dc"])
    out = {"sdA2": sdA2}
    out["g51a"] = abs(phi.sum() - np.sum(w * dc**2))
    m = raw["model"]
    qa = m.predict_quantile(XA, GL_U, "A").mean(0)
    qb = m.predict_quantile(XB, GL_U, "B").mean(0)
    wrec = float(np.sum(GL_W * (qb - qa) ** 2))
    out["W_rec"], out["sum_phi"] = wrec, float(phi.sum())
    out["g51b"] = abs(wrec - phi.sum())
    w2emp = float(np.mean((np.sort(yB) - np.sort(yA)) ** 2))
    mp, rp = x["marg_phi"], x["resid_phi"]
    msum = float(np.sum(mp["comp"]) + np.sum(mp["struct"]) + rp["comp"] + rp["struct"])
    out["W2_emp"], out["W2_trace_marg"], out["marg_sum"] = w2emp, x["w2"]["marginal"], msum
    out["g52"] = abs(msum - w2emp)
    out["g52_internal"] = abs(msum - x["w2"]["marginal"])
    lm = x["lmom"]
    dl = np.asarray(lm["B"]) - np.asarray(lm["A"])
    dck = (2 * np.arange(K) + 1) * dl
    chan = np.asarray(mp["comp"]) + np.asarray(mp["struct"])
    out["g53_chan"] = float(np.max(np.abs(chan - w * dck**2)))
    ls = x["lsplit"]
    out["g53_coef"] = float(np.max(np.abs(np.asarray(ls["comp"]) + np.asarray(ls["struct"]) - dl)))
    out["lmom_harness_dl"] = (shp.lmoments(yB) - shp.lmoments(yA)).tolist()
    # H5.4: period swap
    rs = TR.raw(XB, yB, XA, yA, seed)
    g = [
        np.max(np.abs(np.asarray(rs["bary_phi"]) - phi)),
        abs(
            np.sum(rs["marg_phi"]["comp"])
            + np.sum(rs["marg_phi"]["struct"])
            - np.sum(mp["comp"])
            - np.sum(mp["struct"])
        ),
        np.max(np.abs(np.asarray(rs["marg_phi"]["comp"]) - np.asarray(mp["comp"]))),
        np.max(np.abs(np.asarray(rs["marg_phi"]["struct"]) - np.asarray(mp["struct"]))),
        np.max(np.abs(np.asarray(rs["lsplit"]["comp"]) + np.asarray(ls["comp"]))),
        np.max(np.abs(np.asarray(rs["lsplit"]["struct"]) + np.asarray(ls["struct"]))),
        max(abs(rs["split"][s][i] + x["split_of"][s][i]) for s in rs["split"] for i in (0, 1)),
    ]
    out["g54_parts"] = [float(v) for v in g]
    out["g54"] = float(max(g))
    W = max(1.0, out["W2_emp"] / sdA2)
    out["tol"] = 1e-9 * W * sdA2
    return out


def rival_contrast(XA, yA, XB, yB, seed, base):
    out = {}
    for m in RIVALS:
        r = m.decompose(XB, yB, XA, yA, seed=seed)
        o = {}
        for s, (c, st) in base[m.name]["split"].items():
            tot = dgp.stat(yB, s) - dgp.stat(yA, s)
            cb, sb = r.split[s]
            o[s] = {
                "add_gap": abs(c + st - tot),
                "order_gap": max(abs(c + cb), abs(st + sb)),
                "swap": [cb, sb],
            }
        out[m.name] = o
    return out


# ------------------------------------------------------------- tracks
def t_truth_of(_):
    out = {}
    for s in dgp.SCENARIOS:
        sc = dgp.make(s, 5)
        rng = np.random.default_rng(99)
        n = 2_000_000
        XA, XB = sc.draw_x("A", n, rng), sc.draw_x("B", n, rng)
        yA, yB, yC = sc.A.draw_y(XA, rng), sc.B.draw_y(XB, rng), sc.A.draw_y(XB, rng)
        yCp = sc.B.draw_y(XA, rng)
        o = {}
        for st in dgp.STATS:
            tA, tB, tC, tCp = (dgp.stat(v, st) for v in (yA, yB, yC, yCp))
            o[st] = {
                "comp": 0.5 * ((tB - tCp) + (tC - tA)),
                "struct": 0.5 * ((tCp - tA) + (tB - tC)),
            }
        L4 = {k: shp.lmoments(v) for k, v in (("A", yA), ("B", yB), ("C", yC), ("Cp", yCp))}
        o["lsplit"] = {
            "comp": (0.5 * ((L4["B"] - L4["Cp"]) + (L4["C"] - L4["A"]))).tolist(),
            "struct": (0.5 * ((L4["Cp"] - L4["A"]) + (L4["B"] - L4["C"]))).tolist(),
        }
        out[s] = o
    return out


def t_acc(a):
    scen, n, p, rep = a
    XA, yA, XB, yB = draw(scen, n, p, rep)
    seed = seed_of("m", rep)
    res, raw, sec = run_trace(XA, yA, XB, yB, seed, model=True)
    rec = {
        "scen": scen,
        "n": n,
        "p": p,
        "rep": rep,
        "TRACE": trace_rec(res, sec),
        "h5": h5_checks(XA, yA, XB, yB, seed, res, raw),
    }
    base = {m.name: record(m.decompose(XA, yA, XB, yB, seed=seed)) for m in RIVALS}
    rec["contrast"] = rival_contrast(XA, yA, XB, yB, seed, base)
    return rec


def shape_draw(scen, n, p, rep):
    s = shp.make(scen, p)
    rng = np.random.default_rng(seed_of("data", scen, n, p, rep))
    XA, yA, XB, yB = s.sample(n, rng)
    a, b = centred(XA, XB)
    return XA, yA, XB, yB, a, b


def t_shape(a):
    scen, n, p, rep = a
    _, yA, _, yB, XA, XB = shape_draw(scen, n, p, rep)
    seed = seed_of("m", rep)
    res, raw, sec = run_trace(XA, yA, XB, yB, seed, model=True)
    rec = {
        "scen": scen,
        "n": n,
        "p": p,
        "rep": rep,
        "TRACE": trace_rec(res, sec),
        "h5": h5_checks(XA, yA, XB, yB, seed, res, raw),
        "sdA": float(np.std(yA)),
    }
    for m in (DFL_L(), CFM_L()):
        r = m.decompose(XA, yA, XB, yB, seed=seed)
        rec[m.name] = record(r) | {"lsplit": r.extra["lsplit"]}
    return rec


def t_inv(a):
    scen, n, p, rep = a
    XA, yA, XB, yB = draw(scen, n, p, rep)
    seed = seed_of("m", rep)
    b, _, _ = run_trace(XA, yA, XB, yB, seed)
    rng = np.random.default_rng(seed_of("perm", scen, n, p, rep))
    ra, rb, cp = rng.permutation(n), rng.permutation(n), rng.permutation(p)
    q, _, _ = run_trace(XA[ra][:, cp], yA[ra], XB[rb][:, cp], yB[rb], seed)
    inv = np.argsort(cp)
    d = {}
    for s in b.split:
        d["split_" + s] = max(abs(b.split[s][i] - q.split[s][i]) for i in (0, 1))
        d["split_of_" + s] = max(
            abs(b.extra["split_of"][s][i] - q.extra["split_of"][s][i]) for i in (0, 1)
        )
    d["feat"] = float(
        max(
            np.max(np.abs(b.feat_comp - q.feat_comp[inv])),
            np.max(np.abs(b.feat_struct - q.feat_struct[inv])),
        )
    )
    pb, pq = np.asarray(b.extra["bary_phi"]), np.asarray(q.extra["bary_phi"])
    pq = np.concatenate([pq[:, :1], pq[:, 1:][:, inv]], axis=1)
    d["bary_phi"] = float(np.max(np.abs(pb - pq)))
    d["lsplit"] = float(
        max(
            np.max(np.abs(np.asarray(b.extra["lsplit"][r]) - np.asarray(q.extra["lsplit"][r])))
            for r in ("comp", "struct")
        )
    )
    return {"scen": scen, "n": n, "p": p, "rep": rep, "max_change": max(d.values()), "parts": d}


def _boot_vecs(res):
    x = res.extra
    return {
        "split": res.split,
        "split_of": x["split_of"],
        "lsplit": x["lsplit"],
        "feat_comp": res.feat_comp.tolist(),
        "feat_struct": res.feat_struct.tolist(),
        "intercept": res.intercept,
    }


def t_cover(a):
    scen, rep = a
    n, p = COVER_N, COVER_P
    if scen in shp.SHAPE_SCENARIOS:
        s = shp.make(scen, p)
    else:
        s = dgp.make(scen, p)
    rng = np.random.default_rng(seed_of("data", scen, n, p, rep))
    XA, yA, XB, yB = s.sample(n, rng)
    a_, b_ = centred(XA, XB)
    point, _, _ = run_trace(a_, yA, b_, yB, seed_of("m", rep))
    brng = np.random.default_rng(seed_of("boot", scen, rep))
    boots = []
    for k in range(COVER_B):
        ia, ib = brng.integers(0, n, n), brng.integers(0, n, n)
        a_, b_ = centred(XA[ia], XB[ib])
        r, _, _ = run_trace(a_, yA[ia], b_, yB[ib], seed_of("mb", rep, k))
        boots.append(_boot_vecs(r))
    return {"scen": scen, "rep": rep, "point": _boot_vecs(point), "boots": boots}


def _all_methods_vec(XA, yA, XB, yB, seed):
    out = {}
    r, _, _ = run_trace(XA, yA, XB, yB, seed)
    out["TRACE"] = _boot_vecs(r)
    for m in RIVALS:
        q = m.decompose(XA, yA, XB, yB, seed=seed)
        out[m.name] = {
            "mean": q.split["mean"],
            "feat": None if q.feat_comp is None else np.r_[q.feat_comp, q.feat_struct].tolist(),
        }
    return out


B_NULL = 100


def t_null200(rep):
    n, p = 2000, 5
    s = dgp.make("null", p)
    rng = np.random.default_rng(seed_of("null200", rep))
    XA, yA, XB, yB = s.sample(n, rng)
    a_, b_ = centred(XA, XB)
    point = _all_methods_vec(a_, yA, b_, yB, seed_of("m", rep))
    brng = np.random.default_rng(seed_of("null200boot", rep))
    boots = []
    for k in range(B_NULL):
        ia, ib = brng.integers(0, n, n), brng.integers(0, n, n)
        a_, b_ = centred(XA[ia], XB[ib])
        boots.append(_all_methods_vec(a_, yA[ia], b_, yB[ib], seed_of("mb", rep, k)))
    return {"rep": rep, "point": point, "boots": boots}


# ---------------- real panel
def _real():
    import run_real as rr

    return rr


def t_real(a):
    kind, k = a
    rr = _real()
    d, xs = rr.load()
    units = np.sort(d.unit.unique())
    rng = np.random.default_rng(seed_of("real", kind, k))

    def tr(arr, seed):
        r, _, sec = run_trace(*arr, seed)
        return trace_rec(r, sec)

    if kind == "full":
        return {"kind": kind, "k": k, "TRACE": tr(rr.arrays(d, xs), seed_of("m", 0))}
    if kind == "boot":
        u = rng.choice(units, len(units), replace=True)
        return {"kind": kind, "k": k, "TRACE": tr(rr.arrays(d, xs, u), seed_of("m", k))}
    perm = rng.permutation(units)
    h1, h2 = perm[: len(units) // 2], perm[len(units) // 2 :]
    return {
        "kind": kind,
        "k": k,
        "h1": tr(rr.arrays(d[d.unit.isin(h1)], xs), seed_of("m", k)),
        "h2": tr(rr.arrays(d[d.unit.isin(h2)], xs), seed_of("m", k)),
    }


def t_placebo(k):
    rr = _real()
    d, xs = rr.load()
    d = d[d.period == "A"]
    units = np.sort(d.unit.unique())
    rng = np.random.default_rng(seed_of("placebo", k))
    perm = rng.permutation(units)
    g1, g2 = np.sort(perm[: len(units) // 2]), np.sort(perm[len(units) // 2 :])

    def arr(u1, u2):
        import pandas as pd

        A = pd.concat([d[d.unit == u] for u in u1], ignore_index=True)
        B = pd.concat([d[d.unit == u] for u in u2], ignore_index=True)
        XA, XB = centred(A[xs].to_numpy(float), B[xs].to_numpy(float))
        return XA, A.y.to_numpy(float), XB, B.y.to_numpy(float)

    point = _all_methods_vec(*arr(g1, g2), seed_of("m", k))
    brng = np.random.default_rng(seed_of("placebo_boot", k))
    boots = []
    for b in range(B_NULL):
        u1 = brng.choice(g1, len(g1), replace=True)
        u2 = brng.choice(g2, len(g2), replace=True)
        boots.append(_all_methods_vec(*arr(u1, u2), seed_of("mb", k, b)))
    return {"k": k, "point": point, "boots": boots}


# ---------------- H6 synthetic
H6_SCEN = [s for s in dgp.SCENARIOS if s != "null"]
H6_N, H6_P, H6_REPS, H6_B = 20000, 20, 20, 20


def _feat_vecs(XA, yA, XB, yB, seed):
    out = {}
    r, _, _ = run_trace(XA, yA, XB, yB, seed)
    phi = np.asarray(r.extra["bary_phi"])
    out["TRACE"] = (np.abs(r.feat_comp) + np.abs(r.feat_struct)).tolist()
    out["TRACE_w2"] = np.abs(phi[:, 1:, :].sum(2)).sum(0).tolist()
    for m in PERFEAT:
        q = m.decompose(XA, yA, XB, yB, seed=seed)
        out[m.name] = (np.abs(q.feat_comp) + np.abs(q.feat_struct)).tolist()
    return out


def t_h6syn(a):
    scen, rep = a
    n, p = H6_N, H6_P
    s = dgp.make(scen, p)
    rng = np.random.default_rng(seed_of("data", scen, n, p, rep))
    XA, yA, XB, yB = s.sample(n, rng)
    seed = seed_of("m", rep)
    a_, b_ = centred(XA, XB)
    full = _feat_vecs(a_, yA, b_, yB, seed)
    hr = np.random.default_rng(seed_of("half", scen, n, p, rep))
    pa, pb = hr.permutation(n), hr.permutation(n)
    halves = []
    for sl in (slice(0, n // 2), slice(n // 2, n)):
        ia, ib = pa[sl], pb[sl]
        a_, b_ = centred(XA[ia], XB[ib])
        halves.append(_feat_vecs(a_, yA[ia], b_, yB[ib], seed))
    boots = []
    for b in range(H6_B):
        br = np.random.default_rng(seed_of("sboot", scen, n, p, rep, b))
        ia, ib = br.integers(0, n, n), br.integers(0, n, n)
        a_, b_ = centred(XA[ia], XB[ib])
        boots.append(_feat_vecs(a_, yA[ia], b_, yB[ib], seed))
    return {"scen": scen, "rep": rep, "full": full, "halves": halves, "boots": boots}


# ---------------- scale
def scale_child(n, p):
    XA, yA, XB, yB = draw("mixed", n, p, 0)
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    TR.raw(XA, yA, XB, yB, 0)
    sec = time.perf_counter() - t0
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        json.dumps(
            {
                "method": "TRACE",
                "n": n,
                "p": p,
                "sec": sec,
                "peak_rss_mb": rss1 / 1024,
                "data_mb": 2 * (XA.nbytes + yA.nbytes) / 2**20,
                "extra_mb": max(rss1 - rss0, 0) / 1024,
            }
        )
    )


def run_scale():
    out = []
    for n, p in SCALE_GRID:
        env = dict(os.environ, RAYON_NUM_THREADS="4", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
        try:
            r = subprocess.run(
                [sys.executable, __file__, "scale-child", str(n), str(p)],
                capture_output=True,
                text=True,
                timeout=SCALE_TIMEOUT,
                env=env,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                preexec_fn=lambda: resource.setrlimit(
                    resource.RLIMIT_AS, (SCALE_MEM_BYTES, SCALE_MEM_BYTES)
                ),
            )
            out.append(
                json.loads(r.stdout.strip().splitlines()[-1])
                if r.returncode == 0
                else {
                    "method": "TRACE",
                    "n": n,
                    "p": p,
                    "status": "error",
                    "stderr": r.stderr[-400:],
                }
            )
        except subprocess.TimeoutExpired:
            out.append({"method": "TRACE", "n": n, "p": p, "status": "timeout"})
        print(out[-1], flush=True)
        (RES / "trace_claims_scale.json").write_text(json.dumps(out, indent=1))


def tasks_for(track):
    if track == "acc":
        return t_acc, [
            (s, n, p, r) for n, p, reps in ACC_GRID for s in dgp.SCENARIOS for r in range(reps)
        ]
    if track == "shape":
        return t_shape, [
            (s, n, p, r)
            for n, p, reps in SHAPE_GRID
            for s in shp.SHAPE_SCENARIOS
            for r in range(reps)
        ]
    if track == "inv":
        return t_inv, [
            (s, n, p, r) for n, p in INV_GRID for s in dgp.SCENARIOS for r in range(INV_REPS)
        ]
    if track == "cover":
        return t_cover, [
            (s, r) for s in COVER_SCEN + shp.SHAPE_SCENARIOS for r in range(COVER_REPS)
        ]
    if track == "null200":
        return t_null200, list(range(200))
    if track == "real":
        return t_real, [("full", 0)] + [("boot", k) for k in range(100)] + [
            ("split", k) for k in range(50)
        ]
    if track == "placebo":
        return t_placebo, list(range(100))
    if track == "h6syn":
        return t_h6syn, [(s, r) for s in H6_SCEN for r in range(H6_REPS)]
    raise SystemExit(track)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("track")
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.track == "scale-child":
        return scale_child(int(a.rest[0]), int(a.rest[1]))
    if a.track == "scale":
        return run_scale()
    if a.track == "truth_of":
        (RES / "trace_claims_truth_orderfree.json").write_text(
            json.dumps(t_truth_of(None), indent=1)
        )
        return
    fn, tasks = tasks_for(a.track)
    if a.limit:
        tasks = tasks[: a.limit]
    path = a.out or RES / f"trace_claims_{a.track}.jsonl"
    with Pool(a.workers, maxtasksperchild=20) as pool, open(path, "w") as f:
        for i, rec in enumerate(pool.imap_unordered(fn, tasks)):
            f.write(json.dumps(_tolist(rec)) + "\n")
            f.flush()
            print(f"{a.track} {i + 1}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
