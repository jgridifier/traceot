# ruff: noqa: E501, E731
"""EXPLORATORY diagnosis of the failed TRACE v5 first gate (not a gate, not scored).

Dev seeds 9500-9549, n=2000, skew_comp / tail_comp ONLY (the seeds already spent by
the failed gate). Indices 9600-9619 are never touched.

Question: is v5's graded-channel error mostly *conditional-model* error (the fitted
F_t(y|x)) or *mixture / inversion* error (averaging CDFs on a 256-point grid and
inverting)? We swap the conditional while keeping the v5 mixture construction:

  V5        recorded v5 output (benchmarks/results/v5_gate1_shape.jsonl)
  V5_RERUN  traceot.explain_v5 re-run (checks the recorded numbers reproduce)
  ORACLE    true F_t(y|x): true mean mu(x) and the true noise law of period t
            (deterministic quantiles sigma_t * Phi^-1((i-.5)/n)); sample X only
  ORACLE_MU true mu(x) + the period's empirical residuals y - mu(x)
  OLS       linear least-squares mean + its empirical residuals (estimated,
            correctly specified location family; no oracle information)

Every variant builds the four hybrids with the Rust mixture CDF
(traceot.mixture_cdf_location) on the same 256-point support grid as v5, inverts with
traceot.mixture_quantile, and reads L-moments of the inverted law. ORACLE is also
evaluated exactly (all n^2 location+residual pairs, no grid) to size the
grid / inversion error on its own.
"""

from __future__ import annotations

import json
import os
from multiprocessing import Pool

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np
import sb_dgp_shape as shp
from run_synthetic import RES, centred, seed_of
from scipy import stats
from summarize_v4 import cell_verdict, graded_err

SCEN = ("skew_comp", "tail_comp")
N, P = 2000, 5
REPS = list(range(9500, 9550))
OUT = RES / "v5_diag_gate1.json"
U = (np.arange(8192) + 0.5) / 8192


def pstar(u):
    x = 2 * u - 1
    return np.stack([np.ones_like(u), x, (3 * x**2 - 1) / 2, (5 * x**3 - 3 * x) / 2])


PS = pstar(U)


def lmom_grid(mu, r, grid):
    import traceot

    f = traceot.mixture_cdf_location(mu, np.sort(r), grid)
    q = traceot.mixture_quantile(grid, np.maximum.accumulate(f), U)
    return PS @ q / len(U)


def lmom_exact(mu, r):
    return shp.lmoments((mu[:, None] + r[None, :]).ravel())


def split(L):
    aa, ab, ba, bb = (np.asarray(L[h]) for h in ("aa", "ab", "ba", "bb"))
    return {
        "comp": (0.5 * ((ba - aa) + (bb - ab))).tolist(),
        "struct": (0.5 * ((ab - aa) + (bb - ba))).tolist(),
    }


def hybrids(fn, mA, mB, XA, XB, rA, rB):
    return split(
        {"aa": fn(mA(XA), rA), "ab": fn(mB(XA), rB), "ba": fn(mA(XB), rA), "bb": fn(mB(XB), rB)}
    )


def task(a):
    import traceot

    scen, rep = a
    s = shp.make(scen, P)
    rng = np.random.default_rng(seed_of("data", scen, N, P, rep))
    XA0, yA, XB0, yB = s.sample(N, rng)
    XA, XB = centred(XA0, XB0)
    off = XA0[0] - XA[0]
    r5 = traceot.explain_v5(XA, yA, XB, yB)
    lo = min(yA.min(), yB.min()) - 1.0
    hi = max(yA.max(), yB.max()) + 1.0
    grid = np.linspace(lo - 3, hi + 3, 256)
    g = lambda mu, r: lmom_grid(mu, r, grid)
    mu_true = lambda X: s.mu(X + off)
    sig = {"A": 1.0, "B": s.sigma_b}
    z = stats.norm.ppf((np.arange(N) + 0.5) / N)  # both periods have normal noise here
    out = {
        "scen": scen,
        "rep": rep,
        "V5_RERUN": {"comp": list(r5["lsplit_comp"]), "struct": list(r5["lsplit_struct"])},
    }
    out["ORACLE"] = hybrids(g, mu_true, mu_true, XA, XB, sig["A"] * z, sig["B"] * z)
    out["ORACLE_EXACT"] = hybrids(lmom_exact, mu_true, mu_true, XA, XB, sig["A"] * z, sig["B"] * z)
    out["ORACLE_MU"] = hybrids(g, mu_true, mu_true, XA, XB, yA - mu_true(XA), yB - mu_true(XB))

    def ols(X, y):
        D = np.c_[np.ones(len(X)), X]
        b = np.linalg.lstsq(D, y, rcond=None)[0]
        return lambda Z: np.c_[np.ones(len(Z)), Z] @ b

    fA, fB = ols(XA, yA), ols(XB, yB)
    out["OLS"] = hybrids(g, fA, fB, XA, XB, yA - fA(XA), yB - fB(XB))
    # residual-shape check on the OLS fit: |resid| vs fitted value (heteroskedasticity)
    out["het_corr"] = {
        "A": float(stats.spearmanr(np.abs(yA - fA(XA)), fA(XA))[0]),
        "B": float(stats.spearmanr(np.abs(yB - fB(XB)), fB(XB))[0]),
    }
    return out


def main():
    rows = [json.loads(x) for x in (RES / "v5_gate1_shape.jsonl").read_text().splitlines() if x]
    rec = {(r["scen"], r["rep"]): r for r in rows}
    truth = json.loads((RES / "shiftbench_truth_shape.json").read_text())
    todo = [(s, r) for s in SCEN for r in REPS]
    assert not any(r >= 9600 for _, r in todo)
    with Pool(8) as pool:
        diag = pool.map(task, todo)
    summary = {
        "label": "EXPLORATORY diagnosis on spent dev seeds 9500-9549; not a gate",
        "cells": {},
    }
    for scen in SCEN:
        t = truth[scen]
        sh, k = t["shapley"], t["graded"][0]
        delta = 0.10 * (abs(sh["comp"][k]) + abs(sh["struct"][k]))
        ds = [d for d in diag if d["scen"] == scen]
        riv = {
            nm: np.array(
                [graded_err(rec[(scen, d["rep"])]["rivals"][nm]["lsplit"], sh, k) for d in ds]
            )
            for nm in ("DFL", "CFM")
        }
        cfm = np.minimum(riv["DFL"], riv["CFM"])  # best rival per replication (gate protocol)
        cell = {
            "delta": delta,
            "truth": {"comp": sh["comp"][k], "struct": sh["struct"][k]},
            "CFM_err": float(riv["CFM"].mean()),
            "DFL_err": float(riv["DFL"].mean()),
            "best_rival_err": float(cfm.mean()),
        }
        variants = {"V5": [rec[(scen, d["rep"])]["TRACE_V5"]["lsplit"] for d in ds]}
        for v in ("V5_RERUN", "ORACLE", "ORACLE_EXACT", "ORACLE_MU", "OLS"):
            variants[v] = [d[v] for d in ds]
        for v, ls in variants.items():
            e = np.array([graded_err(x, sh, k) for x in ls])
            verd, ci, hi_ = cell_verdict(e - cfm, delta)
            c = float(np.mean([abs(x["comp"][k]) for x in ls]))
            s_ = float(np.mean([abs(x["struct"][k]) for x in ls]))
            cell[v] = {
                "err": float(e.mean()),
                "mean_d_vs_best_rival": float((e - cfm).mean()),
                "upper95": float(hi_),
                "verdict": verd,
                "abs_c": c,
                "abs_s": s_,
                "role_ok": bool(c > s_),
                "bias_c": float(np.mean([x["comp"][k] for x in ls]) - sh["comp"][k]),
                "bias_s": float(np.mean([x["struct"][k] for x in ls]) - sh["struct"][k]),
            }
        cell["het_corr_mean"] = {p_: float(np.mean([d["het_corr"][p_] for d in ds])) for p_ in "AB"}
        summary["cells"][scen] = cell
    OUT.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
