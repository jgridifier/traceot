# ruff: noqa: E501
"""TRACE v1 checks T1–T14.

Pass criteria are the pre-written bullets in docs/theory/DESIGN_DELTA.md §7.
This script does not edit those bullets. JSON has no NaN.

    python benchmarks/trace_sims/run_v1.py
    python benchmarks/trace_sims/run_v1.py --only t3
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np

from traceot import FrechetV1, RobustV1, TraceRegressor, TraceV1, _core

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks" / "results" / "trace_v1.json"
_NORM = NormalDist()
ORACLE_HE3 = 0.12 * math.sqrt(6.0)


def _z(u: np.ndarray) -> np.ndarray:
    return np.array([_NORM.inv_cdf(float(v)) for v in np.clip(u, 1e-12, 1.0 - 1e-12)])


def _ek(k: int, u: float, k_fit: int) -> float:
    a = np.zeros(k_fit)
    a[k] = 1.0
    return float(_core.v1_eval_a(np.ascontiguousarray(a), float(u)))


def _finite(obj):
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError("NaN in results")
        return obj
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return _finite(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _s1_period2(n: int = 2000, seed: int = 11, p: int = 8):
    rng = np.random.default_rng(seed)
    x = rng.random((n, p))
    z = _z(rng.random(n))
    he3 = z**3 - 3.0 * z
    y = 1.2 * x[:, 0] + (0.4 + 1.2 * x[:, 1]) * z + 0.12 * x[:, 2] * he3
    return x, y


def _fit_v1(x, y, **kw):
    cfg = dict(
        k_fit=6,
        k_report=4,
        n_u=0,
        max_bins=16,
        max_rounds=30,
        tol=1e-9,
        mono="none",
        reporting="legendre",
    )
    cfg.update(kw)
    return TraceV1(**cfg).fit(x, y)


def t1() -> dict:
    rng = np.random.default_rng(1)
    n, p = 1000, 3
    x = rng.random((n, p))
    y = 0.4 * x[:, 0] + (0.3 + 0.5 * x[:, 1]) * _z(rng.random(n)) + rng.normal(scale=0.05, size=n)
    model = _fit_v1(x, y, k_fit=4, k_report=4, max_rounds=12, max_bins=8)
    attr = model.attribute(x)
    phi = attr["phi_fit"]
    w2 = phi.reshape(n, -1).sum(axis=1)
    nodes = np.asarray(model.model_.u_nodes())
    _, w = _core.v1_gauss_legendre(int(nodes.size))
    w = np.asarray(w)
    curves = np.asarray(model.predict(x, nodes, project=False))
    quad = (curves**2) @ w
    gap = float(np.max(np.abs(w2 - quad)))
    rel = float(np.max(np.abs(w2 - quad) / np.maximum(1.0, np.abs(quad))))
    # Sum of squared orthonormal coefficients equals the quadrature of Q^2.
    scale = float(np.max(np.maximum(1.0, np.abs(w2))))
    ok = gap <= 1e-10 * scale and rel <= 1e-10
    return {"max_abs": gap, "max_rel": rel, "pass": bool(ok)}


def t2() -> dict:
    nodes, w = _core.v1_gauss_legendre(8)
    nodes = np.asarray(nodes)
    w = np.asarray(w)
    k = 8
    e = np.column_stack([np.array([_ek(c, float(u), k) for u in nodes]) for c in range(k)])
    gram = (e * w[:, None]).T @ e
    err = float(np.max(np.abs(gram - np.eye(k))))
    return {"max_abs": err, "pass": bool(err <= 1e-12)}


def _energies(model, p: int, channel: int) -> list[float]:
    return [float(model.channel_energy(j, channel)) for j in range(p)]


def t3() -> dict:
    x, y = _s1_period2()
    model = _fit_v1(
        x, y, k_fit=8, k_report=4, reporting="hermite", max_rounds=25, tol=1e-8, max_bins=16
    )
    # Hermite-in-z report is {1, z, He2, He3} / sqrt(m!). He3 is channel 3.
    e = _energies(model, 8, 3)
    shape = e[2]
    others = max(e[j] for j in range(8) if j != 2)
    slope = float(model.channel_slope(2, 3, x[:, 2]))
    rel = abs(slope - ORACLE_HE3) / ORACLE_HE3
    ratio_ok = shape >= 3.0 * others
    slope_ok = rel <= 0.30
    return {
        "energies_he3": e,
        "shape_over_next": float(shape / others) if others else 0.0,
        "slope": slope,
        "oracle_slope": ORACLE_HE3,
        "slope_rel_err": float(rel),
        "rounds": int(model.model_.rounds),
        "pass": bool(ratio_ok and slope_ok),
    }


def _legendre_oracle_channel3() -> dict:
    """Population L2 loads of S1's scale and shape pieces on e_3."""
    nodes, w = _core.v1_gauss_legendre(2000)
    nodes = np.asarray(nodes)
    w = np.asarray(w)
    z = _z(nodes)
    he3 = z**3 - 3.0 * z
    e3 = np.array([_ek(3, float(u), 4) for u in nodes])
    a_scale = float(np.sum(w * (1.2 * z) * e3))
    a_shape = float(np.sum(w * (0.12 * he3) * e3))
    # Centred linear effect on U[0,1], energy averaged over x: coef(x)^2 mean = a^2 / 12.
    return {
        "x_loc": 0.0,
        "x_scale": a_scale**2 / 12.0,
        "x_shape": a_shape**2 / 12.0,
        "noise": 0.0,
        "ratio_shape_scale": (a_shape**2 / a_scale**2) if a_scale else 0.0,
    }


def t4() -> dict:
    oracle = _legendre_oracle_channel3()
    kw = dict(k_fit=8, k_report=4, reporting="legendre", max_rounds=20, tol=1e-7)
    x, y = _s1_period2()
    model = _fit_v1(x, y, **kw)
    names = ["x_loc", "x_scale", "x_shape"] + [f"noise{i}" for i in range(5)]
    point = {names[j]: float(model.channel_energy(j, 3)) for j in range(8)}
    reps = []
    for r in range(200):
        xr, yr = _s1_period2(seed=1000 + r)
        mr = _fit_v1(xr, yr, **kw)
        reps.append([float(mr.channel_energy(j, 3)) for j in range(8)])
        if (r + 1) % 40 == 0:
            print(f"  t4 rep {r + 1}/200", flush=True)
    reps_a = np.asarray(reps)
    se = reps_a.std(axis=0, ddof=1) / math.sqrt(200.0)
    oracle_v = [
        oracle["x_loc"],
        oracle["x_scale"],
        oracle["x_shape"],
        oracle["noise"],
        oracle["noise"],
        oracle["noise"],
        oracle["noise"],
        oracle["noise"],
    ]
    flags = []
    for j, name in enumerate(names):
        est = point[name]
        ora = oracle_v[j]
        within_se = abs(est - ora) <= 2.0 * float(se[j])
        if abs(ora) > 1e-8:
            within_rel = abs(est - ora) <= 0.25 * abs(ora)
        else:
            within_rel = abs(est - ora) <= 0.25 * max(abs(ora), 1e-6)
        flags.append(
            {
                "feature": name,
                "estimate": est,
                "oracle": ora,
                "se": float(se[j]),
                "within_2se": bool(within_se),
                "within_25rel": bool(within_rel),
                "pass": bool(within_se or within_rel),
            }
        )
    return {
        "oracle_ratio_shape_scale": oracle["ratio_shape_scale"],
        "features": flags,
        "pass": bool(all(row["pass"] for row in flags)),
    }


def t5() -> dict:
    rng = np.random.default_rng(11)
    n = 2000
    x = rng.random((n, 5))
    u = rng.random(n)
    k = 6
    y = np.empty(n)
    for i in range(n):
        ui = float(u[i])
        y[i] = 1.2 * x[i, 0] + (0.5 + x[i, 1]) * _ek(1, ui, k) + 0.15 * x[i, 2] * _ek(3, ui, k)
    model = _fit_v1(x, y, k_fit=k, k_report=4, reporting="legendre", max_rounds=30, tol=1e-8)
    e3 = _energies(model, 5, 3)
    shape = e3[2]
    others = [e3[j] for j in range(5) if j != 2]
    ratio = shape / max(others) if max(others) > 0 else 0.0
    leak = e3[1] / shape if shape else 99.0
    # Oracle energies of the centred linear effects, histogram-free L2 in x.
    ora = {
        0: (1.2**2) / 12.0,  # channel 0
        1: (1.0**2) / 12.0,  # channel 1 of scale
        2: (0.15**2) / 12.0,  # channel 3 of shape
    }
    est = {
        "loc_e0": float(model.channel_energy(0, 0)),
        "scale_e1": float(model.channel_energy(1, 1)),
        "shape_e3": float(shape),
    }
    rel = {
        "loc_e0": abs(est["loc_e0"] - ora[0]) / ora[0],
        "scale_e1": abs(est["scale_e1"] - ora[1]) / ora[1],
        "shape_e3": abs(est["shape_e3"] - ora[2]) / ora[2],
    }
    ok = bool(ratio >= 3.0 and leak <= 0.10 and all(v <= 0.25 for v in rel.values()))
    return {
        "energies_ch3": e3,
        "ratio_shape_to_next": float(ratio),
        "scale_leak": float(leak),
        "rel_err": rel,
        "slopes": {
            "loc0": float(model.channel_slope(0, 0, x[:, 0])),
            "scale1": float(model.channel_slope(1, 1, x[:, 1])),
            "shape3": float(model.channel_slope(2, 3, x[:, 2])),
        },
        "pass": ok,
    }


def t6() -> dict:
    rng = np.random.default_rng(6)
    n = 2000
    x = rng.random((n, 1))
    z = _z(rng.random(n))
    y = (0.5 + x[:, 0]) * z
    v0 = TraceRegressor(
        n_channels=3,
        max_bins=16,
        learning_rate=1.0,
        max_rounds=40,
        patience=8,
        val_frac=0.0,
        min_rounds=20,
        u_levels=19,
        max_interactions=0,
        seed=6,
    ).fit(x, y)
    coef, _ = v0.effect(0)
    coef = np.asarray(coef)
    # v0 shape channel is He3, index 2. Energy excludes the missing bin.
    a_energy = float(np.dot(coef[1:, 2], coef[1:, 2]) / (coef.shape[0] - 1))
    v1 = _fit_v1(x, y, k_fit=8, k_report=4, reporting="hermite", max_rounds=20, tol=1e-8)
    b_energy = float(v1.channel_energy(0, 3))
    # Permutation floor: shuffle x and refit. Energy at or below the 95% null quantile.
    null = []
    for _r in range(49):
        xr = rng.permutation(x[:, 0])[:, None]
        mr = _fit_v1(xr, y, k_fit=8, k_report=4, reporting="hermite", max_rounds=20, tol=1e-8)
        null.append(float(mr.channel_energy(0, 3)))
    cutoff = float(np.quantile(null, 0.95))
    ok = bool(b_energy <= 0.05 * a_energy and b_energy <= cutoff)
    return {
        "v0_shape_energy": a_energy,
        "v1_he3_energy": b_energy,
        "ratio_b_over_a": float(b_energy / a_energy) if a_energy else 0.0,
        "perm_cutoff_95": cutoff,
        "pass": ok,
    }


def t7() -> dict:
    rng = np.random.default_rng(7)
    n = 2000
    y = rng.exponential(1.0, size=n)
    x = np.zeros((n, 1))
    m3 = _fit_v1(x, y, k_fit=3, k_report=3, max_rounds=15, mono="none", max_bins=2)
    flagged = bool(m3.model_.non_representable)
    m4 = _fit_v1(x, y, k_fit=4, k_report=4, max_rounds=20, mono="pav", max_bins=2)
    report = m4.model_.mono_report(10001)
    d = float(np.max(np.asarray(m4.attribute(x[:50])["d"])))
    mb = _fit_v1(
        x,
        y,
        k_fit=4,
        k_report=4,
        max_rounds=20,
        mono="bernstein",
        max_bins=2,
        bernstein_elevation=4,
    )
    worst = float(mb.model_.mono_report(10001)["worst_derivative"])
    ok = bool(flagged and d <= 1e-8 and worst >= -1e-8)
    return {
        "k3_flagged": flagged,
        "k4_pav_distance": d,
        "bernstein_worst_derivative": worst,
        "k4_mono_report": {
            k: (float(v) if isinstance(v, (float, int)) else v)
            for k, v in dict(report).items()
            if k != "certified"
        },
        "pass": ok,
    }


def t8() -> dict:
    rng = np.random.default_rng(8)
    n = 500
    x = rng.random((n, 2))
    y = x[:, 0] + rng.normal(size=n) * (0.2 + x[:, 1])
    model = _fit_v1(x, y, k_fit=4, max_rounds=8, max_bins=6, mono="none")
    nodes = np.asarray(model.model_.u_nodes())
    _, w = _core.v1_gauss_legendre(int(nodes.size))
    w = np.asarray(w, dtype=np.float64)
    curves = np.asarray(model.predict(x, nodes, project=False))
    worst = 0.0
    adj_ok = True
    order_ok = True
    for i in range(0, n, 2):
        pair = _core.v1_projection_pair(
            np.ascontiguousarray(curves[i]),
            np.ascontiguousarray(curves[i + 1]),
            np.ascontiguousarray(w),
        )
        d = float(pair["d"])
        d_pi = float(pair["d_pi"])
        d_a = float(pair["d_a"])
        d_b = float(pair["d_b"])
        adj = float(pair["adjustment"])
        order_ok &= d_pi <= d + 1e-10 and d_pi >= d - d_a - d_b - 1e-10
        lo = -2.0 * d * (d_a + d_b) - 1e-10
        adj_ok &= lo <= adj <= 1e-10
        worst = max(worst, abs(min(0.0, d_pi - (d - d_a - d_b))), abs(min(0.0, d - d_pi)))
    return {"worst_slack": float(worst), "pass": bool(order_ok and adj_ok)}


def t9() -> dict:
    rng = np.random.default_rng(9)
    n = 400
    # Same conditional law, different covariate law. Algebraic identity on the product API.
    x_a = rng.random((n, 2))
    x_b = rng.random((n, 2)) * 0.5
    ua, ub = rng.random(n), rng.random(n)
    y_a = 0.8 * x_a[:, 0] + (0.4 + 0.6 * x_a[:, 1]) * _z(ua)
    y_b = 0.8 * x_b[:, 0] + (0.4 + 0.6 * x_b[:, 1]) * _z(ub)
    ma = _fit_v1(x_a, y_a, k_fit=4, max_rounds=12, max_bins=6)
    mb = _fit_v1(x_b, y_b, k_fit=4, max_rounds=12, max_bins=6)
    shift = ma.shift_marginal(mb, x_a, x_b, grid=64)
    total = shift["total"]
    parts = shift["composition"] + shift["structure"] + shift["cross"]
    ident = abs(parts - total)
    # Stratified permutation level of the within-bin L-kurtosis energy.
    # Covariate-only shift: labels are not exchangeable globally, so strata are the bins.
    rejects = 0
    reps = 200
    for r in range(reps):
        rr = np.random.default_rng(200 + r)
        xx = rr.random(n * 2)
        bins = np.minimum((xx * 4).astype(np.int64), 3)
        lab = np.array([0] * n + [1] * n, dtype=np.uint8)
        # Move period B's covariates but keep y|x the same.
        shift_x = np.concatenate([xx[:n], xx[n:] * 0.5])
        bins = np.minimum((shift_x * 4).astype(np.int64), 3)
        zz = rr.normal(size=2 * n)
        yy = 0.3 * shift_x + (1.0 + 0.2 * shift_x) * zz
        p = float(
            _core.v1_permutation_pvalue_lmoment(
                np.ascontiguousarray(yy),
                np.ascontiguousarray(bins),
                np.ascontiguousarray(lab),
                3,
                4,
                199,
                50 + r,
                True,
            )
        )
        rejects += p <= 0.05
    rate = rejects / reps
    return {
        "identity_abs": float(ident),
        "total": float(total),
        "structure": float(shift["structure"]),
        "composition": float(shift["composition"]),
        "cross": float(shift["cross"]),
        "rejection_rate": float(rate),
        "pass": bool(ident <= 1e-8 and rate <= 0.08),
    }


def t10() -> dict:
    reps = 300
    rejects = 0
    for r in range(reps):
        rr = np.random.default_rng(r + 1)
        n = 300
        x = rr.integers(0, 4, size=2 * n)
        lab = np.array([0] * n + [1] * n, dtype=np.uint8)
        zz = rr.normal(size=2 * n)
        y = x * 0.3 + (1.0 + 0.2 * x) * zz
        p = float(
            _core.v1_permutation_pvalue_lmoment(
                np.ascontiguousarray(y),
                np.ascontiguousarray(x.astype(np.int64)),
                np.ascontiguousarray(lab),
                3,
                4,
                199,
                r + 17,
                False,
            )
        )
        rejects += p <= 0.1
        if (r + 1) % 50 == 0:
            print(f"  t10 {r + 1}/{reps} rej {rejects / (r + 1):.3f}", flush=True)
    rate = rejects / reps
    return {"rejection_rate": float(rate), "pass": bool(0.048 <= rate <= 0.152)}


def _frechet_draw(n: int, seed: int):
    rng = np.random.default_rng(seed)
    # Discrete x so the histogram is not the rate bottleneck.
    x = rng.integers(0, 3, size=n).astype(np.float64)
    m = np.array([5, 50, 500], dtype=int)
    lengths = m[np.arange(n) % 3]
    m_max = 500
    samples = np.zeros((n, m_max))
    scale = 1.0 + 0.5 * x
    for i in range(n):
        samples[i, : lengths[i]] = rng.exponential(scale[i], size=int(lengths[i]))
    return x[:, None], samples, lengths, scale


def t11() -> dict:
    def rmse(n, reps, seed0):
        err = []
        for r in range(reps):
            x, samples, lengths, scale = _frechet_draw(n, seed0 + r)
            model = FrechetV1(k_fit=2, max_bins=4, estimate_weights=True).fit(x, samples, lengths)
            pred = np.asarray(model.predict(x))
            # a_1 = sqrt(3) * lambda_2, and lambda_2 = scale/2 for an exponential.
            lam2 = pred[:, 1] / math.sqrt(3.0)
            truth = 0.5 * scale
            err.append(float(np.mean((lam2 - truth) ** 2)))
        return math.sqrt(float(np.mean(err)))

    small = rmse(200, 30, 11)
    big = rmse(3200, 20, 400)
    ratio = big / small if small else 99.0
    # GLS slope variance against equal and proportional-to-m weights.
    rng = np.random.default_rng(35)
    n = 120
    x = rng.random(n)
    lengths = np.array([5] * 40 + [50] * 40 + [500] * 40)
    slopes = {"equal": [], "prop": [], "gls": []}
    reps = 80
    for r in range(reps):
        rr = np.random.default_rng(1000 + r)
        truth = 0.5 + 0.4 * x
        samples = np.zeros((n, 500))
        for i in range(n):
            mi = int(lengths[i])
            samples[i, :mi] = rr.exponential(2.0 * truth[i], size=mi)
        base = dict(k_fit=2, max_bins=8, estimate_weights=False)
        eq = FrechetV1(**base).fit(x[:, None], samples, lengths)
        pr = FrechetV1(**base).fit(x[:, None], samples, lengths, weights=lengths.astype(np.float64))
        gls = FrechetV1(k_fit=2, max_bins=8, estimate_weights=True).fit(
            x[:, None], samples, lengths
        )
        for name, model in (("equal", eq), ("prop", pr), ("gls", gls)):
            pred = np.asarray(model.predict(x[:, None]))[:, 1]
            # Slope of predicted lambda-scale on x. a_1 is proportional to lambda_2.
            xm = x - x.mean()
            ym = pred - pred.mean()
            slopes[name].append(float(np.dot(xm, ym) / np.dot(xm, xm)))
    var = {k: float(np.var(v)) for k, v in slopes.items()}
    gls_ok = var["gls"] <= 1.05 * min(var["equal"], var["prop"])
    ratio_ok = 0.15 <= ratio <= 0.40
    return {
        "rmse_n200": float(small),
        "rmse_n3200": float(big),
        "ratio": float(ratio),
        "slope_var": var,
        "pass": bool(ratio_ok and gls_ok),
    }


def t12() -> dict:
    from scipy.optimize import linprog

    n, p, binc = 50, 3, 6
    width = 1.0 / binc
    configs = list(np.ndindex(binc, binc, binc))
    worst = 0.0
    bound_ok = True
    for seed in range(3):
        rr = np.random.default_rng(seed + 3)
        f = [rr.normal(size=binc) for _ in range(p)]
        bins = rr.integers(0, binc, size=(n, p))
        y = np.zeros(n)
        for i in range(n):
            y[i] = sum(f[j][bins[i, j]] for j in range(p)) + rr.normal()
        kappa = np.full(p, width)
        for ball, cost_pow in (("w1", 1), ("w2", 2)):
            loss = np.zeros((n, len(configs)))
            dist = np.zeros((n, len(configs)))
            u = 0.7
            for i in range(n):
                for c_i, c in enumerate(configs):
                    q = sum(f[j][c[j]] for j in range(p))
                    r = y[i] - q
                    loss[i, c_i] = u * r if r >= 0 else (u - 1.0) * r
                    dist[i, c_i] = (
                        sum(kappa[j] * abs(c[j] - int(bins[i, j])) for j in range(p)) ** cost_pow
                    )
            shapes = [np.asarray(col, dtype=np.float64) for col in f]
            for eps in (0.01, 0.1, 0.5):
                budget = eps if ball == "w1" else eps**2
                # Primal LP, one coupling per row.
                nn, ccount = loss.shape
                a_eq = np.zeros((nn, nn * ccount))
                for i in range(nn):
                    a_eq[i, i * ccount : (i + 1) * ccount] = 1.0
                res = linprog(
                    -loss.ravel(),
                    A_ub=dist.ravel()[None, :],
                    b_ub=[budget],
                    A_eq=a_eq,
                    b_eq=np.full(nn, 1.0 / nn),
                    bounds=(0, None),
                    method="highs",
                )
                primal = float(-res.fun)
                dual = float(
                    _core.v1_robust_risk(
                        0.0,
                        shapes,
                        np.ascontiguousarray(bins.astype(np.int64)),
                        np.ascontiguousarray(y),
                        u,
                        np.ascontiguousarray(kappa),
                        budget if ball == "w2" else eps,
                        ball,
                        None,
                        True,
                    )
                )
                worst = max(worst, abs(primal - dual))
                emp = float(np.mean([loss[i, 0] for i in range(n)]))  # placeholder replaced below
                # Empirical pinball at the observed bins.
                emp = 0.0
                for i in range(n):
                    q = sum(f[j][int(bins[i, j])] for j in range(p))
                    r = y[i] - q
                    emp += u * r if r >= 0 else (u - 1.0) * r
                emp /= n
                ub = float(
                    _core.v1_max_jump_bound(
                        shapes,
                        np.ascontiguousarray(kappa),
                        emp,
                        eps,
                        u,
                    )
                )
                exact = float(
                    _core.v1_robust_risk(
                        0.0,
                        shapes,
                        np.ascontiguousarray(bins.astype(np.int64)),
                        np.ascontiguousarray(y),
                        u,
                        np.ascontiguousarray(kappa),
                        eps,
                        "w1",
                        None,
                        True,
                    )
                )
                if ball == "w1":
                    bound_ok &= ub + 1e-8 >= exact
        # Joint cost, one epsilon, compared with the same LP.
        ky = 0.2
        u = 0.7
        # Joint moves in y are continuous; the grid dual is what the core exposes.
        # Brute force here is the breakpoint dual against an enumerated config LP
        # on the covariate grid with an extra outcome penalty folded into the loss
        # only through kappa_y in the dual. Compare breakpoint vs bisection.
        for eps in (0.01, 0.1, 0.5):
            a = float(
                _core.v1_robust_risk(
                    0.0,
                    shapes,
                    np.ascontiguousarray(bins.astype(np.int64)),
                    np.ascontiguousarray(y),
                    u,
                    np.ascontiguousarray(kappa),
                    eps,
                    "w1",
                    ky,
                    True,
                )
            )
            b = float(
                _core.v1_robust_risk(
                    0.0,
                    shapes,
                    np.ascontiguousarray(bins.astype(np.int64)),
                    np.ascontiguousarray(y),
                    u,
                    np.ascontiguousarray(kappa),
                    eps,
                    "w1",
                    ky,
                    False,
                )
            )
            worst = max(worst, abs(a - b))
    return {
        "max_abs_gap": float(worst),
        "bound_ok": bool(bound_ok),
        "pass": bool(worst <= 1e-8 and bound_ok),
    }


def t13() -> dict:
    wins = 0
    cert = 0
    inside = 0
    reps = 100
    for r in range(reps):
        rng = np.random.default_rng(500 + r)
        n = 300
        x = rng.random((n, 2))
        y = 0.5 * x[:, 0] + 0.3 * x[:, 1] + rng.normal(scale=0.2, size=n)
        edges_model = RobustV1(eps=0.0, max_bins=6, rounds=1).fit(x, y)
        edges = [np.asarray(e, dtype=np.float64) for e in edges_model.model_.edges()]
        kappa = np.asarray(edges_model.model_.kappa(), dtype=np.float64)
        width = float(kappa[0])
        eps = 0.2 * width
        robust = RobustV1(eps=eps, max_bins=6, rounds=30, step=0.4).fit(x, y, kappa=kappa)
        plain = RobustV1(eps=0.0, max_bins=6, rounds=30, step=0.4).fit(x, y, kappa=kappa)
        xb = x.copy()
        from traceot import primitives

        codes0 = np.asarray(
            primitives.bin_column(np.ascontiguousarray(x[:, 0]), np.ascontiguousarray(edges[0])),
            dtype=np.int64,
        )
        last = int(edges[0].size) + 1
        edge0 = edges[0]

        def _in_bin(code: int, e: np.ndarray, last_code: int) -> float:
            if code <= 1:
                step = float(e[1] - e[0]) if e.size > 1 else 0.1
                return float(e[0] - 0.5 * step)
            if code >= last_code:
                step = float(e[-1] - e[-2]) if e.size > 1 else 0.1
                return float(e[-1] + 0.5 * step)
            return float(0.5 * (e[code - 2] + e[code - 1]))

        for i in range(n):
            if rng.random() >= 0.2:
                continue
            c = int(codes0[i])
            dest = c + 1 if c < last else c - 1
            xb[i, 0] = _in_bin(dest, edge0, last)
        yb = y.copy()
        pb = float(robust.pinball(xb, yb))
        pp = float(plain.pinball(xb, yb))
        wins += pb <= pp + 1e-12

        def _codes(block, edges_):
            from traceot import primitives

            out = np.zeros((block.shape[0], 2), dtype=np.int64)
            for j in range(2):
                coded = primitives.bin_column(
                    np.ascontiguousarray(block[:, j]), np.ascontiguousarray(edges_[j])
                )
                out[:, j] = np.asarray(coded, dtype=np.int64)
            return out

        w1 = float(
            _core.v1_coupling_w1(
                np.ascontiguousarray(_codes(x, edges)),
                np.ascontiguousarray(_codes(xb, edges)),
                np.ascontiguousarray(kappa),
            )
        )
        inside += w1 <= eps + 1e-9
        cert += float(robust.model_.robust_value) + 1e-9 >= pb
    return {
        "win_fraction": wins / reps,
        "certificate_fraction": cert / reps,
        "inside_ball_fraction": inside / reps,
        "pass": bool(wins / reps >= 0.60 and cert / reps >= 0.95),
    }


def t14() -> dict:
    rng = np.random.default_rng(14)
    worst = 0.0
    for _ in range(20):
        eps = float(rng.uniform(0.05, 0.8))
        u = float(rng.uniform(0.1, 0.9))
        ky = float(rng.uniform(0.2, 2.0))
        gap = float(_core.v1_outcome_w1_gap(eps, u, ky))
        rho = max(u, 1.0 - u)
        worst = max(worst, abs(gap - eps * rho / ky))
    return {"max_abs": float(worst), "pass": bool(worst <= 1e-9)}


def capability() -> dict:
    """A two-period shift and a distribution-valued fit.

    An explainable boosting machine fits a conditional mean. It has no
    composition/structure split and no row-wise sample of a distribution.
    The numbers below are what TRACE returns. They are not an accuracy contest.
    """
    rng = np.random.default_rng(21)
    n = 800
    x1 = rng.random((n, 2))
    x2 = np.column_stack([rng.random(n) * 0.4 + 0.6, rng.random(n)])
    z1, z2 = _z(rng.random(n)), _z(rng.random(n))
    y1 = x1[:, 0] + 0.5 * z1
    y2 = x2[:, 0] + 0.5 * z2
    a = _fit_v1(x1, y1, k_fit=4, max_rounds=12, max_bins=8)
    b = _fit_v1(x2, y2, k_fit=4, max_rounds=12, max_bins=8)
    shift = a.shift_marginal(b, x1, x2, grid=64)
    x, samples, lengths, scale = _frechet_draw(400, 22)
    fr = FrechetV1(k_fit=2, max_bins=4).fit(x, samples, lengths)
    pred = np.asarray(fr.predict(x))[:, 1] / math.sqrt(3.0)
    rmse = float(np.sqrt(np.mean((pred - 0.5 * scale) ** 2)))
    return {
        "shift_total": float(shift["total"]),
        "composition": float(shift["composition"]),
        "structure": float(shift["structure"]),
        "cross": float(shift["cross"]),
        "frechet_lambda2_rmse": rmse,
        "note": "capability difference: a mean booster has neither object",
    }


TESTS = {
    "t1": t1,
    "t2": t2,
    "t3": t3,
    "t4": t4,
    "t5": t5,
    "t6": t6,
    "t7": t7,
    "t8": t8,
    "t9": t9,
    "t10": t10,
    "t11": t11,
    "t12": t12,
    "t13": t13,
    "t14": t14,
    "capability": capability,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    keys = [args.only] if args.only else list(TESTS)
    payload = {}
    if OUT.exists() and args.only:
        payload = json.loads(OUT.read_text())
    for key in keys:
        t0 = time.perf_counter()
        print(key, "start", flush=True)
        payload[key] = TESTS[key]()
        payload[key]["seconds"] = time.perf_counter() - t0
        flag = "pass" if payload[key].get("pass", True) else "FAIL"
        print(key, flag, f"{payload[key]['seconds']:.1f}s", flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(_finite(payload), indent=2, allow_nan=False))
    print(OUT)


if __name__ == "__main__":
    main()
