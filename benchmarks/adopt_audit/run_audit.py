"""Numerical checks of flaws in ADOPT's method (Song & Müller, AISTATS 2026).

Each block states a short derivation, runs a small simulation through the Rust
core, and checks that the optional fix improves the flaw without making a
commuting problem worse. Paper defaults stay available (composition='given',
centering='frechet', damping=1, certificate off, no binning, no reflection).

Usage::

    python benchmarks/adopt_audit/run_audit.py

Writes ``benchmarks/results/adopt_audit.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from traceot import ADOPTRegressor, primitives

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks" / "results" / "adopt_audit.json"


def w2(a, b) -> float:
    return float(np.mean(np.mean((a - b) ** 2, axis=1)))


def holdout(est, X, truth) -> float:
    return w2(est.predict(X), truth)


def fit(X, Y, h, **kw):
    opts = dict(max_iter=40, tol=1e-10)
    opts.update(kw)
    est = ADOPTRegressor(bandwidth=np.asarray(h, float), **opts)
    est.fit(X, Y)
    return est


def monotone_power(X, q):
    """Non-commuting maps on (0, 1): power, then a fractional linear map.

    f_a(y) = y**(0.45 + 1.1 a) and g_b(y) = (y + 0.7 b) / (1 + 0.7 b) are both
    increasing in y, but f ∘ g ≠ g ∘ f. The response applies f then g to Unif(0, 1).
    """
    u = np.asarray(primitives.quantile_grid(q))
    Y = np.empty((len(X), q))
    for i, (a, b) in enumerate(np.asarray(X)):
        y = np.power(np.clip(u, 1e-12, None), 0.45 + 1.1 * a)
        Y[i] = (y + 0.7 * b) / (1.0 + 0.7 * b)
    return Y


def location_response(X, q):
    """Translations commute, so feature order is irrelevant to the truth."""
    u = np.asarray(primitives.quantile_grid(q))
    return u + X[:, [0]] + X[:, [1]] ** 2


def grid2(n=8):
    g = (np.arange(n) + 0.5) / n
    A, B = np.meshgrid(g, g, indexing="ij")
    return np.column_stack([A.ravel(), B.ravel()])


def disagree(est_ab, est_ba, Xg) -> float:
    return w2(est_ab.predict(Xg), est_ba.predict(Xg[:, ::-1]))


def experiment_order(rng):
    # Analytic counterexample, evaluated at y = 0.
    # f(y) = y^2, g(y) = (y+1)/2. Both increasing on [0, 1].
    y = 0.0
    fg = ((y + 1.0) / 2.0) ** 2
    gf = (y**2 + 1.0) / 2.0
    assert abs(fg - gf) > 0.2

    n, q = 70, 24
    X = rng.uniform(size=(n, 2))
    Y = monotone_power(X, q)
    Xg = grid2(6)
    truth = monotone_power(Xg, q)
    h = [0.28, 0.28]
    given_ab = fit(X, Y, h)
    given_ba = fit(X[:, ::-1], Y, h)
    avg_ab = fit(X, Y, h, composition="average")
    avg_ba = fit(X[:, ::-1], Y, h, composition="average")
    tan_ab = fit(X, Y, h, composition="tangent")
    tan_ba = fit(X[:, ::-1], Y, h, composition="tangent")
    d_given = disagree(given_ab, given_ba, Xg)
    d_avg = disagree(avg_ab, avg_ba, Xg)
    d_tan = disagree(tan_ab, tan_ba, Xg)
    # Commuting location problem: the fixes must not inflate holdout error.
    Yloc = location_response(X, q)
    truth_loc = location_response(Xg, q)
    g_loc = fit(X, Yloc, h)
    a_loc = fit(X, Yloc, h, composition="average")
    t_loc = fit(X, Yloc, h, composition="tangent")
    e_g, e_a, e_t = (
        holdout(g_loc, Xg, truth_loc),
        holdout(a_loc, Xg, truth_loc),
        holdout(t_loc, Xg, truth_loc),
    )
    assert d_given > 1e-4, d_given
    assert d_avg < 0.25 * d_given, (d_avg, d_given)
    assert d_tan < 0.25 * d_given, (d_tan, d_given)
    assert e_a <= 1.25 * e_g + 1e-4
    assert e_t <= 1.25 * e_g + 1e-4
    return {
        "counterexample_y0": {"f_then_g": fg, "g_then_f": gf},
        "order_disagreement_w2": {"given": d_given, "average": d_avg, "tangent": d_tan},
        "location_holdout_w2": {"given": e_g, "average": e_a, "tangent": e_t},
        "noncommute_holdout_w2": {
            "given": holdout(given_ab, Xg, truth),
            "average": holdout(avg_ab, Xg, truth),
            "tangent": holdout(tan_ab, Xg, truth),
        },
    }


def experiment_centering(rng):
    """Eq. (11) already implies the barycentric identity.

    The piecewise-linear map is affine in its target knots, and outside the knot
    range it extends by a translation that is also affine in those knots. If
    ``M = mean_i g_i``, then ``mean_i T_{M, g_i}(z) = z`` for every ``z``.
    Pushforward centering is therefore redundant for that identity. What remains
    is ordinary additive non-identifiability: duplicate features can trade the
    whole effect. Averaged-order composition splits it symmetrically; the
    prediction is unchanged.
    """
    rng_id = np.random.default_rng(1)
    G = np.sort(rng_id.normal(size=(18, 15)), axis=1)
    M = G.mean(axis=0)
    z = np.sort(rng_id.normal(size=15))
    pushed = np.vstack([primitives.transport_apply(M, G[i], z) for i in range(len(G))])
    identity_gap = float(np.max(np.abs(pushed.mean(axis=0) - z)))
    assert identity_gap < 1e-8, identity_gap

    n, q = 50, 16
    x = rng.uniform(size=n)
    X = np.column_stack([x, x])
    u = np.asarray(primitives.quantile_grid(q))
    Y = u + 1.5 * x[:, None]
    h = [0.25, 0.25]
    xg = np.linspace(0.0, 1.0, 12)
    Xg = np.column_stack([xg, xg])
    truth = u + 1.5 * xg[:, None]

    def asymmetry(est):
        g0 = est.component(0, x)["g"]
        g1 = est.component(1, x)["g"]
        return float(np.mean((g0 - g1) ** 2))

    given = fit(X, Y, h)
    avg = fit(X, Y, h, composition="average")
    pus = fit(X, Y, h, centering="pushforward")
    a_given, a_avg, a_pus = asymmetry(given), asymmetry(avg), asymmetry(pus)
    e_given, e_avg, e_pus = (
        holdout(given, Xg, truth),
        holdout(avg, Xg, truth),
        holdout(pus, Xg, truth),
    )
    assert a_given > 1e-2, a_given
    assert a_avg < 1e-8, a_avg
    assert e_avg <= e_given + 1e-8
    assert e_pus <= e_given + 1e-6
    # Pushforward does not remove the duplicate-feature split.
    assert a_pus > 1e-2, a_pus
    return {
        "identity_gap_max_abs": identity_gap,
        "duplicate_component_asymmetry": {"given": a_given, "average": a_avg, "pushforward": a_pus},
        "duplicate_holdout_w2": {"given": e_given, "average": e_avg, "pushforward": e_pus},
    }


def in_sample(est, Y) -> float:
    return w2(np.asarray(est.model_.fitted), Y)


def experiment_convergence(_rng):
    """Correlated features, small bandwidth: in-sample W2 rises after iteration 1.

    A fixed point of Gauss–Seidel plus isotonic projection is not guaranteed to
    be a descent method. The certificate keeps the last accepted cycle.
    """
    rng = np.random.default_rng(0)
    n, q = 40, 12
    x = rng.uniform(size=n)
    X = np.column_stack([x, np.clip(x + 0.02 * rng.normal(size=n), 0.0, 1.0)])
    u = np.asarray(primitives.quantile_grid(q))
    Y = u + np.sin(4.0 * np.pi * x)[:, None] + (x**2)[:, None]
    h = [0.05, 0.05]
    losses = []
    for t in range(1, 9):
        est = ADOPTRegressor(bandwidth=np.asarray(h, float), max_iter=t, tol=0.0)
        est.fit(X, Y)
        losses.append(in_sample(est, Y))
    increases = [float(losses[i + 1] - losses[i]) for i in range(len(losses) - 1)]
    cert = primitives.adopt_w2_fit(X, Y, h, max_iter=8, tol=0.0, certificate=True)
    cert_loss = float(np.mean((cert["fitted"] - Y) ** 2))
    assert max(increases) > 1e-4, increases
    assert cert["n_rejected"] > 0
    assert cert_loss <= min(losses) + 1e-12
    assert cert_loss < losses[-1]

    # Stopping threshold, on a smooth location problem where the loss does descend.
    Xloc = rng.uniform(size=(n, 2))
    Yloc = location_response(Xloc, q)
    Xg = grid2(5)
    truth = location_response(Xg, q)
    h_loc = [0.2, 0.2]
    by_tol = {}
    for tol in (1e-2, 1e-4, 1e-8, 1e-12):
        est = fit(Xloc, Yloc, h_loc, tol=tol, max_iter=80)
        by_tol[f"{tol:.0e}"] = {"n_iter": int(est.n_iter_), "holdout_w2": holdout(est, Xg, truth)}
    n_iters = [by_tol[k]["n_iter"] for k in ("1e-02", "1e-04", "1e-08", "1e-12")]
    tight = by_tol["1e-08"]["holdout_w2"]
    tighter = by_tol["1e-12"]["holdout_w2"]
    # Iteration count keeps growing as the threshold tightens. Holdout on this
    # smooth problem has already flattened by 1e-8.
    assert n_iters == sorted(n_iters) and n_iters[0] < n_iters[-1]
    assert abs(tighter - tight) <= 0.01 * tight + 1e-6
    damped = fit(Xloc, Yloc, h_loc, damping=0.5, certificate=True)
    plain = fit(Xloc, Yloc, h_loc)
    assert holdout(damped, Xg, truth) <= 1.25 * holdout(plain, Xg, truth) + 1e-4
    return {
        "undamped_insample_loss": losses,
        "loss_increments": increases,
        "certificate_insample_loss": cert_loss,
        "n_rejected": int(cert["n_rejected"]),
        "certificate_n_iter": int(cert["n_iter"]),
        "tol_sensitivity": by_tol,
        "location_holdout_w2": {
            "given": holdout(plain, Xg, truth),
            "damped_certificate": holdout(damped, Xg, truth),
        },
    }


def n_decreases(A) -> int:
    d = np.diff(A, axis=1)
    return int(np.sum(d < -1e-8))


def experiment_monotone(rng):
    # Composition of increasing maps is increasing. A sum of centered increasing
    # curves need not be: quantile crossing lives in the tangent representation.
    u = np.asarray(primitives.quantile_grid(30))
    raw = u + 0.2 * np.sin(4.0 * np.pi * u)
    assert n_decreases(raw[None, :]) > 0
    fixed = np.asarray(primitives.pava(raw))
    assert n_decreases(fixed[None, :]) == 0
    assert np.mean((fixed - u) ** 2) <= np.mean((raw - u) ** 2) + 1e-15

    n, q = 80, 30
    X = rng.uniform(size=(n, 2))
    # A wiggle in the quantile direction is not increasing. Training responses are
    # projected onto monotone functions; the tangent-space sum can still cross.
    u = np.asarray(primitives.quantile_grid(q))
    raw_y = u + (0.35 * np.sin(3.0 * np.pi * u)) * np.sin(2.0 * np.pi * X[:, [0]])
    Y = np.asarray(primitives.pava_rows(raw_y))
    h = [0.3, 0.3]
    tan = fit(X, Y, h, composition="tangent")
    given = fit(X, Y, h, composition="given", project_prediction=False)
    projected = fit(X, Y, h, composition="given", project_prediction=True)
    Xn = rng.uniform(size=(30, 2))
    raw_tan = np.asarray(
        primitives.additive_euclid_predict(
            X, np.asarray(tan.model_.residuals), np.asarray(tan.model_.centers), tan.mean_, h, Xn
        )
    )
    pred_tan = tan.predict(Xn)
    pred_given = given.predict(Xn)
    pred_proj = projected.predict(Xn)
    # Translation extension of an increasing knot map stays increasing.
    a = np.asarray(primitives.quantile_grid(12))
    b = a**2
    ext = np.asarray(primitives.transport_apply(a, b, np.array([a[0] - 1.0, a[-1] + 1.0])))
    assert abs(ext[0] - ((a[0] - 1.0) + (b[0] - a[0]))) < 1e-12
    assert abs(ext[1] - ((a[-1] + 1.0) + (b[-1] - a[-1]))) < 1e-12
    assert ext[1] > ext[0]
    cross_raw = n_decreases(raw_tan)
    assert cross_raw > 0, cross_raw
    assert n_decreases(pred_tan) == 0
    assert n_decreases(pred_given) == 0
    assert n_decreases(pred_proj) == 0
    # Already-monotone given predictions are unchanged by the extra projection.
    assert np.allclose(pred_given, pred_proj, atol=1e-8)
    return {
        "analytic_crossing_decreases": n_decreases(raw[None, :]),
        "tangent_raw_decreases": cross_raw,
        "tangent_projected_decreases": n_decreases(pred_tan),
        "given_decreases": n_decreases(pred_given),
        "extension_endpoints": ext.tolist(),
    }


def experiment_scaling():
    """Dense local-linear backfitting is quadratic; equal-width bins are linear."""
    q = 12
    rows = {}
    prev_dense = None
    prev_bin = None
    for n in (200, 400, 800, 1600):
        rng = np.random.default_rng(n)
        X = rng.uniform(size=(n, 2))
        Y = location_response(X, q)
        h = [0.2, 0.2]
        t0 = time.perf_counter()
        dense = fit(X, Y, h, max_iter=3, tol=0.0)
        td = time.perf_counter() - t0
        t0 = time.perf_counter()
        binned = fit(X, Y, h, max_iter=3, tol=0.0, bins=32)
        tb = time.perf_counter() - t0
        Xg = grid2(5)
        truth = location_response(Xg, q)
        ed, eb = holdout(dense, Xg, truth), holdout(binned, Xg, truth)
        rows[str(n)] = {
            "dense_seconds": td,
            "binned_seconds": tb,
            "dense_holdout_w2": ed,
            "binned_holdout_w2": eb,
        }
        if prev_dense is not None:
            rows[str(n)]["dense_growth"] = td / prev_dense
            rows[str(n)]["binned_growth"] = tb / prev_bin
        prev_dense, prev_bin = td, tb
        # Binning stays accurate on this smooth location problem.
        assert eb <= 3.0 * ed + 1e-3, (n, eb, ed)
    # From n=200 to n=1600 is an 8× increase. Dense should grow much faster than bins.
    growth_d = rows["1600"]["dense_seconds"] / rows["200"]["dense_seconds"]
    growth_b = rows["1600"]["binned_seconds"] / rows["200"]["binned_seconds"]
    assert growth_d > 2.5 * growth_b, (growth_d, growth_b)
    rows["growth_200_to_1600"] = {"dense": growth_d, "binned": growth_b}
    return rows


def _nw_weights(x, xe, h):
    """Nadaraya–Watson weights. Benchmark-only; the runtime smoother is local linear."""
    d = (xe[:, None] - x[None, :]) / h
    k = np.exp(-0.5 * d * d)
    return k / k.sum(axis=1, keepdims=True)


def experiment_boundary(rng):
    """Local linear is the boundary correction; even reflection is not.

    Nadaraya–Watson has an O(h) boundary bias. Local linear cancels that term,
    so edge error drops to the same order as the interior. Even reflection of
    the sample (``boundary='reflect'``) assumes a flat derivative at the
    endpoint: it breaks exact fits of straight lines and, on ``exp(3x)``,
    makes the edge worse. It stays available and off by default.
    """
    x = np.linspace(0.05, 0.95, 30)
    xe = np.array([0.0, 0.5, 1.0])
    y = (2.0 * x + 1.0)[:, None]
    truth_line = 2.0 * xe + 1.0
    plain = primitives.local_linear_weights(x, xe, 0.2) @ y
    reflected = primitives.reflected_local_linear_weights(x, xe, 0.2) @ y
    line_err = {
        "local_linear": float(np.max(np.abs(plain.ravel() - truth_line))),
        "reflect": float(np.max(np.abs(reflected.ravel() - truth_line))),
    }
    assert line_err["local_linear"] < 1e-10
    assert line_err["reflect"] > 0.1

    n, h = 80, 0.18
    xe = np.linspace(0.0, 1.0, 41)
    edge = (xe <= 0.08) | (xe >= 0.92)
    interior = (xe >= 0.3) & (xe <= 0.7)
    buckets = {k: [] for k in ("nw_edge", "nw_mid", "ll_edge", "ll_mid", "ref_edge", "ref_mid")}
    for _rep in range(16):
        x = rng.uniform(size=n)
        z = np.exp(3.0 * x)[:, None]
        truth = np.exp(3.0 * xe)[:, None]
        preds = {
            "nw": _nw_weights(x, xe, h) @ z,
            "ll": primitives.local_linear_weights(x, xe, h) @ z,
            "ref": primitives.reflected_local_linear_weights(x, xe, h) @ z,
        }
        for name, pred in preds.items():
            se = (pred - truth) ** 2
            buckets[f"{name}_edge"].append(float(se[edge].mean()))
            buckets[f"{name}_mid"].append(float(se[interior].mean()))
    mse = {k: float(np.mean(v)) for k, v in buckets.items()}
    assert mse["ll_edge"] < 0.2 * mse["nw_edge"]
    assert mse["ll_mid"] <= 1.15 * mse["nw_mid"] + 1e-6
    # Reflection is not an improvement on this curved target.
    assert mse["ref_edge"] > mse["ll_edge"]
    return {"line_max_abs_error": line_err, "exp_mse": mse, "n": n, "h": h, "replicates": 16}


def main():
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    out = {
        "order": experiment_order(rng),
        "centering": experiment_centering(rng),
        "convergence": experiment_convergence(rng),
        "monotone": experiment_monotone(rng),
        "scaling": experiment_scaling(),
        "boundary": experiment_boundary(np.random.default_rng(1)),
    }
    out["wall_seconds"] = time.perf_counter() - t0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
