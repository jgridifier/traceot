"""Parity of the v1 kernels against the numpy reference, absolute tolerance 1e-10."""

import numpy as np

from reference.trace_v1 import (
    crps_quadrature,
    fit_mean,
    gauss_legendre_01,
    hosking_lmoments,
    leave_feature_gains,
    max_jump_bound,
    pinball_gh,
    robust_risk_breakpoints,
    weighted_pav,
)
from traceot import _core


def test_gauss_legendre_matches_numpy():
    for n in (2, 5, 8, 16):
        nodes, weights = _core.v1_gauss_legendre(n)
        rn, rw = gauss_legendre_01(n)
        np.testing.assert_allclose(nodes, rn, atol=1e-10, rtol=0.0)
        np.testing.assert_allclose(weights, rw, atol=1e-10, rtol=0.0)


def test_hosking_matches_reference():
    rng = np.random.default_rng(0)
    x = np.ascontiguousarray(rng.normal(size=40))
    got = np.asarray(_core.v1_hosking(x, 4))
    np.testing.assert_allclose(got, hosking_lmoments(x, 4), atol=1e-10, rtol=0.0)


def test_weighted_pav_matches_reference():
    y = np.array([0.2, 1.5, 0.4, 0.3, 2.0, 1.2])
    w = np.array([1.0, 0.5, 2.0, 0.3, 1.0, 1.4])
    got = np.asarray(_core.v1_weighted_pav(np.ascontiguousarray(y), np.ascontiguousarray(w)))
    np.testing.assert_allclose(got, weighted_pav(y, w), atol=1e-10, rtol=0.0)


def test_robust_dual_matches_reference():
    shapes = [
        np.ascontiguousarray(np.array([0.0, 0.4, 0.1])),
        np.ascontiguousarray(np.array([0.2, -0.1, 0.5])),
    ]
    bins = np.ascontiguousarray(np.array([[0, 1], [2, 0], [1, 2], [0, 0]], dtype=np.int64))
    y = np.ascontiguousarray(np.array([0.2, -0.4, 1.0, 0.3]))
    kappa = np.ascontiguousarray(np.array([0.3, 0.3]))
    u = 0.8
    for eps in (0.01, 0.1, 0.5):
        for ball in ("w1", "w2"):
            got = _core.v1_robust_risk(0.0, shapes, bins, y, u, kappa, eps, ball, None, True)
            ref = robust_risk_breakpoints(0.0, shapes, bins, y, u, kappa, eps, ball)
            assert abs(got - ref) < 1e-8
        fast = _core.v1_robust_risk(0.0, shapes, bins, y, u, kappa, eps, "w1", None, False)
        exact = _core.v1_robust_risk(0.0, shapes, bins, y, u, kappa, eps, "w1", None, True)
        assert abs(fast - exact) < 1e-8
    emp = float(
        np.mean(
            [
                _core.v1_pinball(y[i] - (shapes[0][bins[i, 0]] + shapes[1][bins[i, 1]]), u)
                for i in range(len(y))
            ]
        )
    )
    ub = _core.v1_max_jump_bound(shapes, kappa, emp, 0.1, u)
    ref_ub = max_jump_bound(shapes, kappa, emp, 0.1, u)
    assert abs(ub - ref_ub) < 1e-12
    exact = _core.v1_robust_risk(0.0, shapes, bins, y, u, kappa, 0.1, "w1", None, True)
    assert exact <= ub + 1e-8


def test_outcome_w1_gap_and_cross_fit_sign():
    assert abs(_core.v1_outcome_w1_gap(0.2, 0.7, 0.5) - 0.28) < 1e-15
    energy = _core.v1_cross_fit_energy(
        np.ascontiguousarray(np.array([0.5, -0.25])),
        np.ascontiguousarray(np.array([-0.5, 1.0])),
    )
    assert energy < 0.0


def test_fitted_attribution_parseval():
    from traceot import TraceV1

    rng = np.random.default_rng(1)
    n = 60
    x = rng.uniform(size=(n, 2))
    u = rng.uniform(size=n)
    y = 0.3 * x[:, 0] + (0.6 + 0.4 * x[:, 1]) * (u - 0.5)
    model = TraceV1(k_fit=3, k_report=3, n_u=6, max_bins=4, max_rounds=8, mono="none").fit(x, y)
    attr = model.attribute(x)
    phi = attr["phi_fit"]
    totals = phi.sum(axis=(1, 2))
    # Pre-projection energy is the sum of Shapley shares.
    assert np.all(np.isfinite(totals))
    recon = model.predict(x, np.asarray(model.model_.u_nodes()))
    # Quadrature of a degree-2 polynomial matches the attributed energy.
    w = np.asarray(_core.v1_gauss_legendre(6)[1])
    quad = (recon**2 * w).sum(axis=1)
    np.testing.assert_allclose(totals, quad, atol=1e-8, rtol=1e-8)


def test_mean_booster_matches_reference():
    rng = np.random.default_rng(3)
    x = np.ascontiguousarray(rng.random((40, 2)))
    y = np.ascontiguousarray(0.4 + 1.2 * x[:, 0] - 0.7 * x[:, 1] + 0.05 * rng.normal(size=40))
    kw = dict(
        max_bins=6,
        learning_rate=0.3,
        max_rounds=12,
        early_stopping_rounds=4,
        early_stopping_tolerance=1e-6,
        validation_size=0.2,
        min_samples_leaf=2,
    )
    got = _core.v1_fit_mean(x, y, **kw)
    ref = fit_mean(x, y, **kw)
    assert got["rounds"] == ref["rounds"]
    assert got["best_round"] == ref["best_round"]
    np.testing.assert_allclose(got["intercept"], ref["intercept"], atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(got["prediction"], ref["prediction"], atol=1e-10, rtol=0.0)
    for a, b in zip(got["effects"], ref["effects"], strict=True):
        np.testing.assert_allclose(a, b, atol=1e-10, rtol=0.0)


def test_pinball_gh_matches_reference():
    rng = np.random.default_rng(5)
    k = 4
    nu = 6
    nodes, weights = gauss_legendre_01(nu)
    basis_e = np.ascontiguousarray(rng.normal(size=nu * k))
    basis_de = np.ascontiguousarray(rng.normal(size=nu * k))
    a = np.ascontiguousarray(rng.normal(size=k))
    for smooth, gamma in ((0.0, 0.0), (0.05, 0.0), (0.02, 2.0), (0.0, 1.5)):
        g, h = _core.v1_pinball_gh(
            np.ascontiguousarray(nodes),
            np.ascontiguousarray(weights),
            basis_e,
            basis_de,
            a,
            0.3,
            smooth,
            gamma,
        )
        rg, rh = pinball_gh(nodes, weights, basis_e, basis_de, a, 0.3, smooth, gamma)
        np.testing.assert_allclose(g, rg, atol=1e-10, rtol=0.0)
        np.testing.assert_allclose(h, rh, atol=1e-10, rtol=0.0)


def test_leave_feature_gains_match_reference():
    rng = np.random.default_rng(6)
    n, p, k, nu = 5, 2, 3, 4
    nodes, weights = gauss_legendre_01(nu)
    basis_e = np.ascontiguousarray(rng.normal(size=nu * k))
    players = np.ascontiguousarray(rng.normal(size=n * (p + 1) * k))
    y = np.ascontiguousarray(rng.normal(size=n))
    rows = np.ascontiguousarray(np.array([1, 3, 4], dtype=np.int64))
    got = _core.v1_leave_feature_gains(
        np.ascontiguousarray(nodes),
        np.ascontiguousarray(weights),
        basis_e,
        players,
        y,
        rows,
        p,
        k,
    )
    rg, rs = leave_feature_gains(nodes, weights, basis_e, players, y, rows, p, k)
    np.testing.assert_allclose(got["gains"], rg, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(got["shape_gains"], rs, atol=1e-10, rtol=0.0)


def test_crps_quadrature_matches_reference():
    rng = np.random.default_rng(4)
    y = np.ascontiguousarray(rng.normal(size=15))
    q = np.ascontiguousarray(rng.normal(size=(15, 5)))
    u = np.ascontiguousarray(np.array([0.1, 0.3, 0.5, 0.7, 0.9]))
    w = np.ascontiguousarray(np.array([0.1, 0.2, 0.4, 0.2, 0.1]))
    got = _core.v1_crps_quadrature(y, q, u, w)
    np.testing.assert_allclose(got, crps_quadrature(y, q, u, w), atol=1e-10, rtol=0.0)
