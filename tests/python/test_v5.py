"""TRACE v5 mixture counterfactual: thin-binding smoke tests and DFL toy check."""

import numpy as np

import traceot


def test_mixture_cdf_location_matches_brute_force():
    mu = np.array([0.3, -1.0, 2.0])
    r = np.array([-0.5, 0.1, 0.7, 0.0])
    grid = np.linspace(-3, 4, 31)
    f = traceot.mixture_cdf_location(mu, r, grid)
    brute = np.array([np.mean((mu[:, None] + r[None, :]) <= y) for y in grid])
    np.testing.assert_allclose(f, brute, atol=1e-15)


def test_quantile_inverts():
    grid = np.linspace(-2, 2, 101)
    q = traceot.mixture_quantile(grid, (grid + 2) / 4, np.array([0.1, 0.5, 0.9]))
    np.testing.assert_allclose(q, [-1.6, 0.0, 1.6], atol=1e-12)


def test_mixture_matches_dfl_reweighting():
    """F_[B,A] (covariates B, conditional A) == DFL (1996) reweighting of sample A."""
    rng = np.random.default_rng(1996)
    pa, pb = np.array([0.5, 0.3, 0.2]), np.array([0.2, 0.3, 0.5])
    n = 40_000
    ga = rng.choice(3, n, p=pa)
    gb = rng.choice(3, n, p=pb)
    z = rng.standard_normal(n)
    ya = np.select([ga == 0, ga == 1], [z, 1.5 + 0.5 * z * z], -1.0 + 2.0 * z)
    u = (np.arange(1000) + 0.5) / 1000
    qg = np.stack([np.quantile(ya[ga == g], u, method="inverted_cdf") for g in range(3)])
    grid = np.linspace(-6, 8, 71)
    f_mix = traceot.mixture_cdf_rows(u, qg[gb], grid)
    fa = np.bincount(ga, minlength=3) / n
    fb = np.bincount(gb, minlength=3) / n
    w = (fb / fa)[ga]
    f_dfl = np.array([w[ya <= y].sum() for y in grid]) / w.sum()
    assert np.max(np.abs(f_mix - f_dfl)) < 1e-2


def test_explain_v5_budget_is_exact():
    rng = np.random.default_rng(0)
    xa = rng.standard_normal((3000, 3))
    xb = rng.standard_normal((3000, 3)) + [0.5, 0, 0]
    ya = xa[:, 0] + rng.standard_normal(3000)
    yb = xb[:, 0] + 1.3 * rng.standard_normal(3000)
    r = traceot.explain_v5(xa, ya, xb, yb)
    tot = np.array(r["bb"]["lmom"]) - np.array(r["aa"]["lmom"])
    np.testing.assert_allclose(np.add(r["lsplit_comp"], r["lsplit_struct"]), tot, atol=1e-12)
    assert r["lsplit_comp"][0] > 0.3  # mean moved through x0's shift
    assert r["lsplit_struct"][1] > 0.05  # wider noise moves λ2 through the response
