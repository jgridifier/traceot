"""TRACE_V5_2_SPECIFIED location model. Non-gate seeds only (below 1000)."""

import numpy as np
import pytest

import traceot

BETA = np.array([1.0, -0.8, 0.6, 0.4, -0.3])


def specified_design(x):
    """Numpy reference for the frozen design (raw, uncentred)."""
    x = np.asarray(x, float)
    return np.column_stack([np.ones(x.shape[0]), x, x[:, 2] ** 2, np.sin(x[:, 4])])


def true_mu(x, kappa=1.0):
    x = np.asarray(x, float)
    g = (x[:, 2] ** 2 - 1.0) / np.sqrt(2.0) + np.sqrt(2.0) * np.sin(x[:, 4])
    return 1.0 + x[:, :5] @ BETA + kappa * g


def test_explain_v5_2_specified_matches_mean_model_string():
    rng = np.random.default_rng(17)
    n, p = 400, 5
    xa = rng.standard_normal((n, p))
    xb = rng.standard_normal((n, p))
    xb[:, 0] += 0.5
    ya = true_mu(xa) + 0.4 * rng.standard_normal(n)
    yb = true_mu(xb) + 0.4 * rng.standard_normal(n)
    r = traceot.explain_v5_2_specified(xa, ya, xb, yb)
    r2 = traceot.explain_v5(xa, ya, xb, yb, mean_model="specified")
    assert r["lsplit_comp"] == r2["lsplit_comp"]
    assert r["lsplit_struct"] == r2["lsplit_struct"]
    assert r["dropped_cols_a"] == [] and r["dropped_cols_b"] == []
    tot = np.subtract(r["bb"]["lmom"], r["aa"]["lmom"])
    np.testing.assert_allclose(np.add(r["lsplit_comp"], r["lsplit_struct"]), tot, atol=1e-10)
    with pytest.raises(ValueError, match="specified"):
        traceot.explain_v5(xa, ya, xb, yb, mean_model="tree")


def test_specified_family_contains_true_mean_parity_with_numpy():
    """On noiseless draws the specified fit recovers μ (family is correct by construction).

    Seed 23, below 1000, not a gate index.
    """
    rng = np.random.default_rng(23)
    n, p = 500, 5
    x = rng.standard_normal((n, p))
    y = true_mu(x)
    # A one-period "fit" via explain on two copies: check residuals vanish.
    r = traceot.explain_v5(x, y, x, y, mean_model="specified")
    # budget is exact and total change is ~0
    tot = np.subtract(r["bb"]["lmom"], r["aa"]["lmom"])
    np.testing.assert_allclose(tot, 0.0, atol=1e-10)
    da = specified_design(x)
    coef, *_ = np.linalg.lstsq(da, y, rcond=None)
    mu_np = da @ coef
    np.testing.assert_allclose(mu_np, y, atol=1e-10)
    # recovered coefficients
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    np.testing.assert_allclose(coef[0], 1.0 - inv_sqrt2, atol=1e-10)
    np.testing.assert_allclose(coef[1:6], BETA, atol=1e-10)
    np.testing.assert_allclose(coef[6], inv_sqrt2, atol=1e-10)
    np.testing.assert_allclose(coef[7], np.sqrt(2.0), atol=1e-10)


def test_specified_needs_p_at_least_5():
    rng = np.random.default_rng(43)
    n, p = 80, 4
    xa = rng.standard_normal((n, p))
    ya = rng.standard_normal(n)
    with pytest.raises((ValueError, RuntimeError)):
        traceot.explain_v5(xa, ya, xa, ya, mean_model="specified")


def test_specified_is_not_the_default_and_differs_from_ols_on_curved_mean():
    rng = np.random.default_rng(47)
    n, p = 800, 5
    xa = rng.standard_normal((n, p))
    xb = rng.standard_normal((n, p))
    xb[:, 0] += 0.6
    ya = true_mu(xa) + 0.3 * rng.standard_normal(n)
    yb = true_mu(xb) + 0.3 * rng.standard_normal(n)
    rs = traceot.explain_v5(xa, ya, xb, yb, mean_model="specified")
    ro = traceot.explain_v5_1(xa, ya, xb, yb)
    assert rs["lsplit_comp"] != ro["lsplit_comp"]
    # default remains OLS
    rd = traceot.explain_v5(xa, ya, xb, yb)
    assert rd["lsplit_comp"] == ro["lsplit_comp"]


def test_specified_linear_mean_check_is_report_only():
    rng = np.random.default_rng(53)
    n, p = 600, 5
    xa = rng.standard_normal((n, p))
    xb = rng.standard_normal((n, p))
    ya = true_mu(xa) + 0.4 * rng.standard_normal(n)
    yb = true_mu(xb) + 0.4 * rng.standard_normal(n)
    r1 = traceot.explain_v5(xa, ya, xb, yb, mean_model="specified")
    r2 = traceot.explain_v5(xa, ya, xb, yb, mean_model="specified")
    assert r1["lsplit_comp"] == r2["lsplit_comp"]
    assert r1["linear_mean_check"] == r2["linear_mean_check"]
    assert r1["linear_mean_check"]["a"]["diagnostic_only"] is True
    # family is correct, so period-A curvature signal should be small
    assert abs(r1["linear_mean_check"]["a"]["corr_resid_fit_sq"]) < 0.15
