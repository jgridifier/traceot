"""Behaviour of the reference ADOPT transport backfitting."""

import numpy as np
import pytest

from traceot.reference import (
    additive_euclid_fit,
    additive_euclid_predict,
    adopt_w2_fit,
    adopt_w2_predict,
    coords_to_spd,
    quantile_grid,
    spd_to_coords,
    transport_apply,
    w2_sq,
)
from traceot.reference.adopt import fitted_values, partial_residuals


def toy(rng, n=60, q=25):
    X = rng.uniform(size=(n, 2))
    u = quantile_grid(q)
    Y = (X[:, [0]] - 0.5) + (1.0 + 0.5 * X[:, [1]]) * (u - 0.5)
    return X, Y


def test_partial_residuals_telescope_j0_exact(rng):
    """With noise-free Y = [T_1 ⊕ T_2 ⊕ T_3](mu), P_1(mu) == T_1(mu) exactly (eq. 10).

    For j = 1 nothing is applied before T_{mu,Y}, which is evaluated at its own knots,
    and the remaining inverse maps are exact inverses of piecewise-linear bijections.
    """
    n, q, p = 8, 30, 3
    mu = np.sort(rng.normal(size=q))
    G = np.sort(rng.normal(size=(p, n, q)), axis=2) + mu
    M = G.mean(axis=1)
    active = [True] * p
    Y = fitted_values(G, M, mu, active)
    P = partial_residuals(Y, G, M, mu, active, 0)
    Tj_mu = np.vstack([transport_apply(M[0], G[0, i], mu) for i in range(n)])
    assert np.allclose(P, Tj_mu, atol=1e-10)


def test_partial_residuals_telescope_translations(rng):
    """For translation maps the telescoping identity P_j(mu) = T_j(mu) holds for every j."""
    n, q, p = 6, 20, 3
    mu = np.sort(rng.normal(size=q))
    G = mu + rng.normal(size=(p, n, 1))
    M = G.mean(axis=1)
    active = [True] * p
    Y = fitted_values(G, M, mu, active)
    for j in range(p):
        P = partial_residuals(Y, G, M, mu, active, j)
        assert np.allclose(P, mu + (G[j] - M[j]), atol=1e-10)


def test_partial_residuals_telescope_smooth_maps_converge(rng):
    """For smooth non-commuting maps, the discretisation error of eq. (10) vanishes as q grows."""
    errs = []
    for q in (25, 100, 400):
        u = quantile_grid(q)
        mu = u
        # T_1: power map, T_2: shift, T_3: affine scale about 1/2 (as functions of mu)
        G = np.stack([u**1.5, u + 0.2, 0.5 + 0.7 * (u - 0.5)])[:, None, :]
        M = mu[None, :].repeat(3, 0)
        Y = fitted_values(G, M, mu, [True] * 3)
        P = partial_residuals(Y, G, M, mu, [True] * 3, 1)
        errs.append(np.max(np.abs(P[0] - G[1, 0])))
    assert errs[0] < 2e-2 and errs[-1] < errs[0]


def test_adopt_converges_and_predicts(rng):
    X, Y = toy(rng)
    fit = adopt_w2_fit(X, Y, [0.2, 0.2])
    assert fit["converged"]
    assert np.all(np.diff(fit["fitted"], axis=1) >= -1e-12)
    V = np.mean(w2_sq(Y, fit["mu"]))
    assert np.mean(w2_sq(fit["fitted"], Y)) < 0.02 * V
    pred = adopt_w2_predict(X, fit["P"], fit["M"], fit["mu"], [0.2, 0.2], X)
    assert np.allclose(pred, fit["fitted"])
    # transport centring: Fréchet mean of g_j(X_ij) equals M_j
    assert np.allclose(fit["G"].mean(axis=1), fit["M"])


def test_adopt_constant_response_is_fixed_point(rng):
    X = rng.uniform(size=(20, 2))
    Y = np.tile(np.linspace(0, 1, 10), (20, 1))
    fit = adopt_w2_fit(X, Y, [0.3, 0.3])
    assert np.allclose(fit["fitted"], Y)
    assert fit["converged"]


def test_bounds_respected(rng):
    X, Y = toy(rng)
    Y = np.clip(Y + 0.5, 0, 1)
    fit = adopt_w2_fit(X, Y, [0.1, 0.1], lower=0.0, upper=1.0)
    assert fit["G"].min() >= 0.0 and fit["G"].max() <= 1.0


def test_additive_matches_adopt_for_pure_location_effects(rng):
    """If every effect is a location shift, W2 transports are translations and ADOPT
    coincides with Euclidean backfitting on the quantile vectors (up to projection)."""
    n, q = 50, 20
    X = rng.uniform(size=(n, 2))
    u = quantile_grid(q)
    Y = u + np.sin(3 * X[:, [0]]) + X[:, [1]] ** 2
    a = adopt_w2_fit(X, Y, [0.15, 0.15])
    b = additive_euclid_fit(X, Y, [0.15, 0.15])
    assert np.allclose(a["fitted"], b["fitted"], atol=1e-6)
    pb = additive_euclid_predict(X, b["P"], b["M"], b["mu"], [0.15, 0.15], X)
    assert np.allclose(pb, b["fitted"])


def test_logchol_roundtrip(rng):
    A = rng.normal(size=(4, 5, 5))
    S = A @ np.swapaxes(A, -1, -2) + 0.1 * np.eye(5)
    z = spd_to_coords(S)
    assert z.shape == (4, 15)
    assert np.allclose(coords_to_spd(z, 5), S)


@pytest.mark.parametrize("bad_h", [[0.2], [0.2, 0.0]])
def test_bad_bandwidths(rng, bad_h):
    X, Y = toy(rng, n=10, q=5)
    with pytest.raises(ValueError):
        adopt_w2_fit(X, Y, bad_h)
