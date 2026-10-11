"""Properties of the reference 1-D W2 routines."""

import itertools

import numpy as np
import pytest

from conftest import random_quantiles
from reference import collapse_knots, pava, pava_rows, quantile_grid, transport_apply, w2_sq


def brute_isotonic(y):
    """Exact L2 isotonic fit by checking all block partitions (tiny n only)."""
    n = len(y)
    best, best_val = None, np.inf
    for cuts in itertools.product([0, 1], repeat=n - 1):
        blocks, start = [], 0
        for k, c in enumerate(cuts, start=1):
            if c:
                blocks.append((start, k))
                start = k
        blocks.append((start, n))
        fit = np.concatenate([np.full(b - a, np.mean(y[a:b])) for a, b in blocks])
        if np.all(np.diff(fit) >= -1e-12):
            val = np.sum((fit - y) ** 2)
            if val < best_val:
                best, best_val = fit, val
    return best


def test_quantile_grid():
    assert np.allclose(quantile_grid(4), [0.125, 0.375, 0.625, 0.875])
    with pytest.raises(ValueError):
        quantile_grid(0)


def test_w2_sq_closed_form():
    u = quantile_grid(2000)
    # N(0,1) vs N(1,1) truncated grids differ by a constant shift of 1 -> W2^2 = 1
    assert np.isclose(w2_sq(u, u + 1.0), 1.0)
    # scale change of Unif(0,1) about 1/2: W2^2 = (s-1)^2 Var(U) = (s-1)^2/12
    s = 3.0
    assert np.isclose(w2_sq(u, 0.5 + s * (u - 0.5)), (s - 1) ** 2 / 12, rtol=1e-5)


def test_pava_matches_brute_force(rng):
    for _ in range(40):
        y = rng.normal(size=7)
        assert np.allclose(pava(y), brute_isotonic(y), atol=1e-12)


def test_pava_properties(rng):
    y = rng.normal(size=50)
    p = pava(y)
    assert np.all(np.diff(p) >= 0)
    assert np.isclose(p.sum(), y.sum())
    assert np.array_equal(pava(np.sort(y)), np.sort(y))
    pc = pava(y, lower=-0.5, upper=0.5)
    assert pc.min() >= -0.5 and pc.max() <= 0.5 and np.all(np.diff(pc) >= 0)
    assert pava_rows(np.vstack([y, -y])).shape == (2, 50)


def test_collapse_knots():
    a, b = collapse_knots([0.0, 1.0, 1.0, 1.0, 2.0], [0.0, 1.0, 2.0, 3.0, 5.0])
    assert np.array_equal(a, [0.0, 1.0, 2.0])
    assert np.array_equal(b, [0.0, 2.0, 5.0])


def test_transport_hits_knots_and_is_monotone(rng):
    a, b = random_quantiles(rng, 2, 30)
    assert np.allclose(transport_apply(a, b, a), b)
    y = np.sort(rng.normal(size=200) * 3)
    t = transport_apply(a, b, y)
    assert np.all(np.diff(t) >= -1e-12)


def test_transport_inverse_and_identity(rng):
    a, b = random_quantiles(rng, 2, 25)
    y = rng.normal(size=100) * 2
    assert np.allclose(transport_apply(b, a, transport_apply(a, b, y)), y)
    assert np.allclose(transport_apply(a, a, y), y)


def test_transport_translation_extension():
    a = np.array([0.0, 1.0])
    b = np.array([10.0, 12.0])
    assert np.allclose(transport_apply(a, b, [-1.0, 3.0]), [9.0, 14.0])
    assert np.allclose(transport_apply([1.0, 1.0], [2.0, 4.0], [0.0, 5.0]), [2.0, 7.0])


def test_transport_with_ties_is_generalised_inverse(rng):
    a = random_quantiles(rng, 1, 40, ties=True)[0]
    b = random_quantiles(rng, 1, 40)[0]
    out = transport_apply(a, b, a)
    au, bu = collapse_knots(a, b)
    # every tied block maps to the average of its targets
    for v, w in zip(au, bu, strict=True):
        assert np.allclose(out[a == v], w)


def test_pushforward_of_quantiles_is_ot_geodesic_endpoint(rng):
    """T_{v1,v2}(v1) == v2 on the grid (OT map pushes v1 onto v2)."""
    v1, v2 = random_quantiles(rng, 2, 50)
    assert np.allclose(transport_apply(v1, v2, v1), v2)
