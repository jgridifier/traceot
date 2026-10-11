"""Properties of the reference smoothers."""

import numpy as np
import pytest

from reference import global_frechet_weights, kernel_values, local_linear_weights


@pytest.mark.parametrize("kernel", ["gaussian", "epanechnikov"])
def test_local_linear_reproduces_linear_functions(rng, kernel):
    x = np.sort(rng.uniform(size=40))
    y = 1.5 - 2.0 * x
    xe = np.linspace(-0.1, 1.1, 13)
    W = local_linear_weights(x, xe, 0.25, kernel)
    assert np.allclose(W.sum(axis=1), 1.0)
    assert np.allclose(W @ y, 1.5 - 2.0 * xe)


def test_local_linear_matches_closed_form(rng):
    """Row equals K_h(X_i - x)[u2 - u1 (X_i - x)] / (n sigma0^2) with normalised kernel."""
    x = rng.uniform(size=25)
    h, x0 = 0.2, 0.4
    d = x - x0
    K = np.exp(-0.5 * (d / h) ** 2) / (np.sqrt(2 * np.pi) * h)
    u0, u1, u2 = (np.mean(K * d**k) for k in range(3))
    s = K * (u2 - u1 * d) / (u0 * u2 - u1**2)
    W = local_linear_weights(x, [x0], h)
    assert np.allclose(W[0], s / len(x))


def test_degenerate_fallbacks():
    x = np.array([0.0, 1.0, 2.0])
    W = local_linear_weights(x, [10.0], 0.1, "epanechnikov")
    assert np.array_equal(W[0], [0.0, 0.0, 1.0])
    W = local_linear_weights(x, [0.0], 0.6, "epanechnikov")  # only one point in window
    assert np.allclose(W[0], [1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        local_linear_weights(x, [0.0], 0.0)
    with pytest.raises(ValueError):
        kernel_values(np.zeros(2), "box")


def test_global_weights_reproduce_linear(rng):
    X = rng.normal(size=(30, 3))
    beta = np.array([1.0, -2.0, 0.5])
    y = 0.3 + X @ beta
    Xe = rng.normal(size=(5, 3))
    W = global_frechet_weights(X, Xe)
    assert np.allclose(W.sum(axis=1), 1.0)
    assert np.allclose(W @ y, 0.3 + Xe @ beta)
