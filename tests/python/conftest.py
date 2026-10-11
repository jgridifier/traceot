import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(20261011)


def random_quantiles(rng, n, q, ties=False):
    """Random valid quantile functions (rows non-decreasing), optionally with ties."""
    steps = rng.exponential(1.0, size=(n, q))
    if ties:
        steps[rng.uniform(size=(n, q)) < 0.3] = 0.0
    return rng.normal(size=(n, 1)) + np.cumsum(steps, axis=1) / q
