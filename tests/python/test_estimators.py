"""Scikit-learn-style estimator API."""

import numpy as np
import pytest

from traceot import (
    HAVE_RUST,
    ADOPTRegressor,
    FrechetMeanRegressor,
    GlobalFrechetRegressor,
    L2AdditiveQuantileRegressor,
)
from traceot.reference import quantile_grid


def w2_data(rng, n=60, q=30):
    X = rng.uniform(size=(n, 2))
    u = quantile_grid(q)
    Y = (X[:, [0]] - 0.5) + (1.0 + X[:, [1]]) * (u - 0.5)
    Y = Y + 0.02 * rng.normal(size=(n, 1))
    return X, Y


def spd_data(rng, n=50, m=3):
    X = rng.uniform(size=(n, 2))
    A = np.eye(m)[None] * (1 + X[:, [0]])[:, :, None]
    B = 0.3 * X[:, 1][:, None, None] * np.ones((1, m, m))
    S = A @ A + B + 0.2 * np.eye(m)
    return X, S


def test_adopt_w2_fit_predict_score(rng):
    X, Y = w2_data(rng)
    est = ADOPTRegressor(bandwidth_grid=[0.1, 0.3], cv=3).fit(X, Y)
    pred = est.predict(X[:5])
    assert pred.shape == (5, Y.shape[1])
    assert np.all(np.diff(pred, axis=1) >= -1e-12)
    assert est.score(X, Y) > 0.9
    assert est.cv_results_["best_fraction"] in (0.1, 0.3)
    assert est.converged_ and est.n_iter_ >= 1
    assert est.bandwidths_.shape == (2,)


def test_fixed_bandwidth_and_params(rng):
    X, Y = w2_data(rng)
    est = ADOPTRegressor(bandwidth=0.2)
    assert est.get_params()["bandwidth"] == 0.2
    est.set_params(bandwidth=[0.2, 0.25]).fit(X, Y)
    assert np.allclose(est.bandwidths_, [0.2, 0.25])
    assert est.cv_results_ is None
    with pytest.raises(ValueError):
        est.set_params(nope=1)
    assert "ADOPTRegressor(" in repr(est)


def test_component_transport_consistency_single_feature(rng):
    """With p = 1 the prediction is T_1(x)(mean_)."""
    X, Y = w2_data(rng)
    est = ADOPTRegressor(bandwidth=0.2).fit(X[:, :1], Y)
    x0 = 0.37
    assert np.allclose(est.transport(0, x0, est.mean_), est.predict([[x0]])[0])
    comp = est.component(0, [0.1, 0.9])
    assert comp["g"].shape == (2, Y.shape[1]) and comp["center"].shape == (Y.shape[1],)


@pytest.mark.skipif(not HAVE_RUST, reason="Rust extension not built")
def test_backends_agree_through_estimator(rng):
    X, Y = w2_data(rng, n=40, q=15)
    kw = dict(bandwidth_grid=[0.15, 0.4], cv=4)
    a = ADOPTRegressor(backend="rust", **kw).fit(X, Y)
    b = ADOPTRegressor(backend="reference", **kw).fit(X, Y)
    assert np.allclose(a.cv_results_["mean_dist_sq"], b.cv_results_["mean_dist_sq"], atol=1e-10)
    assert np.allclose(a.predict(X), b.predict(X), atol=1e-9)


def test_logcholesky_space(rng):
    X, S = spd_data(rng)
    est = ADOPTRegressor(space="logcholesky", bandwidth=0.3).fit(X, S)
    P = est.predict(X[:4])
    assert P.shape == (4, 3, 3)
    assert np.allclose(P, np.swapaxes(P, 1, 2))
    assert np.all(np.linalg.eigvalsh(P) > 0)
    assert est.score(X, S) > 0.9
    v = est.mean_
    assert est.transport(0, 0.5, v).shape == v.shape


def test_baselines(rng):
    X, Y = w2_data(rng)
    fm = FrechetMeanRegressor().fit(X, Y)
    assert np.allclose(fm.predict(X[:3]), Y.mean(axis=0))
    assert abs(fm.score(X, Y)) < 1e-12
    gf = GlobalFrechetRegressor().fit(X, Y)
    assert gf.score(X, Y) > 0.9
    l2 = L2AdditiveQuantileRegressor(bandwidth=0.2).fit(X, Y)
    assert l2.score(X, Y) > 0.9
    Xs, S = spd_data(rng)
    gfs = GlobalFrechetRegressor(space="logcholesky").fit(Xs, S)
    assert gfs.predict(Xs[:2]).shape == (2, 3, 3)


def test_input_validation(rng):
    X, Y = w2_data(rng)
    with pytest.raises(ValueError):
        ADOPTRegressor(bandwidth=0.2).fit(X, Y[:, ::-1])  # decreasing rows
    with pytest.raises(ValueError):
        ADOPTRegressor(space="sphere").fit(X, Y)
    with pytest.raises(ValueError):
        ADOPTRegressor(bandwidth=0.2).fit(X[:10], Y)
    with pytest.raises(RuntimeError):
        ADOPTRegressor().predict(X)
    est = ADOPTRegressor(bandwidth=0.2).fit(X, Y)
    with pytest.raises(ValueError):
        est.predict(np.ones((2, 3)))
