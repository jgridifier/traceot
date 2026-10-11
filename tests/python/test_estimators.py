"""Scikit-learn-style estimator API."""

import numpy as np
import pytest

import reference
import reference.estimators as ref
from traceot import (
    ADOPTRegressor,
    FrechetMeanRegressor,
    GlobalFrechetRegressor,
    L2AdditiveQuantileRegressor,
    quantile_grid,
)


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


def _ref_cv_fit(method, space, X, Z, grid, k, seed, **kw):
    cv = ref.select_bandwidth_cv(method, space, X, Z, grid, k, seed, **kw)
    h = cv["best_fraction"] * ref.column_ranges(X)
    return cv, ref.fit(method, space, X, Z, h, **kw)


def test_kfold_matches_reference():
    from traceot import _core

    for n, k, seed in [(10, 2, 0), (23, 5, 7), (101, 5, 2**63 + 5)]:
        a, b = _core.kfold_indices(n, k, seed), ref.kfold_indices(n, k, seed)
        assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize(
    "cls,method,space,bounds",
    [
        (ADOPTRegressor, "adopt", "wasserstein", None),
        (ADOPTRegressor, "adopt", "wasserstein", (-2.0, 2.0)),
        (L2AdditiveQuantileRegressor, "l2additive", "wasserstein", None),
    ],
)
def test_estimator_matches_reference_w2(rng, cls, method, space, bounds):
    X, Y = w2_data(rng, n=40, q=15)
    grid = [0.08, 0.15, 0.4]
    est = cls(bandwidth_grid=grid, cv=4, random_state=11, bounds=bounds).fit(X, Y)
    lo, hi = bounds if bounds else (None, None)
    cv, m = _ref_cv_fit(method, space, X, Y, grid, 4, 11, lower=lo, upper=hi)
    assert np.allclose(est.cv_results_["mean_dist_sq"], cv["mean_dist_sq"], rtol=0, atol=1e-10)
    assert est.cv_results_["best_fraction"] == cv["best_fraction"]
    Xn = rng.uniform(size=(7, 2))
    assert np.allclose(est.predict(Xn), ref.predict(m, Xn), atol=1e-9)
    assert np.allclose(
        est.component(1, [0.2, 0.7])["g"], ref.component(m, 1, np.array([0.2, 0.7])), atol=1e-9
    )
    v = np.sort(rng.normal(size=15))
    assert np.allclose(est.transport(0, 0.3, v), ref.transport(m, 0, 0.3, v), atol=1e-9)
    assert np.isclose(
        est.score(X, Y), ref.frechet_r2(space, ref.predict(m, X), Y, m["mu"]), atol=1e-10
    )


def test_estimator_matches_reference_logcholesky(rng):
    X, S = spd_data(rng, n=40)
    est = ADOPTRegressor(space="logcholesky", bandwidth_grid=[0.1, 0.3], cv=4).fit(X, S)
    Z = reference.spd_to_coords(S)
    cv, m = _ref_cv_fit("adopt", "logcholesky", X, Z, [0.1, 0.3], 4, 0)
    assert np.allclose(est.cv_results_["mean_dist_sq"], cv["mean_dist_sq"], atol=1e-10)
    assert np.allclose(est.predict_encoded(X), ref.predict(m, X), atol=1e-9)


@pytest.mark.parametrize("space", ["wasserstein", "logcholesky"])
def test_global_frechet_matches_reference(rng, space):
    if space == "wasserstein":
        X, Y = w2_data(rng)
        Z = Y
    else:
        X, Y = spd_data(rng)
        Z = reference.spd_to_coords(Y)
    gf = GlobalFrechetRegressor(space=space).fit(X, Y)
    Xn = rng.uniform(size=(5, 2))
    assert np.allclose(
        gf.predict_encoded(Xn), ref.global_frechet_predict(space, X, Z, Xn), atol=1e-10
    )
    fm = FrechetMeanRegressor(space=space).fit(X, Y)
    assert np.allclose(fm.mean_, Z.mean(axis=0), atol=1e-12)


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
