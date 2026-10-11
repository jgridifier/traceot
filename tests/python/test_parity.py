"""The Rust core must agree with the numpy reference on every numerical routine."""

import numpy as np
import pytest

import reference
from conftest import random_quantiles
from traceot import primitives

ATOL = 1e-10


@pytest.fixture(scope="module")
def backends():
    return primitives, reference


def close(a, b, atol=ATOL):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.allclose(a, b, rtol=0, atol=atol), np.max(np.abs(a - b))


def test_quantile_grid(backends):
    R, P = backends
    close(R.quantile_grid(7), P.quantile_grid(7), 0)


def test_w2_sq(backends, rng):
    R, P = backends
    A, B = random_quantiles(rng, 5, 30), random_quantiles(rng, 5, 30)
    close(R.w2_sq(A, B), P.w2_sq(A, B))
    close(R.w2_sq(A, B[0]), P.w2_sq(A, B[0]))


@pytest.mark.parametrize("bounds", [(None, None), (-0.5, 0.5)])
def test_pava(backends, rng, bounds):
    R, P = backends
    for _ in range(25):
        y = rng.normal(size=rng.integers(1, 60))
        close(R.pava(y, *bounds), P.pava(y, *bounds), 0)
    Y = rng.normal(size=(6, 40))
    close(R.pava_rows(Y, *bounds), P.pava_rows(Y, *bounds), 0)


def test_collapse_and_transport(backends, rng):
    R, P = backends
    for ties in (False, True):
        a, b = random_quantiles(rng, 2, 50, ties=ties)
        for x, y in zip(R.collapse_knots(a, b), P.collapse_knots(a, b), strict=True):
            close(x, y, 0)
        pts = np.concatenate([a, rng.normal(size=200) * 3, [a[0] - 1, a[-1] + 1]])
        close(R.transport_apply(a, b, pts), P.transport_apply(a, b, pts))


@pytest.mark.parametrize("kernel", ["gaussian", "epanechnikov"])
def test_local_linear_weights(backends, rng, kernel):
    R, P = backends
    x = rng.uniform(size=50)
    xe = np.concatenate([rng.uniform(-0.2, 1.2, size=20), [5.0]])
    for h in (0.02, 0.1, 0.5):
        close(R.local_linear_weights(x, xe, h, kernel), P.local_linear_weights(x, xe, h, kernel))


def test_global_frechet_weights(backends, rng):
    R, P = backends
    X, Xe = rng.normal(size=(40, 3)), rng.normal(size=(7, 3))
    close(R.global_frechet_weights(X, Xe), P.global_frechet_weights(X, Xe))


def _w2_data(rng, n=40, q=20):
    X = rng.uniform(size=(n, 3))
    u = (np.arange(q) + 0.5) / q
    Y = (u ** (1 + X[:, [0]])) * (1 + X[:, [1]]) + np.sin(3 * X[:, [2]])
    Y = Y + 0.05 * rng.normal(size=(n, 1)) * np.sin(2 * np.pi * Y)
    return X, np.sort(Y, axis=1)


@pytest.mark.parametrize("kernel", ["gaussian", "epanechnikov"])
@pytest.mark.parametrize("bounds", [(None, None), (-1.0, 3.0)])
def test_adopt_fit_predict(backends, rng, kernel, bounds):
    R, P = backends
    X, Y = _w2_data(rng)
    h = np.array([0.15, 0.2, 0.3])
    fr = R.adopt_w2_fit(X, Y, h, kernel, 1e-10, 50, *bounds)
    fp = P.adopt_w2_fit(X, Y, h, kernel, 1e-10, 50, *bounds)
    assert fr["n_iter"] == fp["n_iter"] and fr["converged"] == fp["converged"]
    for k in ("mu", "G", "M", "P", "fitted", "history"):
        close(fr[k], fp[k], 1e-9)
    Xn = rng.uniform(-0.1, 1.1, size=(15, 3))
    close(
        R.adopt_w2_predict(X, fr["P"], fr["M"], fr["mu"], h, Xn, kernel, *bounds),
        P.adopt_w2_predict(X, fp["P"], fp["M"], fp["mu"], h, Xn, kernel, *bounds),
        1e-9,
    )
    close(
        R.adopt_w2_component(X[:, 1], fr["P"][1], h[1], Xn[:, 1], kernel, *bounds),
        P.adopt_w2_component(X[:, 1], fp["P"][1], h[1], Xn[:, 1], kernel, *bounds),
        1e-9,
    )


def test_additive_fit_predict(backends, rng):
    R, P = backends
    X = rng.uniform(size=(45, 3))
    Z = np.column_stack([np.sin(4 * X[:, 0]) + X[:, 1] ** 2, X[:, 2] - X[:, 0]])
    Z = Z + 0.01 * rng.normal(size=Z.shape)
    h = np.array([0.1, 0.2, 0.15])
    fr, fp = R.additive_euclid_fit(X, Z, h), P.additive_euclid_fit(X, Z, h)
    assert fr["n_iter"] == fp["n_iter"]
    for k in ("mu", "G", "M", "P", "fitted", "history"):
        close(fr[k], fp[k], 1e-9)
    Xn = rng.uniform(size=(9, 3))
    close(
        R.additive_euclid_predict(X, fr["P"], fr["M"], fr["mu"], h, Xn),
        P.additive_euclid_predict(X, fp["P"], fp["M"], fp["mu"], h, Xn),
        1e-9,
    )


def test_logcholesky(backends, rng):
    R, P = backends
    A = rng.normal(size=(5, 4, 4))
    S = A @ np.swapaxes(A, -1, -2) + 0.1 * np.eye(4)
    close(R.spd_to_coords(S), P.spd_to_coords(S))
    z = P.spd_to_coords(S)
    close(R.coords_to_spd(z, 4), P.coords_to_spd(z, 4))


def test_reflected_weighted_and_binned(backends, rng):
    R, P = backends
    x = np.linspace(0.0, 1.0, 40)
    xe = np.array([0.0, 0.02, 0.5, 0.97, 1.0, -0.2])
    for kernel in ("gaussian", "epanechnikov"):
        close(
            R.reflected_local_linear_weights(x, xe, 0.2, kernel),
            P.reflected_local_linear_weights(x, xe, 0.2, kernel),
        )
    ones = np.ones_like(x)
    close(
        R.local_linear_weights_weighted(x, ones, xe, 0.15),
        P.local_linear_weights(x, xe, 0.15),
    )
    mult = rng.integers(0, 5, size=x.shape[0]).astype(float)
    close(
        R.local_linear_weights_weighted(x, mult, xe, 0.15),
        P.local_linear_weights_weighted(x, mult, xe, 0.15),
    )
    y = np.column_stack([x, x**2])
    close(R.binned_smooth(x, y, 0.2, xe, 8), P.binned_smooth(x, y, 0.2, xe, 8))
    # A single occupied bin repeats its mean.
    close(
        R.binned_smooth(np.zeros(6), y[:6], 0.3, xe, 4),
        P.binned_smooth(np.zeros(6), y[:6], 0.3, xe, 4),
    )


def _opt_close(fr, fp):
    assert fr["n_iter"] == fp["n_iter"]
    assert fr["converged"] == fp["converged"]
    assert fr["n_rejected"] == fp["n_rejected"]
    for k in ("mu", "G", "M", "P", "fitted", "history"):
        close(fr[k], fp[k])
    assert len(fr["orders"]) == len(fp["orders"])
    for a, b in zip(fr["orders"], fp["orders"], strict=True):
        close(np.asarray(a["perm"], float), np.asarray(b["perm"], float), 0)
        close(a["G"], b["G"])
        close(a["M"], b["M"])
        close(a["P"], b["P"])


def test_adopt_option_paths(backends, rng):
    R, P = backends
    X, Y = _w2_data(rng, n=26, q=10)
    X, h = X[:, :2], np.array([0.22, 0.28])
    specs = [
        dict(damping=0.5, max_iter=6, tol=1e-8),
        dict(certificate=True, max_iter=5, tol=1e-6),
        dict(centering="pushforward", max_iter=5, tol=1e-6),
        dict(composition="average", max_iter=3, tol=1e-4),
        dict(composition="tangent", max_iter=5, tol=1e-6),
        dict(project_prediction=True, max_iter=4, tol=1e-5),
        dict(bins=6, max_iter=4, tol=1e-5),
        dict(boundary="reflect", max_iter=4, tol=1e-5),
        dict(centering="pushforward", damping=0.7, certificate=True, bins=5, max_iter=3, tol=1e-4),
    ]
    Xn = rng.uniform(size=(4, 2))
    for spec in specs:
        fr = R.adopt_w2_fit(X, Y, h, **spec)
        fp = P.adopt_w2_fit(X, Y, h, **spec)
        _opt_close(fr, fp)
        pred_kw = dict(
            bins=spec.get("bins"),
            boundary=spec.get("boundary", "none"),
            project_prediction=spec.get("project_prediction", False),
        )
        if spec.get("composition") == "average":
            close(
                R.adopt_w2_predict_averaged(X, fr["orders"], fr["mu"], h, Xn, **pred_kw),
                P.adopt_w2_predict_averaged(X, fp["orders"], fp["mu"], h, Xn, **pred_kw),
            )
        elif spec.get("composition") != "tangent":
            close(
                R.adopt_w2_predict(X, fr["P"], fr["M"], fr["mu"], h, Xn, **pred_kw),
                P.adopt_w2_predict(X, fp["P"], fp["M"], fp["mu"], h, Xn, **pred_kw),
            )
            ckw = dict(bins=pred_kw["bins"], boundary=pred_kw["boundary"])
            close(
                R.adopt_w2_component(X[:, 0], fr["P"][0], h[0], Xn[:, 0], **ckw),
                P.adopt_w2_component(X[:, 0], fp["P"][0], h[0], Xn[:, 0], **ckw),
            )


def test_additive_option_paths(backends, rng):
    R, P = backends
    X = rng.uniform(size=(28, 2))
    Z = np.column_stack([np.sin(3 * X[:, 0]) + X[:, 1], X[:, 0] - X[:, 1] ** 2])
    h = np.array([0.2, 0.25])
    Xn = rng.uniform(size=(5, 2))
    for spec in (
        dict(damping=0.4, certificate=True, max_iter=6, tol=1e-8),
        dict(bins=5, max_iter=4, tol=1e-5),
        dict(boundary="reflect", max_iter=4, tol=1e-5),
    ):
        fr = R.additive_euclid_fit(X, Z, h, **spec)
        fp = P.additive_euclid_fit(X, Z, h, **spec)
        _opt_close(fr, fp)
        kw = dict(bins=spec.get("bins"), boundary=spec.get("boundary", "none"))
        close(
            R.additive_euclid_predict(X, fr["P"], fr["M"], fr["mu"], h, Xn, **kw),
            P.additive_euclid_predict(X, fp["P"], fp["M"], fp["mu"], h, Xn, **kw),
        )


def test_rust_errors_are_value_errors(backends, rng):
    R, _ = backends
    X, Y = _w2_data(rng, n=10, q=5)
    with pytest.raises(ValueError):
        R.adopt_w2_fit(X, Y, [0.1, 0.1])
    with pytest.raises(ValueError):
        R.adopt_w2_fit(X, Y[:5], [0.1, 0.1, 0.1])
    with pytest.raises(ValueError):
        R.local_linear_weights(X[:, 0], X[:, 0], 0.1, "box")
