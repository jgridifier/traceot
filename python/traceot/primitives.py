"""Low-level numerical primitives of the Rust core, with input conversion only.

Every function here validates shapes, converts to contiguous ``float64`` (or
``uint8``/``uint16``) numpy arrays and calls ``traceot._core``. There is no
numerical logic in Python; the test-only numpy reference
(``tests/python/reference``) mirrors these names and signatures.
"""

from __future__ import annotations

import numpy as np

from . import _core

__all__ = [
    "QuantileSketch",
    "additive_euclid_fit",
    "additive_euclid_predict",
    "adopt_w2_component",
    "adopt_w2_fit",
    "adopt_w2_predict",
    "adopt_w2_predict_averaged",
    "bin_column",
    "bin_edges_from_sorted",
    "binned_smooth",
    "collapse_knots",
    "coords_to_spd",
    "global_frechet_weights",
    "histogram",
    "histograms",
    "local_linear_weights",
    "local_linear_weights_weighted",
    "reflected_local_linear_weights",
    "pava",
    "pava_rows",
    "pinball_grad_location_scale",
    "quantile_grid",
    "spd_to_coords",
    "transport_apply",
    "w2_sq",
]

QuantileSketch = _core.QuantileSketch
quantile_grid = _core.quantile_grid


def _f(a, ndim: int) -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=np.float64)
    if a.ndim != ndim:
        raise ValueError(f"expected a {ndim}-D array, got shape {a.shape}")
    return a


def _h(h, p: int) -> np.ndarray:
    """Bandwidth vector of length p (a scalar is broadcast); validated, not computed."""
    h = np.asarray(h, dtype=np.float64)
    if h.ndim == 0:
        h = np.full(p, float(h))
    if h.shape != (p,):
        raise ValueError(f"expected {p} bandwidths, got shape {h.shape}")
    if not np.all(h > 0):
        raise ValueError("bandwidths must be > 0")
    return np.ascontiguousarray(h)


def w2_sq(q1, q2):
    """Squared W2 distance between quantile functions (last axis), broadcasting."""
    q1, q2 = np.broadcast_arrays(np.asarray(q1, float), np.asarray(q2, float))
    shape = q1.shape[:-1]
    a = np.ascontiguousarray(q1.reshape(-1, q1.shape[-1]))
    b = np.ascontiguousarray(q2.reshape(-1, q2.shape[-1]))
    return _core.w2_sq_rows(a, b).reshape(shape)


def pava(y, lower=None, upper=None):
    return _core.pava(_f(y, 1), lower, upper)


def pava_rows(Y, lower=None, upper=None):
    return _core.pava_rows(_f(np.atleast_2d(Y), 2), lower, upper)


def collapse_knots(a, b):
    return _core.collapse_knots(_f(a, 1), _f(b, 1))


def transport_apply(a, b, y):
    return _core.transport_apply(_f(a, 1), _f(b, 1), _f(np.atleast_1d(y), 1))


def local_linear_weights(x_train, x_eval, h, kernel="gaussian"):
    return _core.local_linear_weights(
        _f(np.ravel(x_train), 1), _f(np.ravel(x_eval), 1), float(h), kernel
    )


def reflected_local_linear_weights(x_train, x_eval, h, kernel="gaussian"):
    return _core.reflected_local_linear_weights(
        _f(np.ravel(x_train), 1), _f(np.ravel(x_eval), 1), float(h), kernel
    )


def local_linear_weights_weighted(x_train, multiplicity, x_eval, h, kernel="gaussian"):
    return _core.local_linear_weights_weighted(
        _f(np.ravel(x_train), 1),
        _f(np.ravel(multiplicity), 1),
        _f(np.ravel(x_eval), 1),
        float(h),
        kernel,
    )


def binned_smooth(x_train, y, h, x_eval, n_bins, kernel="gaussian"):
    return _core.binned_smooth(
        _f(np.ravel(x_train), 1),
        _f(np.atleast_2d(y), 2),
        float(h),
        _f(np.ravel(x_eval), 1),
        int(n_bins),
        kernel,
    )


def global_frechet_weights(X_train, X_eval):
    return _core.global_frechet_weights(_f(np.atleast_2d(X_train), 2), _f(np.atleast_2d(X_eval), 2))


def adopt_w2_fit(
    X,
    Y,
    h,
    kernel="gaussian",
    tol=1e-10,
    max_iter=200,
    lower=None,
    upper=None,
    damping=1.0,
    certificate=False,
    composition="given",
    centering="frechet",
    project_prediction=False,
    bins=None,
    boundary="none",
):
    X, Y = _f(np.atleast_2d(X), 2), _f(np.atleast_2d(Y), 2)
    return _core.adopt_w2_fit(
        X,
        Y,
        _h(h, X.shape[1]),
        kernel,
        float(tol),
        int(max_iter),
        lower,
        upper,
        float(damping),
        bool(certificate),
        composition,
        centering,
        bool(project_prediction),
        bins,
        boundary,
    )


def adopt_w2_predict(
    X_train,
    P,
    M,
    mu,
    h,
    X_new,
    kernel="gaussian",
    lower=None,
    upper=None,
    project_prediction=False,
    bins=None,
    boundary="none",
):
    X_train = _f(np.atleast_2d(X_train), 2)
    return _core.adopt_w2_predict(
        X_train,
        _f(P, 3),
        _f(M, 2),
        _f(mu, 1),
        _h(h, X_train.shape[1]),
        _f(np.atleast_2d(X_new), 2),
        kernel,
        lower,
        upper,
        bool(project_prediction),
        bins,
        boundary,
    )


def adopt_w2_predict_averaged(
    X_train,
    orders,
    mu,
    h,
    X_new,
    kernel="gaussian",
    lower=None,
    upper=None,
    project_prediction=False,
    bins=None,
    boundary="none",
):
    X_train = _f(np.atleast_2d(X_train), 2)
    return _core.adopt_w2_predict_averaged(
        X_train,
        orders,
        _f(mu, 1),
        _h(h, X_train.shape[1]),
        _f(np.atleast_2d(X_new), 2),
        kernel,
        lower,
        upper,
        bool(project_prediction),
        bins,
        boundary,
    )


def adopt_w2_component(
    X_train, P_j, h_j, x_eval, kernel="gaussian", lower=None, upper=None, bins=None, boundary="none"
):
    return _core.adopt_w2_component(
        _f(np.ravel(X_train), 1),
        _f(P_j, 2),
        float(h_j),
        _f(np.ravel(x_eval), 1),
        kernel,
        lower,
        upper,
        bins,
        boundary,
    )


def additive_euclid_fit(
    X,
    Z,
    h,
    kernel="gaussian",
    tol=1e-10,
    max_iter=200,
    damping=1.0,
    certificate=False,
    bins=None,
    boundary="none",
):
    X, Z = _f(np.atleast_2d(X), 2), _f(np.atleast_2d(Z), 2)
    return _core.additive_euclid_fit(
        X,
        Z,
        _h(h, X.shape[1]),
        kernel,
        float(tol),
        int(max_iter),
        float(damping),
        bool(certificate),
        bins,
        boundary,
    )


def additive_euclid_predict(
    X_train, P, M, mu, h, X_new, kernel="gaussian", bins=None, boundary="none"
):
    X_train = _f(np.atleast_2d(X_train), 2)
    return _core.additive_euclid_predict(
        X_train,
        _f(P, 3),
        _f(M, 2),
        _f(mu, 1),
        _h(h, X_train.shape[1]),
        _f(np.atleast_2d(X_new), 2),
        kernel,
        bins,
        boundary,
    )


def spd_to_coords(S):
    S = np.asarray(S, float)
    flat = S.reshape((-1,) + S.shape[-2:])
    return _core.spd_to_coords(_f(flat, 3)).reshape(S.shape[:-2] + (-1,))


def coords_to_spd(z, m):
    z = np.asarray(z, float)
    flat = z.reshape(-1, z.shape[-1])
    return _core.coords_to_spd(_f(flat, 2), int(m)).reshape(z.shape[:-1] + (m, m))


def bin_edges_from_sorted(x_sorted, max_bins):
    return _core.bin_edges_from_sorted(_f(x_sorted, 1), int(max_bins))


def bin_column(x, edges):
    return _core.bin_column(_f(x, 1), _f(edges, 1))


def histograms(columns, grads, n_bins):
    columns = np.ascontiguousarray(np.atleast_2d(columns))
    grads = _f(np.atleast_2d(grads), 2)
    if columns.dtype == np.uint8:
        return _core.histograms_u8(columns, grads, int(n_bins))
    if columns.dtype == np.uint16:
        return _core.histograms_u16(columns, grads, int(n_bins))
    raise ValueError("bin codes must be uint8 or uint16")


def histogram(codes, grads, n_bins):
    s, k = histograms(np.asarray(codes)[None, :], grads, n_bins)
    return s[0], k[0]


def pinball_grad_location_scale(y, loc, scale, z, u, basis):
    return _core.pinball_grad_location_scale(
        _f(y, 1), _f(loc, 1), _f(scale, 1), _f(z, 1), _f(u, 1), _f(np.atleast_2d(basis), 2)
    )
