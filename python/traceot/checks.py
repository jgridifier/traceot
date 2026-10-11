"""Thin wrappers for TRACE's leakage screens, conformal scores and distributional metrics.

Every statistic is computed in the Rust core. This module only checks array shapes
and dtypes, then forwards contiguous buffers.
"""

from __future__ import annotations

import numpy as np

from . import _core


def _matrix(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty 2-D array (n_samples, n_features)")
    if np.any(np.isinf(X)):
        raise ValueError("X contains infinite values")
    return np.ascontiguousarray(X)


def _vector(y, name: str = "y") -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y.size == 0 or not np.all(np.isfinite(y)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return np.ascontiguousarray(y)


def target_proxy_screen(X, y, threshold: float = 0.9, max_shift: int = 1):
    """Flag columns that track ``y`` or a lead/lag of ``y``.

    Returns ``(contemporaneous, flags)``. ``flags`` is a list of
    ``(feature, shift, corr)``. Shift ``+s`` is ``corr(x[:-s], y[s:])``: the
    column leads ``y``, the pattern of a future-period copy.
    """
    X = _matrix(X)
    y = _vector(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y have different numbers of rows")
    return _core.target_proxy_screen(X, y, float(threshold), int(max_shift))


def duplicate_screen(X, near: float = 0.999):
    """Exact bit-copies and pairs with ``|corr| >= near``.

    Returns a list of ``(left, right, exact, corr)``.
    """
    return _core.duplicate_screen(_matrix(X), float(near))


def placebo_screen(X, y, n_perm: int = 20, seed: int = 0):
    """Real absolute correlations, plus the max and mean-of-maxima under shuffled ``y``.

    Returns ``(real_abs_corr, placebo_max_abs_corr, placebo_mean_max_abs_corr)``.
    """
    X = _matrix(X)
    y = _vector(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y have different numbers of rows")
    if int(seed) < 0:
        raise ValueError("seed must be >= 0")
    return _core.placebo_screen(X, y, int(n_perm), int(seed))


def feature_w2(a, b, grid: int = 128) -> np.ndarray:
    """Per-column empirical W2 between two samples (sqrt of mean squared quantile gap)."""
    a = _matrix(a)
    b = _matrix(b)
    if a.shape[1] != b.shape[1]:
        raise ValueError("feature counts differ")
    return _core.feature_w2(a, b, int(grid))


def conformal_pvalues(calibration, test) -> np.ndarray:
    """Split-conformal p-values ``(1 + #{cal >= s}) / (n_cal + 1)``."""
    return _core.conformal_pvalues(_vector(calibration, "calibration"), _vector(test, "test"))


def pinball_scores(y, quantiles, u) -> np.ndarray:
    """Per-row mean pinball loss of a quantile matrix against ``y``."""
    y = _vector(y)
    q = np.asarray(quantiles, dtype=np.float64)
    u = _vector(u, "u")
    if q.ndim != 2 or q.shape[0] != y.shape[0] or q.shape[1] != u.shape[0]:
        raise ValueError("quantiles must have shape (n, len(u))")
    return _core.pinball_scores(y, np.ascontiguousarray(q), u)


def score_quantiles(y, quantiles, u) -> dict:
    """Pinball, CRPS approximation, 80/90% coverage, and MSE of the median and the mean."""
    y = _vector(y)
    q = np.asarray(quantiles, dtype=np.float64)
    u = _vector(u, "u")
    if q.ndim != 2 or q.shape[0] != y.shape[0] or q.shape[1] != u.shape[0]:
        raise ValueError("quantiles must have shape (n, len(u))")
    return _core.score_quantiles(y, np.ascontiguousarray(q), u)


def projection_blur(q00, q01, q10, q11):
    """RMS interaction contrast of four quantile curves, before and after isotonic projection.

    Returns ``(before, after)``.
    """
    curves = [_vector(q, "curve") for q in (q00, q01, q10, q11)]
    n = curves[0].shape[0]
    if any(c.shape[0] != n for c in curves):
        raise ValueError("the four curves must have the same length")
    return _core.projection_blur(*curves)
