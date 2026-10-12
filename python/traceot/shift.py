"""Shift decomposition v2: composition, structure, and entry or exit.

Numerics live in the Rust core. This module checks array shapes and forwards
them. See ``docs/theory/SHIFT_V2.md`` and the learning page
``docs/learn/explaining_shifts.html``.
"""

from __future__ import annotations

import numpy as np

from . import _core
from .trace_v1 import TraceV1, _check_X, _check_y


def _model(model):
    if isinstance(model, TraceV1):
        model._check_fitted()
        return model.model_
    return model


def _mask(mask, n: int) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.shape != (n,):
        raise ValueError(f"mask must have shape ({n},)")
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return np.ascontiguousarray(mask)


def _as_dict(raw) -> dict:
    out = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            out[key] = _as_dict(value)
        else:
            out[key] = value
    return out


def _v1_fit_kwargs(model: TraceV1) -> dict:
    kw = model._kwargs()
    return dict(
        k_fit=kw["k_fit"],
        k_report=kw["k_report"],
        n_u=kw["n_u"],
        max_bins=kw["max_bins"],
        max_rounds=kw["max_rounds"],
        tol=kw["tol"],
        min_count=kw["min_count"],
        ridge=kw["ridge"],
        learning_rate=kw["learning_rate"],
        mono=kw["mono"],
        bernstein_elevation=kw["bernstein_elevation"],
        reporting=kw["reporting"],
    )


def sample_lambda(y, k: int = 6, n_quad: int = 64):
    """L-moments of one sample on the shift quadrature."""
    y = _check_y(y)
    return np.asarray(_core.shift_v2_sample_lambda(y, int(k), int(n_quad)))


def from_laws(y_a, y_b, y_c, y_cp, k: int = 6, n_quad: int = 64) -> dict:
    """Composition and structure from four observed laws. No classifier.

    ``y_c`` is structure B on covariates A. ``y_cp`` is structure A on covariates B.
    """
    y_a = _check_y(y_a)
    y_b = _check_y(y_b)
    y_c = _check_y(y_c)
    y_cp = _check_y(y_cp)
    return _as_dict(_core.shift_v2_from_laws(y_a, y_b, y_c, y_cp, int(k), int(n_quad)))


def quadratic_phi(a, k: int):
    """``φ_i = a_i · sum(a) / (2k+1)``. Negative shares are kept."""
    a = _check_y(a)
    return np.asarray(_core.shift_v2_quadratic_phi(a, int(k)))


def hk_mass(m_a: float, m_b: float) -> tuple[float, float]:
    """Pure-mass Hellinger–Kantorovich term, and the same term over ``m_a + m_b``."""
    hk, hk_n = _core.shift_v2_hk_mass(float(m_a), float(m_b))
    return float(hk), float(hk_n)


def entry_exit(
    stay_a,
    stay_b,
    exit_mean,
    entry_mean,
    e_a: float,
    e_b: float,
    m_a: float,
    m_b: float,
) -> dict:
    """Four-player entry/exit split. One coefficient per channel, same scale on each vector."""
    stay_a = _check_y(stay_a)
    stay_b = _check_y(stay_b)
    exit_mean = _check_y(exit_mean)
    entry_mean = _check_y(entry_mean)
    return _as_dict(
        _core.shift_v2_entry_exit(
            stay_a,
            stay_b,
            exit_mean,
            entry_mean,
            float(e_a),
            float(e_b),
            float(m_a),
            float(m_b),
        )
    )


def marginal(
    X_a,
    y_a,
    X_b,
    y_b,
    k: int = 6,
    n_quad: int = 64,
    n_folds: int = 5,
    trim_quantile: float = 0.995,
    logit_ridge: float = 1e-6,
    logit_iters: int = 25,
    logit_tol: float = 1e-8,
    seed: int = 0,
) -> dict:
    """Cross-fitted period-odds reweighting, then the marginal-law split."""
    X_a = _check_X(X_a)
    X_b = _check_X(X_b, X_a.shape[1])
    y_a = _check_y(y_a, X_a.shape[0])
    y_b = _check_y(y_b, X_b.shape[0])
    return _as_dict(
        _core.shift_v2_marginal(
            X_a,
            y_a,
            X_b,
            y_b,
            int(k),
            int(n_quad),
            int(n_folds),
            float(trim_quantile),
            float(logit_ridge),
            int(logit_iters),
            float(logit_tol),
            int(seed),
        )
    )


def barycenter(model_a, model_b, X_a, X_b) -> dict:
    """Vincent-average players from two fitted TRACE v1 models."""
    X_a = _check_X(X_a)
    X_b = _check_X(X_b, X_a.shape[1])
    return _as_dict(_core.shift_v2_barycenter(_model(model_a), _model(model_b), X_a, X_b))


def explain(
    model_a,
    model_b,
    X_a,
    y_a,
    X_b,
    y_b,
    n_quad: int = 64,
    n_folds: int = 5,
    trim_quantile: float = 0.995,
    logit_ridge: float = 1e-6,
    logit_iters: int = 25,
    logit_tol: float = 1e-8,
    seed: int = 0,
) -> dict:
    """Both targets, and the gap between the marginal law and the barycenter."""
    X_a = _check_X(X_a)
    X_b = _check_X(X_b, X_a.shape[1])
    y_a = _check_y(y_a, X_a.shape[0])
    y_b = _check_y(y_b, X_b.shape[0])
    return _as_dict(
        _core.shift_v2_explain(
            _model(model_a),
            _model(model_b),
            X_a,
            y_a,
            X_b,
            y_b,
            int(n_quad),
            int(n_folds),
            float(trim_quantile),
            float(logit_ridge),
            int(logit_iters),
            float(logit_tol),
            int(seed),
        )
    )


def bootstrap_marginal(
    X_a,
    y_a,
    X_b,
    y_b,
    reps: int,
    seed: int,
    k: int = 6,
    n_quad: int = 64,
    n_folds: int = 5,
    trim_quantile: float = 0.995,
    logit_ridge: float = 1e-6,
    logit_iters: int = 25,
    logit_tol: float = 1e-8,
) -> dict:
    """Stratified row bootstrap of the marginal split, classifier refit each replicate."""
    X_a = _check_X(X_a)
    X_b = _check_X(X_b, X_a.shape[1])
    y_a = _check_y(y_a, X_a.shape[0])
    y_b = _check_y(y_b, X_b.shape[0])
    return _as_dict(
        _core.shift_v2_bootstrap_marginal(
            X_a,
            y_a,
            X_b,
            y_b,
            int(reps),
            int(seed),
            int(k),
            int(n_quad),
            int(n_folds),
            float(trim_quantile),
            float(logit_ridge),
            int(logit_iters),
            float(logit_tol),
        )
    )


def bootstrap_barycenter(
    model_a: TraceV1,
    model_b: TraceV1,
    X_a,
    y_a,
    X_b,
    y_b,
    reps: int,
    seed: int,
) -> dict:
    """Stratified bootstrap that refits both TRACE models, then the barycenter players."""
    if not isinstance(model_a, TraceV1) or not isinstance(model_b, TraceV1):
        raise TypeError("bootstrap_barycenter refits from a TraceV1 configuration")
    X_a = _check_X(X_a)
    X_b = _check_X(X_b, X_a.shape[1])
    y_a = _check_y(y_a, X_a.shape[0])
    y_b = _check_y(y_b, X_b.shape[0])
    return _as_dict(
        _core.shift_v2_bootstrap_barycenter(
            X_a, y_a, X_b, y_b, int(reps), int(seed), **_v1_fit_kwargs(model_a)
        )
    )


def entry_from_models(model_a, model_b, X_a, exit_a, X_b, entry_b, m_a: float, m_b: float) -> dict:
    """Entry/exit split from two fitted models and row masks (1 = mover)."""
    X_a = _check_X(X_a)
    X_b = _check_X(X_b, X_a.shape[1])
    exit_a = _mask(exit_a, X_a.shape[0])
    entry_b = _mask(entry_b, X_b.shape[0])
    return _as_dict(
        _core.shift_v2_entry_from_models(
            _model(model_a),
            _model(model_b),
            X_a,
            exit_a,
            X_b,
            entry_b,
            float(m_a),
            float(m_b),
        )
    )


def bootstrap_entry(
    model_a: TraceV1,
    X_a,
    y_a,
    exit_a,
    X_b,
    y_b,
    entry_b,
    reps: int,
    seed: int,
) -> dict:
    """Refit both models on stratified resamples and recompute the entry/exit split."""
    if not isinstance(model_a, TraceV1):
        raise TypeError("bootstrap_entry refits from a TraceV1 configuration")
    X_a = _check_X(X_a)
    X_b = _check_X(X_b, X_a.shape[1])
    y_a = _check_y(y_a, X_a.shape[0])
    y_b = _check_y(y_b, X_b.shape[0])
    exit_a = _mask(exit_a, X_a.shape[0])
    entry_b = _mask(entry_b, X_b.shape[0])
    return _as_dict(
        _core.shift_v2_bootstrap_entry(
            X_a,
            y_a,
            exit_a,
            X_b,
            y_b,
            entry_b,
            int(reps),
            int(seed),
            **_v1_fit_kwargs(model_a),
        )
    )


def bands(point, replicates) -> dict:
    """Pointwise 2.5/97.5 bands and a uniform max-|t| band. ``replicates`` is ``(reps, dim)``."""
    point = _check_y(point)
    replicates = np.asarray(replicates, dtype=np.float64)
    if replicates.ndim != 2 or replicates.shape[1] != point.shape[0]:
        raise ValueError("replicates must have shape (reps, len(point))")
    replicates = np.ascontiguousarray(replicates)
    return _as_dict(_core.shift_v2_bands(point, replicates))


def permutation_w2(
    y_a,
    y_b,
    reps: int,
    seed: int,
    n_quad: int = 64,
    k: int = 4,
    level: float = 0.95,
) -> dict:
    """Permutation floor of the total outcome W2²."""
    y_a = _check_y(y_a)
    y_b = _check_y(y_b)
    return _as_dict(
        _core.shift_v2_permutation_w2(
            y_a, y_b, int(reps), int(seed), int(n_quad), int(k), float(level)
        )
    )


def permutation_marginal(
    X_a,
    y_a,
    X_b,
    y_b,
    reps: int,
    seed: int,
    k: int = 6,
    n_quad: int = 64,
    n_folds: int = 5,
    trim_quantile: float = 0.995,
    logit_ridge: float = 1e-6,
    logit_iters: int = 25,
    logit_tol: float = 1e-8,
) -> dict:
    """Shuffled period labels, then the marginal split on each shuffle."""
    X_a = _check_X(X_a)
    X_b = _check_X(X_b, X_a.shape[1])
    y_a = _check_y(y_a, X_a.shape[0])
    y_b = _check_y(y_b, X_b.shape[0])
    return _as_dict(
        _core.shift_v2_permutation_marginal(
            X_a,
            y_a,
            X_b,
            y_b,
            int(reps),
            int(seed),
            int(k),
            int(n_quad),
            int(n_folds),
            float(trim_quantile),
            float(logit_ridge),
            int(logit_iters),
            float(logit_tol),
        )
    )
