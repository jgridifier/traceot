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
        pit_rounds=kw.get("pit_rounds", 0),
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


def _resolve_alias(primary, alias, default, label: str):
    if primary is None and alias is None:
        return default
    if primary is None:
        return alias
    if alias is not None and primary != alias:
        raise ValueError(f"{label} was given two different values")
    return primary


def _split_pairs(split: dict) -> dict:
    return {name: (float(pair[0]), float(pair[1])) for name, pair in split.items()}


def _band_pairs(split: dict) -> dict:
    """Map each stat to ``((comp_lo, struct_lo), (comp_hi, struct_hi))``.

    The core reports each role as ``(lo, hi)``. Callers compare a band with
    ``split[stat] = (composition, structure)`` on the same index.
    """
    out = {}
    for name, ends in split.items():
        comp_lo, comp_hi = ends[0]
        struct_lo, struct_hi = ends[1]
        out[name] = (
            (float(comp_lo), float(struct_lo)),
            (float(comp_hi), float(struct_hi)),
        )
    return out


def shift_decompose(
    X_a,
    y_a,
    X_b,
    y_b,
    seed: int = 0,
    *,
    K: int | None = None,
    k: int | None = None,
    folds: int | None = None,
    n_folds: int | None = None,
    trim: float | None = None,
    trim_quantile: float | None = None,
    return_model: bool = False,
    n_quad: int = 64,
    logit_ridge: float = 1e-6,
    logit_iters: int = 40,
    logit_tol: float = 1e-12,
    k_fit: int = 4,
    max_bins: int = 16,
    max_rounds: int = 6,
    fit_tol: float = 1e-6,
    n_boot: int = 0,
    pit_rounds: int = 0,
) -> dict:
    """Order-free composition/structure split of a shift between two periods.

    ``X`` is used as given (features arrive centred at the pooled mean).
    ``split`` maps ``mean``, ``sd``, ``q10``, ``q50`` and ``q90`` to
    ``(composition, structure)``, the average of the two orderings, so swapping
    the periods negates every piece. ``split_anchor_A`` (also ``split_aref``)
    is ``(θ_C − θ_A, θ_B − θ_C)`` with ``C = (covariates B, structure A)``.
    ``split_anchor_B`` is the other ordering. L-moments of the four laws live
    in ``lmom``; their order-free and A-reference splits are ``lsplit`` and
    ``lsplit_aref``.

    ``feat_comp`` and ``feat_struct`` are the mean channel (``k = 0``) of the
    barycenter, one λ₁-scale Shapley piece per feature. ``intercept`` is that
    channel's intercept structure piece. ``bary_a`` is ``(K, p+1, 2)`` on the
    raw coefficient scale and ``bary_phi`` the matching ``W2²`` pieces.
    ``w2`` holds the marginal quantile ``W2²``, the barycenter ``W2²``, and
    their gap. ``total_w2`` repeats the marginal value.

    ``return_model=True`` adds ``model``, a ``ShiftModel`` with
    ``predict_quantile(X, u, period)``, ``coef(X, period)`` and ``rearranged``.
    Folds are ``hash(seed, row) mod folds``. ``n_boot`` draws a stratified
    bootstrap. For each stat, ``bands["split"]`` and ``bands["split_uniform"]``
    are ``((composition_lo, structure_lo), (composition_hi, structure_hi))``.
    The uniform band is the joint max-|t| interval across those pieces.
    ``n_boot=0`` sets every end equal to the point.
    ``pit_rounds`` turns on the PIT pseudo-outcome refit in both period fits
    (``0``, the default, leaves it off).
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    reported = _resolve_alias(K, k, 4, "K")
    nfolds = _resolve_alias(folds, n_folds, 5, "folds")
    trim_q = _resolve_alias(trim, trim_quantile, 0.99, "trim")
    X_a = _check_X(X_a)
    X_b = _check_X(X_b, X_a.shape[1])
    y_a = _check_y(y_a, X_a.shape[0])
    y_b = _check_y(y_b, X_b.shape[0])
    raw = _as_dict(
        _core.shift_v2_decompose(
            X_a,
            y_a,
            X_b,
            y_b,
            int(seed),
            int(reported),
            int(n_quad),
            int(nfolds),
            float(trim_q),
            float(logit_ridge),
            int(logit_iters),
            float(logit_tol),
            int(k_fit),
            int(max_bins),
            int(max_rounds),
            float(fit_tol),
            int(n_boot),
            bool(return_model),
            int(pit_rounds),
        )
    )
    for key in ("split", "split_aref", "split_anchor_A", "split_anchor_B"):
        raw[key] = _split_pairs(raw[key])
    bands = raw["bands"]
    bands["split"] = _band_pairs(bands["split"])
    bands["split_uniform"] = _band_pairs(bands["split_uniform"])
    bands["intercept"] = (float(bands["intercept"][0]), float(bands["intercept"][1]))
    bands["total_w2"] = (float(bands["total_w2"][0]), float(bands["total_w2"][1]))
    bands["critical"] = float(bands["critical"])
    bands["feat_critical"] = float(bands["feat_critical"])
    raw["intercept"] = float(raw["intercept"])
    raw["total_w2"] = float(raw["total_w2"])
    raw["trim_mass"] = float(raw["trim_mass"])
    raw["resid_phi"] = {role: float(value) for role, value in raw["resid_phi"].items()}
    raw["w2"] = {role: float(value) for role, value in raw["w2"].items()}
    return raw
