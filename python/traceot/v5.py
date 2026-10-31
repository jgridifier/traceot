"""Thin Python surface for TRACE v5: the Eulerian mixture counterfactual.

All numerics live in the Rust core (``traceot_core::trace::v5``); this module only
validates array shapes and forwards.

The mixture counterfactual CDF for periods ``s, t`` is

    F_[s,t](y) = E_{x ~ F_s} F_t(y | x)

(covariates from ``s``, conditional law from ``t``), and ``Q_[s,t]`` is its inverse.
"""

from __future__ import annotations

import numpy as np

from . import _core


def _f64(a, ndim):
    a = np.ascontiguousarray(a, dtype=np.float64)
    if a.ndim != ndim:
        raise ValueError(f"expected a {ndim}-D array, got shape {a.shape}")
    return a


def explain_v5(
    x_a,
    y_a,
    x_b,
    y_b,
    *,
    n_y: int = 256,
    k_lmoments: int = 4,
    n_quad: int = 64,
    mean_model: str = "booster",
):
    """Split the A→B change in each L-moment into composition and response pieces.

    Fits a location + empirical-residual conditional per period, builds the four
    mixture hybrids ``[A,A]``, ``[A,B]``, ``[B,A]``, ``[B,B]`` on one grid, and returns
    the two-player Shapley split (order-free, sums exactly to the total change).

    Returns a dict with ``lsplit_comp``, ``lsplit_struct`` (length ``k_lmoments``),
    per-hybrid dicts ``aa/ab/ba/bb`` (``q_mix``, ``q_bary``, ``lmom``, ``lmom_bary``,
    ``mixing_residual``), the ``u_nodes`` / ``y_grid`` used, and timings.

    ``mean_model="booster"`` is v5; ``"ols"`` is TRACE_V5_1 (per-period least squares
    on an intercept plus all columns, no tuning). See :func:`explain_v5_1`.
    """
    xa, xb = _f64(x_a, 2), _f64(x_b, 2)
    ya, yb = _f64(y_a, 1), _f64(y_b, 1)
    if mean_model not in ("booster", "ols"):
        raise ValueError("mean_model must be 'booster' or 'ols'")
    return _core.explain_v5_py(
        xa, ya, xb, yb, n_y=n_y, k_lmoments=k_lmoments, n_quad=n_quad, mean_model=mean_model
    )


def explain_v5_1(x_a, y_a, x_b, y_b, *, n_y: int = 256, k_lmoments: int = 4, n_quad: int = 64):
    """TRACE_V5_1 (PREREG_V5_1 §2): :func:`explain_v5` with an OLS location model."""
    return explain_v5(
        x_a, y_a, x_b, y_b, n_y=n_y, k_lmoments=k_lmoments, n_quad=n_quad, mean_model="ols"
    )


def mixture_cdf_location(mu, resid, y_grid):
    """``F(y) = mean_i G(y - mu_i)`` with ``G`` the empirical CDF of ``resid`` (exact)."""
    return _core.mixture_cdf_location_py(_f64(mu, 1), _f64(resid, 1), _f64(y_grid, 1))


def mixture_cdf_rows(u_nodes, q_rows, y_grid, weights=None):
    """Average of per-row conditional CDFs given as quantile curves on ``u_nodes``."""
    w = None if weights is None else _f64(weights, 1)
    return _core.mixture_cdf_rows_py(_f64(u_nodes, 1), _f64(q_rows, 2), _f64(y_grid, 1), w)


def mixture_quantile(y_grid, f_grid, u):
    """Left-inverse of a tabulated CDF (linear between grid points)."""
    return _core.quantile_from_cdf_py(_f64(y_grid, 1), _f64(f_grid, 1), _f64(u, 1))
