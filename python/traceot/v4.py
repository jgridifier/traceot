"""Thin Python surface for TRACE v4 (`explain_v4`).

All numerics live in the Rust core; this module validates arrays and forwards.
"""

from __future__ import annotations

import numpy as np

from . import _core


def explain_v4(
    x_a,
    y_a,
    x_b,
    y_b,
    *,
    k_channels: int = 6,
    seed: int = 0,
    n_boot: int = 2000,
    alpha: float = 0.05,
    max_bins: int = 64,
    pit_iters: int = 2,
    n_u: int = 64,
) -> dict:
    """Decompose a two-period shift under the v4 mixing law + game.

    Parameters
    ----------
    x_a, x_b : array-like, shape (n, p)
    y_a, y_b : array-like, shape (n,)
    k_channels, seed, n_boot, alpha, max_bins, pit_iters, n_u
        Forwarded to the Rust `ExplainV4Config`.

    Returns
    -------
    dict with keys: split, feat_comp, feat_struct, feat_mix_*, mean, budget,
    budget_se, budget_ci_*, players, config, timings (seconds_*).
    """
    xa = np.ascontiguousarray(np.asarray(x_a, dtype=np.float64))
    xb = np.ascontiguousarray(np.asarray(x_b, dtype=np.float64))
    ya = np.ascontiguousarray(np.asarray(y_a, dtype=np.float64).reshape(-1))
    yb = np.ascontiguousarray(np.asarray(y_b, dtype=np.float64).reshape(-1))
    if xa.ndim != 2 or xb.ndim != 2:
        raise ValueError("x_a and x_b must be 2-D")
    if xa.shape[1] != xb.shape[1]:
        raise ValueError("x_a and x_b must share p")
    if xa.shape[0] != ya.shape[0] or xb.shape[0] != yb.shape[0]:
        raise ValueError("row counts must match y")
    if not (
        np.isfinite(xa).all()
        and np.isfinite(xb).all()
        and np.isfinite(ya).all()
        and np.isfinite(yb).all()
    ):
        raise ValueError("non-finite input")
    return _core.explain_v4(
        xa,
        ya,
        xb,
        yb,
        k_channels=int(k_channels),
        seed=int(seed),
        n_boot=int(n_boot),
        alpha=float(alpha),
        max_bins=int(max_bins),
        pit_iters=int(pit_iters),
        n_u=int(n_u),
    )
