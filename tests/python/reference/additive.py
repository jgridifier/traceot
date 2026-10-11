"""Additive backfitting for responses with flat (Euclidean) coordinates (reference).

Two uses:

1. **ADOPT in a flat metric space.** Under the log-Cholesky metric, SPD matrices
   are isometric to Euclidean vectors (strict lower Cholesky part, log of the
   diagonal), geodesic transports are translations of those vectors, and ``⊕`` is
   addition. Algorithm 1 of the ADOPT paper then *is* classical backfitting with
   local linear smoothers in these coordinates (Fréchet means = coordinate means).
2. **An extrinsic L2-additive baseline for quantile functions** (ours, not in the
   ADOPT paper): backfit quantile functions as vectors and project predictions onto
   monotone functions afterwards.

The iteration mirrors :func:`traceot.reference.adopt.adopt_w2_fit` exactly
(Gauss-Seidel order, centring, stopping rule), with translations in place of maps.
"""

from __future__ import annotations

import numpy as np

from .smoother import ABS_TOL_FLOOR, check_bandwidths, local_linear_weights


def additive_euclid_fit(
    X: np.ndarray,
    Z: np.ndarray,
    h: np.ndarray,
    kernel: str = "gaussian",
    tol: float = 1e-10,
    max_iter: int = 200,
) -> dict:
    """Backfit ``Z_i ≈ mu + sum_j (g_j(X_ij) - M_j)`` (see module docstring)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    n, p = X.shape
    d = Z.shape[1]
    h = check_bandwidths(h, p)
    if Z.shape[0] != n:
        raise ValueError("X and Z must have the same number of rows")
    mu = Z.mean(axis=0)
    V = float(np.mean(np.mean((Z - mu) ** 2, axis=1)))
    S = [local_linear_weights(X[:, j], X[:, j], h[j], kernel) for j in range(p)]
    G = np.zeros((p, n, d))
    M = np.zeros((p, d))
    P = np.zeros((p, n, d))
    fitted_prev = np.tile(mu, (n, 1))
    history: list[float] = []
    converged = False
    it = 0
    for t in range(1, max_iter + 1):
        it = t
        for j in range(p):
            others = np.zeros((n, d))
            for k in range(p):
                if k != j:
                    others += G[k] - M[k]
            P[j] = Z - others
            G[j] = S[j] @ P[j]
            M[j] = G[j].mean(axis=0)
        fitted = mu + np.sum(G - M[:, None, :], axis=0)
        crit = float(np.mean(np.mean((fitted - fitted_prev) ** 2, axis=1)))
        history.append(crit)
        fitted_prev = fitted
        if crit <= max(tol * V, ABS_TOL_FLOOR):
            converged = True
            break
    return {
        "mu": mu,
        "G": G,
        "M": M,
        "P": P,
        "fitted": fitted_prev,
        "n_iter": it,
        "converged": converged,
        "history": np.asarray(history),
    }


def additive_euclid_predict(
    X_train: np.ndarray,
    P: np.ndarray,
    M: np.ndarray,
    mu: np.ndarray,
    h: np.ndarray,
    X_new: np.ndarray,
    kernel: str = "gaussian",
) -> np.ndarray:
    """``mu + sum_j (g_j(x_j) - M_j)`` with ``g_j`` re-smoothed from the stored residuals."""
    X_train = np.atleast_2d(np.asarray(X_train, dtype=float))
    X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
    p = X_train.shape[1]
    h = check_bandwidths(h, p)
    out = np.tile(mu, (X_new.shape[0], 1))
    for j in range(p):
        W = local_linear_weights(X_train[:, j], X_new[:, j], h[j], kernel)
        out += W @ P[j] - M[j]
    return out
