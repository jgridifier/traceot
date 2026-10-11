"""Linear smoothers used by local and global Fréchet regression (reference).

For a metric space where the weighted Fréchet mean of the data is a projection of
the weighted Euclidean average (W2 on quantile functions: isotonic projection;
flat log-Cholesky coordinates: identity), Fréchet regression reduces to a linear
smoother applied to the responses followed by that projection. These functions
return the smoother weight matrices.
"""

from __future__ import annotations

import numpy as np

KERNELS = ("gaussian", "epanechnikov")

#: If 1 - u1^2 / (u0 u2) falls below this, the local linear weights are treated as
#: degenerate and the smoother falls back to Nadaraya-Watson (then nearest neighbour).
DEGENERACY_RTOL = 1e-10


#: Absolute floor of the backfitting stopping threshold (handles V = 0 exactly).
ABS_TOL_FLOOR = 1e-30


def check_bandwidths(h, p: int) -> np.ndarray:
    """Return ``h`` as a float array of shape (p,), all entries > 0, or raise."""
    h = np.asarray(h, dtype=float).ravel()
    if h.shape != (p,):
        raise ValueError(f"expected {p} bandwidths, got {h.shape[0]}")
    if not np.all(h > 0):
        raise ValueError("bandwidths must be > 0")
    return h


def kernel_values(z: np.ndarray, kernel: str = "gaussian") -> np.ndarray:
    """Unnormalised kernel K(z); normalising constants cancel in all smoothers."""
    z = np.asarray(z, dtype=float)
    if kernel == "gaussian":
        return np.exp(-0.5 * z * z)
    if kernel == "epanechnikov":
        return np.where(np.abs(z) < 1.0, 1.0 - z * z, 0.0)
    raise ValueError(f"unknown kernel {kernel!r}; expected one of {KERNELS}")


def local_linear_weights(
    x_train: np.ndarray, x_eval: np.ndarray, h: float, kernel: str = "gaussian"
) -> np.ndarray:
    """Local linear (local Fréchet) weights, Petersen & Müller (2019) eq. (3).

    Row ``r`` holds ``w_i(x_r) = s_i(x_r) / n`` with
    ``s_i(x) = K_h(X_i - x) [u2 - u1 (X_i - x)] / sigma0^2`` and
    ``u_k = n^{-1} sum_i K_h(X_i - x)(X_i - x)^k``, ``sigma0^2 = u0 u2 - u1^2``.
    Rows sum to one (we divide by the row sum, which equals n analytically).
    Weights can be negative.

    Degenerate windows (``1 - u1^2/(u0 u2) < DEGENERACY_RTOL``) fall back to
    Nadaraya-Watson weights, and windows with no kernel mass to the nearest
    training point (lowest index on ties).
    """
    x_train = np.asarray(x_train, dtype=float).ravel()
    x_eval = np.asarray(x_eval, dtype=float).ravel()
    if not h > 0:
        raise ValueError("bandwidth h must be > 0")
    n = x_train.shape[0]
    W = np.zeros((x_eval.shape[0], n))
    for r, x0 in enumerate(x_eval):
        d = x_train - x0
        K = kernel_values(d / h, kernel)
        u0 = float(np.sum(K))
        u1 = float(np.sum(K * d))
        u2 = float(np.sum(K * d * d))
        ok = u0 > 0.0 and u2 > 0.0 and (1.0 - (u1 * u1) / (u0 * u2)) > DEGENERACY_RTOL
        if ok:
            s = K * (u2 - u1 * d)
            W[r] = s / float(np.sum(s))
        elif u0 > 0.0:
            W[r] = K / u0
        else:
            W[r, int(np.argmin(np.abs(d)))] = 1.0
    return W


def global_frechet_weights(X_train: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
    """Global Fréchet regression weights (Petersen & Müller 2019).

    ``w_i(x) = n^{-1} [1 + (X_i - Xbar)^T S^{-1} (x - Xbar)]`` with
    ``S = n^{-1} sum_i (X_i - Xbar)(X_i - Xbar)^T``. Rows sum to one.
    """
    X_train = np.atleast_2d(np.asarray(X_train, dtype=float))
    X_eval = np.atleast_2d(np.asarray(X_eval, dtype=float))
    n = X_train.shape[0]
    xbar = X_train.mean(axis=0)
    C = X_train - xbar
    S = C.T @ C / n
    A = np.linalg.solve(S, (X_eval - xbar).T)  # (p, m)
    return (1.0 + (C @ A).T) / n
