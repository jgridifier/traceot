"""ADOPT transport backfitting for distributional responses in W2 (reference).

Implements Algorithm 1 of Song & Müller, "ADOPT: Additive Optimal Transport
Regression", AISTATS 2026 (PMLR v300, https://proceedings.mlr.press/v300/song26c.html)
for responses in the 1-D Wasserstein space, using the quantile-grid geometry of
:mod:`reference.w2`. Every interpretive choice is recorded in
docs/REPRODUCTION_NOTES.md.

Representation of the fitted model
----------------------------------
* ``mu``  (q,)      sample Fréchet mean of the responses.
* ``G``   (p, n, q) ``g_j(X_ij)``: local Fréchet regression of the j-th partial
  transport residuals, evaluated at the training points.
* ``M``   (p, q)    ``Ê⊕ g_j(X_j)``: Fréchet mean of ``G[j]`` over i (transport centring).
* ``P``   (p, n, q) the final partial transport residuals ``P̂_{⊕ij}(µ̂)``; needed to
  evaluate ``g_j`` at new x.

The j-th transport ``T̂_j(x) = T_{M_j, g_j(x)}`` acts on a distribution with quantile
function ``v`` as ``v -> transport_apply(M_j, g_j(x), v)``; its inverse swaps knots.
``A ⊕ B`` means "apply A, then B", so ``⊕_{j=1..p} T_j`` applies ``T_1`` first.
"""

from __future__ import annotations

import numpy as np

from .smoother import ABS_TOL_FLOOR, check_bandwidths, local_linear_weights
from .w2 import pava_rows, transport_apply, w2_sq


def partial_residuals(Y, G, M, mu, active, j):
    """Partial transport residuals ``P̂_{⊕ij}(µ̂)`` of eq. (10) for all i.

    Order of application to ``mu`` (``⊕`` = "then"):
    ``T_{j-1}^{-1}, ..., T_1^{-1}``, then ``T_{µ̂, Y_i}``, then ``T_p^{-1}, ..., T_{j+1}^{-1}``.
    Identity transports (``active[k]`` False, i.e. not yet fitted) are skipped.
    """
    n = Y.shape[0]
    p = G.shape[0]
    P = np.empty_like(Y)
    for i in range(n):
        z = mu.copy()
        for k in range(j - 1, -1, -1):
            if active[k]:
                z = transport_apply(G[k, i], M[k], z)  # inverse of T_{M_k -> g_k(X_ik)}
        z = transport_apply(mu, Y[i], z)  # T_{µ̂, Y_i}
        for k in range(p - 1, j, -1):
            if active[k]:
                z = transport_apply(G[k, i], M[k], z)
        P[i] = z
    return P


def fitted_values(G, M, mu, active):
    """``Ŷ_i = [T_1(X_i1) ⊕ ... ⊕ T_p(X_ip)](µ̂)`` for every training row i."""
    p, n, _ = G.shape
    out = np.empty((n, mu.shape[0]))
    for i in range(n):
        z = mu.copy()
        for k in range(p):
            if active[k]:
                z = transport_apply(M[k], G[k, i], z)
        out[i] = z
    return out


def adopt_w2_fit(
    X: np.ndarray,
    Y: np.ndarray,
    h: np.ndarray,
    kernel: str = "gaussian",
    tol: float = 1e-10,
    max_iter: int = 200,
    lower: float | None = None,
    upper: float | None = None,
) -> dict:
    """Fit ADOPT by transport backfitting (Algorithm 1).

    Parameters
    ----------
    X : (n, p) predictors.
    Y : (n, q) response quantile functions on a common grid (rows non-decreasing).
    h : (p,) bandwidths, one per predictor.
    kernel : smoothing kernel for the local Fréchet regression.
    tol : stop when ``mean_i d^2(Ŷ_i^(t), Ŷ_i^(t-1)) < tol * V`` where
        ``V = mean_i d^2(Y_i, µ̂)`` (scale-free version of the paper's epsilon).
    max_iter : maximum number of backfitting cycles.
    lower, upper : optional support bounds used in the isotonic projection.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    n, p = X.shape
    q = Y.shape[1]
    h = check_bandwidths(h, p)
    if Y.shape[0] != n:
        raise ValueError("X and Y must have the same number of rows")
    mu = Y.mean(axis=0)
    V = float(np.mean(w2_sq(Y, mu)))
    S = [local_linear_weights(X[:, j], X[:, j], h[j], kernel) for j in range(p)]
    G = np.zeros((p, n, q))
    M = np.zeros((p, q))
    P = np.zeros((p, n, q))
    active = [False] * p
    Yhat_prev = np.tile(mu, (n, 1))
    history: list[float] = []
    converged = False
    it = 0
    for t in range(1, max_iter + 1):
        it = t
        for j in range(p):
            P[j] = partial_residuals(Y, G, M, mu, active, j)
            G[j] = pava_rows(S[j] @ P[j], lower, upper)
            M[j] = G[j].mean(axis=0)
            active[j] = True
        Yhat = fitted_values(G, M, mu, active)
        crit = float(np.mean(w2_sq(Yhat, Yhat_prev)))
        history.append(crit)
        Yhat_prev = Yhat
        if crit <= max(tol * V, ABS_TOL_FLOOR):
            converged = True
            break
    return {
        "mu": mu,
        "G": G,
        "M": M,
        "P": P,
        "fitted": Yhat_prev,
        "n_iter": it,
        "converged": converged,
        "history": np.asarray(history),
    }


def adopt_w2_component(
    X_train: np.ndarray,
    P_j: np.ndarray,
    h_j: float,
    x_eval: np.ndarray,
    kernel: str = "gaussian",
    lower: float | None = None,
    upper: float | None = None,
) -> np.ndarray:
    """``g_j(x)`` at new points: local Fréchet regression of the stored residuals."""
    W = local_linear_weights(X_train, x_eval, h_j, kernel)
    return pava_rows(W @ P_j, lower, upper)


def adopt_w2_predict(
    X_train: np.ndarray,
    P: np.ndarray,
    M: np.ndarray,
    mu: np.ndarray,
    h: np.ndarray,
    X_new: np.ndarray,
    kernel: str = "gaussian",
    lower: float | None = None,
    upper: float | None = None,
) -> np.ndarray:
    """Predict ``[T_1(x_1) ⊕ ... ⊕ T_p(x_p)](µ̂)`` at the rows of ``X_new``."""
    X_train = np.atleast_2d(np.asarray(X_train, dtype=float))
    X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
    p = X_train.shape[1]
    h = check_bandwidths(h, p)
    Gnew = np.stack(
        [
            adopt_w2_component(X_train[:, j], P[j], h[j], X_new[:, j], kernel, lower, upper)
            for j in range(p)
        ]
    )
    return fitted_values(Gnew, M, mu, [True] * p)
