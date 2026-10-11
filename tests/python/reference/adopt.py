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

import itertools

import numpy as np

from .additive import additive_euclid_fit
from .smoother import (
    ABS_TOL_FLOOR,
    binned_smooth,
    check_bandwidths,
    local_linear_weights,
    reflected_local_linear_weights,
)
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


def _paper_w2(damping, certificate, composition, centering, project_prediction, bins, boundary):
    return (
        damping == 1.0
        and not certificate
        and composition == "given"
        and centering == "frechet"
        and not project_prediction
        and bins is None
        and boundary == "none"
    )


def _smooth_matrix(x, resid, h, kernel, bins, boundary, prepared):
    if prepared is not None:
        return prepared @ resid
    if bins is not None:
        return binned_smooth(x, resid, h, x, bins, kernel)
    if boundary == "none":
        return local_linear_weights(x, x, h, kernel) @ resid
    if boundary == "reflect":
        return reflected_local_linear_weights(x, x, h, kernel) @ resid
    raise ValueError(f"unknown boundary {boundary!r}")


def _smooth_at(x_train, resid, h, x_eval, kernel, bins, boundary):
    if bins is not None:
        return binned_smooth(x_train, resid, h, x_eval, bins, kernel)
    if boundary == "none":
        return local_linear_weights(x_train, x_eval, h, kernel) @ resid
    if boundary == "reflect":
        return reflected_local_linear_weights(x_train, x_eval, h, kernel) @ resid
    raise ValueError(f"unknown boundary {boundary!r}")


def _pushforward_center(G, mu, lower, upper):
    """Average pushforward of ``mu`` is ``mu``, then two isotonic projections."""
    anchor = G.mean(axis=0)
    pushed = np.vstack([transport_apply(anchor, G[i], mu) for i in range(G.shape[0])])
    qbar = pushed.mean(axis=0)
    shifted = pushed - qbar + mu
    out = pava_rows(shifted, lower, upper)
    col = out.mean(axis=0)
    out = pava_rows(out + (mu - col), lower, upper)
    return out, out.mean(axis=0)


def _update_component(g_old, raw, mu, lam, centering, lower, upper):
    g_new = pava_rows(raw, lower, upper)
    m_new = g_new.mean(axis=0)
    if centering == "pushforward":
        g_new, m_new = _pushforward_center(g_new, mu, lower, upper)
    if lam != 1.0:
        g_new = (1.0 - lam) * g_old + lam * g_new
        if centering == "pushforward":
            g_new, m_new = _pushforward_center(g_new, mu, lower, upper)
        else:
            m_new = g_new.mean(axis=0)
    return g_new, m_new


def _finish(fitted, project_prediction, composition, lower, upper):
    if project_prediction or composition == "tangent":
        return pava_rows(fitted, lower, upper)
    return fitted


def _pack(mu, G, M, P, fitted, n_iter, converged, history, n_rejected, orders):
    return {
        "mu": mu,
        "G": G,
        "M": M,
        "P": P,
        "fitted": fitted,
        "n_iter": n_iter,
        "converged": converged,
        "history": np.asarray(history, dtype=float),
        "n_rejected": int(n_rejected),
        "orders": orders,
    }


def _adopt_w2_fit_given(
    X, Y, h, kernel, tol, max_iter, lower, upper, damping, certificate, centering, bins, boundary
):
    n, p = X.shape
    q = Y.shape[1]
    mu = Y.mean(axis=0)
    V = float(np.mean(w2_sq(Y, mu)))
    dense = bins is None and boundary == "none"
    prepared = [
        local_linear_weights(X[:, j], X[:, j], h[j], kernel) if dense else None for j in range(p)
    ]
    G = np.zeros((p, n, q))
    M = np.zeros((p, q))
    P = np.zeros((p, n, q))
    active = [False] * p
    prev = np.tile(mu, (n, 1))
    loss_prev = float(np.mean(w2_sq(prev, Y)))
    history: list[float] = []
    converged = False
    it = 0
    n_rejected = 0
    thresh = max(tol * V, ABS_TOL_FLOOR)
    for t in range(1, max_iter + 1):
        it = t
        lam = float(damping)
        accepted = None
        while True:
            Gs, Ms, Ps = G.copy(), M.copy(), P.copy()
            act = list(active)
            for j in range(p):
                Ps[j] = partial_residuals(Y, Gs, Ms, mu, act, j)
                raw = _smooth_matrix(X[:, j], Ps[j], h[j], kernel, bins, boundary, prepared[j])
                Gs[j], Ms[j] = _update_component(Gs[j], raw, mu, lam, centering, lower, upper)
                act[j] = True
            fitted = fitted_values(Gs, Ms, mu, act)
            if not certificate:
                accepted = (Gs, Ms, Ps, act, fitted)
                break
            loss = float(np.mean(w2_sq(fitted, Y)))
            if loss <= loss_prev + 1e-12:
                loss_prev = loss
                accepted = (Gs, Ms, Ps, act, fitted)
                break
            n_rejected += 1
            lam *= 0.5
            if lam < 1e-8:
                converged = True
                it = t - 1
                break
        if accepted is None:
            break
        G, M, P, active, yhat = accepted
        crit = float(np.mean(w2_sq(yhat, prev)))
        history.append(crit)
        prev = yhat
        if crit <= thresh:
            converged = True
            break
    return G, M, P, prev, it, converged, history, n_rejected


def _permutations(p):
    return [list(t) for t in itertools.permutations(range(p))]


def adopt_w2_fit(
    X: np.ndarray,
    Y: np.ndarray,
    h: np.ndarray,
    kernel: str = "gaussian",
    tol: float = 1e-10,
    max_iter: int = 200,
    lower: float | None = None,
    upper: float | None = None,
    damping: float = 1.0,
    certificate: bool = False,
    composition: str = "given",
    centering: str = "frechet",
    project_prediction: bool = False,
    bins: int | None = None,
    boundary: str = "none",
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
    if composition not in ("given", "average", "tangent"):
        raise ValueError(f"unknown composition {composition!r}")
    if centering not in ("frechet", "pushforward"):
        raise ValueError(f"unknown centering {centering!r}")
    if not (damping > 0.0 and damping <= 1.0):
        raise ValueError("damping must lie in (0, 1]")
    if composition == "tangent":
        eu = additive_euclid_fit(
            X,
            Y,
            h,
            kernel,
            tol,
            max_iter,
            damping=damping,
            certificate=certificate,
            bins=bins,
            boundary=boundary,
        )
        fitted = np.tile(mu, (n, 1))
        for j in range(p):
            fitted += eu["G"][j] - eu["M"][j]
        fitted = pava_rows(fitted, lower, upper)
        eu["fitted"] = fitted
        eu["orders"] = []
        return eu
    if composition == "average":
        if p > 5:
            raise ValueError("averaged-order composition supports at most 5 features")
        orders = []
        n_iter = 0
        converged = True
        n_rejected = 0
        for perm in _permutations(p):
            inner = _adopt_w2_fit_given(
                X[:, perm],
                Y,
                h[list(perm)],
                kernel,
                tol,
                max_iter,
                lower,
                upper,
                damping,
                certificate,
                centering,
                bins,
                boundary,
            )
            Gs, Ms, Ps, _yhat, it, conv, _hist, rej = inner
            n_iter = max(n_iter, it)
            converged = converged and conv
            n_rejected += rej
            orders.append(
                {
                    "perm": np.asarray(perm, dtype=np.int64),
                    "G": Gs,
                    "M": Ms,
                    "P": Ps,
                }
            )
        acc = np.zeros((n, q))
        for order in orders:
            acc += fitted_values(order["G"], order["M"], mu, [True] * p)
        acc *= 1.0 / len(orders)
        scale = 1.0 / len(orders)
        G = np.zeros((p, n, q))
        M = np.zeros((p, q))
        P = np.zeros((p, n, q))
        for order in orders:
            for k, j in enumerate(order["perm"]):
                G[j] += order["G"][k] * scale
                P[j] += order["P"][k] * scale
                M[j] += order["M"][k] * scale
        fitted = _finish(acc, project_prediction, composition, lower, upper)
        return _pack(mu, G, M, P, fitted, n_iter, converged, [], n_rejected, orders)
    if _paper_w2(damping, certificate, composition, centering, project_prediction, bins, boundary):
        # Bit-identical to the original dense undamped loop.
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
        return _pack(mu, G, M, P, Yhat_prev, it, converged, history, 0, [])
    G, M, P, prev, it, converged, history, n_rejected = _adopt_w2_fit_given(
        X,
        Y,
        h,
        kernel,
        tol,
        max_iter,
        lower,
        upper,
        damping,
        certificate,
        centering,
        bins,
        boundary,
    )
    fitted = _finish(prev, project_prediction, composition, lower, upper)
    return _pack(mu, G, M, P, fitted, it, converged, history, n_rejected, [])


def adopt_w2_component(
    X_train: np.ndarray,
    P_j: np.ndarray,
    h_j: float,
    x_eval: np.ndarray,
    kernel: str = "gaussian",
    lower: float | None = None,
    upper: float | None = None,
    bins: int | None = None,
    boundary: str = "none",
) -> np.ndarray:
    """``g_j(x)`` at new points: local Fréchet regression of the stored residuals."""
    raw = _smooth_at(np.ravel(X_train), P_j, float(h_j), np.ravel(x_eval), kernel, bins, boundary)
    return pava_rows(raw, lower, upper)


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
    project_prediction: bool = False,
    bins: int | None = None,
    boundary: str = "none",
) -> np.ndarray:
    """Predict ``[T_1(x_1) ⊕ ... ⊕ T_p(x_p)](µ̂)`` at the rows of ``X_new``."""
    X_train = np.atleast_2d(np.asarray(X_train, dtype=float))
    X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
    p = X_train.shape[1]
    h = check_bandwidths(h, p)
    Gnew = np.stack(
        [
            adopt_w2_component(
                X_train[:, j], P[j], h[j], X_new[:, j], kernel, lower, upper, bins, boundary
            )
            for j in range(p)
        ]
    )
    fitted = fitted_values(Gnew, M, mu, [True] * p)
    return _finish(fitted, project_prediction, "given", lower, upper)


def adopt_w2_predict_averaged(
    X_train: np.ndarray,
    orders: list,
    mu: np.ndarray,
    h: np.ndarray,
    X_new: np.ndarray,
    kernel: str = "gaussian",
    lower: float | None = None,
    upper: float | None = None,
    project_prediction: bool = False,
    bins: int | None = None,
    boundary: str = "none",
) -> np.ndarray:
    """Average composed predictions over stored feature orders."""
    if not orders:
        raise ValueError("averaged prediction needs at least one stored order")
    X_train = np.atleast_2d(np.asarray(X_train, dtype=float))
    X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
    h = check_bandwidths(h, X_train.shape[1])
    acc = np.zeros((X_new.shape[0], np.asarray(mu).shape[0]))
    for order in orders:
        perm = list(np.asarray(order["perm"], dtype=int))
        pred = adopt_w2_predict(
            X_train[:, perm],
            order["P"],
            order["M"],
            mu,
            h[perm],
            X_new[:, perm],
            kernel,
            lower,
            upper,
            False,
            bins,
            boundary,
        )
        acc += pred
    acc *= 1.0 / len(orders)
    return _finish(acc, project_prediction, "given", lower, upper)
