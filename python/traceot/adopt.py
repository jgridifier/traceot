"""Scikit-learn-style ADOPT estimator.

ADOPT is "Additive Optimal Transport Regression" by Song & Müller, Proceedings of
AISTATS 2026, PMLR v300:1558-1566, https://proceedings.mlr.press/v300/song26c.html .
This is an independent re-implementation from the paper (no public code exists);
see docs/REPRODUCTION_NOTES.md for every interpretive choice.
"""

from __future__ import annotations

import numpy as np

from ._base import (
    DEFAULT_BANDWIDTH_GRID,
    EstimatorMixin,
    Space,
    check_X,
    kfold_indices,
    resolve_backend,
)


class ADOPTRegressor(EstimatorMixin):
    """Additive optimal transport regression with metric-space responses.

    Parameters
    ----------
    space : {"wasserstein", "logcholesky"}
        ``"wasserstein"``: responses are 1-D distributions given as quantile functions
        on a common grid, shape (n, q). ``"logcholesky"``: responses are SPD matrices,
        shape (n, m, m), with the log-Cholesky metric (Lin 2019).
    bandwidth : "cv", float or array of shape (p,)
        ``"cv"`` selects a common bandwidth *fraction* ``c`` from ``bandwidth_grid`` by
        K-fold cross-validation; predictor j then uses ``h_j = c * range(X_j)``.
        A float or array is used as absolute bandwidth(s).
    bandwidth_grid : sequence of float, optional
        Candidate fractions for ``bandwidth="cv"``.
    cv : int
        Number of folds for bandwidth selection (the paper uses 5).
    kernel : {"gaussian", "epanechnikov"}
    tol, max_iter : stopping rule of the transport backfitting (relative tolerance).
    bounds : (lower, upper) or None
        Optional support bounds applied in the isotonic projection (wasserstein only).
    backend : {"auto", "rust", "reference"}
    random_state : int
        Seed for the CV fold assignment.

    Attributes
    ----------
    mean_ : Fréchet mean of the training responses (encoded).
    bandwidths_ : (p,) bandwidths used in the final fit.
    n_iter_, converged_ : backfitting diagnostics.
    cv_results_ : dict with the grid and mean CV error per candidate (if ``"cv"``).
    """

    def __init__(
        self,
        space: str = "wasserstein",
        bandwidth="cv",
        bandwidth_grid=None,
        cv: int = 5,
        kernel: str = "gaussian",
        tol: float = 1e-10,
        max_iter: int = 200,
        bounds=None,
        backend: str = "auto",
        random_state: int = 0,
    ):
        self.space = space
        self.bandwidth = bandwidth
        self.bandwidth_grid = bandwidth_grid
        self.cv = cv
        self.kernel = kernel
        self.tol = tol
        self.max_iter = max_iter
        self.bounds = bounds
        self.backend = backend
        self.random_state = random_state

    # -- low-level fit / predict on encoded responses ---------------------------------
    def _bounds(self):
        if self.bounds is None or self._space.name != "wasserstein":
            return None, None
        lo, hi = self.bounds
        return lo, hi

    def _fit_raw(self, X, Z, h):
        be = self._be
        if self._space.name == "wasserstein":
            lo, hi = self._bounds()
            return be.adopt_w2_fit(X, Z, h, self.kernel, self.tol, self.max_iter, lo, hi)
        return be.additive_euclid_fit(X, Z, h, self.kernel, self.tol, self.max_iter)

    def _predict_raw(self, X_train, fit, h, X_new):
        be = self._be
        if self._space.name == "wasserstein":
            lo, hi = self._bounds()
            return be.adopt_w2_predict(
                X_train, fit["P"], fit["M"], fit["mu"], h, X_new, self.kernel, lo, hi
            )
        return be.additive_euclid_predict(
            X_train, fit["P"], fit["M"], fit["mu"], h, X_new, self.kernel
        )

    # -- public API -------------------------------------------------------------------
    def fit(self, X, Y):
        self._be = resolve_backend(self.backend)
        self._space = Space(self.space, self._be)
        X = check_X(X)
        Z = self._space.encode(Y)
        if Z.shape[0] != X.shape[0]:
            raise ValueError("X and Y have different numbers of samples")
        n, p = X.shape
        ranges = np.ptp(X, axis=0)
        ranges = np.where(ranges > 0, ranges, 1.0)
        if isinstance(self.bandwidth, str):
            if self.bandwidth != "cv":
                raise ValueError("bandwidth must be 'cv', a float or an array")
            grid = np.asarray(self.bandwidth_grid or DEFAULT_BANDWIDTH_GRID, dtype=float)
            folds = kfold_indices(n, self.cv, self.random_state)
            errs = np.zeros(grid.shape[0])
            for g, c in enumerate(grid):
                h = c * ranges
                tot = 0.0
                for test in folds:
                    train = np.setdiff1d(np.arange(n), test)
                    fit = self._fit_raw(X[train], Z[train], h)
                    pred = self._predict_raw(X[train], fit, h, X[test])
                    tot += float(np.sum(self._space.dist_sq(pred, Z[test])))
                errs[g] = tot / n
            best = int(np.argmin(errs))
            self.cv_results_ = {"grid": grid, "mean_dist_sq": errs, "best_fraction": grid[best]}
            h = grid[best] * ranges
        else:
            h = np.broadcast_to(np.asarray(self.bandwidth, dtype=float), (p,)).copy()
            self.cv_results_ = None
        fit = self._fit_raw(X, Z, h)
        self.X_train_ = X
        self.bandwidths_ = h
        self.fit_ = fit
        self.mean_ = fit["mu"]
        self.n_iter_ = int(fit["n_iter"])
        self.converged_ = bool(fit["converged"])
        self.n_features_in_ = p
        return self

    def predict(self, X):
        """Predicted responses (quantile functions or SPD matrices)."""
        self._check_fitted()
        X = check_X(X, self.n_features_in_)
        return self._space.decode(self._predict_raw(self.X_train_, self.fit_, self.bandwidths_, X))

    def predict_encoded(self, X):
        """Predictions in the flat representation (quantiles / log-Cholesky coordinates)."""
        self._check_fitted()
        X = check_X(X, self.n_features_in_)
        return self._predict_raw(self.X_train_, self.fit_, self.bandwidths_, X)

    def score(self, X, Y) -> float:
        """Fréchet R^2: ``1 - sum d^2(Ŷ_i, Y_i) / sum d^2(mean_, Y_i)``."""
        self._check_fitted()
        Z = self._space.encode(Y)
        pred = self.predict_encoded(X)
        num = float(np.sum(self._space.dist_sq(pred, Z)))
        den = float(np.sum(self._space.dist_sq(self.mean_, Z)))
        return 1.0 - num / den if den > 0 else float("nan")

    def component(self, j: int, x) -> dict:
        """The j-th additive transport ``T_j(x) = T_{M_j, g_j(x)}`` at points ``x``.

        Returns ``{"g": g_j(x) (len(x), d), "center": M_j (d,)}`` in the encoded
        representation. For W2, ``T_j(x)`` maps ``y -> transport_apply(center, g[r], y)``.
        """
        self._check_fitted()
        x = np.atleast_1d(np.asarray(x, dtype=float))
        if self._space.name == "wasserstein":
            lo, hi = self._bounds()
            g = self._be.adopt_w2_component(
                self.X_train_[:, j], self.fit_["P"][j], self.bandwidths_[j], x, self.kernel, lo, hi
            )
        else:
            W = self._be.local_linear_weights(
                self.X_train_[:, j], x, self.bandwidths_[j], self.kernel
            )
            g = W @ self.fit_["P"][j]
        return {"g": g, "center": self.fit_["M"][j]}

    def transport(self, j: int, x: float, v) -> np.ndarray:
        """Apply ``T_j(x)`` to a response ``v`` (encoded). Displacement view of feature j."""
        comp = self.component(j, [x])
        v = np.asarray(v, dtype=float)
        if self._space.name == "wasserstein":
            return self._be.transport_apply(comp["center"], comp["g"][0], v)
        return v + comp["g"][0] - comp["center"]
