"""Comparison methods used in the ADOPT paper's simulations, plus one of ours.

* :class:`FrechetMeanRegressor` - constant Fréchet mean (paper: "FM").
* :class:`GlobalFrechetRegressor` - global Fréchet regression of Petersen & Müller
  (2019) (paper: "GF").
* :class:`L2AdditiveQuantileRegressor` - **not in the paper**: extrinsic additive
  backfitting of quantile functions as L2 vectors with the same smoothers, followed by
  isotonic projection. It is the "additive in quantile space" model that TRACE builds
  on, included for comparison. It is *not* the ADR method of Han et al. (2020).
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


class _SimpleBase(EstimatorMixin):
    def score(self, X, Y) -> float:
        self._check_fitted()
        Z = self._space.encode(Y)
        pred = self.predict_encoded(X)
        num = float(np.sum(self._space.dist_sq(pred, Z)))
        den = float(np.sum(self._space.dist_sq(self.mean_, Z)))
        return 1.0 - num / den if den > 0 else float("nan")

    def predict(self, X):
        return self._space.decode(self.predict_encoded(X))


class FrechetMeanRegressor(_SimpleBase):
    """Predicts the sample Fréchet mean for every x."""

    def __init__(self, space: str = "wasserstein", backend: str = "auto"):
        self.space = space
        self.backend = backend

    def fit(self, X, Y):
        self._space = Space(self.space, resolve_backend(self.backend))
        X = check_X(X)
        Z = self._space.encode(Y)
        self.mean_ = Z.mean(axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def predict_encoded(self, X):
        self._check_fitted()
        X = check_X(X, self.n_features_in_)
        return np.tile(self.mean_, (X.shape[0], 1))


class GlobalFrechetRegressor(_SimpleBase):
    """Global Fréchet regression (Petersen & Müller 2019).

    W2: weighted average of quantile functions with the global weights, projected onto
    monotone functions. Log-Cholesky: the same weights on coordinates (flat space).
    """

    def __init__(self, space: str = "wasserstein", bounds=None, backend: str = "auto"):
        self.space = space
        self.bounds = bounds
        self.backend = backend

    def fit(self, X, Y):
        self._be = resolve_backend(self.backend)
        self._space = Space(self.space, self._be)
        self.X_train_ = check_X(X)
        self.Z_train_ = self._space.encode(Y)
        self.mean_ = self.Z_train_.mean(axis=0)
        self.n_features_in_ = self.X_train_.shape[1]
        return self

    def predict_encoded(self, X):
        self._check_fitted()
        X = check_X(X, self.n_features_in_)
        W = self._be.global_frechet_weights(self.X_train_, X)
        raw = W @ self.Z_train_
        if self._space.name == "wasserstein":
            lo, hi = self.bounds if self.bounds is not None else (None, None)
            return self._be.pava_rows(raw, lo, hi)
        return raw


class L2AdditiveQuantileRegressor(_SimpleBase):
    """Extrinsic additive model for quantile functions (ours; see module docstring)."""

    def __init__(
        self,
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
        self.bandwidth = bandwidth
        self.bandwidth_grid = bandwidth_grid
        self.cv = cv
        self.kernel = kernel
        self.tol = tol
        self.max_iter = max_iter
        self.bounds = bounds
        self.backend = backend
        self.random_state = random_state

    def _predict_raw(self, X_train, fit, h, X_new):
        raw = self._be.additive_euclid_predict(
            X_train, fit["P"], fit["M"], fit["mu"], h, X_new, self.kernel
        )
        lo, hi = self.bounds if self.bounds is not None else (None, None)
        return self._be.pava_rows(raw, lo, hi)

    def fit(self, X, Y):
        self._be = resolve_backend(self.backend)
        self._space = Space("wasserstein", self._be)
        X = check_X(X)
        Z = self._space.encode(Y)
        n, p = X.shape
        ranges = np.ptp(X, axis=0)
        ranges = np.where(ranges > 0, ranges, 1.0)
        fitf = self._be.additive_euclid_fit
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
                    fit = fitf(X[train], Z[train], h, self.kernel, self.tol, self.max_iter)
                    pred = self._predict_raw(X[train], fit, h, X[test])
                    tot += float(np.sum(self._space.dist_sq(pred, Z[test])))
                errs[g] = tot / n
            best = int(np.argmin(errs))
            self.cv_results_ = {"grid": grid, "mean_dist_sq": errs, "best_fraction": grid[best]}
            h = grid[best] * ranges
        else:
            h = np.broadcast_to(np.asarray(self.bandwidth, dtype=float), (p,)).copy()
            self.cv_results_ = None
        self.fit_ = fitf(X, Z, h, self.kernel, self.tol, self.max_iter)
        self.X_train_ = X
        self.bandwidths_ = h
        self.mean_ = self.fit_["mu"]
        self.n_iter_ = int(self.fit_["n_iter"])
        self.converged_ = bool(self.fit_["converged"])
        self.n_features_in_ = p
        return self

    def predict_encoded(self, X):
        self._check_fitted()
        X = check_X(X, self.n_features_in_)
        return self._predict_raw(self.X_train_, self.fit_, self.bandwidths_, X)
