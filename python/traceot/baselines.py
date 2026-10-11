"""Comparison methods used in the ADOPT paper's simulations, plus one of ours.

* :class:`FrechetMeanRegressor` - constant Fréchet mean (paper: "FM").
* :class:`GlobalFrechetRegressor` - global Fréchet regression of Petersen & Müller
  (2019) (paper: "GF").
* :class:`L2AdditiveQuantileRegressor` - **not in the paper**: extrinsic additive
  backfitting of quantile functions as L2 vectors with the same smoothers, followed by
  isotonic projection. It is the "additive in quantile space" model that TRACE builds
  on, included for comparison. It is *not* the ADR method of Han et al. (2020).

All computation happens in the Rust core; these classes only validate and convert.
"""

from __future__ import annotations

import numpy as np

from . import _core
from ._base import AdditiveFitMixin, EstimatorMixin, Space, check_bounds, check_X


class FrechetMeanRegressor(EstimatorMixin):
    """Predicts the sample Fréchet mean for every x."""

    def __init__(self, space: str = "wasserstein"):
        self.space = space

    def fit(self, X, Y):
        self._space = Space(self.space)
        X = check_X(X)
        Z = self._space.encode(Y)
        self.mean_ = _core.frechet_mean(Z)
        self.n_features_in_ = X.shape[1]
        return self

    def predict_encoded(self, X):
        self._check_fitted()
        X = check_X(X, self.n_features_in_)
        return np.broadcast_to(self.mean_, (X.shape[0], self.mean_.shape[0])).copy()


class GlobalFrechetRegressor(EstimatorMixin):
    """Global Fréchet regression (Petersen & Müller 2019).

    W2: weighted average of quantile functions with the global weights, projected onto
    monotone functions. Log-Cholesky: the same weights on coordinates (flat space).
    """

    def __init__(self, space: str = "wasserstein", bounds=None):
        self.space = space
        self.bounds = bounds

    def fit(self, X, Y):
        self._space = Space(self.space)
        self.X_train_ = check_X(X)
        self.Z_train_ = self._space.encode(Y)
        if self.Z_train_.shape[0] != self.X_train_.shape[0]:
            raise ValueError("X and Y have different numbers of samples")
        self.mean_ = _core.frechet_mean(self.Z_train_)
        self.n_features_in_ = self.X_train_.shape[1]
        return self

    def predict_encoded(self, X):
        self._check_fitted()
        X = check_X(X, self.n_features_in_)
        lo, hi = check_bounds(self.bounds, self._space.name)
        return _core.global_frechet_predict(
            self._space.name, self.X_train_, self.Z_train_, X, lo, hi
        )


class L2AdditiveQuantileRegressor(AdditiveFitMixin):
    """Extrinsic additive model for quantile functions (ours; see module docstring)."""

    _method = "l2additive"

    def __init__(
        self,
        bandwidth="cv",
        bandwidth_grid=None,
        cv: int = 5,
        kernel: str = "gaussian",
        tol: float = 1e-10,
        max_iter: int = 200,
        bounds=None,
        random_state: int = 0,
    ):
        self.bandwidth = bandwidth
        self.bandwidth_grid = bandwidth_grid
        self.cv = cv
        self.kernel = kernel
        self.tol = tol
        self.max_iter = max_iter
        self.bounds = bounds
        self.random_state = random_state

    def _space_name(self) -> str:
        return "wasserstein"
