"""Shared estimator plumbing: parameters, input validation, response encoding.

No numerical logic lives here (architecture rule, see CONTRIBUTING.md): fitting,
cross-validation, distances and scoring are all in the Rust core.
"""

from __future__ import annotations

import inspect

import numpy as np

from . import _core, primitives

SPACES = ("wasserstein", "logcholesky")

#: Default bandwidth grid, as fractions of each predictor's observed range
#: (10 log-spaced values from 0.03 to 0.6, rounded to 4 decimals).
DEFAULT_BANDWIDTH_GRID = (0.03, 0.0418, 0.0584, 0.0814, 0.1136, 0.1585, 0.221, 0.3083, 0.4301, 0.6)


class EstimatorMixin:
    """Minimal scikit-learn-compatible ``get_params`` / ``set_params`` / ``__repr__``."""

    @classmethod
    def _param_names(cls):
        sig = inspect.signature(cls.__init__)
        return [p for p in sig.parameters if p != "self"]

    def get_params(self, deep: bool = True) -> dict:
        return {k: getattr(self, k) for k in self._param_names()}

    def set_params(self, **params):
        valid = set(self._param_names())
        for k, v in params.items():
            if k not in valid:
                raise ValueError(f"invalid parameter {k!r} for {type(self).__name__}")
            setattr(self, k, v)
        return self

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"

    def _check_fitted(self):
        if not hasattr(self, "mean_"):
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit() first")

    # Shared API on encoded responses -------------------------------------------------
    def predict(self, X):
        """Predicted responses (quantile functions or SPD matrices)."""
        self._check_fitted()
        return self._space.decode(self.predict_encoded(X))

    def score(self, X, Y) -> float:
        """Fréchet R^2: ``1 - sum d^2(Ŷ_i, Y_i) / sum d^2(mean_, Y_i)``."""
        self._check_fitted()
        Z = self._space.encode(Y)
        return _core.frechet_r2(self._space.name, self.predict_encoded(X), Z, self.mean_)


class Space:
    """Validates responses and converts them to and from the flat encoding."""

    def __init__(self, name: str):
        if name not in SPACES:
            raise ValueError(f"space must be one of {SPACES}, got {name!r}")
        self.name = name
        self.m = None

    def encode(self, Y) -> np.ndarray:
        Y = np.asarray(Y, dtype=np.float64)
        if not np.all(np.isfinite(Y)):
            raise ValueError("responses contain non-finite values")
        if self.name == "wasserstein":
            if Y.ndim != 2:
                raise ValueError("wasserstein responses must be (n, q) quantile functions")
            if np.any(Y[:, 1:] < Y[:, :-1] - 1e-12 * (1 + np.abs(Y[:, 1:]))):
                raise ValueError("each response row must be a non-decreasing quantile function")
            return np.ascontiguousarray(Y)
        if Y.ndim != 3 or Y.shape[1] != Y.shape[2]:
            raise ValueError("logcholesky responses must be (n, m, m) SPD matrices")
        if self.m is None:
            self.m = Y.shape[1]
        elif Y.shape[1] != self.m:
            raise ValueError(f"expected {self.m}x{self.m} matrices")
        return primitives.spd_to_coords(Y)

    def decode(self, Z: np.ndarray) -> np.ndarray:
        if self.name == "wasserstein":
            return Z
        return primitives.coords_to_spd(Z, self.m)


def check_X(X, p: int | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError("X must be 2-D (n_samples, n_features)")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values")
    if p is not None and X.shape[1] != p:
        raise ValueError(f"X has {X.shape[1]} features, expected {p}")
    return np.ascontiguousarray(X)


def check_bounds(bounds, space: str):
    if bounds is None or space != "wasserstein":
        return None, None
    lo, hi = bounds
    return (None if lo is None else float(lo)), (None if hi is None else float(hi))


def check_bandwidth_arg(bandwidth, grid, p: int):
    """Return ("cv", grid array) or ("fixed", h array); validation only."""
    if isinstance(bandwidth, str):
        if bandwidth != "cv":
            raise ValueError("bandwidth must be 'cv', a float or an array")
        g = np.ascontiguousarray(grid if grid is not None else DEFAULT_BANDWIDTH_GRID, float)
        if g.ndim != 1 or g.size == 0 or not np.all(g > 0):
            raise ValueError("bandwidth_grid must be a non-empty 1-D sequence of positive values")
        return "cv", g
    h = np.asarray(bandwidth, dtype=np.float64)
    h = np.full(p, float(h)) if h.ndim == 0 else np.ascontiguousarray(h)
    if h.shape != (p,) or not np.all(h > 0):
        raise ValueError(f"bandwidth must be positive, scalar or of shape ({p},)")
    return "fixed", h


class AdditiveFitMixin(EstimatorMixin):
    """Fit/predict for the backfitting models (ADOPT, L2-additive), delegating to Rust."""

    _method: str = "adopt"

    def _space_name(self) -> str:
        return self.space

    def fit(self, X, Y):
        self._space = Space(self._space_name())
        X = check_X(X)
        Z = self._space.encode(Y)
        if Z.shape[0] != X.shape[0]:
            raise ValueError("X and Y have different numbers of samples")
        n, p = X.shape
        lo, hi = check_bounds(self.bounds, self._space.name)
        mode, val = check_bandwidth_arg(self.bandwidth, self.bandwidth_grid, p)
        opts = dict(
            kernel=self.kernel, tol=float(self.tol), max_iter=int(self.max_iter), lower=lo, upper=hi
        )
        if mode == "cv":
            if not 2 <= int(self.cv) <= n:
                raise ValueError(f"cv must be between 2 and n_samples={n}")
            self.cv_results_ = _core.select_bandwidth_cv(
                self._method,
                self._space.name,
                X,
                Z,
                val,
                int(self.cv),
                int(self.random_state) & (2**64 - 1),
                **opts,
            )
            h = self.cv_results_["best_fraction"] * _core.column_ranges(X)
        else:
            self.cv_results_ = None
            h = val
        self.model_ = _core.AdditiveModel.fit(
            self._method, self._space.name, X, Z, np.ascontiguousarray(h, float), **opts
        )
        self.bandwidths_ = self.model_.bandwidths
        self.mean_ = self.model_.mu
        self.n_iter_ = int(self.model_.n_iter)
        self.converged_ = bool(self.model_.converged)
        self.n_features_in_ = p
        return self

    def predict_encoded(self, X):
        """Predictions in the flat representation (quantiles / log-Cholesky coordinates)."""
        self._check_fitted()
        return self.model_.predict(check_X(X, self.n_features_in_))

    def component(self, j: int, x) -> dict:
        """The j-th additive transport ``T_j(x) = T_{M_j, g_j(x)}`` at points ``x``.

        Returns ``{"g": g_j(x) (len(x), d), "center": M_j (d,)}`` in the encoded
        representation.
        """
        self._check_fitted()
        x = np.ascontiguousarray(np.atleast_1d(x), dtype=np.float64).ravel()
        return {"g": self.model_.component(int(j), x), "center": self.model_.centers[int(j)]}

    def transport(self, j: int, x: float, v) -> np.ndarray:
        """Apply ``T_j(x)`` to an encoded response ``v`` (displacement view of feature j)."""
        self._check_fitted()
        v = np.ascontiguousarray(v, dtype=np.float64).ravel()
        return self.model_.transport(int(j), float(x), v)
