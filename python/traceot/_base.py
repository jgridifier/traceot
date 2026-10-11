"""Shared estimator machinery: parameter handling, spaces, bandwidth cross-validation."""

from __future__ import annotations

import inspect

import numpy as np

from ._backend import get_backend

SPACES = ("wasserstein", "logcholesky")

#: Default bandwidth grid, as fractions of each predictor's observed range.
DEFAULT_BANDWIDTH_GRID = tuple(np.round(np.geomspace(0.03, 0.6, 10), 4))


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


class Space:
    """Encodes responses into flat arrays and back, and measures squared distances."""

    def __init__(self, name: str, backend):
        if name not in SPACES:
            raise ValueError(f"space must be one of {SPACES}, got {name!r}")
        self.name = name
        self.be = backend
        self.m = None

    def encode(self, Y) -> np.ndarray:
        Y = np.asarray(Y, dtype=float)
        if self.name == "wasserstein":
            if Y.ndim != 2:
                raise ValueError("wasserstein responses must be (n, q) quantile functions")
            if not np.all(np.isfinite(Y)):
                raise ValueError("responses contain non-finite values")
            if np.any(np.diff(Y, axis=1) < -1e-12 * (1 + np.abs(Y[:, 1:]))):
                raise ValueError("each response row must be a non-decreasing quantile function")
            return np.ascontiguousarray(Y)
        if Y.ndim != 3 or Y.shape[1] != Y.shape[2]:
            raise ValueError("logcholesky responses must be (n, m, m) SPD matrices")
        if self.m is None:
            self.m = Y.shape[1]
        elif Y.shape[1] != self.m:
            raise ValueError(f"expected {self.m}x{self.m} matrices")
        return self.be.spd_to_coords(Y)

    def decode(self, Z: np.ndarray) -> np.ndarray:
        if self.name == "wasserstein":
            return Z
        return self.be.coords_to_spd(Z, self.m)

    def dist_sq(self, Z1: np.ndarray, Z2: np.ndarray) -> np.ndarray:
        """Squared distances between encoded responses (row-wise, broadcasting)."""
        if self.name == "wasserstein":
            return np.mean((Z1 - Z2) ** 2, axis=-1)
        return np.sum((Z1 - Z2) ** 2, axis=-1)


def check_X(X, p: int | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError("X must be 2-D (n_samples, n_features)")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values")
    if p is not None and X.shape[1] != p:
        raise ValueError(f"X has {X.shape[1]} features, expected {p}")
    return np.ascontiguousarray(X)


def kfold_indices(n: int, k: int, random_state) -> list[np.ndarray]:
    """Shuffled K-fold split (deterministic given ``random_state``)."""
    if k < 2 or k > n:
        raise ValueError(f"cv must be between 2 and n_samples={n}")
    perm = np.random.default_rng(random_state).permutation(n)
    return [np.sort(f) for f in np.array_split(perm, k)]


def resolve_backend(name: str):
    return get_backend(name)
