"""TRACE v1: orthonormal Legendre channels, shifts, distribution rows, robust fit.

The numerical work runs in the Rust core. This module checks inputs and forwards
them. Reporting bases are Legendre, Hermite-in-z, or a reference-adapted basis.
"""

from __future__ import annotations

import numpy as np

from . import _core
from ._base import EstimatorMixin


def _check_X(X, p: int | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty 2-D array (n_samples, n_features)")
    if np.any(np.isinf(X)):
        raise ValueError("X contains infinite values; NaN is the missing bin")
    if p is not None and X.shape[1] != p:
        raise ValueError(f"X has {X.shape[1]} features, expected {p}")
    return np.ascontiguousarray(X)


def _check_y(y, n: int | None = None) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y.size == 0 or not np.all(np.isfinite(y)):
        raise ValueError("y must be a non-empty finite vector")
    if n is not None and y.shape[0] != n:
        raise ValueError("X and y have different numbers of rows")
    return np.ascontiguousarray(y)


class TraceV1(EstimatorMixin):
    """Newton pinball fit on Gauss–Legendre nodes in the shifted Legendre basis.

    Parameters
    ----------
    k_fit, k_report :
        Fitting degree and the number of channels in the reporting basis.
        ``k_fit`` defaults to 6. ``k_report`` defaults to 4.
    reporting :
        ``"legendre"``, ``"hermite"`` or ``"reference"``.
    mono :
        ``"pav"`` (default, post-hoc), ``"bernstein"`` or ``"none"``.
    """

    def __init__(
        self,
        k_fit: int = 6,
        k_report: int = 4,
        n_u: int = 0,
        max_bins: int = 16,
        max_rounds: int = 30,
        tol: float = 1e-9,
        min_count: int = 1,
        ridge: float = 1e-6,
        learning_rate: float = 1.0,
        mono: str = "pav",
        bernstein_elevation: int = 4,
        reporting: str = "legendre",
        smooth: float = 0.0,
        levenberg: float = 0.0,
        sobolev_lambda: float = 0.0,
        sobolev_s: float = 1.0,
        tail_gamma: float = 0.0,
    ):
        self.k_fit = k_fit
        self.k_report = k_report
        self.n_u = n_u
        self.max_bins = max_bins
        self.max_rounds = max_rounds
        self.tol = tol
        self.min_count = min_count
        self.ridge = ridge
        self.learning_rate = learning_rate
        self.mono = mono
        self.bernstein_elevation = bernstein_elevation
        self.reporting = reporting
        self.smooth = smooth
        self.levenberg = levenberg
        self.sobolev_lambda = sobolev_lambda
        self.sobolev_s = sobolev_s
        self.tail_gamma = tail_gamma

    def _kwargs(self) -> dict:
        return dict(
            k_fit=int(self.k_fit),
            k_report=int(self.k_report),
            n_u=int(self.n_u),
            max_bins=int(self.max_bins),
            max_rounds=int(self.max_rounds),
            tol=float(self.tol),
            min_count=int(self.min_count),
            ridge=float(self.ridge),
            learning_rate=float(self.learning_rate),
            mono=str(self.mono),
            bernstein_elevation=int(self.bernstein_elevation),
            reporting=str(self.reporting),
            smooth=float(self.smooth),
            levenberg=float(self.levenberg),
            sobolev_lambda=float(self.sobolev_lambda),
            sobolev_s=float(self.sobolev_s),
            tail_gamma=float(self.tail_gamma),
        )

    def fit(self, X, y):
        X = _check_X(X)
        y = _check_y(y, X.shape[0])
        self.model_ = _core.TraceV1Model.fit(X, y, **self._kwargs())
        self.n_features_in_ = X.shape[1]
        return self

    def _check_fitted(self):
        if not hasattr(self, "model_"):
            raise RuntimeError("TraceV1 is not fitted; call fit() first")

    def predict(self, X, u, project: bool = False):
        self._check_fitted()
        X = _check_X(X, self.n_features_in_)
        u = _check_y(u)
        return np.asarray(self.model_.predict(X, u, bool(project)))

    def attribute(self, X) -> dict:
        self._check_fitted()
        X = _check_X(X, self.n_features_in_)
        raw = self.model_.attribute(X)
        return {k: np.asarray(raw[k]) for k in raw}

    def feature_table(self, j: int):
        self._check_fitted()
        return np.asarray(self.model_.feature_table(int(j)))

    def feature_gains(self, X, y, rows):
        """Held-out pinball gains. A feature is kept when either gain is positive."""
        self._check_fitted()
        X = _check_X(X, self.n_features_in_)
        y = _check_y(y, X.shape[0])
        rows = np.ascontiguousarray(np.asarray(rows, dtype=np.int64).reshape(-1))
        raw = self.model_.feature_gains(X, y, rows)
        return {k: np.asarray(raw[k]) for k in raw}

    def channel_energy(self, feature: int, channel: int) -> float:
        self._check_fitted()
        return float(self.model_.channel_energy(int(feature), int(channel)))

    def channel_slope(self, feature: int, channel: int, x) -> float:
        self._check_fitted()
        x = _check_y(x)
        return float(self.model_.channel_slope(int(feature), int(channel), x))

    def shift_conditional(self, other: TraceV1, X) -> dict:
        self._check_fitted()
        other._check_fitted()
        X = _check_X(X, self.n_features_in_)
        raw = self.model_.shift_conditional(other.model_, X)
        return {
            "total": float(raw["total"]),
            "phi_mean": np.asarray(raw["phi_mean"]),
        }

    def shift_marginal(self, other: TraceV1, X_a, X_b, grid: int = 128) -> dict:
        self._check_fitted()
        other._check_fitted()
        X_a = _check_X(X_a, self.n_features_in_)
        X_b = _check_X(X_b, self.n_features_in_)
        raw = self.model_.shift_marginal(other.model_, X_a, X_b, int(grid))
        out = {}
        for key, value in raw.items():
            if key == "overlap_failures":
                out[key] = value
            else:
                out[key] = float(value)
        return out


class FrechetV1(EstimatorMixin):
    """Per-channel weighted least squares on Hosking L-moment targets."""

    def __init__(
        self,
        k_fit: int = 4,
        max_bins: int = 16,
        max_rounds: int = 40,
        tol: float = 1e-12,
        estimate_weights: bool = True,
    ):
        self.k_fit = k_fit
        self.max_bins = max_bins
        self.max_rounds = max_rounds
        self.tol = tol
        self.estimate_weights = estimate_weights

    def fit(self, X, samples, lengths, weights=None):
        X = _check_X(X)
        samples = np.ascontiguousarray(np.asarray(samples, dtype=np.float64))
        if samples.ndim != 2 or samples.shape[0] != X.shape[0]:
            raise ValueError("samples must have shape (n_samples, m_max)")
        lengths = np.ascontiguousarray(np.asarray(lengths, dtype=np.int64).reshape(-1))
        if lengths.shape[0] != X.shape[0]:
            raise ValueError("lengths must have one entry per row")
        kw = dict(
            k_fit=int(self.k_fit),
            max_bins=int(self.max_bins),
            max_rounds=int(self.max_rounds),
            tol=float(self.tol),
            estimate_weights=bool(self.estimate_weights),
        )
        if weights is not None:
            weights = np.ascontiguousarray(np.asarray(weights, dtype=np.float64).reshape(-1))
            if weights.shape[0] != X.shape[0] or not np.all(np.isfinite(weights)):
                raise ValueError("weights must be a finite vector with one entry per row")
            kw["weights"] = weights
            kw["estimate_weights"] = False
        self.model_ = _core.FrechetModel.fit(X, samples, lengths, **kw)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("FrechetV1 is not fitted")
        X = _check_X(X, self.n_features_in_)
        return np.asarray(self.model_.predict(X))


class RobustV1(EstimatorMixin):
    """Subgradient fit of the exact grid Wasserstein dual. The max-jump formula is separate."""

    def __init__(
        self,
        eps: float = 0.0,
        max_bins: int = 8,
        rounds: int = 40,
        step: float = 0.6,
        ball: str = "w1",
        kappa_y: float | None = None,
    ):
        self.eps = eps
        self.max_bins = max_bins
        self.rounds = rounds
        self.step = step
        self.ball = ball
        self.kappa_y = kappa_y

    def fit(self, X, y, kappa=None, levels=None):
        X = _check_X(X)
        y = _check_y(y, X.shape[0])
        kw = dict(
            eps=float(self.eps),
            max_bins=int(self.max_bins),
            rounds=int(self.rounds),
            step=float(self.step),
            ball=str(self.ball),
            kappa_y=None if self.kappa_y is None else float(self.kappa_y),
        )
        if kappa is not None:
            kw["kappa"] = np.ascontiguousarray(np.asarray(kappa, dtype=np.float64).reshape(-1))
        if levels is not None:
            kw["levels"] = np.ascontiguousarray(np.asarray(levels, dtype=np.float64).reshape(-1))
        self.model_ = _core.RobustQuantileModel.fit(X, y, **kw)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("RobustV1 is not fitted")
        return np.asarray(self.model_.predict(_check_X(X, self.n_features_in_)))

    def pinball(self, X, y) -> float:
        if not hasattr(self, "model_"):
            raise RuntimeError("RobustV1 is not fitted")
        X = _check_X(X, self.n_features_in_)
        y = _check_y(y, X.shape[0])
        return float(self.model_.pinball(X, y))


class TraceHybrid(EstimatorMixin):
    """Squared-loss mean channel, then CRPS residual channels.

    The mean booster uses the additive histogram settings of the mean baseline.
    Residual channels are the v1 Newton fit with channel 0 held at zero.
    ``predict(..., project=True)`` is a mean-preserving isotonic map: the
    residual curve is projected, recentred, and the mean is added back.
    """

    def __init__(
        self,
        k_fit: int = 6,
        k_report: int = 4,
        n_u: int = 0,
        max_bins: int = 16,
        max_rounds: int = 40,
        tol: float = 1e-8,
        min_count: int = 1,
        ridge: float = 1e-6,
        learning_rate: float = 1.0,
        mono: str = "none",
        bernstein_elevation: int = 4,
        reporting: str = "legendre",
        smooth: float = 0.0,
        levenberg: float = 0.0,
        sobolev_lambda: float = 0.0,
        sobolev_s: float = 1.0,
        tail_gamma: float = 0.0,
        mean_max_bins: int = 1024,
        mean_learning_rate: float = 0.04,
        mean_max_rounds: int = 50000,
        mean_early_stopping_rounds: int = 100,
        mean_early_stopping_tolerance: float = 1e-5,
        mean_validation_size: float = 0.15,
        mean_min_samples_leaf: int = 4,
    ):
        self.k_fit = k_fit
        self.k_report = k_report
        self.n_u = n_u
        self.max_bins = max_bins
        self.max_rounds = max_rounds
        self.tol = tol
        self.min_count = min_count
        self.ridge = ridge
        self.learning_rate = learning_rate
        self.mono = mono
        self.bernstein_elevation = bernstein_elevation
        self.reporting = reporting
        self.smooth = smooth
        self.levenberg = levenberg
        self.sobolev_lambda = sobolev_lambda
        self.sobolev_s = sobolev_s
        self.tail_gamma = tail_gamma
        self.mean_max_bins = mean_max_bins
        self.mean_learning_rate = mean_learning_rate
        self.mean_max_rounds = mean_max_rounds
        self.mean_early_stopping_rounds = mean_early_stopping_rounds
        self.mean_early_stopping_tolerance = mean_early_stopping_tolerance
        self.mean_validation_size = mean_validation_size
        self.mean_min_samples_leaf = mean_min_samples_leaf

    def _kwargs(self) -> dict:
        return dict(
            k_fit=int(self.k_fit),
            k_report=int(self.k_report),
            n_u=int(self.n_u),
            max_bins=int(self.max_bins),
            max_rounds=int(self.max_rounds),
            tol=float(self.tol),
            min_count=int(self.min_count),
            ridge=float(self.ridge),
            learning_rate=float(self.learning_rate),
            mono=str(self.mono),
            bernstein_elevation=int(self.bernstein_elevation),
            reporting=str(self.reporting),
            smooth=float(self.smooth),
            levenberg=float(self.levenberg),
            sobolev_lambda=float(self.sobolev_lambda),
            sobolev_s=float(self.sobolev_s),
            tail_gamma=float(self.tail_gamma),
            mean_max_bins=int(self.mean_max_bins),
            mean_learning_rate=float(self.mean_learning_rate),
            mean_max_rounds=int(self.mean_max_rounds),
            mean_early_stopping_rounds=int(self.mean_early_stopping_rounds),
            mean_early_stopping_tolerance=float(self.mean_early_stopping_tolerance),
            mean_validation_size=float(self.mean_validation_size),
            mean_min_samples_leaf=int(self.mean_min_samples_leaf),
        )

    def fit(self, X, y):
        X = _check_X(X)
        y = _check_y(y, X.shape[0])
        self.model_ = _core.TraceHybridModel.fit(X, y, **self._kwargs())
        self.n_features_in_ = X.shape[1]
        return self

    def _check_fitted(self):
        if not hasattr(self, "model_"):
            raise RuntimeError("TraceHybrid is not fitted; call fit() first")

    def predict(self, X, u, project: bool = True):
        self._check_fitted()
        X = _check_X(X, self.n_features_in_)
        u = _check_y(u)
        return np.asarray(self.model_.predict(X, u, bool(project)))

    def mean(self, X):
        self._check_fitted()
        X = _check_X(X, self.n_features_in_)
        return np.asarray(self.model_.mean(X))

    def channel_energy(self, feature: int, channel: int) -> float:
        self._check_fitted()
        return float(self.model_.channel_energy(int(feature), int(channel)))
