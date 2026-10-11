"""TRACE: a binned additive model of the conditional quantile function.

TRACE is our own unpublished method. It is not ADOPT. Each feature contributes a
location channel (constant in the quantile level), a scale channel (proportional to
the normal quantile) and a shape channel (a higher Hermite direction). The sum is
order-invariant, then projected onto monotone quantile functions.

All fitting, prediction and scoring run in the Rust core.
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


class TraceRegressor(EstimatorMixin):
    """Cyclic histogram booster for conditional quantiles.

    Parameters
    ----------
    n_channels : {2, 3, 4}
        Basis size. 2 is location and scale. 3 adds the tail channel
        ``z^3 - 3z``. 4 adds the even channel ``z^2 - 1`` after the tail, so
        indices 0, 1 and 2 stay location, scale and tail.
    max_bins : int
        Histogram bins per feature, including the missing-value bin. At most 255
        uses ``uint8`` codes.
    learning_rate : float
        Step length of each Jacobi update.
    max_rounds, patience, tol : early stopping on the validation pinball loss.
        ``patience=0`` or ``val_frac=0`` keeps the last round.
    val_frac : float
        Fraction of each bag held out for the pinball loss and feature gains.
    n_bags, bag_frac : bootstrap bags. A feature is kept when it is selected in
        at least ``select_frac`` of the bags that had a validation split.
    gain_threshold : float
        A feature is selected in a bag only when its held-out pinball gain is
        strictly above this value.
    max_interactions, interaction_rounds, screen_rows :
        FAST-style residual-gradient screen, then boosting of the selected pairs.
        ``max_interactions=0`` disables pairs.
    min_count : int
        Bins with fewer training rows are not updated.
    u_levels : int
        Size of the fixed quantile grid (a CRPS approximation via mean pinball).
    seed : int
        SplitMix64 seed for bags, the validation split and the interaction screen.

    Attributes
    ----------
    model_ : fitted ``traceot._core.TraceModel``.
    q0_, u_, order_, selected_, gains_ : copies of the fitted summaries.
    train_loss_, val_loss_ : mean pinball on the report split, after interactions.
    """

    def __init__(
        self,
        n_channels: int = 3,
        max_bins: int = 32,
        learning_rate: float = 1.0,
        max_rounds: int = 40,
        patience: int = 8,
        tol: float = 0.0,
        val_frac: float = 0.2,
        n_bags: int = 1,
        bag_frac: float = 1.0,
        gain_threshold: float = 0.0,
        select_frac: float = 0.7,
        max_interactions: int = 0,
        interaction_rounds: int = 15,
        screen_rows: int = 4096,
        min_count: int = 1,
        u_levels: int = 19,
        seed: int = 0,
        min_rounds: int = 20,
    ):
        self.n_channels = n_channels
        self.max_bins = max_bins
        self.learning_rate = learning_rate
        self.max_rounds = max_rounds
        self.patience = patience
        self.tol = tol
        self.val_frac = val_frac
        self.n_bags = n_bags
        self.bag_frac = bag_frac
        self.gain_threshold = gain_threshold
        self.select_frac = select_frac
        self.max_interactions = max_interactions
        self.interaction_rounds = interaction_rounds
        self.screen_rows = screen_rows
        self.min_count = min_count
        self.u_levels = u_levels
        self.seed = seed
        self.min_rounds = min_rounds

    def _check_fitted(self):
        if not hasattr(self, "model_"):
            raise RuntimeError("TraceRegressor is not fitted; call fit() first")

    def _kwargs(self) -> dict:
        if int(self.seed) < 0:
            raise ValueError("seed must be >= 0")
        return dict(
            n_channels=int(self.n_channels),
            max_bins=int(self.max_bins),
            learning_rate=float(self.learning_rate),
            max_rounds=int(self.max_rounds),
            patience=int(self.patience),
            tol=float(self.tol),
            val_frac=float(self.val_frac),
            n_bags=int(self.n_bags),
            bag_frac=float(self.bag_frac),
            gain_threshold=float(self.gain_threshold),
            select_frac=float(self.select_frac),
            max_interactions=int(self.max_interactions),
            interaction_rounds=int(self.interaction_rounds),
            screen_rows=int(self.screen_rows),
            min_count=int(self.min_count),
            u_levels=int(self.u_levels),
            seed=int(self.seed),
            min_rounds=int(self.min_rounds),
        )

    def _remember(self, p: int):
        m = self.model_
        self.n_features_in_ = p
        self.q0_ = np.asarray(m.q0)
        self.u_ = np.asarray(m.u)
        self.order_ = np.asarray(m.order)
        self.selected_ = np.asarray(m.selected)
        self.gains_ = np.asarray(m.gains)
        self.shape_gains_ = np.asarray(m.shape_gains)
        self.train_loss_ = float(m.train_loss)
        self.val_loss_ = float(m.val_loss)
        return self

    def fit(self, X, y):
        """Fit from a dense ``X`` of shape ``(n, p)``. Column order does not matter."""
        X = _check_X(X)
        y = _check_y(y, X.shape[0])
        self.model_ = _core.TraceModel.fit(X, y, **self._kwargs())
        return self._remember(X.shape[1])

    def fit_codes(self, codes, edges, y):
        """Fit from columnar codes of shape ``(p, n)`` and one edge vector per feature.

        Unlike :meth:`fit`, columns are accumulated in index order. Pass the same
        edges to every period when comparing coefficients.
        """
        codes = np.asarray(codes)
        if codes.ndim != 2:
            raise ValueError("codes must have shape (n_features, n_samples)")
        y = _check_y(y, codes.shape[1])
        edge_list = [np.ascontiguousarray(np.asarray(e, dtype=np.float64)) for e in edges]
        if len(edge_list) != codes.shape[0]:
            raise ValueError("edges must contain one vector per feature")
        codes = np.ascontiguousarray(codes)
        kw = self._kwargs()
        if codes.dtype == np.uint8:
            self.model_ = _core.TraceModel.fit_codes_u8(codes, edge_list, y, **kw)
        elif codes.dtype == np.uint16:
            self.model_ = _core.TraceModel.fit_codes_u16(codes, edge_list, y, **kw)
        else:
            raise ValueError("codes must be uint8 or uint16")
        return self._remember(codes.shape[0])

    def predict(self, X, project: bool = True, calibrated: bool = False) -> np.ndarray:
        """Quantile functions of shape ``(n, u_levels)``.

        ``project=True`` applies isotonic regression in the quantile level.
        ``calibrated=True`` adds split-conformal offsets from :meth:`calibrate`
        before that projection.
        """
        self._check_fitted()
        X = _check_X(X, self.n_features_in_)
        if calibrated:
            return self.model_.predict_calibrated(X, project)
        if project:
            return self.model_.predict(X)
        return self.model_.predict_raw(X)

    def calibrate(self, X, y):
        """Fit per-level split-conformal offsets on a held-out sample.

        Residuals are ``y - Q(u|x)`` on the projected raw curve. The offset at
        each level is the conformal order statistic ``ceil((n+1) u) - 1``.
        """
        self._check_fitted()
        X = _check_X(X, self.n_features_in_)
        y = _check_y(y, X.shape[0])
        self.model_.calibrate(X, y)
        return self

    def score(self, X, y) -> dict:
        """Pinball, CRPS approximation, coverage and squared errors on ``(X, y)``.

        When :meth:`calibrate` has been called, calibrated coverage and pinball
        are included under the ``_calibrated`` suffix.
        """
        self._check_fitted()
        X = _check_X(X, self.n_features_in_)
        y = _check_y(y, X.shape[0])
        out = dict(self.model_.score(X, y))
        if np.asarray(self.model_.calibration).size:
            extra = dict(self.model_.score_calibrated(X, y))
            for key, value in extra.items():
                out[f"{key}_calibrated"] = value
        return out

    def effect(self, j: int):
        """``(coef, std)`` for feature ``j``, each of shape ``(n_bins, n_channels)``.

        Bin 0 is missing values. ``std`` is the bag standard deviation before a
        dropped feature's coefficients are zeroed.
        """
        self._check_fitted()
        coef, std = self.model_.effect(int(j))
        return np.asarray(coef), np.asarray(std)

    def edges(self, j: int) -> np.ndarray:
        """Cut points for feature ``j``. Codes are ``1 + #{edges <= x}``."""
        self._check_fitted()
        return np.asarray(self.model_.edges(int(j)))

    def pairs(self):
        """Selected interactions: dicts with ``left``, ``right``, ``coef`` and ``score``."""
        self._check_fitted()
        return self.model_.pairs()

    def permutation_importance(self, X, y, repeats: int = 5, seed: int = 0) -> np.ndarray:
        """Mean pinball increase when each column's bin codes are shuffled."""
        self._check_fitted()
        X = _check_X(X, self.n_features_in_)
        y = _check_y(y, X.shape[0])
        if int(seed) < 0:
            raise ValueError("seed must be >= 0")
        return self.model_.permutation_importance(X, y, int(repeats), int(seed))
