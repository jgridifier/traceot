"""Scikit-learn-style ADOPT estimator (thin wrapper over the Rust core).

ADOPT is "Additive Optimal Transport Regression" by Song & Müller, Proceedings of
AISTATS 2026, PMLR v300:1558-1566, https://proceedings.mlr.press/v300/song26c.html .
This is an independent re-implementation from the paper (no public code exists);
see docs/REPRODUCTION_NOTES.md for every interpretive choice.
"""

from __future__ import annotations

from ._base import AdditiveFitMixin


class ADOPTRegressor(AdditiveFitMixin):
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
    random_state : int
        Seed for the CV fold assignment (SplitMix64 shuffle in the Rust core).

    Attributes
    ----------
    mean_ : Fréchet mean of the training responses (encoded).
    bandwidths_ : (p,) bandwidths used in the final fit.
    n_iter_, converged_ : backfitting diagnostics.
    cv_results_ : dict with the grid and mean CV error per candidate (if ``"cv"``).
    model_ : the fitted ``traceot._core.AdditiveModel``.
    """

    _method = "adopt"

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
        self.random_state = random_state
