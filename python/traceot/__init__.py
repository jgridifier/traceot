"""traceot: interpretable distributional modelling for large tabular data.

**Status: research, unvalidated.**

All numerical methods live in the Rust core (``traceot._core``); this package is a
thin binding layer (input validation, numpy conversion, scikit-learn-style API).

Milestone 1 implements ADOPT (Song & Müller, "ADOPT: Additive Optimal Transport
Regression", AISTATS 2026, PMLR v300, https://proceedings.mlr.press/v300/song26c.html).
TRACE, our own proposed (unpublished) method, is the long-term goal; its scalable
primitives are in :mod:`traceot.primitives`.
"""

from . import primitives
from .adopt import ADOPTRegressor
from .baselines import FrechetMeanRegressor, GlobalFrechetRegressor, L2AdditiveQuantileRegressor
from .primitives import quantile_grid

__version__ = "0.1.0"

__all__ = [
    "ADOPTRegressor",
    "FrechetMeanRegressor",
    "GlobalFrechetRegressor",
    "L2AdditiveQuantileRegressor",
    "primitives",
    "quantile_grid",
]
