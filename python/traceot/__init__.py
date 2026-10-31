"""traceot: interpretable distributional modelling for large tabular data.

**Status: research, unvalidated.**

All numerical methods live in the Rust core (``traceot._core``); this package is a
thin binding layer (input validation, numpy conversion, scikit-learn-style API).

Milestone 1 implements ADOPT (Song & Müller, "ADOPT: Additive Optimal Transport
Regression", AISTATS 2026, PMLR v300, https://proceedings.mlr.press/v300/song26c.html).
TRACE v0, our own unpublished method, is a binned additive model of the conditional
quantile function (:class:`traceot.TraceRegressor`). TRACE v1
(:class:`traceot.TraceV1`) fits orthonormal Legendre channels with Newton steps.
Shift v2 (:mod:`traceot.shift`) splits a change between two periods into
composition and structure.
TRACE v2 (:class:`traceot.TraceHybrid`) fits the mean by squared loss and the
residual channels by CRPS. The pre-written bar was missed, including a second
attempt that writes one two-leaf split per visit, so it is not the default.
:class:`traceot.TraceRegressor` stays the v0 path and
:class:`traceot.TraceV1` stays the Newton path.
ADOPT remains a reproduction only.
"""

from . import checks, primitives, shift
from .adopt import ADOPTRegressor
from .baselines import FrechetMeanRegressor, GlobalFrechetRegressor, L2AdditiveQuantileRegressor
from .primitives import quantile_grid
from .shift import shift_decompose
from .trace import TraceRegressor
from .trace_v1 import FrechetV1, RobustV1, TraceHybrid, TraceV1
from .v4 import explain_v4
from .v5 import explain_v5, explain_v5_1, mixture_cdf_location, mixture_cdf_rows, mixture_quantile

__version__ = "0.1.0"

__all__ = [
    "ADOPTRegressor",
    "FrechetMeanRegressor",
    "GlobalFrechetRegressor",
    "L2AdditiveQuantileRegressor",
    "TraceRegressor",
    "TraceV1",
    "TraceHybrid",
    "FrechetV1",
    "RobustV1",
    "checks",
    "primitives",
    "quantile_grid",
    "shift",
    "shift_decompose",
    "explain_v4",
    "explain_v5",
    "explain_v5_1",
    "mixture_cdf_location",
    "mixture_cdf_rows",
    "mixture_quantile",
]
