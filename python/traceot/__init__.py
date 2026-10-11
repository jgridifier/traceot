"""traceot: glass-box optimal-transport regression.

**Status: research, unvalidated.**

Milestone 1 implements ADOPT (Song & Müller, "ADOPT: Additive Optimal Transport
Regression", AISTATS 2026, PMLR v300, https://proceedings.mlr.press/v300/song26c.html)
with a Rust core and a pure-numpy reference. TRACE, our own proposed (unpublished)
method, is the long-term goal and is not implemented yet.
"""

from ._backend import HAVE_RUST, get_backend
from .adopt import ADOPTRegressor
from .baselines import FrechetMeanRegressor, GlobalFrechetRegressor, L2AdditiveQuantileRegressor

__version__ = "0.1.0"

__all__ = [
    "ADOPTRegressor",
    "FrechetMeanRegressor",
    "GlobalFrechetRegressor",
    "HAVE_RUST",
    "L2AdditiveQuantileRegressor",
    "get_backend",
]
