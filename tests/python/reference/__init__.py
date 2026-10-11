"""Pure-numpy reference implementations of every numerical routine in traceot.

These are written for clarity, not speed. The Rust core (``traceot._core``) must
agree with them to a tight tolerance; ``tests/python/test_parity.py`` checks this.
Function names and signatures here are the contract the Rust bindings follow.
"""

from .additive import additive_euclid_fit, additive_euclid_predict
from .adopt import adopt_w2_component, adopt_w2_fit, adopt_w2_predict
from .logchol import coords_to_spd, spd_to_coords
from .smoother import global_frechet_weights, kernel_values, local_linear_weights
from .w2 import (
    collapse_knots,
    pava,
    pava_rows,
    quantile_grid,
    transport_apply,
    w2_sq,
)

__all__ = [
    "additive_euclid_fit",
    "additive_euclid_predict",
    "adopt_w2_component",
    "adopt_w2_fit",
    "adopt_w2_predict",
    "collapse_knots",
    "coords_to_spd",
    "global_frechet_weights",
    "kernel_values",
    "local_linear_weights",
    "pava",
    "pava_rows",
    "quantile_grid",
    "spd_to_coords",
    "transport_apply",
    "w2_sq",
]
