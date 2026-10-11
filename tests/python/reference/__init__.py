"""Pure-numpy reference implementations of every numerical routine in traceot.

**Test-only.** All runtime numerics live in the Rust core (``crates/core``); the
``traceot`` package never imports this module. These implementations are written for
clarity, not speed, and exist solely to verify the Rust results
(``tests/python/test_parity.py``, ``test_scalable.py``, ``test_estimators.py``).
Function names and signatures match ``traceot.primitives``.
"""

from .additive import additive_euclid_fit, additive_euclid_predict
from .adopt import adopt_w2_component, adopt_w2_fit, adopt_w2_predict, adopt_w2_predict_averaged
from .logchol import coords_to_spd, spd_to_coords
from .scalable import (
    QuantileSketch,
    bin_column,
    bin_edges_from_sorted,
    histogram,
    histograms,
    pinball_grad_location_scale,
)
from .smoother import (
    binned_smooth,
    global_frechet_weights,
    kernel_values,
    local_linear_weights,
    local_linear_weights_weighted,
    reflected_local_linear_weights,
)
from .w2 import (
    collapse_knots,
    pava,
    pava_rows,
    quantile_grid,
    transport_apply,
    w2_sq,
)

__all__ = [
    "QuantileSketch",
    "bin_column",
    "bin_edges_from_sorted",
    "histogram",
    "histograms",
    "pinball_grad_location_scale",
    "additive_euclid_fit",
    "additive_euclid_predict",
    "adopt_w2_component",
    "adopt_w2_fit",
    "adopt_w2_predict",
    "adopt_w2_predict_averaged",
    "binned_smooth",
    "collapse_knots",
    "coords_to_spd",
    "global_frechet_weights",
    "kernel_values",
    "local_linear_weights",
    "local_linear_weights_weighted",
    "reflected_local_linear_weights",
    "pava",
    "pava_rows",
    "quantile_grid",
    "spd_to_coords",
    "transport_apply",
    "w2_sq",
]
