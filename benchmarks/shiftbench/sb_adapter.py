"""The method interface every ShiftBench contestant implements (rivals and TRACE).

A method receives two periods (X_A, y_A), (X_B, y_B) with features already centred
at the pooled mean, and returns a Decomposition. Anything a method cannot produce is
None and is reported as "n/a", never imputed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Decomposition:
    # stat name -> (composition, structure); stats are those in sb_dgp.STATS
    split: dict[str, tuple[float, float]]
    # mean-channel per-feature split, length p each, or None
    feat_comp: np.ndarray | None = None
    feat_struct: np.ndarray | None = None
    intercept: float | None = None
    extra: dict = field(default_factory=dict)


class ShiftMethod:
    """Subclass and implement decompose(). Must be deterministic given seed."""

    name = "base"
    stats: tuple[str, ...] = ()  # stats the method supports
    per_feature = False

    def decompose(self, XA, yA, XB, yB, seed: int = 0) -> Decomposition:
        raise NotImplementedError


class TraceMethod(ShiftMethod):
    """Slot for TRACE shift decomposition v2 (branch shift-v2, not yet merged).

    Contract for the TRACE side, via the traceot Python bindings only:
        traceot.shift_decompose(XA, yA, XB, yB, seed=...) -> mapping with keys
          "split":   {stat: (composition, structure)} for any of mean, sd, q10, q50, q90
          "feat_comp", "feat_struct": per-feature mean-channel arrays (optional)
          "intercept": float (optional)
    Until that function exists this adapter raises, and the harness records TRACE as
    "pending". Wiring it in needs only this class.
    """

    name = "TRACE"
    stats = ("mean", "sd", "q10", "q50", "q90")
    per_feature = True

    def decompose(self, XA, yA, XB, yB, seed: int = 0) -> Decomposition:
        import traceot

        fn = getattr(traceot, "shift_decompose", None)
        if fn is None:
            raise NotImplementedError("traceot.shift_decompose not available yet")
        r = fn(XA, yA, XB, yB, seed=seed)
        fc, fs = r.get("feat_comp"), r.get("feat_struct")
        return Decomposition(
            split={k: (float(a), float(b)) for k, (a, b) in r["split"].items()},
            feat_comp=None if fc is None else np.asarray(fc, float),
            feat_struct=None if fs is None else np.asarray(fs, float),
            intercept=r.get("intercept"),
        )
