"""TRACE v4 adapter for ShiftBench (TRACE_V4).

Calls `traceot.explain_v4` and maps the mean-channel split into the frozen
`Decomposition` shape. Extra mixing pieces live in `Decomposition.extra`.
"""

from __future__ import annotations

import numpy as np

from sb_adapter import Decomposition, ShiftMethod


class TraceV4Method(ShiftMethod):
    """TRACE v4 mixing-law decomposition."""

    name = "TRACE_V4"
    stats = ("mean",)
    per_feature = True

    def __init__(self, k_channels: int = 6, n_boot: int = 2000):
        self.k_channels = k_channels
        self.n_boot = n_boot

    def decompose(self, XA, yA, XB, yB, seed: int = 0) -> Decomposition:
        import traceot

        fn = getattr(traceot, "explain_v4", None)
        if fn is None:
            raise NotImplementedError("traceot.explain_v4 not available yet")
        r = fn(
            XA,
            yA,
            XB,
            yB,
            seed=int(seed),
            k_channels=self.k_channels,
            n_boot=self.n_boot,
        )
        split_raw = r.get("split") or {}
        split = {}
        for k, v in split_raw.items():
            if isinstance(v, (tuple, list)) and len(v) == 2:
                split[k] = (float(v[0]), float(v[1]))
        fc = r.get("feat_comp")
        fs = r.get("feat_struct")
        return Decomposition(
            split=split,
            feat_comp=None if fc is None else np.asarray(fc, float),
            feat_struct=None if fs is None else np.asarray(fs, float),
            intercept=r.get("intercept"),
            extra={
                "feat_mix_comp": r.get("feat_mix_comp"),
                "feat_mix_struct": r.get("feat_mix_struct"),
                "budget": r.get("budget"),
                "budget_se": r.get("budget_se"),
                "budget_ci": (r.get("budget_ci_lo"), r.get("budget_ci_hi")),
                "mean": r.get("mean"),
                "config": r.get("config"),
                "seconds_total": r.get("seconds_total"),
                "interaction_mass": r.get("interaction_mass"),
                "mixing_residual": r.get("mixing_residual"),
                "lsplit": r.get("lsplit"),
            },
        )


# Convenience alias matching the hand-off name
TRACE_V4 = TraceV4Method
