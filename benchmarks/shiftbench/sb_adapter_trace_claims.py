# ruff: noqa: E501, E731, B007, B023, F841
"""TRACE adapter for the TRACE-claims runs (TRACE_ADAPTER_SPEC.md).

Subclasses the frozen TraceMethod contract and keeps the whole result in `extra`.
`split` is the A-reference split (Amendment 1, D4: same estimand as the frozen truth);
the order-free split is kept in extra["split_of"].
"""

from __future__ import annotations

import numpy as np
from sb_adapter import Decomposition, TraceMethod

CFG = dict(K=4, folds=5, trim=0.99)  # Amendment 1, D5


def _tolist(v):
    if isinstance(v, dict):
        return {k: _tolist(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return [_tolist(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


KEEP = (
    "split",
    "split_aref",
    "split_anchor_B",
    "lsplit",
    "lsplit_aref",
    "lmom",
    "bary_a",
    "bary_dc",
    "bary_phi",
    "marg_phi",
    "resid_phi",
    "w2",
    "trim_mass",
    "interaction",
    "config",
    "intercept",
)


class TraceFull(TraceMethod):
    name = "TRACE"

    def __init__(self, return_model=False):
        self.return_model = return_model

    def raw(self, XA, yA, XB, yB, seed=0):
        import traceot

        return traceot.shift_decompose(
            XA, yA, XB, yB, seed=seed, return_model=self.return_model, **CFG
        )

    def decompose(self, XA, yA, XB, yB, seed=0, r=None):
        if r is None:
            r = self.raw(XA, yA, XB, yB, seed)
        extra = {k: _tolist(r[k]) for k in KEEP if k in r}
        extra["split_of"] = extra.pop("split")
        if "model" in r:
            extra["_model"] = r["model"]
        return Decomposition(
            split={k: (float(a), float(b)) for k, (a, b) in r["split_aref"].items()},
            feat_comp=np.asarray(r["feat_comp"], float),
            feat_struct=np.asarray(r["feat_struct"], float),
            intercept=float(r["intercept"]),
            extra=extra,
        )
