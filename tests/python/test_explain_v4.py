"""Smoke tests for traceot.explain_v4."""

from __future__ import annotations

import numpy as np
import pytest

traceot = pytest.importorskip("traceot")


@pytest.mark.skipif(
    not hasattr(traceot, "explain_v4"), reason="explain_v4 binding missing"
)
def test_explain_v4_keys_and_shapes():
    rng = np.random.default_rng(0)
    n, p = 60, 3
    xa = rng.normal(size=(n, p))
    xb = xa + 0.05
    ya = xa[:, 0] + 0.3 * xa[:, 1]
    yb = ya + 0.8
    r = traceot.explain_v4(xa, ya, xb, yb, k_channels=3, n_boot=50, seed=1, max_bins=8)
    assert "split" in r and "mean" in r["split"]
    assert r["feat_comp"].shape == (p,)
    assert r["feat_struct"].shape == (p,)
    assert np.isfinite(r["budget"])
    assert r["budget_ci_lo"] <= r["budget_ci_hi"]


@pytest.mark.skipif(
    not hasattr(traceot, "explain_v4"), reason="explain_v4 binding missing"
)
def test_trace_v4_adapter():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "shiftbench"))
    from sb_adapter_trace_v4 import TraceV4Method

    rng = np.random.default_rng(2)
    n, p = 40, 2
    xa = rng.normal(size=(n, p))
    xb = xa.copy()
    ya = xa.sum(1)
    yb = ya + 1.0
    m = TraceV4Method(k_channels=3, n_boot=40)
    d = m.decompose(xa, ya, xb, yb, seed=3)
    assert "mean" in d.split
    assert d.feat_comp is not None and d.feat_comp.shape == (p,)
    assert "budget" in d.extra
