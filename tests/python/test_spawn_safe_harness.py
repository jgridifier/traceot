"""Regression test: a spawn pool completes after a pre-flight TRACE smoke.

The V5.2b abort (docs/benchmark/V5_2B_ABORT.md) was a deadlock: the
pre-flight TRACE smoke ran in the parent, initialized Rayon, and a
forked Pool hung. This test runs the spawn-safe sequence on tiny
non-gate seeds (96, 95, 94) and asserts completion within a timeout.

The sequence itself runs in a freshly spawned process so other tests
in the same pytest session, which may have imported TRACE, cannot
contaminate the parent. This test does not touch any study seed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SYS = ROOT / "benchmarks" / "shiftbench"
if str(SYS) not in sys.path:
    sys.path.insert(0, str(SYS))

import spawn_safe  # noqa: E402

# Generous relative to a local run of ~0.3 s; CI is a cold start.
TIMEOUT = spawn_safe.DEFAULT_REGRESSION_TIMEOUT


def test_regression_seeds_are_non_gate_and_not_sealed():
    seeds = [spawn_safe.REGRESSION_SMOKE_SEED, *spawn_safe.REGRESSION_POOL_SEEDS]
    assert seeds == [96, 95, 94]
    for s in seeds:
        assert 0 < s < 1000
        assert s not in range(9600, 9620)
        assert s not in range(10400, 10800)
        assert s not in range(11000, 11050)
        assert s not in range(11100, 11120)
        assert s not in range(12000, 12050)
        assert s not in range(12100, 12120)
        assert s not in range(13000, 13050)
        assert s not in range(13100, 13120)


def test_spawn_context_is_spawn():
    ctx = spawn_safe.spawn_context()
    assert ctx.get_start_method() == "spawn"


def test_spawn_pool_completes_after_preflight_within_timeout(tmp_path):
    """Run the pool after a pre-flight TRACE smoke; assert completion in time.

    Mirrors the V5.2c protocol: pre-flight in a spawned child, then a
    spawn pool from a parent that never called TRACE. A hang of the
    V5.2b kind fails this test by timeout rather than completing.
    """
    out = tmp_path / "spawn_safe_regression.json"
    ctx = spawn_safe.spawn_context()
    proc = ctx.Process(
        target=spawn_safe.write_regression_sequence,
        args=(str(out), TIMEOUT),
    )
    proc.start()
    proc.join(timeout=TIMEOUT + 15)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
        pytest.fail(
            f"regression process still alive after {TIMEOUT + 15}s; "
            "spawn-safe pool did not complete (possible deadlock)"
        )
    assert proc.exitcode == 0, (
        f"regression process exited {proc.exitcode}; output exists={out.exists()}"
    )
    assert out.exists()
    rec = json.loads(out.read_text())
    assert rec["completed"] is True
    assert rec["smoke_ok"] is True
    assert rec["pool_ok"] is True
    assert rec["pool_n"] == 2
    assert rec["parent_imported_traceot"] is False
    assert rec["seeds"] == [96, 95, 94]
    assert rec["elapsed_sec"] < TIMEOUT
