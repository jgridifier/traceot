"""Spawn-safe process isolation for TRACE gate runners.

Rayon, once initialized in a process, is not safe to fork. The V5.2b
specified-family pass deadlocked because the pre-flight TRACE smoke ran in
the parent process and ``multiprocessing.Pool`` then used the default
``'fork'`` start method (see ``docs/benchmark/V5_2B_ABORT.md``).

Rules this module encodes:

* Any TRACE call (``import traceot`` or a fit) happens in a *spawned*
  child or in a *spawned* pool worker, never in the process that would
  create a pool by forking.
* Pools are created with multiprocessing's ``'spawn'`` start method from
  a parent that has never called TRACE.

This module does not import TRACE at module level, and importing it does
not initialize Rayon. It is not a study runner and touches no gate seed.
"""

from __future__ import annotations

import os
import pickle
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# Throwaway non-gate seeds for the regression sequence only. Not study indices.
# 96, 95, 94 sit below 1000; they are not in 9600-9619 or any spent range.
REGRESSION_SMOKE_SEED = 96
REGRESSION_POOL_SEEDS = (95, 94)
REGRESSION_N = 200
REGRESSION_P = 5
DEFAULT_REGRESSION_TIMEOUT = 90.0


def spawn_context():
    """Return a multiprocessing context whose start method is 'spawn'."""
    return get_context("spawn")


def parent_has_called_trace() -> bool:
    """True if this process has already imported the TRACE package."""
    return "traceot" in sys.modules


def assert_parent_has_not_called_trace() -> None:
    if parent_has_called_trace():
        raise RuntimeError(
            "this process has already imported TRACE; a forked pool from here "
            "would be unsafe, and a spawn-safe parent must never call TRACE"
        )


def _child_call(module_name: str, fn_name: str, args_path: str, out_path: str) -> None:
    """Entry point for a spawned child. Must stay at module level."""
    import importlib

    mod = importlib.import_module(module_name)
    fn = getattr(mod, fn_name)
    args, kwargs = pickle.loads(Path(args_path).read_bytes())
    try:
        result = fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001 — child reports, then exits
        Path(out_path).write_bytes(
            pickle.dumps(
                {
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                }
            )
        )
        raise SystemExit(1) from e
    Path(out_path).write_bytes(pickle.dumps({"ok": True, "result": result}))


def run_in_spawned_child(fn: Callable, *args, timeout: float | None = None, **kwargs):
    """Run ``fn(*args, **kwargs)`` in a freshly spawned child process.

    ``fn`` must be a top-level function of an importable module (spawn
    re-imports it in the child). The calling process is not required to
    stay TRACE-free, but the V5.2c protocol requires that of the gate
    parent; ``assert_parent_has_not_called_trace`` enforces that when
    asked. The child's TRACE imports die with the child.
    """
    ctx = spawn_context()
    tmp = Path(tempfile.mkdtemp(prefix="spawn_safe_"))
    args_path = tmp / "args.pkl"
    out_path = tmp / "out.pkl"
    args_path.write_bytes(pickle.dumps((args, kwargs)))
    proc = ctx.Process(
        target=_child_call,
        args=(fn.__module__, fn.__name__, str(args_path), str(out_path)),
    )
    try:
        proc.start()
        proc.join(timeout=timeout)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
            raise TimeoutError(f"spawned child {fn.__name__} exceeded {timeout}s")
        if not out_path.exists():
            raise RuntimeError(
                f"spawned child {fn.__name__} exited {proc.exitcode} with no result file"
            )
        payload = pickle.loads(out_path.read_bytes())
        if not payload["ok"]:
            raise RuntimeError(payload["error"])
        return payload["result"]
    finally:
        for p in (args_path, out_path):
            try:
                p.unlink()
            except OSError:
                pass
        try:
            tmp.rmdir()
        except OSError:
            pass


def spawn_pool(workers: int, **kwargs):
    """Create a pool with the 'spawn' start method.

    The calling process should not have imported TRACE (see
    ``assert_parent_has_not_called_trace``). Workers are fresh interpreters
    and may import TRACE themselves.
    """
    return spawn_context().Pool(workers, **kwargs)


def tiny_trace_fit(
    seed: int = REGRESSION_SMOKE_SEED,
    n: int = REGRESSION_N,
    p: int = REGRESSION_P,
) -> dict:
    """One small TRACE specified-family fit. Used only in spawned children.

    Random Gaussian data, not a study simulator, not a gate seed. Seed
    arguments must be non-gate (the regression sequence uses 96, 95, 94).
    """
    import numpy as np

    import traceot

    if not hasattr(traceot, "explain_v5_2_specified"):
        raise RuntimeError("traceot.explain_v5_2_specified is missing")
    rng = np.random.default_rng(seed)
    xa = rng.normal(size=(n, p))
    ya = xa[:, 0] + rng.normal(size=n) * 0.1
    xb = rng.normal(size=(n, p))
    yb = xb[:, 0] + rng.normal(size=n) * 0.1
    result = traceot.explain_v5_2_specified(xa, ya, xb, yb)
    if "lsplit_comp" not in result or "lsplit_struct" not in result:
        raise RuntimeError("tiny TRACE fit did not return lsplit fields")
    return {"ok": True, "seed": seed, "n": n, "p": p}


def run_regression_sequence(timeout: float = DEFAULT_REGRESSION_TIMEOUT) -> dict:
    """The sequence a gate runner must follow, on tiny non-gate seeds.

    1. Assert this process has not imported TRACE.
    2. Run a TRACE smoke fit in a spawned child (seed 96).
    3. After the child exits, assert this process still has not imported TRACE.
    4. Create a 2-worker spawn pool and run two tiny TRACE fits (seeds 95, 94).
    5. Assert the pool completes within ``timeout`` and the parent still has
       not imported TRACE.

    This is a harness test, not a study result. It does not touch any gate
    index, including 13000-13119, 12000-12119, 11000-11119, 10400-10799, or
    the sealed range 9600-9619.
    """
    t0 = time.perf_counter()
    assert_parent_has_not_called_trace()
    smoke = run_in_spawned_child(tiny_trace_fit, REGRESSION_SMOKE_SEED, timeout=timeout)
    assert_parent_has_not_called_trace()
    if not smoke.get("ok"):
        raise RuntimeError(f"smoke fit was not ok: {smoke}")
    pool = spawn_pool(2)
    try:
        async_result = pool.map_async(tiny_trace_fit, list(REGRESSION_POOL_SEEDS))
        records = async_result.get(timeout=timeout)
    finally:
        pool.terminate()
        pool.join()
    elapsed = time.perf_counter() - t0
    assert_parent_has_not_called_trace()
    if len(records) != len(REGRESSION_POOL_SEEDS):
        raise RuntimeError(f"pool returned {len(records)} records, expected 2")
    if not all(r.get("ok") for r in records):
        raise RuntimeError(f"pool fit was not ok: {records}")
    return {
        "completed": True,
        "smoke_ok": True,
        "pool_n": len(records),
        "pool_ok": True,
        "elapsed_sec": elapsed,
        "parent_imported_traceot": parent_has_called_trace(),
        "seeds": [REGRESSION_SMOKE_SEED, *REGRESSION_POOL_SEEDS],
        "timeout": timeout,
    }


def write_regression_sequence(out_path: str, timeout: float = DEFAULT_REGRESSION_TIMEOUT) -> None:
    """Module-level entry for a spawned test process. Writes a JSON record."""
    import json

    rec = run_regression_sequence(timeout=timeout)
    Path(out_path).write_text(json.dumps(rec, indent=2) + "\n")
    if not rec.get("completed") or rec.get("parent_imported_traceot"):
        raise SystemExit(1)
