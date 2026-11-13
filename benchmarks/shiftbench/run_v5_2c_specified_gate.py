"""TRACE v5.2c specified-family replication gate (PREREG_V5_2C_SPECIFIED_REPLICATION.md).

    python run_v5_2c_specified_gate.py [--workers 8]

Replication of the V5.2 specified-family gate: same hypothesis, method, simulator,
rivals, Holm / BEST / NON-INFERIOR rule, δ, and 30-minute cap. The V5.2b
pre-flight rule is incorporated by reference and is executed in a spawned
child. Fresh seeds: skew_comp / tail_comp at n=2000, p=5 (indices 13000-13049)
and n=20000, p=20 (indices 13100-13119), kappa=1, with paired DFL / CFM rivals.

A mandatory rehearsal of the full parallel pipeline (all three methods, same
task function, spawn-safe pool, watchdog) runs on throwaway non-gate seeds
98 and 97, two per cell (8 tasks), before any gate seed is touched. Rehearsal
outputs go to a separate file and never enter scoring. Any rehearsal failure
aborts without consuming gate seeds.

The gate pool and the rehearsal pool use multiprocessing's 'spawn' start
method, from a parent that never calls TRACE. A 5-minute no-completion
watchdog aborts and records ABORTED (not FAIL).

This module is the runner: invoking it executes the study; importing it does
not. This file's presence in a PR is not a run.

Sealed indices 9600-9619 are never run. Spent coverage indices 10400-10799,
spent specified-family indices 11000-11049 / 11100-11119, and spent V5.2b
indices 12000-12049 / 12100-12119 are not re-run. A scenario whose sigma_B
calibration is infeasible has no tasks and its cells are INCOMPLETE; H is
then not decided and gate seeds are not touched. Tasks not finished inside
the cap are not imputed.

The method under test receives raw (uncentred) draws, as in
PREREG_V5_2_SPECIFIED §2 (by reference). Rivals receive the same centred
arrays as in the V5.1 / V5.2 gates.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import sb_dgp_shape_v5_2_specified as sp
import spawn_safe
from run_synthetic import RES, centred, seed_of

# Freeze point: the merge commit that added PREREG_V5_2C_SPECIFIED_REPLICATION.md
# to main (PR #64).
PREREG_SHA = "f022ba3e42ebe4b7cee1dbdf2bf1c254b248b1f6"
IMPL_SHA = "1aab4778abf8f84c8c8f7d9603eb8c322f1ba3c9"
SCEN = ("skew_comp", "tail_comp")
KAPPA = 1.0
SIZES = {2000: (5, list(range(13000, 13050))), 20000: (20, list(range(13100, 13120)))}
ALLOWED = set(range(13000, 13050)) | set(range(13100, 13120))
FORBIDDEN = (
    set(range(0, 2000))
    | set(range(2101, 2120))
    | set(range(3101, 3121))
    | set(range(5000, 6500))
    | set(range(7007, 7013))
    | set(range(7900, 7910))
    | set(range(8000, 8500))
    | set(range(9100, 9150))
    | set(range(9200, 9220))
    | set(range(9300, 9380))
    | set(range(9400, 9430))
    | set(range(9500, 9550))
    | set(range(9600, 9620))  # sealed
    | set(range(9700, 9750))
    | set(range(9800, 9820))
    | set(range(10100, 10150))
    | set(range(10200, 10220))
    | set(range(10220, 10400))  # buffer after v5.2
    | set(range(10400, 10800))  # spent V5.1 coverage
    | set(range(10800, 11000))
    | set(range(11000, 11050))  # spent V5.2 specified n=2000
    | set(range(11100, 11120))  # spent V5.2 specified n=20000
    | set(range(11120, 12000))  # buffer after spent specified
    | set(range(12000, 12050))  # spent V5.2b n=2000
    | set(range(12050, 12100))  # gap between V5.2b blocks
    | set(range(12100, 12120))  # spent V5.2b n=20000
    | set(range(12120, 13000))  # buffer after spent V5.2b
    | set(range(13050, 13100))  # gap between V5.2c blocks
    | set(range(81001, 81021))
    | set(range(83001, 83021))
    | set(range(84001, 84051))
)
SEALED = set(range(9600, 9620))
COVERAGE_SPENT = set(range(10400, 10800))
SPECIFIED_SPENT = set(range(11000, 11050)) | set(range(11100, 11120))
V52B_SPENT = set(range(12000, 12050)) | set(range(12100, 12120))
CAP_SEC = 30 * 60
WATCHDOG_SEC = 5 * 60
# Rehearsal has no wall cap of its own; only the watchdog binds. A large
# number so min(cap, watchdog) equals the watchdog.
REHEARSAL_CAP_SEC = 24 * 60 * 60
SMOKE_SEED = 99
REHEARSAL_SEEDS = (98, 97)
OUT = RES / "v5_2c_specified_gate_shape.jsonl"
REHEARSAL_OUT = RES / "v5_2c_specified_gate_rehearsal.jsonl"
WALL = RES / "v5_2c_specified_gate_wall.json"
PREFLIGHT_OUT = RES / "v5_2c_specified_gate_preflight.json"
PREFLIGHT_TIMEOUT = 5 * 60


def assert_rep(rep: int, n: int) -> None:
    assert rep in ALLOWED, f"index {rep} is not in 13000-13049 or 13100-13119"
    assert rep in SIZES[n][1], f"index {rep} is not in the range for n={n}"
    assert rep not in FORBIDDEN, f"index {rep} is in the FORBIDDEN set"
    assert rep not in SEALED, f"index {rep} is in the sealed range 9600-9619"
    assert rep not in COVERAGE_SPENT, f"index {rep} is a spent V5.1 coverage index"
    assert rep not in SPECIFIED_SPENT, f"index {rep} is a spent V5.2 specified index"
    assert rep not in V52B_SPENT, f"index {rep} is a spent V5.2b index"


def assert_rehearsal_rep(rep: int) -> None:
    assert rep in REHEARSAL_SEEDS, f"rehearsal index {rep} is not 98 or 97"
    assert rep not in ALLOWED
    assert rep not in SEALED
    assert rep not in COVERAGE_SPENT
    assert rep not in SPECIFIED_SPENT
    assert rep not in V52B_SPENT


def _run_one(scen: str, kappa: float, n: int, rep: int, *, gate: bool) -> dict:
    """One TRACE + DFL + CFM fit. Used by gate and rehearsal workers."""
    from sb_shape_readout import CFM_L, DFL_L

    import traceot

    p = SIZES[n][0]
    if gate:
        assert_rep(rep, n)
    else:
        assert_rehearsal_rep(rep)
        assert n in SIZES
    s = sp.make(scen, kappa, p)
    skey = sp.key(scen, kappa)
    rng = np.random.default_rng(seed_of("data", skey, n, p, rep))
    xa0, ya, xb0, yb = s.sample(n, rng)
    # Method under test: raw draws (PREREG_V5_2_SPECIFIED §2, by reference).
    seed = seed_of("m", rep)
    out = {
        "scen": scen,
        "kappa": kappa,
        "n": n,
        "p": p,
        "rep": rep,
        "skey": skey,
        "gate": gate,
    }
    t0 = time.perf_counter()
    r = traceot.explain_v5_2_specified(xa0, ya, xb0, yb)
    out["TRACE_V5_2_SPECIFIED"] = {
        "lsplit": {"comp": list(r["lsplit_comp"]), "struct": list(r["lsplit_struct"])},
        "dropped_cols_a": r["dropped_cols_a"],
        "dropped_cols_b": r["dropped_cols_b"],
        "sec": time.perf_counter() - t0,
        "sec_core": r["seconds_total"],
    }
    # Rivals: same centred arrays and settings as the V5.1 / V5.2 gates.
    xa, xb = centred(xa0, xb0)
    rivals = {}
    for name, meth in (("DFL", DFL_L()), ("CFM", CFM_L())):
        t1 = time.perf_counter()
        try:
            rd = meth.decompose(xa, ya, xb, yb, seed=seed)
            rivals[name] = {"lsplit": rd.extra.get("lsplit"), "sec": time.perf_counter() - t1}
        except Exception as e:  # noqa: BLE001 — keep failure on record
            rivals[name] = {"error": str(e), "sec": time.perf_counter() - t1}
    out["rivals"] = rivals
    return out


def task(a):
    scen, kappa, n, rep = a
    return _run_one(scen, kappa, n, rep, gate=True)


def rehearsal_task(a):
    scen, kappa, n, rep = a
    return _run_one(scen, kappa, n, rep, gate=False)


def plan():
    todo, skipped = [], []
    assert ALLOWED.isdisjoint(FORBIDDEN)
    assert SEALED <= FORBIDDEN
    assert COVERAGE_SPENT <= FORBIDDEN
    assert SPECIFIED_SPENT <= FORBIDDEN
    assert V52B_SPENT <= FORBIDDEN
    assert SMOKE_SEED not in ALLOWED
    assert SMOKE_SEED not in SEALED
    assert set(REHEARSAL_SEEDS).isdisjoint(ALLOWED)
    assert set(REHEARSAL_SEEDS).isdisjoint(SEALED)
    for n in (20000, 2000):  # large cells first so the cap binds on the expensive work
        for rep in SIZES[n][1]:
            assert_rep(rep, n)
            for scen in SCEN:
                if sp.load_sigma(scen, KAPPA) is None:
                    skipped.append(sp.key(scen, KAPPA))
                    continue
                todo.append((scen, KAPPA, n, rep))
    return todo, sorted(set(skipped))


def rehearsal_plan():
    todo = []
    for n in (20000, 2000):
        for rep in REHEARSAL_SEEDS:
            assert_rehearsal_rep(rep)
            for scen in SCEN:
                if sp.load_sigma(scen, KAPPA) is None:
                    continue
                todo.append((scen, KAPPA, n, rep))
    return todo


def _pkg_version(name: str) -> str:
    try:
        mod = __import__(name)
        return str(getattr(mod, "__version__", "unknown"))
    except Exception as e:  # noqa: BLE001
        return f"IMPORT_FAILED: {type(e).__name__}: {e}"


def collect_versions() -> dict:
    return {
        "python": sys.version,
        "traceot": _pkg_version("traceot"),
        "numpy": _pkg_version("numpy"),
        "scipy": _pkg_version("scipy"),
        "scikit-learn": _pkg_version("sklearn"),
    }


def preflight() -> dict:
    """Mandatory pre-flight (PREREG_V5_2B §2, by reference). Runs in a child."""
    record: dict = {
        "versions": collect_versions(),
        "imports": {},
        "smoke": {},
        "ok": False,
        "smoke_seed": SMOKE_SEED,
        "smoke_scen": "skew_comp",
        "smoke_n": 2000,
        "smoke_p": 5,
        "prereg_sha": PREREG_SHA,
    }
    try:
        import sklearn  # noqa: F401
        from sb_shape_readout import CFM_L, DFL_L

        import traceot

        if not hasattr(traceot, "explain_v5_2_specified"):
            raise RuntimeError("traceot.explain_v5_2_specified is missing")
        record["imports"]["sklearn"] = "ok"
        record["imports"]["traceot"] = "ok"
        record["imports"]["DFL"] = "ok"
        record["imports"]["CFM"] = "ok"
        record["versions"] = collect_versions()

        if sp.load_sigma("skew_comp", KAPPA) is None:
            raise RuntimeError("smoke: calibrated sigma_B for skew_comp is missing")
        s = sp.make("skew_comp", KAPPA, 5)
        rng = np.random.default_rng(SMOKE_SEED)
        xa0, ya, xb0, yb = s.sample(2000, rng)

        r = traceot.explain_v5_2_specified(xa0, ya, xb0, yb)
        if "lsplit_comp" not in r or "lsplit_struct" not in r:
            raise RuntimeError("smoke TRACE fit did not return lsplit fields")
        record["smoke"]["TRACE"] = "ok"

        xa, xb = centred(xa0, xb0)
        for name, meth in (("DFL", DFL_L()), ("CFM", CFM_L())):
            rd = meth.decompose(xa, ya, xb, yb, seed=SMOKE_SEED)
            if rd is None or rd.extra is None or rd.extra.get("lsplit") is None:
                raise RuntimeError(f"smoke {name} fit did not return an lsplit")
            record["smoke"][name] = "ok"

        record["ok"] = True
    except Exception as e:  # noqa: BLE001 — abort is the protocol
        record["ok"] = False
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc()
    return record


def run_preflight_spawned() -> dict:
    """Execute pre-flight in a spawned child. The parent never imports TRACE."""
    spawn_safe.assert_parent_has_not_called_trace()
    rec = spawn_safe.run_in_spawned_child(preflight, timeout=PREFLIGHT_TIMEOUT)
    spawn_safe.assert_parent_has_not_called_trace()
    return rec


def drain_pool(pool, fn, todo, out_path: Path, *, cap_sec: float, watchdog_sec: float):
    """Write completed records. Returns (done, wall_sec, stop).

    ``stop`` is None if every task finished, ``'watchdog_abort'`` if 5 minutes
    elapsed with no completion, or ``'wall_cap'`` if the cap expired.
    """
    t0 = time.perf_counter()
    last = t0
    done = 0
    stop = None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as fh:
        it = pool.imap_unordered(fn, todo)
        while done < len(todo):
            now = time.perf_counter()
            left_wd = watchdog_sec - (now - last)
            left_cap = cap_sec - (now - t0)
            if left_wd <= 0:
                stop = "watchdog_abort"
                break
            if left_cap <= 0:
                stop = "wall_cap"
                break
            try:
                rec = it.next(timeout=min(left_wd, left_cap))
            except Exception as e:  # noqa: BLE001
                if type(e).__name__ == "TimeoutError":
                    now = time.perf_counter()
                    if (now - last) >= watchdog_sec:
                        stop = "watchdog_abort"
                    else:
                        stop = "wall_cap"
                    break
                raise
            rec["wall_at_finish"] = time.perf_counter() - t0
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done += 1
            last = time.perf_counter()
    return done, time.perf_counter() - t0, stop


def rehearsal_ok(path: Path, done: int, total: int, stop: str | None) -> tuple[bool, str]:
    """Any failure aborts before gate seeds are touched."""
    if stop == "watchdog_abort":
        return False, "watchdog tripped during rehearsal"
    if stop is not None:
        return False, f"rehearsal stopped: {stop}"
    if done != total:
        return False, f"rehearsal completed {done}/{total} tasks"
    if not path.exists():
        return False, "rehearsal file is missing"
    rows = [json.loads(x) for x in path.read_text().splitlines() if x]
    if len(rows) != total:
        return False, f"rehearsal file has {len(rows)} records, expected {total}"
    for r in rows:
        tr = r.get("TRACE_V5_2_SPECIFIED", {})
        if "lsplit" not in tr or "error" in tr:
            return False, (
                f"TRACE failed on rehearsal {r.get('scen')} n={r.get('n')} "
                f"seed={r.get('rep')}: {tr.get('error', 'no lsplit')}"
            )
        for nm in ("DFL", "CFM"):
            rv = r.get("rivals", {}).get(nm, {})
            if "error" in rv or "lsplit" not in rv:
                return False, (
                    f"{nm} failed on rehearsal {r.get('scen')} n={r.get('n')} "
                    f"seed={r.get('rep')}: {rv.get('error', 'no lsplit')}"
                )
        if r.get("rep") not in REHEARSAL_SEEDS:
            return False, f"rehearsal record has non-rehearsal seed {r.get('rep')}"
        if r.get("gate") is True:
            return False, "rehearsal record is marked as a gate task"
    return True, "ok"


def _write_wall(payload: dict) -> None:
    WALL.write_text(json.dumps(payload) + "\n")


def _base_wall(workers: int, skipped: list, **extra) -> dict:
    payload = {
        "tasks_preregistered": 140,
        "cells_infeasible": skipped,
        "cap_sec": CAP_SEC,
        "watchdog_sec": WATCHDOG_SEC,
        "workers": workers,
        "prereg_sha": PREREG_SHA,
        "impl_sha": IMPL_SHA,
    }
    payload.update(extra)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    spawn_safe.assert_parent_has_not_called_trace()
    assert not OUT.exists(), "single pass: results file already exists"
    assert not REHEARSAL_OUT.exists(), "single pass: rehearsal file already exists"
    todo, skipped = plan()
    r_todo = rehearsal_plan()
    print(
        f"v5.2c specified gate: {len(todo)} tasks -> {OUT} (cap {CAP_SEC}s, "
        f"watchdog {WATCHDOG_SEC}s); "
        f"cells with infeasible calibration (INCOMPLETE, no tasks): {skipped}",
        flush=True,
    )
    if skipped:
        _write_wall(
            _base_wall(
                args.workers,
                skipped,
                wall_sec=0.0,
                tasks_done=0,
                tasks_total=0,
                stop="calibration_incomplete",
            )
        )
        print(
            "calibration INCOMPLETE: not running any gate index; H is not decided",
            flush=True,
        )
        return

    print("pre-flight: spawned child, imports and smoke fits on seed 99", flush=True)
    pf = run_preflight_spawned()
    PREFLIGHT_OUT.write_text(json.dumps(pf, indent=2) + "\n")
    spawn_safe.assert_parent_has_not_called_trace()
    if not pf.get("ok"):
        _write_wall(
            _base_wall(
                args.workers,
                skipped,
                wall_sec=0.0,
                tasks_done=0,
                tasks_total=0,
                stop="preflight_abort",
                preflight_error=pf.get("error"),
                preflight_path=str(PREFLIGHT_OUT),
            )
        )
        print(
            f"pre-flight ABORT: {pf.get('error')}; not running any gate index; "
            "H is not decided; seeds remain unspent",
            flush=True,
        )
        return
    print(
        "pre-flight OK: "
        + ", ".join(f"{k}={v}" for k, v in pf["versions"].items() if k != "python"),
        flush=True,
    )

    print(
        f"rehearsal: {len(r_todo)} tasks on seeds {REHEARSAL_SEEDS} -> {REHEARSAL_OUT}",
        flush=True,
    )
    spawn_safe.assert_parent_has_not_called_trace()
    r_pool = spawn_safe.spawn_pool(args.workers)
    r_error = None
    try:
        r_done, r_wall, r_stop = drain_pool(
            r_pool,
            rehearsal_task,
            r_todo,
            REHEARSAL_OUT,
            cap_sec=REHEARSAL_CAP_SEC,
            watchdog_sec=WATCHDOG_SEC,
        )
    except Exception as e:  # noqa: BLE001 — rehearsal failure aborts
        r_done, r_wall, r_stop = 0, 0.0, "rehearsal_error"
        r_error = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        r_pool.terminate()
        r_pool.join()
    spawn_safe.assert_parent_has_not_called_trace()
    ok, reason = rehearsal_ok(REHEARSAL_OUT, r_done, len(r_todo), r_stop)
    if r_error is not None:
        ok, reason = False, r_error
    if not ok:
        stop = "watchdog_abort" if r_stop == "watchdog_abort" else "rehearsal_abort"
        _write_wall(
            _base_wall(
                args.workers,
                skipped,
                wall_sec=0.0,
                rehearsal_wall_sec=r_wall,
                tasks_done=0,
                tasks_total=0,
                rehearsal_tasks_done=r_done,
                rehearsal_tasks_total=len(r_todo),
                stop=stop,
                rehearsal_error=reason,
                rehearsal_path=str(REHEARSAL_OUT),
                preflight_ok=True,
                preflight_path=str(PREFLIGHT_OUT),
            )
        )
        print(
            f"rehearsal ABORT: {reason}; not running any gate index; "
            "H is not decided; seeds remain unspent",
            flush=True,
        )
        return
    print(f"rehearsal OK: {r_done}/{len(r_todo)} tasks in {r_wall:.1f}s", flush=True)

    spawn_safe.assert_parent_has_not_called_trace()
    print("gate pass: spawn pool, 30-minute cap starts now", flush=True)
    pool = spawn_safe.spawn_pool(args.workers)
    try:
        done, wall, stop = drain_pool(
            pool,
            task,
            todo,
            OUT,
            cap_sec=CAP_SEC,
            watchdog_sec=WATCHDOG_SEC,
        )
    finally:
        pool.terminate()
        pool.join()
    _write_wall(
        _base_wall(
            args.workers,
            skipped,
            wall_sec=wall,
            tasks_done=done,
            tasks_total=len(todo),
            rehearsal_wall_sec=r_wall,
            rehearsal_tasks_done=r_done,
            rehearsal_tasks_total=len(r_todo),
            stop=stop,
            preflight_ok=True,
            preflight_path=str(PREFLIGHT_OUT),
            rehearsal_ok=True,
            rehearsal_path=str(REHEARSAL_OUT),
        )
    )
    if stop == "watchdog_abort":
        print(
            f"gate WATCHDOG ABORT after {wall:.1f}s, {done}/{len(todo)} tasks; "
            "H is not decided; seeds are spent",
            flush=True,
        )
    else:
        print(f"wall {wall:.1f}s, {done}/{len(todo)} tasks, stop={stop}", flush=True)


if __name__ == "__main__":
    main()
