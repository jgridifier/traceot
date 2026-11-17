"""TRACE v5.1 structure/mixed-cell gate (PREREG_V5_1_STRUCT.md).

    python run_v5_1_struct_gate.py [--workers 8]
    python run_v5_1_struct_gate.py --confirmatory [--workers 8]

Pre-registered hypothesis: TRACE_V5_1, unchanged, is BEST or NON-INFERIOR
against the best paired rival (DFL, CFM) after Holm on skew_struct,
tail_struct, and shape_mixed at n=2000 (indices 14000-14049) and n=20000
(indices 14100-14119), and passes the role check on all six cells and the
false-attribution check on the four structure-only cells.

This module is the runner: invoking it may execute a step; importing it
does not. This file's presence in a PR is not a run.

Default (no --confirmatory): mandatory pre-flight in a spawned child
(imports, smoke fits, and the §2 method-identity check on oracle seed 99)
and a rehearsal of the full parallel pipeline on throwaway seeds 98 and 97.
Rehearsal outputs go to a separate file and never enter scoring. Gate
indices 14000-14049 and 14100-14119 are refused. Any pre-flight or
rehearsal failure aborts without consuming gate seeds.

--confirmatory: after a successful pre-flight and rehearsal (reused if
already on disk and OK, otherwise run first), one single pass over the
210 gate tasks. A 5-minute no-completion watchdog and a 30-minute wall
cap apply. n=20000 cells are scheduled first. No task is dropped,
retried, or re-run.

The gate pool and the rehearsal pool use multiprocessing's 'spawn' start
method, from a parent that never calls TRACE.

Sealed indices 9600-9619 are never run. Spent coverage indices
10400-10799 and every other inventory range in PREREG_V5_1_STRUCT.md §4
are in FORBIDDEN and are refused.

Four commits after the V5.1 implementation merge 665939b touched
crates/core (5db5806, 3b8a1d4, 3b382f0, 4cf9754). The pre-flight
therefore compares explain_v5_1 output on one centred skew_struct draw
at oracle seed 99 to the frozen 665939b record in
benchmarks/results/v5_1_struct_method_check.json. If the split is not
bit-identical, the run stops: that is a method change and needs its own
pre-registration. Timings and the later report-only linear_mean_check /
dropped_cols fields are excluded from the comparison; they do not enter
the split.

Method and rivals receive the same centred arrays as in run_v5_1_gate.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import sb_dgp_shape as shp
import spawn_safe
from run_synthetic import RES, centred, seed_of

# Freeze point: the merge commit that added PREREG_V5_1_STRUCT.md to main (PR #70).
PREREG_SHA = "85a8fccfac44b6a1506037d785d4be6cf56c3bdb"
IMPL_SHA = "665939bd4aa78ffc7474b91d482d18ac0191d595"
CORE_COMMITS_AFTER_IMPL = (
    "5db58067e4356035eab43078ec80ccff97ecb304",
    "3b8a1d4c827444bbbb86ba2ffa5149897e179746",
    "3b382f0f47426f4f6f1c1badcd452aeb25936378",
    "4cf9754ff4deeb7ed1da9d2a0267529563ed87b3",
)
SCEN = ("skew_struct", "tail_struct", "shape_mixed")
SIZES = {2000: (5, list(range(14000, 14050))), 20000: (20, list(range(14100, 14120)))}
ALLOWED = set(range(14000, 14050)) | set(range(14100, 14120))
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
    | set(range(10400, 10800))  # spent V5.1 coverage; named explicitly
    | set(range(10800, 11000))
    | set(range(11000, 11050))
    | set(range(11100, 11120))
    | set(range(11120, 12000))
    | set(range(12000, 12050))
    | set(range(12050, 12100))
    | set(range(12100, 12120))
    | set(range(12120, 13000))
    | set(range(13000, 13050))  # spent V5.2c n=2000
    | set(range(13050, 13100))
    | set(range(13100, 13120))  # spent V5.2c n=20000
    | set(range(13120, 14000))  # buffer after V5.2c
    | set(range(14050, 14100))  # gap between this suite's blocks
    | set(range(81001, 81021))
    | set(range(83001, 83021))
    | set(range(84001, 84051))
)
SEALED = set(range(9600, 9620))
COVERAGE_SPENT = set(range(10400, 10800))
CAP_SEC = 30 * 60
WATCHDOG_SEC = 5 * 60
REHEARSAL_CAP_SEC = 24 * 60 * 60
SMOKE_SEED = 99
REHEARSAL_SEEDS = (98, 97)
N_TASKS = 210
OUT = RES / "v5_1_struct_gate_shape.jsonl"
REHEARSAL_OUT = RES / "v5_1_struct_gate_rehearsal.jsonl"
WALL = RES / "v5_1_struct_gate_wall.json"
PREFLIGHT_OUT = RES / "v5_1_struct_gate_preflight.json"
METHOD_CHECK_INPUTS = RES / "v5_1_struct_method_check_inputs.npz"
METHOD_CHECK_REF = RES / "v5_1_struct_method_check.json"
PREFLIGHT_TIMEOUT = 5 * 60


def assert_rep(rep: int, n: int) -> None:
    assert rep in ALLOWED, f"index {rep} is not in 14000-14049 or 14100-14119"
    assert rep in SIZES[n][1], f"index {rep} is not in the range for n={n}"
    assert rep not in FORBIDDEN, f"index {rep} is in the FORBIDDEN set"
    assert rep not in SEALED, f"index {rep} is in the sealed range 9600-9619"
    assert rep not in COVERAGE_SPENT, f"index {rep} is a spent V5.1 coverage index"


def assert_rehearsal_rep(rep: int) -> None:
    assert rep in REHEARSAL_SEEDS, f"rehearsal index {rep} is not 98 or 97"
    assert rep not in ALLOWED
    assert rep not in SEALED
    assert rep not in COVERAGE_SPENT


def _run_one(scen: str, n: int, rep: int, *, gate: bool) -> dict:
    """One TRACE_V5_1 + DFL + CFM fit. Used by gate and rehearsal workers."""
    from sb_shape_readout import CFM_L, DFL_L

    import traceot

    p = SIZES[n][0]
    if gate:
        assert_rep(rep, n)
    else:
        assert_rehearsal_rep(rep)
        assert n in SIZES
    s = shp.make(scen, p)
    rng = np.random.default_rng(seed_of("data", scen, n, p, rep))
    xa0, ya, xb0, yb = s.sample(n, rng)
    xa, xb = centred(xa0, xb0)
    seed = seed_of("m", rep)
    out = {"scen": scen, "n": n, "p": p, "rep": rep, "gate": gate}
    t0 = time.perf_counter()
    r = traceot.explain_v5_1(xa, ya, xb, yb)
    out["TRACE_V5_1"] = {
        "lsplit": {"comp": list(r["lsplit_comp"]), "struct": list(r["lsplit_struct"])},
        "mixing_residual": {h: r[h]["mixing_residual"] for h in ("aa", "ab", "ba", "bb")},
        "sec": time.perf_counter() - t0,
        "sec_core": r["seconds_total"],
    }
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
    scen, n, rep = a
    return _run_one(scen, n, rep, gate=True)


def rehearsal_task(a):
    scen, n, rep = a
    return _run_one(scen, n, rep, gate=False)


def plan():
    todo = []
    assert ALLOWED.isdisjoint(FORBIDDEN)
    assert SEALED <= FORBIDDEN
    assert COVERAGE_SPENT <= FORBIDDEN
    assert SMOKE_SEED not in ALLOWED
    assert SMOKE_SEED not in SEALED
    assert set(REHEARSAL_SEEDS).isdisjoint(ALLOWED)
    assert set(REHEARSAL_SEEDS).isdisjoint(SEALED)
    for n in (20000, 2000):  # large cells first so the cap binds on the expensive work
        for rep in SIZES[n][1]:
            assert_rep(rep, n)
            for scen in SCEN:
                todo.append((scen, n, rep))
    assert len(todo) == N_TASKS
    return todo


def rehearsal_plan():
    todo = []
    for n in (20000, 2000):
        for rep in REHEARSAL_SEEDS:
            assert_rehearsal_rep(rep)
            for scen in SCEN:
                todo.append((scen, n, rep))
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


def _sha256_arr(a) -> str:
    return hashlib.sha256(np.asarray(a, dtype=np.float64).tobytes()).hexdigest()


def method_identity_check() -> dict:
    """Bit-identity of explain_v5_1 vs the frozen 665939b record (seed 99 only)."""
    import traceot

    rec: dict = {
        "ok": False,
        "impl_sha": IMPL_SHA,
        "core_commits_after_impl": list(CORE_COMMITS_AFTER_IMPL),
        "seed": SMOKE_SEED,
        "scen": "skew_struct",
    }
    if not METHOD_CHECK_INPUTS.exists() or not METHOD_CHECK_REF.exists():
        rec["error"] = "method-check fixture is missing"
        return rec
    ref = json.loads(METHOD_CHECK_REF.read_text())
    data = np.load(METHOD_CHECK_INPUTS)
    xa, ya, xb, yb = data["XA"], data["yA"], data["XB"], data["yB"]
    got_inputs = {k: _sha256_arr(data[k]) for k in ("XA", "yA", "XB", "yB")}
    rec["input_sha256"] = got_inputs
    if got_inputs != ref["draw"]["input_sha256"]:
        rec["error"] = "method-check inputs do not match the frozen SHA-256 record"
        return rec

    # Regenerating the same draw must match the frozen arrays (DGP unchanged).
    s = shp.make("skew_struct", 5)
    rng = np.random.default_rng(SMOKE_SEED)
    xa0, ya_g, xb0, yb_g = s.sample(2000, rng)
    xa_g, xb_g = centred(xa0, xb0)
    if not (
        _sha256_arr(xa_g) == got_inputs["XA"]
        and _sha256_arr(ya_g) == got_inputs["yA"]
        and _sha256_arr(xb_g) == got_inputs["XB"]
        and _sha256_arr(yb_g) == got_inputs["yB"]
    ):
        rec["error"] = "regenerated seed-99 skew_struct draw does not match frozen inputs"
        return rec

    r = traceot.explain_v5_1(xa, ya, xb, yb)
    mismatches = []
    for key in ("lsplit_comp", "lsplit_struct"):
        got = _sha256_arr(r[key])
        exp = ref["reference"][key]["sha256"]
        rec[f"{key}_sha256"] = got
        if got != exp:
            mismatches.append(key)
    for h in ("aa", "ab", "ba", "bb"):
        for k, meta in ref["reference"][h].items():
            got = _sha256_arr(r[h][k])
            if got != meta["sha256"]:
                mismatches.append(f"{h}.{k}")
    rec["mismatches"] = mismatches
    rec["ok"] = not mismatches
    if mismatches:
        rec["error"] = (
            "explain_v5_1 output is not bit-identical to the 665939b record on "
            f"the seed-99 skew_struct draw: {mismatches}. Stop: this is a method "
            "change and needs its own pre-registration."
        )
    return rec


def preflight() -> dict:
    """Mandatory pre-flight (PREREG_V5_1_STRUCT §6 / §2). Runs in a child."""
    record: dict = {
        "versions": collect_versions(),
        "imports": {},
        "smoke": {},
        "method_identity": {},
        "ok": False,
        "smoke_seed": SMOKE_SEED,
        "smoke_scen": "skew_struct",
        "smoke_n": 2000,
        "smoke_p": 5,
        "prereg_sha": PREREG_SHA,
        "impl_sha": IMPL_SHA,
        "core_commits_after_impl": list(CORE_COMMITS_AFTER_IMPL),
    }
    try:
        import sklearn  # noqa: F401
        from sb_shape_readout import CFM_L, DFL_L

        import traceot

        if not hasattr(traceot, "explain_v5_1"):
            raise RuntimeError("traceot.explain_v5_1 is missing")
        record["imports"]["sklearn"] = "ok"
        record["imports"]["traceot"] = "ok"
        record["imports"]["DFL"] = "ok"
        record["imports"]["CFM"] = "ok"
        record["versions"] = collect_versions()

        ident = method_identity_check()
        record["method_identity"] = ident
        if not ident.get("ok"):
            raise RuntimeError(ident.get("error") or "method-identity check failed")

        s = shp.make("skew_struct", 5)
        rng = np.random.default_rng(SMOKE_SEED)
        xa0, ya, xb0, yb = s.sample(2000, rng)
        xa, xb = centred(xa0, xb0)

        r = traceot.explain_v5_1(xa, ya, xb, yb)
        if "lsplit_comp" not in r or "lsplit_struct" not in r:
            raise RuntimeError("smoke TRACE fit did not return lsplit fields")
        record["smoke"]["TRACE"] = "ok"

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
        tr = r.get("TRACE_V5_1", {})
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
        if r.get("rep") in ALLOWED or r.get("rep") in SEALED:
            return False, f"rehearsal record has a reserved seed {r.get('rep')}"
        if r.get("gate") is True:
            return False, "rehearsal record is marked as a gate task"
    return True, "ok"


def existing_preflight_ok() -> dict | None:
    if not PREFLIGHT_OUT.exists():
        return None
    rec = json.loads(PREFLIGHT_OUT.read_text())
    return rec if rec.get("ok") else None


def existing_rehearsal_ok() -> tuple[bool, str, int]:
    if not REHEARSAL_OUT.exists():
        return False, "missing", 0
    rows = [json.loads(x) for x in REHEARSAL_OUT.read_text().splitlines() if x]
    ok, reason = rehearsal_ok(REHEARSAL_OUT, len(rows), len(rehearsal_plan()), None)
    return ok, reason, len(rows)


def _write_wall(payload: dict) -> None:
    WALL.write_text(json.dumps(payload) + "\n")


def _base_wall(workers: int, **extra) -> dict:
    payload = {
        "tasks_preregistered": N_TASKS,
        "cap_sec": CAP_SEC,
        "watchdog_sec": WATCHDOG_SEC,
        "workers": workers,
        "prereg_sha": PREREG_SHA,
        "impl_sha": IMPL_SHA,
    }
    payload.update(extra)
    return payload


def _run_preflight_step(workers: int) -> dict | None:
    existing = existing_preflight_ok()
    if existing is not None:
        print("pre-flight: reusing existing OK record", flush=True)
        return existing
    if PREFLIGHT_OUT.exists():
        pf = json.loads(PREFLIGHT_OUT.read_text())
        _write_wall(
            _base_wall(
                workers,
                wall_sec=0.0,
                tasks_done=0,
                tasks_total=0,
                stop="preflight_abort",
                preflight_error=pf.get("error"),
                preflight_path=str(PREFLIGHT_OUT),
            )
        )
        print(
            f"pre-flight ABORT: existing record is not ok: {pf.get('error')}; "
            "not running any gate index; H is not decided; seeds remain unspent",
            flush=True,
        )
        return None
    print(
        "pre-flight: spawned child, imports, seed-99 method-identity, smoke fits",
        flush=True,
    )
    pf = run_preflight_spawned()
    PREFLIGHT_OUT.write_text(json.dumps(pf, indent=2) + "\n")
    spawn_safe.assert_parent_has_not_called_trace()
    if not pf.get("ok"):
        _write_wall(
            _base_wall(
                workers,
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
        return None
    print(
        "pre-flight OK: method-identity bit-identical to 665939b; "
        + ", ".join(f"{k}={v}" for k, v in pf["versions"].items() if k != "python"),
        flush=True,
    )
    return pf


def _run_rehearsal_step(workers: int) -> tuple[bool, int, float]:
    r_todo = rehearsal_plan()
    ok_existing, reason_existing, n_existing = existing_rehearsal_ok()
    if ok_existing:
        print(
            f"rehearsal: reusing existing OK file ({n_existing}/{len(r_todo)} tasks)",
            flush=True,
        )
        return True, n_existing, 0.0
    if REHEARSAL_OUT.exists():
        _write_wall(
            _base_wall(
                workers,
                wall_sec=0.0,
                tasks_done=0,
                tasks_total=0,
                stop="rehearsal_abort",
                rehearsal_error=reason_existing,
                rehearsal_path=str(REHEARSAL_OUT),
                preflight_ok=True,
                preflight_path=str(PREFLIGHT_OUT),
            )
        )
        print(
            f"rehearsal ABORT: existing file is not ok ({reason_existing}); "
            "not running any gate index; H is not decided; seeds remain unspent",
            flush=True,
        )
        return False, n_existing, 0.0

    print(
        f"rehearsal: {len(r_todo)} tasks on seeds {REHEARSAL_SEEDS} -> {REHEARSAL_OUT}",
        flush=True,
    )
    spawn_safe.assert_parent_has_not_called_trace()
    r_pool = spawn_safe.spawn_pool(workers)
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
                workers,
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
        return False, r_done, r_wall
    print(f"rehearsal OK: {r_done}/{len(r_todo)} tasks in {r_wall:.1f}s", flush=True)
    return True, r_done, r_wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--confirmatory",
        action="store_true",
        help=(
            "Run the single pre-registered pass on 14000-14049 and 14100-14119. "
            "Without this flag those gate seeds are refused."
        ),
    )
    args = ap.parse_args()
    spawn_safe.assert_parent_has_not_called_trace()
    todo = plan()
    print(
        f"v5.1 struct gate: {len(todo)} pre-registered tasks, cap {CAP_SEC}s, "
        f"watchdog {WATCHDOG_SEC}s; confirmatory={args.confirmatory}",
        flush=True,
    )

    pf = _run_preflight_step(args.workers)
    if pf is None:
        return
    ok, r_done, r_wall = _run_rehearsal_step(args.workers)
    if not ok:
        return

    if not args.confirmatory:
        _write_wall(
            _base_wall(
                args.workers,
                wall_sec=0.0,
                tasks_done=0,
                tasks_total=0,
                rehearsal_wall_sec=r_wall,
                rehearsal_tasks_done=r_done,
                rehearsal_tasks_total=len(rehearsal_plan()),
                stop="awaiting_confirmatory",
                preflight_ok=True,
                preflight_path=str(PREFLIGHT_OUT),
                rehearsal_ok=True,
                rehearsal_path=str(REHEARSAL_OUT),
            )
        )
        print(
            "pre-flight and rehearsal OK; gate seeds 14000-14049 / 14100-14119 "
            "were not touched. Pass --confirmatory to run the single pass.",
            flush=True,
        )
        return

    assert not OUT.exists(), "single pass: results file already exists"
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
            wall_sec=wall,
            tasks_done=done,
            tasks_total=len(todo),
            rehearsal_wall_sec=r_wall,
            rehearsal_tasks_done=r_done,
            rehearsal_tasks_total=len(rehearsal_plan()),
            stop=stop,
            preflight_ok=True,
            preflight_path=str(PREFLIGHT_OUT),
            rehearsal_ok=True,
            rehearsal_path=str(REHEARSAL_OUT),
            confirmatory=True,
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
