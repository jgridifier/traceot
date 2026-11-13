"""TRACE v5.2b specified-family replication gate (PREREG_V5_2B_SPECIFIED_REPLICATION.md).

    python run_v5_2b_specified_gate.py [--workers 8]

Replication of the V5.2 specified-family gate: same hypothesis, method, simulator,
rivals, Holm / BEST / NON-INFERIOR rule, δ, and 30-minute cap. Fresh seeds:
skew_comp / tail_comp at n=2000, p=5 (indices 12000-12049) and n=20000, p=20
(indices 12100-12119), kappa=1, with paired DFL / CFM rivals.

A mandatory pre-flight (import TRACE and both rivals, one smoke fit per method
on non-gate seed 99, record package versions) runs in this exact environment
before any gate seed is touched. Any failure aborts: gate seeds are not used,
H is not decided, and this is not a FAIL and not a park.

This module is the runner: invoking it executes the study; importing it does
not. This file's presence in a PR is not a run.

Sealed indices 9600-9619 are never run. Spent coverage indices 10400-10799 and
spent specified-family indices 11000-11049 / 11100-11119 are not re-run. A
scenario whose sigma_B calibration is infeasible has no tasks and its cells are
INCOMPLETE; H is then not decided and gate seeds are not touched. Tasks not
finished inside the cap are not imputed.

The method under test receives raw (uncentred) draws, as in
PREREG_V5_2_SPECIFIED §2 (by reference). Rivals receive the same centred
arrays as in the V5.1 / V5.2 gates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool

os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import sb_dgp_shape_v5_2_specified as sp
from run_synthetic import RES, centred, seed_of

# Freeze point: the merge commit that added PREREG_V5_2B_SPECIFIED_REPLICATION.md
# to main (PR #61).
PREREG_SHA = "19ed3202c701a5ee924399885265b1996d4a13cc"
IMPL_SHA = "1aab4778abf8f84c8c8f7d9603eb8c322f1ba3c9"
SCEN = ("skew_comp", "tail_comp")
KAPPA = 1.0
SIZES = {2000: (5, list(range(12000, 12050))), 20000: (20, list(range(12100, 12120)))}
ALLOWED = set(range(12000, 12050)) | set(range(12100, 12120))
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
    | set(range(81001, 81021))
    | set(range(83001, 83021))
    | set(range(84001, 84051))
)
SEALED = set(range(9600, 9620))
COVERAGE_SPENT = set(range(10400, 10800))
SPECIFIED_SPENT = set(range(11000, 11050)) | set(range(11100, 11120))
CAP_SEC = 30 * 60
SMOKE_SEED = 99
OUT = RES / "v5_2b_specified_gate_shape.jsonl"
WALL = RES / "v5_2b_specified_gate_wall.json"
PREFLIGHT_OUT = RES / "v5_2b_specified_gate_preflight.json"


def assert_rep(rep: int, n: int) -> None:
    assert rep in ALLOWED, f"index {rep} is not in 12000-12049 or 12100-12119"
    assert rep in SIZES[n][1], f"index {rep} is not in the range for n={n}"
    assert rep not in FORBIDDEN, f"index {rep} is in the FORBIDDEN set"
    assert rep not in SEALED, f"index {rep} is in the sealed range 9600-9619"
    assert rep not in COVERAGE_SPENT, f"index {rep} is a spent V5.1 coverage index"
    assert rep not in SPECIFIED_SPENT, f"index {rep} is a spent V5.2 specified index"


def task(a):
    from sb_shape_readout import CFM_L, DFL_L

    import traceot

    scen, kappa, n, rep = a
    p = SIZES[n][0]
    assert_rep(rep, n)
    s = sp.make(scen, kappa, p)
    skey = sp.key(scen, kappa)
    rng = np.random.default_rng(seed_of("data", skey, n, p, rep))
    XA0, yA, XB0, yB = s.sample(n, rng)
    # Method under test: raw draws (PREREG_V5_2_SPECIFIED §2, by reference).
    seed = seed_of("m", rep)
    out = {"scen": scen, "kappa": kappa, "n": n, "p": p, "rep": rep, "skey": skey}
    t0 = time.perf_counter()
    r = traceot.explain_v5_2_specified(XA0, yA, XB0, yB)
    out["TRACE_V5_2_SPECIFIED"] = {
        "lsplit": {"comp": list(r["lsplit_comp"]), "struct": list(r["lsplit_struct"])},
        "dropped_cols_a": r["dropped_cols_a"],
        "dropped_cols_b": r["dropped_cols_b"],
        "sec": time.perf_counter() - t0,
        "sec_core": r["seconds_total"],
    }
    # Rivals: same centred arrays and settings as the V5.1 / V5.2 gates.
    XA, XB = centred(XA0, XB0)
    rivals = {}
    for name, meth in (("DFL", DFL_L()), ("CFM", CFM_L())):
        t1 = time.perf_counter()
        try:
            rd = meth.decompose(XA, yA, XB, yB, seed=seed)
            rivals[name] = {"lsplit": rd.extra.get("lsplit"), "sec": time.perf_counter() - t1}
        except Exception as e:  # noqa: BLE001 — keep failure on record
            rivals[name] = {"error": str(e), "sec": time.perf_counter() - t1}
    out["rivals"] = rivals
    return out


def plan():
    todo, skipped = [], []
    assert ALLOWED.isdisjoint(FORBIDDEN)
    assert SEALED <= FORBIDDEN
    assert COVERAGE_SPENT <= FORBIDDEN
    assert SPECIFIED_SPENT <= FORBIDDEN
    assert SMOKE_SEED not in ALLOWED
    assert SMOKE_SEED not in SEALED
    for n in (20000, 2000):  # large cells first so the cap binds on the expensive work
        for rep in SIZES[n][1]:
            assert_rep(rep, n)
            for scen in SCEN:
                if sp.load_sigma(scen, KAPPA) is None:
                    skipped.append(sp.key(scen, KAPPA))
                    continue
                todo.append((scen, KAPPA, n, rep))
    return todo, sorted(set(skipped))


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
    """Mandatory pre-flight (PREREG_V5_2B_SPECIFIED_REPLICATION §2).

    Import TRACE and both rivals in this exact environment, run one smoke fit
    of each method on non-gate seed 99, and record package versions. Does not
    write the official results file and does not touch any gate index.
    """
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
        # Re-record versions now that imports succeeded (fills sklearn if it
        # had been missing at collect_versions, which would already have failed).
        record["versions"] = collect_versions()

        if sp.load_sigma("skew_comp", KAPPA) is None:
            raise RuntimeError("smoke: calibrated sigma_B for skew_comp is missing")
        s = sp.make("skew_comp", KAPPA, 5)
        rng = np.random.default_rng(SMOKE_SEED)
        XA0, yA, XB0, yB = s.sample(2000, rng)

        r = traceot.explain_v5_2_specified(XA0, yA, XB0, yB)
        if "lsplit_comp" not in r or "lsplit_struct" not in r:
            raise RuntimeError("smoke TRACE fit did not return lsplit fields")
        record["smoke"]["TRACE"] = "ok"

        XA, XB = centred(XA0, XB0)
        for name, meth in (("DFL", DFL_L()), ("CFM", CFM_L())):
            rd = meth.decompose(XA, yA, XB, yB, seed=SMOKE_SEED)
            if rd is None or rd.extra is None or rd.extra.get("lsplit") is None:
                raise RuntimeError(f"smoke {name} fit did not return an lsplit")
            record["smoke"][name] = "ok"

        record["ok"] = True
    except Exception as e:  # noqa: BLE001 — abort is the protocol
        record["ok"] = False
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc()
    return record


def _write_wall(payload: dict) -> None:
    WALL.write_text(json.dumps(payload) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    assert not OUT.exists(), "single pass: results file already exists"
    todo, skipped = plan()
    print(
        f"v5.2b specified gate: {len(todo)} tasks -> {OUT} (cap {CAP_SEC}s); "
        f"cells with infeasible calibration (INCOMPLETE, no tasks): {skipped}",
        flush=True,
    )
    if skipped:
        # Calibration-INCOMPLETE stop: H is not decided; gate seeds are not touched.
        # Pre-flight is not run; parent §3 stop comes first.
        _write_wall(
            {
                "wall_sec": 0.0,
                "tasks_done": 0,
                "tasks_total": 0,
                "tasks_preregistered": 140,
                "cells_infeasible": skipped,
                "stop": "calibration_incomplete",
                "cap_sec": CAP_SEC,
                "workers": args.workers,
                "prereg_sha": PREREG_SHA,
                "impl_sha": IMPL_SHA,
            }
        )
        print(
            "calibration INCOMPLETE: not running any gate index; H is not decided",
            flush=True,
        )
        return

    print("pre-flight: importing TRACE and rivals, smoke fits on seed 99", flush=True)
    pf = preflight()
    PREFLIGHT_OUT.write_text(json.dumps(pf, indent=2) + "\n")
    if not pf.get("ok"):
        _write_wall(
            {
                "wall_sec": 0.0,
                "tasks_done": 0,
                "tasks_total": 0,
                "tasks_preregistered": 140,
                "cells_infeasible": skipped,
                "stop": "preflight_abort",
                "preflight_error": pf.get("error"),
                "preflight_path": str(PREFLIGHT_OUT),
                "cap_sec": CAP_SEC,
                "workers": args.workers,
                "prereg_sha": PREREG_SHA,
                "impl_sha": IMPL_SHA,
            }
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

    t0 = time.perf_counter()
    done = 0
    pool = Pool(args.workers)
    try:
        with OUT.open("a") as fh:
            it = pool.imap_unordered(task, todo)
            while done < len(todo):
                left = CAP_SEC - (time.perf_counter() - t0)
                if left <= 0:
                    break
                try:
                    rec = it.next(timeout=left)
                except Exception as e:  # noqa: BLE001
                    if type(e).__name__ == "TimeoutError":
                        break
                    raise
                rec["wall_at_finish"] = time.perf_counter() - t0
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                done += 1
    finally:
        pool.terminate()
    wall = time.perf_counter() - t0
    _write_wall(
        {
            "wall_sec": wall,
            "tasks_done": done,
            "tasks_total": len(todo),
            "tasks_preregistered": 140,
            "cells_infeasible": skipped,
            "stop": None,
            "cap_sec": CAP_SEC,
            "workers": args.workers,
            "prereg_sha": PREREG_SHA,
            "impl_sha": IMPL_SHA,
            "preflight_ok": True,
            "preflight_path": str(PREFLIGHT_OUT),
        }
    )
    print(f"wall {wall:.1f}s, {done}/{len(todo)} tasks", flush=True)


if __name__ == "__main__":
    main()
