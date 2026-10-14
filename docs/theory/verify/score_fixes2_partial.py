# ruff: noqa
"""Score the completed FIXES2 seeds against the frozen bars (FIXES2_PLAN.md @ cc699b2).

Early stop (deviation log, 2026-10-14). Reads the frozen runner's per-seed cache, applies the
same per-seed hit definitions, and reports k/n. A bar is PASS/FAIL only when determined by the
completed seeds; otherwise INCOMPLETE. Writes verify/FIXES2_RESULTS.txt.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = Path("/tmp/prereg_fixes2_cache")
OUT = ROOT / "FIXES2_RESULTS.txt"
SEEDS = range(6000, 6020)
K = 6


def load(name):
    out = []
    for s in SEEDS:
        p = CACHE / f"{name}_{s}.json"
        if p.exists():
            out.append(json.loads(p.read_text()))
    return out


def verdict_atleast(hits, n, need=18, total=20):
    """'hits >= need of total' with total-n seeds missing."""
    if hits >= need:
        return "PASS" if n == total else f"PASS (determined: {hits} >= {need} already)"
    if hits + (total - n) < need:
        return (
            "FAIL"
            if n == total
            else f"FAIL (determined: at most {hits + total - n}/{total} reachable)"
        )
    return "INCOMPLETE"


def verdict_atmost(bad, n, allow=1, total=20):
    if bad > allow:
        return "FAIL" if n == total else f"FAIL (determined: {bad} > {allow} already)"
    if n == total:
        return "PASS"
    return "INCOMPLETE"


recs = {n: load(n) for n in ("S1", "S2", "S3", "S4", "S5")}
L = [
    "FIXES2 validation results (bars in docs/theory/FIXES2_PLAN.md, frozen at cc699b2)",
    "EARLY STOP (2026-10-14, user request; see the deviation log in FIXES2_PLAN.md).",
    "Scored from the frozen runner's per-seed cache with its own per-seed hit definitions.",
    "k/n = hits over completed seeds; bars need 20 seeds. INCOMPLETE = not determined.",
    "Completed seeds: " + ", ".join(f"{k} {len(v)}/20" for k, v in recs.items()),
    "",
    "Bars",
]


def flag_count(name, pred):
    rows = recs[name]
    hits = sum(bool(pred(r)) for r in rows)
    failed = [r["seed"] for r in rows if not pred(r)]
    return hits, len(rows), failed


h, n, f = flag_count("S1", lambda r: r["share_hit"])
L.append(f"B1  {h}/{n}  failed_seeds={f or '-'}  {verdict_atleast(h, n)}")
h, n, f = flag_count("S2", lambda r: r["share_hit"])
L.append(f"B2  {h}/{n}  failed_seeds={f or '-'}  {verdict_atleast(h, n)}")
labels = [f"comp k={k}" for k in range(3)] + [f"struct k={k}" for k in range(3)]
rows, vs = [], []
for i, lab in enumerate(labels):
    h, n, f = flag_count("S3", lambda r, i=i: r["b3_hit"][i])
    v = verdict_atleast(h, n)
    vs.append(v)
    rows.append(f"    {lab}  {h}/{n}  failed_seeds={f or '-'}  {v}")


def combine(vs):
    if any(v.startswith("FAIL") for v in vs):
        return "FAIL (a component is determined to fail)"
    if all(v == "PASS" for v in vs):
        return "PASS"
    return "INCOMPLETE"


L.append(f"B3  {combine(vs)}")
L += rows
rows, vs = [], []
for i in range(3):
    h, n, f = flag_count("S3", lambda r, i=i: r["b4_hit"][i])
    v = verdict_atleast(h, n)
    vs.append(v)
    rows.append(f"    feature {i + 1}  {h}/{n}  failed_seeds={f or '-'}  {v}")
L.append(f"B4  {combine(vs)}")
L += rows
L.append("B5  0/20 S4 seeds run  INCOMPLETE")
L.append("B5b calibration (seeds 6100-6499) not run  INCOMPLETE")
L.append("B6  0/20 S4 seeds run  INCOMPLETE")
for bar, key in (("B7", "order_gap"), ("B8", "player_sum_gap")):
    bad = [f"{nm}:{r['seed']}" for nm, rs in recs.items() for r in rs if r[key] > 1e-9]
    done = sum(len(rs) for rs in recs.values())
    L.append(
        f"B7/B8 {bar}  {done - len(bad)}/{done} completed seeds within 1e-9 (bar needs all 100)  "
        f"failed={bad or '-'}  {'FAIL' if bad else 'INCOMPLETE (no failure so far)'}"
    )
L.append("B9  0/20 S5 seeds run  INCOMPLETE")
proc = subprocess.run(
    [sys.executable, str(ROOT / "shift_v2_checks.py")], capture_output=True, text=True, check=False
)
checks = proc.stdout + proc.stderr
ok10 = "ALL PASS" in checks and "SOME FAILED" not in checks
L.append(f"B10 {'PASS' if ok10 else 'FAIL'}")
sp = Path(ROOT.parents[2] / "benchmarks" / "results" / "shift_v2_speed_fixes2.json")
L.append(
    "B11 speed: "
    + (sp.read_text().replace("\n", " ") if sp.exists() else "not available")
    + " (8 threads, see deviation log; not a correctness gate)"
)
L += ["", "Per seed"]
for _nm, rs in recs.items():
    for r in rs:
        L.append(json.dumps({k: r[k] for k in r if k not in ("b3_hat",)}))
L += ["", "B10 verify/shift_v2_checks.py", checks.rstrip(), ""]
OUT.write_text("\n".join(L) + "\n")
print("\n".join(L[:30]))
