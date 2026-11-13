"""Guards for the V5.2b specified-family replication runner.

Does not run the study and does not touch any gate seed. Pre-flight is tested
with mocks, and (when scikit-learn is absent) via the real abort path, writing
only to a temporary directory.
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

import run_v5_2b_specified_gate as gate  # noqa: E402
from summarize_v5_2_specified_gate import after_holm  # noqa: E402
from summarize_v5_2b_specified_gate import after_holm as after_holm_b  # noqa: E402


def test_allowed_ranges_and_forbidden_include_sealed_spent_and_buffers():
    assert gate.ALLOWED == set(range(12000, 12050)) | set(range(12100, 12120))
    assert set(range(9600, 9620)) <= gate.FORBIDDEN
    assert set(range(10400, 10800)) <= gate.FORBIDDEN
    assert set(range(11000, 11050)) <= gate.FORBIDDEN
    assert set(range(11100, 11120)) <= gate.FORBIDDEN
    assert set(range(11120, 12000)) <= gate.FORBIDDEN
    assert gate.ALLOWED.isdisjoint(gate.FORBIDDEN)
    assert gate.SEALED == set(range(9600, 9620))
    assert gate.COVERAGE_SPENT == set(range(10400, 10800))
    assert gate.SPECIFIED_SPENT == set(range(11000, 11050)) | set(range(11100, 11120))
    for i in (9600, 9619, 10400, 10799, 11000, 11049, 11100, 11119, 11120, 11999, 9700, 10100, 0):
        with pytest.raises(AssertionError):
            n = 2000 if i < 12100 else 20000
            gate.assert_rep(i, n)
    gate.assert_rep(12000, 2000)
    gate.assert_rep(12049, 2000)
    gate.assert_rep(12100, 20000)
    gate.assert_rep(12119, 20000)
    with pytest.raises(AssertionError):
        gate.assert_rep(12000, 20000)  # wrong size
    with pytest.raises(AssertionError):
        gate.assert_rep(12100, 2000)


def test_smoke_seed_is_non_gate_and_not_sealed():
    assert gate.SMOKE_SEED == 99
    assert gate.SMOKE_SEED not in gate.ALLOWED
    assert gate.SMOKE_SEED not in gate.SEALED
    assert gate.SMOKE_SEED not in gate.COVERAGE_SPENT
    assert gate.SMOKE_SEED not in gate.SPECIFIED_SPENT
    assert gate.SMOKE_SEED not in gate.FORBIDDEN or gate.SMOKE_SEED < 2000
    # 99 sits in 0-1999, which is FORBIDDEN for gate indices; that is fine,
    # because the smoke path does not go through assert_rep.


def test_single_pass_constants():
    assert gate.CAP_SEC == 30 * 60
    assert gate.KAPPA == 1.0
    assert gate.OUT.name == "v5_2b_specified_gate_shape.jsonl"
    assert gate.PREFLIGHT_OUT.name == "v5_2b_specified_gate_preflight.json"
    assert len(gate.SIZES[2000][1]) == 50
    assert len(gate.SIZES[20000][1]) == 20
    assert 2 * (50 + 20) == 140
    assert gate.PREREG_SHA == "19ed3202c701a5ee924399885265b1996d4a13cc"
    assert gate.IMPL_SHA == "1aab4778abf8f84c8c8f7d9603eb8c322f1ba3c9"
    # plan() only inspects calibration; it does not run a gate index.
    todo, skipped = gate.plan()
    assert skipped == []
    assert len(todo) == 140
    reps = {t[3] for t in todo}
    assert reps == gate.ALLOWED


def test_scoring_helpers_are_the_parent_helpers():
    # Byte-identical scoring: the replication imports the parent's after_holm.
    assert after_holm_b is after_holm
    assert after_holm("BEST", hi=-0.01, delta=0.1, p_best_holm=0.01, p_ni_holm=0.001) == "BEST"
    assert (
        after_holm("BEST", hi=-0.01, delta=0.1, p_best_holm=0.06, p_ni_holm=0.01) == "NON-INFERIOR"
    )
    assert (
        after_holm("NON-INFERIOR", hi=0.05, delta=0.1, p_best_holm=0.2, p_ni_holm=0.01)
        == "NON-INFERIOR"
    )
    assert after_holm("NON-INFERIOR", hi=0.05, delta=0.1, p_best_holm=0.2, p_ni_holm=0.06) == "FAIL"
    assert after_holm("FAIL", hi=0.2, delta=0.1, p_best_holm=0.001, p_ni_holm=0.001) == "FAIL"


def test_main_aborts_before_seeds_on_failed_preflight(tmp_path, monkeypatch):
    """A failed pre-flight must not create the official results file or start a pool."""

    import run_v5_2b_specified_gate as g

    monkeypatch.setattr(g, "RES", tmp_path)
    monkeypatch.setattr(g, "OUT", tmp_path / "v5_2b_specified_gate_shape.jsonl")
    monkeypatch.setattr(g, "WALL", tmp_path / "v5_2b_specified_gate_wall.json")
    monkeypatch.setattr(g, "PREFLIGHT_OUT", tmp_path / "v5_2b_specified_gate_preflight.json")
    monkeypatch.setattr(
        g,
        "preflight",
        lambda: {
            "ok": False,
            "error": "ModuleNotFoundError: No module named 'sklearn'",
            "versions": {"scikit-learn": "IMPORT_FAILED"},
            "imports": {},
            "smoke": {},
            "smoke_seed": 99,
        },
    )
    pool_started = {"n": 0}

    class _BoomPool:
        def __init__(self, *a, **k):
            pool_started["n"] += 1
            raise AssertionError("Pool must not start after a pre-flight abort")

        def terminate(self):
            pass

    monkeypatch.setattr(g, "Pool", _BoomPool)
    monkeypatch.setattr(sys, "argv", ["run_v5_2b_specified_gate.py", "--workers", "2"])
    g.main()
    assert pool_started["n"] == 0
    assert not g.OUT.exists()
    wall = json.loads(g.WALL.read_text())
    assert wall["stop"] == "preflight_abort"
    assert wall["tasks_done"] == 0
    assert wall["prereg_sha"] == g.PREREG_SHA
    pf = json.loads(g.PREFLIGHT_OUT.read_text())
    assert pf["ok"] is False
    assert "sklearn" in pf["error"]


def test_main_does_not_run_when_results_file_exists(tmp_path, monkeypatch):
    import run_v5_2b_specified_gate as g

    out = tmp_path / "v5_2b_specified_gate_shape.jsonl"
    out.write_text("{}\n")
    monkeypatch.setattr(g, "OUT", out)
    monkeypatch.setattr(sys, "argv", ["run_v5_2b_specified_gate.py"])
    with pytest.raises(AssertionError, match="single pass"):
        g.main()


def test_summariser_preflight_abort_leaves_h_undecided(tmp_path, monkeypatch):
    import summarize_v5_2b_specified_gate as sm

    (tmp_path / "v5_2b_specified_gate_wall.json").write_text(
        json.dumps(
            {
                "wall_sec": 0.0,
                "tasks_done": 0,
                "tasks_total": 0,
                "tasks_preregistered": 140,
                "stop": "preflight_abort",
                "preflight_error": "ModuleNotFoundError: No module named 'sklearn'",
                "cap_sec": 1800,
                "prereg_sha": sm.PREREG_SHA,
            }
        )
    )
    (tmp_path / "v5_2b_specified_gate_preflight.json").write_text(
        json.dumps(
            {
                "ok": False,
                "error": "ModuleNotFoundError: No module named 'sklearn'",
                "versions": {"scikit-learn": "IMPORT_FAILED: ModuleNotFoundError"},
            }
        )
    )
    monkeypatch.setattr(sm, "RES", tmp_path)
    monkeypatch.setattr(sm, "OUT_MD", tmp_path / "results.md")
    monkeypatch.setattr(sm, "OUT_JSON", tmp_path / "summary.json")
    monkeypatch.setattr(sm, "JSONL", tmp_path / "v5_2b_specified_gate_shape.jsonl")
    monkeypatch.setattr(sm, "WALL_PATH", tmp_path / "v5_2b_specified_gate_wall.json")
    monkeypatch.setattr(sm, "PREFLIGHT_PATH", tmp_path / "v5_2b_specified_gate_preflight.json")
    sm.main()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["overall"] == "INCOMPLETE (pre-flight abort)"
    assert not (tmp_path / "v5_2b_specified_gate_shape.jsonl").exists()
    md = (tmp_path / "results.md").read_text()
    assert "H is not decided" in md
    assert "not touched" in md
    verdict_line = [ln for ln in md.splitlines() if ln.startswith("**Gate verdict:**")][0]
    assert "PASS" not in verdict_line
    assert "FAIL" not in verdict_line


def test_collect_versions_records_sklearn_failure_without_raising():
    versions = gate.collect_versions()
    assert "python" in versions
    assert "numpy" in versions
    assert "scikit-learn" in versions
    # CI does not install scikit-learn; a missing install must be recorded, not raised.
    try:
        import sklearn  # noqa: F401
    except ImportError:
        assert versions["scikit-learn"].startswith("IMPORT_FAILED")


def test_preflight_aborts_on_missing_sklearn_without_touching_seeds(monkeypatch):
    """The failure mode of the V5.2 specified pass: sklearn missing must abort.

    Forced even when sklearn is installed, so the test is deterministic on CI
    and on a local machine that has the package. Does not write official files
    and does not call assert_rep on any index.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sklearn" or name.startswith("sklearn."):
            raise ModuleNotFoundError("No module named 'sklearn'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    rec = gate.preflight()
    assert rec["ok"] is False
    assert "sklearn" in rec["error"]
    assert rec["smoke_seed"] == 99
    # smoke fits must not be recorded as ok
    assert rec.get("smoke", {}).get("TRACE") != "ok"
    assert rec.get("smoke", {}).get("DFL") != "ok"
    assert rec.get("smoke", {}).get("CFM") != "ok"
