"""Guards for the V5.2c specified-family replication runner.

Does not run the study and does not touch any gate seed. Pre-flight and
rehearsal are tested with mocks, writing only to a temporary directory.
The spawn-safe deadlock regression is in test_spawn_safe_harness.py.
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

import run_v5_2c_specified_gate as gate  # noqa: E402
from summarize_v5_2_specified_gate import after_holm  # noqa: E402
from summarize_v5_2c_specified_gate import after_holm as after_holm_c  # noqa: E402


def test_allowed_ranges_and_forbidden_include_sealed_spent_and_buffers():
    assert gate.ALLOWED == set(range(13000, 13050)) | set(range(13100, 13120))
    assert set(range(9600, 9620)) <= gate.FORBIDDEN
    assert set(range(10400, 10800)) <= gate.FORBIDDEN
    assert set(range(11000, 11050)) <= gate.FORBIDDEN
    assert set(range(11100, 11120)) <= gate.FORBIDDEN
    assert set(range(11120, 12000)) <= gate.FORBIDDEN
    assert set(range(12000, 12050)) <= gate.FORBIDDEN
    assert set(range(12100, 12120)) <= gate.FORBIDDEN
    assert set(range(12120, 13000)) <= gate.FORBIDDEN
    assert set(range(13050, 13100)) <= gate.FORBIDDEN
    assert gate.ALLOWED.isdisjoint(gate.FORBIDDEN)
    assert gate.SEALED == set(range(9600, 9620))
    assert gate.COVERAGE_SPENT == set(range(10400, 10800))
    assert gate.SPECIFIED_SPENT == set(range(11000, 11050)) | set(range(11100, 11120))
    assert gate.V52B_SPENT == set(range(12000, 12050)) | set(range(12100, 12120))
    banned = (
        9600,
        9619,
        10400,
        10799,
        11000,
        11049,
        11100,
        11119,
        11120,
        11999,
        12000,
        12049,
        12100,
        12119,
        12120,
        12999,
        13050,
        0,
        9700,
        10100,
    )
    for i in banned:
        with pytest.raises(AssertionError):
            n = 2000 if i < 13100 else 20000
            gate.assert_rep(i, n)
    gate.assert_rep(13000, 2000)
    gate.assert_rep(13049, 2000)
    gate.assert_rep(13100, 20000)
    gate.assert_rep(13119, 20000)
    with pytest.raises(AssertionError):
        gate.assert_rep(13000, 20000)
    with pytest.raises(AssertionError):
        gate.assert_rep(13100, 2000)


def test_smoke_and_rehearsal_seeds_are_non_gate_and_not_sealed():
    assert gate.SMOKE_SEED == 99
    assert gate.REHEARSAL_SEEDS == (98, 97)
    for s in (gate.SMOKE_SEED, *gate.REHEARSAL_SEEDS):
        assert s not in gate.ALLOWED
        assert s not in gate.SEALED
        assert s not in gate.COVERAGE_SPENT
        assert s not in gate.SPECIFIED_SPENT
        assert s not in gate.V52B_SPENT
        gate.assert_rehearsal_rep(s) if s != 99 else None
    with pytest.raises(AssertionError):
        gate.assert_rehearsal_rep(13000)
    with pytest.raises(AssertionError):
        gate.assert_rehearsal_rep(96)


def test_single_pass_constants():
    assert gate.CAP_SEC == 30 * 60
    assert gate.WATCHDOG_SEC == 5 * 60
    assert gate.KAPPA == 1.0
    assert gate.OUT.name == "v5_2c_specified_gate_shape.jsonl"
    assert gate.REHEARSAL_OUT.name == "v5_2c_specified_gate_rehearsal.jsonl"
    assert gate.PREFLIGHT_OUT.name == "v5_2c_specified_gate_preflight.json"
    assert len(gate.SIZES[2000][1]) == 50
    assert len(gate.SIZES[20000][1]) == 20
    assert 2 * (50 + 20) == 140
    assert gate.PREREG_SHA == "f022ba3e42ebe4b7cee1dbdf2bf1c254b248b1f6"
    assert gate.IMPL_SHA == "1aab4778abf8f84c8c8f7d9603eb8c322f1ba3c9"
    todo, skipped = gate.plan()
    assert skipped == []
    assert len(todo) == 140
    reps = {t[3] for t in todo}
    assert reps == gate.ALLOWED
    r_todo = gate.rehearsal_plan()
    assert len(r_todo) == 8
    assert {t[3] for t in r_todo} == {98, 97}
    assert {t[0] for t in r_todo} == {"skew_comp", "tail_comp"}
    assert {t[2] for t in r_todo} == {2000, 20000}


def test_scoring_helpers_are_the_parent_helpers():
    assert after_holm_c is after_holm
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


def _patch_paths(g, tmp_path, monkeypatch):
    monkeypatch.setattr(g, "RES", tmp_path)
    monkeypatch.setattr(g, "OUT", tmp_path / "v5_2c_specified_gate_shape.jsonl")
    monkeypatch.setattr(g, "REHEARSAL_OUT", tmp_path / "v5_2c_specified_gate_rehearsal.jsonl")
    monkeypatch.setattr(g, "WALL", tmp_path / "v5_2c_specified_gate_wall.json")
    monkeypatch.setattr(g, "PREFLIGHT_OUT", tmp_path / "v5_2c_specified_gate_preflight.json")


def test_main_aborts_before_seeds_on_failed_preflight(tmp_path, monkeypatch):
    import run_v5_2c_specified_gate as g
    import spawn_safe

    _patch_paths(g, tmp_path, monkeypatch)
    monkeypatch.setattr(
        g,
        "run_preflight_spawned",
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

    def _boom(*a, **k):
        pool_started["n"] += 1
        raise AssertionError("Pool must not start after a pre-flight abort")

    monkeypatch.setattr(g.spawn_safe, "spawn_pool", _boom)
    monkeypatch.setattr(spawn_safe, "spawn_pool", _boom)
    monkeypatch.setattr(sys, "argv", ["run_v5_2c_specified_gate.py", "--workers", "2"])
    g.main()
    assert pool_started["n"] == 0
    assert not g.OUT.exists()
    assert not g.REHEARSAL_OUT.exists()
    wall = json.loads(g.WALL.read_text())
    assert wall["stop"] == "preflight_abort"
    assert wall["tasks_done"] == 0
    assert wall["prereg_sha"] == g.PREREG_SHA
    pf = json.loads(g.PREFLIGHT_OUT.read_text())
    assert pf["ok"] is False
    assert "sklearn" in pf["error"]


def test_main_aborts_before_seeds_on_failed_rehearsal(tmp_path, monkeypatch):
    import run_v5_2c_specified_gate as g

    _patch_paths(g, tmp_path, monkeypatch)
    monkeypatch.setattr(
        g,
        "run_preflight_spawned",
        lambda: {
            "ok": True,
            "versions": {"traceot": "0.1.0", "scikit-learn": "1.9.1"},
            "smoke": {"TRACE": "ok", "DFL": "ok", "CFM": "ok"},
        },
    )

    class _FakePool:
        def imap_unordered(self, fn, todo):
            raise AssertionError("drain is mocked")

        def terminate(self):
            pass

        def join(self):
            pass

    monkeypatch.setattr(g.spawn_safe, "spawn_pool", lambda *a, **k: _FakePool())
    monkeypatch.setattr(
        g,
        "drain_pool",
        lambda *a, **k: (0, 0.0, "watchdog_abort"),
    )
    # drain_pool is mocked so it will not write the rehearsal file; rehearsal_ok
    # then fails. Gate pool must not start: drain_pool is called at most once.
    calls = {"n": 0}

    def counted(*a, **k):
        calls["n"] += 1
        if calls["n"] > 1:
            raise AssertionError("gate drain_pool must not run after rehearsal abort")
        return (0, 1.0, "watchdog_abort")

    monkeypatch.setattr(g, "drain_pool", counted)
    monkeypatch.setattr(sys, "argv", ["run_v5_2c_specified_gate.py", "--workers", "2"])
    g.main()
    assert calls["n"] == 1
    assert not g.OUT.exists()
    wall = json.loads(g.WALL.read_text())
    assert wall["stop"] == "watchdog_abort"
    assert wall["tasks_done"] == 0
    assert (
        "rehearsal" in wall.get("rehearsal_error", "").lower() or wall["stop"] == "watchdog_abort"
    )
    assert wall.get("rehearsal_ok") is not True


def test_main_does_not_run_when_results_file_exists(tmp_path, monkeypatch):
    import run_v5_2c_specified_gate as g

    out = tmp_path / "v5_2c_specified_gate_shape.jsonl"
    out.write_text("{}\n")
    monkeypatch.setattr(g, "OUT", out)
    monkeypatch.setattr(sys, "argv", ["run_v5_2c_specified_gate.py"])
    with pytest.raises(AssertionError, match="single pass"):
        g.main()


def test_summariser_rehearsal_abort_leaves_h_undecided(tmp_path, monkeypatch):
    import summarize_v5_2c_specified_gate as sm

    (tmp_path / "v5_2c_specified_gate_wall.json").write_text(
        json.dumps(
            {
                "wall_sec": 0.0,
                "tasks_done": 0,
                "tasks_total": 0,
                "tasks_preregistered": 140,
                "stop": "rehearsal_abort",
                "rehearsal_error": "DFL failed on rehearsal",
                "rehearsal_tasks_done": 3,
                "rehearsal_tasks_total": 8,
                "cap_sec": 1800,
                "prereg_sha": sm.PREREG_SHA,
            }
        )
    )
    (tmp_path / "v5_2c_specified_gate_preflight.json").write_text(
        json.dumps({"ok": True, "versions": {"scikit-learn": "1.9.1"}})
    )
    monkeypatch.setattr(sm, "RES", tmp_path)
    monkeypatch.setattr(sm, "OUT_MD", tmp_path / "results.md")
    monkeypatch.setattr(sm, "OUT_JSON", tmp_path / "summary.json")
    monkeypatch.setattr(sm, "JSONL", tmp_path / "v5_2c_specified_gate_shape.jsonl")
    monkeypatch.setattr(sm, "WALL_PATH", tmp_path / "v5_2c_specified_gate_wall.json")
    monkeypatch.setattr(sm, "PREFLIGHT_PATH", tmp_path / "v5_2c_specified_gate_preflight.json")
    sm.main()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["overall"] == "ABORTED (rehearsal)"
    assert not (tmp_path / "v5_2c_specified_gate_shape.jsonl").exists()
    md = (tmp_path / "results.md").read_text()
    assert "H is not decided" in md
    assert "not touched" in md or "remain unspent" in md
    verdict_line = [ln for ln in md.splitlines() if ln.startswith("**Gate verdict:**")][0]
    assert "PASS" not in verdict_line
    assert "FAIL" not in verdict_line
    assert "ABORTED" in verdict_line


def test_summariser_watchdog_abort_leaves_h_undecided(tmp_path, monkeypatch):
    import summarize_v5_2c_specified_gate as sm

    (tmp_path / "v5_2c_specified_gate_wall.json").write_text(
        json.dumps(
            {
                "wall_sec": 300.0,
                "tasks_done": 0,
                "tasks_total": 140,
                "tasks_preregistered": 140,
                "stop": "watchdog_abort",
                "rehearsal_ok": True,
                "cap_sec": 1800,
                "watchdog_sec": 300,
                "prereg_sha": sm.PREREG_SHA,
            }
        )
    )
    (tmp_path / "v5_2c_specified_gate_preflight.json").write_text(
        json.dumps({"ok": True, "versions": {"scikit-learn": "1.9.1"}})
    )
    (tmp_path / "v5_2c_specified_gate_shape.jsonl").write_text("")
    monkeypatch.setattr(sm, "RES", tmp_path)
    monkeypatch.setattr(sm, "OUT_MD", tmp_path / "results.md")
    monkeypatch.setattr(sm, "OUT_JSON", tmp_path / "summary.json")
    monkeypatch.setattr(sm, "JSONL", tmp_path / "v5_2c_specified_gate_shape.jsonl")
    monkeypatch.setattr(sm, "WALL_PATH", tmp_path / "v5_2c_specified_gate_wall.json")
    monkeypatch.setattr(sm, "PREFLIGHT_PATH", tmp_path / "v5_2c_specified_gate_preflight.json")
    sm.main()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["overall"] == "ABORTED (watchdog)"
    md = (tmp_path / "results.md").read_text()
    assert "H is not decided" in md
    assert "spent" in md.lower()
    verdict_line = [ln for ln in md.splitlines() if ln.startswith("**Gate verdict:**")][0]
    assert "PASS" not in verdict_line
    assert "FAIL" not in verdict_line


def test_summariser_preflight_abort_leaves_h_undecided(tmp_path, monkeypatch):
    import summarize_v5_2c_specified_gate as sm

    (tmp_path / "v5_2c_specified_gate_wall.json").write_text(
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
    (tmp_path / "v5_2c_specified_gate_preflight.json").write_text(
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
    monkeypatch.setattr(sm, "JSONL", tmp_path / "v5_2c_specified_gate_shape.jsonl")
    monkeypatch.setattr(sm, "WALL_PATH", tmp_path / "v5_2c_specified_gate_wall.json")
    monkeypatch.setattr(sm, "PREFLIGHT_PATH", tmp_path / "v5_2c_specified_gate_preflight.json")
    sm.main()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["overall"] == "INCOMPLETE (pre-flight abort)"
    assert not (tmp_path / "v5_2c_specified_gate_shape.jsonl").exists()
    md = (tmp_path / "results.md").read_text()
    assert "H is not decided" in md
    verdict_line = [ln for ln in md.splitlines() if ln.startswith("**Gate verdict:**")][0]
    assert "PASS" not in verdict_line
    assert "FAIL" not in verdict_line


def test_collect_versions_records_sklearn_failure_without_raising():
    versions = gate.collect_versions()
    assert "python" in versions
    assert "numpy" in versions
    assert "scikit-learn" in versions
    try:
        import sklearn  # noqa: F401
    except ImportError:
        assert versions["scikit-learn"].startswith("IMPORT_FAILED")


def test_preflight_aborts_on_missing_sklearn_without_touching_seeds(monkeypatch):
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
    assert rec.get("smoke", {}).get("TRACE") != "ok"
    assert rec.get("smoke", {}).get("DFL") != "ok"
    assert rec.get("smoke", {}).get("CFM") != "ok"


def test_watchdog_timer_is_five_minutes():
    assert gate.WATCHDOG_SEC == 300


def test_rehearsal_ok_rejects_method_error(tmp_path):
    path = tmp_path / "reh.jsonl"
    good = {
        "scen": "skew_comp",
        "n": 2000,
        "rep": 98,
        "gate": False,
        "TRACE_V5_2_SPECIFIED": {"lsplit": {"comp": [0], "struct": [0]}},
        "rivals": {"DFL": {"lsplit": {}}, "CFM": {"lsplit": {}}},
    }
    bad = dict(good)
    bad = {
        **good,
        "rep": 97,
        "rivals": {"DFL": {"error": "boom"}, "CFM": {"lsplit": {}}},
    }
    path.write_text(json.dumps(good) + "\n" + json.dumps(bad) + "\n")
    ok, reason = gate.rehearsal_ok(path, 2, 2, None)
    assert ok is False
    assert "DFL" in reason
