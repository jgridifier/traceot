"""Guards for the V5.1 structure/mixed-cell gate runner.

Does not run the study and does not touch any gate seed. Pre-flight and
rehearsal are tested with mocks, writing only to a temporary directory.
The spawn-safe deadlock regression is in test_spawn_safe_harness.py.
The live 665939b bit-identity check is a pre-flight step, not a CI test.
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

import run_v5_1_struct_gate as gate  # noqa: E402
from summarize_v5_1_struct_gate import FROZEN_DELTA, false_attr_verdict  # noqa: E402
from summarize_v5_1_struct_gate import after_holm as after_holm_struct  # noqa: E402
from summarize_v5_2_specified_gate import after_holm  # noqa: E402


def test_allowed_ranges_and_forbidden_include_sealed_spent_and_buffers():
    assert gate.ALLOWED == set(range(14000, 14050)) | set(range(14100, 14120))
    assert set(range(9600, 9620)) <= gate.FORBIDDEN
    assert set(range(10400, 10800)) <= gate.FORBIDDEN
    assert set(range(13000, 13050)) <= gate.FORBIDDEN
    assert set(range(13100, 13120)) <= gate.FORBIDDEN
    assert set(range(13120, 14000)) <= gate.FORBIDDEN
    assert set(range(14050, 14100)) <= gate.FORBIDDEN
    assert gate.ALLOWED.isdisjoint(gate.FORBIDDEN)
    assert gate.SEALED == set(range(9600, 9620))
    assert gate.COVERAGE_SPENT == set(range(10400, 10800))
    assert gate.SEALED <= gate.FORBIDDEN
    assert gate.COVERAGE_SPENT <= gate.FORBIDDEN
    banned = (
        9600,
        9619,
        10400,
        10799,
        13000,
        13119,
        13120,
        13999,
        14050,
        14099,
        0,
        9700,
        10100,
        12000,
        99,
        97,
        98,
    )
    for i in banned:
        with pytest.raises(AssertionError):
            n = 2000 if i < 14100 else 20000
            gate.assert_rep(i, n)
    gate.assert_rep(14000, 2000)
    gate.assert_rep(14049, 2000)
    gate.assert_rep(14100, 20000)
    gate.assert_rep(14119, 20000)
    with pytest.raises(AssertionError):
        gate.assert_rep(14000, 20000)
    with pytest.raises(AssertionError):
        gate.assert_rep(14100, 2000)


def test_smoke_and_rehearsal_seeds_are_non_gate_and_not_sealed():
    assert gate.SMOKE_SEED == 99
    assert gate.REHEARSAL_SEEDS == (98, 97)
    for s in (gate.SMOKE_SEED, *gate.REHEARSAL_SEEDS):
        assert s not in gate.ALLOWED
        assert s not in gate.SEALED
        assert s not in gate.COVERAGE_SPENT
        if s != 99:
            gate.assert_rehearsal_rep(s)
    with pytest.raises(AssertionError):
        gate.assert_rehearsal_rep(14000)
    with pytest.raises(AssertionError):
        gate.assert_rehearsal_rep(96)


def test_single_pass_constants_and_plan():
    assert gate.CAP_SEC == 30 * 60
    assert gate.WATCHDOG_SEC == 5 * 60
    assert gate.N_TASKS == 210
    assert gate.OUT.name == "v5_1_struct_gate_shape.jsonl"
    assert gate.REHEARSAL_OUT.name == "v5_1_struct_gate_rehearsal.jsonl"
    assert gate.PREFLIGHT_OUT.name == "v5_1_struct_gate_preflight.json"
    assert len(gate.SIZES[2000][1]) == 50
    assert len(gate.SIZES[20000][1]) == 20
    assert 3 * (50 + 20) == 210
    assert gate.PREREG_SHA == "85a8fccfac44b6a1506037d785d4be6cf56c3bdb"
    assert gate.IMPL_SHA == "665939bd4aa78ffc7474b91d482d18ac0191d595"
    assert len(gate.CORE_COMMITS_AFTER_IMPL) == 4
    todo = gate.plan()
    assert len(todo) == 210
    assert {t[1] for t in todo} == {2000, 20000}
    assert {t[0] for t in todo} == {"skew_struct", "tail_struct", "shape_mixed"}
    reps = {t[2] for t in todo}
    assert reps == gate.ALLOWED
    # Large cells first.
    assert todo[0][1] == 20000
    assert todo[-1][1] == 2000
    r_todo = gate.rehearsal_plan()
    assert len(r_todo) == 12
    assert {t[2] for t in r_todo} == {98, 97}
    assert {t[0] for t in r_todo} == {"skew_struct", "tail_struct", "shape_mixed"}
    assert {t[1] for t in r_todo} == {2000, 20000}


def test_frozen_deltas_match_the_prereg_table():
    from summarize_v4 import planted_mag

    truth = json.loads(
        (ROOT / "benchmarks" / "results" / "shiftbench_truth_shape.json").read_text()
    )
    for scen, frozen in FROZEN_DELTA.items():
        sh, k = truth[scen]["shapley"], truth[scen]["graded"][0]
        delta = 0.10 * planted_mag(sh, k)
        assert abs(delta - frozen) / frozen < 5e-4, (scen, delta, frozen)


def test_scoring_helpers_are_the_parent_helpers():
    assert after_holm_struct is after_holm
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


def test_false_attr_requires_bound_and_holm_p():
    assert false_attr_verdict(0.001, 0.005, 0.01) == "PASS"
    assert false_attr_verdict(0.006, 0.005, 0.01) == "FAIL"
    assert false_attr_verdict(0.001, 0.005, 0.06) == "FAIL"
    assert false_attr_verdict(0.005, 0.005, 0.01) == "FAIL"  # bound must be strictly < δ


def test_method_check_fixture_is_well_formed():
    ref = json.loads(gate.METHOD_CHECK_REF.read_text())
    assert ref["bit_identical"] is True
    assert ref["impl_sha"] == gate.IMPL_SHA
    assert ref["draw"]["seed"] == 99
    assert ref["draw"]["scenario"] == "skew_struct"
    assert ref["draw"]["centred"] is True
    assert len(ref["core_commits_after_impl_touching_crates_core"]) == 4
    assert gate.METHOD_CHECK_INPUTS.exists()
    assert set(ref["draw"]["input_sha256"]) == {"XA", "yA", "XB", "yB"}
    assert "lsplit_comp" in ref["reference"]
    assert "lsplit_struct" in ref["reference"]


def _patch_paths(g, tmp_path, monkeypatch):
    monkeypatch.setattr(g, "RES", tmp_path)
    monkeypatch.setattr(g, "OUT", tmp_path / "v5_1_struct_gate_shape.jsonl")
    monkeypatch.setattr(g, "REHEARSAL_OUT", tmp_path / "v5_1_struct_gate_rehearsal.jsonl")
    monkeypatch.setattr(g, "WALL", tmp_path / "v5_1_struct_gate_wall.json")
    monkeypatch.setattr(g, "PREFLIGHT_OUT", tmp_path / "v5_1_struct_gate_preflight.json")
    monkeypatch.setattr(g.spawn_safe, "assert_parent_has_not_called_trace", lambda: None)


def _ok_preflight():
    return {
        "ok": True,
        "versions": {"traceot": "0.1.0", "scikit-learn": "1.9.1", "numpy": "2.5.5"},
        "smoke": {"TRACE": "ok", "DFL": "ok", "CFM": "ok"},
        "method_identity": {"ok": True},
        "smoke_seed": 99,
    }


def test_main_without_confirmatory_does_not_touch_gate_seeds(tmp_path, monkeypatch):
    import run_v5_1_struct_gate as g

    _patch_paths(g, tmp_path, monkeypatch)
    monkeypatch.setattr(g, "run_preflight_spawned", lambda: _ok_preflight())

    class _FakePool:
        def imap_unordered(self, fn, todo):
            raise AssertionError("drain is mocked")

        def terminate(self):
            pass

        def join(self):
            pass

    monkeypatch.setattr(g.spawn_safe, "spawn_pool", lambda *a, **k: _FakePool())
    calls = {"n": 0}

    def counted(pool, fn, todo, out_path, *, cap_sec, watchdog_sec):
        calls["n"] += 1
        assert fn is g.rehearsal_task
        assert all(t[2] in (97, 98) for t in todo)
        # Write a complete rehearsal file so rehearsal_ok passes.
        recs = []
        for scen, n, rep in todo:
            recs.append(
                {
                    "scen": scen,
                    "n": n,
                    "rep": rep,
                    "gate": False,
                    "TRACE_V5_1": {"lsplit": {"comp": [0], "struct": [0]}},
                    "rivals": {"DFL": {"lsplit": {}}, "CFM": {"lsplit": {}}},
                }
            )
        out_path.write_text("".join(json.dumps(r) + "\n" for r in recs))
        return len(recs), 1.2, None

    monkeypatch.setattr(g, "drain_pool", counted)
    monkeypatch.setattr(sys, "argv", ["run_v5_1_struct_gate.py", "--workers", "2"])
    g.main()
    assert calls["n"] == 1
    assert not g.OUT.exists()
    wall = json.loads(g.WALL.read_text())
    assert wall["stop"] == "awaiting_confirmatory"
    assert wall["tasks_done"] == 0
    assert wall.get("confirmatory") is not True
    assert wall["rehearsal_ok"] is True


def test_main_confirmatory_starts_gate_after_ok_preflight_and_rehearsal(tmp_path, monkeypatch):
    import run_v5_1_struct_gate as g

    _patch_paths(g, tmp_path, monkeypatch)
    # Existing OK pre-flight and rehearsal: confirmatory should not re-run them,
    # only start the gate drain.
    g.PREFLIGHT_OUT.write_text(json.dumps(_ok_preflight()))
    r_todo = g.rehearsal_plan()
    recs = [
        {
            "scen": scen,
            "n": n,
            "rep": rep,
            "gate": False,
            "TRACE_V5_1": {"lsplit": {"comp": [0], "struct": [0]}},
            "rivals": {"DFL": {"lsplit": {}}, "CFM": {"lsplit": {}}},
        }
        for scen, n, rep in r_todo
    ]
    g.REHEARSAL_OUT.write_text("".join(json.dumps(r) + "\n" for r in recs))

    def _boom(*a, **k):
        raise AssertionError("pre-flight must be reused, not re-run")

    monkeypatch.setattr(g, "run_preflight_spawned", _boom)
    gate_calls = {"n": 0}

    def counted(pool, fn, todo, out_path, *, cap_sec, watchdog_sec):
        gate_calls["n"] += 1
        assert fn is g.task
        assert len(todo) == 210
        assert all(t[2] in g.ALLOWED for t in todo)
        return 210, 12.0, None

    monkeypatch.setattr(g, "drain_pool", counted)

    class _FakePool:
        def terminate(self):
            pass

        def join(self):
            pass

    monkeypatch.setattr(g.spawn_safe, "spawn_pool", lambda *a, **k: _FakePool())
    monkeypatch.setattr(
        sys, "argv", ["run_v5_1_struct_gate.py", "--confirmatory", "--workers", "2"]
    )
    g.main()
    assert gate_calls["n"] == 1
    wall = json.loads(g.WALL.read_text())
    assert wall["confirmatory"] is True
    assert wall["tasks_done"] == 210
    assert wall["stop"] is None


def test_main_aborts_before_seeds_on_failed_preflight(tmp_path, monkeypatch):
    import run_v5_1_struct_gate as g

    _patch_paths(g, tmp_path, monkeypatch)
    monkeypatch.setattr(
        g,
        "run_preflight_spawned",
        lambda: {
            "ok": False,
            "error": "RuntimeError: explain_v5_1 output is not bit-identical",
            "versions": {},
            "imports": {},
            "smoke": {},
            "method_identity": {"ok": False},
            "smoke_seed": 99,
        },
    )
    pool_started = {"n": 0}

    def _boom(*a, **k):
        pool_started["n"] += 1
        raise AssertionError("Pool must not start after a pre-flight abort")

    monkeypatch.setattr(g.spawn_safe, "spawn_pool", _boom)
    monkeypatch.setattr(
        sys, "argv", ["run_v5_1_struct_gate.py", "--confirmatory", "--workers", "2"]
    )
    g.main()
    assert pool_started["n"] == 0
    assert not g.OUT.exists()
    assert not g.REHEARSAL_OUT.exists()
    wall = json.loads(g.WALL.read_text())
    assert wall["stop"] == "preflight_abort"
    assert wall["tasks_done"] == 0
    pf = json.loads(g.PREFLIGHT_OUT.read_text())
    assert pf["ok"] is False
    assert "bit-identical" in pf["error"]


def test_main_aborts_before_seeds_on_failed_rehearsal(tmp_path, monkeypatch):
    import run_v5_1_struct_gate as g

    _patch_paths(g, tmp_path, monkeypatch)
    monkeypatch.setattr(g, "run_preflight_spawned", lambda: _ok_preflight())

    class _FakePool:
        def terminate(self):
            pass

        def join(self):
            pass

    monkeypatch.setattr(g.spawn_safe, "spawn_pool", lambda *a, **k: _FakePool())
    calls = {"n": 0}

    def counted(*a, **k):
        calls["n"] += 1
        if calls["n"] > 1:
            raise AssertionError("gate drain_pool must not run after rehearsal abort")
        return (0, 1.0, "watchdog_abort")

    monkeypatch.setattr(g, "drain_pool", counted)
    monkeypatch.setattr(
        sys, "argv", ["run_v5_1_struct_gate.py", "--confirmatory", "--workers", "2"]
    )
    g.main()
    assert calls["n"] == 1
    assert not g.OUT.exists()
    wall = json.loads(g.WALL.read_text())
    assert wall["stop"] == "watchdog_abort"
    assert wall["tasks_done"] == 0
    assert wall.get("rehearsal_ok") is not True


def test_main_does_not_run_when_results_file_exists(tmp_path, monkeypatch):
    import run_v5_1_struct_gate as g

    _patch_paths(g, tmp_path, monkeypatch)
    # Reuse OK pre-flight / rehearsal so we reach the single-pass assertion.
    g.PREFLIGHT_OUT.write_text(json.dumps(_ok_preflight()))
    recs = [
        {
            "scen": scen,
            "n": n,
            "rep": rep,
            "gate": False,
            "TRACE_V5_1": {"lsplit": {"comp": [0], "struct": [0]}},
            "rivals": {"DFL": {"lsplit": {}}, "CFM": {"lsplit": {}}},
        }
        for scen, n, rep in g.rehearsal_plan()
    ]
    g.REHEARSAL_OUT.write_text("".join(json.dumps(r) + "\n" for r in recs))
    g.OUT.write_text("{}\n")
    monkeypatch.setattr(g.spawn_safe, "assert_parent_has_not_called_trace", lambda: None)
    monkeypatch.setattr(sys, "argv", ["run_v5_1_struct_gate.py", "--confirmatory"])
    with pytest.raises(AssertionError, match="single pass"):
        g.main()


def test_summariser_preflight_abort_leaves_h_undecided(tmp_path, monkeypatch):
    import summarize_v5_1_struct_gate as sm

    (tmp_path / "v5_1_struct_gate_wall.json").write_text(
        json.dumps(
            {
                "wall_sec": 0.0,
                "tasks_done": 0,
                "tasks_total": 0,
                "tasks_preregistered": 210,
                "stop": "preflight_abort",
                "preflight_error": "not bit-identical",
                "cap_sec": 1800,
                "prereg_sha": sm.PREREG_SHA,
            }
        )
    )
    (tmp_path / "v5_1_struct_gate_preflight.json").write_text(
        json.dumps({"ok": False, "error": "not bit-identical", "method_identity": {"ok": False}})
    )
    monkeypatch.setattr(sm, "RES", tmp_path)
    monkeypatch.setattr(sm, "OUT_MD", tmp_path / "results.md")
    monkeypatch.setattr(sm, "OUT_JSON", tmp_path / "summary.json")
    monkeypatch.setattr(sm, "JSONL", tmp_path / "v5_1_struct_gate_shape.jsonl")
    monkeypatch.setattr(sm, "WALL_PATH", tmp_path / "v5_1_struct_gate_wall.json")
    monkeypatch.setattr(sm, "PREFLIGHT_PATH", tmp_path / "v5_1_struct_gate_preflight.json")
    sm.main()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["overall"] == "ABORTED (pre-flight)"
    assert summary["seeds_spent"] is False
    assert not (tmp_path / "v5_1_struct_gate_shape.jsonl").exists()
    md = (tmp_path / "results.md").read_text()
    assert "H is not decided" in md
    assert "not touched" in md or "remain unspent" in md
    verdict_line = [ln for ln in md.splitlines() if ln.startswith("**Gate verdict:**")][0]
    assert "PASS" not in verdict_line
    assert "FAIL" not in verdict_line
    assert "ABORTED" in verdict_line


def test_summariser_rehearsal_abort_leaves_h_undecided(tmp_path, monkeypatch):
    import summarize_v5_1_struct_gate as sm

    (tmp_path / "v5_1_struct_gate_wall.json").write_text(
        json.dumps(
            {
                "wall_sec": 0.0,
                "tasks_done": 0,
                "tasks_total": 0,
                "tasks_preregistered": 210,
                "stop": "rehearsal_abort",
                "rehearsal_error": "DFL failed on rehearsal",
                "rehearsal_tasks_done": 3,
                "rehearsal_tasks_total": 12,
                "cap_sec": 1800,
                "prereg_sha": sm.PREREG_SHA,
            }
        )
    )
    (tmp_path / "v5_1_struct_gate_preflight.json").write_text(
        json.dumps({"ok": True, "method_identity": {"ok": True}})
    )
    monkeypatch.setattr(sm, "RES", tmp_path)
    monkeypatch.setattr(sm, "OUT_MD", tmp_path / "results.md")
    monkeypatch.setattr(sm, "OUT_JSON", tmp_path / "summary.json")
    monkeypatch.setattr(sm, "JSONL", tmp_path / "v5_1_struct_gate_shape.jsonl")
    monkeypatch.setattr(sm, "WALL_PATH", tmp_path / "v5_1_struct_gate_wall.json")
    monkeypatch.setattr(sm, "PREFLIGHT_PATH", tmp_path / "v5_1_struct_gate_preflight.json")
    sm.main()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["overall"] == "ABORTED (rehearsal)"
    assert summary["seeds_spent"] is False
    md = (tmp_path / "results.md").read_text()
    assert "H is not decided" in md
    verdict_line = [ln for ln in md.splitlines() if ln.startswith("**Gate verdict:**")][0]
    assert "PASS" not in verdict_line
    assert "FAIL" not in verdict_line


def test_summariser_watchdog_abort_leaves_h_undecided(tmp_path, monkeypatch):
    import summarize_v5_1_struct_gate as sm

    (tmp_path / "v5_1_struct_gate_wall.json").write_text(
        json.dumps(
            {
                "wall_sec": 300.0,
                "tasks_done": 0,
                "tasks_total": 210,
                "tasks_preregistered": 210,
                "stop": "watchdog_abort",
                "rehearsal_ok": True,
                "confirmatory": True,
                "cap_sec": 1800,
                "watchdog_sec": 300,
                "prereg_sha": sm.PREREG_SHA,
            }
        )
    )
    (tmp_path / "v5_1_struct_gate_preflight.json").write_text(
        json.dumps({"ok": True, "method_identity": {"ok": True}})
    )
    (tmp_path / "v5_1_struct_gate_shape.jsonl").write_text("")
    monkeypatch.setattr(sm, "RES", tmp_path)
    monkeypatch.setattr(sm, "OUT_MD", tmp_path / "results.md")
    monkeypatch.setattr(sm, "OUT_JSON", tmp_path / "summary.json")
    monkeypatch.setattr(sm, "JSONL", tmp_path / "v5_1_struct_gate_shape.jsonl")
    monkeypatch.setattr(sm, "WALL_PATH", tmp_path / "v5_1_struct_gate_wall.json")
    monkeypatch.setattr(sm, "PREFLIGHT_PATH", tmp_path / "v5_1_struct_gate_preflight.json")
    sm.main()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["overall"] == "ABORTED (watchdog)"
    assert summary["seeds_spent"] is True
    md = (tmp_path / "results.md").read_text()
    assert "H is not decided" in md
    assert "spent" in md.lower()
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
    assert rec.get("method_identity", {}).get("ok") is not True


def test_watchdog_timer_is_five_minutes():
    assert gate.WATCHDOG_SEC == 300


def test_rehearsal_ok_rejects_method_error(tmp_path):
    path = tmp_path / "reh.jsonl"
    good = {
        "scen": "skew_struct",
        "n": 2000,
        "rep": 98,
        "gate": False,
        "TRACE_V5_1": {"lsplit": {"comp": [0], "struct": [0]}},
        "rivals": {"DFL": {"lsplit": {}}, "CFM": {"lsplit": {}}},
    }
    bad = {
        **good,
        "rep": 97,
        "rivals": {"DFL": {"error": "boom"}, "CFM": {"lsplit": {}}},
    }
    path.write_text(json.dumps(good) + "\n" + json.dumps(bad) + "\n")
    ok, reason = gate.rehearsal_ok(path, 2, 2, None)
    assert ok is False
    assert "DFL" in reason


def test_rehearsal_ok_rejects_a_gate_index(tmp_path):
    path = tmp_path / "reh.jsonl"
    rec = {
        "scen": "skew_struct",
        "n": 2000,
        "rep": 14000,
        "gate": False,
        "TRACE_V5_1": {"lsplit": {"comp": [0], "struct": [0]}},
        "rivals": {"DFL": {"lsplit": {}}, "CFM": {"lsplit": {}}},
    }
    path.write_text(json.dumps(rec) + "\n")
    ok, reason = gate.rehearsal_ok(path, 1, 1, None)
    assert ok is False
    assert "14000" in reason or "reserved" in reason or "non-rehearsal" in reason
