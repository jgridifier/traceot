"""Guards for the V5.2 specified-family runner. Does not run the study or touch any seed."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SYS = ROOT / "benchmarks" / "shiftbench"
if str(SYS) not in sys.path:
    sys.path.insert(0, str(SYS))

import run_v5_2_specified_gate as gate  # noqa: E402
import sb_dgp_shape_v5_2_specified as dgp  # noqa: E402
from run_synthetic import seed_of  # noqa: E402
from summarize_v5_2_specified_gate import after_holm  # noqa: E402


def test_allowed_ranges_and_forbidden_include_sealed_and_coverage():
    assert gate.ALLOWED == set(range(11000, 11050)) | set(range(11100, 11120))
    assert set(range(9600, 9620)) <= gate.FORBIDDEN
    assert set(range(10400, 10800)) <= gate.FORBIDDEN
    assert gate.ALLOWED.isdisjoint(gate.FORBIDDEN)
    assert gate.SEALED == set(range(9600, 9620))
    assert gate.COVERAGE_RESERVED == set(range(10400, 10800))
    for i in (9600, 9619, 10400, 10799, 9700, 10100, 0, 10219):
        with pytest.raises(AssertionError):
            n = 2000 if i < 11100 else 20000
            # assert_rep also checks the size range; pick a size so the FORBIDDEN
            # check is reached for sealed / reserved indices.
            try:
                gate.assert_rep(i, n)
            except AssertionError:
                raise
    gate.assert_rep(11000, 2000)
    gate.assert_rep(11049, 2000)
    gate.assert_rep(11100, 20000)
    gate.assert_rep(11119, 20000)
    with pytest.raises(AssertionError):
        gate.assert_rep(11000, 20000)  # wrong size


def test_scenario_key_and_g_on_unshifted_columns():
    assert dgp.key("skew_comp", 1.0) == "skew_comp|k1"
    assert dgp.key("tail_comp") == "tail_comp|k1"
    rng = __import__("numpy").random.default_rng(13)
    x = rng.standard_normal((20, 5))
    g = dgp.g(x)
    expected = (x[:, 2] ** 2 - 1.0) / __import__("numpy").sqrt(2.0) + __import__("numpy").sqrt(
        2.0
    ) * __import__("numpy").sin(x[:, 4])
    __import__("numpy").testing.assert_allclose(g, expected)
    # shifting columns 0 or 1 (the composition-shifted ones) does not change g
    x2 = x.copy()
    x2[:, 0] += 3.0
    x2[:, 1] -= 2.0
    __import__("numpy").testing.assert_allclose(dgp.g(x2), g)


def test_data_stream_includes_kappa_and_is_disjoint_from_linear_mean_stream():
    s_spec = seed_of("data", "skew_comp|k1", 2000, 5, 11000)
    s_lin = seed_of("data", "skew_comp", 2000, 5, 11000)
    s_v52 = seed_of("data", "skew_comp|k1.0", 2000, 5, 11000)
    # :g format of 1.0 is "1", so the key is k1, not k1.0
    assert s_spec != s_lin
    assert s_spec != s_v52
    assert dgp.key("skew_comp", 1.0) == "skew_comp|k1"


def test_single_pass_constants():
    assert gate.CAP_SEC == 30 * 60
    assert gate.KAPPA == 1.0
    assert gate.OUT.name == "v5_2_specified_gate_shape.jsonl"
    assert len(gate.SIZES[2000][1]) == 50
    assert len(gate.SIZES[20000][1]) == 20
    # 2 cells × (50 + 20) = 140 tasks
    assert 2 * (50 + 20) == 140


def test_after_holm_requires_both_ci_and_p():
    # BEST CI + Holm p BEST < 0.05
    assert after_holm("BEST", hi=-0.01, delta=0.1, p_best_holm=0.01, p_ni_holm=0.001) == "BEST"
    # BEST CI but Holm p BEST fails; NI still holds
    assert (
        after_holm("BEST", hi=-0.01, delta=0.1, p_best_holm=0.06, p_ni_holm=0.01) == "NON-INFERIOR"
    )
    # NI CI + Holm p NI < 0.05
    assert (
        after_holm("NON-INFERIOR", hi=0.05, delta=0.1, p_best_holm=0.2, p_ni_holm=0.01)
        == "NON-INFERIOR"
    )
    # NI CI but Holm p NI fails
    assert after_holm("NON-INFERIOR", hi=0.05, delta=0.1, p_best_holm=0.2, p_ni_holm=0.06) == "FAIL"
    # CI FAIL cannot be rescued
    assert after_holm("FAIL", hi=0.2, delta=0.1, p_best_holm=0.001, p_ni_holm=0.001) == "FAIL"
