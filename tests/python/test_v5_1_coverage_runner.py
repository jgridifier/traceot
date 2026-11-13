"""Guards for the V5.1 coverage runner. Does not run the study or touch any seed."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SYS = ROOT / "benchmarks" / "shiftbench"
if str(SYS) not in sys.path:
    sys.path.insert(0, str(SYS))

import run_v5_1_coverage as cov  # noqa: E402
from run_synthetic import seed_of  # noqa: E402


def test_allowed_range_is_exactly_10400_10799():
    assert cov.ALLOWED == set(range(10400, 10800))
    assert min(cov.ALLOWED) == 10400 and max(cov.ALLOWED) == 10799
    assert len(cov.ALLOWED) == 400


def test_sealed_range_is_forbidden_and_disjoint_from_allowed():
    sealed = set(range(9600, 9620))
    assert cov.SEALED == sealed
    assert sealed <= cov.FORBIDDEN
    assert cov.ALLOWED.isdisjoint(cov.FORBIDDEN)
    assert cov.ALLOWED.isdisjoint(sealed)
    for i in range(9600, 9620):
        with pytest.raises(AssertionError):
            cov.assert_rep(i)


def test_spent_ranges_from_prereg_section_4_are_forbidden():
    spent_samples = [
        0,
        199,
        1999,
        5000,
        6499,
        8000,
        8499,
        9100,
        9500,
        9549,
        9700,
        9749,
        9800,
        9819,
        10100,
        10149,
        10200,
        10219,
        11000,
        11049,
        11100,
        11119,
    ]
    for i in spent_samples:
        assert i in cov.FORBIDDEN
        assert i not in cov.ALLOWED
        with pytest.raises(AssertionError):
            cov.assert_rep(i)


def test_fresh_indices_pass_the_guard():
    for i in (10400, 10401, 10600, 10799):
        cov.assert_rep(i)


def test_v51cov_stream_is_a_new_name_and_varies_with_b():
    # Stream name frozen in PREREG_V5_1_COVERAGE §2; used by nothing else as a seed part.
    s0 = seed_of("v51cov", "skew_comp", 2000, 5, 10400, 0)
    s1 = seed_of("v51cov", "skew_comp", 2000, 5, 10400, 1)
    data = seed_of("data", "skew_comp", 2000, 5, 10400)
    assert s0 != s1
    assert s0 != data
    assert isinstance(s0, int)


def test_single_pass_constants():
    assert cov.B == 500
    assert cov.N == 2000 and cov.P == 5 and cov.R == 400
    assert cov.CAP_SEC == 60 * 60
    assert cov.OUT.name == "v5_1_coverage.jsonl"
    assert cov.SCEN == ("skew_comp", "tail_comp")


def test_pass_interval_helper():
    from summarize_v5_1_coverage import decide

    assert decide(360, 400)[0] == "PASS"
    assert decide(344, 400)[0] == "PASS"
    assert decide(374, 400)[0] == "PASS"
    assert decide(343, 400)[0] == "FAIL (under-covers)"
    assert decide(375, 400)[0] == "FAIL (over-covers)"
    # Already decided with missing reps.
    assert decide(0, 350)[0] == "FAIL (under-covers)"  # max k = 50 < 344, wait 0+50=50
    assert decide(380, 380)[0] == "FAIL (over-covers)"  # min k = 380 even if 20 miss
    # Not yet decided.
    assert decide(340, 360)[0] == "INCOMPLETE"  # min 340, max 380
    # Still inside whatever the missing do: 350 covers, 10 missing -> 350..360, all in [344,374]
    assert decide(350, 390)[0] == "PASS"
