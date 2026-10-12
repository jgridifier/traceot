"""Binding tests for the shift decomposition. Numerics are in the Rust core."""

import numpy as np
import pytest

from traceot import TraceV1, shift, shift_decompose


def test_quadratic_phi_matches_the_closed_form_and_keeps_a_negative():
    coefficients = np.array([1.5, -0.5, 0.25], dtype=np.float64)
    channel = 3
    phi = shift.quadratic_phi(coefficients, channel)
    total = float(coefficients.sum())
    expected = coefficients * total / (2 * channel + 1)
    assert np.max(np.abs(phi - expected)) < 1e-12
    assert phi[1] < 0
    channel_w2 = total * total / (2 * channel + 1)
    assert abs(float(phi.sum()) - channel_w2) < 1e-12


def test_entry_exit_players_sum_and_mass_stays_out_of_the_sum():
    stay_a = np.array([1.0, 0.2, -0.1])
    stay_b = np.array([1.4, 0.0, 0.2])
    exit_mean = np.array([0.5, -0.3, 0.1])
    entry_mean = np.array([2.0, 0.4, -0.2])
    got = shift.entry_exit(stay_a, stay_b, exit_mean, entry_mean, 0.2, 0.3, 100.0, 80.0)
    delta = got["stay"] + got["mix"] + got["entry"] + got["exit"]
    assert np.max(np.abs(delta - got["delta"])) < 1e-12
    for channel in range(3):
        players = np.array(
            [
                got["stay"][channel],
                got["mix"][channel],
                got["entry"][channel],
                got["exit"][channel],
            ]
        )
        phi = shift.quadratic_phi(players, channel)
        assert abs(float(phi[0]) - got["phi_stay"][channel]) < 1e-12
        assert abs(float(phi.sum()) - float(players.sum()) ** 2 / (2 * channel + 1)) < 1e-12
    hk, hk_norm = shift.hk_mass(100.0, 80.0)
    assert abs(hk - got["hk2_mass"]) < 1e-12
    assert abs(hk_norm - got["hk2_normalized"]) < 1e-12
    assert abs(hk - 4 * (10.0 - 80.0**0.5) ** 2) < 1e-12


def test_explain_is_order_free_and_players_sum_to_the_channel():
    rng = np.random.default_rng(0)
    n, p = 400, 3
    x_a = rng.normal(size=(n, p))
    x_b = rng.normal(size=(n, p)) + np.array([0.4, -0.2, 0.0])
    y_a = np.tanh(x_a[:, 0]) + 0.3 * rng.normal(size=n)
    y_b = 1.3 * np.tanh(x_b[:, 0]) - 0.4 * np.tanh(x_b[:, 1]) + 0.3 * rng.normal(size=n)
    fit = dict(k_fit=3, k_report=3, max_bins=8, max_rounds=6, tol=1e-6, mono="none")
    model_a = TraceV1(**fit).fit(x_a, y_a)
    model_b = TraceV1(**fit).fit(x_b, y_b)
    report = shift.explain(model_a, model_b, x_a, y_a, x_b, y_b, n_folds=2, seed=1)
    assert report["order_gap"] <= 1e-9
    assert report["player_sum_gap"] <= 1e-9
    bary = report["bary"]
    phi = np.asarray(bary["phi"])
    assert phi.shape == (2 * p + 1, 3)
    for channel in range(3):
        assert abs(float(phi[:, channel].sum()) - bary["w2_channel"][channel]) <= 1e-9
    marginal = report["marginal"]
    for channel in range(3):
        share = marginal["comp_w2"][channel] + marginal["struct_w2"][channel]
        delta = marginal["lambda_b"][channel] - marginal["lambda_a"][channel]
        assert abs(share - (2 * channel + 1) * delta * delta) <= 1e-8
    assert "gap_comp_w2" in report and "gap_struct_w2" in report


def test_bands_mark_a_constant_replicate_as_unresolved_at_zero_only_when_away():
    point = np.array([0.0, 2.0])
    replicates = np.column_stack(
        [
            np.full(40, 0.0),
            np.linspace(1.5, 2.5, 40),
        ]
    )
    band = shift.bands(point, replicates)
    assert band["resolved"][0] is False
    assert band["lo"][1] <= 2.0 <= band["hi"][1]


def test_shift_decompose_rejects_short_samples_and_nonfinite_values():
    x = np.zeros((4, 1))
    y = np.zeros(4)
    with pytest.raises(ValueError):
        shift_decompose(x, y, x, y, folds=5)
    bad = y.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError):
        shift_decompose(x, bad, x, y, folds=2)


def _assert_channel_sums(got):
    bary_a = np.asarray(got["bary_a"])
    bary_phi = np.asarray(got["bary_phi"])
    dc = bary_a.sum(axis=(1, 2))
    assert np.allclose(dc, got["bary_dc"], atol=1e-9)
    assert abs(float(bary_phi.sum()) - got["w2"]["bary"]) < 1e-9
    marg = float(np.sum(got["marg_phi"]["comp"]) + np.sum(got["marg_phi"]["struct"]))
    marg += got["resid_phi"]["comp"] + got["resid_phi"]["struct"]
    assert abs(marg - got["w2"]["marginal"]) < 1e-9
    assert np.allclose(
        got["lsplit"]["comp"] + got["lsplit"]["struct"],
        got["lmom"]["B"] - got["lmom"]["A"],
        atol=1e-9,
    )
    assert abs(got["w2"]["gap"] - (got["w2"]["marginal"] - got["w2"]["bary"])) < 1e-12
    assert got["config"]["centring"] == "pooled_mean"
    assert np.allclose(got["feat_comp"], bary_a[0, 1:, 0], atol=0.0)
    assert np.allclose(got["feat_struct"], bary_a[0, 1:, 1], atol=0.0)
    assert abs(got["intercept"] - bary_a[0, 0, 1]) < 1e-15


def test_shift_decompose_matches_the_adapter_and_ignores_row_order():
    rng = np.random.default_rng(2)
    n, p = 160, 3
    x_a = rng.normal(size=(n, p))
    x_b = rng.normal(size=(n, p))
    x_b[:, 0] += 0.35
    y_a = x_a[:, 0] - 0.4 * x_a[:, 1] + 0.2 * rng.normal(size=n)
    y_b = 0.15 + 1.2 * x_b[:, 0] - 0.4 * x_b[:, 1] + 0.2 * rng.normal(size=n)
    kw = dict(k=3, n_quad=16, n_folds=2, k_fit=3, max_bins=8, max_rounds=4, n_boot=0)
    base = shift_decompose(x_a, y_a, x_b, y_b, seed=5, **kw)
    order_a = rng.permutation(n)
    order_b = rng.permutation(n)
    perm = shift_decompose(x_a[order_a], y_a[order_a], x_b[order_b], y_b[order_b], seed=5, **kw)
    for name in ("mean", "sd", "q10", "q50", "q90"):
        assert name in base["split"]
        assert np.allclose(base["split"][name], perm["split"][name], atol=1e-9)
        anchor = 0.5 * (
            np.array(base["split_anchor_A"][name]) + np.array(base["split_anchor_B"][name])
        )
        assert np.allclose(anchor, base["split"][name], atol=1e-12)
        assert np.allclose(base["split_aref"][name], base["split_anchor_A"][name], atol=0.0)
    assert "lambda1" not in base["split"]
    assert np.allclose(base["feat_comp"], perm["feat_comp"], atol=1e-9)
    assert np.allclose(base["feat_struct"], perm["feat_struct"], atol=1e-9)
    comp, struct = base["split"]["mean"]
    assert abs((comp + struct) - (y_b.mean() - y_a.mean())) < 1e-9
    assert base["total_w2"] > 0
    _assert_channel_sums(base)
    cols = np.array([2, 0, 1])
    inv = np.argsort(cols)
    swapped = shift_decompose(x_a[:, cols], y_a, x_b[:, cols], y_b, seed=5, **kw)
    assert np.allclose(base["split"]["q50"], swapped["split"]["q50"], atol=1e-9)
    assert np.allclose(base["feat_comp"], swapped["feat_comp"][inv], atol=1e-9)
    assert np.allclose(base["feat_struct"], swapped["feat_struct"][inv], atol=1e-9)
    assert np.allclose(base["bary_a"][:, 1:], swapped["bary_a"][:, 1:][:, inv], atol=1e-9)
    assert np.allclose(base["bary_a"][:, 0], swapped["bary_a"][:, 0], atol=1e-9)
    assert np.allclose(base["interaction"][:, 1:], swapped["interaction"][:, 1:][:, inv], atol=1e-9)

    flipped = shift_decompose(x_b, y_b, x_a, y_a, seed=5, **kw)
    for name in base["split"]:
        assert np.allclose(base["split"][name], -np.array(flipped["split"][name]), atol=1e-9)
        assert np.allclose(
            base["split_anchor_A"][name], -np.array(flipped["split_anchor_B"][name]), atol=1e-9
        )
    assert np.allclose(base["lsplit"]["comp"], -flipped["lsplit"]["comp"], atol=1e-9)
    assert np.allclose(base["bary_a"], -flipped["bary_a"], atol=1e-9)
    assert np.allclose(base["bary_phi"], flipped["bary_phi"], atol=1e-9)
    assert np.allclose(base["marg_phi"]["comp"], flipped["marg_phi"]["comp"], atol=1e-9)
    assert abs(base["resid_phi"]["comp"] - flipped["resid_phi"]["comp"]) < 1e-9
    assert abs(base["resid_phi"]["struct"] - flipped["resid_phi"]["struct"]) < 1e-9

    band_kw = dict(kw)
    band_kw["n_boot"] = 4
    banded = shift_decompose(x_a, y_a, x_b, y_b, seed=5, **band_kw)
    assert banded["bands"]["reps"] == 4
    assert banded["bands"]["critical"] >= 0.0
    lo, hi = banded["bands"]["split_uniform"]["mean"]
    assert lo[0] <= banded["split"]["mean"][0] <= hi[0]
    assert banded["bands"]["total_w2"][1] >= banded["bands"]["total_w2"][0]

    modeled = shift_decompose(x_a, y_a, x_b, y_b, seed=5, return_model=True, **kw)
    model = modeled["model"]
    assert model.rearranged is False
    u = np.linspace(0.05, 0.95, 32)
    qa = model.predict_quantile(x_a, u, "A")
    qb = model.predict_quantile(x_b, u, "B")
    assert qa.shape == (n, u.size)
    assert qb.shape == (n, u.size)
    ca = model.coef(x_a, "A")
    assert ca.shape[0] == n
    assert ca.shape[1] == modeled["config"]["K_fitted"]
