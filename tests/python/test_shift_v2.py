"""Binding tests for the shift decomposition. Numerics are in the Rust core."""

import numpy as np

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


def test_shift_decompose_matches_the_adapter_and_ignores_row_order():
    rng = np.random.default_rng(2)
    n, p = 160, 3
    x_a = rng.normal(size=(n, p))
    x_b = rng.normal(size=(n, p))
    x_b[:, 0] += 0.35
    y_a = x_a[:, 0] - 0.4 * x_a[:, 1] + 0.2 * rng.normal(size=n)
    y_b = 0.15 + 1.2 * x_b[:, 0] - 0.4 * x_b[:, 1] + 0.2 * rng.normal(size=n)
    kw = dict(k=3, n_quad=16, n_folds=2, k_fit=3, max_bins=8, max_rounds=4, n_boot=4)
    base = shift_decompose(x_a, y_a, x_b, y_b, seed=5, **kw)
    order_a = rng.permutation(n)
    order_b = rng.permutation(n)
    perm = shift_decompose(x_a[order_a], y_a[order_a], x_b[order_b], y_b[order_b], seed=5, **kw)
    for name in ("mean", "sd", "q10", "q50", "q90", "lambda1"):
        assert name in base["split"]
        assert np.allclose(base["split"][name], perm["split"][name], atol=1e-8)
    assert np.allclose(base["feat_comp"], perm["feat_comp"], atol=1e-8)
    assert np.allclose(base["feat_struct"], perm["feat_struct"], atol=1e-8)
    comp, struct = base["split"]["mean"]
    assert abs((comp + struct) - (y_b.mean() - y_a.mean())) < 1e-8
    assert abs(base["intercept"] + base["feat_struct"].sum() - struct) < 1e-8
    assert base["total_w2"] > 0
    assert base["bands"]["reps"] == 4
    assert base["bands"]["total_w2"][1] >= base["bands"]["total_w2"][0]
    cols = np.array([2, 0, 1])
    inv = np.argsort(cols)
    swapped = shift_decompose(x_a[:, cols], y_a, x_b[:, cols], y_b, seed=5, **kw)
    assert np.allclose(base["split"]["q50"], swapped["split"]["q50"], atol=1e-7)
    assert np.allclose(base["feat_comp"], swapped["feat_comp"][inv], atol=1e-7)
    assert np.allclose(base["feat_struct"], swapped["feat_struct"][inv], atol=1e-7)
