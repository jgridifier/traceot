import numpy as np

from traceot import TraceV1, shift


def _data(seed, n=4000):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    e = rng.exponential(size=n) - 1.0
    y = x[:, 0] + (1.0 + 0.3 * np.tanh(x[:, 1])) * e
    return x, y


def test_pit_rounds_round_trips_and_keeps_identities():
    x_a, y_a = _data(1)
    x_b, y_b = _data(2)
    x_b = x_b + np.array([0.3, 0.0, 0.0])
    kw = dict(k_fit=4, k_report=4, max_bins=8, max_rounds=6, tol=1e-6, mono="none")
    m_a = TraceV1(**kw, pit_rounds=8).fit(x_a, y_a)
    m_b = TraceV1(**kw, pit_rounds=8).fit(x_b, y_b)
    assert m_a.get_params()["pit_rounds"] == 8
    bary = shift.barycenter(m_a, m_b, x_a, x_b)
    assert abs(float(bary["player_sum_gap"])) < 1e-9
    assert abs(float(bary["order_gap"])) < 1e-9
    draws = shift.bootstrap_barycenter(m_a, m_b, x_a, y_a, x_b, y_b, 3, 5)
    assert np.asarray(draws["phi"]).shape[0] == 3


def test_pit_refit_moves_shape_channels_toward_lmoments():
    # Exp(1) - 1 noise: L-moment ratio λ3/λ2 = 1/3.
    x, y = _data(3, n=20000)
    kw = dict(k_fit=4, k_report=4, max_bins=8, max_rounds=8, tol=1e-6, mono="none")
    mask = np.ones(len(y), dtype=np.uint8)
    mask[: len(y) // 2] = 0
    ratios = []
    for rounds in (0, 10):
        m = TraceV1(**kw, pit_rounds=rounds).fit(x, y)
        got = shift.entry_from_models(m, m, x, mask, x, mask, float(len(y)), float(len(y)))
        c = -np.asarray(got["exit"])
        ratios.append((c[2] / 5.0) / (c[1] / 3.0))
    assert abs(ratios[1] - 1.0 / 3.0) < abs(ratios[0] - 1.0 / 3.0) + 0.01
    assert abs(ratios[1] - 1.0 / 3.0) < 0.05
