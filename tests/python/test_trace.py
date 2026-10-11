"""TRACE properties: order invariance, monotone quantiles, channel recovery, projection blur."""

import numpy as np

from reference.trace import inv_norm
from traceot import TraceRegressor, checks, primitives


def _reg(**kw):
    base = dict(
        n_channels=3,
        max_bins=8,
        learning_rate=0.5,
        max_rounds=50,
        patience=0,
        val_frac=0.0,
        n_bags=1,
        max_interactions=0,
        u_levels=11,
        min_count=5,
        seed=1,
    )
    base.update(kw)
    return TraceRegressor(**base)


def _energy(coef, channel):
    sl = np.asarray(coef)[1:, channel]
    return float(np.dot(sl, sl) / sl.size)


def test_order_invariance():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(48, 4))
    y = 0.8 * x[:, 0] + 0.3 * x[:, 1] + 0.15 * rng.normal(size=48)
    perm = np.array([2, 0, 3, 1])
    kw = dict(max_rounds=8, learning_rate=0.4, max_bins=6, u_levels=7, min_count=1, seed=4)
    a = _reg(**kw).fit(x, y)
    b = _reg(**kw).fit(x[:, perm], y)
    np.testing.assert_allclose(a.predict(x), b.predict(x[:, perm]), atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        a.predict(x, project=False),
        b.predict(x[:, perm], project=False),
        atol=1e-10,
        rtol=0,
    )
    inv = np.empty(4, dtype=int)
    inv[perm] = np.arange(4)
    for j in range(4):
        ca, _ = a.effect(j)
        cb, _ = b.effect(int(inv[j]))
        np.testing.assert_allclose(ca, cb, atol=1e-10, rtol=0)


def test_projected_quantiles_are_monotone():
    rng = np.random.default_rng(5)
    x = rng.uniform(size=(80, 2))
    z = rng.normal(size=80)
    y = x[:, 0] + (0.5 + x[:, 1]) * z + 0.2 * x[:, 0] * (z**3 - 3 * z)
    model = _reg(max_rounds=12, learning_rate=0.3, val_frac=0.2, patience=4, u_levels=15).fit(x, y)
    q = model.predict(x)
    assert np.all(q[:, 1:] + 1e-12 >= q[:, :-1])
    # A shape term that decreases on part of the grid is repaired by the isotonic projection.
    u = (np.arange(21) + 0.5) / 21
    zz = np.array([inv_norm(float(ui)) for ui in u])
    raw = zz + 0.6 * (zz**3 - 3 * zz)
    fixed = primitives.pava(raw)
    assert np.any(raw[1:] < raw[:-1])
    assert np.all(fixed[1:] + 1e-12 >= fixed[:-1])


def test_pure_location_scale_and_shape_recovery():
    rng = np.random.default_rng(6)
    n = 500
    x = rng.uniform(0, 1, size=(n, 3))
    loc = _reg(seed=6).fit(x, 2.0 * x[:, 0])
    c0, _ = loc.effect(0)
    assert _energy(c0, 0) > 20 * _energy(c0, 1)
    assert _energy(c0, 0) > 20 * _energy(c0, 2)
    for j in (1, 2):
        cj, _ = loc.effect(j)
        assert _energy(cj, 0) < 0.05 * _energy(c0, 0)

    z = rng.normal(size=n)
    scale = _reg(seed=6).fit(x, (0.3 + 1.6 * x[:, 0]) * z)
    s0, _ = scale.effect(0)
    assert _energy(s0, 1) > 3 * _energy(s0, 0)
    assert _energy(s0, 1) > 5 * _energy(s0, 2)
    others = max(_energy(scale.effect(j)[0], 1) for j in (1, 2))
    assert _energy(s0, 1) > 5 * others

    # He3 lives in the tails, so the grid has to reach them. 0.25 < 1/3 keeps Q increasing.
    n = 2000
    x = rng.uniform(0, 1, size=(n, 3))
    z = rng.normal(size=n)
    he3 = z**3 - 3 * z
    shape = _reg(
        seed=6, max_rounds=100, learning_rate=0.45, max_bins=10, u_levels=41, min_count=15
    ).fit(x, z + 0.25 * x[:, 0] * he3)
    h0, _ = shape.effect(0)
    others = max(_energy(shape.effect(j)[0], 2) for j in (1, 2))
    assert _energy(h0, 2) > 3 * others
    assert _energy(h0, 2) > _energy(h0, 0)
    assert _energy(h0, 2) > _energy(h0, 1)


def test_projection_blurs_shape_interactions_and_not_location():
    k = 21
    u = (np.arange(k) + 0.5) / k
    z = np.array([inv_norm(float(ui)) for ui in u])
    he3 = z**3 - 3 * z
    # Location interaction: contrast is constant in u, so the isotonic map leaves it.
    loc = [z + 1.2 * a * b for a in (0.0, 1.0) for b in (0.0, 1.0)]
    # order above is (0,0), (0,1), (1,0), (1,1)
    before, after = checks.projection_blur(*loc)
    assert before > 0
    assert after / before >= 0.99
    shape = [z + 0.6 * a * b * he3 for a in (0.0, 1.0) for b in (0.0, 1.0)]
    before_s, after_s = checks.projection_blur(*shape)
    ratio = after_s / before_s
    assert ratio < 0.90
    assert before_s > after_s


def test_held_out_gain_keeps_the_signal():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(240, 2))
    y = 1.4 * x[:, 0] + 0.05 * rng.normal(size=240)
    model = TraceRegressor(
        n_channels=2,
        max_bins=8,
        learning_rate=0.4,
        max_rounds=30,
        patience=8,
        val_frac=0.25,
        n_bags=1,
        max_interactions=0,
        u_levels=9,
        seed=8,
    ).fit(x, y)
    assert model.gains_[0] > model.gains_[1]
    assert model.gains_[0] > 0
    assert bool(model.selected_[0])
    mid = 0.5 * (float(model.gains_[0]) + float(model.gains_[1]))
    dropped = TraceRegressor(
        n_channels=2,
        max_bins=8,
        learning_rate=0.4,
        max_rounds=30,
        patience=8,
        val_frac=0.25,
        n_bags=1,
        max_interactions=0,
        u_levels=9,
        gain_threshold=mid,
        seed=8,
    ).fit(x, y)
    assert bool(dropped.selected_[0])
    assert not bool(dropped.selected_[1])


def test_split_conformal_reports_raw_and_calibrated_coverage():
    rng = np.random.default_rng(3)
    n = 800
    x = rng.random((n, 2))
    y = x[:, 0] + 0.08 * rng.normal(size=n)
    model = TraceRegressor(
        max_rounds=20,
        patience=0,
        val_frac=0.0,
        min_rounds=0,
        learning_rate=1.0,
        max_bins=8,
        u_levels=15,
        seed=3,
    ).fit(x[:500], y[:500])
    raw = model.score(x[650:], y[650:])
    model.calibrate(x[500:650], y[500:650])
    both = model.score(x[650:], y[650:])
    assert both["cover80"] == raw["cover80"]
    assert "cover80_calibrated" in both
    assert "cover90_calibrated" in both
    assert 0.0 <= both["cover80_calibrated"] <= 1.0
    q_raw = model.predict(x[650:])
    q_cal = model.predict(x[650:], calibrated=True)
    assert q_cal.shape == q_raw.shape
    assert not np.allclose(q_raw, q_cal)
