"""TRACE Rust core against the numpy reference, absolute tolerance 1e-10."""

import numpy as np

from reference.trace import (
    column_order,
    conformal_pvalues,
    conformal_quantile,
    duplicate_screen,
    empirical_quantile,
    feature_w2,
    fit_trace,
    pinball_grad_channels,
    pinball_scores,
    placebo_screen,
    predict_trace,
    projection_blur,
    quantile_basis,
    score_quantiles,
    target_proxy_screen,
)
from traceot import _core

ATOL = 1e-10

FIT = dict(
    n_channels=3,
    max_bins=4,
    learning_rate=0.35,
    max_rounds=4,
    patience=2,
    tol=0.0,
    val_frac=0.25,
    n_bags=2,
    bag_frac=0.7,
    gain_threshold=0.0,
    select_frac=0.7,
    max_interactions=1,
    interaction_rounds=2,
    screen_rows=20,
    min_count=1,
    u_levels=7,
    seed=7,
)


def _data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(36, 3))
    x[0, 1] = np.nan
    y = np.nan_to_num(x[:, 0], nan=0.0) + 0.25 * x[:, 2] + 0.05 * rng.normal(size=36)
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def test_basis_quantile_and_gradient_match():
    u, basis = quantile_basis(11, 3)
    ru, rb = _core.trace_basis(11, 3)
    np.testing.assert_allclose(ru, u, atol=ATOL, rtol=0)
    np.testing.assert_allclose(rb, basis, atol=ATOL, rtol=0)
    y = np.array([0.2, -1.0, 0.5, 3.0, -0.3])
    uq = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(
        _core.trace_empirical_quantile(y, uq), empirical_quantile(y, uq), atol=ATOL, rtol=0
    )
    for level in (0.1, 0.25, 0.5, 0.9):
        np.testing.assert_allclose(
            _core.trace_conformal_quantile(y, level),
            conformal_quantile(y, level),
            atol=ATOL,
            rtol=0,
        )
    theta = np.array([[0.1, -0.2, 0.0], [0.0, 0.3, -0.1], [0.4, 0.0, 0.2], [-0.2, 0.1, 0.05]])
    q0 = np.linspace(-1.0, 1.0, 11)
    g, loss = pinball_grad_channels(y[:4], theta, q0, basis, u)
    rg, rl = _core.trace_pinball_grad(y[:4], theta, q0, basis, u)
    np.testing.assert_allclose(rg, g, atol=ATOL, rtol=0)
    np.testing.assert_allclose(rl, loss, atol=ATOL, rtol=0)


def test_fit_and_predict_match():
    x, y = _data()
    ref = fit_trace(x, y, **FIT)
    model = _core.TraceModel.fit(x, y, **FIT)
    np.testing.assert_allclose(model.q0, ref["q0"], atol=ATOL, rtol=0)
    np.testing.assert_allclose(model.basis, ref["basis"], atol=ATOL, rtol=0)
    np.testing.assert_array_equal(np.asarray(model.order), np.asarray(ref["order"]))
    np.testing.assert_allclose(model.gains, ref["gains"], atol=ATOL, rtol=0)
    np.testing.assert_allclose(model.shape_gains, ref["shape_gains"], atol=ATOL, rtol=0)
    np.testing.assert_array_equal(np.asarray(model.selected), np.asarray(ref["selected"]))
    np.testing.assert_array_equal(np.asarray(model.select_counts), np.asarray(ref["select_counts"]))
    np.testing.assert_allclose(model.train_loss, ref["train_loss"], atol=ATOL, rtol=0)
    np.testing.assert_allclose(model.val_loss, ref["val_loss"], atol=ATOL, rtol=0)
    np.testing.assert_array_equal(np.asarray(model.bag_rounds), np.asarray(ref["bag_rounds"]))
    assert model.interaction_rounds == ref["interaction_rounds"]
    for j in range(x.shape[1]):
        coef, std = model.effect(j)
        np.testing.assert_allclose(coef, ref["coef"][j], atol=ATOL, rtol=0)
        np.testing.assert_allclose(std, ref["std"][j], atol=ATOL, rtol=0)
        np.testing.assert_allclose(model.edges(j), ref["edges"][j], atol=ATOL, rtol=0)
    got = model.pairs()
    assert len(got) == len(ref["pairs"])
    for a, b in zip(got, ref["pairs"], strict=True):
        assert (a["left"], a["right"]) == (b["left"], b["right"])
        np.testing.assert_allclose(a["score"], b["score"], atol=ATOL, rtol=0)
        np.testing.assert_allclose(
            np.asarray(a["coef"]).ravel(),
            np.asarray(b["coef"]).ravel(),
            atol=ATOL,
            rtol=0,
        )
    pred = model.predict(x)
    np.testing.assert_allclose(pred, predict_trace(ref, x, project=True), atol=ATOL, rtol=0)
    raw = model.predict_raw(x)
    np.testing.assert_allclose(raw, predict_trace(ref, x, project=False), atol=ATOL, rtol=0)
    order = _core.trace_column_order(x)
    np.testing.assert_array_equal(order, column_order(x))


def test_screens_scores_and_blur_match():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 3))
    x[:, 2] = x[:, 0]
    y = x[:, 0] + 0.1 * rng.normal(size=40)
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    corr, flags = _core.target_proxy_screen(x, y, 0.8, 1)
    rcorr, rflags = target_proxy_screen(x, y, 0.8, 1)
    np.testing.assert_allclose(corr, rcorr, atol=ATOL, rtol=0)
    for got, exp in zip(flags, rflags, strict=True):
        assert (got[0], got[1]) == (exp[0], exp[1])
        np.testing.assert_allclose(got[2], exp[2], atol=ATOL, rtol=0)
    dups = list(_core.duplicate_screen(x, 0.999))
    rdups = duplicate_screen(x, 0.999)
    assert len(dups) == len(rdups)
    for got, exp in zip(dups, rdups, strict=True):
        assert (got[0], got[1], bool(got[2])) == (exp[0], exp[1], exp[2])
        np.testing.assert_allclose(got[3], exp[3], atol=ATOL, rtol=0)
    real, pmax, pmean = _core.placebo_screen(x, y, 5, 3)
    rreal, rpmax, rpmean = placebo_screen(x, y, 5, 3)
    np.testing.assert_allclose(real, rreal, atol=ATOL, rtol=0)
    np.testing.assert_allclose(pmax, rpmax, atol=ATOL, rtol=0)
    np.testing.assert_allclose(pmean, rpmean, atol=ATOL, rtol=0)
    w = _core.feature_w2(x[:20], x[20:], 32)
    np.testing.assert_allclose(w, feature_w2(x[:20], x[20:], 32), atol=ATOL, rtol=0)
    u, _ = quantile_basis(9, 2)
    q = np.ascontiguousarray(rng.normal(size=(40, 9)).cumsum(axis=1))
    np.testing.assert_allclose(
        _core.conformal_pvalues(y[:15], y[15:]),
        conformal_pvalues(y[:15], y[15:]),
        atol=ATOL,
        rtol=0,
    )
    np.testing.assert_allclose(
        _core.pinball_scores(y, q, u), pinball_scores(y, q, u), atol=ATOL, rtol=0
    )
    got = _core.score_quantiles(y, q, u)
    exp = score_quantiles(y, q, u)
    for key in exp:
        np.testing.assert_allclose(got[key], exp[key], atol=ATOL, rtol=0)
    z = np.linspace(-2, 2, 21)
    before, after = _core.projection_blur(z, z, z, z + 0.4)
    rb, ra = projection_blur(z, z, z, z + 0.4)
    np.testing.assert_allclose((before, after), (rb, ra), atol=ATOL, rtol=0)
