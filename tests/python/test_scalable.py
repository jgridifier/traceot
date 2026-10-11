"""Scalable (TRACE) primitives: reference properties and Rust parity."""

import numpy as np
import pytest

import reference as P
from reference import scalable as sc
from traceot import primitives as R


def test_sketch_exact_when_small(rng):
    x = rng.normal(size=500)
    s = sc.QuantileSketch(k=1024).update(x)
    lv = np.linspace(0, 0.999, 50)
    xs = np.sort(x)
    assert np.array_equal(s.quantiles(lv), xs[np.floor(lv * 500).astype(int)])
    assert np.array_equal(s.bin_edges(16), sc.bin_edges_from_sorted(xs, 16))


def test_sketch_rank_error_and_merge(rng):
    x = rng.standard_t(3, size=60_000)
    lv = np.linspace(0.01, 0.99, 49)
    xs = np.sort(x)
    a = sc.QuantileSketch(k=256).update(x[:30_000])
    b = sc.QuantileSketch(k=256).update(x[30_000:])
    a.merge(b)
    assert a.count == 60_000
    err = np.abs(np.searchsorted(xs, a.quantiles(lv)) / x.size - lv)
    assert err.max() < 0.02
    assert sum(map(len, a.levels)) < 256 * 12


def test_binning_and_histogram_reference(rng):
    x = rng.normal(size=1000)
    x[::97] = np.nan
    e = sc.bin_edges_from_sorted(np.sort(x[~np.isnan(x)]), 32)
    c = sc.bin_column(x, e)
    assert c.dtype == np.uint8 and c[0] == 0 and c.max() == len(e) + 1
    g = rng.normal(size=(2, 1000))
    s, k = sc.histogram(c, g, len(e) + 2)
    assert k.sum() == 1000 and np.allclose(s.sum(axis=0), g.sum(axis=1))
    assert sc.bin_column(x, np.arange(300.0)).dtype == np.uint16


def test_pinball_gradient_matches_dense(rng):
    n, K = 200, 9
    y, loc, scale = rng.normal(size=n), rng.normal(size=n) * 0.1, np.exp(rng.normal(size=n) * 0.1)
    u = np.arange(1, K + 1) / (K + 1)
    z = np.sort(rng.normal(size=K))
    basis = np.column_stack([np.ones(K), z])
    G, loss = sc.pinball_grad_location_scale(y, loc, scale, z, u, basis)
    Q = loc[:, None] + scale[:, None] * z[None, :]
    r = u[None, :] - (y[:, None] < Q)
    assert np.allclose(G, (r @ basis).T)
    assert np.isclose(loss, np.mean((y[:, None] - Q) * r))


def test_sketch_parity(rng):
    for k, n in [(64, 50), (64, 5_000), (128, 40_000)]:
        x = rng.normal(size=n)
        x[::31] = np.nan
        a, b = R.QuantileSketch(k), sc.QuantileSketch(k)
        for chunk in np.array_split(x, 3):
            a.update(np.ascontiguousarray(chunk))
            b.update(chunk)
        lv = np.linspace(0, 1, 41)
        assert np.array_equal(a.quantiles(lv), b.quantiles(lv))
        assert a.count == b.count and a.retained == sum(map(len, b.levels))
        assert np.array_equal(a.bin_edges(32), b.bin_edges(32))
        a2, b2 = R.QuantileSketch(k), sc.QuantileSketch(k)
        a2.update(x[:100])
        b2.update(x[:100])
        a.merge(a2)
        b.merge(b2)
        assert np.array_equal(a.quantiles(lv), b.quantiles(lv))


@pytest.mark.parametrize("max_bins", [8, 255, 1000])
def test_binning_parity(rng, max_bins):
    x = np.round(rng.normal(size=20_000), 2)  # many ties
    x[::53] = np.nan
    xs = np.sort(x[~np.isnan(x)])
    e1, e2 = R.bin_edges_from_sorted(xs, max_bins), P.bin_edges_from_sorted(xs, max_bins)
    assert np.array_equal(e1, e2)
    c1, c2 = R.bin_column(x, e1), P.bin_column(x, e2)
    assert c1.dtype == c2.dtype and np.array_equal(c1, c2)


@pytest.mark.parametrize("dtype,n_bins", [(np.uint8, 256), (np.uint16, 700)])
def test_histogram_parity(rng, dtype, n_bins):
    n = 150_000  # > 2 chunks
    cols = rng.integers(0, n_bins, size=(4, n)).astype(dtype)
    g = rng.normal(size=(3, n))
    s1, k1 = R.histograms(cols, g, n_bins)
    s2, k2 = P.histograms(cols, g, n_bins)
    assert np.array_equal(k1, k2)
    assert np.allclose(s1, s2, rtol=0, atol=1e-9)
    s3, k3 = R.histogram(cols[1], g, n_bins)
    assert np.array_equal(k3, k2[1]) and np.allclose(s3, s2[1], atol=1e-9)
    with pytest.raises(ValueError):
        R.histograms(cols, g, 3)


def test_pinball_parity(rng):
    n, K = 140_000, 19
    y = rng.normal(size=n)
    loc, scale = 0.1 * rng.normal(size=n), np.exp(0.2 * rng.normal(size=n))
    u = np.arange(1, K + 1) / (K + 1)
    z = np.sort(rng.normal(size=K))
    basis = np.column_stack([np.ones(K), z, z**2 - 1])
    G1, l1 = R.pinball_grad_location_scale(y, loc, scale, z, u, basis)
    G2, l2 = P.pinball_grad_location_scale(y, loc, scale, z, u, basis)
    assert np.allclose(G1, G2, atol=1e-10) and np.isclose(l1, l2, rtol=1e-10)


def test_sketch_chunked_parity(rng):
    x = rng.normal(size=30_000)
    a, b = R.QuantileSketch(128), sc.QuantileSketch(128)
    a.update_chunked(x, 4_000)
    b.update_chunked(x, 4_000)
    lv = np.linspace(0, 1, 33)
    assert np.array_equal(a.quantiles(lv), b.quantiles(lv))
    err = np.abs(np.searchsorted(np.sort(x), a.quantiles(lv[1:-1])) / x.size - lv[1:-1])
    assert err.max() < 0.03
