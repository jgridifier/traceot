"""Leakage and evaluation checks for the ADOPT reproduction.

These tests fail if a future change puts held-out covariates into the bandwidth,
lets a CV fold see its own responses, drops the paper's noise, changes the MISE
quadrature, or entangles the data seed with the CV seed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from traceot import _core

PAPER = Path(__file__).resolve().parents[2] / "benchmarks" / "adopt_paper"
sys.path.insert(0, str(PAPER))

import reproduce  # noqa: E402

import dgp  # noqa: E402


def test_kfold_is_a_partition():
    n, k, seed = 37, 5, 11
    folds = [np.asarray(f) for f in _core.kfold_indices(n, k, seed)]
    for f in folds:
        assert np.all(f[1:] > f[:-1])  # each fold is sorted, no duplicates
    stacked = np.concatenate(folds)
    assert stacked.size == n
    assert np.array_equal(np.sort(stacked), np.arange(n))
    # Every pair of folds is disjoint.
    for i in range(k):
        for j in range(i + 1, k):
            assert np.intersect1d(folds[i], folds[j]).size == 0


def test_cv_bandwidth_uses_training_range_only():
    """A held-out fold with a huge covariate range must not set h.

    The score from the Rust CV is compared with two hand-rolled loops: one that
    scales h by the training-fold range, and one that scales it by the full
    sample (the bug). Only the training-fold loop may match.
    """
    n, q, k, seed, c = 30, 8, 5, 0, 0.2
    rng = np.random.default_rng(1)
    X = rng.uniform(0.0, 1.0, size=(n, 1))
    folds = [np.asarray(f) for f in _core.kfold_indices(n, k, seed)]
    # One fold sits far to the right, so its covariate range dominates the full sample.
    X[folds[0], 0] = np.linspace(0.0, 30.0, folds[0].size)
    u = np.asarray(_core.quantile_grid(q))
    Y = u * (0.4 + X)
    full = np.asarray(_core.column_ranges(X))
    train_ranges = []
    for test in folds:
        train = np.setdiff1d(np.arange(n), test)
        assert np.intersect1d(train, test).size == 0
        train_ranges.append(np.asarray(_core.column_ranges(X[train])))
    assert any(not np.allclose(r, full) for r in train_ranges)

    def score(use_full):
        err = 0.0
        for test, ranges in zip(folds, train_ranges, strict=True):
            train = np.setdiff1d(np.arange(n), test)
            h = c * (full if use_full else ranges)
            model = _core.AdditiveModel.fit(
                "adopt",
                "wasserstein",
                X[train],
                Y[train],
                np.ascontiguousarray(h),
                max_iter=25,
                tol=1e-8,
            )
            pred = model.predict(X[test])
            err += float(np.sum(_core.dist_sq_rows("wasserstein", pred, Y[test])))
        return err / n

    rust = _core.select_bandwidth_cv(
        "adopt",
        "wasserstein",
        X,
        Y,
        np.array([c]),
        k,
        seed,
        max_iter=25,
        tol=1e-8,
    )
    got = float(rust["mean_dist_sq"][0])
    train_score = score(False)
    full_score = score(True)
    assert got == pytest.approx(train_score, abs=1e-9)
    assert abs(got - full_score) > 1e-6


def test_mise_grid_is_disjoint_midpoint_cube():
    grid = dgp.mise_grid(10)
    assert grid.shape == (1000, 3)
    ticks = (np.arange(10) + 0.5) / 10
    for j in range(3):
        assert np.allclose(np.sort(np.unique(grid[:, j])), ticks)
    rng = np.random.default_rng(0)
    X = dgp.draw_predictors(400, rng)
    assert reproduce.chebyshev_gap(X, grid) > 0.0
    # The detector itself must fire when a grid point is copied into the sample.
    leaked = np.vstack([X[:5], grid[:1]])
    assert reproduce.chebyshev_gap(leaked, grid) == 0.0


def test_mise_quadrature_matches_the_paper_integral():
    pred = np.ones((1000, 7))
    truth = np.zeros((1000, 7))
    # W2² on the grid is a mean over quantile levels, then a mean over the cube.
    assert reproduce.mise_w2(pred, truth) == pytest.approx(1.0)
    # A sum over levels would be q times larger; a sum over the cube would be 1000.
    assert reproduce.mise_w2(pred, truth) != pytest.approx(7.0)
    # Log-Cholesky d² is the sum of squared coordinates, then a mean over the cube.
    assert reproduce.mise_spd(pred, truth) == pytest.approx(7.0)
    assert reproduce.mise_spd(pred, truth) != pytest.approx(1.0)


def test_responses_carry_the_stated_noise():
    rng = np.random.default_rng(2)
    Q = np.linspace(0.2, 0.8, 6)[None, :]
    Q = np.repeat(Q, 12, axis=0)
    Y = dgp.w2_noise(Q, rng)
    amp = np.sin(2.0 * np.pi * Q) / (2.0 * np.pi)
    ratio = (Y - Q) / amp
    assert np.all(np.abs(ratio) <= 1.0 + 1e-12)
    assert np.allclose(ratio, ratio[:, :1])
    assert not np.allclose(Y, Q)

    truth = dgp.W2Truth.build("II")
    X, Y2 = dgp.simulate_w2("II", 20, np.random.default_rng(3), q=30, truth=truth)
    clean = truth.true_quantiles(X, np.asarray(_core.quantile_grid(30)))
    assert not np.allclose(Y2, clean)

    m = 6
    means = dgp.spd_population_means("I", m)
    Xs, Z = dgp.simulate_spd("I", 25, m, np.random.default_rng(4), sigma=0.01, means=means)
    clean_z = dgp.spd_true_coords("I", Xs, m, means)
    k = m * (m - 1) // 2
    assert not np.allclose(Z[:, :k], clean_z[:, :k])
    assert np.allclose(Z[:, k:], clean_z[:, k:])


def test_seeds_are_separated():
    a = reproduce.data_seed("table1", "I", 100, 0, 0)
    b = reproduce.data_seed("table1", "II", 100, 0, 0)
    c = reproduce.data_seed("table2", "I", 100, 10, 0)
    d = reproduce.data_seed("table1", "I", 100, 0, 1)
    entropies = [tuple(s.entropy) for s in (a, b, c, d)]
    assert len(set(entropies)) == 4
    # The CV seed is the replicate index, not a child of the data generator.
    assert reproduce.cv_seed(4) == 4
    assert reproduce.cv_seed(4) not in a.entropy
    rng_data = np.random.default_rng(a)
    rng_cv = np.random.default_rng(reproduce.cv_seed(0))
    assert rng_data.random() != pytest.approx(rng_cv.random())
