"""Reference for the estimator layer (``crates/core/src/estimators.rs``).

Readable numpy versions of: response distances, Fréchet R², SplitMix64 K-fold split,
bandwidth cross-validation, model fit/predict dispatch and global Fréchet regression.
Test-only: used to verify the Rust results, never imported by the ``traceot`` package.
"""

from __future__ import annotations

import numpy as np

from .additive import additive_euclid_fit, additive_euclid_predict
from .adopt import adopt_w2_component, adopt_w2_fit, adopt_w2_predict, adopt_w2_predict_averaged
from .smoother import (
    binned_smooth,
    global_frechet_weights,
    local_linear_weights,
    reflected_local_linear_weights,
)
from .w2 import pava_rows, transport_apply

_MASK = (1 << 64) - 1


def dist_sq_rows(space, a, b):
    s = np.sum((np.asarray(a) - np.asarray(b)) ** 2, axis=-1)
    return s / np.shape(a)[-1] if space == "wasserstein" else s


def frechet_r2(space, pred, z, mean):
    num = float(np.sum(dist_sq_rows(space, pred, z)))
    den = float(np.sum(dist_sq_rows(space, np.broadcast_to(mean, z.shape), z)))
    return 1.0 - num / den if den > 0 else float("nan")


def splitmix64(seed):
    state = seed & _MASK
    while True:
        state = (state + 0x9E3779B97F4A7C15) & _MASK
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        yield z ^ (z >> 31)


def kfold_indices(n, k, seed):
    if k < 2 or k > n:
        raise ValueError(f"cv must be between 2 and n_samples={n}")
    perm = list(range(n))
    rng = splitmix64(seed)
    for i in range(n - 1, 0, -1):
        j = next(rng) % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return [np.sort(np.array(f, dtype=np.int64)) for f in np.array_split(perm, k)]


def column_ranges(X):
    r = np.ptp(X, axis=0)
    return np.where(r > 0, r, 1.0)


def fit(
    method,
    space,
    X,
    Z,
    h,
    kernel="gaussian",
    tol=1e-10,
    max_iter=200,
    lower=None,
    upper=None,
    damping=1.0,
    certificate=False,
    composition="given",
    centering="frechet",
    project_prediction=False,
    bins=None,
    boundary="none",
):
    opts = dict(
        damping=damping,
        certificate=certificate,
        bins=bins,
        boundary=boundary,
    )
    if method == "adopt" and space == "wasserstein":
        f = adopt_w2_fit(
            X,
            Z,
            h,
            kernel,
            tol,
            max_iter,
            lower,
            upper,
            composition=composition,
            centering=centering,
            project_prediction=project_prediction,
            **opts,
        )
    else:
        f = additive_euclid_fit(X, Z, h, kernel, tol, max_iter, **opts)
    return {
        "method": method,
        "space": space,
        "X": X,
        "h": np.asarray(h, float),
        "kernel": kernel,
        "lower": lower,
        "upper": upper,
        "composition": composition,
        "centering": centering,
        "project_prediction": project_prediction,
        "bins": bins,
        "boundary": boundary,
        **f,
    }


def _smooth_component(m, j, x):
    xj = m["X"][:, j]
    if m.get("bins") is not None:
        return binned_smooth(xj, m["P"][j], m["h"][j], x, m["bins"], m["kernel"])
    if m.get("boundary") == "reflect":
        return reflected_local_linear_weights(xj, x, m["h"][j], m["kernel"]) @ m["P"][j]
    return local_linear_weights(xj, x, m["h"][j], m["kernel"]) @ m["P"][j]


def predict(m, X_new):
    comp = m.get("composition", "given")
    if (
        m["method"] == "adopt"
        and m["space"] == "wasserstein"
        and comp == "average"
        and m.get("orders")
    ):
        return adopt_w2_predict_averaged(
            m["X"],
            m["orders"],
            m["mu"],
            m["h"],
            X_new,
            m["kernel"],
            m["lower"],
            m["upper"],
            m.get("project_prediction", False),
            m.get("bins"),
            m.get("boundary", "none"),
        )
    if m["method"] == "adopt" and m["space"] == "wasserstein" and comp == "tangent":
        raw = additive_euclid_predict(
            m["X"],
            m["P"],
            m["M"],
            m["mu"],
            m["h"],
            X_new,
            m["kernel"],
            m.get("bins"),
            m.get("boundary", "none"),
        )
        return pava_rows(raw, m["lower"], m["upper"])
    if m["method"] == "adopt" and m["space"] == "wasserstein":
        return adopt_w2_predict(
            m["X"],
            m["P"],
            m["M"],
            m["mu"],
            m["h"],
            X_new,
            m["kernel"],
            m["lower"],
            m["upper"],
            m.get("project_prediction", False),
            m.get("bins"),
            m.get("boundary", "none"),
        )
    raw = additive_euclid_predict(
        m["X"],
        m["P"],
        m["M"],
        m["mu"],
        m["h"],
        X_new,
        m["kernel"],
        m.get("bins"),
        m.get("boundary", "none"),
    )
    if m["method"] == "l2additive":
        return pava_rows(raw, m["lower"], m["upper"])
    return raw


def component(m, j, x):
    if (
        m["method"] == "adopt"
        and m["space"] == "wasserstein"
        and m.get("composition", "given") != "tangent"
    ):
        return adopt_w2_component(
            m["X"][:, j],
            m["P"][j],
            m["h"][j],
            x,
            m["kernel"],
            m["lower"],
            m["upper"],
            m.get("bins"),
            m.get("boundary", "none"),
        )
    return _smooth_component(m, j, x)


def transport(m, j, x0, v):
    g = component(m, j, np.array([x0]))[0]
    tangent = m.get("composition") == "tangent"
    if m["method"] == "adopt" and m["space"] == "wasserstein" and not tangent:
        return transport_apply(m["M"][j], g, v)
    return v + g - m["M"][j]


def select_bandwidth_cv(
    method,
    space,
    X,
    Z,
    grid,
    k,
    seed,
    kernel="gaussian",
    tol=1e-10,
    max_iter=200,
    lower=None,
    upper=None,
    damping=1.0,
    certificate=False,
    composition="given",
    centering="frechet",
    project_prediction=False,
    bins=None,
    boundary="none",
):
    n = X.shape[0]
    folds = kfold_indices(n, k, seed)
    errs = np.zeros(len(grid))
    for g, c in enumerate(grid):
        for test in folds:
            train = np.setdiff1d(np.arange(n), test)
            # Training-fold ranges only: the held-out fold must not set the scale of h.
            h = c * column_ranges(X[train])
            m = fit(
                method,
                space,
                X[train],
                Z[train],
                h,
                kernel,
                tol,
                max_iter,
                lower,
                upper,
                damping,
                certificate,
                composition,
                centering,
                project_prediction,
                bins,
                boundary,
            )
            errs[g] += float(np.sum(dist_sq_rows(space, predict(m, X[test]), Z[test])))
    errs /= n
    best = int(np.argmin(errs))
    return {"grid": np.asarray(grid, float), "mean_dist_sq": errs, "best_fraction": grid[best]}


def global_frechet_predict(space, X_train, Z_train, X_new, lower=None, upper=None):
    raw = global_frechet_weights(X_train, X_new) @ Z_train
    return pava_rows(raw, lower, upper) if space == "wasserstein" else raw
