"""Numpy reference for the v1 kernels. Not imported by the package."""

from __future__ import annotations

import math

import numpy as np


def gauss_legendre_01(n: int):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def pstar_coef(r: int, k: int) -> float:
    return ((-1) ** (r - k)) * math.comb(r, k) * math.comb(r + k, k)


def hosking_lmoments(x, order: int):
    x = np.sort(np.asarray(x, dtype=np.float64))
    m = x.size
    j = np.arange(1, m + 1, dtype=np.float64)
    b = []
    for r in range(order):
        w = np.ones(m)
        for s in range(1, r + 1):
            w = w * (j - s) / (m - s)
        w[:r] = 0.0
        b.append(float(np.dot(x, w) / m))
    lam = []
    for r in range(order):
        lam.append(sum(pstar_coef(r, k) * b[k] for k in range(r + 1)))
    return np.asarray(lam, dtype=np.float64)


def weighted_pav(y, w):
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    blocks = []
    for i in range(y.size):
        blocks.append([w[i] * y[i], w[i], i, i + 1])
        while len(blocks) >= 2 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            b2 = blocks.pop()
            b1 = blocks.pop()
            blocks.append([b1[0] + b2[0], b1[1] + b2[1], b1[2], b2[3]])
    out = np.empty_like(y)
    for sw, wt, start, end in blocks:
        out[start:end] = sw / wt
    return out


def pinball(r, u):
    r = np.asarray(r, dtype=np.float64)
    return np.where(r >= 0.0, u * r, (u - 1.0) * r)


def _configs(sizes):
    out = [[]]
    for s in sizes:
        out = [row + [c] for row in out for c in range(s)]
    return out


def robust_risk_breakpoints(intercept, shapes, bins, y, u, kappa, eps, ball="w1", kappa_y=None):
    """Exact dual by evaluating every kink of the piecewise-linear dual."""
    rbar = max(u, 1.0 - u)
    lam_min = 0.0 if kappa_y is None else rbar / kappa_y
    y = np.asarray(y, dtype=np.float64)
    bins = np.asarray(bins, dtype=int)
    sizes = [len(col) for col in shapes]
    configs = _configs(sizes)
    cand = [lam_min]
    tables = []
    for i in range(y.size):
        loss = []
        dist = []
        for c in configs:
            q = intercept + sum(shapes[j][c[j]] for j in range(len(shapes)))
            d = sum(kappa[j] * abs(c[j] - bins[i, j]) for j in range(len(shapes)))
            if ball == "w2":
                d = d * d
            rr = float(pinball(y[i] - q, u))
            loss.append(rr)
            dist.append(d)
        loss = np.asarray(loss)
        dist = np.asarray(dist)
        for a in range(len(loss)):
            for b in range(len(loss)):
                dd = dist[a] - dist[b]
                if dd > 1e-15:
                    slope = (loss[a] - loss[b]) / dd
                    if slope >= lam_min and np.isfinite(slope):
                        cand.append(float(slope))
        tables.append((loss, dist))
    best = np.inf
    for lam in cand:
        mean = np.mean([np.max(ls - lam * ds) for ls, ds in tables])
        best = min(best, lam * eps + mean)
    return float(best)


def max_jump_bound(shapes, kappa, emp, eps, u):
    rbar = max(u, 1.0 - u)
    lip = 0.0
    for j, col in enumerate(shapes):
        g = max(abs(col[i + 1] - col[i]) for i in range(len(col) - 1)) if len(col) > 1 else 0.0
        lip = max(lip, g / max(kappa[j], 1e-15))
    return emp + eps * rbar * lip
