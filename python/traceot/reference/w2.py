"""1-D Wasserstein-2 geometry on a common quantile grid (reference implementation).

A distribution is represented by its quantile function evaluated on a fixed grid
``u_1 < ... < u_q`` in (0, 1). Under this representation

* the W2 distance is the L2 distance between quantile functions (approximated by
  the mean over the grid, i.e. the midpoint rule on the default grid);
* the Fréchet (W2 barycentre) mean is the pointwise mean of quantile functions;
* the optimal transport map from ``v1`` to ``v2`` is ``F2^{-1} o F1``, which on
  the grid is the monotone map sending ``Q1(u_k)`` to ``Q2(u_k)``.
"""

from __future__ import annotations

import numpy as np


def quantile_grid(q: int) -> np.ndarray:
    """Midpoint grid ``u_k = (k - 1/2) / q``, k = 1..q."""
    if q < 1:
        raise ValueError("q must be >= 1")
    return (np.arange(q, dtype=float) + 0.5) / q


def w2_sq(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Squared W2 distance between quantile functions on the same grid.

    Works on the last axis, so ``q1`` and ``q2`` may be stacks (broadcasting).
    """
    d = np.asarray(q1, dtype=float) - np.asarray(q2, dtype=float)
    return np.mean(d * d, axis=-1)


def pava(y: np.ndarray, lower: float | None = None, upper: float | None = None) -> np.ndarray:
    """L2 projection of ``y`` onto non-decreasing vectors (equal weights), then clip.

    Pool-adjacent-violators. Every element of a pooled block receives the *same*
    floating-point value (the block mean), so ties are exact. Clipping an isotonic
    vector to ``[lower, upper]`` gives the projection onto the bounded monotone
    cone, because clipping preserves order and acts coordinatewise.
    """
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    # blocks: (sum, count)
    sums: list[float] = []
    counts: list[int] = []
    for k in range(n):
        s, c = float(y[k]), 1
        while sums and sums[-1] / counts[-1] > s / c:
            s += sums.pop()
            c += counts.pop()
        sums.append(s)
        counts.append(c)
    out = np.empty(n)
    pos = 0
    for s, c in zip(sums, counts, strict=True):
        out[pos : pos + c] = s / c
        pos += c
    if lower is not None:
        out = np.maximum(out, lower)
    if upper is not None:
        out = np.minimum(out, upper)
    return out


def pava_rows(Y: np.ndarray, lower: float | None = None, upper: float | None = None) -> np.ndarray:
    """Apply :func:`pava` to every row of a 2-D array."""
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    return np.vstack([pava(row, lower, upper) for row in Y])


def collapse_knots(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge runs of exactly equal source knots ``a``, averaging the targets ``b``.

    ``a`` must be non-decreasing. Returns strictly increasing knots and the matching
    (non-decreasing, if ``b`` is) targets. This is the tie rule of the generalised
    inverse used throughout (see docs/REPRODUCTION_NOTES.md, "ties").
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    au: list[float] = []
    bu: list[float] = []
    k = 0
    n = a.shape[0]
    while k < n:
        m = k + 1
        s = float(b[k])
        while m < n and a[m] == a[k]:
            s += float(b[m])
            m += 1
        au.append(float(a[k]))
        bu.append(s / (m - k))
        k = m
    return np.asarray(au), np.asarray(bu)


def transport_apply(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate the monotone transport map with knots ``a -> b`` at points ``y``.

    ``a`` = quantile function of the source distribution on the grid, ``b`` = quantile
    function of the target. The map is the piecewise-linear interpolant through the
    (tie-collapsed) knots ``(a_k, b_k)``; outside ``[a_1, a_q]`` it is extended as a
    translation by the boundary displacement (``y + b_1 - a_1`` below, ``y + b_q - a_q``
    above), which keeps it strictly increasing and invertible.
    """
    au, bu = collapse_knots(a, b)
    y = np.asarray(y, dtype=float)
    if au.shape[0] == 1:
        return y + (bu[0] - au[0])
    out = np.interp(y, au, bu)
    lo = y < au[0]
    hi = y > au[-1]
    out[lo] = y[lo] + (bu[0] - au[0])
    out[hi] = y[hi] + (bu[-1] - au[-1])
    return out
