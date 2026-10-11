"""Scalable primitives for the TRACE core (reference implementation).

These are the building blocks of an EBM-style cyclic booster for additive models of
the conditional quantile function, designed for millions of rows and ~100 features:

* :class:`QuantileSketch` - a deterministic, mergeable compactor sketch (MRL/KLL
  style) for streaming quantiles, used to choose bin edges without sorting all data.
* :func:`bin_edges_from_sorted` / :func:`bin_column` - EBM-style histogram binning
  to small integer codes (bin 0 = missing, value bins 1..len(edges)+1).
* :func:`histogram` - per-bin sums of gradient channels and counts: one O(n) pass.
* :func:`pinball_grad_location_scale` - per-row pinball (quantile) subgradients on a
  level grid, projected onto u-basis channels, for Q(u|x) = loc + scale * z_u. O(n K)
  time and O(n M) memory (the n x K quantile matrix is never materialised).

Nothing here is O(n^2). The Rust core (``traceot_core::scalable``) must match these
functions exactly (sketch, binning, counts) or to floating tolerance (sums).
"""

from __future__ import annotations

import numpy as np


class QuantileSketch:
    """Deterministic mergeable compactor sketch.

    Level ``h`` holds items of weight ``2**h``. When a level reaches ``k`` items it is
    sorted and compacted: items at positions ``offset, offset+2, ...`` are promoted to
    level ``h+1`` (``offset`` alternates 0/1 per level on each compaction), the rest
    are dropped, and an odd leftover (the largest item) stays. Rank error is
    O(log(n/k)/k) relative. With ``n <= k`` the sketch is exact.
    """

    def __init__(self, k: int = 2048):
        if k < 2:
            raise ValueError("k must be >= 2")
        self.k = int(k)
        self.levels: list[list[float]] = [[]]
        self.flips: list[int] = [0]
        self.count = 0  # number of non-NaN values seen

    def _compact(self, h: int) -> None:
        buf = sorted(self.levels[h])
        keep_last = None
        if len(buf) % 2 == 1:
            keep_last = buf.pop()
        off = self.flips[h]
        self.flips[h] ^= 1
        promoted = buf[off::2]
        if h + 1 == len(self.levels):
            self.levels.append([])
            self.flips.append(0)
        self.levels[h + 1].extend(promoted)
        self.levels[h] = [] if keep_last is None else [keep_last]

    def _settle(self) -> None:
        h = 0
        while h < len(self.levels):
            if len(self.levels[h]) >= self.k:
                self._compact(h)
            h += 1

    def update(self, values) -> QuantileSketch:
        for v in np.asarray(values, dtype=float).ravel():
            if v != v:  # NaN
                continue
            self.levels[0].append(float(v))
            self.count += 1
            if len(self.levels[0]) >= self.k:
                self._settle()
        return self

    def update_chunked(self, values, chunk: int) -> QuantileSketch:
        """Sketch consecutive ``chunk``-sized blocks separately, then merge in order."""
        if chunk <= 0:
            raise ValueError("chunk must be > 0")
        values = np.asarray(values, dtype=float).ravel()
        for s in range(0, values.shape[0], chunk):
            self.merge(QuantileSketch(self.k).update(values[s : s + chunk]))
        return self

    def merge(self, other: QuantileSketch) -> QuantileSketch:
        if other.k != self.k:
            raise ValueError("cannot merge sketches with different k")
        while len(self.levels) < len(other.levels):
            self.levels.append([])
            self.flips.append(0)
        for h, lev in enumerate(other.levels):
            self.levels[h].extend(lev)
        self.count += other.count
        self._settle()
        return self

    def weighted_items(self) -> tuple[np.ndarray, np.ndarray]:
        vals, wts = [], []
        for h, lev in enumerate(self.levels):
            vals.extend(lev)
            wts.extend([1 << h] * len(lev))
        v = np.asarray(vals, dtype=float)
        w = np.asarray(wts, dtype=np.int64)
        order = np.argsort(v, kind="stable")
        return v[order], w[order]

    def quantiles(self, levels) -> np.ndarray:
        """Item whose cumulative weight first exceeds ``floor(level * W)``."""
        v, w = self.weighted_items()
        if v.size == 0:
            return np.full(np.shape(levels), np.nan)
        cw = np.cumsum(w)
        total = int(cw[-1])
        out = []
        for lv in np.atleast_1d(levels):
            r = min(int(np.floor(float(lv) * total)), total - 1)
            out.append(v[int(np.searchsorted(cw, r, side="right"))])
        return np.asarray(out)

    def bin_edges(self, max_bins: int) -> np.ndarray:
        return edges_from_quantile_fn(self.quantiles, max_bins)


def edges_from_quantile_fn(qfn, max_bins: int) -> np.ndarray:
    """Cut points ``qfn(k / max_bins)``, k = 1..max_bins-1, de-duplicated (sorted)."""
    if not 2 <= max_bins <= 65535:
        raise ValueError("max_bins must be in [2, 65535]")
    levels = np.arange(1, max_bins) / max_bins
    e = np.asarray(qfn(levels), dtype=float)
    e = e[np.isfinite(e)]
    return np.unique(e)


def bin_edges_from_sorted(x_sorted: np.ndarray, max_bins: int) -> np.ndarray:
    """Exact edges from sorted non-NaN data: order statistics at ranks floor(k n / B)."""
    n = x_sorted.shape[0]
    if n == 0:
        return np.empty(0)
    return edges_from_quantile_fn(
        lambda lv: x_sorted[np.minimum(np.floor(lv * n).astype(np.int64), n - 1)], max_bins
    )


def bin_column(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin codes: 0 for NaN, else ``1 + #{edges <= x}``. uint8 if it fits, else uint16."""
    x = np.asarray(x, dtype=float)
    n_bins = edges.shape[0] + 2
    dtype = np.uint8 if n_bins <= 256 else np.uint16
    codes = 1 + np.searchsorted(edges, x, side="right")
    codes[np.isnan(x)] = 0
    return codes.astype(dtype)


def histogram(codes: np.ndarray, grads: np.ndarray, n_bins: int):
    """Per-bin gradient sums (n_bins, M) and counts (n_bins,) for one binned feature.

    ``grads`` is channel-major, shape (M, n).
    """
    grads = np.atleast_2d(np.asarray(grads, dtype=float))
    M = grads.shape[0]
    sums = np.zeros((n_bins, M))
    for m in range(M):
        sums[:, m] = np.bincount(codes, weights=grads[m], minlength=n_bins)[:n_bins]
    counts = np.bincount(codes, minlength=n_bins)[:n_bins].astype(np.uint64)
    return sums, counts


def pinball_grad_location_scale(y, loc, scale, z, u, basis):
    """Negative pinball subgradient channels and mean pinball loss.

    ``Q_ik = loc_i + scale_i z_k``; ``r_ik = u_k - 1{y_i < Q_ik}``;
    ``G[m, i] = sum_k r_ik basis[k, m]``; loss = mean_i mean_k rho_{u_k}(y_i - Q_ik).
    Returns (G of shape (M, n), loss).
    """
    y, loc, scale = (np.asarray(a, dtype=float) for a in (y, loc, scale))
    z, u = np.asarray(z, dtype=float), np.asarray(u, dtype=float)
    basis = np.atleast_2d(np.asarray(basis, dtype=float))
    K, M = basis.shape
    G = np.zeros((M, y.shape[0]))
    loss = 0.0
    for k in range(K):  # loop over levels keeps memory O(n M)
        res = y - (loc + scale * z[k])
        ind = (res < 0).astype(float)
        r = u[k] - ind
        G += basis[k][:, None] * r[None, :]
        loss += float(np.sum(res * r))
    return G, loss / (y.shape[0] * K)


def histograms(columns: np.ndarray, grads: np.ndarray, n_bins: int):
    """Histograms of many features; ``columns`` is columnar (p, n). Returns
    sums (p, n_bins, M) and counts (p, n_bins)."""
    out = [histogram(c, grads, n_bins) for c in np.atleast_2d(columns)]
    return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])
