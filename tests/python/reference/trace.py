"""Numpy reference for TRACE (test-only).

Mirrors ``traceot_core::trace``. Loops follow the Rust accumulation order so the
two agree to a tight tolerance. Not used at runtime.
"""

from __future__ import annotations

import math
import struct

import numpy as np

from .scalable import bin_column, bin_edges_from_sorted
from .w2 import pava

_MASK64 = (1 << 64) - 1


def _u64(x: int) -> int:
    return x & _MASK64


class SplitMix64:
    def __init__(self, seed: int):
        self.s = _u64(seed)

    def next_u64(self) -> int:
        self.s = _u64(self.s + 0x9E3779B97F4A7C15)
        z = self.s
        z = _u64((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9)
        z = _u64((z ^ (z >> 27)) * 0x94D049BB133111EB)
        return z ^ (z >> 31)


def fisher_yates(items: list, rng: SplitMix64) -> None:
    n = len(items)
    if n < 2:
        return
    for i in range(n - 1, 0, -1):
        j = rng.next_u64() % (i + 1)
        items[i], items[j] = items[j], items[i]


def total_key(v: float) -> int:
    """Signed key matching ``f64::total_cmp``."""
    u = int(np.float64(v).view(np.uint64))
    left = u if u < (1 << 63) else u - (1 << 64)
    mask = 0 if left >= 0 else (1 << 63) - 1
    out = _u64(left) ^ _u64(mask)
    return out - (1 << 64) if out >= (1 << 63) else out


def rust_round(x: float) -> int:
    """``f64::round`` for a non-negative value: half away from zero."""
    return int(math.floor(x + 0.5))


def column_order(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n, p = x.shape
    idx = list(range(p))
    # Stable sort. The key is lexicographic ``f64::total_cmp``.
    idx.sort(key=lambda j: tuple(total_key(float(x[i, j])) for i in range(n)))
    return np.asarray(idx, dtype=np.int64)


def inv_norm(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("p in (0, 1)")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        return num / den
    if p > 1.0 - plow:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        return -(num / den)
    q = p - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    den = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
    return num / den


def quantile_basis(n_levels: int, n_channels: int):
    if n_channels not in (2, 3, 4):
        raise ValueError("n_channels")
    if not 3 <= n_levels <= 401:
        raise ValueError("u_levels")
    u = (np.arange(n_levels, dtype=np.float64) + 0.5) / n_levels
    z = np.array([inv_norm(float(v)) for v in u])
    cols = [np.ones(n_levels), z]
    if n_channels >= 3:
        cols.append(z**3 - 3.0 * z)
    if n_channels == 4:
        cols.append(z**2 - 1.0)
    basis = np.column_stack(cols)
    for m in range(n_channels):
        rms = math.sqrt(float(np.dot(basis[:, m], basis[:, m])) / n_levels)
        if rms > 1e-15:
            basis[:, m] /= rms
    return u, basis


def empirical_quantile(y: np.ndarray, u: np.ndarray) -> np.ndarray:
    ys = np.sort(np.asarray(y, dtype=np.float64))
    n = ys.shape[0]
    idx = np.minimum(np.floor(np.asarray(u, dtype=np.float64) * n).astype(np.int64), n - 1)
    return ys[idx]


def pinball_grad_channels(y, theta, q0, basis, u):
    y = np.asarray(y, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    q0 = np.asarray(q0, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    n, m_ch = theta.shape
    k = u.shape[0]
    grad = np.zeros((n, m_ch))
    row_loss = np.zeros(n)
    for i in range(n):
        th = theta[i]
        acc = 0.0
        g = grad[i]
        for kk in range(k):
            b = basis[kk]
            q = float(q0[kk])
            for m in range(m_ch):
                q += float(th[m]) * float(b[m])
            res = float(y[i]) - q
            r = (float(u[kk]) - 1.0) if res < 0.0 else float(u[kk])
            acc += res * r
            for m in range(m_ch):
                g[m] += r * float(b[m])
        row_loss[i] = acc / k
    return grad, row_loss


def _masked_means(row_loss, mask):
    train_sum = 0.0
    val_sum = 0.0
    n_train = 0
    n_val = 0
    for i, m in enumerate(mask):
        if m == 1:
            train_sum += float(row_loss[i])
            n_train += 1
        elif m == 2:
            val_sum += float(row_loss[i])
            n_val += 1
    train = math.nan if n_train == 0 else train_sum / n_train
    val = train if n_val == 0 else val_sum / n_val
    return train, val, n_train, n_val


def _split_mask(n, seed, val_frac):
    idx = list(range(n))
    fisher_yates(idx, SplitMix64(seed))
    n_val = 0 if val_frac <= 0.0 or n < 4 else min(n - 1, max(1, rust_round(val_frac * n)))
    mask = np.ones(n, dtype=np.uint8)
    for i in idx[:n_val]:
        mask[i] = 2
    return mask


def _bag_mask(n, cfg, bag):
    full = cfg["n_bags"] == 1 and cfg["bag_frac"] >= 1.0 - 1e-15
    seed = cfg["seed"]
    if full:
        return _split_mask(n, _u64(seed + 20_011 * (bag + 1)), cfg["val_frac"])
    perm = list(range(n))
    fisher_yates(perm, SplitMix64(_u64(seed + 10_007 * (bag + 1))))
    m = min(n, max(2, rust_round(cfg["bag_frac"] * n)))
    perm = perm[:m]
    fisher_yates(perm, SplitMix64(_u64(seed + 20_011 * (bag + 1))))
    n_val = (
        0
        if cfg["val_frac"] <= 0.0 or len(perm) < 4
        else min(len(perm) - 1, max(1, rust_round(cfg["val_frac"] * len(perm))))
    )
    mask = np.zeros(n, dtype=np.uint8)
    for t, i in enumerate(perm):
        mask[i] = 2 if t < n_val else 1
    return mask


def _feature_delta(codes, grad, mask, n_bins, k, lr, min_count):
    m_ch = grad.shape[1]
    sums = np.zeros((n_bins, m_ch))
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(codes.shape[0]):
        if mask[i] != 1:
            continue
        b = int(codes[i])
        counts[b] += 1
        for m in range(m_ch):
            sums[b, m] += float(grad[i, m])
    delta = np.zeros((n_bins, m_ch))
    weight = 0.0
    scale = lr / k
    for b in range(n_bins):
        if counts[b] >= min_count:
            weight += float(counts[b])
            inv = scale / float(counts[b])
            for m in range(m_ch):
                delta[b, m] = float(sums[b, m]) * inv
    if weight > 0.0:
        for m in range(m_ch):
            mean = 0.0
            for b in range(n_bins):
                if counts[b] >= min_count:
                    mean += float(delta[b, m]) * float(counts[b])
            mean /= weight
            for b in range(n_bins):
                if counts[b] >= min_count:
                    delta[b, m] -= mean
    return delta


def _apply_delta(theta, coef, codes, delta):
    coef += delta
    m_ch = theta.shape[1]
    for i in range(theta.shape[0]):
        b = int(codes[i])
        for m in range(m_ch):
            theta[i, m] += float(delta[b, m])


def _rebuild(theta, codes, order, coef, pairs):
    theta[:] = 0.0
    m_ch = theta.shape[1]
    for j in order:
        col = codes[j]
        cf = coef[j]
        for i in range(theta.shape[0]):
            b = int(col[i])
            for m in range(m_ch):
                theta[i, m] += float(cf[b, m])
    for pair in pairs:
        cl, cr = codes[pair["left"]], codes[pair["right"]]
        nb_r = pair["n_bins_right"]
        pc = pair["coef"]
        for i in range(theta.shape[0]):
            base = (int(cl[i]) * nb_r + int(cr[i])) * m_ch
            for m in range(m_ch):
                theta[i, m] += float(pc[base + m])


def _loss_on_rows(y, theta, q0, basis, u, rows):
    if not rows:
        return math.nan
    k = len(u)
    m_ch = theta.shape[1]
    total = 0.0
    for i in rows:
        th = theta[i]
        acc = 0.0
        for kk in range(k):
            q = float(q0[kk])
            b = basis[kk]
            for m in range(m_ch):
                q += float(th[m]) * float(b[m])
            res = float(y[i]) - q
            r = (float(u[kk]) - 1.0) if res < 0.0 else float(u[kk])
            acc += res * r
        total += acc / k
    return total / len(rows)


def _feature_gains(y, theta, q0, basis, u, codes, coef, val_rows, threshold):
    p = len(coef)
    m_ch = theta.shape[1]
    base = _loss_on_rows(y, theta, q0, basis, u, val_rows)
    gains = np.zeros(p)
    selected = np.zeros(p, dtype=bool)
    for j in range(p):
        col = codes[j]
        cf = coef[j]
        for i in val_rows:
            b = int(col[i])
            for m in range(m_ch):
                theta[i, m] -= float(cf[b, m])
        loss_j = _loss_on_rows(y, theta, q0, basis, u, val_rows)
        gains[j] = loss_j - base
        selected[j] = gains[j] > threshold
        for i in val_rows:
            b = int(col[i])
            for m in range(m_ch):
                theta[i, m] += float(cf[b, m])
    return gains, selected


def _boost_main_nb(codes, n_bins, order, y, mask, q0, basis, u, cfg):
    m_ch = basis.shape[1]
    k = len(u)
    p = len(codes)
    coef = [np.zeros((n_bins[j], m_ch)) for j in range(p)]
    theta = np.zeros((y.shape[0], m_ch))
    grad, row_loss = pinball_grad_channels(y, theta, q0, basis, u)
    _, val0, _, n_val = _masked_means(row_loss, mask)
    use_es = cfg["patience"] > 0 and n_val > 0 and cfg["val_frac"] > 0.0
    best_val = val0
    best_coef = [c.copy() for c in coef]
    stale = 0
    rounds = 0
    for _r in range(cfg["max_rounds"]):
        for j in order:
            delta = _feature_delta(
                codes[j], grad, mask, n_bins[j], k, cfg["learning_rate"], cfg["min_count"]
            )
            _apply_delta(theta, coef[j], codes[j], delta)
        grad, row_loss = pinball_grad_channels(y, theta, q0, basis, u)
        _, val_loss, _, _ = _masked_means(row_loss, mask)
        rounds += 1
        if use_es:
            if val_loss < best_val - cfg["tol"]:
                best_val = val_loss
                best_coef = [c.copy() for c in coef]
                stale = 0
            else:
                stale += 1
                if stale >= cfg["patience"]:
                    break
    if use_es:
        coef = best_coef
    return coef, rounds


def _pair_delta(left, right, grad, mask, nb_l, nb_r, k, lr, min_count):
    m_ch = grad.shape[1]
    n_cells = nb_l * nb_r
    sums = np.zeros((n_cells, m_ch))
    counts = np.zeros(n_cells, dtype=np.int64)
    for i in range(left.shape[0]):
        if mask[i] != 1:
            continue
        cell = int(left[i]) * nb_r + int(right[i])
        counts[cell] += 1
        for m in range(m_ch):
            sums[cell, m] += float(grad[i, m])
    delta = np.zeros(n_cells * m_ch)
    weight = 0.0
    scale = lr / k
    for c in range(n_cells):
        if counts[c] >= min_count:
            weight += float(counts[c])
            inv = scale / float(counts[c])
            for m in range(m_ch):
                delta[c * m_ch + m] = float(sums[c, m]) * inv
    if weight > 0.0:
        for m in range(m_ch):
            mean = 0.0
            for c in range(n_cells):
                if counts[c] >= min_count:
                    mean += float(delta[c * m_ch + m]) * float(counts[c])
            mean /= weight
            for c in range(n_cells):
                if counts[c] >= min_count:
                    delta[c * m_ch + m] -= mean
    return delta


def _pair_score(left, right, grad, rows, nb_l, nb_r, k):
    m_ch = grad.shape[1]
    n_cells = nb_l * nb_r
    sums = np.zeros((n_cells, m_ch))
    counts = np.zeros(n_cells, dtype=np.int64)
    for i in rows:
        cell = int(left[i]) * nb_r + int(right[i])
        counts[cell] += 1
        for m in range(m_ch):
            sums[cell, m] += float(grad[i, m])
    score = 0.0
    inv_k = 1.0 / k
    for c in range(n_cells):
        if counts[c] == 0:
            continue
        energy = 0.0
        for m in range(m_ch):
            mean = float(sums[c, m]) / float(counts[c]) * inv_k
            energy += mean * mean
        score += float(counts[c]) * energy
    return score


def _boost_pairs(codes, n_bins, order, y, mask, q0, basis, u, coef, selected, cfg):
    if cfg["max_interactions"] == 0 or cfg["interaction_rounds"] == 0:
        return [], 0
    eligible = [j for j in order if selected[j]]
    if len(eligible) < 2:
        return [], 0
    m_ch = basis.shape[1]
    k = len(u)
    p = len(codes)
    pos = [0] * p
    for rank, j in enumerate(order):
        pos[j] = rank
    theta = np.zeros((y.shape[0], m_ch))
    _rebuild(theta, codes, order, coef, [])
    grad, row_loss = pinball_grad_channels(y, theta, q0, basis, u)
    _, val0, _, n_val = _masked_means(row_loss, mask)
    rows = [i for i, m in enumerate(mask) if m == 1]
    fisher_yates(rows, SplitMix64(_u64(cfg["seed"] + 42_042)))
    rows = rows[: min(len(rows), cfg["screen_rows"])]
    rows.sort()
    cands = []
    for a in range(len(eligible)):
        for b in range(a + 1, len(eligible)):
            j, kft = eligible[a], eligible[b]
            score = _pair_score(codes[j], codes[kft], grad, rows, n_bins[j], n_bins[kft], k)
            if score > 0.0:
                cands.append((score, pos[j], pos[kft], j, kft))
    # Stable two-pass sort: position order, then score descending (finite scores).
    cands.sort(key=lambda t: (t[1], t[2]))
    cands.sort(key=lambda t: t[0], reverse=True)
    cands = cands[: cfg["max_interactions"]]
    if not cands:
        return [], 0
    pairs = []
    for score, _, _, j, kft in cands:
        pairs.append(
            {
                "left": j,
                "right": kft,
                "n_bins_left": n_bins[j],
                "n_bins_right": n_bins[kft],
                "coef": np.zeros(n_bins[j] * n_bins[kft] * m_ch),
                "score": score,
            }
        )
    use_es = cfg["patience"] > 0 and n_val > 0 and cfg["val_frac"] > 0.0
    best_val = val0
    best_pairs = [{**p, "coef": p["coef"].copy()} for p in pairs]
    stale = 0
    rounds = 0
    for _r in range(cfg["interaction_rounds"]):
        deltas = [
            _pair_delta(
                codes[pair["left"]],
                codes[pair["right"]],
                grad,
                mask,
                pair["n_bins_left"],
                pair["n_bins_right"],
                k,
                cfg["learning_rate"],
                cfg["min_count"],
            )
            for pair in pairs
        ]
        for pair, delta in zip(pairs, deltas, strict=True):
            pair["coef"] += delta
            cl, cr = codes[pair["left"]], codes[pair["right"]]
            nb_r = pair["n_bins_right"]
            for i in range(theta.shape[0]):
                base = (int(cl[i]) * nb_r + int(cr[i])) * m_ch
                for m in range(m_ch):
                    theta[i, m] += float(delta[base + m])
        grad, row_loss = pinball_grad_channels(y, theta, q0, basis, u)
        _, val_loss, _, _ = _masked_means(row_loss, mask)
        rounds += 1
        if use_es:
            if val_loss < best_val - cfg["tol"]:
                best_val = val_loss
                best_pairs = [{**p, "coef": p["coef"].copy()} for p in pairs]
                stale = 0
            else:
                stale += 1
                if stale >= cfg["patience"]:
                    break
    if use_es:
        pairs = best_pairs
    return pairs, rounds


def default_config(**overrides):
    cfg = {
        "n_channels": 3,
        "max_bins": 32,
        "learning_rate": 0.3,
        "max_rounds": 40,
        "patience": 8,
        "tol": 0.0,
        "val_frac": 0.2,
        "n_bags": 1,
        "bag_frac": 1.0,
        "gain_threshold": 0.0,
        "select_frac": 0.7,
        "max_interactions": 0,
        "interaction_rounds": 15,
        "screen_rows": 4096,
        "min_count": 1,
        "u_levels": 19,
        "seed": 0,
    }
    cfg.update(overrides)
    return cfg


def _frequency_keeps(count, votes, frac):
    if votes == 0:
        return True
    return count + 1e-12 >= frac * votes


def fit_trace(x, y, **overrides):
    cfg = default_config(**overrides)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n, p = x.shape
    u, basis = quantile_basis(cfg["u_levels"], cfg["n_channels"])
    order = [int(j) for j in column_order(x)]
    edges = []
    codes = []
    n_bins = []
    for j in range(p):
        col = x[:, j]
        finite = col[np.isfinite(col)]
        finite = np.array(sorted(finite.tolist(), key=total_key), dtype=np.float64)
        e = bin_edges_from_sorted(finite, cfg["max_bins"])
        c = bin_column(col, e)
        edges.append(np.asarray(e, dtype=np.float64))
        codes.append(c)
        n_bins.append(int(e.shape[0]) + 2)
    m_ch = cfg["n_channels"]
    bags = []
    for b in range(cfg["n_bags"]):
        mask = _bag_mask(n, cfg, b)
        train_y = y[mask == 1]
        q0 = empirical_quantile(train_y, u)
        coef, rounds = _boost_main_nb(codes, n_bins, order, y, mask, q0, basis, u, cfg)
        theta = np.zeros((n, m_ch))
        _rebuild(theta, codes, order, coef, [])
        val_rows = [i for i in range(n) if mask[i] == 2]
        had_val = len(val_rows) > 0
        if had_val:
            gains, selected = _feature_gains(
                y, theta, q0, basis, u, codes, coef, val_rows, cfg["gain_threshold"]
            )
        else:
            gains = np.zeros(p)
            selected = np.ones(p, dtype=bool)
        bags.append(
            {
                "coef": coef,
                "q0": q0,
                "gains": gains,
                "selected": selected,
                "had_val": had_val,
                "rounds": rounds,
            }
        )
    votes = sum(1 for bag in bags if bag["had_val"])
    select_counts = np.zeros(p, dtype=np.int64)
    gain_sum = np.zeros(p)
    for bag in bags:
        if bag["had_val"]:
            select_counts += bag["selected"].astype(np.int64)
            gain_sum += bag["gains"]
    selected = np.array(
        [_frequency_keeps(int(select_counts[j]), votes, cfg["select_frac"]) for j in range(p)]
    )
    gains = gain_sum / votes if votes else np.zeros(p)
    coef = [np.zeros((n_bins[j], m_ch)) for j in range(p)]
    std = [np.zeros((n_bins[j], m_ch)) for j in range(p)]
    n_bags_f = float(cfg["n_bags"])
    for j in range(p):
        for t in range(coef[j].size):
            flat_idx = np.unravel_index(t, coef[j].shape)
            mean = 0.0
            for bag in bags:
                mean += float(bag["coef"][j][flat_idx])
            mean /= n_bags_f
            var = 0.0
            if cfg["n_bags"] > 1:
                for bag in bags:
                    d = float(bag["coef"][j][flat_idx]) - mean
                    var += d * d
                var /= cfg["n_bags"] - 1
            std[j][flat_idx] = math.sqrt(var)
            coef[j][flat_idx] = mean if selected[j] else 0.0
    q0 = np.zeros(len(u))
    for bag in bags:
        q0 += bag["q0"]
    q0 /= n_bags_f
    report = _split_mask(n, _u64(cfg["seed"] + 20_011), cfg["val_frac"])
    pairs, interaction_rounds = _boost_pairs(
        codes, n_bins, order, y, report, q0, basis, u, coef, selected, cfg
    )
    theta = np.zeros((n, m_ch))
    _rebuild(theta, codes, order, coef, pairs)
    grad, row_loss = pinball_grad_channels(y, theta, q0, basis, u)
    train_loss, val_loss, _, _ = _masked_means(row_loss, report)
    model = {
        "u": u,
        "basis": basis,
        "q0": q0,
        "edges": edges,
        "n_bins": n_bins,
        "order": order,
        "coef": coef,
        "std": std,
        "selected": selected,
        "gains": gains,
        "select_counts": select_counts,
        "pairs": pairs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "bag_rounds": [bag["rounds"] for bag in bags],
        "interaction_rounds": interaction_rounds,
        "n_channels": m_ch,
    }
    model["train_pred"] = predict_trace(model, x, project=True)
    return model


def predict_trace(model, x, project=True):
    x = np.asarray(x, dtype=np.float64)
    n, _p = x.shape
    m_ch = model["n_channels"]
    k = len(model["u"])
    codes = [bin_column(x[:, j], model["edges"][j]) for j in range(x.shape[1])]
    theta = np.zeros((n, m_ch))
    _rebuild(theta, codes, model["order"], model["coef"], model["pairs"])
    basis = model["basis"]
    q0 = model["q0"]
    out = np.zeros((n, k))
    for i in range(n):
        th = theta[i]
        for kk in range(k):
            q = float(q0[kk])
            b = basis[kk]
            for m in range(m_ch):
                q += float(th[m]) * float(b[m])
            out[i, kk] = q
        if project:
            out[i] = pava(out[i])
    return out


def pearson(x, y) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = min(x.shape[0], y.shape[0])
    count = 0
    sx = 0.0
    sy = 0.0
    for i in range(n):
        if math.isfinite(x[i]) and math.isfinite(y[i]):
            sx += float(x[i])
            sy += float(y[i])
            count += 1
    if count < 2:
        return math.nan
    mx = sx / count
    my = sy / count
    sxx = syy = sxy = 0.0
    for i in range(n):
        if math.isfinite(x[i]) and math.isfinite(y[i]):
            dx = float(x[i]) - mx
            dy = float(y[i]) - my
            sxx += dx * dx
            syy += dy * dy
            sxy += dx * dy
    if sxx <= 0.0 or syy <= 0.0:
        return math.nan
    return sxy / math.sqrt(sxx * syy)


def projection_blur(q00, q01, q10, q11):
    qs = [np.asarray(q, dtype=np.float64) for q in (q00, q01, q10, q11)]

    def rms(curves):
        c = curves[3] - curves[2] - curves[1] + curves[0]
        return math.sqrt(float(np.dot(c, c)) / c.shape[0])

    before = rms(qs)
    after = rms([pava(q) for q in qs])
    return before, after


def conformal_pvalues(cal, test):
    cal = np.sort(np.asarray(cal, dtype=np.float64))
    test = np.asarray(test, dtype=np.float64)
    n = cal.shape[0]
    out = np.empty(test.shape[0])
    for i, v in enumerate(test):
        below = int(np.searchsorted(cal, v, side="left"))
        ge = n - below
        out[i] = (1.0 + ge) / (n + 1.0)
    return out


def score_quantiles(y, q, u):
    y = np.asarray(y, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    n, k = q.shape

    def q_at(row, level):
        if level <= u[0]:
            return float(row[0])
        if level >= u[-1]:
            return float(row[-1])
        j = int(np.searchsorted(u, level, side="right")) - 1
        t = (level - float(u[j])) / (float(u[j + 1]) - float(u[j]))
        return float(row[j]) * (1.0 - t) + float(row[j + 1]) * t

    pin_sum = 0.0
    c80 = c90 = 0
    sse_med = sse_mean = 0.0
    for i in range(n):
        row = q[i]
        acc = 0.0
        for kk in range(k):
            res = float(y[i]) - float(row[kk])
            r = (float(u[kk]) - 1.0) if res < 0.0 else float(u[kk])
            acc += res * r
        pin_sum += acc / k
        lo80, hi80 = q_at(row, 0.1), q_at(row, 0.9)
        lo90, hi90 = q_at(row, 0.05), q_at(row, 0.95)
        if lo80 <= y[i] <= hi80:
            c80 += 1
        if lo90 <= y[i] <= hi90:
            c90 += 1
        med = q_at(row, 0.5)
        mean = float(row.sum()) / k
        sse_med += (med - float(y[i])) ** 2
        sse_mean += (mean - float(y[i])) ** 2
    pinball = pin_sum / n
    return {
        "pinball": pinball,
        "crps": 2.0 * pinball,
        "cover80": c80 / n,
        "cover90": c90 / n,
        "mse_median": sse_med / n,
        "mse_mean": sse_mean / n,
    }


def _to_bits(v: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(v)))[0]


def _column_hash(col: np.ndarray) -> int:
    h = 0xCBF29CE484222325
    for v in col:
        h ^= _to_bits(float(v))
        h = (h * 0x100000001B3) & _MASK64
    return h


def target_proxy_screen(x, y, threshold=0.9, max_shift=1):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n, p = x.shape
    flags = []
    contemporaneous = np.empty(p)
    for j in range(p):
        col = x[:, j]
        c0 = pearson(col, y)
        contemporaneous[j] = c0
        if math.isfinite(c0) and abs(c0) >= threshold:
            flags.append((j, 0, c0))
        for s in range(1, max_shift + 1):
            if s >= n:
                break
            lead = pearson(col[: n - s], y[s:])
            if math.isfinite(lead) and abs(lead) >= threshold:
                flags.append((j, s, lead))
            lag = pearson(col[s:], y[: n - s])
            if math.isfinite(lag) and abs(lag) >= threshold:
                flags.append((j, -s, lag))
    return contemporaneous, flags


def duplicate_screen(x, near=0.999):
    x = np.asarray(x, dtype=np.float64)
    n, p = x.shape
    cols = [x[:, j].copy() for j in range(p)]
    hashes = [_column_hash(c) for c in cols]
    flags = []
    for i in range(p):
        for j in range(i + 1, p):
            exact = hashes[i] == hashes[j] and all(
                _to_bits(float(x[t, i])) == _to_bits(float(x[t, j])) for t in range(n)
            )
            corr = pearson(cols[i], cols[j])
            if exact or (math.isfinite(corr) and abs(corr) >= near):
                flags.append((i, j, exact, corr))
    return flags


def placebo_screen(x, y, n_perm=20, seed=0):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = x.shape[1]
    cols = [x[:, j] for j in range(p)]
    real = np.array([abs(pearson(c, y)) for c in cols])
    rng = SplitMix64(_u64(seed))
    shuffled = [float(v) for v in y]
    placebo_max = 0.0
    placebo_sum = 0.0
    for _ in range(n_perm):
        fisher_yates(shuffled, rng)
        m = 0.0
        for c in cols:
            a = abs(pearson(c, shuffled))
            if math.isfinite(a) and a > m:
                m = a
        placebo_sum += m
        if m > placebo_max:
            placebo_max = m
    return real, placebo_max, placebo_sum / n_perm


def feature_w2(a, b, grid=128):
    from reference.w2 import quantile_grid

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    u = quantile_grid(grid)
    out = np.empty(a.shape[1])
    for j in range(a.shape[1]):
        aa = a[:, j][np.isfinite(a[:, j])]
        bb = b[:, j][np.isfinite(b[:, j])]
        if aa.size == 0 or bb.size == 0:
            out[j] = math.nan
            continue
        qa = empirical_quantile(aa, u)
        qb = empirical_quantile(bb, u)
        d = qa - qb
        out[j] = math.sqrt(float(np.dot(d, d)) / grid)
    return out


def pinball_scores(y, q, u):
    y = np.asarray(y, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    n, k = q.shape
    out = np.empty(n)
    for i in range(n):
        acc = 0.0
        for kk in range(k):
            res = float(y[i]) - float(q[i, kk])
            r = (float(u[kk]) - 1.0) if res < 0.0 else float(u[kk])
            acc += res * r
        out[i] = acc / k
    return out
