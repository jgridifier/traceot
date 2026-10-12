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


def edges_from_sorted(x_sorted, max_bins: int):
    """Order statistics at floor(k n / max_bins), de-duplicated. Matches the Rust binner."""
    import math

    n = len(x_sorted)
    if n == 0:
        return []
    q = []
    for k in range(1, max_bins):
        idx = min(int(math.floor((k / max_bins) * n)), n - 1)
        q.append(float(x_sorted[idx]))
    edges = []
    for v in q:
        if not edges or v != edges[-1]:
            edges.append(v)
    return edges


def _bin_codes(col, edges):
    codes = np.empty(len(col), dtype=np.int64)
    for i, v in enumerate(col):
        if not np.isfinite(v):
            codes[i] = 0
            continue
        n_le = 0
        for e in edges:
            if e <= v:
                n_le += 1
            else:
                break
        codes[i] = 1 + n_le
    return codes


def _pooled_steps(sum_b, cnt, learning_rate, min_samples_leaf):
    nb = len(sum_b)
    raw = np.zeros(nb)
    need = max(int(min_samples_leaf), 1)
    pools = []
    i = 0
    while i < nb:
        if cnt[i] == 0:
            i += 1
            continue
        start = i
        c = 0.0
        s = 0.0
        while i < nb and c < need:
            c += cnt[i]
            s += sum_b[i]
            i += 1
        pools.append([start, i, c, s])
    if len(pools) >= 2 and pools[-1][2] < need:
        tail = pools.pop()
        prev = pools[-1]
        prev[1] = tail[1]
        prev[2] += tail[2]
        prev[3] += tail[3]
    for start, end, c, s in pools:
        if c == 0:
            continue
        delta = learning_rate * s / c
        for b in range(start, end):
            if cnt[b] > 0:
                raw[b] = delta
    return raw


def fit_mean(
    x,
    y,
    max_bins=8,
    learning_rate=0.5,
    max_rounds=20,
    early_stopping_rounds=5,
    early_stopping_tolerance=1e-8,
    validation_size=0.0,
    min_samples_leaf=1,
):
    """Cyclic squared-loss histogram booster. The last validation_size fraction is held out."""
    import math

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n, p = x.shape
    n_val = min(int(math.floor(validation_size * n)), n - 1)
    train_end = n - n_val
    edges = []
    codes = []
    for j in range(p):
        col = x[:, j]
        train = col[:train_end]
        finite = np.sort(train[np.isfinite(train)])
        e = edges_from_sorted(finite, max_bins)
        edges.append(e)
        codes.append(_bin_codes(col, e))
    intercept = float(y[:train_end].sum() / train_end)
    effects = [np.zeros(len(e) + 2) for e in edges]
    pred = np.full(n, intercept)
    best_mse = float(np.mean((pred[train_end:] - y[train_end:]) ** 2)) if n_val else 0.0
    best = (intercept, [e.copy() for e in effects], 0)
    patience = 0
    executed = 0
    for rnd in range(max_rounds):
        for j in range(p):
            nb = len(effects[j])
            sum_b = np.zeros(nb)
            cnt = np.zeros(nb)
            for i in range(train_end):
                b = int(codes[j][i])
                sum_b[b] += y[i] - pred[i]
                cnt[b] += 1.0
            raw = _pooled_steps(sum_b, cnt, learning_rate, min_samples_leaf)
            total = float(cnt.sum())
            moment = float((cnt * raw).sum())
            mean = moment / total if total > 0 else 0.0
            intercept += mean
            for b in range(nb):
                if cnt[b] > 0:
                    effects[j][b] += raw[b] - mean
            for i in range(n):
                pred[i] += raw[int(codes[j][i])]
        executed = rnd + 1
        if n_val == 0:
            best = (intercept, [e.copy() for e in effects], executed)
            continue
        cur = float(np.mean((pred[train_end:] - y[train_end:]) ** 2))
        improve = best_mse - cur
        bar = early_stopping_tolerance * max(abs(best_mse), 1e-12)
        if improve > bar:
            best_mse = cur
            patience = 0
            best = (intercept, [e.copy() for e in effects], executed)
        else:
            patience += 1
            if patience >= early_stopping_rounds:
                break
    if n_val > 0:
        intercept, effects, best_round = best
    else:
        best_round = executed
    pred = np.full(n, intercept)
    for j in range(p):
        for i in range(n):
            pred[i] += effects[j][int(codes[j][i])]
    return {
        "intercept": intercept,
        "effects": effects,
        "prediction": pred,
        "rounds": executed,
        "best_round": best_round,
    }


def crps_quadrature(y, q, u, w):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    q = np.asarray(q, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    acc = 0.0
    for i, yi in enumerate(y):
        for t in range(len(u)):
            r = yi - q[i, t]
            rho = u[t] * r if r >= 0.0 else (u[t] - 1.0) * r
            acc += w[t] * rho
    return 2.0 * acc / max(len(y), 1)


def pinball_gh(nodes, weights, basis_e, basis_de, a, y, smooth, tail_gamma):
    """One-curve pinball score and Hessian. Matches `pinball_gh` in v1fit.rs."""
    nodes = np.asarray(nodes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    basis_e = np.asarray(basis_e, dtype=np.float64)
    basis_de = np.asarray(basis_de, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    k = int(a.size)
    g = np.zeros(k)
    h = np.zeros(k * k)
    for t, u in enumerate(nodes):
        q = 0.0
        qp = 0.0
        for c in range(k):
            q += a[c] * basis_e[t * k + c]
            qp += a[c] * basis_de[t * k + c]
        ind = 1.0 if y < q else 0.0
        if smooth > 0.0:
            z = min(40.0, max(-40.0, (q - y) / smooth))
            sig = 1.0 / (1.0 + math.exp(-z))
            dens = min(1e6, max(1e-6, sig * (1.0 - sig) / smooth))
        elif qp > 1e-3:
            dens = min(40.0, max(1e-2, 1.0 / qp))
        else:
            dens = 1.0
        shape_w = 1.0 if tail_gamma == 0.0 else 1.0 + tail_gamma * (u - 0.5) * (u - 0.5)
        for c in range(k):
            mult = shape_w if c >= 2 else 1.0
            g[c] += weights[t] * mult * (ind - u) * basis_e[t * k + c]
        for c in range(k):
            mc = shape_w if c >= 2 else 1.0
            for d in range(k):
                md = shape_w if d >= 2 else 1.0
                h[c * k + d] += (
                    weights[t]
                    * math.sqrt(mc)
                    * math.sqrt(md)
                    * dens
                    * basis_e[t * k + c]
                    * basis_e[t * k + d]
                )
    return g, h


def leave_feature_gains(nodes, weights, basis_e, players, y, rows, p, k):
    """Leave-one-feature pinball gains. Matches `leave_feature_gains` in v1fit.rs."""
    nodes = np.asarray(nodes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    basis_e = np.asarray(basis_e, dtype=np.float64)
    players = np.asarray(players, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rows = [int(i) for i in rows]
    nu = int(nodes.size)
    shape_w = (nodes - 0.5) ** 2
    sw = float(shape_w.sum())
    if sw > 0.0:
        shape_w = shape_w / sw

    def pin_of(coeff, override):
        total = 0.0
        for t, i in enumerate(rows):
            a = coeff[t]
            for s in range(nu):
                q = float(np.dot(a, basis_e[s * k : (s + 1) * k]))
                r = y[i] - q
                rho = nodes[s] * r if r >= 0.0 else (nodes[s] - 1.0) * r
                w = shape_w[s] if override else weights[s]
                total += w * rho
        return total / len(rows)

    full = np.zeros((len(rows), k))
    for t, i in enumerate(rows):
        for j in range(p + 1):
            sl = i * (p + 1) * k + j * k
            full[t] += players[sl : sl + k]
    base = pin_of(full, False)
    base_shape = pin_of(full, True)
    gains = np.zeros(p)
    shape_gains = np.zeros(p)
    for j in range(p):
        dropped = full.copy()
        shape_only = full.copy()
        for t, i in enumerate(rows):
            sl = i * (p + 1) * k + (j + 1) * k
            coef = players[sl : sl + k]
            dropped[t] -= coef
            shape_only[t, 2:] -= coef[2:]
        gains[j] = pin_of(dropped, False) - base
        shape_gains[j] = pin_of(shape_only, True) - base_shape
    return gains, shape_gains


def max_jump_bound(shapes, kappa, emp, eps, u):
    rbar = max(u, 1.0 - u)
    lip = 0.0
    for j, col in enumerate(shapes):
        g = max(abs(col[i + 1] - col[i]) for i in range(len(col) - 1)) if len(col) > 1 else 0.0
        lip = max(lip, g / max(kappa[j], 1e-15))
    return emp + eps * rbar * lip
