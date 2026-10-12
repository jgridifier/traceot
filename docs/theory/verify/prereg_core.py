"""Shared pieces for PREREG_TAIL_v2: the fitter's span, reporting matrices, conditional law, oracles."""
import numpy as np
from scipy import special, stats, optimize

K_FIT, N_U, K_REP, N_BINS, N_FINE = 8, 12, 4, 16, 1024

def gl01(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1), 0.5 * w

def e_mat(u, K=K_FIT):
    """Orthonormal shifted Legendre e_k(u)=sqrt(2k+1) P_k(2u-1), shape (len(u), K)."""
    return np.stack([np.sqrt(2 * k + 1) * special.eval_sh_legendre(k, u) for k in range(K)], 1)

UN, WN = gl01(N_U)          # fitter nodes (n_u = max(K_fit,12) = 12 for K_fit = 8)
EN = e_mat(UN)              # (12, 8)
UF, WF = gl01(N_FINE)       # fine grid used by the fitter's report_matrix
EF = e_mat(UF)

def report_matrix(kind, ref=None):
    if kind == "legendre":
        return np.eye(K_FIT)[:, :K_REP]
    z = stats.norm.ppf(np.clip(UF, 1e-12, 1 - 1e-12))
    cols = np.zeros((N_FINE, K_REP))
    if kind == "hermite":
        for m in range(K_REP):
            cols[:, m] = special.eval_hermitenorm(m, z) / np.sqrt(special.factorial(m))
    elif kind == "reference":
        r = (z if ref is None else ref).astype(float).copy()
        cols[:, 0] = 1.0
        r -= np.sum(WF * r); cols[:, 1] = r / np.sqrt(np.sum(WF * r * r))
        for m in range(2, K_REP):
            v = EF[:, min(m, K_FIT - 1)].copy()
            for pv in range(m):
                v -= np.sum(WF * v * cols[:, pv]) * cols[:, pv]
            cols[:, m] = v / np.sqrt(np.sum(WF * v * v))
    else:
        raise ValueError(kind)
    return (EF * WF[:, None]).T @ cols      # (K_fit, K_rep): g[k,m] = <e_k, col_m>

# ---- conditional law Y = mu + sig Z + s He3(Z), s >= 0, sig - 3 s > 0 ----
def zstar(q, mu, sig, s):
    """Unique real root c of mu + sig c + s(c^3 - 3c) = q (Cardano; linear when s = 0)."""
    q, mu, sig, s = np.broadcast_arrays(*(np.asarray(v, float) for v in (q, mu, sig, s)))
    out = np.empty(q.shape)
    lin = s < 1e-9
    out[lin] = (q[lin] - mu[lin]) / sig[lin]
    nl = ~lin
    if nl.any():
        P = (sig[nl] - 3 * s[nl]) / s[nl]; Qc = (mu[nl] - q[nl]) / s[nl]
        D = np.sqrt((Qc / 2) ** 2 + (P / 3) ** 3)
        c = np.cbrt(-Qc / 2 + D) + np.cbrt(-Qc / 2 - D)
        for _ in range(3):  # polish (cancellation when P is large)
            g = mu[nl] + sig[nl] * c + s[nl] * (c ** 3 - 3 * c) - q[nl]
            c -= g / (sig[nl] + 3 * s[nl] * (c * c - 1))
        out[nl] = c
    return out

def cdf_pdf(q, mu, sig, s):
    c = zstar(q, mu, sig, s)
    return stats.norm.cdf(c), stats.norm.pdf(c) / (sig + 3 * s * (c * c - 1))

def exp_pinball(q, u, mu, sig, s):
    """Closed form E rho_u(Y - q): u(mu - q) - E[(Y - q) 1{Y<q}], E[Y 1{Z<c}] = mu Phi - sig phi + s(1-c^2) phi."""
    c = zstar(q, mu, sig, s); Ph, ph = stats.norm.cdf(c), stats.norm.pdf(c)
    M = mu * Ph - sig * ph + s * (1 - c * c) * ph
    return u * (mu - q) - (M - q * Ph)

def quantile_true(u, mu, sig, s):
    z = stats.norm.ppf(u); return mu + sig * z + s * (z ** 3 - 3 * z)

# ---- population oracle over x ~ U[0,1]^d (relevant features only), equal-mass bins b/16 ----
def xgrid(m):
    """Gauss-Legendre m nodes inside each of the 16 bins: points, weights (sum 1), bin index."""
    u, w = gl01(m)
    pts = ((np.arange(N_BINS)[:, None] + u[None]) / N_BINS).ravel()
    wts = np.tile(w / N_BINS, N_BINS); bins = np.repeat(np.arange(N_BINS), m)
    return pts, wts, bins

class Design:
    def __init__(self, name, mu, sig, s, d):
        self.name, self.mu, self.sig, self.s, self.d = name, mu, sig, s, d   # callables of X (P, d)

def build(design, m):
    pts, wts, bins = xgrid(m)
    d = design.d
    grids = np.meshgrid(*([pts] * d), indexing="ij")
    X = np.stack([g.ravel() for g in grids], 1)
    W = np.ones(len(X))
    for g in np.meshgrid(*([wts] * d), indexing="ij"): W = W * g.ravel()
    B = np.stack([g.ravel() for g in np.meshgrid(*([bins] * d), indexing="ij")], 1)
    return X, W, B, design.mu(X), design.sig(X), design.s(X)

def n_par(d): return K_FIT * (1 + d * N_BINS)

def coef_rows(theta, B):
    d = B.shape[1]; a = np.tile(theta[:K_FIT], (len(B), 1))
    G = theta[K_FIT:].reshape(d, N_BINS, K_FIT)
    for j in range(d): a = a + G[j][B[:, j]]
    return a

def centre(theta, d):
    th = theta.copy(); G = th[K_FIT:].reshape(d, N_BINS, K_FIT)
    for j in range(d):
        mval = G[j].mean(0); G[j] -= mval; th[:K_FIT] += mval
    th[K_FIT:] = G.ravel(); return th

def grad_from_rowgrad(R, B):
    """R: (P, K) row gradients in coefficient space -> parameter gradient."""
    d = B.shape[1]; g = [R.sum(0)]
    for j in range(d):
        g.append(np.stack([np.bincount(B[:, j], R[:, k], N_BINS) for k in range(K_FIT)], 1).ravel())
    return np.concatenate(g)

def oracle_newton(design, m=5, iters=60, tol=1e-12, verbose=False):
    """Method 1: damped Newton on the population first-order condition E[(F(q|x)-u) e(u)] = 0."""
    X, W, B, mu, sig, s = build(design, m); d = design.d; P = n_par(d)
    # start: L2 node fit of the true quantile at each x, averaged into the additive span by least squares
    theta = np.zeros(P); theta[0] = np.sum(W * mu)
    def gh_clean(th):
        a = coef_rows(th, B); q = a @ EN.T
        F, f = cdf_pdf(q, mu[:, None], sig[:, None], s[:, None])
        R = (W[:, None] * (F - UN[None]) * WN[None]) @ EN
        g = grad_from_rowgrad(R, B)
        Wt = W[:, None] * f * WN[None]
        nb = 1 + d * N_BINS; S = np.zeros((nb, nb, N_U))
        idx = [np.zeros(len(B), int)] + [1 + j * N_BINS + B[:, j] for j in range(d)]
        for i1 in range(d + 1):
            for i2 in range(d + 1):
                lin = idx[i1] * nb + idx[i2]
                for t in range(N_U):
                    S[:, :, t] += np.bincount(lin, Wt[:, t], nb * nb).reshape(nb, nb)
        Hm = np.einsum("abt,tk,tl->akbl", S, EN, EN).reshape(nb * K_FIT, nb * K_FIT)
        return g, Hm
    for it in range(iters):
        g, Hm = gh_clean(theta)
        gn = np.abs(g).max()
        if verbose: print(it, gn)
        if gn < tol: break
        step = np.linalg.lstsq(Hm, g, rcond=1e-12)[0]
        lam = 1.0
        while lam > 1e-4:
            th2 = theta - lam * step
            if np.abs(gh_grad_only(th2, B, W, mu, sig, s)).max() < gn: break
            lam /= 2
        theta = th2
    g, _ = gh_clean(theta)
    return centre(theta, d), float(np.abs(g).max()), it + 1

def gh_grad_only(th, B, W, mu, sig, s):
    a = coef_rows(th, B); q = a @ EN.T
    F, _ = cdf_pdf(q, mu[:, None], sig[:, None], s[:, None])
    R = (W[:, None] * (F - UN[None]) * WN[None]) @ EN
    return grad_from_rowgrad(R, B)

def oracle_lbfgs(design, m=3, start=None):
    """Method 2: L-BFGS on the closed-form expected pinball loss (different x quadrature)."""
    X, W, B, mu, sig, s = build(design, m); d = design.d; P = n_par(d)
    def fg(th):
        a = coef_rows(th, B); q = a @ EN.T
        Lr = exp_pinball(q, UN[None], mu[:, None], sig[:, None], s[:, None])
        val = np.sum(W[:, None] * WN[None] * Lr)
        F, _ = cdf_pdf(q, mu[:, None], sig[:, None], s[:, None])
        return val, grad_from_rowgrad((W[:, None] * (F - UN[None]) * WN[None]) @ EN, B)
    th0 = np.zeros(P); th0[0] = np.sum(W * mu); th0[1] = 0.5
    r = optimize.minimize(fg, th0 if start is None else start, jac=True, method="L-BFGS-B",
                          options={"maxiter": 20000, "maxfun": 40000, "ftol": 1e-16, "gtol": 1e-12, "maxcor": 50})
    return centre(r.x, d), float(np.abs(fg(r.x)[1]).max()), r.nit, float(r.fun)

def energies(theta, d, Gm):
    G = theta[K_FIT:].reshape(d, N_BINS, K_FIT) @ Gm      # (d, 16, K_rep)
    return (G ** 2).mean(1)                                 # (d, K_rep)

def slopes(theta, d, Gm):
    G = theta[K_FIT:].reshape(d, N_BINS, K_FIT) @ Gm
    xc = (np.arange(N_BINS) + 0.5) / N_BINS; xc = xc - xc.mean()
    return np.einsum("b,jbm->jm", xc, G) / np.sum(xc ** 2)

# ---- reference estimator for the power analysis: exact empirical pinball minimiser (LP, HiGHS) ----
import scipy.sparse as sp
from scipy.optimize import linprog

def sample_bins(x):
    """Fitter binning: edges at order statistics floor(k n / 16), code = #edges <= x (0-based value bin)."""
    n = len(x); xs = np.sort(x)
    e = np.unique(xs[np.minimum((np.arange(1, N_BINS) / N_BINS * n).astype(int), n - 1)])
    return np.searchsorted(e, x, side="right")

def draw(design_full, n, seed, p=8):
    rng = np.random.default_rng(seed)
    x = rng.random((n, p)); z = stats.norm.ppf(rng.random(n))
    mu, sig, s = design_full(x)
    return x, mu + sig * z + s * (z ** 3 - 3 * z)

def fit_lp(x, y):
    """min_theta sum_i sum_t w_t rho_{u_t}(y_i - e_t . a(x_i)); a(x) = a0 + sum_j g_j[bin_j(x)].
    Returns centred theta (equal-count centring as in the fitter), objective."""
    n, p = x.shape
    B = np.stack([sample_bins(x[:, j]) for j in range(p)], 1)
    counts = [np.bincount(B[:, j], minlength=N_BINS) for j in range(p)]
    # drop one bin per feature (bin 0) for identifiability; centre afterwards
    cols_per_row = []
    rows, cols, vals = [], [], []
    nb = N_BINS - 1; P = K_FIT * (1 + p * nb)
    rid = np.arange(n * N_U)
    i_idx = np.repeat(np.arange(n), N_U); t_idx = np.tile(np.arange(N_U), n)
    for k in range(K_FIT):
        rows.append(rid); cols.append(np.full(n * N_U, k)); vals.append(EN[t_idx, k])
        for j in range(p):
            b = B[i_idx, j]; m = b > 0
            rows.append(rid[m]); cols.append(K_FIT + (j * nb + (b[m] - 1)) * K_FIT + k); vals.append(EN[t_idx[m], k])
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n * N_U, P))
    M = n * N_U
    # y = A theta + r+ - r- ; cost w_t (u r+ + (1-u) r-)
    Aeq = sp.hstack([A, sp.eye(M), -sp.eye(M)]).tocsc()
    wt = WN[t_idx]; c = np.concatenate([np.zeros(P), wt * UN[t_idx], wt * (1 - UN[t_idx])])
    bounds = [(None, None)] * P + [(0, None)] * (2 * M)
    res = linprog(c, A_eq=Aeq, b_eq=y[i_idx], bounds=bounds, method="highs")
    th = res.x[:P]
    a0 = th[:K_FIT].copy(); G = np.zeros((p, N_BINS, K_FIT))
    G[:, 1:, :] = th[K_FIT:].reshape(p, nb, K_FIT)
    for j in range(p):
        mval = (counts[j][:, None] * G[j]).sum(0) / n; G[j] -= mval; a0 += mval
    return np.concatenate([a0, G.ravel()]), res.fun / n, B

def design_matrix(x):
    n, p = x.shape
    B = np.stack([sample_bins(x[:, j]) for j in range(p)], 1)
    nb = N_BINS - 1; P = K_FIT * (1 + p * nb)
    rid = np.arange(n * N_U); i_idx = np.repeat(np.arange(n), N_U); t_idx = np.tile(np.arange(N_U), n)
    rows, cols, vals = [], [], []
    for k in range(K_FIT):
        rows.append(rid); cols.append(np.full(n * N_U, k)); vals.append(EN[t_idx, k])
        for j in range(p):
            b = B[i_idx, j]; m = b > 0
            rows.append(rid[m]); cols.append(K_FIT + (j * nb + (b[m] - 1)) * K_FIT + k); vals.append(EN[t_idx[m], k])
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n * N_U, P))
    return A, B, i_idx, t_idx

def uncentre_to_theta(th, B, n, p):
    nb = N_BINS - 1
    a0 = th[:K_FIT].copy(); G = np.zeros((p, N_BINS, K_FIT)); G[:, 1:, :] = th[K_FIT:].reshape(p, nb, K_FIT)
    for j in range(p):
        cnt = np.bincount(B[:, j], minlength=N_BINS)
        mval = (cnt[:, None] * G[j]).sum(0) / n; G[j] -= mval; a0 += mval
    return np.concatenate([a0, G.ravel()])

def fit_smooth(x, y, hs=(0.05, 0.02, 0.01), iters=80, tol=1e-12):
    """Convolution-smoothed pinball (Gaussian kernel, bandwidth h) minimised by damped Newton.
    rho_h(r) = r(u - Phi(-r/h)) + h phi(r/h). Proxy for the exact empirical minimiser (fit_lp)."""
    n, p = x.shape
    A, B, i_idx, t_idx = design_matrix(x)
    u = UN[t_idx]; w = WN[t_idx] / n; yy = y[i_idx]
    th = np.zeros(A.shape[1]); th[0] = np.median(y); th[1] = np.std(y) / 2
    At = A.T.tocsr()
    for h in hs:
      def loss(th):
        r = yy - A @ th; return np.sum(w * (r * (u - stats.norm.cdf(-r / h)) + h * stats.norm.pdf(r / h)))
      f = loss(th)
      for it in range(iters):
        r = yy - A @ th
        g = -(At @ (w * (u - stats.norm.cdf(-r / h))))
        H = (At @ sp.diags(w * stats.norm.pdf(r / h) / h) @ A).toarray() + 1e-12 * np.eye(A.shape[1])
        step = np.linalg.solve(H, g); lam = 1.0
        while lam > 1e-8:
            f2 = loss(th - lam * step)
            if f2 <= f: break
            lam /= 2
        if f2 > f: break
        th = th - lam * step; df = f - f2; f = f2
        if df < tol * max(1.0, abs(f)): break
    return uncentre_to_theta(th, B, n, p), f, B

# ---- asymptotic sampling law of the (reduced) estimator at the oracle: sandwich H^-1 S H^-1 / n ----
def sandwich(design, theta_c, p_total=8, m=3):
    """Population H and score covariance S for the full p_total-feature model (relevant features first,
    the rest independent noise), reduced parameterisation (bin 0 of each feature dropped)."""
    X, W, B, mu, sig, s = build(design, m); d = design.d
    a = coef_rows(theta_c, B); q = a @ EN.T
    F, f = cdf_pdf(q, mu[:, None], sig[:, None], s[:, None])
    Hw = W[:, None] * f * WN[None]                                     # (P, 12)
    Fm = np.minimum(F[:, :, None], F[:, None, :])
    C = (Fm - UN[None, :, None] * F[:, None, :] - UN[None, None, :] * F[:, :, None] + UN[None, :, None] * UN[None, None, :])
    C = C * (WN[:, None] * WN[None, :])[None] * W[:, None, None]       # (P, 12, 12)
    nb = 1 + d * N_BINS
    idx = [np.zeros(len(B), int)] + [1 + j * N_BINS + B[:, j] for j in range(d)]
    Hs = np.zeros((nb, nb, N_U)); Ss = np.zeros((nb, nb, N_U, N_U))
    for i1 in range(d + 1):
        for i2 in range(d + 1):
            lin = idx[i1] * nb + idx[i2]
            for t in range(N_U):
                Hs[:, :, t] += np.bincount(lin, Hw[:, t], nb * nb).reshape(nb, nb)
                for t2 in range(N_U):
                    Ss[:, :, t, t2] += np.bincount(lin, C[:, t, t2], nb * nb).reshape(nb, nb)
    Hr = np.einsum("abt,tk,tl->akbl", Hs, EN, EN)              # (nb, K, nb, K)
    Sr = np.einsum("abtv,tk,vl->akbl", Ss, EN, EN)
    # extend with noise features: blocks from the intercept row (index 0) scaled by bin masses
    nn = p_total - d; NB = nb + nn * N_BINS
    def extend(M):
        out = np.zeros((NB, K_FIT, NB, K_FIT)); out[:nb, :, :nb, :] = M
        tot = M[0, :, 0, :]
        for j in range(nn):
            o = nb + j * N_BINS
            for b in range(N_BINS):
                out[o + b, :, 0, :] = out[0, :, o + b, :] = tot / N_BINS
                out[o + b, :, o + b, :] = tot / N_BINS
                for r in range(1, nb):  # relevant bins: E[1{rel bin} M] / 16
                    out[o + b, :, r, :] = out[r, :, o + b, :] = M[0, :, r, :] / N_BINS
                for j2 in range(nn):
                    if j2 != j:
                        o2 = nb + j2 * N_BINS
                        for b2 in range(N_BINS):
                            out[o + b, :, o2 + b2, :] = tot / N_BINS ** 2
        return out
    He, Se = extend(Hr), extend(Sr)
    keep = [0] + [1 + j * N_BINS + b for j in range(p_total) for b in range(1, N_BINS)]
    Hk = He[keep][:, :, keep].reshape(len(keep) * K_FIT, len(keep) * K_FIT)
    Sk = Se[keep][:, :, keep].reshape(len(keep) * K_FIT, len(keep) * K_FIT)
    return Hk, Sk

def reduced_from_centred(theta_c, d, p_total=8):
    G = theta_c[K_FIT:].reshape(d, N_BINS, K_FIT)
    a0 = theta_c[:K_FIT] + G[:, 0, :].sum(0)
    R = np.zeros((p_total, N_BINS - 1, K_FIT)); R[:d] = G[:, 1:, :] - G[:, :1, :]
    return np.concatenate([a0, R.ravel()])

def simulate_energies(th_red, Hk, Sk, n, Gm, reps, seed, p_total=8):
    """Draw theta-hat ~ N(theta*, H^-1 S H^-1 / n), centre with equal bin weights, return (reps, p, K_rep) energies."""
    Hi = np.linalg.inv(Hk); V = Hi @ Sk @ Hi / n
    L = np.linalg.cholesky(0.5 * (V + V.T) + 1e-14 * np.eye(len(V)))
    rng = np.random.default_rng(seed)
    out = np.empty((reps, p_total, K_REP))
    for r0 in range(0, reps, 500):
        rr = min(500, reps - r0)
        dr = th_red[None] + (L @ rng.standard_normal((len(V), rr))).T
        G = np.zeros((rr, p_total, N_BINS, K_FIT)); G[:, :, 1:, :] = dr[:, K_FIT:].reshape(rr, p_total, N_BINS - 1, K_FIT)
        G -= G.mean(2, keepdims=True)
        out[r0:r0 + rr] = ((G @ Gm) ** 2).mean(2)
    return out
