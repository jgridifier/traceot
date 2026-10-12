"""Shared helpers for the TRACE theory checks (numpy/scipy only)."""
import math
import numpy as np
from scipy import special, integrate

RESULTS = []  # (script, check_id, status, detail)


def record(script, cid, ok, detail):
    status = "PASS" if ok else "FAIL"
    RESULTS.append((script, cid, status, detail))
    print(f"[{status}] {script}:{cid}  {detail}", flush=True)
    return ok


def info(script, cid, detail):
    RESULTS.append((script, cid, "INFO", detail))
    print(f"[INFO] {script}:{cid}  {detail}", flush=True)


def gl01(n):
    """Gauss-Legendre nodes/weights on [0,1]; exact for polynomials of degree <= 2n-1."""
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def Pstar(k, u):
    """Shifted Legendre P*_k(u) = P_k(2u-1) (Hosking 1990 convention, P*_k(1)=1)."""
    return special.eval_sh_legendre(k, u)


def Pstar_matrix(K, u):
    return np.stack([Pstar(k, u) for k in range(K)], axis=1)


def Pstar_deriv_matrix(K, u):
    """d/du P*_k(u) = 2 P_k'(2u-1)."""
    out = []
    for k in range(K):
        c = np.zeros(k + 1); c[k] = 1.0
        dc = np.polynomial.legendre.legder(c)
        out.append(2.0 * np.polynomial.legendre.legval(2 * u - 1, dc) if k > 0 else np.zeros_like(u))
    return np.stack(out, axis=1)


def pstar_coef(r, k):
    """Hosking (1990) p*_{r,k} = (-1)^{r-k} C(r,k) C(r+k,k)."""
    return (-1) ** (r - k) * math.comb(r, k) * math.comb(r + k, k)


def quad01(f, **kw):
    kw.setdefault("limit", 500)
    kw.setdefault("epsabs", 1e-13)
    kw.setdefault("epsrel", 1e-12)
    return integrate.quad(f, 0.0, 1.0, **kw)[0]


def lmoments_from_Q(Q, R):
    """Population L-moments lambda_1..lambda_R = int Q(u) P*_{r-1}(u) du by adaptive quadrature."""
    return np.array([quad01(lambda u, r=r: Q(u) * Pstar(r - 1, u)) for r in range(1, R + 1)])


def sample_lmoments(x, R):
    """Unbiased sample L-moments l_1..l_R (Hosking 1990, via unbiased PWMs b_r). x: (..., m) sorted along last axis not required."""
    x = np.sort(np.asarray(x, float), axis=-1)
    m = x.shape[-1]
    j = np.arange(1, m + 1, dtype=float)
    b = []
    for r in range(R):
        wts = np.ones(m)
        for s in range(1, r + 1):
            wts = wts * (j - s) / (m - s)
        wts[: r] = 0.0
        b.append((x * wts).sum(-1) / m)
    b = np.stack(b, axis=-1)
    l = []
    for r in range(R):
        l.append(sum(pstar_coef(r, k) * b[..., k] for k in range(r + 1)))
    return np.stack(l, axis=-1)


def pav(y, w=None):
    """Weighted pool-adjacent-violators: L2(w) projection of y onto nondecreasing vectors."""
    y = np.asarray(y, float)
    n = len(y)
    w = np.ones(n) if w is None else np.asarray(w, float)
    vals, wts, cnts = [], [], []
    for i in range(n):
        vals.append(y[i]); wts.append(w[i]); cnts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnts.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnts.pop()
            ww = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / ww); wts.append(ww); cnts.append(c1 + c2)
    return np.repeat(vals, cnts)


def wnorm(v, w):
    return float(np.sqrt(np.sum(w * v * v)))
