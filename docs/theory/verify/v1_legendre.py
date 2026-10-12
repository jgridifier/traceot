"""Item 1: shifted-Legendre / L-moment channel basis. Every check prints PASS/FAIL with its tolerance."""
import warnings; warnings.filterwarnings("ignore")
import itertools, math
import numpy as np
from scipy import stats, special, optimize
from common import (record, info, gl01, Pstar, Pstar_matrix, Pstar_deriv_matrix, quad01,
                    lmoments_from_Q, pav, wnorm)

S = "v1"
rng = np.random.default_rng(20261011)


def random_monotone_poly_coeffs(K):
    """Coefficients c_0..c_{K-1} of a strictly increasing polynomial Q in the P* basis (built from Q' = sum of squares + const)."""
    uu, ww = gl01(64)
    a = rng.normal(size=K - 1)
    dQ = 0.3 + np.polynomial.polynomial.polyval(uu, a) ** 2  # >0, degree 2(K-2) <= K-2? enforce by fitting
    # integrate to get Q on nodes, then least-squares project to degree K-1 and check monotone
    for _ in range(100):
        q = rng.normal() + np.cumsum(np.r_[0, np.diff(uu)] * dQ)
        P = Pstar_matrix(K, uu)
        c = np.linalg.lstsq(P * np.sqrt(ww)[:, None], q * np.sqrt(ww), rcond=None)[0]
        d = Pstar_deriv_matrix(K, np.linspace(0, 1, 2001)) @ c
        if d.min() > 0:
            return c
        a = 0.5 * a
        dQ = 0.3 + np.polynomial.polynomial.polyval(uu, a) ** 2
    raise RuntimeError("could not build monotone poly")


# 1.1 orthogonality, exact with Gauss-Legendre (n = K nodes integrate degree 2K-1 >= 2K-2)
K = 12
u, w = gl01(K)
P = Pstar_matrix(K, u)
G = (P * w[:, None]).T @ P
err = np.abs(G - np.diag(1.0 / (2 * np.arange(K) + 1))).max()
record(S, "1.1 orthogonality int P*_j P*_k = delta_jk/(2k+1) (GL, n=K)", err < 1e-13, f"max abs err {err:.2e} (tol 1e-13)")

# 1.2 Hosking indexing/sign: lambda_r = int Q P*_{r-1} vs order-statistic definition and known closed forms
def lam_orderstat(Q, r):
    # lambda_r = r^{-1} sum_{k=0}^{r-1} (-1)^k C(r-1,k) E X_{r-k:r};  E X_{j:r} = r!/((j-1)!(r-j)!) int Q u^{j-1}(1-u)^{r-j}
    tot = 0.0
    for k in range(r):
        j = r - k
        c = math.factorial(r) / (math.factorial(j - 1) * math.factorial(r - j))
        EX = c * quad01(lambda uu: Q(uu) * uu ** (j - 1) * (1 - uu) ** (r - j))
        tot += (-1) ** k * math.comb(r - 1, k) * EX
    return tot / r

dists = {
    "uniform": (lambda uu: uu, [0.5, 1 / 6, 0.0, 0.0]),
    "exponential": (lambda uu: -np.log1p(-uu), [1.0, 0.5, 1 / 6, 1 / 12]),
    "normal": (stats.norm.ppf, [0.0, 1 / math.sqrt(math.pi), 0.0, (30 / math.pi * math.atan(math.sqrt(2)) - 9) / math.sqrt(math.pi)]),
    "gumbel": (lambda uu: -np.log(-np.log(uu)), [np.euler_gamma, math.log(2), (2 * math.log(3) / math.log(2) - 3) * math.log(2), (16 - 10 * math.log(3) / math.log(2)) * math.log(2)]),
}
worst = 0.0
for name, (Q, known) in dists.items():
    lq = lmoments_from_Q(Q, 4)
    lo = np.array([lam_orderstat(Q, r) for r in range(1, 5)])
    worst = max(worst, np.abs(lq - known).max(), np.abs(lq - lo).max())
    info(S, f"1.2 L-moments {name}", "lambda1..4 = " + ", ".join(f"{v:.6f}" for v in lq))
record(S, "1.2 lambda_r = int Q P*_{r-1} matches Hosking order-statistic definition and closed forms",
       worst < 1e-7, f"max abs diff {worst:.2e} (tol 1e-7; uniform, exponential, normal, gumbel)")

# 1.3 expansion coefficients c_k = (2k+1) lambda_{k+1}
K = 6
c = random_monotone_poly_coeffs(K)
Qp = lambda uu: Pstar_matrix(K, np.atleast_1d(uu)) @ c
lam = np.array([quad01(lambda uu, k=k: float(Qp(uu)[0]) * Pstar(k, uu)) for k in range(K)])
err = np.abs(c - (2 * np.arange(K) + 1) * lam).max()
record(S, "1.3 c_k = (2k+1) lambda_{k+1}", err < 1e-11, f"max abs err {err:.2e} (tol 1e-11)")

# 1.4 Parseval on polynomial quantile functions, and 1.5 the refuted variant
c1, c2 = random_monotone_poly_coeffs(K), random_monotone_poly_coeffs(K)
dc = c1 - c2
uu, ww = gl01(K + 2)
W2sq = float(np.sum(ww * (Pstar_matrix(K, uu) @ dc) ** 2))
viaC = float(np.sum(dc ** 2 / (2 * np.arange(K) + 1)))
dlam = dc / (2 * np.arange(K) + 1)
viaL = float(np.sum((2 * np.arange(K) + 1) * dlam ** 2))
rel = max(abs(W2sq - viaC), abs(W2sq - viaL)) / W2sq
record(S, "1.4 W2^2 = sum dc_k^2/(2k+1) = sum (2k+1) dlambda_{k+1}^2", rel < 1e-12, f"W2^2={W2sq:.6f}, rel err {rel:.1e} (tol 1e-12)")
wrong = float(np.sum((2 * np.arange(K) + 1) * dc ** 2))
record(S, "1.5 'W2^2 = sum (2k+1) db_k^2' with b_k = expansion coefficients is FALSE (refutation reproduced)",
       abs(wrong - W2sq) / W2sq > 1e-2, f"that formula gives {wrong:.4f} vs true {W2sq:.4f}; correct iff b_k = lambda_{{k+1}}")

# 1.6 truncation: exact split = K channels + orthogonal residual (normal vs exponential)
Q1, Q2 = stats.norm.ppf, (lambda x: -np.log1p(-x))
D = lambda x: Q1(x) - Q2(x)
W2sq = quad01(lambda x: D(x) ** 2)
lamD = lmoments_from_Q(D, 10)
prev, ok, rows = np.inf, True, []
for KK in range(1, 11):
    ch = np.sum((2 * np.arange(KK) + 1) * lamD[:KK] ** 2)
    cK = (2 * np.arange(KK) + 1) * lamD[:KK]
    resid = quad01(lambda x: (D(x) - float((Pstar_matrix(KK, np.atleast_1d(x)) @ cK)[0])) ** 2)
    ok &= abs(ch + resid - W2sq) < 1e-8 and resid >= -1e-12 and resid <= prev + 1e-12
    prev = resid
    rows.append(f"K={KK}: channels {ch:.6f} resid {resid:.2e}")
record(S, "1.6 W2^2 = sum_{k<K}(2k+1)dlambda^2 + ||residual||^2, residual >=0 and nonincreasing in K",
       ok, f"W2^2={W2sq:.6f}; " + "; ".join(rows[i] for i in (0, 2, 4, 9)) + " (tol 1e-8)")

# 1.7 Shapley value of v(S) = (sum_{i in S} a_i)^2/(2k+1)
def shapley_bruteforce(vfun, p):
    phi = np.zeros(p)
    for i in range(p):
        others = [j for j in range(p) if j != i]
        for r in range(p):
            for Sset in itertools.combinations(others, r):
                wgt = math.factorial(r) * math.factorial(p - r - 1) / math.factorial(p)
                phi[i] += wgt * (vfun(set(Sset) | {i}) - vfun(set(Sset)))
    return phi

ok, neg_seen = True, False
for trial in range(50):
    p = int(rng.integers(2, 8)); kk = int(rng.integers(0, 5))
    a = rng.normal(size=p)
    v = lambda Sset: (sum(a[j] for j in Sset)) ** 2 / (2 * kk + 1)
    phi = shapley_bruteforce(v, p)
    closed = a * a.sum() / (2 * kk + 1)
    ok &= np.abs(phi - closed).max() < 1e-12 and abs(phi.sum() - v(set(range(p)))) < 1e-12
    neg_seen |= (closed < 0).any()
record(S, "1.7 Shapley of (sum a_i)^2/(2k+1) is a_j*(sum_i a_i)/(2k+1); efficiency exact", ok,
       f"50 random games, p in 2..7, brute force over all coalitions (tol 1e-12); negative values occur: {neg_seen}")

# 1.8 averaged attribution, exact by enumeration over a discrete product distribution (independent features)
p, levels = 3, 4
vals = [rng.normal(size=levels) for _ in range(p)]
probs = [rng.dirichlet(np.ones(levels)) for _ in range(p)]
vals = [v - (v * q).sum() for v, q in zip(vals, probs)]  # centered under the evaluation distribution
kk = 2
Ephi, Etot = np.zeros(p), 0.0
for idx in itertools.product(range(levels), repeat=p):
    pr = np.prod([probs[j][idx[j]] for j in range(p)])
    a = np.array([vals[j][idx[j]] for j in range(p)])
    Ephi += pr * a * a.sum() / (2 * kk + 1); Etot += pr * a.sum() ** 2 / (2 * kk + 1)
varj = np.array([(q * v * v).sum() for v, q in zip(vals, probs)]) / (2 * kk + 1)
ok = np.abs(Ephi - varj).max() < 1e-12 and abs(Ephi.sum() - Etot) < 1e-12
record(S, "1.8 independence + centering: E phi_jk = Var f_jk/(2k+1) (= functional-ANOVA share), sums to E dc_k^2/(2k+1)", ok,
       f"exact enumeration, max err {np.abs(Ephi - varj).max():.1e} (tol 1e-12)")
# dependent case: covariance terms appear, sums still exact
joint = rng.dirichlet(np.ones(levels ** 2)).reshape(levels, levels)
f0, f1 = rng.normal(size=levels), rng.normal(size=levels)
m0 = (joint.sum(1) * f0).sum(); m1 = (joint.sum(0) * f1).sum(); f0 -= m0; f1 -= m1
E0 = sum(joint[i, j] * f0[i] * (f0[i] + f1[j]) for i in range(levels) for j in range(levels))
E1 = sum(joint[i, j] * f1[j] * (f0[i] + f1[j]) for i in range(levels) for j in range(levels))
tot = sum(joint[i, j] * (f0[i] + f1[j]) ** 2 for i in range(levels) for j in range(levels))
cov = sum(joint[i, j] * f0[i] * f1[j] for i in range(levels) for j in range(levels))
record(S, "1.8b dependent features: E phi_j = Var f_j + Cov(f_j, f_other); still sums exactly", abs(E0 + E1 - tot) < 1e-12 and abs(cov) > 1e-3,
       f"Cov = {cov:.4f} (nonzero); |E0+E1-total| = {abs(E0+E1-tot):.1e}")

# 1.9 monotonicity K=3: exact iff |c2| <= c1/3  (equivalently |tau3| <= 1/5)
ug = np.linspace(0, 1, 4001)
D3 = Pstar_deriv_matrix(3, ug)
agree = 0; N = 20000
for _ in range(N):
    cc = rng.normal(size=3) * [1, 1, 0.5]
    grid_ok = (D3 @ cc).min() >= -1e-12
    rule = abs(cc[2]) <= cc[1] / 3 + 1e-12
    agree += grid_ok == rule
record(S, "1.9 K=3: Q nondecreasing iff |c2| <= c1/3 (iff |tau3| <= 1/5)", agree == N, f"{agree}/{N} random coefficient vectors agree with dense-grid check")

# 1.10 K=4 exact closed-form condition
def mono_K4(c):
    c1, c2, c3 = c[1], c[2], c[3]
    a0 = 2 * c1 - 6 * c2 + 12 * c3; a1 = 2 * c1 + 6 * c2 + 12 * c3  # Q'(0), Q'(1)
    if a0 < -1e-12 or a1 < -1e-12: return False
    if c3 > 0:
        us = 0.5 - c2 / (10 * c3)
        if 0 < us < 1:
            return 2 * c1 + c2 * (12 * us - 6) + c3 * (60 * us * us - 60 * us + 12) >= -1e-12
    return True
D4 = Pstar_deriv_matrix(4, ug)
agree = 0
for _ in range(N):
    cc = rng.normal(size=4) * [1, 1, 0.4, 0.3]
    agree += ((D4 @ cc).min() >= -1e-9) == mono_K4(cc)
record(S, "1.10 K=4: endpoint + interior-vertex closed form is exact", agree >= N - 2, f"{agree}/{N} agree (<=2 boundary ties allowed)")

# 1.11 Bernstein sufficient (linear) condition and degree elevation
def bernstein_coefs(c, K, elev=0):
    """Bernstein coefficients (degree d=K-2+elev) of Q'(u)."""
    d = K - 2 + elev
    # sample Q' at d+1 Chebyshev points and solve for Bernstein coefs
    us = 0.5 - 0.5 * np.cos(np.pi * (np.arange(d + 1) + 0.5) / (d + 1))
    B = np.stack([math.comb(d, i) * us ** i * (1 - us) ** (d - i) for i in range(d + 1)], 1)
    return np.linalg.solve(B, Pstar_deriv_matrix(K, us) @ c)
K = 5; Dk = Pstar_deriv_matrix(K, ug)
viol, mono_cnt, acc = 0, 0, {0: 0, 4: 0, 16: 0}
for _ in range(5000):
    cc = rng.normal(size=K) * [1, 1, 0.3, 0.2, 0.1]
    is_mono = (Dk @ cc).min() >= 0
    mono_cnt += is_mono
    for e in acc:
        if bernstein_coefs(cc, K, e).min() >= 0:
            acc[e] += 1
            viol += not is_mono
record(S, "1.11 Bernstein coefficients of Q' >= 0  =>  Q nondecreasing (sufficient, linear constraints)", viol == 0,
       f"0 violations expected, got {viol}; monotone cases {mono_cnt}/5000, certified by Bernstein at elevation 0/4/16: {acc[0]}/{acc[4]}/{acc[16]}")
lam_exp = np.array([1, .5, 1 / 6, 1 / 12, 1 / 20]);  cexp = (2 * np.arange(5) + 1) * lam_exp
min3 = (Pstar_deriv_matrix(3, ug) @ cexp[:3]).min(); min4 = (Pstar_deriv_matrix(4, ug) @ cexp[:4]).min()
record(S, "1.11b exponential law truncated at K=3 is not monotone (tau3=1/3>1/5); at K=4 it is", min3 < 0 and min4 > 0,
       f"min Q' at K=3: {min3:.3f}; at K=4: {min4:.3f}")

# 1.12 isotonic (PAV) projection: nonexpansive, triangle lower bound, explicit distance bound, cone characterization
Ng = 2000
ugrid = (np.arange(Ng) + 0.5) / Ng; wg = np.full(Ng, 1.0 / Ng)
ok = True; maxrel = 0.0
for _ in range(300):
    def curve():
        cc = rng.normal(size=7) * [1, 1, .6, .5, .4, .3, .2]
        return Pstar_matrix(7, ugrid) @ cc
    q1, q2 = curve(), curve()
    p1, p2 = pav(q1, wg), pav(q2, wg)
    pre, post = wnorm(q1 - q2, wg), wnorm(p1 - p2, wg)
    d1, d2 = wnorm(q1 - p1, wg), wnorm(q2 - p2, wg)
    def bound(q):
        dq = np.diff(q, prepend=q[0]); M = q[0] + np.cumsum(np.maximum(dq, 0))
        return wnorm(q - M, wg), np.sum(np.maximum(-dq, 0))
    b1, t1 = bound(q1); b2, t2 = bound(q2)
    ok &= post <= pre + 1e-12 and post >= pre - d1 - d2 - 1e-12
    ok &= d1 <= b1 + 1e-12 and b1 <= t1 + 1e-12 and d2 <= b2 + 1e-12
    ok &= abs(np.sum(wg * (q1 - p1) * p1)) < 1e-10 and np.sum(wg * (q1 - p1) * ugrid) <= 1e-12
    maxrel = max(maxrel, (pre - post) / max(pre, 1e-12))
record(S, "1.12 PAV = L2 projection on monotone cone: post<=pre, post>=pre-d1-d2, d<=||cum. neg. part||<=TV_-, Moreau orthogonality", ok,
       f"300 random degree-6 pairs on 2000-pt grid; largest relative shrink of distance {maxrel:.3f}")

# 1.13 discrete orthogonality of grids
def gram_offdiag(B, wts):
    G = (B * wts[:, None]).T @ B; d = np.sqrt(np.diag(G)); C = G / np.outer(d, d)
    return np.abs(C - np.eye(len(d))).max()
rows = []
for Ngr in (19, 31, 101):
    um = (np.arange(Ngr) + 0.5) / Ngr
    rows.append(f"midpoint N={Ngr}: Legendre K=4 max|corr| {gram_offdiag(Pstar_matrix(4, um), np.full(Ngr, 1/Ngr)):.2e}")
    z = stats.norm.ppf(um); H = np.c_[np.ones(Ngr), z, z ** 3 - 3 * z]
    rows.append(f"v0 Hermite {{1,z,He3}} corr(z,He3) {((H[:,1]*H[:,2]).mean()/np.sqrt((H[:,1]**2).mean()*(H[:,2]**2).mean())):+.4f}")
uG, wG = gl01(4)
gl_err = gram_offdiag(Pstar_matrix(4, uG), wG)
herm_cont = quad01(lambda x: stats.norm.ppf(x) * (stats.norm.ppf(x) ** 3 - 3 * stats.norm.ppf(x)))
record(S, "1.13 Gauss-Legendre (n>=K) makes the discrete Gram exactly diagonal; v0 Hermite basis is orthogonal in continuous L2[0,1]",
       gl_err < 1e-13 and abs(herm_cont) < 1e-8, f"GL n=4 max|corr| {gl_err:.1e}; continuous <z,He3> = {herm_cont:.1e}")
info(S, "1.13 grids", "; ".join(rows))

# 1.14 pinball curvature metric H_jk = int P_j P_k f(Q(u)) du is not diagonal
uq, wq = gl01(400)
def hess(Pm, fq):
    H = (Pm * (wq * fq)[:, None]).T @ Pm; d = np.sqrt(np.diag(H)); return H / np.outer(d, d)
fq_norm = stats.norm.pdf(stats.norm.ppf(uq)); fq_exp = 1 - uq
Cn = hess(Pstar_matrix(4, uq), fq_norm); Ce = hess(Pstar_matrix(4, uq), fq_exp)
zq = stats.norm.ppf(uq); Ch = hess(np.c_[np.ones_like(zq), zq, zq ** 3 - 3 * zq], fq_norm)
record(S, "1.14 pinball (CRPS) curvature metric weights u by f(Q(u)); it couples channels",
       abs(Cn[0, 2]) > 0.1 and abs(Ce[0, 1]) > 0.1 and abs(Ch[1, 2]) > 0.5,
       f"Gaussian, Legendre corr(P0,P2)={Cn[0,2]:+.3f}, corr(P1,P3)={Cn[1,3]:+.3f}; Exponential corr(P0,P1)={Ce[0,1]:+.3f}; Gaussian v0 Hermite corr(z,He3)={Ch[1,2]:+.4f} (=-sqrt(0.6)={-math.sqrt(.6):.4f})")

# 1.15 optimisation-path leakage: early-stopped gradient descent on the local quadratic, pure-scale truth
Hh = np.array([[quad01(lambda x, a=a, b=b: a(x) * b(x) * stats.norm.pdf(stats.norm.ppf(x)))
                for b in (lambda x: 1.0, stats.norm.ppf, lambda x: stats.norm.ppf(x) ** 3 - 3 * stats.norm.ppf(x))]
               for a in (lambda x: 1.0, stats.norm.ppf, lambda x: stats.norm.ppf(x) ** 3 - 3 * stats.norm.ppf(x))])
theta_star = np.array([0.0, 1.2, 0.0])
eta = 1.0 / np.linalg.eigvalsh(Hh).max()
th = np.zeros(3); path = []
for t in range(2000):
    th = th - eta * Hh @ (th - theta_star); path.append(th.copy())
path = np.array(path)
peak = np.abs(path[:, 2]).max(); final = abs(path[-1, 2])
newton = (np.zeros(3) - np.linalg.solve(Hh, Hh @ (np.zeros(3) - theta_star)))[2]
record(S, "1.15 early-stopped GD leaks a pure scale effect into He3; converged GD and a Newton step do not",
       peak > 0.05 and final < 1e-6 and abs(newton) < 1e-12,
       f"peak |shape coef| {peak:.3f} (scale truth 1.2) at step {np.abs(path[:,2]).argmax()+1}; after 2000 steps {final:.1e}; Newton {abs(newton):.1e}")

# 1.16 S1 truth in plain Legendre channels vs a reference-adapted orthonormal basis
z = stats.norm.ppf; he3 = lambda x: z(x) ** 3 - 3 * z(x)
lz, lh = lmoments_from_Q(z, 6), lmoments_from_Q(he3, 6)
E_scale = (2 * 3 + 1) * (1.2 * lz[3]) ** 2; E_shape = (2 * 3 + 1) * (0.12 * lh[3]) ** 2
E2s, E2h = 3 * (1.2 * lz[1]) ** 2, 3 * (0.12 * lh[1]) ** 2
record(S, "1.16 in plain Legendre channels the S1 scale feature out-weighs the He3 feature in the L-kurtosis channel (k=3)",
       E_shape / E_scale < 3, f"channel-3 energy over x in [0,1]: x_shape {E_shape:.5f}, x_scale {E_scale:.5f}, ratio {E_shape/E_scale:.3f} (<3: the S1 3x criterion is unattainable in this basis even with no noise); channel-1: x_scale {E2s:.4f}, x_shape {E2h:.5f}")
# adapted basis with reference shape Q_ref = z: e0=1, e1=z, e2.. = Gram-Schmidt of P*_2.. ; exact orthonormal, S1 scale stays in e1
uq, wq = gl01(3000)  # z is not polynomial; heavy GL to keep quadrature error small
basis = [np.ones_like(uq), z(uq)]
for k in range(2, 6):
    v = Pstar(k, uq).copy()
    for b in basis: v -= np.sum(wq * v * b) / np.sum(wq * b * b) * b
    basis.append(v)
E = np.stack([b / np.sqrt(np.sum(wq * b * b)) for b in basis], 1)
Gm = (E * wq[:, None]).T @ E
sc = (E * wq[:, None]).T @ (1.2 * z(uq)); sh = (E * wq[:, None]).T @ (0.12 * he3(uq))
record(S, "1.16b reference-adapted basis (e1 = Q_ref = z): orthonormal; the scale feature has zero energy in shape channels",
       np.abs(Gm - np.eye(6)).max() < 1e-10 and np.abs(sc[2:]).max() < 1e-6,
       f"max|Gram-I| {np.abs(Gm-np.eye(6)).max():.1e}; x_scale shape-channel loads {np.abs(sc[2:]).max():.1e}; x_shape loads on e2..e5 {np.round(sh[2:],4)}")
# reference = standardized S1 period-2 marginal (what a fitter would actually have)
M = 400000; X = rng.uniform(size=(M, 3)); Z = rng.standard_normal(M)
Y = 1.2 * X[:, 0] + (0.4 + 1.2 * X[:, 1]) * Z + 0.12 * X[:, 2] * (Z ** 3 - 3 * Z)
Qref = np.quantile(Y, uq)
basis = [np.ones_like(uq), Qref - np.sum(wq * Qref)]
for k in range(2, 6):
    v = Pstar(k, uq).copy()
    for b in basis: v -= np.sum(wq * v * b) / np.sum(wq * b * b) * b
    basis.append(v)
E = np.stack([b / np.sqrt(np.sum(wq * b * b)) for b in basis], 1)
sc = (E * wq[:, None]).T @ (1.2 * z(uq)); sh = (E * wq[:, None]).T @ (0.12 * he3(uq))
best = int(np.argmax(sh[2:] ** 2 / np.maximum(sc[2:] ** 2, 1e-300))) + 2
info(S, "1.16c Q_ref = S1 period-2 marginal", f"shape-channel energies e2..e5: x_scale {np.round(sc[2:]**2,5)}, x_shape {np.round(sh[2:]**2,5)}; total shape energy ratio x_shape/x_scale = {np.sum(sh[2:]**2)/np.sum(sc[2:]**2):.2f}")

# 1.17 misspecification: CRPS-projection != L2 (W2) projection; richer fit then report first K channels
s = 0.6; Qln = lambda x: np.exp(s * stats.norm.ppf(x)); Fln = lambda t: stats.norm.cdf(np.log(np.maximum(t, 1e-300)) / s)
uq, wq = gl01(600)
lam_ln = lmoments_from_Q(Qln, 8); cL2 = (2 * np.arange(8) + 1) * lam_ln
def crps_fit(Kf):
    Pm = Pstar_matrix(Kf, uq)
    g = lambda cc: (Pm * wq[:, None]).T @ (Fln(Pm @ cc) - uq)
    return optimize.fsolve(g, cL2[:Kf], xtol=1e-13)
c3 = crps_fit(3); c8 = crps_fit(8)
e3 = np.abs(c3 - cL2[:3]).max(); e8 = np.abs(c8[:3] - cL2[:3]).max()
record(S, "1.17 truncated pinball/CRPS fit is biased away from the W2 (L-moment) channels; fitting K_fit=8 and reporting 3 shrinks the bias",
       e3 > 1e-3 and e8 < e3 / 3, f"lognormal(0,{s}): max|c_CRPS,K=3 - c_L2| = {e3:.4f}; with K_fit=8 = {e8:.5f}")
# 1.16d reference = average L-scale-standardized conditional shape (estimable from a first-pass fit)
uq, wq = gl01(3000)
zz = z(uq); hh = zz ** 3 - 3 * zz
bb, cc_ = np.meshgrid((np.arange(50) + .5) / 50, (np.arange(50) + .5) / 50)
a_ = (0.4 + 1.2 * bb).ravel(); b_ = (0.12 * cc_).ravel()
l2 = a_ * lz[1] + b_ * lh[1]
Qref = np.mean((a_[:, None] * zz[None] + b_[:, None] * hh[None]) / l2[:, None], 0)
basis = [np.ones_like(uq), Qref - np.sum(wq * Qref)]
for k in range(2, 6):
    v = Pstar(k, uq).copy()
    for b in basis: v -= np.sum(wq * v * b) / np.sum(wq * b * b) * b
    basis.append(v)
E = np.stack([b / np.sqrt(np.sum(wq * b * b)) for b in basis], 1)
sc = (E * wq[:, None]).T @ (1.2 * zz); sh = (E * wq[:, None]).T @ (0.12 * hh)
ratio = np.sum(sh[2:] ** 2) / np.sum(sc[2:] ** 2)
record(S, "1.16d even a mean-standardised-conditional reference does not isolate S1's He3 feature (x_scale changes the standardised shape)",
       ratio < 3, f"total shape-channel energy ratio x_shape/x_scale = {ratio:.2f} (plain Legendre 0.75, marginal reference 1.67). "
       "An earlier hypothesis that this reference would reach >=3x was tested here and is false.")
# 1.16e Hermite-in-z basis: exact Parseval with weights m!, so S1 is separated exactly in that orthogonal basis
th1 = rng.normal(size=5); th2 = rng.normal(size=5)
Hm = lambda m, x: special.eval_hermitenorm(m, stats.norm.ppf(x))
lhs = quad01(lambda x: sum((th1[m] - th2[m]) * Hm(m, x) for m in range(5)) ** 2)
rhs = sum(math.factorial(m) * (th1[m] - th2[m]) ** 2 for m in range(5))
record(S, "1.16e v0's Hermite channels already satisfy W2^2 = sum_m m! dtheta_m^2 in continuous L2 (Parseval, weights m!)",
       abs(lhs - rhs) / rhs < 1e-8, f"quadrature {lhs:.8f} vs sum {rhs:.8f}")
