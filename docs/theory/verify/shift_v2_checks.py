"""Numeric checks for docs/theory/SHIFT_V2.md (numpy only). Run: python shift_v2_checks.py"""
import numpy as np
from numpy.polynomial import legendre as L
rng = np.random.default_rng(20261012)
K = 6
U, WQ = L.leggauss(400); U = (U + 1) / 2; WQ = WQ / 2          # Gauss-Legendre on [0,1]
P = np.stack([L.legval(2*U-1, np.eye(K)[k]) for k in range(K)])  # shifted Legendre P*_k(u)

def lam(Q):  # L-moments lambda_{k+1} = int Q P*_k du
    return P @ (WQ * Q)

ok = True
def check(name, err, tol):
    global ok; good = err <= tol; ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] {name}: err={err:.3e} tol={tol:.0e}")

# C1: W2^2 = sum (2k+1) dLambda^2 for curves in span(P*_0..P*_{K-1})
cA, cB = rng.normal(size=K), rng.normal(size=K)
QA, QB = cA @ P, cB @ P
w2 = np.sum(WQ * (QA-QB)**2)
dl = lam(QA) - lam(QB)
check("C1 W2^2 Legendre identity", abs(w2 - np.sum((2*np.arange(K)+1)*dl**2)), 1e-10)

# C2: quadratic-game Shapley phi_i = a_i*sum(a) (brute force over permutations)
from itertools import permutations
from math import factorial
a = rng.normal(size=4)
v = lambda S: sum(a[list(S)])**2
phi = np.zeros(4)
for perm in permutations(range(4)):
    S = []
    for i in perm:
        phi[i] += v(S+[i]) - v(S); S.append(i)
phi /= factorial(4)
check("C2 quadratic Shapley closed form", np.max(abs(phi - a*a.sum())), 1e-12)

# C3: two-ordering Shapley split exact & order-free (scalar per channel)
lA, lB, lC, lCp = rng.normal(size=(4, K))   # A, B, C=(struct B, x A), C'=(struct A, x B)
comp = 0.5*((lB-lC)+(lCp-lA)); struct = 0.5*((lC-lA)+(lB-lCp))
check("C3 comp+struct = total", np.max(abs(comp+struct-(lB-lA))), 1e-12)
comp2 = 0.5*((lCp-lA)+(lB-lC)); check("C3 order invariance (swap roles)", np.max(abs(comp2-comp)), 1e-12)

# C4: barycenter (Vincent average) linearity + per-feature additive split sums to W2^2
n, p = 3000, 5
def f(X, B):  # additive channels: c_k(x) = b0_k + sum_j B[k,j]*tanh(x_j)
    return B[:, :1].T + np.tanh(X) @ B[:, 1:].T
XA = rng.normal(size=(n, p)); XB = rng.normal(0.3, 1.1, size=(n, p))
BA = rng.normal(size=(K, p+1)) * 0.3; BB = BA + rng.normal(size=(K, p+1)) * 0.1
g = lambda X, B: np.tanh(X).mean(0)[None, :] * B[:, 1:]   # per-feature mean contributions (K,p)
comp_j = 0.5*((g(XB,BA)-g(XA,BA)) + (g(XB,BB)-g(XA,BB)))
str_j = 0.5*((g(XA,BB)-g(XA,BA)) + (g(XB,BB)-g(XB,BA)))
str_0 = BB[:, 0]-BA[:, 0]
a_all = np.concatenate([comp_j, str_j, str_0[:, None]], 1)       # coefficient-scale players
dc = f(XB, BB).mean(0) - f(XA, BA).mean(0)
check("C4 per-feature players sum to channel delta", np.max(abs(a_all.sum(1)-dc)), 1e-12)
wk = 2*np.arange(K)+1
phi = a_all * a_all.sum(1, keepdims=True) / wk[:, None]
# W2^2 between Vincent averages, computed directly on the quantile grid
Qa = f(XA, BA).mean(0) @ P; Qb = f(XB, BB).mean(0) @ P
check("C4 sum phi = W2^2(barycenters)", abs(phi.sum() - np.sum(WQ*(Qa-Qb)**2)), 1e-10)

# C5: DFL reweighting recovers counterfactual law F_C = int F_B(y|x) dF_A(x)
N = 200000
xa = rng.normal(0, 1, N); xb = rng.normal(0.5, 1, N)
yb = 2*xb + rng.normal(size=N)
# true odds p(A|x)/p(B|x) for these Gaussians: exp(-(x^2)/2 + (x-.5)^2/2)
psi = np.exp(-(xb**2)/2 + (xb-0.5)**2/2); psi /= psi.mean()
# logistic fit (Newton) on pooled sample with 2-fold cross-fitting
def logit_fit(x, t):
    Xd = np.c_[np.ones_like(x), x]; b = np.zeros(2)
    for _ in range(25):
        pr = 1/(1+np.exp(-Xd@b)); H = Xd.T @ (Xd*(pr*(1-pr))[:, None]); b += np.linalg.solve(H, Xd.T@(t-pr))
    return b
xs = np.r_[xa, xb]; t = np.r_[np.ones(N), np.zeros(N)]; fold = rng.integers(0, 2, 2*N)
psi_hat = np.empty(N)
for k in (0, 1):
    b = logit_fit(xs[fold != k], t[fold != k]); m = fold[N:] == k
    psi_hat[m] = np.exp(b[0] + b[1]*xb[m])
psi_hat /= psi_hat.mean()
qs = np.linspace(0.05, 0.95, 19)
def wq(y, w, q):
    o = np.argsort(y); cw = np.cumsum(w[o]); return np.interp(q*cw[-1], cw, y[o])
truth = np.quantile(2*xa + rng.normal(size=N), qs)
check("C5 DFL counterfactual quantiles (true odds)", np.max(abs(wq(yb, psi, qs)-truth)), 0.03)
check("C5 DFL counterfactual quantiles (cross-fit logit)", np.max(abs(wq(yb, psi_hat, qs)-truth)), 0.03)

# C6: entry/exit decomposition of barycenter is exact (linear in unit weights)
cs_A, cs_B, cx, cn = rng.normal(size=(4, K)); eA, eB = 0.2, 0.3
cbA = (1-eA)*cs_A + eA*cx; cbB = (1-eB)*cs_B + eB*cn
stay = 0.5*((1-eA)+(1-eB))*(cs_B-cs_A)
mix = 0.5*(cs_A+cs_B)*((1-eB)-(1-eA))
entry = eB*cn; exit_ = -eA*cx
check("C6 stayers+entry+exit = total", np.max(abs(stay+mix+entry+exit_-(cbB-cbA))), 1e-12)

print("ALL PASS" if ok else "SOME FAILED"); raise SystemExit(0 if ok else 1)
