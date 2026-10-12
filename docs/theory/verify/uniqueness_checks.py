"""Numerical checks for docs/theory/UNIQUENESS.md (numpy only).

Each check prints PASS/FAIL. U1-U7 check lemmas/theorem; X* check counterexamples
(each shows a rule satisfying all axioms but one, and that it differs from TRACE's rule).
Run: python docs/theory/verify/uniqueness_checks.py
"""
from itertools import combinations, permutations
from math import factorial

import numpy as np
from numpy.polynomial import legendre as L

rng = np.random.default_rng(20261012)
TOL = 1e-10
fails = []


def check(name, ok, info=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {info}")
    if not ok:
        fails.append(name)


# ---------- basis ----------
def pstar(k, u):
    c = np.zeros(k + 1); c[k] = 1
    return L.legval(2 * u - 1, c)


xg, wg = L.leggauss(64)
U, WU = (xg + 1) / 2, wg / 2  # Gauss-Legendre on [0,1]


def inner(f, g):
    return float(np.sum(WU * f * g))


# ---------- games ----------
def subsets(n):
    for r in range(n + 1):
        for S in combinations(range(n), r):
            yield frozenset(S)


def shapley(v, n):
    phi = np.zeros(n)
    for i in range(n):
        for S in subsets(n):
            if i in S:
                continue
            w = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
            phi[i] += w * (v(S | {i}) - v(S))
    return phi


def path(v, n, order=None):  # fixed-order marginal contributions
    order = order or list(range(n))
    phi = np.zeros(n); S = frozenset()
    for i in order:
        phi[i] = v(S | {i}) - v(S); S = S | {i}
    return phi


def equal_split(v, n):
    return np.full(n, (v(frozenset(range(n))) - v(frozenset())) / n)


def sq_prop(v, n):  # nonlinear: Shapley^2-proportional
    s = shapley(v, n); d = np.sum(s * s)
    return s if d == 0 else s * s * (v(frozenset(range(n))) - v(frozenset())) / d


def half(v, n):
    return 0.5 * shapley(v, n)


def stage1_trace(G):  # G[s,t]: s=covariate period, t=structure period (0=A,1=B)
    comp = 0.5 * ((G[1, 0] - G[0, 0]) + (G[1, 1] - G[0, 1]))
    struct = 0.5 * ((G[0, 1] - G[0, 0]) + (G[1, 1] - G[1, 0]))
    return np.array([comp, struct])


def game2(G):  # player 0 = comp (switch covariates), 1 = struct
    def v(S):
        return G[int(0 in S), int(1 in S)] - G[0, 0]
    return v


# ---------- U1: Parseval / Prop 1 ----------
K = 6
cA, cB = rng.normal(size=K), rng.normal(size=K)
QA = sum(cA[k] * pstar(k, U) for k in range(K)); QB = sum(cB[k] * pstar(k, U) for k in range(K))
w2 = inner(QB - QA, QB - QA)
check("U1 Parseval: W2^2 = sum (dc_k)^2/(2k+1)", abs(w2 - np.sum((cB - cA) ** 2 / (2 * np.arange(K) + 1))) < TOL)

# ---------- U2: 2-player Shapley = SHIFT_V2 formula, uniqueness via unanimity basis ----------
ok = True
for _ in range(200):
    G = rng.normal(size=(2, 2))
    ok &= np.allclose(shapley(game2(G), 2), stage1_trace(G), atol=TOL)
check("U2 stage-1 Shapley equals SHIFT_V2 comp/struct", ok)
# uniqueness: games on 2 players with v(0)=0 are a 3-dim space; axioms fix values on basis
basis = {"u1": lambda S: float(0 in S), "u2": lambda S: float(1 in S), "u12": lambda S: float(S == {0, 1})}
forced = {"u1": [1, 0], "u2": [0, 1], "u12": [0.5, 0.5]}
check("U2b Shapley on unanimity basis equals axiom-forced values",
      all(np.allclose(shapley(basis[b], 2), forced[b]) for b in basis))

# ---------- U3: Hilbert quadratic game Shapley = <a_i, sum a> ----------
ok = True
for _ in range(20):
    n = 5
    a = [sum(rng.normal() * pstar(k, U) for k in range(4)) for _ in range(n)]
    tot = sum(a)
    v = lambda S: inner(sum((a[i] for i in S), 0 * U), sum((a[i] for i in S), 0 * U))
    phi = shapley(v, n)
    ok &= np.allclose(phi, [inner(ai, tot) for ai in a], atol=1e-9)
check("U3 Shapley of ||sum_S a||^2 is <a_i, sum a>", ok)

# ---------- U4: orthogonal reduction -> phi_(k,i) = a_ki dc_k/(2k+1); channel efficiency ----------
Kc, n_per = 3, 3
A = rng.normal(size=(Kc, n_per))  # coefficient pieces a_{k,i}
players = [(k, i) for k in range(Kc) for i in range(n_per)]
vecs = [A[k, i] * pstar(k, U) for k, i in players]
v = lambda S: (lambda f: inner(f, f))(sum((vecs[m] for m in S), 0 * U))
phi = shapley(v, len(players))
dc = A.sum(1)
closed = np.array([A[k, i] * dc[k] / (2 * k + 1) for k, i in players])
check("U4 joint Shapley over (channel,player) = a_ki*dc_k/(2k+1)", np.allclose(phi, closed, atol=1e-9))
check("U4b per-channel efficiency is implied", np.allclose(phi.reshape(Kc, n_per).sum(1), dc ** 2 / (2 * np.arange(Kc) + 1), atol=1e-9))


# ---------- U5: full theorem on simulated per-feature 2x2 tables ----------
def trace_rule(Gt):  # Gt shape (K, p+1, 2, 2)
    Kk, P = Gt.shape[:2]
    a = np.array([[stage1_trace(Gt[k, j]) for j in range(P)] for k in range(Kk)])  # (K,P,2)
    dcb = np.array([Gt[k, :, 1, 1].sum() - Gt[k, :, 0, 0].sum() for k in range(Kk)])
    return a * (dcb / (2 * np.arange(Kk) + 1))[:, None, None], dcb


Gt = rng.normal(size=(4, 4, 2, 2)); Gt[:, 0, 1, :] = Gt[:, 0, 0, :]  # intercept: no covariate effect
phi, dcb = trace_rule(Gt)
W = np.sum(dcb ** 2 / (2 * np.arange(4) + 1))
check("U5 exact budget: sum of all pieces = barycenter W2^2", abs(phi.sum() - W) < TOL, f"gap={abs(phi.sum()-W):.1e}")
check("U5b intercept composition piece is 0", np.allclose(phi[:, 0, 0], 0))

# ---------- U6: invariances ----------
Gs = Gt[:, :, ::-1, ::-1]  # swap period labels
phis, _ = trace_rule(Gs)
check("U6 period swap leaves pieces unchanged", np.allclose(phis, phi, atol=TOL))
perm = [0, 3, 1, 2]
check("U6b feature relabelling is equivariant", np.allclose(trace_rule(Gt[:, perm])[0], phi[:, perm], atol=TOL))
sig, m = 2.7, 5.0
Gy = Gt * sig; Gy[0, 0] += m  # y -> sig*y + m : c_0 shifts by m in all four cells, all c scale by sig
check("U6c y-affine: pieces scale by sig^2", np.allclose(trace_rule(Gy)[0], sig ** 2 * phi, atol=1e-9))
check("U6d order symmetry: comp-first vs struct-first averaged == stage1", np.allclose(
    0.5 * (path(game2(Gt[1, 2]), 2, [0, 1]) + path(game2(Gt[1, 2]), 2, [1, 0])), stage1_trace(Gt[1, 2])))

# ---------- U7: marginal law: lambda_2 is not linear in mixtures; lambda_1 is ----------
def lmom(y):
    y = np.sort(y); n = len(y); i = np.arange(n)
    b0 = y.mean(); b1 = np.sum(i / (n - 1) * y) / n
    return b0, 2 * b1 - b0


N = 400_000
y1, y2 = rng.normal(0, 1, N), rng.normal(3, 1, N)
mix = np.concatenate([y1, y2])
l1 = [lmom(y) for y in (y1, y2, mix)]
check("U7 lambda_1 linear in mixture", abs(l1[2][0] - 0.5 * (l1[0][0] + l1[1][0])) < 1e-9)
gap = l1[2][1] - 0.5 * (l1[0][1] + l1[1][1])
check("U7b lambda_2 NOT linear in mixture (per-feature marginal split not exact)", abs(gap) > 0.1, f"gap={gap:.3f}")

# ---------- Counterexamples: each rule satisfies all axioms but one ----------
def axioms(rule, n, trials=60):
    E = S = Nl = A = True
    for _ in range(trials):
        t1 = {S_: rng.normal() for S_ in subsets(n)}; t1[frozenset()] = 0.0
        t2 = {S_: rng.normal() for S_ in subsets(n)}; t2[frozenset()] = 0.0
        v, w = t1.__getitem__, t2.__getitem__
        E &= abs(rule(v, n).sum() - v(frozenset(range(n)))) < 1e-9
        A &= np.allclose(rule(lambda T: v(T) + w(T), n), rule(v, n) + rule(w, n), atol=1e-9)
        # symmetric game in players 0,1
        sym = {S_: 0.5 * (t1[S_] + t1[frozenset({1 - x if x < 2 else x for x in S_})]) for S_ in subsets(n)}
        r = rule(sym.__getitem__, n); S &= abs(r[0] - r[1]) < 1e-9
        # player n-1 null
        nl = {S_: t1[S_ - {n - 1}] for S_ in subsets(n)}
        Nl &= abs(rule(nl.__getitem__, n)[n - 1]) < 1e-9
    return dict(E=bool(E), S=bool(S), N=bool(Nl), A=bool(A))


for n in (2, 3):
    check(f"X0 Shapley satisfies E,S,N,A (n={n})", all(axioms(shapley, n).values()))
    for nm, rule, drop in [("half (drop E)", half, "E"), ("fixed-order path (drop S)", path, "S"),
                           ("equal split (drop N)", equal_split, "N"), ("Shapley^2-proportional (drop A)", sq_prop, "A")]:
        ax = axioms(rule, n)
        check(f"X {nm} n={n}", (not ax[drop]) and all(v for k, v in ax.items() if k != drop), str(ax))

# X-F: dropping factorisation: Shapley of the single 'switch' game over (j,role) players differs
p_ = 2
Gf = rng.normal(size=(p_, 2, 2))  # one channel (k=1, weight 1/3), per-feature 2x2 tables with interaction
wk = 1 / 3
def switch(S):  # S over players 2j (comp of j), 2j+1 (struct of j)
    d = sum(Gf[j, int(2 * j in S), int(2 * j + 1 in S)] - Gf[j, 0, 0] for j in range(p_))
    return wk * d * d
sw = shapley(switch, 2 * p_)
a = np.concatenate([stage1_trace(Gf[j]) for j in range(p_)])
tr = wk * a * a.sum()
check("XF switch-game Shapley is efficient too", abs(sw.sum() - tr.sum()) < TOL)
check("XF ...but differs from TRACE when comp x struct interactions exist", np.max(abs(sw - tr)) > 1e-3, f"maxdiff={np.max(abs(sw-tr)):.3f}")
Gn = Gf.copy(); Gn[:, 1, 1] = Gn[:, 1, 0] + Gn[:, 0, 1] - Gn[:, 0, 0]  # no interaction
Gf_save, Gf = Gf, Gn
check("XF ...and coincides when interactions vanish", np.allclose(shapley(switch, 2 * p_), wk * np.concatenate([stage1_trace(Gf[j]) for j in range(p_)]) * sum(stage1_trace(Gf[j]).sum() for j in range(p_)), atol=1e-10))
Gf = Gf_save

# X-O: orthogonality needed: in the monomial basis the per-channel formula fails to sum to W2^2
cA, cB = rng.normal(size=3), rng.normal(size=3)
mono = [U ** k for k in range(3)]
d = sum((cB[k] - cA[k]) * mono[k] for k in range(3))
naive = sum((cB[k] - cA[k]) ** 2 * inner(mono[k], mono[k]) for k in range(3))
check("XO non-orthogonal basis: sum of per-channel terms != W2^2", abs(naive - inner(d, d)) > 1e-3, f"gap={naive-inner(d,d):.3f}")

print("\nALL PASS" if not fails else f"\nFAILED: {fails}")
raise SystemExit(1 if fails else 0)
