"""Item 4: Wasserstein-robust pinball fitting for a binned additive model."""
import itertools, math, warnings
import numpy as np
from scipy.optimize import linprog
from common import record, info
warnings.filterwarnings("ignore")
S = "v4"
rng = np.random.default_rng(44)
p, Bn, n, u = 2, 5, 12, 0.8
rho = lambda r, uu=u: np.maximum(uu * r, (uu - 1) * r)
rbar = max(u, 1 - u)
edges = np.linspace(0, 1, Bn + 1); width = edges[1] - edges[0]; centers = 0.5 * (edges[1:] + edges[:-1])
configs = list(itertools.product(range(Bn), repeat=p))

def instance(seed=None):
    r = np.random.default_rng(seed)
    f = r.normal(size=(p, Bn)) * [[1.0], [0.6]]
    x = r.uniform(size=(n, p)); b = np.minimum((x / width).astype(int), Bn - 1)
    y = f[0, b[:, 0]] + f[1, b[:, 1]] + r.normal(size=n)
    return f, x, b, y

def q_of(f, c):
    return sum(f[j, c[j]] for j in range(p))

def lp_worst(lossmat, dist, eps):
    """max (1/n) sum_ic pi_ic loss_ic  s.t. sum_c pi_ic = 1/n, sum pi_ic d_ic <= eps."""
    nn, C = lossmat.shape
    A_eq = np.zeros((nn, nn * C))
    for i in range(nn): A_eq[i, i * C:(i + 1) * C] = 1
    res = linprog(-lossmat.ravel(), A_ub=dist.ravel()[None], b_ub=[eps], A_eq=A_eq, b_eq=np.full(nn, 1.0 / nn),
                  bounds=(0, None), method="highs")
    return -res.fun

def dual_exact(lossmat, dist, eps, lam_min=0.0):
    """min_{lam>=lam_min} lam*eps + (1/n) sum_i max_c (loss_ic - lam d_ic); exact via breakpoints of a convex PL function."""
    base = lossmat[np.arange(len(lossmat)), 0:1]  # unused
    cand = [lam_min]
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(len(lossmat)):
            for c in range(lossmat.shape[1]):
                for c2 in range(lossmat.shape[1]):
                    dd = dist[i, c] - dist[i, c2]
                    if dd > 1e-15:
                        lv = (lossmat[i, c] - lossmat[i, c2]) / dd
                        if lv >= lam_min: cand.append(lv)
    cand = np.unique(cand)
    h = [l * eps + np.mean(np.max(lossmat - l * dist, 1)) for l in cand]
    return float(np.min(h))

# 4.1 ordinal-grid cost: exact LP == dual == separable inner maximisation
kap = np.array([width, width])
ok = True; rows = []
for seed in range(6):
    f, x, b, y = instance(seed)
    L = np.array([[rho(y[i] - q_of(f, c)) for c in configs] for i in range(n)])
    Dg = np.array([[np.sum(kap * np.abs(np.array(c) - b[i])) for c in configs] for i in range(n)])
    # separable evaluation of max_c (loss - lam d) for a given lam
    lam = 0.37
    sep = []
    for i in range(n):
        best = -np.inf
        for s in (u, u - 1):
            val = s * y[i] + sum(np.max(-s * f[j] - lam * kap[j] * np.abs(np.arange(Bn) - b[i, j])) for j in range(p))
            best = max(best, val)
        sep.append(best)
    ok &= np.allclose(sep, np.max(L - lam * Dg, 1), atol=1e-12)
    for eps in (0.005, 0.02, 0.1, 0.5):
        a, d = lp_worst(L, Dg, eps), dual_exact(L, Dg, eps)
        ok &= abs(a - d) < 1e-8
        if seed == 0: rows.append(f"eps={eps}: {a:.6f}")
record(S, "4.1 grid cost: worst-case pinball risk (primal LP) = 1-D dual = separable per-feature inner max", ok, "seed 0 values " + "; ".join(rows) + " (tol 1e-8)")

# 4.2 Lipschitz-type upper bound with the max adjacent jump (grid-Lipschitz constant), and how loose it is
ok = True; ratios = []
for seed in range(6):
    f, x, b, y = instance(seed)
    L = np.array([[rho(y[i] - q_of(f, c)) for c in configs] for i in range(n)])
    Dg = np.array([[np.sum(kap * np.abs(np.array(c) - b[i])) for c in configs] for i in range(n)])
    emp = np.mean(L[np.arange(n), [configs.index(tuple(bb)) for bb in b]])
    G = np.abs(np.diff(f, axis=1)).max(1)
    for eps in (0.005, 0.02, 0.1, 0.5):
        exact = dual_exact(L, Dg, eps); ub = emp + eps * rbar * np.max(G / kap)
        ok &= exact <= ub + 1e-10
        ratios.append((exact - emp) / (ub - emp))
record(S, "4.2 R(eps) <= emp + eps*rbar*max_j G_j/kappa_j (G_j = max adjacent jump); an upper bound, not exact", ok,
       f"(exact-emp)/(bound-emp) ranges {min(ratios):.2f}..{max(ratios):.2f} over 24 cases")

# 4.3 raw-x cost: exact via point-to-bin distances; slope blows up as a point approaches an edge (no Lipschitz equivalence)
def dist_to_bin(xv, bb):
    lo, hi = edges[bb], edges[bb + 1]
    return 0.0 if lo <= xv <= hi else min(abs(xv - lo), abs(xv - hi))
f, x, b, y = instance(3)
ok = True; slopes = []
for delta in (0.05, 0.01, 0.002, 0.0004):
    x2 = x.copy(); x2[0, 0] = edges[2] - delta; b2 = np.minimum((x2 / width).astype(int), Bn - 1)
    y2 = y.copy(); y2[0] = q_of(f, b2[0]) + (1.0 if f[0, 2] < f[0, 1] else -1.0)  # crossing the edge raises the loss
    L = np.array([[rho(y2[i] - q_of(f, c)) for c in configs] for i in range(n)])
    Dx = np.array([[sum(dist_to_bin(x2[i, j], c[j]) for j in range(p)) for c in configs] for i in range(n)])
    emp = np.mean(L[np.arange(n), [configs.index(tuple(bb)) for bb in b2]])
    e0 = 1e-6
    a, d = lp_worst(L, Dx, e0), dual_exact(L, Dx, e0)
    ok &= abs(a - d) < 1e-9
    slopes.append((d - emp) / e0)
    i0 = 0; cfg = list(b2[0]); cfg[0] = 2
    gain = rho(y2[0] - q_of(f, cfg)) - rho(y2[0] - q_of(f, b2[0]))
    ok &= slopes[-1] >= gain / delta - 1e-6
ok &= slopes[-1] > 10 * slopes[0]
record(S, "4.3 raw-x transport cost: exact LP == dual with point-to-bin distances; initial slope ~ jump/delta grows without bound",
       ok, "slope dR/deps at eps->0 for delta=0.05,0.01,0.002,0.0004: " + ", ".join(f"{s:.1f}" for s in slopes))

# 4.4 outcome perturbation: W1 on y alone adds exactly eps*rbar/kappa_y (a constant: no regularisation of the fit)
f, x, b, y = instance(1)
ky = 1.0; ok = True
cur = np.array([q_of(f, bb) for bb in b])
emp = np.mean(rho(y - cur))
for T in (0.5, 5.0, 50.0):
    cand = np.stack([y, y + T, y - T], 1)
    L = rho(cand - cur[:, None]); Dy = ky * np.abs(cand - y[:, None])
    for eps in (0.01, 0.1):
        val = lp_worst(L, Dy, eps)
        ok &= abs(val - (emp + eps * rbar / ky)) < 1e-9 or (T == 0.5 and val <= emp + eps * rbar / ky + 1e-12)
record(S, "4.4 outcome-only W1 ball: worst case = emp + eps*max(u,1-u)/kappa_y exactly (independent of the model)", ok,
       f"emp {emp:.5f}; any point with a residual of the right sign attains rate rbar={rbar}")
# joint (x,y), additive cost: exact value = min over lam >= rbar/ky of the covariate dual
f, x, b, y = instance(2)
Lx = np.array([[rho(y[i] - q_of(f, c)) for c in configs] for i in range(n)])
Dg = np.array([[np.sum(kap * np.abs(np.array(c) - b[i])) for c in configs] for i in range(n)])
ok = True; rows = []
for ky, eps in ((1.0, 0.01), (1.0, 0.4), (0.05, 0.01), (0.05, 0.4)):
    formula = dual_exact(Lx, Dg, eps, lam_min=rbar / ky)
    T = 40.0
    Lj = np.concatenate([np.array([[rho(y[i] + s - q_of(f, c)) for c in configs] for i in range(n)]) for s in (0, T, -T)], 1)
    Dj = np.concatenate([Dg + ky * abs(s) for s in (0, T, -T)], 1)
    val = lp_worst(Lj, Dj, eps)
    ok &= abs(val - formula) < 1e-7
    rows.append(f"kappa_y={ky}, eps={eps}: LP {val:.6f} formula {formula:.6f} covariate-only {dual_exact(Lx, Dg, eps):.6f}")
record(S, "4.4b joint (x,y) W1 with cost kappa.|db| + kappa_y|dy|: R = min_{lam>=rbar/kappa_y} covariate dual", ok, "; ".join(rows))

# 4.5 W-infinity (each row moved at most r bins per feature and |dy|<=delta): closed form vs brute force; TV-type bound
f, x, b, y = instance(5); r, dlt = 1, 0.1
ok = True; exact_sum = 0.0; tv_sum = 0.0
for i in range(n):
    wins = [range(max(0, b[i, j] - r), min(Bn, b[i, j] + r + 1)) for j in range(p)]
    brute = max(rho(y[i] + s - q_of(f, c)) for c in itertools.product(*wins) for s in (-dlt, 0, dlt))
    qmin = sum(min(f[j, k] for k in wins[j]) for j in range(p)); qmax = sum(max(f[j, k] for k in wins[j]) for j in range(p))
    closed = max(rho(y[i] + dlt - qmin), rho(y[i] - dlt - qmax))
    ok &= abs(brute - closed) < 1e-12
    exact_sum += closed
    tv = sum(np.sum(np.abs(np.diff(f[j, list(wins[j])]))) for j in range(p))
    tv_sum += rho(y[i] - q_of(f, b[i])) + rbar * (dlt + tv)
ok &= exact_sum <= tv_sum + 1e-12
record(S, "4.5 W-inf box: exact worst case uses per-feature window min/max (O(npB)); window total variation gives an upper bound", ok,
       f"exact mean {exact_sum/n:.5f} <= TV bound {tv_sum/n:.5f}")

# 4.6 the exact W1-robust risk is convex in the shape-function values (LP-representable)
def robust_risk(fvec, b, y, eps):
    f = fvec.reshape(p, Bn)
    L = np.array([[rho(y[i] - q_of(f, c)) for c in configs] for i in range(n)])
    Dg = np.array([[np.sum(kap * np.abs(np.array(c) - b[i])) for c in configs] for i in range(n)])
    return dual_exact(L, Dg, eps)
f, x, b, y = instance(9); ok = True; worst = -np.inf
for _ in range(60):
    a1, a2 = rng.normal(size=p * Bn), rng.normal(size=p * Bn)
    mid = robust_risk((a1 + a2) / 2, b, y, 0.05); avg = (robust_risk(a1, b, y, 0.05) + robust_risk(a2, b, y, 0.05)) / 2
    worst = max(worst, mid - avg); ok &= mid <= avg + 1e-10
record(S, "4.6 exact W1-robust pinball risk is convex in theta (midpoint test, 60 random pairs)", ok, f"max(mid - avg) = {worst:.2e}")

# 4.7 sum of marginal W1's is a LOWER bound on the joint W1 with additive cost (calibrating eps from marginals under-covers)
ok = True; gaps = []
for _ in range(60):
    P1 = rng.dirichlet(np.ones(Bn * Bn)); P2 = rng.dirichlet(np.ones(Bn * Bn))
    cost = np.array([[np.sum(kap * np.abs(np.array(c1) - np.array(c2))) for c2 in configs] for c1 in configs])
    C = len(configs)
    A_eq = np.vstack([np.kron(np.eye(C), np.ones(C)), np.kron(np.ones(C), np.eye(C))])
    res = linprog(cost.ravel(), A_eq=A_eq, b_eq=np.r_[P1, P2], bounds=(0, None), method="highs")
    joint = res.fun
    m1 = P1.reshape(Bn, Bn); m2 = P2.reshape(Bn, Bn)
    marg = sum(kap[j] * np.sum(np.abs(np.cumsum(m1.sum(1 - j)) - np.cumsum(m2.sum(1 - j)))) for j in range(p))
    ok &= marg <= joint + 1e-9; gaps.append(joint / max(marg, 1e-12))
record(S, "4.7 sum_j kappa_j W1(marginal_j) <= joint W1 (additive cost)", ok, f"joint/marginal-sum ratio range {min(gaps):.2f}..{max(gaps):.2f}")

# 4.8 integrated (multi-level) pinball: per-level worst cases summed is an upper bound on the common-adversary worst case
us = np.array([0.1, 0.3, 0.5, 0.7, 0.9]); f_u = rng.normal(size=(len(us), p, Bn)) * 0.5 + (np.arange(len(us)) - 2)[:, None, None] * 0.6
f, x, b, y = instance(11)
Dg = np.array([[np.sum(kap * np.abs(np.array(c) - b[i])) for c in configs] for i in range(n)])
Lc = sum(np.array([[rho(y[i] - sum(f_u[t, j, c[j]] for j in range(p)), us[t]) for c in configs] for i in range(n)]) for t in range(len(us))) / len(us)
ok = True
for eps in (0.01, 0.1):
    common = lp_worst(Lc, Dg, eps)
    sep = np.mean([lp_worst(np.array([[rho(y[i] - sum(f_u[t, j, c[j]] for j in range(p)), us[t]) for c in configs] for i in range(n)]), Dg, eps) for t in range(len(us))])
    ok &= common <= sep + 1e-10
record(S, "4.8 integrated loss: common-adversary worst case <= average of per-level worst cases", ok, f"eps=0.1: common {common:.5f} <= separate {sep:.5f}")

# 4.9 W2 ball on the covariate grid: same finite LP with cost d^2 and budget eps^2 (exact); it is dominated by the W1 value
ok = True; rows = []
for seed in range(4):
    f, x, b, y = instance(20 + seed)
    L = np.array([[rho(y[i] - q_of(f, c)) for c in configs] for i in range(n)])
    Dg = np.array([[np.sum(kap * np.abs(np.array(c) - b[i])) for c in configs] for i in range(n)])
    for eps in (0.02, 0.1, 0.3):
        a2, d2 = lp_worst(L, Dg ** 2, eps ** 2), dual_exact(L, Dg ** 2, eps ** 2)
        w1 = dual_exact(L, Dg, eps)
        ok &= abs(a2 - d2) < 1e-8 and d2 <= w1 + 1e-10
        if seed == 0: rows.append(f"eps={eps}: W2 {d2:.5f} <= W1 {w1:.5f}")
record(S, "4.9 covariate W2 ball on the grid: exact LP == dual with cost d^2, budget eps^2; value <= W1-ball value", ok, "; ".join(rows) + " (tol 1e-8)")

# 4.10 outcome-only W2 ball: exact value depends on the residuals (unlike W1); bounded by emp + rbar*eps/kappa_y
ky = 1.0; Dl = np.linspace(-6, 6, 1201); ok = True; rows = []
for name, shift in (("fit", 0.0), ("fit+0.8", 0.8)):
    f, x, b, y = instance(1)
    cur = np.array([q_of(f, bb) for bb in b]) + shift
    emp = np.mean(rho(y - cur))
    L = rho((y[:, None] + Dl[None]) - cur[:, None]); Dc = (ky * Dl[None]) ** 2 * np.ones((n, 1))
    for eps in (0.05, 0.3):
        a, d = lp_worst(L, Dc, eps ** 2), dual_exact(L, Dc, eps ** 2)
        ok &= abs(a - d) < 1e-7 and a <= emp + rbar * eps / ky + 1e-9
        rows.append(f"{name} eps={eps}: gain {a-emp:.5f} (W1 gain {rbar*eps/ky:.5f})")
gains = [float(r.split("gain ")[1].split(" ")[0]) for r in rows]
ok &= abs(gains[0] - gains[2]) > 1e-3
record(S, "4.10 outcome-only W2 ball: exact LP == dual; gain <= rbar*eps/kappa_y and depends on the model's residuals (not a constant)", ok, "; ".join(rows))
