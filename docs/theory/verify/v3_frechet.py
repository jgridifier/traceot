"""Item 3: distribution-valued responses; Frechet regression in 1-D Wasserstein space."""
import math, warnings, itertools
import numpy as np
from scipy import stats
from common import record, info, gl01, Pstar, Pstar_matrix, sample_lmoments, pav, wnorm
warnings.filterwarnings("ignore")
S = "v3"
rng = np.random.default_rng(3)
K = 4; kk = 2 * np.arange(K) + 1

# 3.1 W2 Frechet objective == per-channel least squares on (2k+1)-scaled L-moment targets, weights 1/(2k+1), + constant
n, p, Bn = 120, 2, 4
X = rng.integers(0, Bn, size=(n, p)); w = rng.uniform(0.2, 2.0, n)
uq, wq = gl01(200); Pq = Pstar_matrix(K, uq)
loc = 0.5 * X[:, 0] - 0.3 * X[:, 1]; sc = 1 + 0.2 * X[:, 1]; sk = 0.2 * X[:, 0]
Qrows = loc[:, None] + sc[:, None] * (np.exp(sk[:, None] * stats.norm.ppf(uq)[None]) - 1) / np.maximum(sk[:, None], 1e-9) * (sk[:, None] > 0) \
        + sc[:, None] * stats.norm.ppf(uq)[None] * (sk[:, None] == 0)
# design for one channel: intercept + dummies (reference bin 0) per feature
D = np.column_stack([np.ones(n)] + [(X[:, j] == b).astype(float) for j in range(p) for b in range(1, Bn)])
# (a) direct minimisation of sum_i w_i int (Q_i - sum_k c_k(x_i) P*_k)^2 du : stacked weighted LS over (row, node)
big = np.kron(D, np.ones((len(uq), 1)))  # (n*nodes, d)
Areg = np.hstack([big * np.tile(Pq[:, k], n)[:, None] for k in range(K)])
sw = np.sqrt(np.repeat(w, len(uq)) * np.tile(wq, n))
theta_a = np.linalg.lstsq(Areg * sw[:, None], Qrows.ravel() * sw, rcond=None)[0].reshape(K, -1)
# (b) per-channel LS on targets t_ik = (2k+1) int Q_i P*_k, weights w_i/(2k+1) (irrelevant within a channel)
T = (Qrows * wq[None]) @ Pq * kk[None]
theta_b = np.stack([np.linalg.lstsq(D * np.sqrt(w)[:, None], T[:, k] * np.sqrt(w), rcond=None)[0] for k in range(K)])
obj_a = np.sum(w[:, None] * wq[None] * (Qrows - (D @ theta_a.T) @ Pq.T) ** 2)
resid_const = np.sum(w[:, None] * wq[None] * (Qrows - T @ Pq.T) ** 2)
obj_b = np.sum(w[:, None] * (T - D @ theta_b.T) ** 2 / kk[None]) + resid_const
record(S, "3.1 additive Frechet-W2 fit == K separate weighted LS fits on c_k targets; objective = sum_k SSR_k/(2k+1) + const",
       np.abs(theta_a - theta_b).max() < 1e-9 and abs(obj_a - obj_b) < 1e-9 * obj_a,
       f"max|theta diff| {np.abs(theta_a-theta_b).max():.1e}; objectives {obj_a:.8f} vs {obj_b:.8f}; constant (truncation residual) {resid_const:.5f}")

# 3.2 Hosking sample L-moments are unbiased; plug-in empirical-quantile coefficients are not
lam_exp = np.array([1.0, 0.5, 1 / 6, 1 / 12]); R = 200000
ok = True; rows = []
for m in (5, 20):
    xs = rng.exponential(size=(R, m))
    L = sample_lmoments(xs, 4)
    mean, se = L.mean(0), L.std(0) / math.sqrt(R)
    ok &= np.all(np.abs(mean - lam_exp) < 4 * se)
    # plug-in: int Qhat P*_k = sum_i x_(i) int_{(i-1)/m}^{i/m} P*_k
    edges = np.linspace(0, 1, m + 1)
    def antider(k, x):
        c = np.zeros(k + 1); c[k] = 1
        ic = np.polynomial.legendre.legint(c)
        return 0.5 * np.polynomial.legendre.legval(2 * x - 1, ic)
    Wt = np.stack([antider(k, edges[1:]) - antider(k, edges[:-1]) for k in range(4)], 1)
    plug = np.sort(xs, 1) @ Wt
    pb = plug.mean(0) - lam_exp; pse = plug.std(0) / math.sqrt(R)
    rows.append(f"m={m}: unbiased max|z|={np.max(np.abs(mean-lam_exp)/se):.2f}; plug-in bias {np.round(pb,4)} (z={np.round(pb/pse,0)})")
    if m == 5: ok &= abs(pb[1]) > 10 * pse[1]
record(S, "3.2 sample L-moments (Hosking 1990 PWM form) unbiased for m>=r; plug-in quantile-function coefficients biased O(1/m)", ok, "; ".join(rows))

# 3.3 consistency with FIXED per-row size m: n -> infinity suffices for LS on unbiased targets
def fit_err(nrows, m, unbiased, reps=60):
    errs = []
    for _ in range(reps):
        x = rng.integers(0, 3, nrows)
        scale = 1.0 + 0.5 * x + 0.1 * rng.standard_normal(nrows)
        xs = rng.exponential(size=(nrows, m)) * scale[:, None]
        if unbiased:
            t = sample_lmoments(xs, 2)[:, 1]
        else:
            q = np.sort(xs, 1); edges = np.linspace(0, 1, m + 1)
            t = q @ ((edges[1:] ** 2 - edges[1:]) - (edges[:-1] ** 2 - edges[:-1]))  # int Qhat (2u-1) du
        Dm = np.column_stack([np.ones(nrows), x == 1, x == 2]).astype(float)
        beta = np.linalg.lstsq(Dm, t, rcond=None)[0]
        truth = np.array([0.5, 0.25, 0.5])  # lambda2 = 0.5*scale for Exp
        errs.append(np.sum((beta - truth) ** 2))
    return math.sqrt(np.mean(errs))
e_small, e_big = fit_err(200, 5, True), fit_err(3200, 5, True)
p_big = fit_err(3200, 5, False)
record(S, "3.3 m_i = 5 fixed: error of LS on unbiased L-moment targets falls like n^-1/2; plug-in targets plateau at their bias",
       0.15 < e_big / e_small < 0.40 and p_big > 3 * e_big,
       f"RMSE n=200: {e_small:.4f}, n=3200: {e_big:.4f} (ratio {e_big/e_small:.3f}, sqrt-n predicts 0.25); plug-in at n=3200: {p_big:.4f}")

# 3.4 Bobkov-Ledoux type bound E W2^2(F_m, F) <= 2 J2(F)/(m+1); uniform J2 = 1/6
ok = True; rows = []
for m in (5, 20, 100):
    xs = np.sort(rng.uniform(size=(40000, m)), 1)
    a = np.arange(m) / m; b = (np.arange(m) + 1) / m
    # int_a^b (x-u)^2 du = ((x-a)^3 - (x-b)^3)/3
    W = np.sum(((xs - a) ** 3 - (xs - b) ** 3) / 3, 1)
    bound = 2 * (1 / 6) / (m + 1)
    ok &= W.mean() <= bound
    rows.append(f"m={m}: E W2^2={W.mean():.5f} <= {bound:.5f}")
record(S, "3.4 empirical quantile error E W2^2(F_m,F) <= 2 J2/(m+1) for the uniform law (constant from secondary source; checked here)", ok, "; ".join(rows))
mvals = (10, 100, 1000); vals = []
for m in mvals:
    xs = np.sort(rng.standard_normal(size=(4000, m)), 1)
    a = np.arange(m) / m; b = (np.arange(m) + 1) / m
    uu = (np.arange(400) + .5) / 400
    # fine numerical integral per cell
    Wn = []
    for i in range(m):
        uc = a[i] + (b[i] - a[i]) * uu
        Wn.append(((xs[:, [i]] - stats.norm.ppf(uc)[None]) ** 2).mean(1) * (b[i] - a[i]))
    vals.append(m * np.sum(Wn, 0).mean())
info(S, "3.4b normal law (J2 = infinity)", "m * E W2^2 = " + ", ".join(f"{v:.3f} (m={m})" for v, m in zip(vals, mvals)) + "; grows with m, so the 1/m rate fails for Gaussian rows")

# 3.5 unequal row sizes: GLS weights 1/(tau^2 + v/m_i) vs equal and proportional-to-m weights
tau, reps = 0.15, 3000
ms = np.array([5] * 40 + [50] * 40 + [500] * 40)
xcov = rng.uniform(size=len(ms))
v_l2 = sample_lmoments(rng.exponential(size=(200000, 50)), 2)[:, 1].var() * 50  # Var(l2) ~ v/m
ests = {"equal": [], "prop_m": [], "gls": []}
for _ in range(reps):
    truth = 0.5 + 0.4 * xcov + tau * rng.standard_normal(len(ms))  # row-level lambda2
    t = np.array([sample_lmoments(rng.exponential(size=mi) * 2 * lt, 2)[1] for mi, lt in zip(ms, truth)])
    Dm = np.column_stack([np.ones(len(ms)), xcov])
    for name, wt in (("equal", np.ones(len(ms))), ("prop_m", ms.astype(float)), ("gls", 1 / (tau ** 2 + v_l2 * (2 * truth.mean()) ** 2 / ms))):
        sw = np.sqrt(wt); ests[name].append(np.linalg.lstsq(Dm * sw[:, None], t * sw, rcond=None)[0][1])
var = {k: np.var(v) for k, v in ests.items()}
bias = {k: np.mean(v) - 0.4 for k, v in ests.items()}
record(S, "3.5 unequal m_i: inverse-variance weights 1/(tau_k^2 + v_k/m_i) beat equal and m-proportional weights; all unbiased",
       var["gls"] <= 1.05 * min(var["equal"], var["prop_m"]) and all(abs(b) < 4 * math.sqrt(var[k] / reps) for k, b in bias.items()),
       "slope variance: " + ", ".join(f"{k} {v:.2e}" for k, v in var.items()) + "; bias: " + ", ".join(f"{k} {b:+.4f}" for k, b in bias.items()))

# 3.6 projecting a fitted curve onto the monotone cone never moves it away from a monotone truth
Ng = 1000; ug = (np.arange(Ng) + .5) / Ng; wg = np.full(Ng, 1 / Ng); ok = True
for _ in range(500):
    truth = np.sort(rng.standard_normal(Ng)).cumsum() / Ng + stats.norm.ppf(ug)
    fit = truth + Pstar_matrix(6, ug) @ (rng.standard_normal(6) * 0.4)
    ok &= wnorm(pav(fit, wg) - truth, wg) <= wnorm(fit - truth, wg) + 1e-12
record(S, "3.6 if the target is monotone, ||Pi(Qhat) - Q|| <= ||Qhat - Q|| (fit-then-project is safe)", ok, "500 random cases")
