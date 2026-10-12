"""Item 2: shift attribution between periods A and B."""
import itertools, math, warnings
import numpy as np
from common import record, info, gl01, Pstar_matrix, Pstar_deriv_matrix, sample_lmoments
warnings.filterwarnings("ignore")
S = "v2"
rng = np.random.default_rng(7)
K, p, B = 4, 3, 5
uq, wq = gl01(16)  # exact for products of degree <= 31
Pq = Pstar_matrix(K, uq)
kk = 2 * np.arange(K) + 1

def model(scale_lvl=0.15):
    c0 = np.array([0.0, 4.0, 0.0, 0.0]) + rng.normal(size=K) * [1, .2, .1, .05]
    f = rng.normal(size=(p, B, K)) * np.array([1.0, scale_lvl, 0.08, 0.04])
    return c0, f

def coefs(c0, f, idx):
    return c0 + sum(f[j, idx[j]] for j in range(p))

def joint_cov(dep):
    if not dep:
        m = [rng.dirichlet(np.ones(B) * 2) for _ in range(p)]
        J = np.einsum('i,j,k->ijk', *m)
    else:
        J = rng.dirichlet(np.ones(B ** p) * 0.5).reshape((B,) * p)
    return J

cA, fA = model(); cB, fB = model()
# centre each shape function under the evaluation distribution P (here P_B) so the intercept carries the common shift
PB = joint_cov(dep=True); PA = joint_cov(dep=True)
def centre(c0, f, J):
    f = f.copy(); c0 = c0.copy()
    for j in range(p):
        mj = J.sum(axis=tuple(a for a in range(p) if a != j))
        mean = (mj[:, None] * f[j]).sum(0); f[j] -= mean; c0 += mean
    return c0, f
cA, fA = centre(cA, fA, PB); cB, fB = centre(cB, fB, PB)
dense = np.linspace(0, 1, 2001); Dd = Pstar_deriv_matrix(K, dense)
allmono = all((Dd @ coefs(c, f, idx)).min() > 0 for c, f in ((cA, fA), (cB, fB)) for idx in itertools.product(range(B), repeat=p))

# 2.1 exact conditional-shift decomposition by channel and feature (Shapley, intercept is player 0)
Dcond_quad, phi = 0.0, np.zeros((p + 1, K))
for idx in itertools.product(range(B), repeat=p):
    pr = PB[idx]
    dcurve = Pq @ (coefs(cB, fB, idx) - coefs(cA, fA, idx))
    Dcond_quad += pr * np.sum(wq * dcurve ** 2)
    a = np.vstack([cB - cA] + [fB[j, idx[j]] - fA[j, idx[j]] for j in range(p)])  # players x channels
    phi += pr * a * a.sum(0)[None, :] / kk[None, :]
record(S, "2.1 E_{x~P_B} W2^2(Q_A(.|x),Q_B(.|x)) = sum_{players,channels} E phi (exact, pre-projection)",
       abs(phi.sum() - Dcond_quad) < 1e-12 and allmono, f"|diff| {abs(phi.sum()-Dcond_quad):.1e} (tol 1e-12); total {Dcond_quad:.6f}; per-channel {np.round(phi.sum(0),5)}; per-player {np.round(phi.sum(1),5)}; all curves monotone={allmono}")

# helpers for marginal (mixture) quantile functions
ufine = (np.arange(4000) + 0.5) / 4000
Pf = Pstar_matrix(K, ufine)
def mixture_quantile(c0, f, J, u_out):
    curves, wts = [], []
    for idx in itertools.product(range(B), repeat=p):
        if J[idx] > 0:
            curves.append(Pf @ coefs(c0, f, idx)); wts.append(J[idx])
    curves = np.array(curves); wts = np.array(wts)
    ygrid = np.linspace(curves.min() - 1e-9, curves.max() + 1e-9, 60001)
    F = np.zeros_like(ygrid)
    for cq, wt in zip(curves, wts):
        F += wt * np.interp(ygrid, cq, ufine, left=0.0, right=1.0)
    F = np.maximum.accumulate(F)
    return np.interp(u_out, F, ygrid)

uo = (np.arange(2000) + 0.5) / 2000
QA = mixture_quantile(cA, fA, PB, uo); QB = mixture_quantile(cB, fB, PB, uo)
marg = np.mean((QA - QB) ** 2)
record(S, "2.2 same covariates: W2^2(F_A,F_B) <= E_x W2^2(conditionals) (mixture/joint convexity)", marg <= Dcond_quad + 1e-4,
       f"marginal {marg:.5f} <= conditional {Dcond_quad:.5f} (tol 1e-4 for the numerical mixture inversion)")
# counterexample: conditional shifts cancel in the mixture
from scipy import stats
zq = stats.norm.ppf(uo)
QA0, QA1, QB0, QB1 = zq, zq + 2, zq + 2, zq
def mix2(q0, q1):
    y = np.linspace(-10, 12, 200001)
    F = 0.5 * stats.norm.cdf(y - (q0 - zq)[0]) + 0.5 * stats.norm.cdf(y - (q1 - zq)[0])
    return np.interp(uo, F, y)
m2 = np.mean((mix2(QA0, QA1) - mix2(QB0, QB1)) ** 2); c2 = 0.5 * 4 + 0.5 * 4
record(S, "2.2b marginal shift is not an average of conditional shifts: swap example", m2 < 1e-10 and c2 == 4.0,
       f"marginal W2^2 = {m2:.1e}, mean conditional W2^2 = {c2}")

# 2.3 composition / structure decomposition with counterfactual C = (A model, B covariates)
QA_ = mixture_quantile(cA, fA, PA, uo); QB_ = mixture_quantile(cB, fB, PB, uo)
QC = mixture_quantile(cA, fA, PB, uo); QC2 = mixture_quantile(cB, fB, PA, uo)
tot = np.mean((QB_ - QA_) ** 2); comp = np.mean((QC - QA_) ** 2); struc = np.mean((QB_ - QC) ** 2)
cross = 2 * np.mean((QB_ - QC) * (QC - QA_))
# channel-wise exact linear split of L-moment differences (GL on the fine grid via midpoint, same rule both sides)
Pu = Pstar_matrix(K, uo)
dl = lambda q: (Pu * (1 / len(uo))).T @ q
lin_err = np.abs(dl(QB_ - QA_) - (dl(QB_ - QC) + dl(QC - QA_))).max()
v = {(): 0.0, ("cov",): comp, ("str",): np.mean((QC2 - QA_) ** 2), ("cov", "str"): tot}
sh_cov = 0.5 * (v[("cov",)] - v[()]) + 0.5 * (v[("cov", "str")] - v[("str",)])
sh_str = 0.5 * (v[("str",)] - v[()]) + 0.5 * (v[("cov", "str")] - v[("cov",)])
record(S, "2.3 Q_B-Q_A = (Q_B-Q_C)+(Q_C-Q_A) exactly; W2^2 = structure + composition + 2*cross; 2-player Shapley sums exactly",
       abs(tot - (comp + struc + cross)) < 1e-10 and lin_err < 1e-10 and abs(sh_cov + sh_str - tot) < 1e-12,
       f"total {tot:.5f} = struct {struc:.5f} + comp {comp:.5f} + cross {cross:+.5f}; Shapley cov {sh_cov:.5f}, struct {sh_str:.5f}")
info(S, "2.3 path dependence", f"composition first: comp={comp:.5f}; structure first: comp'={tot - np.mean((QC2-QA_)**2):.5f} (differ: DFL-style decompositions are path dependent)")

# 2.4 cross-fitting removes the noise bias of squared distances and of the Shapley products
R = 40000; a_true = rng.normal(size=4) * 0.3; sd = 0.25
e1 = rng.normal(size=(R, 4)) * sd; e2 = rng.normal(size=(R, 4)) * sd
h1 = a_true + e1; h2 = a_true + e2; full = a_true + (e1 + e2) / 2
plug = (full.sum(1)) ** 2; cf = h1.sum(1) * h2.sum(1); truth = a_true.sum() ** 2
phi_plug = full[:, 0] * full.sum(1); phi_cf = 0.5 * (h1[:, 0] * h2.sum(1) + h2[:, 0] * h1.sum(1)); phi_true = a_true[0] * a_true.sum()
se = lambda x: x.std() / math.sqrt(R)
ok = abs(cf.mean() - truth) < 4 * se(cf) and abs(phi_cf.mean() - phi_true) < 4 * se(phi_cf) and (plug.mean() - truth) > 10 * se(plug)
record(S, "2.4 cross-fit <D1,D2> is unbiased for ||D||^2 and for phi_j; the plug-in is biased upward by tr(Cov)", ok,
       f"truth {truth:.4f}: plug-in {plug.mean():.4f} (bias {(plug.mean()-truth):.4f}, theory {4*sd**2/2:.4f}), cross-fit {cf.mean():.4f}+-{se(cf):.4f}; phi_1 truth {phi_true:.4f}, cross-fit {phi_cf.mean():.4f}+-{se(phi_cf):.4f}")

# 2.5 permutation noise floor: exact level under exchangeability, power under a planted tail shift
nb, n = 4, 300
def stat(y, xb, lab):
    T = 0.0
    for b in range(nb):
        ya = y[(xb == b) & (lab == 0)]; yb = y[(xb == b) & (lab == 1)]
        la = sample_lmoments(ya, 4); lb = sample_lmoments(yb, 4)
        T += 7 * (la[3] - lb[3]) ** 2 * (len(ya) + len(yb)) / (2 * n)  # channel k=3 (L-kurtosis) energy
    return T
def draw(tail):
    x = rng.integers(0, nb, size=2 * n); lab = np.r_[np.zeros(n, int), np.ones(n, int)]
    zz = rng.standard_normal(2 * n)
    y = x * 0.3 + (1 + 0.2 * x) * zz + tail * lab * (x / nb) * (zz ** 3 - 3 * zz)
    return y, x, lab
def perm_p(y, x, lab, nperm=199):
    t0 = stat(y, x, lab); cnt = 0
    for _ in range(nperm):
        cnt += stat(y, x, rng.permutation(lab)) >= t0
    return (1 + cnt) / (1 + nperm)
reps = 300
rej0 = np.mean([perm_p(*draw(0.0)) <= 0.1 for _ in range(reps)])
lo, hi = 0.1 - 3 * math.sqrt(.09 / reps), 0.1 + 3 * math.sqrt(.09 / reps)
record(S, "2.5 permutation test of 'no shift' (labels exchangeable) has level alpha", lo <= rej0 <= hi,
       f"rejection rate {rej0:.3f} at alpha=0.10, {reps} null replicates, 199 permutations; band [{lo:.3f},{hi:.3f}]")
pw = np.mean([perm_p(*draw(0.12)) <= 0.1 for _ in range(100)])
info(S, "2.5 power", f"planted tail shift 0.12*(x/4)*He3, n=300 per period: power {pw:.2f} (descriptive)")

# 2.6 bootstrap noise floor tracks the true sampling SD of a feature's shift statistic
def phi_stat(y, x, lab):
    return stat(y, x, lab)
true_sd = np.std([phi_stat(*draw(0.12)) for _ in range(300)])
ratios = []
for _ in range(25):
    y, x, lab = draw(0.12)
    bs = []
    for _ in range(150):
        ia = rng.integers(0, n, n); ib = n + rng.integers(0, n, n); ii = np.r_[ia, ib]
        bs.append(phi_stat(y[ii], x[ii], lab[ii]))
    ratios.append(np.std(bs) / true_sd)
med = float(np.median(ratios))
record(S, "2.6 within-period row bootstrap SD of the statistic is within 35% of its Monte-Carlo SD (median over 25 datasets)",
       0.65 <= med <= 1.35, f"median ratio bootstrap SD / MC SD = {med:.2f} (iid rows; time-dependent rows need a block bootstrap)")
