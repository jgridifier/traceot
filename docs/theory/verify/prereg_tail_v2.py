"""PREREG_TAIL_v2: population pinball oracles in TRACE v1's span, separation by basis, power/size, bars.
numpy/scipy only. Does not run TRACE. Output: verify/PREREG_TAIL_v2_RESULTS.txt (via tee)."""
import sys, os, time, math, hashlib
sys.path.insert(0, os.path.dirname(__file__))
from prereg_core import *
from scipy.stats import binom

np.set_printoptions(linewidth=160)
TOL_PAR, TOL_EREL, TOL_EABS = 5e-5, 1e-3, 1e-8
BASES = ("legendre", "hermite", "reference")
GM = {b: report_matrix(b) for b in BASES}
NAMES = ["x_loc", "x_scale", "x_shape", "noise1", "noise2", "noise3", "noise4", "noise5"]
print("PREREG_TAIL_v2 computation, run", time.strftime("%Y-%m-%d %H:%M %Z"))
print(f"fitter span: K_fit={K_FIT}, n_u={N_U} Gauss-Legendre nodes on (0,1), weights = GL weights (no extra tail weight), "
      f"{N_BINS} equal-mass bins, K_report={K_REP}, report matrix by {N_FINE}-node GL")

# ---------------- A0. the disputed unconditional example ----------------
mu_, sig_, s_ = 0.0, 0.8, 0.12
def L0(a): return np.sum(WN * exp_pinball(EN @ a, UN, mu_, sig_, s_))
def G0(a): F, _ = cdf_pdf(EN @ a, mu_, sig_, s_); return EN.T @ (WN * (F - UN))
a_start = (EN * WN[:, None]).T @ quantile_true(UN, mu_, sig_, s_)
rA = optimize.minimize(L0, a_start, jac=G0, method="BFGS", options={"gtol": 1e-13})
a_new = a_start.copy()
for _ in range(50):  # independent: Newton on the first-order condition
    F, f = cdf_pdf(EN @ a_new, mu_, sig_, s_)
    a_new -= np.linalg.solve((EN * (WN * f)[:, None]).T @ EN, EN.T @ (WN * (F - UN)))
h_bfgs, h_newt = (rA.x @ GM["hermite"])[3], (a_new @ GM["hermite"])[3]
l2_exact = ((EF * WF[:, None]).T @ quantile_true(UF, mu_, sig_, s_)) @ GM["hermite"]
l2_nodes = ((EN * WN[:, None]).T @ quantile_true(UN, mu_, sig_, s_)) @ GM["hermite"]
print("\n[A0] Y = 0.8 z + 0.12 He3(z), intercept only, reported Hermite channel 3 (orthonormal He3/sqrt6):")
print(f"  population pinball optimum: BFGS {h_bfgs:.6f}, Newton {h_newt:.6f} (|diff| {abs(h_bfgs-h_newt):.1e})")
print(f"  L2 projection into the span then reported: exact {l2_exact[3]:.6f}; 12-node quadrature {l2_nodes[3]:.6f}; "
      f"truth (no span restriction) {0.12*math.sqrt(6):.6f}")

# ---------------- A. oracles for the three designs, two methods ----------------
D_alt = Design("S1-alt", lambda X: 1.2 * X[:, 0], lambda X: 0.4 + 1.2 * X[:, 1], lambda X: 0.12 * X[:, 2], 3)
D_null = Design("S1-null", lambda X: 1.2 * X[:, 0], lambda X: 0.4 + 1.2 * X[:, 1], lambda X: 0.0 * X[:, 2], 3)
D_t6 = Design("T6-scale", lambda X: 0.0 * X[:, 0], lambda X: 0.5 + X[:, 0], lambda X: 0.0 * X[:, 0], 1)
ORA = {}
for D in (D_alt, D_null, D_t6):
    t = time.time(); th1, g1, it1 = oracle_newton(D, m=5)
    th2, g2, it2, _ = oracle_lbfgs(D, m=3)
    dpar = np.abs(th1 - th2).max(); ok = dpar <= TOL_PAR
    rel = []
    for b in BASES:
        e1, e2 = energies(th1, D.d, GM[b]), energies(th2, D.d, GM[b])
        big = np.abs(e1) > TOL_EABS
        rel.append(np.max(np.abs(e1 - e2)[big] / np.abs(e1)[big]))
        ok &= np.all(np.abs(e1 - e2)[~big] <= TOL_EABS)
    ok &= max(rel) <= TOL_EREL
    ORA[D.name] = th1
    print(f"\n[A] {D.name}: method 1 (Newton on FOC, 5 GL x-nodes/bin) |grad| {g1:.1e} in {it1} it; "
          f"method 2 (L-BFGS on closed-form loss, 3 GL x-nodes/bin) |grad| {g2:.1e} in {it2} it")
    print(f"  agreement: max|theta1-theta2| {dpar:.2e} (tol {TOL_PAR}); max rel energy diff {max(rel):.2e} (tol {TOL_EREL}) -> {'PASS' if ok else 'FAIL'}")
    for b in BASES:
        E = energies(th1, D.d, GM[b]); Sl = slopes(th1, D.d, GM[b])
        print(f"  basis {b:9s} energies (rows=features, cols=channels 0..3):")
        for j in range(D.d):
            print(f"    {NAMES[j] if D.d == 3 else 'x':8s} " + " ".join(f"{v:.6e}" for v in E[j]) + f"   ch3 slope {Sl[j,3]:+.6f}")
        if D.d == 3:
            r = E[2, 3] / max(E[0, 3], E[1, 3]) if max(E[0, 3], E[1, 3]) > 0 else np.inf
            print(f"    oracle separation ratio E(x_shape,3)/max(E(x_loc,3),E(x_scale,3)) = {r:.4g} (noise features are 0 at the oracle)")
    print(f"  ({time.time()-t:.0f}s)")
print("  note: noise features have zero oracle effect (Jensen + independence; see PREREG section 1).")

# ---------------- B. asymptotic sampling law, power scan ----------------
SW = {}
for nm, D in (("alt", D_alt), ("null", D_null)):
    SW[nm] = (reduced_from_centred(ORA[D.name], 3), *sandwich(D, ORA[D.name], p_total=8))
SW["t6"] = (reduced_from_centred(ORA[D_t6.name], 1, p_total=1), *sandwich(D_t6, ORA[D_t6.name], p_total=1))
def sep(E):  # E: (reps, 8, K_rep) -> separation ratio in channel 3
    e3 = E[:, :, 3]; return e3[:, 2] / np.delete(e3, 2, 1).max(1)
print("\n[B] asymptotic power scan (theta-hat ~ N(theta*, H^-1 S H^-1/n), 4000 draws), criterion E(x_shape,3) >= 3 x max other")
print("  basis      n      P_alt(pass)  P_null(pass)  alt q05/q50 of ratio   null q95/q99 of ratio")
for b in BASES:
    for n in (2000, 4000, 8000, 16000, 32000):
        Ea = simulate_energies(*SW["alt"], n, GM[b], 4000, 1); En = simulate_energies(*SW["null"], n, GM[b], 4000, 2)
        ra, rn = sep(Ea), sep(En)
        print(f"  {b:9s} {n:6d}   {np.mean(ra>=3):.4f}       {np.mean(rn>=3):.4f}        {np.quantile(ra,.05):6.2f} / {np.quantile(ra,.5):6.2f}        {np.quantile(rn,.95):5.2f} / {np.quantile(rn,.99):5.2f}")

# ---------------- C. chosen design: reference basis (e1 = z), n = 16000 ----------------
N_CH, B_CH, C_SEP, NSIM = 16000, "reference", 3.0, 20000
Ea = simulate_energies(*SW["alt"], N_CH, GM[B_CH], NSIM, 11); En = simulate_energies(*SW["null"], N_CH, GM[B_CH], NSIM, 12)
pi_alt, pi_null = np.mean(sep(Ea) >= C_SEP), np.mean(sep(En) >= C_SEP)
EaL = simulate_energies(*SW["alt"], N_CH, GM["legendre"], NSIM, 13)
Et6 = simulate_energies(*SW["t6"], 2000, GM[B_CH], NSIM, 14, p_total=1)
print(f"\n[C] chosen: basis={B_CH}, n={N_CH}, c={C_SEP}; {NSIM} asymptotic draws")
print(f"  per-seed separation pass prob: alt {pi_alt:.4f}, null twin {pi_null:.4f}")
def band(x, lo=0.005, hi=0.995): return np.quantile(x, lo), np.quantile(x, hi)

# ---------------- D. finite-sample validation (reference estimator) and consistency bands ----------------
here = os.path.dirname(__file__)
V = dict(np.load(os.path.join(here, "prereg_validation.npz")))
if os.path.exists(os.path.join(here, "prereg_validation_null.npz")):
    V["null"] = np.load(os.path.join(here, "prereg_validation_null.npz"))["null"]
V["alt"] = V["alt"][:20]; V["null"] = V["null"][:10]
bi, bl = BASES.index(B_CH), BASES.index("legendre")
def wilson(k, n, z=2.576):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)
print("\n[D] finite-sample Monte Carlo with the reference estimator (smoothed-pinball Newton, h = 0.05 -> 0.02 -> 0.01;"
      " seeds 900000+ (T6), 910000+ (alt), 920000+ (null), disjoint from the test seeds)")
kA, nA = int(np.sum(sep(V["alt"][:, bi]) >= C_SEP)), len(V["alt"])
kN, nN = int(np.sum(sep(V["null"][:, bi]) >= C_SEP)), len(V["null"])
print(f"  alt  (n=16000, {nA} reps): separation passes {kA}/{nA} (99% Wilson {wilson(kA,nA)[0]:.3f}..{wilson(kA,nA)[1]:.3f}); "
      f"ratio min {sep(V['alt'][:, bi]).min():.2f}, median {np.median(sep(V['alt'][:, bi])):.2f}")
print(f"  null (n=16000, {nN} reps): false positives {kN}/{nN} (99% Wilson upper {wilson(kN,nN)[1]:.3f}); ratio max {sep(V['null'][:, bi]).max():.2f}")
print("  mean ch3 energies, asymptotic sim vs Monte Carlo (reference basis):")
for j, nmj in enumerate(NAMES[:3]):
    print(f"    {nmj:8s} sim {Ea[:, j, 3].mean():.4e}  MC {V['alt'][:, bi, j, 3].mean():.4e} (sd {V['alt'][:, bi, j, 3].std(ddof=1):.2e})")
print(f"    noise    sim {Ea[:, 3:, 3].mean():.4e}  MC {V['alt'][:, bi, 3:, 3].mean():.4e}")
def mcband(x): m, sd = x.mean(), x.std(ddof=1); return m - 3 * sd, m + 3 * sd
print("  consistency bands = Monte Carlo mean +- 3 SD of the reference estimator (used for T3v2-secondary, T4v2, T6v2):")
B3 = {nmj: mcband(V["alt"][:, bi, j, 3]) for j, nmj in enumerate(NAMES[:3])}
B4 = {nmj: mcband(V["alt"][:, bl, j, 3]) for j, nmj in enumerate(NAMES[:3])}
for nmj in NAMES[:3]:
    print(f"    T3v2 reference ch3 {nmj:8s} [{B3[nmj][0]:.4e}, {B3[nmj][1]:.4e}]   T4v2 Legendre ch3 {nmj:8s} [{B4[nmj][0]:.4e}, {B4[nmj][1]:.4e}]"
          f"  (Legendre oracle {energies(ORA['S1-alt'],3,GM['legendre'])[NAMES.index(nmj),3]:.4e})")
E6 = V["t6"][:, bi, 0]; B6 = mcband(E6[:, 3]); B6r = mcband(E6[:, 3] / E6[:, 1])
print(f"    T6v2 reference ch3 energy [{B6[0]:.4e}, {B6[1]:.4e}]; ch3/ch1 ratio upper {B6r[1]:.4e}  "
      f"(MC n=2000, {len(E6)} reps; oracle ch3 {energies(ORA['T6-scale'],1,GM[B_CH])[0,3]:.4e}, oracle ratio "
      f"{energies(ORA['T6-scale'],1,GM[B_CH])[0,3]/energies(ORA['T6-scale'],1,GM[B_CH])[0,1]:.3e})")
inside4 = np.mean(np.all([(V["alt"][:, bl, j, 3] >= B4[nm][0]) & (V["alt"][:, bl, j, 3] <= B4[nm][1]) for j, nm in enumerate(NAMES[:3])], 0))
print(f"    in-sample check: fraction of MC reps inside all T4v2 bands {inside4:.3f}")

# ---------------- E. binomial bars over 20 preregistered seeds ----------------
R = 20
def kmin(pi, alpha=0.01):
    for k in range(R, -1, -1):
        if binom.cdf(k - 1, R, pi) <= alpha: return k
    return 0
pi_sep = min(pi_alt, wilson(kA, nA)[0])
pi_band3, pi_band2 = 1 - 3 * 0.0027, 1 - 2 * 0.0027   # normal approx per stat, Bonferroni over stats
k3, k4, k6 = kmin(pi_sep), kmin(pi_band3), kmin(pi_band2)
fp_bar = 1
print(f"\n[E] bars over R={R} seeds (bar chosen so that an estimator behaving like the reference fails it with probability <= 0.01):")
print(f"  T3v2 separation (primary): rate used min(sim {pi_alt:.4f}, MC Wilson lower {wilson(kA,nA)[0]:.4f}) = {pi_sep:.4f} -> pass if >= {k3}/{R}; "
      f"P(fail | rate) = {binom.cdf(k3-1,R,pi_sep):.4f}; P(fail | sim rate) = {binom.cdf(k3-1,R,pi_alt):.2e}")
print(f"  T3v2 null twin: pass if false positives <= {fp_bar}/{R}; P(fail) at sim rate {pi_null:.4f}: {binom.sf(fp_bar,R,pi_null):.2e}; "
      f"at MC Wilson upper {wilson(kN,nN)[1]:.3f}: {binom.sf(fp_bar,R,wilson(kN,nN)[1]):.3f}; at a 5% per-seed FP rate: {binom.sf(fp_bar,R,0.05):.3f}")
print(f"  T4v2 Legendre consistency (3 bands): rate {pi_band3:.4f} -> pass if >= {k4}/{R}; P(fail) = {binom.cdf(k4-1,R,pi_band3):.4f}")
print(f"  T6v2 scale-leak consistency (2 bands): rate {pi_band2:.4f} -> pass if >= {k6}/{R}; P(fail) = {binom.cdf(k6-1,R,pi_band2):.4f}")
print(f"  T3v2 secondary consistency (3 bands, reported, not a pass condition): rate {pi_band3:.4f}, bar {k4}/{R}")
print("\nend")
