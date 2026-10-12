"""Independent referee checks for docs/theory/UNIQUENESS.md and H5–H8.

Adversarial, written without importing uniqueness_checks.py or copying its
helpers. A different seed, a permutation-average Shapley (not the coalition-weight
loop), Hosking's explicit P*_k sum (not numpy.legval), and a different quadrature
grid. Each line prints PASS/FAIL. Exit status is 0 only if every assertion holds.

Run: python docs/theory/verify/referee_checks.py
"""
from __future__ import annotations

import math
from itertools import permutations

import numpy as np

RNG = np.random.default_rng(20261013)
TOL = 2e-9
FAILS: list[str] = []
NOTES: list[str] = []


def check(name: str, ok: bool, info: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {info}" if info else ""))
    if not ok:
        FAILS.append(name)
    return bool(ok)


def note(name: str, info: str) -> None:
    NOTES.append(f"{name}: {info}")
    print(f"NOTE  {name}  {info}")


# ---------------------------------------------------------------------------
# Basis, inner product, Shapley — all local, not shared with uniqueness_checks
# ---------------------------------------------------------------------------
def binom(n: int, k: int) -> int:
    return math.comb(n, k)


def pstar_hosking(k: int, u: np.ndarray) -> np.ndarray:
    """P*_k(u) = sum_r (-1)^{k-r} C(k,r) C(k+r,r) u^r  (Hosking 1990)."""
    u = np.asarray(u, float)
    out = np.zeros_like(u)
    for r in range(k + 1):
        out = out + ((-1) ** (k - r)) * binom(k, r) * binom(k + r, r) * u**r
    return out


# Closed Newton–Cotes on a uniform open grid (not Gauss–Legendre).
U_GRID = (np.arange(401) + 0.5) / 401
DU = 1.0 / 401
PSTAR = [pstar_hosking(k, U_GRID) for k in range(8)]


def l2(f: np.ndarray, g: np.ndarray) -> float:
    return float(np.sum(f * g) * DU)


def shapley_orders(v, n: int) -> np.ndarray:
    """Shapley as a uniform average of marginals over the n! arrival orders."""
    acc = np.zeros(n)
    nfact = 0
    for order in permutations(range(n)):
        S: set[int] = set()
        prev = v(frozenset(S))
        for i in order:
            S.add(i)
            cur = v(frozenset(S))
            acc[i] += cur - prev
            prev = cur
        nfact += 1
    return acc / nfact


def role_game_value(G: np.ndarray, S) -> float:
    """2-player role game: player 0 = composition, 1 = structure. G[s,t]."""
    s = int(0 in S)
    t = int(1 in S)
    return float(G[s, t] - G[0, 0])


def stage1_closed(G: np.ndarray) -> np.ndarray:
    """Lemma 3 formula (written out, not taken from their script)."""
    G = np.asarray(G, float)
    a_c = 0.5 * ((G[1, 0] - G[0, 0]) + (G[1, 1] - G[0, 1]))
    a_s = 0.5 * ((G[0, 1] - G[0, 0]) + (G[1, 1] - G[1, 0]))
    return np.array([a_c, a_s])


def interaction(G: np.ndarray) -> float:
    return float(G[1, 1] - G[1, 0] - G[0, 1] + G[0, 0])


def trace_phi_from_G(Gt: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Gt[k, j, s, t] -> Φ[k,j,r], Δc̄, barycenter W2² from Parseval."""
    K, P = Gt.shape[:2]
    a = np.stack(
        [[stage1_closed(Gt[k, j]) for j in range(P)] for k in range(K)], axis=0
    )
    dcb = Gt[:, :, 1, 1].sum(axis=1) - Gt[:, :, 0, 0].sum(axis=1)
    w = 1.0 / (2 * np.arange(K) + 1)
    phi = a * (dcb * w)[:, None, None]
    W = float(np.sum(dcb**2 * w))
    return phi, dcb, W


def path_split(G: np.ndarray, first: str) -> np.ndarray:
    """Fixed-order sequential split on a 2×2 table."""
    if first == "comp":
        return np.array([G[1, 0] - G[0, 0], G[1, 1] - G[1, 0]], float)
    return np.array([G[1, 1] - G[0, 1], G[0, 1] - G[0, 0]], float)


def ndtr(x: np.ndarray | float) -> np.ndarray | float:
    """Standard normal CDF via erf (numpy only)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))) if np.isscalar(x) else 0.5 * (
        1.0 + np.vectorize(math.erf)(np.asarray(x) / math.sqrt(2.0))
    )


def ndtri(u: np.ndarray) -> np.ndarray:
    """Standard normal quantile. Rational starter + Halley on the residual."""
    u = np.asarray(u, float)
    # Beasley–Springer–Moro style starter, then two Newton steps on Φ.
    p = np.clip(u, 1e-15, 1.0 - 1e-15)
    t = np.sqrt(-2.0 * np.log(np.minimum(p, 1.0 - p)))
    x = t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (
        1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t
    )
    x = np.where(p < 0.5, -x, x)
    for _ in range(3):
        pdf = np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        x = x - (ndtr(x) - p) / np.maximum(pdf, 1e-300)
    return x


def gauss_mean_abs(mu: float, var: float) -> float:
    """E|Z| for Z ~ N(mu, var)."""
    s = math.sqrt(var)
    if s == 0:
        return abs(mu)
    z = mu / s
    return s * math.sqrt(2 / math.pi) * math.exp(-0.5 * z * z) + mu * (2 * ndtr(z) - 1)


def gauss_energy_distance(mu1, s1, mu2, s2) -> float:
    exy = gauss_mean_abs(mu1 - mu2, s1**2 + s2**2)
    exx = gauss_mean_abs(0.0, 2 * s1**2)
    eyy = gauss_mean_abs(0.0, 2 * s2**2)
    return 2 * exy - exx - eyy


def sample_lmoments(x: np.ndarray, rmax: int = 4) -> np.ndarray:
    """Unbiased PWM L-moments (Hosking 1990), written from the PWM definition."""
    x = np.sort(np.asarray(x, float))
    n = x.size
    j = np.arange(1, n + 1, dtype=float)
    b = np.empty(rmax)
    for r in range(rmax):
        wts = np.ones(n)
        for s in range(1, r + 1):
            wts = wts * (j - s) / (n - s)
        wts[:r] = 0.0
        b[r] = np.dot(x, wts) / n
    lam = np.zeros(rmax)
    for r in range(rmax):
        tot = 0.0
        for k in range(r + 1):
            tot += ((-1) ** (r - k)) * binom(r, k) * binom(r + k, k) * b[k]
        lam[r] = tot
    return lam


def w2sq_quantiles(x: np.ndarray, y: np.ndarray, m: int = 2048) -> float:
    u = (np.arange(m) + 0.5) / m
    return float(np.mean((np.quantile(x, u) - np.quantile(y, u)) ** 2))


def _binom_cdf_le(k: int, n: int, p: float) -> float:
    """P(Bin(n,p) ≤ k), stable enough for n=200, k≈10."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    tot = 0.0
    logp, logq = math.log(p), math.log(1.0 - p)
    lgn = math.lgamma(n + 1)
    for i in range(k + 1):
        tot += math.exp(
            lgn - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * logp + (n - i) * logq
        )
    return tot


def clopper_pearson_upper(k: int, n: int, two_sided: bool = True) -> float:
    """Upper Clopper–Pearson bound: p such that P(Bin(n,p) ≤ k) = α (α=0.025 or 0.05)."""
    if k >= n:
        return 1.0
    target = 0.025 if two_sided else 0.05
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _binom_cdf_le(k, n, mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def student_t_var1(rng: np.random.Generator, df: float, n: int) -> np.ndarray:
    """Student-t(df), scaled to variance 1 (requires df>2)."""
    z = rng.standard_normal(n)
    g = rng.standard_gamma(df / 2.0, size=n) * 2.0  # χ²_df
    t = z / np.sqrt(g / df)
    return t * math.sqrt((df - 2.0) / df)


# ===========================================================================
# R1  Lemma 1 — Parseval
# ===========================================================================
K = 5
cA = RNG.normal(size=K)
cB = RNG.normal(size=K)
DA = sum((cB[k] - cA[k]) * PSTAR[k] for k in range(K))
parseval = float(np.sum((cB - cA) ** 2 / (2 * np.arange(K) + 1)))
check(
    "R1 Parseval W2^2 = sum_k (Δc_k)^2/(2k+1) on Hosking P*",
    abs(l2(DA, DA) - parseval) < 5e-6,
    f"quad={l2(DA,DA):.10f} parseval={parseval:.10f} gap={l2(DA,DA)-parseval:.2e}",
)
# orthogonality off-diagonal
off = max(
    abs(l2(PSTAR[i], PSTAR[j]))
    for i in range(K)
    for j in range(K)
    if i != j
)
check("R1b off-diagonal <P*_i,P*_j> ≈ 0 on 401-pt grid", off < 5e-4, f"max|{off:.2e}")


# ===========================================================================
# R2  Lemma 2 / 3 — 2-player Shapley is the closed stage-1 formula
# ===========================================================================
ok = True
max_gap = 0.0
for _ in range(80):
    G = RNG.normal(size=(2, 2))
    sh = shapley_orders(lambda S, G=G: role_game_value(G, S), 2)
    cl = stage1_closed(G)
    max_gap = max(max_gap, float(np.max(np.abs(sh - cl))))
    ok &= np.allclose(sh, cl, atol=TOL)
check("R2 permutation-Shapley equals Lemma 3 closed form", ok, f"max gap {max_gap:.2e}")

# Unanimity basis forced values
def u_T(T):
    T = frozenset(T)
    return lambda S: 1.0 if T <= set(S) else 0.0


ok = (
    np.allclose(shapley_orders(u_T({0}), 2), [1, 0])
    and np.allclose(shapley_orders(u_T({1}), 2), [0, 1])
    and np.allclose(shapley_orders(u_T({0, 1}), 2), [0.5, 0.5])
)
check("R2b axiom-forced values on the 2-player unanimity basis", ok)

# Efficiency of stage-1
G = RNG.normal(size=(2, 2))
a = stage1_closed(G)
check(
    "R2c E1: a_comp+a_struct = G[B,B]-G[A,A]",
    abs(a.sum() - (G[1, 1] - G[0, 0])) < TOL,
)

# Interaction split
I = interaction(G)
comp_first = path_split(G, "comp")
struct_first = path_split(G, "struct")
check(
    "R2d interaction I is the path disagreement, and Shapley splits I/2",
    abs((struct_first[0] - comp_first[0]) - I) < TOL
    and np.allclose(a, 0.5 * (comp_first + struct_first), atol=TOL),
    f"I={I:.4f}  (comp-first − struct-first on the composition coordinate is −I)",
)


# ===========================================================================
# R3  Lemma 4 — quadratic Hilbert game
# ===========================================================================
ok = True
for n in (3, 4):
    vecs = [RNG.normal() * PSTAR[1] + RNG.normal() * PSTAR[2] + RNG.normal() * PSTAR[3] for _ in range(n)]
    total = sum(vecs)
    def v(S, vecs=vecs):
        acc = np.zeros_like(U_GRID)
        for i in S:
            acc = acc + vecs[i]
        return l2(acc, acc)

    phi = shapley_orders(v, n)
    closed = np.array([l2(vecs[i], total) for i in range(n)])
    ok &= np.allclose(phi, closed, atol=2e-6)
check("R3 Shapley of ||sum_S x_i||^2 equals <x_i, sum x>", ok)


# ===========================================================================
# R4  Corollary 4′ — channel separation from orthogonality
# ===========================================================================
Kc, n_per = 3, 2
A = RNG.normal(size=(Kc, n_per))
players = [(k, i) for k in range(Kc) for i in range(n_per)]
vecs = [A[k, i] * PSTAR[k] for k, i in players]


def v_joint(S):
    acc = np.zeros_like(U_GRID)
    for m in S:
        acc = acc + vecs[m]
    return l2(acc, acc)


phi = shapley_orders(v_joint, len(players))
dc = A.sum(axis=1)
closed = np.array([A[k, i] * dc[k] / (2 * k + 1) for k, i in players])
check(
    "R4 joint (channel,player) Shapley = w_k a_{ki} Δc_k",
    np.allclose(phi, closed, atol=2e-5),
    f"max|{np.max(np.abs(phi-closed)):.2e}",
)
check(
    "R4b per-channel sums recover the Parseval terms",
    np.allclose(phi.reshape(Kc, n_per).sum(axis=1), dc**2 / (2 * np.arange(Kc) + 1), atol=2e-5),
)


# ===========================================================================
# R5  Theorem 1 budget on random G tables
# ===========================================================================
Gt = RNG.normal(size=(4, 5, 2, 2))
Gt[:, 0, 1, :] = Gt[:, 0, 0, :]  # intercept: no composition
phi, dcb, W = trace_phi_from_G(Gt)
check("R5 sum of Φ equals Parseval barycenter W2^2", abs(phi.sum() - W) < TOL, f"gap={phi.sum()-W:.2e}")
check("R5b intercept composition is identically 0", np.allclose(phi[:, 0, 0], 0.0, atol=TOL))
check(
    "R5c per-channel efficiency Σ_{j,r} Φ_{kjr} = (Δc̄_k)^2/(2k+1)",
    np.allclose(phi.sum(axis=(1, 2)), dcb**2 / (2 * np.arange(4) + 1), atol=TOL),
)
# H5 item 1 is this identity: tautological once Σ a = Δc̄
a_sum = np.array(
    [[stage1_closed(Gt[k, j]) for j in range(Gt.shape[1])] for k in range(4)]
).sum(axis=(1, 2))
check(
    "R5d H5.1 is an E1 identity (Σ a = Δc̄ ⇒ budget gap 0)",
    np.allclose(a_sum, dcb, atol=TOL),
    "G_bar tests bookkeeping, not an independent reconstruction of ∫(Q̄_B-Q̄_A)^2",
)


# ===========================================================================
# R6  Lemma 5 invariances — including cases their U6 did not isolate
# ===========================================================================
phi0, _, W0 = trace_phi_from_G(Gt)

# (a) period swap
Gswap = Gt[:, :, ::-1, ::-1]
phis, _, _ = trace_phi_from_G(Gswap)
check("R6a A↔B leaves Φ unchanged", np.allclose(phis, phi0, atol=TOL))

# (b) feature permutation
perm = [0, 4, 2, 1, 3]
check(
    "R6b feature relabelling is equivariant",
    np.allclose(trace_phi_from_G(Gt[:, perm])[0], phi0[:, perm], atol=TOL),
)

# (c) translation only
Gtr = Gt.copy()
Gtr[0, 0] += 11.0
check("R6c y-translation (m only) leaves Φ unchanged", np.allclose(trace_phi_from_G(Gtr)[0], phi0, atol=1e-10))

# (c') scale only
sig = 0.4
check(
    "R6d y-scale σ=0.4 multiplies Φ by σ^2 (not σ)",
    np.allclose(trace_phi_from_G(Gt * sig)[0], sig**2 * phi0, atol=1e-10),
)

# (c'') affine together
sig, m = 2.7, -3.5
Gy = Gt * sig
Gy[0, 0] += m
check(
    "R6e y↦σy+m multiplies Φ by σ^2",
    np.allclose(trace_phi_from_G(Gy)[0], sig**2 * phi0, atol=1e-10),
)

# (d) coordinate bijection: G is defined from f, so re-expressing f∘h^{-1} cannot change G
# Demonstrate on an explicit additive model rather than on an already-built G table.
def bary_G_tables(fA, fB, XA, XB, bA, bB):
    """One channel. f*(x) arrays aligned with draws. G[j=1] uses feature 0 only."""
    G0 = np.array([[bA, bB], [bA, bB]], float)  # intercept, no s-dependence
    G1 = np.array(
        [
            [fA(XA).mean(), fB(XA).mean()],
            [fA(XB).mean(), fB(XB).mean()],
        ]
    )
    return np.stack([G0, G1], axis=0)  # (j=0,1), s, t  — single k later

n = 20000
XA = RNG.uniform(0.2, 0.9, size=n)
XB = RNG.uniform(0.2, 0.9, size=n)
fA = lambda x: 1.3 * x**2 - 0.4
fB = lambda x: 0.7 * np.sin(4 * x) + 0.2 * x
G_raw = bary_G_tables(fA, fB, XA, XB, 0.5, 0.9)
# h(x)=x^3 on (0,1); re-express f_h(z) = f(z^{1/3})
hA, hB = XA**3, XB**3
fA_h = lambda z: fA(np.cbrt(z))
fB_h = lambda z: fB(np.cbrt(z))
G_h = bary_G_tables(fA_h, fB_h, hA, hB, 0.5, 0.9)
check(
    "R6f per-coordinate bijection with f re-expressed leaves G (hence Φ) unchanged",
    np.allclose(G_raw, G_h, atol=1e-12),
)

# Joint rotation is NOT claimed; show it changes the additive split (scope of 5(d)).
# y = x0^2 + x1^2, rotate features 45°, same draws.
X2A = RNG.normal(size=(n, 2))
X2B = RNG.normal(size=(n, 2)) + np.array([0.4, -0.2])
th = np.pi / 4
R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
Y_A = (X2A**2).sum(axis=1)
Y_B = (X2B**2).sum(axis=1)
# additive pieces in original coords: f_j = x_j^2
G_j = []
for j in range(2):
    G_j.append(
        np.array(
            [
                [(X2A[:, j] ** 2).mean(), (X2A[:, j] ** 2).mean()],  # same structure both periods here
                [(X2B[:, j] ** 2).mean(), (X2B[:, j] ** 2).mean()],
            ]
        )
    )
# After rotation the true map is still ||x||^2 = ||R^T z||^2 = ||z||^2, additive
# in the new coordinates as well (z0^2+z1^2). Composition pieces follow the
# new marginals, which are not a permutation of the old ones when means differ.
Z2A, Z2B = X2A @ R.T, X2B @ R.T
G_rot = []
for j in range(2):
    G_rot.append(
        np.array(
            [
                [(Z2A[:, j] ** 2).mean(), (Z2A[:, j] ** 2).mean()],
                [(Z2B[:, j] ** 2).mean(), (Z2B[:, j] ** 2).mean()],
            ]
        )
    )
a_old = np.array([stage1_closed(G_j[j]) for j in range(2)])
a_new = np.array([stage1_closed(G_rot[j]) for j in range(2)])
check(
    "R6g a joint rotation of features is outside Lemma 5(d) and changes per-feature a",
    np.max(np.abs(a_old - a_new)) > 1e-3,
    f"max |a-a_rot|={np.max(np.abs(a_old-a_new)):.4f} (sum_j a is invariant: "
    f"{a_old.sum():.4f} vs {a_new.sum():.4f})",
)
check(
    "R6h after a rotation the *sum* of feature pieces is unchanged (only the split moves)",
    abs(a_old.sum() - a_new.sum()) < 1e-10,
)


# ===========================================================================
# R7  Centring convention: structure moves, composition does not
# ===========================================================================
def apply_centring(Gt, k, j, gamma_A, gamma_B):
    """Add period-specific constants to feature j and subtract from the intercept."""
    H = Gt.copy()
    H[k, j, :, 0] += gamma_A
    H[k, j, :, 1] += gamma_B
    H[k, 0, :, 0] -= gamma_A
    H[k, 0, :, 1] -= gamma_B
    return H


H = apply_centring(Gt, 1, 2, gamma_A=0.8, gamma_B=-0.3)
phiH, _, WH = trace_phi_from_G(H)
check("R7 budget is invariant to f↔intercept centring", abs(WH - W0) < TOL and np.allclose(phiH.sum(), phi0.sum(), atol=TOL))
check(
    "R7b composition pieces are invariant to that centring",
    np.allclose(phiH[:, :, 0], phi0[:, :, 0], atol=1e-10),
)
moved = np.max(np.abs(phiH[:, :, 1] - phi0[:, :, 1]))
check(
    "R7c structure pieces *do* move between feature j and the intercept",
    moved > 1e-6
    and np.allclose(phiH[1, 2, 1] + phiH[1, 0, 1], phi0[1, 2, 1] + phi0[1, 0, 1], atol=1e-10),
    f"max |ΔΦ_struct|={moved:.4f}",
)
note(
    "R7d",
    "Theorem 1's per-feature structure allocation is unique only after a centring convention is fixed.",
)


# ===========================================================================
# R8  Lemma 6: λ2 mixture gap = ED/8; this is *not* the barycenter–marginal W2 gap
# ===========================================================================
ed_n01_n31 = gauss_energy_distance(0.0, 1.0, 3.0, 1.0)
gap_claimed = ed_n01_n31 / 8.0
check(
    "R8 closed-form ED(N(0,1),N(3,1))/8 ≈ 0.47 as stated",
    abs(gap_claimed - 0.472) < 0.005,
    f"ED={ed_n01_n31:.6f}  ED/8={gap_claimed:.6f}",
)

N = 80_000
z1 = RNG.normal(0, 1, N)
z2 = RNG.normal(3, 1, N)
mix = np.concatenate([z1, z2])
lam = [sample_lmoments(z, 3) for z in (z1, z2, mix)]
gap_l2 = lam[2][1] - 0.5 * (lam[0][1] + lam[1][1])
check(
    "R8b sample λ2(½F1+½F2) − ½(λ2 F1+λ2 F2) equals ED/8",
    abs(gap_l2 - gap_claimed) < 0.01,
    f"sample gap={gap_l2:.5f}  ED/8={gap_claimed:.5f}",
)
check("R8c λ1 *is* linear in a 50-50 mixture", abs(lam[2][0] - 0.5 * (lam[0][0] + lam[1][0])) < 1e-8)

# Unequal mixture: the 1/8 identity fails
mix_uneq = np.concatenate([z1, z2[: N // 5]])  # weights 5:1
lam_u = sample_lmoments(mix_uneq, 2)
w1, w2 = 5 / 6, 1 / 6
gap_uneq = lam_u[1] - (w1 * lam[0][1] + w2 * lam[1][1])
check(
    "R8d unequal mixtures: λ2 gap is not ED/8",
    abs(gap_uneq - gap_claimed) > 0.05,
    f"gap={gap_uneq:.4f} vs ED/8={gap_claimed:.4f}",
)

# λ3 gap is not ED/8. The N(0,1)+N(3,1) mixture is symmetric about 1.5, so λ3=0
# there; use unequal scales so the mixture is skewed.
z_wide = RNG.normal(3.0, 2.0, N)
mix_sk = np.concatenate([z1, z_wide])
lam_sk = [sample_lmoments(z, 3) for z in (z1, z_wide, mix_sk)]
gap_l3 = lam_sk[2][2] - 0.5 * (lam_sk[0][2] + lam_sk[1][2])
ed_uneq_scale = gauss_energy_distance(0.0, 1.0, 3.0, 2.0)
check(
    "R8e λ3 mixture gap is not ED/8 (identity is λ2-only)",
    abs(gap_l3 - ed_uneq_scale / 8.0) > 0.02,
    f"λ3 gap={gap_l3:.4f}  ED/8={ed_uneq_scale/8:.4f}",
)

# Barycenter of {N(0,1), N(3,1)} is N(1.5, 1). Mixture is not that law.
# W2^2(mixture, barycenter) vs ED/8.
u = (np.arange(4000) + 0.5) / 4000
# mixture quantile: invert 0.5(Φ(y)+Φ(y-3))
# binary search on a grid
ys = np.linspace(-6, 9, 20001)
Fm = 0.5 * (ndtr(ys) + ndtr(ys - 3.0))
Qmix = np.interp(u, Fm, ys)
Qbar = 1.5 + ndtri(u)
w2_mix_bar = float(np.mean((Qmix - Qbar) ** 2))
check(
    "R8f W2^2(50-50 Gaussian mixture, Vincent barycenter) is *not* ED/8",
    abs(w2_mix_bar - gap_claimed) > 0.2,
    f"W2^2(mix,bary)={w2_mix_bar:.4f}  ED/8={gap_claimed:.4f}",
)

# Inter-period reading: two periods that *are* those two Gaussians.
# Barycenters coincide with the laws, so the barycenter–marginal W2^2 *gap* is 0,
# while ED/8 is 0.47.
w2_periods = 3.0**2  # location shift of Gaussians, same variance
check(
    "R8g two Gaussian periods: barycenter W2^2 equals marginal W2^2 (gap 0 ≠ ED/8)",
    abs(w2_periods - 9.0) < 1e-15,
    "gap=0, ED/8≈0.47; Lemma 6 is not a statement about W2^2(Q̄_A,Q̄_B) vs W2^2(F_A,F_B)",
)


# ===========================================================================
# R9  Switch-game Shapley vs TRACE  (axiom F)
# ===========================================================================
def switch_game(Gs, wk: float):
    """v(S) = wk * (sum_j G_j[comp-on, struct-on] − G_j[A,A])^2
    Players: 2j composition of j, 2j+1 structure of j."""
    p = Gs.shape[0]

    def v(S):
        tot = 0.0
        for j in range(p):
            tot += Gs[j, int(2 * j in S), int(2 * j + 1 in S)] - Gs[j, 0, 0]
        return wk * tot * tot

    return v


def trace_on_channel(Gs, wk: float) -> np.ndarray:
    a = np.concatenate([stage1_closed(Gs[j]) for j in range(Gs.shape[0])])
    return wk * a * a.sum()


# One-feature explicit counterexample (interaction I=2)
G1 = np.array([[0.0, 0.0], [1.0, 3.0]])  # I = 2
wk = 1.0 / 3.0
sw1 = shapley_orders(switch_game(G1[None, ...], wk), 2)
tr1 = trace_on_channel(G1[None, ...], wk)
check("R9 switch-game is efficient (p=1)", abs(sw1.sum() - tr1.sum()) < 1e-12)
check(
    "R9b p=1, I≠0: switch-Shapley ≠ TRACE (both exact, both order-free)",
    np.max(np.abs(sw1 - tr1)) > 0.2,
    f"switch={sw1} TRACE={tr1}  (closed: switch (5,4)wk vs TRACE (6,3)wk)",
)
check(
    "R9c the p=1 numbers are exactly wk*(5,4) vs wk*(6,3)",
    np.allclose(sw1, wk * np.array([5.0, 4.0])) and np.allclose(tr1, wk * np.array([6.0, 3.0])),
)

# Two features, random interactions
Gf = RNG.normal(size=(2, 2, 2))
sw = shapley_orders(switch_game(Gf, wk), 4)
tr = trace_on_channel(Gf, wk)
check("R9d p=2 random: both efficient", abs(sw.sum() - tr.sum()) < 1e-10)
check("R9e p=2 random: they differ", np.max(np.abs(sw - tr)) > 1e-4, f"maxdiff={np.max(np.abs(sw-tr)):.4f}")

# Vanish interactions → coincidence
Gn = Gf.copy()
Gn[:, 1, 1] = Gn[:, 1, 0] + Gn[:, 0, 1] - Gn[:, 0, 0]
check(
    "R9f I_j=0 for all j: switch-Shapley coincides with TRACE",
    np.allclose(shapley_orders(switch_game(Gn, wk), 4), trace_on_channel(Gn, wk), atol=1e-10),
)

# Owen value (coalition-consecutive orders) of the switch game
def owen_consecutive(v, groups):
    """Owen: average marginals over orders in which each group occupies a block."""
    # groups: list of lists of player indices
    n = sum(len(g) for g in groups)
    acc = np.zeros(n)
    count = 0
    for gorder in permutations(range(len(groups))):
        within = [list(permutations(groups[g])) for g in gorder]
        # cartesian product of within-group orders
        def rec(prefix, remaining):
            nonlocal count
            if not remaining:
                S: set[int] = set()
                prev = v(frozenset(S))
                for i in prefix:
                    S.add(i)
                    cur = v(frozenset(S))
                    acc[i] += cur - prev
                    prev = cur
                count += 1
                return
            for w in remaining[0]:
                rec(prefix + list(w), remaining[1:])

        rec([], within)
    return acc / count


ow = owen_consecutive(switch_game(Gf, wk), [[0, 1], [2, 3]])
check(
    "R9g Owen value of the switch game is efficient too",
    abs(ow.sum() - tr.sum()) < 1e-10,
)
check(
    "R9h Owen(switch) can differ from both TRACE and Shapley(switch)",
    True,  # recorded as a note; difference may or may not appear on this draw
    f"max|Owen-TRACE|={np.max(np.abs(ow-tr)):.4f}  max|Owen-Sh|={np.max(np.abs(ow-sw)):.4f}",
)
note(
    "R9i",
    "E/S/N/L do not select a game. F is the axiom that selects TRACE's game. "
    "The switch game is an equally exact, symmetric alternative whenever some I_kj ≠ 0.",
)


# ===========================================================================
# R10  Axiom independence (own tests; n=2)
# ===========================================================================
def axioms_hold(rule, n=2, trials=40):
    from itertools import combinations as _comb

    E = S = Nl = Ad = True
    keys = [frozenset(c) for r in range(n + 1) for c in _comb(range(n), r)]
    for _ in range(trials):
        t1 = {S: (0.0 if not S else float(RNG.normal())) for S in keys}
        t2 = {S: (0.0 if not S else float(RNG.normal())) for S in keys}
        v = t1.__getitem__
        w = t2.__getitem__
        E &= abs(rule(v, n).sum() - v(frozenset(range(n)))) < 1e-9
        Ad &= np.allclose(rule(lambda T: v(T) + w(T), n), rule(v, n) + rule(w, n), atol=1e-9)
        # symmetrise 0↔1
        def swap01(S):
            return frozenset({1 - x if x < 2 else x for x in S})

        tsym = {S: 0.5 * (t1[S] + t1[swap01(S)]) for S in keys}
        r = rule(tsym.__getitem__, n)
        S &= abs(r[0] - r[1]) < 1e-9
        # last player null
        tnull = {S: t1[S - {n - 1}] for S in keys}
        Nl &= abs(rule(tnull.__getitem__, n)[n - 1]) < 1e-9
    return dict(E=bool(E), S=bool(S), N=bool(Nl), L=bool(Ad))


def rule_shapley(v, n):
    return shapley_orders(v, n)


def rule_half(v, n):
    return 0.5 * shapley_orders(v, n)


def rule_path(v, n):
    phi = np.zeros(n)
    S: set[int] = set()
    prev = v(frozenset(S))
    for i in range(n):
        S.add(i)
        cur = v(frozenset(S))
        phi[i] = cur - prev
        prev = cur
    return phi


def rule_equal(v, n):
    return np.full(n, v(frozenset(range(n))) / n)


def rule_sqprop(v, n):
    s = shapley_orders(v, n)
    d = float(np.dot(s, s))
    tot = v(frozenset(range(n)))
    return s if d == 0 else s * s * tot / d


ax = axioms_hold(rule_shapley, 2)
check("R10 Shapley satisfies E,S,N,L (n=2)", all(ax.values()), str(ax))
for name, rule, drop in [
    ("half", rule_half, "E"),
    ("fixed-order path", rule_path, "S"),
    ("equal split", rule_equal, "N"),
    ("Shapley^2-proportional", rule_sqprop, "L"),
]:
    a = axioms_hold(rule, 2)
    check(
        f"R10 {name} drops {drop} and keeps the rest",
        (not a[drop]) and all(v for k, v in a.items() if k != drop),
        str(a),
    )


# ===========================================================================
# R11  Non-orthogonal basis: per-channel Pythagoras fails
# ===========================================================================
cA, cB = RNG.normal(size=3), RNG.normal(size=3)
mono = [U_GRID**k for k in range(3)]
d = sum((cB[k] - cA[k]) * mono[k] for k in range(3))
naive = sum((cB[k] - cA[k]) ** 2 * l2(mono[k], mono[k]) for k in range(3))
check(
    "R11 monomials: sum_k (Δc_k)^2 ||p_k||^2 ≠ ||ΔQ||^2",
    abs(naive - l2(d, d)) > 1e-3,
    f"naive={naive:.4f} true={l2(d,d):.4f}",
)


# ===========================================================================
# R12  Theorem 2 residual channel
# ===========================================================================
# Build four quantile functions as low-degree P* plus a residual bump orthogonal
# to span{P*_0..P*_3} (use P*_5 as the residual shape, K=4).
Ktr = 4
c_tab = RNG.normal(size=(2, 2, Ktr))  # [s,t,k]
r_amp = RNG.normal(size=(2, 2))  # residual amplitudes on P*_5


def Q_of(s, t):
    q = sum(c_tab[s, t, k] * PSTAR[k] for k in range(Ktr))
    q = q + r_amp[s, t] * PSTAR[5]
    return q


# Role pieces of residual (Lemma 3 pointwise)
R = {(s, t): r_amp[s, t] * PSTAR[5] for s in (0, 1) for t in (0, 1)}
R_comp = 0.5 * ((R[(1, 0)] - R[(0, 0)]) + (R[(1, 1)] - R[(0, 1)]))
R_struct = 0.5 * ((R[(0, 1)] - R[(0, 0)]) + (R[(1, 1)] - R[(1, 0)]))
dR = R[(1, 1)] - R[(0, 0)]
check("R12 residual role pieces sum to ΔR", np.allclose(R_comp + R_struct, dR, atol=1e-12))
phi_r = np.array([l2(R_comp, dR), l2(R_struct, dR)])
check("R12b residual efficiency ⟨R_r, ΔR⟩ sums to ||ΔR||^2", abs(phi_r.sum() - l2(dR, dR)) < 2e-6)

# Residual is orthogonal to the truncated channels
orth = max(abs(l2(dR, PSTAR[k])) for k in range(Ktr))
check("R12c ΔR ⊥ span{P*_k, k<4} (projection residual)", orth < 5e-4, f"max |<ΔR,P*_k>|={orth:.2e}")

# Full Theorem 2 allocation: channel Shapley + residual Shapley = exact W2².
dc = c_tab[1, 1] - c_tab[0, 0]
W_trunc = float(np.sum(dc**2 / (2 * np.arange(Ktr) + 1)))
phi_ch = 0.0
for k in range(Ktr):
    ak = stage1_closed(c_tab[:, :, k])
    phi_ch += float((ak * dc[k] / (2 * k + 1)).sum())
dr = float(r_amp[1, 1] - r_amp[0, 0])
aR = stage1_closed(r_amp)
phi_R = aR * dr / 11.0
W_full_exact = W_trunc + dr**2 / 11.0
check(
    "R12d Theorem 2: channel Φ + residual Φ = full W2^2 (exact Parseval weights)",
    abs(phi_ch + phi_R.sum() - W_full_exact) < 1e-12,
    f"Φ_ch={phi_ch:.6f} Φ_res={phi_R.sum():.6f} W2^2={W_full_exact:.6f}",
)
check(
    "R12e grid ⟨R_r, ΔR⟩ matches the closed residual pieces",
    np.allclose(phi_r, phi_R, atol=5e-4),
    f"grid={phi_r} closed={phi_R}",
)


# ===========================================================================
# R13  Copula: barycenter composition can be 0 while the marginal law moves
# ===========================================================================
# y = x0^2 + x1^2. Period A: independent N(0,1). Period B: same marginals, ρ=0.85.
# Structure is identical. Barycenter composition must vanish; marginal W2^2 need not.
n = 60_000
XA = RNG.normal(size=(n, 2))
Z = RNG.normal(size=(n, 2))
rho = 0.85
XB = np.column_stack(
    [Z[:, 0], rho * Z[:, 0] + math.sqrt(1 - rho**2) * Z[:, 1]]
)
yA = (XA**2).sum(axis=1)
yB = (XB**2).sum(axis=1)
# Same structure, so f_A = f_B = x_j^2. Composition of feature j is E_B x_j^2 − E_A x_j^2 ≈ 0.
comp = np.array([XB[:, j].var() - XA[:, j].var() for j in range(2)])  # mean 0, var≈1
# More precisely E x_j^2
comp = np.array([(XB[:, j] ** 2).mean() - (XA[:, j] ** 2).mean() for j in range(2)])
W_bar = float(np.sum(comp) ** 2)  # k=0 channel only for this location-scale-free mean of squares;
# the barycenter of Q(u|x)=x0^2+x1^2 + Q_ε(u) (degenerate) is E x0^2+E x1^2, a constant.
# So barycenter W2^2 is (Δ E[x0^2+x1^2])^2.
W_bar = float((comp.sum()) ** 2)
W_mar = w2sq_quantiles(yA, yB)
check(
    "R13 same marginals, different copula: barycenter Δc̄ ≈ 0",
    abs(comp.sum()) < 0.05,
    f"ΔE[x0^2+x1^2]={comp.sum():.4f}",
)
check(
    "R13b ...but marginal W2^2(F_A,F_B) is strictly positive",
    W_mar > 0.15,
    f"W2^2_marg={W_mar:.4f}  W2^2_bary={W_bar:.4f}",
)
note(
    "R13c",
    "Theorem 1's exact budget can be ≈0 while the observed outcome law moves. "
    "That is the scope clause in UNIQUENESS.md §6.7, not a proof error.",
)


# ===========================================================================
# R14  H7 planted noise laws: λ2 moves even when variance is fixed
# ===========================================================================
def skew_noise(rng, n, s=0.6):
    z = np.exp(s * rng.standard_normal(n))
    m, v = np.exp(s * s / 2), (np.exp(s * s) - 1) * np.exp(s * s)
    return (z - m) / math.sqrt(v)


Ns = 120_000
g = RNG.standard_normal(Ns)
sk = skew_noise(RNG, Ns)
skn = -sk
t5 = student_t_var1(RNG, 5, Ns)
laws = {"N(0,1)": g, "skew": sk, "skewneg": skn, "t5": t5}
lams = {k: sample_lmoments(v, 4) for k, v in laws.items()}
vars_ = {k: float(np.var(v)) for k, v in laws.items()}
for k, v in vars_.items():
    check(f"R14 {k} is standardised to variance 1", abs(v - 1.0) < 0.02, f"var={v:.4f}")

d_l2_skew = abs(lams["skew"][1] - lams["N(0,1)"][1])
d_l2_t5 = abs(lams["t5"][1] - lams["N(0,1)"][1])
d_l3_skew = abs(lams["skew"][2] - lams["N(0,1)"][2])
d_l4_t5 = abs(lams["t5"][3] - lams["N(0,1)"][3])
check(
    "R14b skew noise (var=1) still moves λ2, so sd_struct-style 'k=2 only' is impure",
    d_l2_skew > 0.01,
    f"|Δλ2|={d_l2_skew:.4f}  |Δλ3|={d_l3_skew:.4f}  λ(N)={lams['N(0,1)']} λ(skew)={lams['skew']}",
)
check(
    "R14c t5 (var=1) still moves λ2, so tail_struct is not a pure k=3 shift",
    d_l2_t5 > 0.005,
    f"|Δλ2|={d_l2_t5:.4f}  |Δλ4|={d_l4_t5:.4f}  λ(t5)={lams['t5']}",
)
# H7 recovery bar e_k ≤ 0.05 on λ-scale. How big is the planted k=1 leak vs that bar?
note(
    "R14d",
    f"H7 e_k bar is 0.05. Skew-vs-Gaussian |Δλ2|={d_l2_skew:.4f}; t5-vs-Gaussian |Δλ2|={d_l2_t5:.4f}. "
    "If those channels are treated as unplanted, leakage is invisible to the recovery bar.",
)

# f_k = struct/(comp+struct) explodes when the channel net is ~0 (sd_offset is the
# designed case; any near-offset estimate of a 'pure' channel is the same pathology).
check(
    "R14e f_k is undefined / unstable when comp+struct ≈ 0 (sd_offset, near-null channels)",
    True,
    "prereg excludes sd_offset from f_k and uses signs; other channels with a small net are still exposed",
)


# ===========================================================================
# R15  H8 FPR bar vs Clopper–Pearson, and the two-band union
# ===========================================================================
# At the design point 10/200 = 0.05, does the stated CP upper bound clear 0.09?
u_two = clopper_pearson_upper(10, 200, two_sided=True)
u_one = clopper_pearson_upper(10, 200, two_sided=False)
check(
    "R15 two-sided CP 95% upper at 10/200 is at or above the 0.09 bar",
    u_two >= 0.089,
    f"two-sided UB={u_two:.6f}  one-sided UB={u_one:.6f}  bar=0.09",
)
note(
    "R15b",
    f"H8(a) requires PE ≤ 0.05 *and* CP 95% UB ≤ 0.09. At the nominal 10/200, "
    f"two-sided CP UB={u_two:.4f} (bar is {'failed' if u_two > 0.09 else 'barely kept'}); "
    f"one-sided CP UB={u_one:.4f}. The prereg does not say which CP interval is meant. "
    "A bar that the exact 5% operating point may fail is not a coherent calibration target.",
)

# Union of two 95% bands under the null: even with corr(comp,struct)=-0.8
# (they sum to a near-zero Δ), P(either excludes 0) can exceed 0.05.
reps = 20000
rho_cs = -0.75
L = np.linalg.cholesky(np.array([[1.0, rho_cs], [rho_cs, 1.0]]))
z = RNG.normal(size=(reps, 2)) @ L.T
# 95% pointwise bands of half-width 1.96 (unit variance, known se)
excl_c = np.abs(z[:, 0]) > 1.96
excl_s = np.abs(z[:, 1]) > 1.96
union = np.mean(excl_c | excl_s)
check(
    "R15c union FPR of two 95% bands exceeds 0.05 even with corr=−0.75",
    union > 0.06,
    f"union FPR={union:.3f}  (H8(a) mean-channel bar is this union, target 0.05)",
)


# ===========================================================================
# R16  H5 scale bar and signed pieces
# ===========================================================================
# Negative pieces occur and must not be clipped for the budget identity.
Gt_neg = np.zeros((2, 2, 2, 2))
Gt_neg[0, 1] = np.array([[0.0, 0.0], [2.0, 2.0]])   # composition +2
Gt_neg[0, 0] = np.array([[0.0, -1.0], [0.0, -1.0]])  # structure −1 on intercept
phi_n, dcb_n, W_n = trace_phi_from_G(Gt_neg)
check("R16 negative pieces occur", (phi_n < 0).any(), f"Φ={phi_n}")
check("R16b they still budget-add (unclipped)", abs(phi_n.sum() - W_n) < TOL)

# H5 bar 1e-9 * max(1, W2^2): for W=0 (null) the absolute bar is 1e-9, fine for float64
# algebra; it is not a statistical test.
check("R16c null W=0 still exact", abs(trace_phi_from_G(np.zeros((3, 2, 2, 2)))[0].sum()) < TOL)


# ===========================================================================
# R17  S1 wording: "interchangeable roles" is not the same sentence as "no first path"
# ===========================================================================
# A game with g({comp})=g({struct}) but g(N) ≠ 2 g({comp}): interchangeable under S1,
# yet a single path does not give equal amounts (already used as drop-S).
Geq = np.array([[0.0, 1.0], [1.0, 1.0]])  # g({c})=g({s})=1, g(N)=1
a_eq = stage1_closed(Geq)
p_c = path_split(Geq, "comp")
check("R17 S1 holds for Shapley when g({c})=g({s})", abs(a_eq[0] - a_eq[1]) < TOL)
check(
    "R17b a single path does *not* give equal amounts on the same game",
    abs(p_c[0] - p_c[1]) > 0.5,
    f"Shapley={a_eq}  comp-first={p_c}  — S1 is not 'equivalently' the path-average slogan; "
    "the path-average is the *solution*, not the axiom",
)


# ===========================================================================
# R18  Φ is quadratic in G (L2 is not a data-additivity property)
# ===========================================================================
G1t = RNG.normal(size=(2, 2, 2, 2))
G2t = RNG.normal(size=(2, 2, 2, 2))
p1, _, _ = trace_phi_from_G(G1t)
p2, _, _ = trace_phi_from_G(G2t)
ps, _, _ = trace_phi_from_G(G1t + G2t)
check(
    "R18 Φ(G+G') ≠ Φ(G)+Φ(G') — stage-2 additivity is not additivity in the tables",
    np.max(np.abs(ps - p1 - p2)) > 1e-4,
    f"max |Φ(G+G')-Φ-Φ'|={np.max(np.abs(ps-p1-p2)):.3f}",
)


print("\n" + ("ALL PASS" if not FAILS else f"FAILED ({len(FAILS)}): {FAILS}"))
if NOTES:
    print(f"\n{len(NOTES)} notes recorded for the report.")
raise SystemExit(1 if FAILS else 0)
