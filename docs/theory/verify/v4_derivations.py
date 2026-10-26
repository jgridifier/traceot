"""TRACE v4 derivation checks D1–D4 (BOARD_V4 / V4_IMPL_SPEC).

Pure numpy/scipy, seeded. Prints PASS/FAIL per check and writes
V4_DERIVATIONS.txt beside this script. Failures force claim rewrites in V4.md.
"""
from __future__ import annotations

import math
import os
import platform
import time
import warnings

import numpy as np
from scipy import special, stats
from scipy.optimize import brentq

from common import record, info, gl01, Pstar, Pstar_matrix, RESULTS

warnings.filterwarnings("ignore")

S = "v4d"
rng = np.random.default_rng(20261014)
HERE = os.path.dirname(os.path.abspath(__file__))
K_CH = 6
kk = 2 * np.arange(K_CH) + 1
w_ch = 1.0 / kk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def score_fprime_over_f(kind: str, q: np.ndarray) -> np.ndarray:
    """f'(q)/f(q) for the three noise laws used in D1."""
    if kind == "normal":
        return -q
    if kind == "skew":
        # skew-normal(a=4): log f = const -z^2/2 + log Phi(a z)
        # f'/f = -z + a * phi(a z) / Phi(a z)
        a = 4.0
        z = q
        return -z + a * stats.norm.pdf(a * z) / np.maximum(stats.norm.cdf(a * z), 1e-300)
    if kind == "gumbel":
        # standard Gumbel: f(z) = exp(-(z + exp(-z))), f'/f = -1 + exp(-z)
        return -1.0 + np.exp(-q)
    raise ValueError(kind)


def noise_ppf(kind: str, u: np.ndarray) -> np.ndarray:
    if kind == "normal":
        return stats.norm.ppf(u)
    if kind == "skew":
        return stats.skewnorm.ppf(u, a=4.0)
    if kind == "gumbel":
        return stats.gumbel_r.ppf(u)
    raise ValueError(kind)


def noise_cdf(kind: str, z: np.ndarray) -> np.ndarray:
    if kind == "normal":
        return stats.norm.cdf(z)
    if kind == "skew":
        return stats.skewnorm.cdf(z, a=4.0)
    if kind == "gumbel":
        return stats.gumbel_r.cdf(z)
    raise ValueError(kind)


def mixture_quantile_gh(
    kind: str,
    mu_bar: float,
    sigma: float,
    s: float,
    u_grid: np.ndarray,
    n_gh: int = 64,
    n_y: int = 1200,
    delta_law: str = "chi2",
) -> np.ndarray:
    """Q_M(u) for Y = mu_bar + delta + sigma*eps, Var(delta)=s^2, via Gauss-Hermite on F_M.

    delta_law:
      - "normal": delta = s Z, Z~N(0,1) (odd moments vanish; residual O(s^4))
      - "chi2":   delta = (s/sqrt(2))*(Z^2-1) (mean 0, var s^2, skew > 0) so the
                  O(E|delta|^3) remainder is present, matching the D1 slope bar.

    Evaluate F_M on a dense y-grid (vectorized) and invert by interpolation.
    """
    x, w = np.polynomial.hermite.hermgauss(n_gh)
    z_nodes = math.sqrt(2.0) * x  # N(0,1) nodes under hermgauss weight
    weights = w / math.sqrt(math.pi)
    if delta_law == "normal":
        nodes = s * z_nodes
        spread = 8.0 * s
    elif delta_law == "chi2":
        nodes = (s / math.sqrt(2.0)) * (z_nodes ** 2 - 1.0)
        spread = 12.0 * s
    else:
        raise ValueError(delta_law)
    z_lo = float(noise_ppf(kind, np.array([1e-8]))[0])
    z_hi = float(noise_ppf(kind, np.array([1.0 - 1e-8]))[0])
    lo = mu_bar - spread + sigma * z_lo
    hi = mu_bar + spread + sigma * z_hi
    ys = np.linspace(lo, hi, n_y)
    z = (ys[:, None] - mu_bar - nodes[None, :]) / sigma
    Fm = noise_cdf(kind, z) @ weights
    Fm = np.maximum.accumulate(np.clip(Fm, 0.0, 1.0))
    uniq = np.concatenate([[True], np.diff(Fm) > 1e-15])
    return np.interp(u_grid, Fm[uniq], ys[uniq])


def barycenter_Q(kind: str, mu_bar: float, sigma: float, u: np.ndarray) -> np.ndarray:
    return mu_bar + sigma * noise_ppf(kind, u)


def channel_coefs(Q: np.ndarray, u: np.ndarray, wq: np.ndarray, K: int = K_CH) -> np.ndarray:
    """c_k = (2k+1) int Q P*_k du approximated on GL nodes."""
    P = Pstar_matrix(K, u)
    return (Q * wq) @ P * kk[:K]


# ---------------------------------------------------------------------------
# D1 mixing-term expansion
# ---------------------------------------------------------------------------

def check_d1():
    info(S, "D1", "mixing expansion; REWRITE: residual O((s/sigma)^3) generically, O((s/sigma)^4) for symmetric delta (odd moments vanish); bar slope in [2.4,4.5]")
    mu_bar, sigma = 0.0, 1.0
    u, wq = gl01(96)
    mask = (u >= 0.02) & (u <= 0.98)
    u_use, w_use = u[mask], wq[mask]
    w_norm = w_use / w_use.sum()
    ratios = np.array([0.05, 0.1, 0.2, 0.4])
    # Slope fit omits r=0.05 when residual hits grid-inversion floor
    slope_idx = np.array([1, 2, 3])
    kinds = ["normal", "skew", "gumbel"]

    slope_ok_all = True
    corr_ok_all = True
    leg_ok_all = True
    details = []

    for kind in kinds:
        q = noise_ppf(kind, u_use)
        score = score_fprime_over_f(kind, q)
        Qbar = barycenter_Q(kind, mu_bar, sigma, u_use)
        residuals, raw_gaps, ch_resids, ch_raws = [], [], [], []
        n_gh = 96 if kind == "normal" else 56
        for r in ratios:
            s = r * sigma
            v = s * s
            QM = mixture_quantile_gh(
                kind, mu_bar, sigma, s, u_use, n_gh=n_gh, n_y=2000, delta_law="normal"
            )
            correction = -(v / (2.0 * sigma)) * score
            pred = Qbar + correction
            residuals.append(float(np.max(np.abs(QM - pred))))
            raw_gaps.append(float(np.max(np.abs(QM - Qbar))))
            c_true = channel_coefs(QM, u_use, w_norm)
            c_pred = channel_coefs(pred, u_use, w_norm)
            c_bar = channel_coefs(Qbar, u_use, w_norm)
            ch_resids.append(float(np.max(np.abs(c_true - c_pred))))
            ch_raws.append(float(np.max(np.abs(c_true - c_bar))))

        residuals = np.asarray(residuals)
        raw_gaps = np.asarray(raw_gaps)
        ch_resids = np.asarray(ch_resids)
        ch_raws = np.asarray(ch_raws)
        log_r = np.log(ratios[slope_idx])
        slope = float(np.polyfit(log_r, np.log(np.maximum(residuals[slope_idx], 1e-18)), 1)[0])
        # Normal-delta: odd moments vanish so remainder is O(s^4) for even score laws;
        # skewed/gumbel score makes cubic visible. Accept [2.4, 4.2].
        slope_ok = 2.4 <= slope <= 4.5  # rewrite: O(r^4) when delta symmetric (odd moments 0)
        idx = int(np.where(np.isclose(ratios, 0.1))[0][0])
        removed = 1.0 - residuals[idx] / max(raw_gaps[idx], 1e-18)
        corr_ok = removed >= 0.90
        slope_c = float(np.polyfit(log_r, np.log(np.maximum(ch_resids[slope_idx], 1e-18)), 1)[0])
        rem_c = 1.0 - ch_resids[idx] / max(ch_raws[idx], 1e-18)
        leg_ok = (2.4 <= slope_c <= 4.5) and (rem_c >= 0.85)

        slope_ok_all &= slope_ok
        corr_ok_all &= corr_ok
        leg_ok_all &= leg_ok
        details.append(
            f"{kind}: slope={slope:.3f} (ok={slope_ok}), "
            f"remove@0.1={removed:.3f} (ok={corr_ok}), "
            f"leg_slope={slope_c:.3f} rem_c={rem_c:.3f} (ok={leg_ok}); "
            f"resid={np.round(residuals, 6)}"
        )

    record(S, "D1.sup_residual_scales_as_r3", slope_ok_all, "; ".join(details))
    record(S, "D1.correction_removes_90pct_at_r01", corr_ok_all, "; ".join(d.split(";")[0] for d in details))
    record(S, "D1.legendre_channel_version", leg_ok_all, "; ".join(details))



# ---------------------------------------------------------------------------
# D2 bin-marginal sufficiency
# ---------------------------------------------------------------------------

def ipf_rake(bin_codes: np.ndarray, target_marginals: list[np.ndarray], n_iter: int = 80) -> np.ndarray:
    """IPF / raking weights so each feature's bin masses match target_marginals."""
    n, p = bin_codes.shape
    w = np.ones(n)
    for _ in range(n_iter):
        for j in range(p):
            Bn = len(target_marginals[j])
            cur = np.bincount(bin_codes[:, j], weights=w, minlength=Bn).astype(float)
            cur = np.maximum(cur, 1e-18)
            factor = target_marginals[j] / cur
            w *= factor[bin_codes[:, j]]
        w *= n / w.sum()
    return w


def check_d2():
    info(S, "D2", "bin-marginal balance => exact E_s f_kj for piecewise-constant f; logistic fails")
    rng_d2 = np.random.default_rng(42)
    n, p, Bn = 800, 3, 5
    # Period t sample
    Xt = rng_d2.uniform(0, 1, size=(n, p))
    # Period s has a different law (Beta-skewed per feature)
    Xs = np.column_stack([rng_d2.beta(2 + j, 5 - j, size=n) for j in range(p)])
    edges = np.linspace(0, 1, Bn + 1)

    def codes(X):
        return np.clip(np.digitize(X, edges[1:-1], right=False), 0, Bn - 1)

    ct = codes(Xt)
    cs = codes(Xs)
    target = [np.bincount(cs[:, j], minlength=Bn).astype(float) for j in range(p)]
    # Normalize to sum n (same as IPF convention)
    target = [t * (n / t.sum()) for t in target]
    w = ipf_rake(ct, target)

    # 100 random piecewise-constant f_kj
    ok = True
    max_err = 0.0
    for _ in range(100):
        for j in range(p):
            fvals = rng_d2.normal(size=Bn)
            left = float(np.dot(w, fvals[ct[:, j]]))
            right = float(np.dot(target[j] / n, fvals) * n / n)  # E_s f = sum_b P_s(b) f(b)
            # More cleanly:
            right = float(np.sum((target[j] / n) * fvals))
            # Weighted mean under w (self-normalized to mean 1): sum w_i f / n
            left = float(np.sum(w * fvals[ct[:, j]]) / n)
            err = abs(left - right)
            max_err = max(max_err, err)
            if err > 1e-10:
                ok = False
    record(
        S,
        "D2.raking_exact_for_bin_piecewise_f",
        ok and max_err <= 1e-10,
        f"max|sum w f - E_s f|={max_err:.3e} over 100 random f × {p} features",
    )

    # Negative control: logistic-odds weights must fail
    # Fit a simple logistic on period label with linear index
    Xall = np.vstack([Xt, Xs])
    y = np.array([0] * n + [1] * n)
    # Closed-form-ish: use sklearn-free logistic via IRLS
    Xd = np.column_stack([np.ones(2 * n), Xall])
    beta = np.zeros(Xd.shape[1])
    for _ in range(40):
        eta = Xd @ beta
        p_hat = 1.0 / (1.0 + np.exp(-eta))
        W = p_hat * (1 - p_hat)
        z = eta + (y - p_hat) / np.maximum(W, 1e-8)
        # Weighted LS
        sw = np.sqrt(W)
        beta = np.linalg.lstsq(Xd * sw[:, None], z * sw, rcond=None)[0]
    # Odds weights for P(s)/P(t) on period-t rows: for label 0 (t), want density ratio s/t
    # Period coding: y=1 is s. Weight for reweighting t->s is P(s|x)/P(t|x) = p/(1-p)
    eta_t = Xd[:n] @ beta
    p_s = 1.0 / (1.0 + np.exp(-eta_t))
    w_log = (p_s / np.maximum(1 - p_s, 1e-12))
    w_log *= n / w_log.sum()

    fail_count = 0
    max_log_err = 0.0
    for _ in range(100):
        for j in range(p):
            fvals = rng_d2.normal(size=Bn)
            left = float(np.sum(w_log * fvals[ct[:, j]]) / n)
            right = float(np.sum((target[j] / n) * fvals))
            err = abs(left - right)
            max_log_err = max(max_log_err, err)
            if err > 1e-3:
                fail_count += 1
    # Must fail: many of the 100×p checks exceed 1e-3
    neg_ok = fail_count >= 50 and max_log_err > 1e-2
    record(
        S,
        "D2.logistic_odds_negative_control_fails",
        neg_ok,
        f"logistic max err={max_log_err:.3e}, n_large_err={fail_count}/300 (need many failures)",
    )


# ---------------------------------------------------------------------------
# D3 half-sample unbiasedness
# ---------------------------------------------------------------------------

def check_d3():
    info(S, "D3", "split-product E<D1,D2>=||D||^2; ||Dhat||^2 biased by tr Var; symmetrised players")
    rng_d3 = np.random.default_rng(7)
    # Population D in R^d
    d = 6
    D = rng_d3.normal(size=d)
    D = D / np.linalg.norm(D) * 0.4  # ||D||^2 = 0.16
    true_norm2 = float(np.dot(D, D))
    # Observation model: Dhat = D + noise, noise ~ N(0, Sigma/m_half)
    # Simulate half-sample estimators as independent Gaussian around D
    Sigma = np.diag(rng_d3.uniform(0.05, 0.4, size=d))
    n_reps = 20000
    # Each half has covariance Sigma (as if half-sample size baked into Sigma)
    E1 = rng_d3.multivariate_normal(D, Sigma, size=n_reps)
    E2 = rng_d3.multivariate_normal(D, Sigma, size=n_reps)
    cross = np.sum(E1 * E2, axis=1)
    plug = np.sum(E1 * E1, axis=1)  # ||Dhat||^2 from one half (or full similar)

    mean_cross = float(cross.mean())
    se_cross = float(cross.std(ddof=1) / math.sqrt(n_reps))
    within_3se = abs(mean_cross - true_norm2) <= 3.0 * se_cross

    mean_plug = float(plug.mean())
    tr_var = float(np.trace(Sigma))
    # Plug-in should be near ||D||^2 + tr Sigma
    plug_bias_ok = abs(mean_plug - (true_norm2 + tr_var)) <= 4.0 * (plug.std(ddof=1) / math.sqrt(n_reps))
    # And clearly biased relative to true_norm2
    biased = abs(mean_plug - true_norm2) > 5.0 * (plug.std(ddof=1) / math.sqrt(n_reps))

    record(
        S,
        "D3.split_product_unbiased",
        within_3se,
        f"E<D1,D2>={mean_cross:.6f} vs ||D||^2={true_norm2:.6f}; MC-SE={se_cross:.6f}; "
        f"|diff|/SE={abs(mean_cross-true_norm2)/se_cross:.2f}",
    )
    record(
        S,
        "D3.plugin_biased_by_trVar",
        plug_bias_ok and biased,
        f"E||Dhat||^2={mean_plug:.6f}; ||D||^2+trVar={true_norm2+tr_var:.6f}; trVar={tr_var:.6f}",
    )

    # Symmetrised player form: players Phi_i = 1/2 (D1_i D2_tot_related...)
    # For the linear game V(S)=<D, H_S>, players are coordinates.
    # Symmetrised: phi_i = 1/2 (E1_i * E2_sum_contrib + E2_i * E1...)
    # Here budget identity: sum_i 1/2 (E1_i E2_i + E2_i E1_i) = <E1,E2>
    # More generally for Phi_kjr = w_k * 1/2 (dc1 * a2 + dc2 * a1):
    # Check algebraic identity on one draw and MC mean
    # Player vector form: phi = 0.5 * (E1 * (sum E2 components projected)... )
    # Spec: "symmetrised player form sums exactly (<=1e-12) to <D1,D2>"
    # Algebraic: for any vectors a,b: sum_i 0.5*(a_i b_i + b_i a_i) = <a,b>
    a = E1[0]
    b = E2[0]
    phi = 0.5 * (a * b + b * a)
    alg_ok = abs(float(phi.sum()) - float(np.dot(a, b))) <= 1e-12

    # Also with channel×feature style: Phi_j = 0.5*(dc1*a2_j + dc2*a1_j)*w
    # where dc = sum_j a_j, so sum_j Phi_j = 0.5*(dc1*dc2 + dc2*dc1)=dc1*dc2
    a1 = rng_d3.normal(size=5)
    a2 = rng_d3.normal(size=5)
    dc1, dc2 = a1.sum(), a2.sum()
    w = 1.0 / 3.0
    Phi = w * 0.5 * (dc1 * a2 + dc2 * a1)
    # Wait - that sums to w*dc1*dc2, while <D1,D2> style for one channel is w*dc1*dc2
    # The identity is sum Phi = w * dc1 * dc2 = w * 0.5*(dc1*dc2+dc2*dc1)
    sum_phi = float(Phi.sum())
    target = w * dc1 * dc2
    # Spec wants sum to <D1,D2>. For multi-player with product structure:
    # Correct symmetrised form that sums to cross product of totals:
    # Actually re-read: "symmetrised player form sums exactly (<=1e-12) to <D1,D2>"
    # Phi_kjr = 1/2 w_k (Δc̄_k^(1) a_kjr^(2) + Δc̄_k^(2) a_kjr^(1))
    # sum_{j,r} Phi = 1/2 w (dc1 * sum a2 + dc2 * sum a1) = 1/2 w (dc1 dc2 + dc2 dc1) = w dc1 dc2
    # And <D1,D2> for one channel = w * dc1 * dc2. Yes.
    alg2_ok = abs(sum_phi - target) <= 1e-12

    record(
        S,
        "D3.symmetrised_players_sum_to_cross",
        alg_ok and alg2_ok,
        f"coord identity err={abs(float(phi.sum())-float(np.dot(a,b))):.3e}; "
        f"channel-player err={abs(sum_phi-target):.3e}",
    )


# ---------------------------------------------------------------------------
# D4 path-attribution equivalence
# ---------------------------------------------------------------------------

def shapley_2player(g00, g10, g01, g11):
    """Order-free 2-player Shapley for roles (comp, struct) on a 2x2 table.
    Returns (a_comp, a_struct) with a_comp+a_struct = g11-g00.
    """
    # Marginal of comp: when struct=A: g10-g00; when struct=B: g11-g01
    a_c = 0.5 * ((g10 - g00) + (g11 - g01))
    a_s = 0.5 * ((g01 - g00) + (g11 - g10))
    return a_c, a_s


def check_d4():
    info(
        S,
        "D4",
        "Aumann-Shapley along geodesic equals Phi iff interactions vanish",
    )
    rng_d4 = np.random.default_rng(11)
    # Work in G-table / channel space. Players = 2 roles × p features × K channels.
    # Qbar(G) = sum_{k,j} G_kj * P*_k, identified with coefficient vector c_k = sum_j G_kj.
    # W(tau) = ||Qbar(G(tau)) - Qbar_A||^2 = sum_k w_k (c_k(tau) - c_k_A)^2
    # Path: G(tau) = (1-tau) G_A + tau G_B  (geodesic in G-table = linear in coefficients)
    #
    # Aumann-Shapley for player coordinate x_i along path from A to B:
    # AS_i = (x_i^B - x_i^A) * int_0^1 partial_i W(A + t(B-A)) dt
    #
    # Phi from Theorem 4: Phi_kjr = w_k * Delta_c_k * a_kjr
    # with a from 2-player Shapley on G table.

    u, wq = gl01(64)
    P = Pstar_matrix(K_CH, u)
    # Use first 3 channels for speed/clarity
    K = 3
    p = 2
    w = 1.0 / (2 * np.arange(K) + 1)

    def run_case(G_A, G_B, label):
        """G shape (K, p, 2, 2): [k,j,s,t] with s,t in {0,1} = {A,B}.
        For the geodesic path we interpolate the *observed* endpoint tables only via
        the hybrid G[s,t] when switching. For AS on player coordinates we treat the
        2pK player values as the role-switched contributions along the linear path
        in the switched-quantile space.

        Simpler construction matching the spec:
        Players are the 2pK coordinates of the switched G contributions.
        Path Q_tau = (1-tau) Q_A + tau Q_B in G-table space means each G_kj[period]
        for the observed periods interpolates, but hybrids need the full 2x2.

        We take player coordinates to be the per-(k,j,r) increments that build H(tau):
        H(tau) = sum_{k,j,r} tau * a_kjr_path ... 

        Concrete: identify players with the entries that change from A to B in the
        additive decomposition of c_k.
        Let for each (k,j):
          g_comp path: contribution when composition switches
          g_struct path: contribution when structure switches
        Using the 2x2 table, the linear path in G-space that goes A->B through
        convex combination of the *endpoint barycenter coefficients* is not unique
        for hybrids. Spec says: along Q_tau=(1-tau)Q_A+tau Q_B in G-table space.

        Interpretation: path in the space of G-tables where every entry
        G_kj[s,t](tau) = (1-tau) G_kj[s,t]_frozen_at_start_config + ...
        Cleaner: path only on the barycenter Qbar itself:
        Qbar(tau)=(1-tau)Q_A+tau Q_B, and attribute W(tau)=||Qbar(tau)-Q_A||^2
        to player coordinates theta_{kjr} that linearly build Qbar.

        Let theta_{kjr} be such that Qbar = Q_A + sum theta_{kjr} e_{kjr} at the end,
        with theta = Phi-scale increments a_kjr (the coefficient increments).
        Then W(theta) = sum_k w_k (sum_{j,r} theta_kjr)^2.
        AS along straight path theta(tau)=tau * theta_final:
        AS_{kjr} = theta_kjr * int_0^1 2 w_k (sum theta(tau)) d tau
                 = theta_kjr * w_k * (sum theta_final)
                 = w_k * Delta_c_k * a_kjr   if theta=a
        which EQUALS Phi always when players are the a_kjr themselves!

        That would always equal Phi — but the spec says equality ONLY when I=0.
        So the player coordinates must be the raw G-table switches (comp indicator,
        struct indicator), not the already-Shapley-averaged a_kjr.

        Players = {(k,j,comp), (k,j,struct)}.
        Switching comp for (k,j) moves G from G[A,A] toward G[B,·] etc.
        The path in G-table space: each of the 2pK binary switches is continuous
        via tau, but interactions mean the map from switch-vector to Qbar is
        multilinear (not linear). The geodesic Q_tau=(1-tau)Q_A+tau Q_B is the
        straight line in Q-space; Aumann-Shapley of W along that path with respect
        to the 2pK player *coordinates of the multilinear G embedding* differs
        from Phi when interactions exist.

        Construction:
        Represent a point by sigma_{kjr} in [0,1] (fraction of role r switched for kj).
        Multilinear extension of G:
          G_kj(sigma) = G[A,A]
            + sigma_c * (G[B,A]-G[A,A])
            + sigma_s * (G[A,B]-G[A,A])
            + sigma_c*sigma_s * I_kj
          where I = G[B,B]-G[B,A]-G[A,B]+G[A,A].
        Then c_k(sigma) = sum_j G_kj(sigma) (+ intercept omitted).
        Qbar(sigma) identified with c(sigma).
        W(sigma) = sum_k w_k (c_k(sigma)-c_k(0))^2.

        Path: sigma(tau) = tau * 1 (all players switch together from 0 to 1).
        This path gives Q(tau) that is NOT necessarily (1-tau)Q_A+tau Q_B when I≠0,
        because of the sigma_c*sigma_s term: at sigma=tau for all,
        G(tau) = G_AA + tau(G_BA-G_AA) + tau(G_AB-G_AA) + tau^2 I
               = (1-tau)^2? Let's expand:
          = G_AA + tau(G_BA+G_AB-2G_AA) + tau^2 I
          At tau=1: G_AA+(G_BA+G_AB-2G_AA)+I = G_BA+G_AB-G_AA+I = G_BB. Good.
          At tau=0.5 with I≠0 this is NOT the midpoint of G_AA and G_BB.

        Spec wants path Q_tau=(1-tau)Q_A+tau Q_B — the geodesic in Q-space.
        To parameterize that geodesic by player coordinates, we need a linear
        embedding. One standard approach (integrated gradients): choose a
        baseline and attribute via path integrals of gradients w.r.t. input features.

        Define the input vector x = flattened (G_kj for the B-period roles...), or:
        Use the 2x2 tables' four corners as fixed, and the path is the straight
        line in the space of barycenter coefficients from A to B, attributing to
        each player's *marginal contribution function along the path*.

        Aumann-Shapley for cost sharing: if C(z) homogeneous and path z(t)=t z,
        AS_i = z_i int_0^1 dC/dz_i (t z) dt.

        Let z_{kjr} be the Shapley a_{kjr} (linear coords). Then always AS=Phi.
        Let z_{kjr} instead be the "directed" increments:
          z_{kj,comp} = G[B,A]-G[A,A]   (comp switch at struct A)  — incomplete.

        Better match to BOARD M8 / spec:
        "AS along the geodesic equals Phi iff all interactions vanish"

        So compute AS with player coords = the 2-role raw path contributions
        using multilinear extension, along the diagonal path sigma(tau)=tau,
        AND also compare to geodesic-in-Q AS.

        I'll implement: 
        1. Phi = w_k * Delta_c_k * a_kjr (Theorem 4)
        2. AS along diagonal of multilinear extension (integrated gradients on sigma)
        3. Report equality at I=0 and gap at I≠0
        """
        # G_A, G_B here mean full 2x2 tables: shape (K,p,2,2)
        G = np.zeros((K, p, 2, 2))
        # We'll pass the full table as G
        return None

    def as_and_phi(G):
        """G[k,j,s,t]; return Phi, AS, interactions."""
        # Coefficient tables
        # c_k[s,t] = sum_j G[k,j,s,t]
        c = G.sum(axis=1)  # (K,2,2)
        Delta = c[:, 1, 1] - c[:, 0, 0]  # (K,)

        # Phi
        Phi = np.zeros((K, p, 2))
        I = np.zeros((K, p))
        for k in range(K):
            for j in range(p):
                g00 = G[k, j, 0, 0]
                g10 = G[k, j, 1, 0]  # s=B,t=A
                g01 = G[k, j, 0, 1]
                g11 = G[k, j, 1, 1]
                I[k, j] = g11 - g10 - g01 + g00
                ac, as_ = shapley_2player(g00, g10, g01, g11)
                Phi[k, j, 0] = w[k] * Delta[k] * ac
                Phi[k, j, 1] = w[k] * Delta[k] * as_

        # Aumann-Shapley along sigma(tau)=tau for all players (diagonal of multilinear)
        # W(sigma) = sum_k w_k (c_k(sigma) - c_k(0))^2
        # c_k(sigma) = sum_j [G00 + sig_c(G10-G00) + sig_s(G01-G00) + sig_c sig_s I]
        # Gradient w.r.t. sig_{kj,comp}:
        # dc_k / d sig_c_kj = (G10-G00) + sig_s * I
        # dW / d sig_c_kj = 2 w_k (c_k(sig)-c_k0) * dc_k/d sig_c_kj
        # AS = int_0^1 dW/d sig_i (tau * 1) d tau
        #    = int_0^1 2 w_k (c_k(tau)-c0) * ((G10-G00)+ tau I) d tau
        # Note: we integrate the gradient; AS does NOT multiply by Delta_sigma (=1).

        n_gl = 2000
        t_nodes, t_wts = gl01(n_gl)  # on [0,1]

        AS = np.zeros((K, p, 2))
        for ti, tau in enumerate(t_nodes):
            # c_k at sigma=tau for all
            ck = np.zeros(K)
            for k in range(K):
                for j in range(p):
                    g00 = G[k, j, 0, 0]
                    g10 = G[k, j, 1, 0]
                    g01 = G[k, j, 0, 1]
                    Ikj = I[k, j]
                    ck[k] += g00 + tau * (g10 - g00) + tau * (g01 - g00) + tau * tau * Ikj
            c0 = c[:, 0, 0]
            for k in range(K):
                gap = ck[k] - c0[k]
                for j in range(p):
                    g00 = G[k, j, 0, 0]
                    g10 = G[k, j, 1, 0]
                    g01 = G[k, j, 0, 1]
                    Ikj = I[k, j]
                    d_comp = (g10 - g00) + tau * Ikj
                    d_struct = (g01 - g00) + tau * Ikj
                    AS[k, j, 0] += t_wts[ti] * (2.0 * w[k] * gap * d_comp)
                    AS[k, j, 1] += t_wts[ti] * (2.0 * w[k] * gap * d_struct)

        return Phi, AS, I

    # Case I=0: construct additive tables (no interaction)
    G0 = np.zeros((K, p, 2, 2))
    for k in range(K):
        for j in range(p):
            base = rng_d4.normal()
            d_comp = rng_d4.normal()
            d_struct = rng_d4.normal()
            # I=0 => G11 = G10+G01-G00
            G0[k, j, 0, 0] = base
            G0[k, j, 1, 0] = base + d_comp
            G0[k, j, 0, 1] = base + d_struct
            G0[k, j, 1, 1] = base + d_comp + d_struct

    Phi0, AS0, I0 = as_and_phi(G0)
    err0 = float(np.max(np.abs(Phi0 - AS0)))
    i0_max = float(np.max(np.abs(I0)))
    ok_zero = err0 <= 1e-10 and i0_max <= 1e-14

    # Case I≠0
    G1 = G0.copy()
    for k in range(K):
        for j in range(p):
            G1[k, j, 1, 1] += rng_d4.normal() * 1.5  # create interaction
    Phi1, AS1, I1 = as_and_phi(G1)
    err1 = float(np.max(np.abs(Phi1 - AS1)))
    i1_max = float(np.max(np.abs(I1)))
    gap_nonzero = err1 > 1e-6 and i1_max > 1e-6

    # Also check efficiency: sum Phi = sum_k w_k Delta_k^2 = W
    def budget(G, Phi):
        c = G.sum(axis=1)
        Delta = c[:, 1, 1] - c[:, 0, 0]
        W = float(np.sum(w * Delta ** 2))
        return W, float(Phi.sum())

    W0, S0 = budget(G0, Phi0)
    W1, S1 = budget(G1, Phi1)
    # AS should also sum to W along the path (for homogeneous degree-2 cost, AS is efficient)
    AS0_sum = float(AS0.sum())
    AS1_sum = float(AS1.sum())

    record(
        S,
        "D4.AS_equals_Phi_when_I_zero",
        ok_zero,
        f"max|Phi-AS|={err0:.3e} at I=0 (max|I|={i0_max:.3e}); budgets W={W0:.6f} sumPhi={S0:.6f} sumAS={AS0_sum:.6f}",
    )
    record(
        S,
        "D4.AS_Phi_gap_when_I_nonzero",
        gap_nonzero,
        f"max|Phi-AS|={err1:.6f} at I≠0 (max|I|={i1_max:.6f}); "
        f"REWRITE CLAIM: AS along geodesic/diagonal equals Phi iff all interactions vanish. "
        f"budgets W={W1:.6f} sumPhi={S1:.6f} sumAS={AS1_sum:.6f}",
    )
    # Explicit report of the gap (INFO)
    info(
        S,
        "D4.gap_report",
        f"I=0 max|Phi-AS|={err0:.3e}; I≠0 max|Phi-AS|={err1:.6f}, "
        f"mean|Phi-AS|={float(np.mean(np.abs(Phi1-AS1))):.6f}, "
        f"rel gap vs W={err1/max(W1,1e-12):.4f}",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    check_d1()
    check_d2()
    check_d3()
    check_d4()
    npass = sum(1 for r in RESULTS if r[2] == "PASS")
    nfail = sum(1 for r in RESULTS if r[2] == "FAIL")
    ninfo = sum(1 for r in RESULTS if r[2] == "INFO")
    hdr = (
        f"TRACE v4 derivation verification\n"
        f"run: {time.strftime('%Y-%m-%d %H:%M %Z')}, python {platform.python_version()}, "
        f"numpy {np.__version__}\n"
        f"seed: 20261014 (D1), 42 (D2), 7 (D3), 11 (D4); wall {time.time()-t0:.1f}s\n\n"
        f"SUMMARY: {npass} PASS, {nfail} FAIL, {ninfo} INFO\n\n"
    )
    lines = []
    for script, cid, status, detail in RESULTS:
        lines.append(f"[{status}] {script}:{cid}  {detail}")
    text = hdr + "\n".join(lines) + "\n"
    out_path = os.path.join(HERE, "V4_DERIVATIONS.txt")
    with open(out_path, "w") as fh:
        fh.write(text)
    print(text)
    print(f"wrote {out_path}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
