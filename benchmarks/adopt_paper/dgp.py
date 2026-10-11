"""Data-generating processes of Song & Müller (AISTATS 2026), Section 5.

Source: "ADOPT: Additive Optimal Transport Regression", PMLR v300, pp. 1558-1566,
https://proceedings.mlr.press/v300/song26c.html . Settings are transcribed from
Sections 5.1 and 5.2 of the paper; interpretive choices are listed in
docs/REPRODUCTION_NOTES.md and flagged ``# CHOICE`` below.

Requires scipy (``pip install traceot[sim]``). This is benchmark code that generates
simulated data; it is not part of the ``traceot`` runtime package.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import betaincinv
from scipy.stats import norm

from traceot import quantile_grid

#: Predictor correlation of V ~ N3(0, Sigma), X = Phi(V) (both sections).
SIGMA = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.5], [0.3, 0.5, 1.0]])


def draw_predictors(n: int, rng: np.random.Generator) -> np.ndarray:
    """``X_i = (Phi(V_i1), Phi(V_i2), Phi(V_i3))`` with ``V_i ~ N3(0, SIGMA)``."""
    V = rng.multivariate_normal(np.zeros(3), SIGMA, size=n)
    return norm.cdf(V)


# Gauss-Legendre rule on [0, 1] for population Fréchet means over X_j ~ Unif(0, 1)
# (each X_j = Phi(V_j) is marginally uniform).
_GL_T, _GL_W = np.polynomial.legendre.leggauss(128)
_GL_T = (_GL_T + 1.0) / 2.0
_GL_W = _GL_W / 2.0


# ---------------------------------------------------------------------------
# Section 5.1: distributional responses in W2
# ---------------------------------------------------------------------------


def _beta_q(a, b, u):
    return betaincinv(a, b, u)


def w2_component_quantile(case: str, j: int, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Quantile function of ``g_j(x)`` on levels ``u``; returns (len(x), len(u))."""
    x = np.asarray(x, dtype=float)[:, None]
    u = np.asarray(u, dtype=float)[None, :]
    if case == "I":
        if j == 0:  # Beta(1 + 2 x1, 1)
            return u ** (1.0 / (1.0 + 2.0 * x))
        if j == 1:  # Beta(1, 2 + 3 x2)
            return 1.0 - (1.0 - u) ** (1.0 / (2.0 + 3.0 * x))
        if j == 2:  # Beta(1/2 + x3/2, 1/2 + x3/2)
            a = 0.5 + 0.5 * x
            return _beta_q(a + 0 * u, a + 0 * u, u + 0 * x)
    elif case == "II":
        z = norm.ppf(u)
        if j == 0:  # N(x1, 1)
            return x + z
        if j == 1:  # N(x2^2, 1)
            return x**2 + z
        if j == 2:  # N(exp(-x3), 1)
            return np.exp(-x) + z
    raise ValueError(f"unknown case/component {case!r}/{j}")


@dataclass
class W2Truth:
    """Population objects for the W2 design: true maps T_j(x) = Q_{g_j(x)} o F_{m_j}."""

    case: str
    fine_u: np.ndarray  # fine grid used to tabulate the population means m_j
    m_q: list  # quantile functions of m_j = E⊕ g_j(X_j) on fine_u

    @classmethod
    def build(cls, case: str, n_fine: int = 20001) -> W2Truth:
        if case == "I":
            fine_u = np.linspace(0.0, 1.0, n_fine)  # Beta supports are [0, 1]
        else:
            # Normal case: maps are exact translations (see apply_map); no table needed.
            fine_u = np.linspace(0.0, 1.0, 3)
        m_q = []
        for j in range(3):
            if case == "I":
                Qt = w2_component_quantile(case, j, _GL_T, fine_u)  # (T, U)
                m_q.append(_GL_W @ Qt)
            else:
                m_q.append(None)
        return cls(case=case, fine_u=fine_u, m_q=m_q)

    def apply_map(self, j: int, x: float, y: np.ndarray) -> np.ndarray:
        """``T_j(x)(y) = Q_{g_j(x)}(F_{m_j}(y))`` evaluated pointwise."""
        y = np.asarray(y, dtype=float)
        if self.case == "II":
            # g_j(x) = N(c_j(x), 1) and m_j = N(E c_j(X_j), 1): T_j is the shift c_j(x) - E c_j.
            c = {0: lambda t: t, 1: lambda t: t**2, 2: lambda t: np.exp(-t)}[j]
            shift = float(c(np.asarray(x))) - float(_GL_W @ c(_GL_T))
            return y + shift
        F = np.interp(y, self.m_q[j], self.fine_u)  # F_{m_j}, m_j continuous on [0, 1]
        return w2_component_quantile(self.case, j, np.array([x]), F)[0]

    def true_quantiles(self, X: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Noise-free ``[T_1(x_1) ⊕ T_2(x_2) ⊕ T_3(x_3)](U)`` on levels ``u``; U = Unif(0,1)."""
        X = np.atleast_2d(X)
        out = np.empty((X.shape[0], u.shape[0]))
        for i, x in enumerate(X):
            z = u.copy()  # quantile function of Unif(0, 1)
            for j in range(3):
                z = self.apply_map(j, x[j], z)
            out[i] = z
        return out


def w2_noise(Q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """``eps_i(f)(u) = f(u) + xi_i sin(2 pi f(u)) / (2 pi)``, ``xi_i ~ Unif(-1, 1)``."""
    xi = rng.uniform(-1.0, 1.0, size=(Q.shape[0], 1))
    return Q + xi * np.sin(2.0 * np.pi * Q) / (2.0 * np.pi)


def simulate_w2(case: str, n: int, rng: np.random.Generator, q: int = 100, truth=None):
    """One Monte Carlo sample (X, Y) for Section 5.1, Y on the midpoint grid of size q."""
    truth = truth or W2Truth.build(case)
    u = quantile_grid(q)
    X = draw_predictors(n, rng)
    Y = w2_noise(truth.true_quantiles(X, u), rng)  # CHOICE: responses fully observed on grid
    return X, Y


# ---------------------------------------------------------------------------
# Section 5.2: SPD responses under the log-Cholesky metric
# ---------------------------------------------------------------------------


def spd_component_coords(case: str, j: int, x: np.ndarray, m: int) -> np.ndarray:
    """Log-Cholesky coordinates of ``g_j(x) = L_j(x) L_j(x)^T``; (len(x), m(m+1)/2)."""
    x = np.asarray(x, dtype=float)[:, None]
    r, c = np.tril_indices(m, -1)
    dd = np.abs(r - c)[None, :].astype(float)
    if case == "I":
        a, s = {0: (2.0, 2.0), 1: (3.0, 4.0), 2: (4.0, 8.0)}[j]
        off = np.exp(-dd / a) * (x + 1.0) / s
        diag = np.log((x + 1.0) / s)  # |k - l| = 0 on the diagonal
    elif case == "II":
        if j == 0:
            f = np.sin((x + 1.0) * np.pi / 4.0)
            off, diag = np.exp(-dd / 2.0) * f, np.log(f)
        elif j == 1:
            f = (x**2 + 1.0) / 2.0
            off, diag = np.exp(-dd / 4.0) * f, np.log(f)
        elif j == 2:
            f = np.exp(-x)
            off, diag = np.exp(-dd / 4.0) * f, -x
        else:
            raise ValueError(j)
    else:
        raise ValueError(case)
    return np.concatenate([off, np.repeat(diag, m, axis=1)], axis=1)


def spd_population_means(case: str, m: int) -> np.ndarray:
    """Coordinates of ``E⊕ g_j(X_j)`` (coordinate means over X_j ~ Unif(0,1)); (3, dim)."""
    return np.stack([_GL_W @ spd_component_coords(case, j, _GL_T, m) for j in range(3)])


def spd_true_coords(case: str, X: np.ndarray, m: int, means=None) -> np.ndarray:
    """Noise-free ``[⊕_j T_j(x_j)](I_m)`` in coordinates: ``sum_j phi(g_j(x_j)) - phi(m_j)``."""
    means = spd_population_means(case, m) if means is None else means
    X = np.atleast_2d(X)
    out = np.zeros((X.shape[0], m * (m + 1) // 2))  # phi(I_m) = 0
    for j in range(3):
        out += spd_component_coords(case, j, X[:, j], m) - means[j]
    return out


def simulate_spd(
    case: str, n: int, m: int, rng: np.random.Generator, sigma: float = 0.01, means=None
):
    """One Monte Carlo sample for Section 5.2, responses returned as coordinates.

    Noise: ``e(L)_{kl} = L_{kl} + xi_{kl}`` for k > l, ``xi ~ N(0, sigma^2)``; diagonal
    unchanged. The strict lower part of L *is* a block of the coordinates, so the
    noise is additive there.
    """
    X = draw_predictors(n, rng)
    Z = spd_true_coords(case, X, m, means)
    k = m * (m - 1) // 2
    Z[:, :k] += rng.normal(0.0, sigma, size=(n, k))
    return X, Z


def mise_grid(per_axis: int = 10) -> np.ndarray:
    """Midpoint grid on [0, 1]^3 used to approximate the integral in eq. (12)."""
    g = (np.arange(per_axis) + 0.5) / per_axis
    A, B, C = np.meshgrid(g, g, g, indexing="ij")
    return np.column_stack([A.ravel(), B.ravel(), C.ravel()])
