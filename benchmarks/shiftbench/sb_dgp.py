"""ShiftBench synthetic suite: data-generating processes with a planted split.

Every scenario defines two periods, A (reference) and B. Each period has a feature
law X_t, a conditional mean mu_t(x) = a_t + sum_j g_tj(x_j), and a noise law
y = mu_t(x) + s_t(x) * e_t. The planted split of a statistic theta of y is

    composition = theta(Y_c) - theta(Y_A),   structure = theta(Y_B) - theta(Y_c),

where Y_c is the counterfactual outcome: X drawn from period B, everything else
(mean, noise scale, noise law) from period A. Per-feature truth exists for the mean
because the mean is additive: composition_j = E_B g_Aj - E_A g_Aj and
structure_j = E_B[g_Bj - g_Aj] - E_P[g_Bj - g_Aj], with P the 50/50 pooled law
of X (the same centring the harness applies to every method). The intercept term is
the structure residual. See docs/benchmark/SHIFTBENCH.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

N_ACTIVE = 5
STATS = ("mean", "sd", "q10", "q50", "q90")
SCENARIOS = (
    "null",
    "loc_comp",
    "loc_struct",
    "mixed",
    "scale",
    "tail",
    "correlated",
    "offsetting",
    "hetero",
    "nonlinear",
    "entry_exit",
)


def stat(y: np.ndarray, name: str, w: np.ndarray | None = None) -> float:
    """Statistic of a (weighted) sample. Weighted quantiles use the step CDF."""
    if w is None:
        if name == "mean":
            return float(np.mean(y))
        if name == "sd":
            return float(np.std(y))
        return float(np.quantile(y, int(name[1:]) / 100))
    w = w / w.sum()
    m = float(np.sum(w * y))
    if name == "mean":
        return m
    if name == "sd":
        return float(np.sqrt(np.sum(w * (y - m) ** 2)))
    o = np.argsort(y)
    cw = np.cumsum(w[o])
    k = min(int(np.searchsorted(cw, int(name[1:]) / 100)), len(y) - 1)
    return float(y[o][k])


def _skew_noise(rng, n):
    """Standardised centred lognormal: mean 0, variance 1, right skew."""
    s = 0.6
    z = np.exp(s * rng.standard_normal(n))
    m, v = np.exp(s * s / 2), (np.exp(s * s) - 1) * np.exp(s * s)
    return (z - m) / np.sqrt(v)


def _skew_feature(rng, n):
    """Standardised centred chi-square(3): mean 0, variance 1, right skew."""
    return (rng.chisquare(3, n) - 3) / np.sqrt(6)


@dataclass
class Period:
    a: float
    beta: np.ndarray  # coefficients on the active basis
    basis: tuple  # per active feature: "lin", "sq", "sin"
    sigma: float
    het: float = 0.0  # s(x) = sigma * exp(het * x0)
    noise: str = "normal"

    def g(self, X: np.ndarray) -> np.ndarray:
        """Per-active-feature contributions, shape (n, N_ACTIVE)."""
        out = np.empty((X.shape[0], N_ACTIVE))
        for j in range(N_ACTIVE):
            x = X[:, j]
            b = self.basis[j]
            h = x if b == "lin" else (x * x if b == "sq" else np.sin(2 * x))
            out[:, j] = self.beta[j] * h
        return out

    def draw_y(self, X, rng):
        e = rng.standard_normal(len(X)) if self.noise == "normal" else _skew_noise(rng, len(X))
        s = self.sigma * np.exp(self.het * X[:, 0]) if self.het else self.sigma
        return self.a + self.g(X).sum(axis=1) + s * e


@dataclass
class Scenario:
    name: str
    p: int
    A: Period
    B: Period
    xa: dict = field(default_factory=dict)  # feature-law parameters
    xb: dict = field(default_factory=dict)

    def draw_x(self, which: str, n: int, rng) -> np.ndarray:
        prm = self.xa if which == "A" else self.xb
        rho = prm.get("rho", 0.0)
        Z = rng.standard_normal((n, self.p))
        if rho:
            # AR(1) correlation among the active block.
            for j in range(1, N_ACTIVE):
                Z[:, j] = rho * Z[:, j - 1] + np.sqrt(1 - rho * rho) * Z[:, j]
        if prm.get("skew1"):
            Z[:, 1] = _skew_feature(rng, n)
        Z = Z * prm.get("scale", np.ones(self.p)) + prm.get("shift", np.zeros(self.p))
        if prm.get("entrants"):
            # A fraction of period-B units are entrants from a shifted law.
            frac, sh = prm["entrants"]
            m = rng.random(n) < frac
            Z[m] = Z[m] + sh
        return Z

    def sample(self, n: int, rng):
        XA = self.draw_x("A", n, rng)
        XB = self.draw_x("B", n, rng)
        return XA, self.A.draw_y(XA, rng), XB, self.B.draw_y(XB, rng)

    def truth(self, n: int = 2_000_000, seed: int = 99) -> dict:
        """Oracle planted split by Monte Carlo on n draws per law."""
        rng = np.random.default_rng(seed)
        XA, XB = self.draw_x("A", n, rng), self.draw_x("B", n, rng)
        yA, yB, yC = self.A.draw_y(XA, rng), self.B.draw_y(XB, rng), self.A.draw_y(XB, rng)
        out = {"sd_A": float(np.std(yA))}
        for s in STATS:
            tA, tB, tC = stat(yA, s), stat(yB, s), stat(yC, s)
            out[s] = {"total": tB - tA, "comp": tC - tA, "struct": tB - tC}
        gAA, gAB, gBB = self.A.g(XA), self.A.g(XB), self.B.g(XB)
        dpool = 0.5 * ((self.B.g(XA) - self.A.g(XA)).mean(0) + (gBB - gAB).mean(0))
        comp = np.zeros(self.p)
        struct = np.zeros(self.p)
        comp[:N_ACTIVE] = gAB.mean(0) - gAA.mean(0)
        struct[:N_ACTIVE] = (gBB - gAB).mean(0) - dpool
        out["feat_comp"] = comp.tolist()
        out["feat_struct"] = struct.tolist()
        out["intercept"] = out["mean"]["struct"] - float(struct.sum())
        return out


def make(name: str, p: int = 5) -> Scenario:
    beta = np.array([1.0, -0.8, 0.6, 0.4, -0.3])
    lin = ("lin",) * N_ACTIVE
    A = Period(a=1.0, beta=beta, basis=lin, sigma=1.0)
    B = Period(a=1.0, beta=beta.copy(), basis=lin, sigma=1.0)
    xa: dict = {}
    xb: dict = {}

    def shift(**kw):
        v = np.zeros(p)
        for k, val in kw.items():
            v[int(k[1:])] = val
        return v

    if name == "null":
        pass
    elif name == "loc_comp":
        xb["shift"] = shift(x0=0.5, x1=-0.3, x2=0.2)
    elif name == "loc_struct":
        B.beta = beta + np.array([0.3, 0.0, -0.2, 0.0, 0.0])
        B.a = 1.1
    elif name == "mixed":
        xb["shift"] = shift(x0=0.5, x3=0.4)
        B.beta = beta + np.array([0.0, 0.3, 0.0, -0.2, 0.0])
        B.a = 0.9
    elif name == "scale":
        sc = np.ones(p)
        sc[0] = 1.5
        xb["scale"] = sc
        B.sigma = 1.3
    elif name == "tail":
        xb["skew1"] = True
        B.noise = "skew"
    elif name == "correlated":
        xa["rho"] = xb["rho"] = 0.8
        xb["shift"] = shift(x0=0.6)
        B.beta = beta + np.array([0.0, 0.3, 0.0, 0.0, 0.0])
    elif name == "offsetting":
        xb["shift"] = shift(x0=0.4)
        B.a = 0.6
    elif name == "hetero":
        A.het = B.het = 0.4
        xb["shift"] = shift(x0=0.5)
    elif name == "nonlinear":
        nl = ("sq", "sin", "lin", "lin", "lin")
        A.basis = B.basis = nl
        xb["shift"] = shift(x0=0.5, x1=0.4)
        B.beta = beta + np.array([0.0, 0.0, 0.3, 0.0, 0.0])
    elif name == "entry_exit":
        xb["entrants"] = (0.2, shift(x0=0.8, x2=0.8))
        B.beta = beta + np.array([0.0, 0.25, 0.0, 0.0, 0.0])
    else:
        raise ValueError(name)
    return Scenario(name, p, A, B, xa, xb)
