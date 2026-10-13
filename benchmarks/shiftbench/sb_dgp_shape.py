# ruff: noqa: E501, E731, B007, B023, F841
"""H7 shape scenarios (PREREG_TRACE_CLAIMS_AMENDMENT_1.md, A2): mean AND L-scale fixed.

Imports the frozen sb_dgp and does not modify it. Each scenario changes only the shape of
the outcome law between periods; period-B noise scale sigma_B is calibrated on the oracle
sample (2e6 draws, seed 99, common random numbers) so that lambda_2(Y_B) = lambda_2(Y_A).

Truth: L-moments lambda_1..lambda_4 (channels k=0..3) of Y_A, Y_B, Y_c=(X_B, structure A),
Y_c'=(X_A, structure B); the order-free Shapley split (TRACE estimand) and the A-reference
split (rival estimand); plus the frozen harness statistics (A-reference) for completeness.
"""

from __future__ import annotations

import json
import os

import numpy as np
import sb_dgp as dgp

SHAPE_SCENARIOS = ("skew_struct", "tail_struct", "skew_comp", "tail_comp", "shape_mixed")
GRADED = {
    "skew_struct": (2,),
    "tail_struct": (3,),
    "skew_comp": (2,),
    "tail_comp": (3,),
    "shape_mixed": (2,),
}
BETA = np.array([1.0, -0.8, 0.6, 0.4, -0.3])
ORACLE_N, ORACLE_SEED = 2_000_000, 99
TRUTH_FILE = os.path.join(os.path.dirname(__file__), "..", "results", "shiftbench_truth_shape.json")


def _t5(rng, n):
    return rng.standard_t(5, n) * np.sqrt(3 / 5)


def noise(kind, rng, n):
    if kind == "normal":
        return rng.standard_normal(n)
    if kind == "skew":
        return dgp._skew_noise(rng, n)
    if kind == "skewneg":
        return -dgp._skew_noise(rng, n)
    if kind == "t5":
        return _t5(rng, n)
    raise ValueError(kind)


def feature(kind, rng, n):
    if kind == "chi3":
        return dgp._skew_feature(rng, n)
    if kind == "t5":
        return _t5(rng, n)
    raise ValueError(kind)


SPEC = {  # name: (B feature replacements {col: law}, B noise law)
    "skew_struct": ({}, "skew"),
    "tail_struct": ({}, "t5"),
    "skew_comp": ({0: "chi3"}, "normal"),
    "tail_comp": ({1: "t5"}, "normal"),
    "shape_mixed": ({0: "chi3"}, "skewneg"),
}


def lmoments(y):
    """Unbiased sample L-moments lambda_1..lambda_4 (Hosking 1990)."""
    y = np.sort(np.asarray(y, float))
    n = len(y)
    i = np.arange(n, dtype=float)
    b0 = y.mean()
    b1 = np.sum(i / (n - 1) * y) / n
    b2 = np.sum(i * (i - 1) / ((n - 1) * (n - 2)) * y) / n
    b3 = np.sum(i * (i - 1) * (i - 2) / ((n - 1) * (n - 2) * (n - 3)) * y) / n
    return np.array([b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0, 20 * b3 - 30 * b2 + 12 * b1 - b0])


class ShapeScenario:
    def __init__(self, name: str, p: int = 5, sigma_b: float | None = None):
        self.name, self.p = name, p
        self.feat_b, self.noise_b = SPEC[name]
        self.sigma_b = sigma_b if sigma_b is not None else load_sigma(name)

    # X ------------------------------------------------------------------
    def draw_x(self, which, n, rng):
        Z = rng.standard_normal((n, self.p))
        if which == "B":
            for j, law in self.feat_b.items():
                Z[:, j] = feature(law, rng, n)
        return Z

    @staticmethod
    def mu(X):
        return 1.0 + X[:, : dgp.N_ACTIVE] @ BETA

    def y_struct(self, which, X, e_std):
        """e_std: draws already from the period's noise law (mean 0)."""
        s = 1.0 if which == "A" else self.sigma_b
        return self.mu(X) + s * e_std

    def sample(self, n, rng):
        XA = self.draw_x("A", n, rng)
        XB = self.draw_x("B", n, rng)
        yA = self.y_struct("A", XA, noise("normal", rng, n))
        yB = self.y_struct("B", XB, noise(self.noise_b, rng, n))
        return XA, yA, XB, yB

    # oracle ---------------------------------------------------------------
    def _oracle_draws(self, n=ORACLE_N, seed=ORACLE_SEED):
        rng = np.random.default_rng(seed)
        XA, XB = self.draw_x("A", n, rng), self.draw_x("B", n, rng)
        eA, eB = noise("normal", rng, n), noise(self.noise_b, rng, n)
        eA2, eB2 = noise("normal", rng, n), noise(self.noise_b, rng, n)
        return XA, XB, eA, eB, eA2, eB2

    def calibrate(self, n=ORACLE_N, seed=ORACLE_SEED, tol=1e-6):
        XA, XB, eA, eB, _, _ = self._oracle_draws(n, seed)
        target = lmoments(self.mu(XA) + eA)[1]
        mB = self.mu(XB)
        lo, hi = 0.3, 3.0
        f = lambda s: lmoments(mB + s * eB)[1] - target
        flo = f(lo)
        while hi - lo > tol * lo:
            mid = 0.5 * (lo + hi)
            fm = f(mid)
            if (fm > 0) == (flo > 0):
                lo, flo = mid, fm
            else:
                hi = mid
        self.sigma_b = 0.5 * (lo + hi)
        return self.sigma_b

    def truth(self, n=ORACLE_N, seed=ORACLE_SEED):
        XA, XB, eA, eB, eA2, eB2 = self._oracle_draws(n, seed)
        Y = {
            "A": self.y_struct("A", XA, eA),
            "B": self.y_struct("B", XB, eB),
            "C": self.y_struct("A", XB, eA2),  # covariates B, structure A
            "Cp": self.y_struct("B", XA, eB2),  # covariates A, structure B
        }
        L = {k: lmoments(v) for k, v in Y.items()}
        out = {
            "sigma_b": self.sigma_b,
            "sd_A": float(np.std(Y["A"])),
            "graded": list(GRADED[self.name]),
            "lmom": {k: v.tolist() for k, v in L.items()},
            "dlam1": float(L["B"][0] - L["A"][0]),
            "dlam2": float(L["B"][1] - L["A"][1]),
        }
        sh_c = 0.5 * ((L["B"] - L["Cp"]) + (L["C"] - L["A"]))
        sh_s = 0.5 * ((L["Cp"] - L["A"]) + (L["B"] - L["C"]))
        out["shapley"] = {"comp": sh_c.tolist(), "struct": sh_s.tolist()}
        out["aref"] = {"comp": (L["C"] - L["A"]).tolist(), "struct": (L["B"] - L["C"]).tolist()}
        w = 1 / (2 * np.arange(4) + 1)
        dc = (2 * np.arange(4) + 1) * (L["B"] - L["A"])
        out["w2_share"] = {
            "comp": (w * (2 * np.arange(4) + 1) * sh_c * dc).tolist(),
            "struct": (w * (2 * np.arange(4) + 1) * sh_s * dc).tolist(),
        }
        for s in dgp.STATS:
            tA, tB, tC = (dgp.stat(Y[k], s) for k in ("A", "B", "C"))
            out[s] = {"total": tB - tA, "comp": tC - tA, "struct": tB - tC}
        return out


def load_sigma(name):
    if os.path.exists(TRUTH_FILE):
        with open(TRUTH_FILE) as f:
            return json.load(f)[name]["sigma_b"]
    return None


def make(name, p=5):
    return ShapeScenario(name, p)


def build_truth():
    """Calibrate sigma_B and write the oracle truth. Fails loudly if |dlam1| or |dlam2| > 2e-3."""
    res = {}
    for name in SHAPE_SCENARIOS:
        sc = ShapeScenario(name, 5, sigma_b=1.0)
        sc.calibrate()
        t = sc.truth()
        # lambda_1 is equal by construction (every law has mean 0); the Monte Carlo check uses
        # 4 standard errors (Amendment 1, deviation D1); lambda_2 is matched by calibration.
        se1 = np.sqrt(2 * t["sd_A"] ** 2 / ORACLE_N)
        t["dlam1_se"] = float(se1)
        assert abs(t["dlam1"]) <= 4 * se1 and abs(t["dlam2"]) <= 2e-3, (
            name,
            t["dlam1"],
            se1,
            t["dlam2"],
        )
        res[name] = t
        print(
            name,
            f"sigma_B={sc.sigma_b:.6f} dlam1={t['dlam1']:+.2e} dlam2={t['dlam2']:+.2e}",
            "graded",
            GRADED[name],
            "shapley comp",
            np.round(t["shapley"]["comp"], 4),
            "struct",
            np.round(t["shapley"]["struct"], 4),
        )
    with open(TRUTH_FILE, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    build_truth()
