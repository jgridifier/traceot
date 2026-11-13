# ruff: noqa: E501
"""Curved-mean shape scenarios for the v5.2 specified-family gate (PREREG_V5_2_SPECIFIED §3).

Imports the frozen H7 generator ``sb_dgp_shape`` and does not modify it. Only the mean
changes:

    mu_kappa(X) = 1 + X[:, :5] @ BETA + kappa * g(X),
    g(X) = (X_3^2 - 1)/sqrt(2) + sqrt(2) * sin(X_5)
         (X_3 = code column 2, X_5 = code column 4; both unshifted in skew_comp / tail_comp)

One curvature level, pre-stated: kappa = 1. Noise laws are unchanged (``SPEC``). sigma_B
is re-calibrated per scenario with the frozen procedure (oracle 2e6 draws, oracle seed 99,
common random numbers) so lambda_2(Y_B) = lambda_2(Y_A); if no sigma_B in the frozen
bracket solves it, the entry is marked infeasible and the cell is INCOMPLETE. Truth
(Shapley split, graded channel) uses the same oracle code and is written to
``benchmarks/results/shiftbench_truth_shape_v5_2_specified.json``.

    python sb_dgp_shape_v5_2_specified.py      # builds the truth file (oracle seed only)

This module does not touch gate indices 11000-11049 or 11100-11119, nor sealed 9600-9619,
nor reserved 10400-10799.
"""

from __future__ import annotations

import json
import os

import numpy as np
import sb_dgp as dgp
import sb_dgp_shape as shp

KAPPA = 1.0
# (scenario, kappa) pairs used by the four gate cells (PREREG_V5_2_SPECIFIED §4)
TRUTH_KEYS = (
    ("skew_comp", 1.0),
    ("tail_comp", 1.0),
)
TRUTH_FILE = os.path.join(
    os.path.dirname(__file__), "..", "results", "shiftbench_truth_shape_v5_2_specified.json"
)
SIGMA_BRACKET = (0.3, 3.0)  # the frozen bracket of sb_dgp_shape.ShapeScenario.calibrate


def key(name: str, kappa: float = KAPPA) -> str:
    """Scenario key with kappa, e.g. ``skew_comp|k1`` (also the ``seed_of`` key)."""
    return f"{name}|k{kappa:g}"


def g(X):
    """g(X) = (X_3^2 - 1)/√2 + √2 sin(X_5); X_3 = col 2, X_5 = col 4."""
    return (X[:, 2] ** 2 - 1.0) / np.sqrt(2.0) + np.sqrt(2.0) * np.sin(X[:, 4])


class SpecifiedShapeScenario(shp.ShapeScenario):
    def __init__(self, name: str, kappa: float = KAPPA, p: int = 5, sigma_b: float | None = None):
        if kappa != KAPPA:
            raise ValueError(kappa)
        self.kappa = float(kappa)
        if sigma_b is None:
            sigma_b = load_sigma(name, kappa)
        super().__init__(name, p, sigma_b=sigma_b)

    def mu(self, X):  # overrides the frozen linear mean (instance method on purpose)
        return 1.0 + X[:, : dgp.N_ACTIVE] @ shp.BETA + self.kappa * g(X)

    def feasibility(self, n=shp.ORACLE_N, seed=shp.ORACLE_SEED):
        """lambda_2(Y_B) - lambda_2(Y_A) at sigma_B -> 0+ and at the bracket ends.

        Feasible iff the frozen bracket has a sign change."""
        XA, XB, eA, eB, _, _ = self._oracle_draws(n, seed)
        target = shp.lmoments(self.mu(XA) + eA)[1]
        mB = self.mu(XB)
        f = {s: float(shp.lmoments(mB + s * eB)[1] - target) for s in (1e-9, *SIGMA_BRACKET)}
        lo, hi = f[SIGMA_BRACKET[0]], f[SIGMA_BRACKET[1]]
        return (lo > 0) != (hi > 0), f


def load_sigma(name, kappa=KAPPA):
    if os.path.exists(TRUTH_FILE):
        with open(TRUTH_FILE) as f:
            ent = json.load(f).get(key(name, kappa))
        if ent is not None:
            return ent["sigma_b"]
    return None


def make(name, kappa=KAPPA, p=5):
    sb = load_sigma(name, kappa)
    if sb is None:
        raise RuntimeError(f"no calibrated sigma_B for {key(name, kappa)} (INCOMPLETE)")
    return SpecifiedShapeScenario(name, kappa, p, sigma_b=sb)


def build_truth():
    res = {}
    for name, kappa in TRUTH_KEYS:
        sc = SpecifiedShapeScenario(name, kappa, 5, sigma_b=1.0)
        k = key(name, kappa)
        ok, fvals = sc.feasibility()
        if not ok:
            res[k] = {
                "feasible": False,
                "sigma_b": None,
                "scen": name,
                "kappa": kappa,
                "dlam2_at_sigma": {f"{s:g}": v for s, v in fvals.items()},
            }
            print(k, "calibration infeasible -> INCOMPLETE")
            continue
        sc.calibrate()
        t = sc.truth()
        se1 = np.sqrt(2 * t["sd_A"] ** 2 / shp.ORACLE_N)
        t["dlam1_se"] = float(se1)
        t["kappa"] = kappa
        t["scen"] = name
        t["feasible"] = True
        assert abs(t["dlam1"]) <= 4 * se1 and abs(t["dlam2"]) <= 2e-3, (
            k,
            t["dlam1"],
            se1,
            t["dlam2"],
        )
        res[k] = t
        gk = t["graded"][0]
        print(
            k,
            f"sigma_B={sc.sigma_b:.6f} dlam1={t['dlam1']:+.2e} dlam2={t['dlam2']:+.2e}",
            f"graded lambda{gk + 1}: comp {t['shapley']['comp'][gk]:+.5f} struct {t['shapley']['struct'][gk]:+.5f}",
            flush=True,
        )
    with open(TRUTH_FILE, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    build_truth()
