"""H8 joint 5% test and the exact one-sided Clopper-Pearson bar (Amendment 1, A3)."""
from __future__ import annotations

import numpy as np
from scipy.stats import beta

N_NULL, B_NULL, CP_LEVEL, CP_BAR = 200, 100, 0.95, 0.09


def joint_reject(theta_hat, boot, level=0.95):
    """One max-|t| test of theta = 0 for the whole vector.

    theta_hat: (d,) point estimate; boot: (B, d) bootstrap replicates.
    Reject iff max_i |theta_hat_i| / sd*_i > q_level( max_i |theta*_i - theta_hat_i| / sd*_i ).
    Entries with zero bootstrap sd are dropped (reported by the caller)."""
    theta_hat = np.asarray(theta_hat, float)
    boot = np.asarray(boot, float)
    sd = boot.std(0, ddof=1)
    keep = sd > 0
    if not keep.any():
        return False, np.nan, np.nan
    t = np.abs(theta_hat[keep]) / sd[keep]
    tstar = np.max(np.abs(boot[:, keep] - theta_hat[keep]) / sd[keep], axis=1)
    c = float(np.quantile(tstar, level))
    return bool(t.max() > c), float(t.max()), c


def cp_upper_one_sided(x, n, level=CP_LEVEL):
    """Exact one-sided Clopper-Pearson upper bound."""
    return 1.0 if x >= n else float(beta.ppf(level, x + 1, n - x))


def h8a_pass(x, n=N_NULL):
    """Pass iff the one-sided 95% CP upper bound <= 0.09 (equivalently x <= 11 at n = 200)."""
    return cp_upper_one_sided(x, n) <= CP_BAR


if __name__ == "__main__":
    assert max(x for x in range(N_NULL) if h8a_pass(x)) == 11
    rng = np.random.default_rng(0)
    rej = 0
    for r in range(2000):  # calibration smoke test on a known null: Gaussian mean vector
        d = rng.standard_normal((400, 2)) @ np.array([[1, -0.75], [0, 0.66]])
        th = d.mean(0)
        bs = np.array([d[rng.integers(0, 400, 400)].mean(0) for _ in range(B_NULL)])
        rej += joint_reject(th, bs)[0]
    print("max x passing:", 11, "| joint-test null rejection on toy null:", rej / 2000)
