"""ADOPT quickstart on neutral simulated data.

Each row has three predictors and a response that is a whole distribution (given by
its quantile function). x1 shifts the distribution (non-linearly), x2 widens it, x3 skews it.

    python examples/adopt_quickstart.py
"""

import numpy as np

from traceot import ADOPTRegressor, GlobalFrechetRegressor
from traceot.reference import quantile_grid

rng = np.random.default_rng(0)
n, q = 200, 50
u = quantile_grid(q)
X = rng.uniform(size=(n, 3))
base = np.log(u / (1 - u))  # logistic quantile function
Y = (
    np.sin(2 * np.pi * X[:, [0]])  # location (non-linear in x1)
    + (1.0 + X[:, [1]]) * base  # scale
    + 0.8 * X[:, [2]] * np.maximum(base, 0.0)  # upper tail only
    + 0.1 * rng.normal(size=(n, 1))  # row-level noise
)

X_train, X_test, Y_train, Y_test = X[:150], X[150:], Y[:150], Y[150:]
adopt = ADOPTRegressor().fit(X_train, Y_train)
gf = GlobalFrechetRegressor().fit(X_train, Y_train)
print(f"selected bandwidth fraction: {adopt.cv_results_['best_fraction']}")
print(f"held-out Frechet R^2  ADOPT: {adopt.score(X_test, Y_test):.3f}")
print(f"held-out Frechet R^2  GF   : {gf.score(X_test, Y_test):.3f}")

# How does feature 2 (the 'scale' feature) move the average distribution?
for x2 in (0.1, 0.5, 0.9):
    moved = adopt.transport(1, x2, adopt.mean_)
    iqr = np.interp(0.75, u, moved) - np.interp(0.25, u, moved)
    print(f"x2 = {x2:.1f}: interquartile range after T_2(x2) = {iqr:.2f}")
