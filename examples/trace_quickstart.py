"""TRACE quickstart on a neutral simulated draw.

One feature shifts the outcome, one widens it, one changes the tail. The rest is noise.

    python examples/trace_quickstart.py
"""

from statistics import NormalDist

import numpy as np

from traceot import TraceRegressor

rng = np.random.default_rng(0)
n = 800
x = rng.uniform(size=(n, 4))
u = np.clip(rng.random(n), 1e-6, 1 - 1e-6)
z = np.array([NormalDist().inv_cdf(float(v)) for v in u])
y = 1.5 * x[:, 0] + (0.4 + x[:, 1]) * z + 0.1 * x[:, 2] * (z**3 - 3 * z)

model = TraceRegressor(max_rounds=40, max_bins=12, u_levels=19, seed=0).fit(x[:600], y[:600])
held = model.score(x[600:], y[600:])
print(f"held-out pinball {held['pinball']:.4f}  CRPS approx {held['crps']:.4f}")
print(f"80% coverage {held['cover80']:.2f}  90% coverage {held['cover90']:.2f}")
for j, name in enumerate(("location feature", "scale feature", "shape feature", "noise")):
    coef, _std = model.effect(j)
    spread = coef[1:].max(axis=0) - coef[1:].min(axis=0)
    print(f"{name}: channel ranges loc/scale/shape {spread[0]:.3f} {spread[1]:.3f} {spread[2]:.3f}")
