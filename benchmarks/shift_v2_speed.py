"""Wall time and peak memory of the shift decomposition at 1e6 rows by 100 features.

The fits are built on a small draw and are not part of the timed section. The
timed section is the decomposition with those models already in hand.

    python benchmarks/shift_v2_speed.py
"""

from __future__ import annotations

import json
import os
import resource
import time
from pathlib import Path

import numpy as np

from traceot import TraceV1, shift

N = 1_000_000
P = 100
FIT_N = 4_000
OUT = Path(
    os.environ.get(
        "SHIFT_SPEED_OUT", Path(__file__).resolve().parent / "results" / "shift_v2_speed.json"
    )
)


def rss_mb() -> float:
    # Linux reports ru_maxrss in kilobytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    rng = np.random.default_rng(123)
    x_fit = rng.normal(size=(FIT_N, P))
    y_fit = np.tanh(x_fit[:, 0]) + 0.1 * rng.normal(size=FIT_N)
    t0 = time.perf_counter()
    model = TraceV1(
        k_fit=6,
        k_report=6,
        max_bins=16,
        max_rounds=4,
        tol=1e-5,
        mono="none",
    ).fit(x_fit, y_fit)
    fit_seconds = time.perf_counter() - t0
    del x_fit, y_fit

    x_a = rng.normal(size=(N, P))
    x_b = x_a + 0.05
    y_a = np.tanh(x_a[:, 0]) + 0.1 * rng.normal(size=N)
    y_b = np.tanh(x_b[:, 0]) + 0.1 * rng.normal(size=N)
    rss_after_data = rss_mb()

    t0 = time.perf_counter()
    bary = shift.barycenter(model, model, x_a, x_b)
    bary_seconds = time.perf_counter() - t0
    rss_after_bary = rss_mb()

    t0 = time.perf_counter()
    report = shift.explain(
        model,
        model,
        x_a,
        y_a,
        x_b,
        y_b,
        n_quad=64,
        n_folds=5,
        trim_quantile=0.995,
        seed=1,
    )
    explain_seconds = time.perf_counter() - t0
    rss_after_explain = rss_mb()

    out = {
        "rows_per_period": N,
        "features": P,
        "fit_rows": FIT_N,
        "fit_seconds_not_in_decomposition": fit_seconds,
        "barycenter_seconds": bary_seconds,
        "explain_seconds": explain_seconds,
        "barycenter_w2": bary["w2_total"],
        "explain_order_gap": report["order_gap"],
        "rss_mb_after_data": rss_after_data,
        "rss_mb_after_barycenter": rss_after_bary,
        "rss_mb_after_explain": rss_after_explain,
        "peak_rss_mb": rss_mb(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
