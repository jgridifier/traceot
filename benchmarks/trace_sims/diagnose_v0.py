"""Per-round pinball of the v0 fitter. Diagnosis only; not a reported cell.

``benchmarks/results/trace_diagnosis.json`` is that v0 log. Re-running this
script on the current fitter would overwrite it, so the write is refused
unless ``TRACE_DIAG_FORCE=1`` is set.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from traceot import TraceRegressor

OUT = Path(__file__).resolve().parents[2] / "benchmarks" / "results" / "trace_diagnosis.json"


def _at(q, u, level):
    u = np.asarray(u, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1:
        return float(np.interp(level, u, q))
    hi = int(np.searchsorted(u, level, side="left"))
    hi = min(max(hi, 1), u.shape[0] - 1)
    lo = hi - 1
    span = float(u[hi] - u[lo])
    t = 0.0 if span == 0.0 else (level - float(u[lo])) / span
    return q[:, lo] * (1.0 - t) + q[:, hi] * t


def _spread(q, u, lo=0.1, hi=0.9):
    return _at(q, u, hi) - _at(q, u, lo)


def _case(name, x, y, kw, hold=0.3):
    n = int(hold * len(y))
    # Last `hold` is untouched by the fit, so the spread comparison is out of sample.
    cut = len(y) - n
    model = TraceRegressor(**kw).fit(x[:cut], y[:cut])
    m = model.model_
    q = model.predict(x[cut:])
    u = np.asarray(m.u)
    q0 = np.asarray(m.q0)
    spread_hat = _spread(q, u)
    spread_q0 = float(_spread(q0, u))
    med = _at(q, u, 0.5)
    return {
        "name": name,
        "n_train": int(cut),
        "n_test": int(n),
        "kw": {k: (float(v) if isinstance(v, float) else int(v)) for k, v in kw.items()},
        "bag_rounds": [int(v) for v in np.asarray(m.bag_rounds)],
        "round_train": [float(v) for v in np.asarray(m.round_train)],
        "round_val": [float(v) for v in np.asarray(m.round_val)],
        "round_step": [float(v) for v in np.asarray(m.round_step)],
        "round_center_kept": [float(v) for v in np.asarray(m.round_center_kept)],
        "learning_rate": float(kw["learning_rate"]),
        "saturated_location_step": float(kw["learning_rate"]) / 2.0,
        "mean_abs_location_step": (
            float(np.mean(np.asarray(m.round_step))) if len(m.round_step) else None
        ),
        "mean_center_kept": (
            float(np.mean(np.asarray(m.round_center_kept))) if len(m.round_center_kept) else None
        ),
        "q0_spread_10_90": spread_q0,
        "pred_spread_10_90_mean": float(np.mean(spread_hat)),
        "pred_spread_over_q0": float(np.mean(spread_hat) / spread_q0) if spread_q0 else None,
        "mse_median_test": float(np.mean((med - y[cut:]) ** 2)),
        "gains": [float(v) for v in np.asarray(m.gains)],
        "selected": [bool(v) for v in np.asarray(m.selected)],
        "train_loss": float(m.train_loss),
        "val_loss": float(m.val_loss),
    }


def main():
    rows = []
    rng = np.random.default_rng(0)
    n, p = 20_000, 10
    x4 = rng.random((n, p))
    y4 = x4[:, 0] + 0.1 * rng.normal(size=n)
    base = dict(
        n_channels=3,
        max_bins=32,
        learning_rate=0.35,
        u_levels=19,
        n_bags=1,
        max_interactions=0,
        seed=0,
        gain_threshold=-1.0,
    )
    rows.append(
        _case(
            "n2e4_v0_comparison_3rounds",
            x4,
            y4,
            {**base, "max_rounds": 3, "patience": 0, "val_frac": 0.0},
        )
    )
    rows.append(
        _case(
            "n2e4_path_patience0",
            x4,
            y4,
            {**base, "max_rounds": 60, "patience": 0, "val_frac": 0.2},
        )
    )
    rows.append(
        _case(
            "n2e4_defaults_patience8_lr0.3",
            x4,
            y4,
            {
                **base,
                "max_bins": 16,
                "learning_rate": 0.3,
                "max_rounds": 40,
                "patience": 8,
                "val_frac": 0.2,
                "gain_threshold": 0.0,
            },
        )
    )
    rng = np.random.default_rng(44)
    log_units = rng.uniform(0, 2, size=4000)
    unit_price = rng.uniform(0.5, 5, size=4000)
    deliveries = rng.integers(1, 12, size=4000).astype(np.float64)
    distance = rng.uniform(0, 1, size=4000)
    quality = rng.uniform(0, 1, size=4000)
    season = rng.uniform(0, 1, size=4000)
    noise = rng.normal(size=(4000, 3))
    xv = np.column_stack(
        [log_units, unit_price, deliveries, distance, quality, season, noise]
    )
    from statistics import NormalDist

    z = np.array([NormalDist().inv_cdf(float(v)) for v in rng.random(4000)])
    he3 = z**3 - 3 * z
    yv = 10 + 4 * log_units + 2 * unit_price + (0.8 + 0.5 * log_units) * z + 0.05 * quality * he3
    rows.append(
        _case(
            "vendor_v0_comparison",
            xv,
            yv,
            dict(
                n_channels=3,
                max_bins=16,
                learning_rate=0.35,
                max_rounds=40,
                patience=8,
                val_frac=0.2,
                n_bags=1,
                max_interactions=0,
                u_levels=19,
                seed=0,
            ),
        )
    )
    rows.append(
        _case(
            "vendor_path_patience0",
            xv,
            yv,
            dict(
                n_channels=3,
                max_bins=16,
                learning_rate=0.35,
                max_rounds=80,
                patience=0,
                val_frac=0.2,
                n_bags=1,
                max_interactions=0,
                u_levels=19,
                seed=0,
                gain_threshold=-1.0,
            ),
        )
    )
    rng = np.random.default_rng(11)
    xs = rng.uniform(size=(2000, 8))
    z = np.array([NormalDist().inv_cdf(float(v)) for v in rng.random(2000)])
    ys = 1.2 * xs[:, 0] + (0.4 + 1.2 * xs[:, 1]) * z + 0.12 * xs[:, 2] * (z**3 - 3 * z)
    rows.append(
        _case(
            "s1_period2_locked",
            xs,
            ys,
            dict(
                max_bins=16,
                u_levels=31,
                n_channels=3,
                learning_rate=0.4,
                max_rounds=80,
                patience=12,
                val_frac=0.2,
                n_bags=1,
                max_interactions=0,
                min_count=5,
                seed=11,
            ),
        )
    )
    if OUT.exists() and os.environ.get("TRACE_DIAG_FORCE") != "1":
        raise SystemExit(
            f"{OUT} is the v0 diagnosis log. Set TRACE_DIAG_FORCE=1 to overwrite it."
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2))
    for row in rows:
        print(
            row["name"],
            "rounds",
            row["bag_rounds"],
            "step",
            None
            if row["mean_abs_location_step"] is None
            else round(row["mean_abs_location_step"], 5),
            "kept",
            None if row["mean_center_kept"] is None else round(row["mean_center_kept"], 4),
            "spread_ratio",
            None if row["pred_spread_over_q0"] is None else round(row["pred_spread_over_q0"], 4),
            "mse",
            round(row["mse_median_test"], 5),
            "val0",
            round(row["round_val"][0], 5) if row["round_val"] else None,
            "val_last",
            round(row["round_val"][-1], 5) if row["round_val"] else None,
            "selected",
            row["selected"],
        )


if __name__ == "__main__":
    main()
