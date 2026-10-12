# ruff: noqa: E501
"""Baseline T3v2, T3v2-null, T4v2 and T6v2 on the unchanged v1 fitter.

Settings are the constructor in docs/theory/PREREG_TAIL_v2.md §4. The bars in
that file are not edited here. JSON has no NaN.

    python benchmarks/trace_sims/run_v2_baseline.py
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks" / "results" / "trace_v2_baseline.json"
_NORM = NormalDist()

# Frozen constructor. ridge, learning_rate and min_count stay at the fitter defaults
# because the prereg does not override them.
FIT = dict(
    k_fit=8,
    k_report=4,
    n_u=0,
    max_bins=16,
    mono="none",
    max_rounds=200,
    tol=1e-10,
)

T3_SEEDS = [11, *range(2101, 2120)]
T3_NULL_SEEDS = list(range(3101, 3121))
T6_SEEDS = [6, *range(6101, 6120)]

# Secondary T3v2 bands (reported, not a pass condition) and T4v2 / T6v2 bands.
T3_BANDS = {
    0: (-5.6e-6, 4.07e-4),
    1: (-1.20e-4, 7.20e-4),
    2: (7.29e-4, 3.11e-3),
}
T4_BANDS = {
    0: (-2.02e-5, 5.86e-4),
    1: (1.68e-3, 4.83e-3),
    2: (4.53e-4, 3.05e-3),
}
T6_CH3 = (-3.49e-4, 3.75e-3)
T6_RATIO = 4.02e-2


def _z(u: np.ndarray) -> np.ndarray:
    return np.array([_NORM.inv_cdf(float(v)) for v in np.clip(u, 1e-12, 1.0 - 1e-12)])


def _s1(n: int, seed: int, p: int = 8, shape: float = 0.12):
    """Same draw as benchmarks/trace_sims/run_v1.py `_s1_period2`, with a shape multiplier."""
    rng = np.random.default_rng(seed)
    x = rng.random((n, p))
    z = _z(rng.random(n))
    he3 = z**3 - 3.0 * z
    y = 1.2 * x[:, 0] + (0.4 + 1.2 * x[:, 1]) * z + shape * x[:, 2] * he3
    return x, y


def _t6_draw(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.random((n, 1))
    z = _z(rng.random(n))
    y = (0.5 + x[:, 0]) * z
    return x, y


def _in_band(value: float, band: tuple[float, float]) -> bool:
    lo, hi = band
    return lo <= value <= hi


def _fit_energies(x, y, reporting: str) -> dict:
    from traceot import TraceV1

    t0 = time.perf_counter()
    model = TraceV1(**FIT, reporting=reporting).fit(x, y)
    elapsed = time.perf_counter() - t0
    p = x.shape[1]
    energies = [[float(model.channel_energy(j, m)) for m in range(4)] for j in range(p)]
    return {
        "energies": energies,
        "rounds": int(model.model_.rounds),
        "train_loss": float(model.model_.train_loss),
        "n_u": int(len(model.model_.u_nodes())),
        "seconds": elapsed,
        "hit_max_rounds": int(model.model_.rounds) >= int(FIT["max_rounds"]),
    }


def _ratio(shape: float, others: float):
    if others == 0.0:
        return None
    return shape / others


def _one(spec: dict) -> dict:
    kind = spec["kind"]
    seed = int(spec["seed"])
    if kind == "t6":
        x, y = _t6_draw(2000, seed)
        fitted = _fit_energies(x, y, "reference")
        e3 = fitted["energies"][0][3]
        e1 = fitted["energies"][0][1]
        ratio = _ratio(e3, e1)
        ok = _in_band(e3, T6_CH3) and ratio is not None and ratio <= T6_RATIO
        row = {
            "seed": seed,
            "channel3": e3,
            "channel1": e1,
            "ratio": ratio,
            "pass": ok,
        }
    else:
        shape = 0.0 if kind == "t3null" else 0.12
        x, y = _s1(16000, seed, shape=shape)
        fitted = _fit_energies(x, y, spec["reporting"])
        ch3 = [row[3] for row in fitted["energies"]]
        shape_e = ch3[2]
        others = max(ch3[j] for j in range(len(ch3)) if j != 2)
        ratio = _ratio(shape_e, others)
        separated = shape_e >= 3.0 * others
        bands = T3_BANDS if spec["reporting"] == "reference" else T4_BANDS
        in_bands = all(_in_band(ch3[j], bands[j]) for j in (0, 1, 2))
        row = {
            "seed": seed,
            "channel3": ch3,
            "shape": shape_e,
            "max_other": others,
            "ratio": ratio,
            "separated": separated,
            "in_bands": in_bands,
        }
    row.update(
        {
            "rounds": fitted["rounds"],
            "train_loss": fitted["train_loss"],
            "n_u": fitted["n_u"],
            "seconds": fitted["seconds"],
            "hit_max_rounds": fitted["hit_max_rounds"],
        }
    )
    print(
        f"{kind} seed {seed} reporting {spec['reporting']} rounds {row['rounds']} {row['seconds']:.1f}s",
        flush=True,
    )
    return {"kind": kind, "row": row}


def _finite(obj):
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError("non-finite result")
        return obj
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite(v) for v in obj]
    raise TypeError(type(obj))


def _summarise(rows: list[dict], kind: str) -> dict:
    if kind == "t6":
        n_pass = sum(bool(r["pass"]) for r in rows)
        return {
            "n": len(rows),
            "n_pass": n_pass,
            "pass": n_pass >= 19,
            "bar": ">= 19/20 seeds inside the channel-3 band and ratio <= 4.02e-2",
            "seeds": rows,
        }
    if kind == "t3null":
        n_fp = sum(bool(r["separated"]) for r in rows)
        return {
            "n": len(rows),
            "false_positives": n_fp,
            "pass": n_fp <= 1,
            "bar": "false positives <= 1 of 20",
            "seeds": rows,
        }
    if kind == "t3":
        n_sep = sum(bool(r["separated"]) for r in rows)
        n_band = sum(bool(r["in_bands"]) for r in rows)
        return {
            "n": len(rows),
            "n_separated": n_sep,
            "pass": n_sep >= 9,
            "bar": ">= 9 of 20 seeds with E(shape,3) >= 3 * max other",
            "secondary_in_bands": n_band,
            "secondary_bar": ">= 18/20 inside MC mean ± 3 SD; reported, not a pass condition",
            "seeds": rows,
        }
    n_band = sum(bool(r["in_bands"]) for r in rows)
    return {
        "n": len(rows),
        "n_in_bands": n_band,
        "pass": n_band >= 18,
        "bar": ">= 18/20 seeds inside the Legendre channel-3 bands",
        "oracle_separation_ratio": 0.497,
        "note": "3× separation is impossible in Legendre: the oracle ratio is 0.497",
        "seeds": rows,
    }


def main() -> None:
    specs = []
    for seed in T3_SEEDS:
        specs.append({"kind": "t3", "seed": seed, "reporting": "reference"})
    for seed in T3_NULL_SEEDS:
        specs.append({"kind": "t3null", "seed": seed, "reporting": "reference"})
    for seed in T3_SEEDS:
        specs.append({"kind": "t4", "seed": seed, "reporting": "legendre"})
    for seed in T6_SEEDS:
        specs.append({"kind": "t6", "seed": seed, "reporting": "reference"})
    grouped: dict[str, list[dict]] = {k: [] for k in ("t3", "t3null", "t4", "t6")}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_one, spec) for spec in specs]
        done = 0
        for fut in as_completed(futures):
            item = fut.result()
            grouped[item["kind"]].append(item["row"])
            done += 1
            payload = {
                "fitter": "TraceV1 unchanged",
                "settings": {
                    **FIT,
                    "reporting_per_test": {
                        "t3": "reference",
                        "t3null": "reference",
                        "t4": "legendre",
                        "t6": "reference",
                    },
                },
                "completed": done,
                "of": len(specs),
                "seconds": time.perf_counter() - t0,
            }
            for kind, rows in grouped.items():
                if rows:
                    payload[kind] = _summarise(rows, kind)
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(_finite(payload), indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
