# ruff: noqa: E501
"""T3v2, T3v2-null, T4v2 and T6v2 after each v2 fitter change.

The bars are the frozen ones in docs/theory/PREREG_TAIL_v2.md. They are not
edited here. The step-1 baseline on the unchanged TraceV1 fitter stays in
benchmarks/results/trace_v2_baseline.json.

Three fitters, each written down before its run:
- hybrid: TraceHybrid. Residual channels use the prereg constructor
  (k_fit=8, k_report=4, n_u=0, max_bins=16, mono="none", max_rounds=200,
  tol=1e-10). The mean channel uses the item-1 booster defaults.
- smooth: TraceV1 with that same constructor, plus the item-2 curvature
  (smooth=0.05, levenberg=1e-3).
- ridge: TraceV1 with that same constructor, plus the middle of the item-3
  grid (sobolev_s=1, sobolev_lambda=1e-3, tail_gamma=1). The middle is fixed
  here so the tail run does not wait on, or copy, a grid point that won on
  the CRPS cases.

    python benchmarks/trace_sims/run_v2_tail_after.py hybrid
    python benchmarks/trace_sims/run_v2_tail_after.py smooth
    python benchmarks/trace_sims/run_v2_tail_after.py ridge
"""

from __future__ import annotations

import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_NORM = NormalDist()
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
T3_BANDS = {0: (-5.6e-6, 4.07e-4), 1: (-1.20e-4, 7.20e-4), 2: (7.29e-4, 3.11e-3)}
T4_BANDS = {0: (-2.02e-5, 5.86e-4), 1: (1.68e-3, 4.83e-3), 2: (4.53e-4, 3.05e-3)}
T6_CH3 = (-3.49e-4, 3.75e-3)
T6_RATIO = 4.02e-2


def _z(u: np.ndarray) -> np.ndarray:
    return np.array([_NORM.inv_cdf(float(v)) for v in np.clip(u, 1e-12, 1.0 - 1e-12)])


def _s1(n: int, seed: int, p: int = 8, shape: float = 0.12):
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


def _fit(x, y, reporting: str, mode: str) -> dict:
    from traceot import TraceHybrid, TraceV1

    t0 = time.perf_counter()
    if mode == "hybrid":
        model = TraceHybrid(**FIT, reporting=reporting).fit(x, y)
        rounds = int(model.model_.rounds)
        n_u = int(len(model.model_.u_nodes))
        loss = None
    elif mode == "smooth":
        model = TraceV1(**FIT, reporting=reporting, smooth=0.05, levenberg=1e-3).fit(x, y)
        rounds = int(model.model_.rounds)
        n_u = int(len(model.model_.u_nodes()))
        loss = float(model.model_.train_loss)
    elif mode == "ridge":
        model = TraceV1(
            **FIT, reporting=reporting, sobolev_s=1.0, sobolev_lambda=1e-3, tail_gamma=1.0
        ).fit(x, y)
        rounds = int(model.model_.rounds)
        n_u = int(len(model.model_.u_nodes()))
        loss = float(model.model_.train_loss)
    else:
        raise ValueError(mode)
    elapsed = time.perf_counter() - t0
    p = x.shape[1]
    energies = [[float(model.channel_energy(j, m)) for m in range(4)] for j in range(p)]
    return {
        "energies": energies,
        "rounds": rounds,
        "train_loss": loss,
        "n_u": n_u,
        "seconds": elapsed,
        "hit_max_rounds": rounds >= int(FIT["max_rounds"]),
    }


def _ratio(shape: float, others: float):
    if others == 0.0:
        return None
    return shape / others


def _one(spec: dict) -> dict:
    kind = spec["kind"]
    seed = int(spec["seed"])
    mode = spec["mode"]
    if kind == "t6":
        x, y = _t6_draw(2000, seed)
        fitted = _fit(x, y, "reference", mode)
        e3 = fitted["energies"][0][3]
        e1 = fitted["energies"][0][1]
        ratio = _ratio(e3, e1)
        ok = _in_band(e3, T6_CH3) and ratio is not None and ratio <= T6_RATIO
        row = {"seed": seed, "channel3": e3, "channel1": e1, "ratio": ratio, "pass": ok}
    else:
        shape = 0.0 if kind == "t3null" else 0.12
        x, y = _s1(16000, seed, shape=shape)
        fitted = _fit(x, y, spec["reporting"], mode)
        ch3 = [r[3] for r in fitted["energies"]]
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
    print(f"{mode} {kind} seed {seed} rounds {row['rounds']} {row['seconds']:.1f}s", flush=True)
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
    mode = sys.argv[1]
    if mode not in {"hybrid", "smooth", "ridge"}:
        raise SystemExit("mode must be hybrid, smooth or ridge")
    out = ROOT / "benchmarks" / "results" / f"trace_v2_tail_{mode}.json"
    specs = []
    for seed in T3_SEEDS:
        specs.append({"kind": "t3", "seed": seed, "reporting": "reference", "mode": mode})
    for seed in T3_NULL_SEEDS:
        specs.append({"kind": "t3null", "seed": seed, "reporting": "reference", "mode": mode})
    for seed in T3_SEEDS:
        specs.append({"kind": "t4", "seed": seed, "reporting": "legendre", "mode": mode})
    for seed in T6_SEEDS:
        specs.append({"kind": "t6", "seed": seed, "reporting": "reference", "mode": mode})
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
                "fitter": mode,
                "completed": done,
                "of": len(specs),
                "seconds": time.perf_counter() - t0,
            }
            for kind, rows in grouped.items():
                if rows:
                    payload[kind] = _summarise(rows, kind)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(_finite(payload), indent=2, allow_nan=False) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
