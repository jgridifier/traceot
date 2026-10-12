# ruff: noqa: E501
"""Landscape item 2, scored before any result is used to change a bar.

Protocol, fixed from docs/research/LANDSCAPE.md §4 #2 before this run:
- Candidate: TraceV1, k_fit=8, k_report=4, n_u=0 (12 Gauss–Legendre nodes), max_bins=16, learning_rate=1.0, max_rounds=40, tol=1e-8, mono="none".
- Smoothed pinball density with bandwidth 0.05, annealed by the fitter to 0.02 then 0.01. Levenberg damping 1e-3. These two numbers are not tuned on the cases below.
- Baseline: the same constructor with smooth=0 and levenberg=0. That is the current boosting fit at learning rate 1.0 and 40 rounds.
- Leakage: y = (0.5 + x) z, n=2000, p=1, reporting="hermite", channel-3 energy divided by the recorded v0 He3 energy 0.0018508010614343975 (benchmarks/results/trace_v1.json, the denominator of the v1 ratio 0.496). Seed 6 is the historical T6 seed. Fresh seeds 83001..83020 are the paired sample.
- Leakage pass: the seed-6 ratio ≤ 0.10, the fresh-seed mean ratio ≤ 0.10, and a one-sided paired t of (fresh ratio − 0.10) has p < 0.05.
- CRPS: S1 period 2 (n=2000, p=8), vendor spend (n=4000, p=9), and n=2e4 p=10. Fresh seeds 81001..81020. First 70% train, last 30% test. Both models fit on the whole train block. CRPS is the Gauss quadrature on the shared 12 nodes.
- CRPS pass: seed-mean candidate CRPS ≤ 1.005 × baseline on every case, and seed-mean CRPS is lower on ≥ 2 of 3, each of those with a one-sided paired t p < 0.05.
- The whole item passes only if leakage and CRPS both pass.

    python benchmarks/trace_sims/run_v2_item2.py
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
OUT = ROOT / "benchmarks" / "results" / "trace_v2_item2.json"
_NORM = NormalDist()
V0_HE3 = 0.0018508010614343975
FRESH = list(range(81001, 81021))
LEAK_FRESH = list(range(83001, 83021))
COMMON = dict(
    k_fit=8,
    k_report=4,
    n_u=0,
    max_bins=16,
    learning_rate=1.0,
    max_rounds=40,
    tol=1e-8,
    mono="none",
)


def _z(u: np.ndarray) -> np.ndarray:
    return np.array([_NORM.inv_cdf(float(v)) for v in np.clip(u, 1e-12, 1.0 - 1e-12)])


def _vendor(n: int, seed: int):
    rng = np.random.default_rng(seed)
    log_units = rng.uniform(0, 2, size=n)
    unit_price = rng.uniform(0.5, 5, size=n)
    deliveries = rng.integers(1, 12, size=n).astype(np.float64)
    distance = rng.uniform(0, 1, size=n)
    quality = rng.uniform(0, 1, size=n)
    season = rng.uniform(0, 1, size=n)
    noise = rng.normal(size=(n, 3))
    x = np.column_stack([log_units, unit_price, deliveries, distance, quality, season, noise])
    z = _z(rng.random(n))
    he3 = z**3 - 3 * z
    y = 10 + 4 * log_units + 2 * unit_price + (0.8 + 0.5 * log_units) * z + 0.05 * quality * he3
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def _s1(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(n, 8))
    z = _z(rng.random(n))
    y = 1.2 * x[:, 0] + (0.4 + 1.2 * x[:, 1]) * z + 0.12 * x[:, 2] * (z**3 - 3 * z)
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def _scale(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.random((n, 10))
    y = x[:, 0] + 0.1 * rng.normal(size=n)
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def _draw(case: str, seed: int):
    if case == "s1_period2":
        return _s1(2000, seed)
    if case == "vendor_spend":
        return _vendor(4000, seed)
    if case == "scale_small":
        return _scale(20_000, seed)
    raise ValueError(case)


def _split(x, y, frac=0.7):
    n = int(frac * len(y))
    return x[:n], y[:n], x[n:], y[n:]


def _ttest_less(diff: list[float]) -> dict:
    from scipy import stats

    a = np.asarray(diff, dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) < 3:
        return {"p": None, "mean": None, "n": int(len(a))}
    if float(np.std(a)) == 0.0:
        return {"p": 1.0 if float(a[0]) >= 0 else 0.0, "mean": float(a.mean()), "n": int(len(a))}
    res = stats.ttest_rel(a, np.zeros_like(a), alternative="less")
    return {"p": float(res.pvalue), "mean": float(a.mean()), "n": int(len(a))}


def _one(spec: dict) -> dict:
    from traceot import TraceV1, _core

    kind = spec["kind"]
    if kind == "leak":
        rng = np.random.default_rng(int(spec["seed"]))
        x = np.ascontiguousarray(rng.random((2000, 1)))
        z = _z(rng.random(2000))
        y = np.ascontiguousarray((0.5 + x[:, 0]) * z)
        t0 = time.perf_counter()
        model = TraceV1(**COMMON, reporting="hermite", smooth=0.05, levenberg=1e-3).fit(x, y)
        elapsed = time.perf_counter() - t0
        energy = float(model.channel_energy(0, 3))
        ratio = energy / V0_HE3
        row = {
            "kind": "leak",
            "seed": int(spec["seed"]),
            "historical": bool(spec["historical"]),
            "he3": energy,
            "ratio": ratio,
            "rounds": int(model.model_.rounds),
            "seconds": elapsed,
        }
        print(f"leak seed {spec['seed']} ratio {ratio:.4f} rounds {row['rounds']}", flush=True)
        return row

    case, seed = spec["case"], int(spec["seed"])
    x, y = _draw(case, seed)
    xtr, ytr, xte, yte = _split(x, y)
    t0 = time.perf_counter()
    cand = TraceV1(**COMMON, reporting="legendre", smooth=0.05, levenberg=1e-3).fit(xtr, ytr)
    cand_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    base = TraceV1(**COMMON, reporting="legendre", smooth=0.0, levenberg=0.0).fit(xtr, ytr)
    base_s = time.perf_counter() - t0
    nodes = np.asarray(cand.model_.u_nodes(), dtype=np.float64)
    weights = np.asarray(_core.v1_gauss_legendre(int(nodes.shape[0]))[1], dtype=np.float64)
    q_c = cand.predict(xte, nodes, project=False)
    q_b = base.predict(xte, nodes, project=False)
    crps_c = float(
        _core.v1_crps_quadrature(
            np.ascontiguousarray(yte), np.ascontiguousarray(q_c), nodes, weights
        )
    )
    crps_b = float(
        _core.v1_crps_quadrature(
            np.ascontiguousarray(yte), np.ascontiguousarray(q_b), nodes, weights
        )
    )
    row = {
        "kind": "crps",
        "case": case,
        "seed": seed,
        "crps_candidate": crps_c,
        "crps_baseline": crps_b,
        "rounds_candidate": int(cand.model_.rounds),
        "rounds_baseline": int(base.model_.rounds),
        "seconds_candidate": cand_s,
        "seconds_baseline": base_s,
    }
    print(
        f"{case} seed {seed} crps {crps_c:.4f}/{crps_b:.4f} rounds {row['rounds_candidate']}/{row['rounds_baseline']}",
        flush=True,
    )
    return row


def _summarise(rows: list[dict]) -> dict:
    leaks = [r for r in rows if r["kind"] == "leak"]
    fresh = [r for r in leaks if not r["historical"]]
    hist = [r for r in leaks if r["historical"]]
    ratios = [r["ratio"] for r in fresh]
    t_leak = _ttest_less([r - 0.10 for r in ratios])
    seed6 = hist[0]["ratio"] if hist else None
    leak_pass = bool(
        seed6 is not None
        and seed6 <= 0.10
        and float(np.mean(ratios)) <= 0.10
        and t_leak["p"] is not None
        and t_leak["p"] < 0.05
    )
    by = {}
    for case in ("s1_period2", "vendor_spend", "scale_small"):
        block = [r for r in rows if r["kind"] == "crps" and r["case"] == case]
        c = [r["crps_candidate"] for r in block]
        b = [r["crps_baseline"] for r in block]
        ratio = float(np.mean(c) / np.mean(b))
        t = _ttest_less([ci - bi for ci, bi in zip(c, b, strict=True)])
        by[case] = {
            "n": len(block),
            "crps_candidate": float(np.mean(c)),
            "crps_baseline": float(np.mean(b)),
            "ratio": ratio,
            "magnitude": ratio <= 1.005,
            "better_mean": float(np.mean(c)) < float(np.mean(b)),
            "better_ttest": t["p"] is not None and t["p"] < 0.05,
            "ttest": t,
            "rounds_candidate": float(np.mean([r["rounds_candidate"] for r in block])),
            "rounds_baseline": float(np.mean([r["rounds_baseline"] for r in block])),
        }
    wins = sum(1 for c in by if by[c]["better_mean"] and by[c]["better_ttest"])
    crps_pass = all(by[c]["magnitude"] for c in by) and wins >= 2
    return {
        "leakage": {
            "v0_he3": V0_HE3,
            "seed6_ratio": seed6,
            "fresh_mean_ratio": float(np.mean(ratios)),
            "ttest": t_leak,
            "pass": leak_pass,
        },
        "cases": by,
        "crps_wins": wins,
        "crps_pass": crps_pass,
        "pass": bool(leak_pass and crps_pass),
    }


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


def main() -> None:
    specs = [{"kind": "leak", "seed": 6, "historical": True}]
    specs += [{"kind": "leak", "seed": s, "historical": False} for s in LEAK_FRESH]
    for case in ("s1_period2", "vendor_spend", "scale_small"):
        for seed in FRESH:
            specs.append({"kind": "crps", "case": case, "seed": seed})
    rows = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_one, spec) for spec in specs]
        for fut in as_completed(futures):
            rows.append(fut.result())
            payload = {
                "completed": len(rows),
                "of": len(specs),
                "seconds": time.perf_counter() - t0,
                "rows": rows,
            }
            if len(rows) == len(specs):
                payload["summary"] = _summarise(rows)
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(_finite(payload), indent=2, allow_nan=False) + "\n")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
