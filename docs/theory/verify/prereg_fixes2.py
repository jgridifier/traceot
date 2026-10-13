"""Validation runner for FIXES2_PLAN.md (shift-v2 fixes, new seeds 6000..6019).

Copied from prereg_shift_v2.py (which stays frozen). Differences are listed in
FIXES2_PLAN.md §Validation and below: seeds, truth seeds and common random
numbers, TRACE fit with pit_rounds=20, trim 1.0, B_refit=100, B5b calibration.

Data rows use numpy's PCG64 (``np.random.default_rng``). Folds, bootstrap
indices, and permutations use the core's SplitMix64, seeded with the same
integer as the scenario seed. β_A is not in the pre-registration; it is fixed
below and is not revised after the bars are scored.

    python docs/theory/verify/prereg_fixes2.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from statistics import NormalDist

from traceot import TraceV1, shift

ROOT = Path(__file__).resolve().parent
CACHE = Path("/tmp/prereg_fixes2_cache")
OUT = ROOT / "FIXES2_RESULTS.txt"

# Locked before any seed is scored.
BETA_A = np.array([1.0, 0.5, -0.7, 0.3, -0.2, 0.15, 0.1, -0.05], dtype=np.float64)
FIT = dict(
    k_fit=6,
    k_report=6,
    n_u=0,
    max_bins=32,
    max_rounds=12,
    tol=1e-6,
    min_count=1,
    ridge=1e-6,
    learning_rate=1.0,
    mono="none",
    bernstein_elevation=4,
    reporting="legendre",
    pit_rounds=20,
)
N = 20_000
P = 8
K = 6
N_QUAD = 64
N_FOLDS = 5
TRIM = 1.0
B = 200
B_REFIT = 100
CALIB_SEEDS = list(range(6100, 6500))
R = 200
MC_N = 2_000_000
PROJ_BINS = 64
PROJ_ROUNDS = 40
SEEDS = list(range(6000, 6020))
TRUTH_SEED_EPS = 7007
TRUTH_SEED = {"S1": 7008, "S2": 7009, "S3": 7010, "S5": 7012}
TRUTH_SEED_FEATURES = 7011


def cov_factor(p: int) -> np.ndarray:
    idx = np.arange(p)
    sigma = 0.3 ** np.abs(idx[:, None] - idx[None, :])
    return np.linalg.cholesky(sigma)


def skew_eps(rng: np.random.Generator, n: int) -> np.ndarray:
    alpha = 3.0
    delta = alpha / np.sqrt(1.0 + alpha * alpha)
    z = delta * np.abs(rng.standard_normal(n)) + np.sqrt(1.0 - delta * delta) * rng.standard_normal(
        n
    )
    mean = delta * np.sqrt(2.0 / np.pi)
    var = 1.0 - 2.0 * delta * delta / np.pi
    return (z - mean) / np.sqrt(var)


def draw_x(rng: np.random.Generator, n: int, mu: np.ndarray, chol: np.ndarray) -> np.ndarray:
    return rng.standard_normal((n, chol.shape[0])) @ chol.T + mu


def scale_of(x: np.ndarray, factor: float) -> np.ndarray:
    return factor * np.exp(0.2 * x[:, 0] - 0.1 * x[:, 1])


def mean_of(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.tanh(x) @ beta


def response(x, beta, factor, eps) -> np.ndarray:
    return mean_of(x, beta) + scale_of(x, factor) * eps


def scenario(name: str) -> dict:
    mu = np.zeros(P)
    beta_b = BETA_A.copy()
    factor = 1.0
    if name in ("S1", "S3"):
        mu = np.array([0.5, -0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    if name in ("S2", "S3"):
        beta_b = BETA_A + np.array([0.4, 0.0, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
        factor = 1.2
    return {"mu": mu, "beta_b": beta_b, "factor": factor}


def draw_period(rng, n, mu, beta, factor, chol):
    x = draw_x(rng, n, mu, chol)
    y = response(x, beta, factor, skew_eps(rng, n))
    return x, y


def gamma_of_eps() -> np.ndarray:
    path = CACHE / "gamma.json"
    if path.exists():
        return np.asarray(json.loads(path.read_text()), dtype=np.float64)
    rng = np.random.default_rng(TRUTH_SEED_EPS)
    lam = np.asarray(shift.sample_lambda(skew_eps(rng, MC_N), K, N_QUAD))
    gamma = (2 * np.arange(K) + 1) * lam
    path.write_text(json.dumps(gamma.tolist()))
    return gamma


def marginal_truth(name: str, chol: np.ndarray) -> dict:
    path = CACHE / f"truth_{name}_marginal.json"
    if path.exists():
        raw = json.loads(path.read_text())
        return {k: np.asarray(v) if isinstance(v, list) else v for k, v in raw.items()}
    spec = scenario(name)
    rng = np.random.default_rng(TRUTH_SEED[name])
    # Common random numbers: each law is sampled exactly, but A and C share
    # (x_A, ε_1), B and C' share (x_B, ε_2), and when the covariate laws coincide
    # x_B reuses x_A. The four laws are unchanged; only the MC noise in their
    # differences is removed (S1 struct and S2 comp are then exactly 0).
    xa = draw_x(rng, MC_N, np.zeros(P), chol)
    e1 = skew_eps(rng, MC_N)
    if np.all(spec["mu"] == 0.0):
        xb, e2 = xa, e1
    else:
        xb = draw_x(rng, MC_N, spec["mu"], chol)
        e2 = skew_eps(rng, MC_N)
    ya = response(xa, BETA_A, 1.0, e1)
    yc = response(xa, spec["beta_b"], spec["factor"], e1)
    yb = response(xb, spec["beta_b"], spec["factor"], e2)
    ycp = response(xb, BETA_A, 1.0, e2)
    view = shift.from_laws(ya, yb, yc, ycp, K, N_QUAD)
    w2 = float(view["w2_total"])
    out = {
        "comp_w2": np.asarray(view["comp_w2"]).tolist(),
        "struct_w2": np.asarray(view["struct_w2"]).tolist(),
        "comp_share": float(view["comp_w2_total"]) / w2,
        "struct_share": float(view["struct_w2_total"]) / w2,
        "w2_total": w2,
    }
    path.write_text(json.dumps(out))
    return {k: np.asarray(v) if isinstance(v, list) else v for k, v in out.items()}


def _bin_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    probs = np.arange(1, n_bins) / n_bins
    return np.quantile(x, probs)


def additive_means(x_train, y, x_a, x_b, n_bins=PROJ_BINS, rounds=PROJ_ROUNDS):
    """Centered equal-mass backfit. Returns mean effect of each feature on A and on B."""
    n, p = x_train.shape
    edges = []
    codes = []
    for j in range(p):
        edge = _bin_edges(x_train[:, j], n_bins)
        edges.append(edge)
        codes.append(np.searchsorted(edge, x_train[:, j], side="right"))
    effects = [np.zeros(n_bins) for _ in range(p)]
    pred = np.zeros(n)
    for _ in range(rounds):
        for j in range(p):
            current = effects[j][codes[j]]
            resid = y - pred + current
            counts = np.bincount(codes[j], minlength=n_bins).astype(np.float64)
            sums = np.bincount(codes[j], weights=resid, minlength=n_bins)
            means = np.zeros(n_bins)
            nz = counts > 0
            means[nz] = sums[nz] / counts[nz]
            means -= np.dot(means, counts) / n
            new = means[codes[j]]
            pred += new - current
            effects[j] = means

    def average(x_eval):
        out = np.zeros(p)
        for j in range(p):
            code = np.searchsorted(edges[j], x_eval[:, j], side="right")
            code = np.clip(code, 0, n_bins - 1)
            out[j] = float(effects[j][code].mean())
        return out

    return average(x_a), average(x_b)


def feature_truth(chol: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Population per-feature composition W2 shares, summed over channels. Shape (p,)."""
    path = CACHE / "truth_S3_features.json"
    if path.exists():
        return np.asarray(json.loads(path.read_text()), dtype=np.float64)
    spec = scenario("S3")
    rng = np.random.default_rng(TRUTH_SEED_FEATURES)
    xa = draw_x(rng, MC_N, np.zeros(P), chol)
    xb = draw_x(rng, MC_N, spec["mu"], chol)
    phi = np.zeros(P)
    # Channel 0 is exactly additive: c = β · tanh(x).
    ta = np.tanh(xa).mean(axis=0)
    tb = np.tanh(xb).mean(axis=0)
    g_aa = BETA_A * (ta - ta)
    g_ab = BETA_A * (tb - ta)
    g_ba = spec["beta_b"] * (ta - tb)
    g_bb = spec["beta_b"] * (tb - tb)
    comp = 0.5 * ((g_ab - g_aa) + (g_bb - g_ba))
    delta = float(mean_of(xb, spec["beta_b"]).mean() - mean_of(xa, BETA_A).mean())
    phi += comp * delta
    for channel in range(1, K):
        sa = scale_of(xa, 1.0)
        sb = scale_of(xb, spec["factor"])
        g_aa, g_ab = additive_means(xa, sa, xa, xb)
        g_ba, g_bb = additive_means(xb, sb, xa, xb)
        comp = 0.5 * ((g_ab - g_aa) + (g_bb - g_ba))
        delta_c = float(gamma[channel]) * float(sb.mean() - sa.mean())
        comp = comp * float(gamma[channel])
        phi += comp * delta_c / (2 * channel + 1)
    path.write_text(json.dumps(phi.tolist()))
    return phi


def entry_truth(chol: np.ndarray, gamma: np.ndarray) -> dict:
    path = CACHE / "truth_S5.json"
    if path.exists():
        raw = json.loads(path.read_text())
        return {k: np.asarray(v, dtype=np.float64) for k, v in raw.items()}
    z = NormalDist().inv_cdf(0.85)
    rng = np.random.default_rng(TRUTH_SEED["S5"])

    def conditional(n, above):
        chunks = []
        have = 0
        while have < n:
            x = draw_x(rng, n, np.zeros(P), chol)
            keep = x[:, 0] > z if above else x[:, 0] <= z
            chunks.append(x[keep])
            have += int(keep.sum())
        return np.vstack(chunks)[:n]

    stay = conditional(MC_N, above=False)
    exited = conditional(MC_N, above=True)
    entered = draw_x(rng, MC_N, np.array([1.0, 0, 0, 0, 0, 0, 0, 0]), chol)
    e_a, e_b = 0.15, 0.25

    def curve(x):
        c = np.zeros(K)
        c[0] = float(mean_of(x, BETA_A).mean())
        s = float(scale_of(x, 1.0).mean())
        for channel in range(1, K):
            c[channel] = float(gamma[channel]) * s
        return c

    entry = e_b * curve(entered)
    exit_ = -e_a * curve(exited)
    stay_term = np.zeros(K)
    # Stayers share a law, so the stay term is 0. ``curve(stay)`` is unused in the term
    # and retained so the group is actually drawn under the declared seed.
    _ = curve(stay)
    out = {"entry": entry.tolist(), "exit": exit_.tolist(), "stay": stay_term.tolist()}
    path.write_text(json.dumps(out))
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def covers(lo, hi, truth) -> bool:
    return float(lo) <= float(truth) <= float(hi)


def fit_pair(x_a, y_a, x_b, y_b):
    model_a = TraceV1(**FIT).fit(x_a, y_a)
    model_b = TraceV1(**FIT).fit(x_b, y_b)
    return model_a, model_b


def marginal_gaps(view) -> float:
    gap = 0.0
    for channel in range(K):
        delta = float(view["lambda_b"][channel] - view["lambda_a"][channel])
        share = float(view["comp_w2"][channel] + view["struct_w2"][channel])
        gap = max(gap, abs(share - (2 * channel + 1) * delta * delta))
    resolved = float(view["comp_w2_total"] + view["struct_w2_total"])
    gap = max(gap, abs(resolved - float(view["w2_resolved"])))
    return gap


def entry_gaps(got) -> float:
    gap = 0.0
    for channel in range(K):
        players = np.array(
            [
                got["stay"][channel],
                got["mix"][channel],
                got["entry"][channel],
                got["exit"][channel],
            ],
            dtype=np.float64,
        )
        gap = max(gap, abs(float(players.sum()) - float(got["delta"][channel])))
        pred = float(got["delta"][channel]) ** 2 / (2 * channel + 1)
        got_phi = (
            float(got["phi_stay"][channel])
            + float(got["phi_mix"][channel])
            + float(got["phi_entry"][channel])
            + float(got["phi_exit"][channel])
        )
        gap = max(gap, abs(got_phi - pred))
    return gap


def share_band(point, replicates, truth):
    band = shift.bands(
        np.asarray([point], dtype=np.float64),
        np.asarray(replicates, dtype=np.float64).reshape(-1, 1),
    )
    hit = covers(band["lo"][0], band["hi"][0], truth)
    return hit, float(band["lo"][0]), float(band["hi"][0])


def component_hits(point, replicates, truth) -> list[bool]:
    band = shift.bands(
        np.asarray(point, dtype=np.float64), np.asarray(replicates, dtype=np.float64)
    )
    return [covers(band["lo"][i], band["hi"][i], truth[i]) for i in range(len(point))]


def any_uniform_resolved(point, replicates) -> bool:
    band = shift.bands(
        np.asarray(point, dtype=np.float64), np.asarray(replicates, dtype=np.float64)
    )
    return any(bool(flag) for flag in band["resolved"])


def build_s5(rng, chol):
    x_a, y_a = draw_period(rng, N, np.zeros(P), BETA_A, 1.0, chol)
    threshold = float(np.quantile(x_a[:, 0], 0.85))
    exit_a = (x_a[:, 0] > threshold).astype(np.uint8)
    stay = exit_a == 0
    n_stay = int(stay.sum())
    n_entry = int(round(n_stay * 0.25 / 0.75))
    x_entry = draw_x(rng, n_entry, np.array([1.0, 0, 0, 0, 0, 0, 0, 0.0]), chol)
    y_entry = response(x_entry, BETA_A, 1.0, skew_eps(rng, n_entry))
    x_b = np.vstack([x_a[stay], x_entry])
    y_b = np.concatenate([y_a[stay], y_entry])
    entry_b = np.zeros(x_b.shape[0], dtype=np.uint8)
    entry_b[n_stay:] = 1
    return x_a, y_a, exit_a, x_b, y_b, entry_b


def run_seed(name: str, seed: int, chol, truths) -> dict:
    path = CACHE / f"{name}_{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    spec = scenario(name)
    if name == "S5":
        x_a, y_a, exit_a, x_b, y_b, entry_b = build_s5(rng, chol)
    else:
        x_a, y_a = draw_period(rng, N, np.zeros(P), BETA_A, 1.0, chol)
        x_b, y_b = draw_period(rng, N, spec["mu"], spec["beta_b"], spec["factor"], chol)
        exit_a = entry_b = None
    model_a, model_b = fit_pair(x_a, y_a, x_b, y_b)
    report = shift.explain(
        model_a,
        model_b,
        x_a,
        y_a,
        x_b,
        y_b,
        n_quad=N_QUAD,
        n_folds=N_FOLDS,
        trim_quantile=TRIM,
        seed=seed,
    )
    marginal = report["marginal"]
    bary = report["bary"]
    rec = {
        "scenario": name,
        "seed": seed,
        "order_gap": float(report["order_gap"]),
        "player_sum_gap": max(float(report["player_sum_gap"]), marginal_gaps(marginal)),
        "trim_c": float(marginal["trim_mass_c"]),
        "trim_cp": float(marginal["trim_mass_cp"]),
        "w2_total": float(marginal["w2_total"]),
        "w2_resolved": float(marginal["w2_resolved"]),
        "gap_w2": float(report["gap_w2"]),
    }
    draws = shift.bootstrap_marginal(
        x_a, y_a, x_b, y_b, B, seed, k=K, n_quad=N_QUAD, n_folds=N_FOLDS, trim_quantile=TRIM
    )
    if name == "S1":
        point = float(marginal["struct_w2_total"]) / float(marginal["w2_total"])
        reps = np.asarray(draws["struct_total"]) / np.asarray(draws["w2_total"])
        hit, lo, hi = share_band(point, reps, float(truths["S1"]["struct_share"]))
        rec.update(
            share=point,
            share_lo=lo,
            share_hi=hi,
            share_hit=hit,
            share_truth=float(truths["S1"]["struct_share"]),
        )
    if name == "S2":
        point = float(marginal["comp_w2_total"]) / float(marginal["w2_total"])
        reps = np.asarray(draws["comp_total"]) / np.asarray(draws["w2_total"])
        hit, lo, hi = share_band(point, reps, float(truths["S2"]["comp_share"]))
        rec.update(
            share=point,
            share_lo=lo,
            share_hi=hi,
            share_hit=hit,
            share_truth=float(truths["S2"]["comp_share"]),
        )
    if name == "S3":
        point = np.concatenate(
            [np.asarray(marginal["comp_w2"])[:3], np.asarray(marginal["struct_w2"])[:3]]
        )
        reps = np.concatenate(
            [np.asarray(draws["comp_w2"])[:, :3], np.asarray(draws["struct_w2"])[:, :3]], axis=1
        )
        truth = np.concatenate([truths["S3"]["comp_w2"][:3], truths["S3"]["struct_w2"][:3]])
        rec["b3_hit"] = component_hits(point, reps, truth)
        rec["b3_hat"] = point.tolist()
        rec["b3_truth"] = truth.tolist()
        bdraw = shift.bootstrap_barycenter(model_a, model_b, x_a, y_a, x_b, y_b, B_REFIT, seed)
        # Point estimate is the fitted barycenter, not a bootstrap mean.
        phi = np.asarray(bary["phi"])
        hat = phi[:P].sum(axis=1)
        truth_f = truths["features"][:3]
        rec["b4_hit"] = component_hits(hat[:3], np.asarray(bdraw["comp_feature"])[:, :3], truth_f)
        rec["b4_hat"] = hat[:3].tolist()
        rec["b4_truth"] = truth_f.tolist()
    if name == "S4":
        perm = shift.permutation_w2(y_a, y_b, R, seed, n_quad=N_QUAD, k=K, level=0.95)
        rec["exceeds"] = bool(perm["exceeds"])
        rec["perm_observed"] = float(perm["observed"])
        rec["perm_floor"] = float(perm["floor"])
        point_m = np.concatenate(
            [np.asarray(marginal["comp_w2"]), np.asarray(marginal["struct_w2"])]
        )
        reps_m = np.concatenate(
            [np.asarray(draws["comp_w2"]), np.asarray(draws["struct_w2"])], axis=1
        )
        resolved = any_uniform_resolved(point_m, reps_m)
        bdraw = shift.bootstrap_barycenter(model_a, model_b, x_a, y_a, x_b, y_b, B_REFIT, seed)
        phi = np.asarray(bary["phi"]).reshape(-1)
        resolved = resolved or any_uniform_resolved(phi, np.asarray(bdraw["phi"]))
        rec["resolved"] = bool(resolved)
    if name == "S5":
        got = shift.entry_from_models(
            model_a, model_b, x_a, exit_a, x_b, entry_b, float(len(y_a)), float(len(y_b))
        )
        rec["player_sum_gap"] = max(rec["player_sum_gap"], entry_gaps(got))
        rec["hk2_mass"] = float(got["hk2_mass"])
        edraw = shift.bootstrap_entry(model_a, x_a, y_a, exit_a, x_b, y_b, entry_b, B_REFIT, seed)
        truth = truths["S5"]
        hits = {}
        for key in ("entry", "exit", "stay"):
            point = np.asarray(got[key])
            reps = np.asarray(edraw[key])
            hits[key] = component_hits(point, reps, truth[key])
        rec["b9_hit"] = hits
        rec["b9_hat_entry0"] = float(got["entry"][0])
        rec["b9_hat_exit0"] = float(got["exit"][0])
        rec["b9_hat_stay0"] = float(got["stay"][0])
    rec["seconds"] = time.perf_counter() - t0
    path.write_text(json.dumps(rec))
    print(
        f"{name} seed {seed} {rec['seconds']:.1f}s order {rec['order_gap']:.2e} players {rec['player_sum_gap']:.2e}",
        flush=True,
    )
    return rec


PLAN_SHA = "unknown"
CALIB_LINE = "B5b not run"


def plan_sha() -> str:
    import subprocess

    out = subprocess.run(
        ["git", "log", "-1", "--format=%h", "--", "docs/theory/FIXES2_PLAN.md"],
        cwd=ROOT.parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip() or "unknown"


def calibration(chol) -> str:
    """B5b: permutation false-positive rate on 400 fresh null datasets (S4 law)."""
    path = CACHE / "calibration.json"
    if path.exists():
        hits = json.loads(path.read_text())
    else:
        hits = []
        for seed in CALIB_SEEDS:
            rng = np.random.default_rng(seed)
            _, y_a = draw_period(rng, N, np.zeros(P), BETA_A, 1.0, chol)
            _, y_b = draw_period(rng, N, np.zeros(P), BETA_A, 1.0, chol)
            perm = shift.permutation_w2(y_a, y_b, R, seed, n_quad=N_QUAD, k=K, level=0.95)
            hits.append(bool(perm["exceeds"]))
        path.write_text(json.dumps(hits))
    n = len(hits)
    k = int(sum(hits))
    rate_ = k / n
    z = 1.959963984540054
    centre = (rate_ + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(rate_ * (1 - rate_) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    lo, hi = centre - half, centre + half
    ok = lo <= 0.05 <= hi and rate_ <= 0.075
    return (
        f"B5b permutation calibration {k}/{n} = {rate_:.4f}  Wilson95 [{lo:.4f}, {hi:.4f}]  "
        f"{'PASS' if ok else 'FAIL'}"
    )


def rate(flags) -> tuple[int, list[int]]:
    failed = [seed for seed, flag in flags if not flag]
    return len(flags) - len(failed), failed


def line(bar, hits, failed, ok) -> str:
    return f"{bar}  {hits}/20  failed_seeds={failed or '-'}  {'PASS' if ok else 'FAIL'}"


def write_results(records, truths, checks_text: str) -> None:
    by = {}
    for rec in records:
        by.setdefault(rec["scenario"], []).append(rec)
    for rows in by.values():
        rows.sort(key=lambda r: r["seed"])

    def flags(name, pred):
        return [(r["seed"], pred(r)) for r in by[name]]

    b1_hits, b1_fail = rate(flags("S1", lambda r: r["share_hit"]))
    b2_hits, b2_fail = rate(flags("S2", lambda r: r["share_hit"]))
    b3_rows = []
    b3_ok = True
    labels = [f"comp k={k}" for k in range(3)] + [f"struct k={k}" for k in range(3)]
    for i, label in enumerate(labels):
        hits, failed = rate(flags("S3", lambda r, i=i: r["b3_hit"][i]))
        b3_ok = b3_ok and hits >= 18
        b3_rows.append(f"    {label}  {hits}/20  failed_seeds={failed or '-'}")
    b4_rows = []
    b4_ok = True
    for i in range(3):
        hits, failed = rate(flags("S3", lambda r, i=i: r["b4_hit"][i]))
        b4_ok = b4_ok and hits >= 18
        b4_rows.append(f"    feature {i + 1}  {hits}/20  failed_seeds={failed or '-'}")
    b5_fp, b5_fail = rate(flags("S4", lambda r: not r["exceeds"]))
    # rate() counts True as hits. For B5 the pass count is the number of seeds that do NOT exceed.
    b5_hits = b5_fp
    b5_ok = (20 - b5_hits) <= 1
    b6_quiet, b6_fail = rate(flags("S4", lambda r: not r["resolved"]))
    b6_ok = (20 - b6_quiet) <= 1
    b7_by = []
    b8_by = []
    b7_fail = []
    b8_fail = []
    for name in ("S1", "S2", "S3", "S4", "S5"):
        ok7 = ok8 = 0
        for rec in by[name]:
            if rec["order_gap"] > 1e-9:
                b7_fail.append(f"{name}:{rec['seed']}")
            else:
                ok7 += 1
            if rec["player_sum_gap"] > 1e-9:
                b8_fail.append(f"{name}:{rec['seed']}")
            else:
                ok8 += 1
        b7_by.append(f"    {name}  {ok7}/20")
        b8_by.append(f"    {name}  {ok8}/20")
    b7_min = min(int(row.split()[1].split("/")[0]) for row in b7_by)
    b8_min = min(int(row.split()[1].split("/")[0]) for row in b8_by)
    b9_rows = []
    b9_ok = True
    for key in ("entry", "exit", "stay"):
        for channel in range(K):
            hits, failed = rate(
                flags("S5", lambda r, key=key, channel=channel: r["b9_hit"][key][channel])
            )
            b9_ok = b9_ok and hits >= 18
            b9_rows.append(f"    {key} k={channel}  {hits}/20  failed_seeds={failed or '-'}")

    b3_min = min(int(row.split()[2].split("/")[0]) for row in b3_rows)
    b4_min = min(int(row.split()[2].split("/")[0]) for row in b4_rows)
    # feature rows look like "    feature 1  18/20 ..."
    b9_min = min(int(row.split()[2].split("/")[0]) for row in b9_rows)

    checks_ok = "ALL PASS" in checks_text and "SOME FAILED" not in checks_text
    lines = [
        "FIXES2 validation results (bars in docs/theory/FIXES2_PLAN.md)",
        f"plan freeze sha: {PLAN_SHA}",
        "bars were not edited. A failed bar stays failed.",
        "",
        "Implementation choices fixed before scoring (not in the frozen pre-registration):",
        f"  beta_A = {BETA_A.tolist()}",
        "  numpy PCG64 via np.random.default_rng(seed) for every row of X, Y, and ε.",
        "  ε is skew-normal(α=3) by the |U0| construction, then standardized with the",
        "  population mean δ√(2/π) and variance 1−2δ²/π, δ=α/√(1+α²).",
        "  Folds, bootstrap indices, and permutations: SplitMix64 in the Rust core,",
        "  seeded with the scenario seed (6000..6019).",
        f"  TRACE fit: {FIT}",
        f"  quadrature nodes n_quad={N_QUAD}, folds={N_FOLDS}, trim={TRIM}, B={B} (marginal), B_refit={B_REFIT}, R={R}.",
        "  Logistic period classifier: intercept plus raw covariates, ridge 1e-6 on slopes,",
        "  at most 25 Newton steps, tolerance 1e-8. Same code path as the unit tests.",
        "  Monte Carlo truth n=2,000,000 with common random numbers. ε coefficients seed 7007.",
        "  Marginal laws seeds S1=7008, S2=7009, S3=7010. B4 projection 7011. Entry/exit 7012.",
        f"  B4 truth is the population additive projection (equal-mass bins={PROJ_BINS},",
        f"  backfit rounds={PROJ_ROUNDS}) of the true channel, then the same Shapley share",
        "  the barycenter reports: sum over channels of φ for that composition feature.",
        "  Channel 0 uses the exact tanh components instead of a backfit.",
        "  B1 scores the marginal structure share struct_w2_total / w2_total against the",
        "  Monte Carlo share. B2 scores the marginal composition share the same way.",
        "  B3 scores marginal comp_w2 and struct_w2 on channels k=0,1,2. A component",
        "  passes at ≥18/20. The bar passes only when every component does.",
        "  Bands for B1–B4 and B9 are the pointwise 2.5/97.5 bootstrap percentiles.",
        "  B6 uses the uniform max-|t| band, separately on the 2K marginal shares and on",
        "  the barycenter players. A seed counts if either band resolves any player.",
        "  B5 is the outcome-quantile W2² against the permutation 95% floor (R=200).",
        "  B7 is explain_shift.order_gap on every scenario. B8 is the barycenter player",
        "  sum, the marginal channel identity, and on S5 the entry/exit Prop. 4 identity.",
        "  B9 scores c-scale entry, exit, and stay on every channel k=0..5. Stay's truth",
        "  is 0. Population exit rate 0.15 uses Φ⁻¹(0.85); the estimator uses the sample",
        "  0.85 quantile, and the entry count is rounded so the sample entry share is 0.25.",
        "",
        "Truth snapshots:",
        f"  S1 struct share {float(truths['S1']['struct_share']):.6e}  w2 {float(truths['S1']['w2_total']):.6e}",
        f"  S2 comp share   {float(truths['S2']['comp_share']):.6e}  w2 {float(truths['S2']['w2_total']):.6e}",
        f"  S3 comp_w2[:3]  {np.asarray(truths['S3']['comp_w2'])[:3].tolist()}",
        f"  S3 struct_w2[:3] {np.asarray(truths['S3']['struct_w2'])[:3].tolist()}",
        f"  S3 feature shares 1..3 {np.asarray(truths['features'])[:3].tolist()}",
        f"  S5 entry {np.asarray(truths['S5']['entry']).tolist()}",
        f"  S5 exit  {np.asarray(truths['S5']['exit']).tolist()}",
        "",
        "Bars",
        line("B1", b1_hits, b1_fail, b1_hits >= 18),
        line("B2", b2_hits, b2_fail, b2_hits >= 18),
        f"B3  min {b3_min}/20  {'PASS' if b3_ok else 'FAIL'}",
        *b3_rows,
        f"B4  min {b4_min}/20  {'PASS' if b4_ok else 'FAIL'}",
        *b4_rows,
        f"B5  false_positives {20 - b5_hits}/20  seeds={b5_fail or '-'}  {'PASS' if b5_ok else 'FAIL'}",
        f"B6  resolved_seeds {20 - b6_quiet}/20  seeds={b6_fail or '-'}  {'PASS' if b6_ok else 'FAIL'}",
        f"B7  {b7_min}/20  failed={b7_fail or '-'}  {'PASS' if not b7_fail else 'FAIL'}",
        *b7_by,
        f"B8  {b8_min}/20  failed={b8_fail or '-'}  {'PASS' if not b8_fail else 'FAIL'}",
        *b8_by,
        f"B9  min {b9_min}/20  {'PASS' if b9_ok else 'FAIL'}",
        *b9_rows,
        f"B10 {'PASS' if checks_ok else 'FAIL'}",
        CALIB_LINE,
        "B11 see the speed section; not a correctness gate.",
        "",
        "Per seed",
    ]
    for name in ("S1", "S2", "S3", "S4", "S5"):
        for rec in by[name]:
            brief = {k: rec[k] for k in rec if k not in ("b3_hat",)}
            lines.append(json.dumps(brief))
    lines.extend(["", "B10 verify/shift_v2_checks.py", checks_text.rstrip(), ""])
    OUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    global PLAN_SHA, CALIB_LINE
    CACHE.mkdir(parents=True, exist_ok=True)
    PLAN_SHA = plan_sha()
    only = sys.argv[1:] or ["S1", "S2", "S3", "S4", "S5"]
    chol = cov_factor(P)
    gamma = gamma_of_eps()
    truths = {}
    if any(name in only for name in ("S1", "S2", "S3")):
        for name in ("S1", "S2", "S3"):
            if name in only or name == "S3" and "S3" in only:
                print(f"truth {name}", flush=True)
                truths[name] = marginal_truth(name, chol)
    if "S3" in only:
        print("truth features", flush=True)
        truths["features"] = feature_truth(chol, gamma)
    if "S5" in only:
        print("truth entry", flush=True)
        truths["S5"] = entry_truth(chol, gamma)
    # Reload every truth the writer needs when assembling a complete run.
    for name in ("S1", "S2", "S3"):
        truths[name] = marginal_truth(name, chol)
    truths["features"] = feature_truth(chol, gamma)
    truths["S5"] = entry_truth(chol, gamma)
    records = []
    for name in only:
        for seed in SEEDS:
            records.append(run_seed(name, seed, chol, truths))
    if only == ["S1", "S2", "S3", "S4", "S5"]:
        import subprocess

        proc = subprocess.run(
            [sys.executable, str(ROOT / "shift_v2_checks.py")],
            check=False,
            capture_output=True,
            text=True,
        )
        checks = proc.stdout + proc.stderr
        CALIB_LINE = calibration(chol)
        # Include cached seeds from every scenario.
        all_recs = []
        for name in ("S1", "S2", "S3", "S4", "S5"):
            for seed in SEEDS:
                all_recs.append(json.loads((CACHE / f"{name}_{seed}.json").read_text()))
        write_results(all_recs, truths, checks)
        print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
