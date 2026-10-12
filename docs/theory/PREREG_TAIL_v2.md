# PREREG_TAIL_v2: corrected tail-identification tests (replaces v1 T3, T4, T6)

Status: preregistration. This file was written before any TRACE fit was run against these criteria. Computations: `verify/prereg_tail_v2.py`, with shared code in `verify/prereg_core.py`. Finite-sample Monte Carlo: `verify/prereg_validate.py` and `verify/prereg_validate_null.py`, cached in `verify/prereg_validation*.npz`. Output: `verify/PREREG_TAIL_v2_RESULTS.txt` (run 2026-10-12 10:25 EDT). numpy/scipy only. TRACE's fitter was not run.

**The v1 results for T3, T4 and T6 stay on record as failed.** Those three tests compared a pinball-loss fit against L2-projection oracles, and a truncated pinball projection is not the L2 projection (`v1:1.17`). The build agent's dispute is correct. The tests below replace them for future rounds; they do not re-score the v1 round.

## 0. Fitter span the oracle is computed in (read from `crates/core/src/trace/v1fit.rs`, `legendre.rs`, `binning.rs` at main 4ce6430)

- Coefficients: orthonormal shifted Legendre `e_k(u) = √(2k+1) P*_k(u)`, `K_fit = 8`.
- u-nodes: `n_u = 0` resolves to `max(K_fit, 12) = 12` Gauss–Legendre nodes on (0,1).
- Loss: `Σ_i Σ_t w_t ρ_{u_t}(y_i − e(u_t)·a(x_i))`. The weights `w_t` are the GL weights. There is no extra tail weighting.
- Model: additive, `a(x) = a_0 + Σ_j g_j[bin_j(x)]`. There are 16 equal-count bins whose edges are the order statistics at `⌊kn/16⌋`, and bin 0 is the missing-value bin. Effects are centred by training counts.
- Reporting: `r = Gᵀ g`, with `G[k,m] = ⟨e_k, col_m⟩` computed by 1024-node GL, `K_report = 4`.
- Energy: `channel_energy(j, m)` = mean over the 16 value bins of the squared reported coefficient.
- Reporting bases considered:
  - `legendre`;
  - `hermite` (`He_m(z)/√m!`);
  - `reference` with an empty reference, which the code defaults to `z = Φ⁻¹(u)`. Its columns are 1, centred-normalised z, and Gram–Schmidt of `e_2`, `e_3`.

## 1. Population oracle

**Definition.** For a design with `x ~ U[0,1]^p` and `Y | x = μ(x) + σ(x)Z + s(x)He3(Z)`, where `Z ~ N(0,1)`, the oracle `θ*` minimises the population loss `E Σ_t w_t ρ_{u_t}(Y − e(u_t)·a_θ(x))` over the span above. Population bin edges are `b/16`.

**Noise features are exactly zero at the oracle.** Take a feature that is independent of `(Y, relevant x)`. For any effect it carries, Jensen's inequality applied for each value of the relevant x gives a loss at least that of the effect's mean, which is a constant absorbed by the intercept. The oracle is therefore computed over the relevant features only.

**Two independent methods, both required to agree to `max|Δθ| ≤ 5e-5` and relative energy difference ≤ 1e-3:**
1. Damped Newton on the first-order condition `E[(F(q|x) − u_t) e(u_t)] = 0`, with the Hessian `E[f(q|x) e eᵀ]`.
   - `F` and `f` are exact: `F = Φ(c)`, where `c` is the real root of the monotone cubic.
   - The integral over x uses 5 GL nodes per bin per relevant feature (80³ points for S1).
2. L-BFGS on the closed-form expected loss `E ρ_u(Y−q) = u(μ−q) − (μΦ(c) − σφ(c) + s(1−c²)φ(c) − qΦ(c))`.
   - The integral over x uses 3 GL nodes per bin.
   - It starts from a different initial point.

Both methods agree for every design:

| Design | max \|Δθ\| | max rel. energy diff |
|---|---:|---:|
| S1-alt | 1.54e-5 | 2.6e-4 |
| S1-null | 1.1e-7 | 2.4e-6 |
| T6 | 1.2e-7 | 7.3e-7 |

All three pass.

**Oracle channel energies.** These are population values, `E(j,m)`, with rows for features and columns for channels 0–3. The oracle values for location and scale are ≈ 0.1195 and ≈ 0.112–0.116 (channel 0 and channel 1).

S1-alt (`μ = 1.2x_loc`, `σ = 0.4+1.2x_scale`, `s = 0.12x_shape`), tail channel 3:

| Basis | x_loc | x_scale | x_shape | Oracle separation E(shape,3)/max(other,3) |
|---|---:|---:|---:|---:|
| legendre | ~0 | 3.463e-3 | 1.722e-3 | **0.497 (fails)** |
| hermite | ~0 | 2.511e-4 | 1.860e-3 | 7.41 |
| reference (e1 = z) | ~0 | 1.213e-5 | 2.025e-3 | 167 |

The noise features are 0 in every basis. The x_shape channel-3 slopes against the bin centre at the oracle are 0.144 (legendre), 0.150 (hermite) and 0.156 (reference). The v1 bar of 0.294 = 0.12√6 is the L2 value with no span restriction.

S1-null (s = 0): x_shape has zero energy in every channel. x_scale's channel-3 energy is 3.40e-3 (legendre), 2.76e-4 (hermite) and 1.56e-5 (reference).

T6 (pure scale `(0.5+x)z`, one feature): the channel-3 energy is 2.37e-3 (legendre), 1.91e-4 (hermite) and 1.05e-5 (reference). The channel-1 energy is 0.078–0.080. At the population pinball optimum, scale therefore has a small but non-zero channel-3 load in every basis; it is not zero.

**The disputed example** is `0.8z + 0.12He3(z)`, intercept only, reported in the Hermite He3 channel:
- the population pinball optimum is **0.11253** (BFGS and Newton agree to 4e-10);
- the L2 projection into the span gives 0.2050 when computed exactly and 0.1635 on the 12 fitting nodes;
- the truth outside the span is 0.2939.

The build agent's fitted value of ≈ 0.118 is consistent with the pinball optimum. Their "≈ 0.18" for the L2 value is not reproduced exactly; that value depends on the quadrature used.

## 2. Which identification claim

- **(b) Separation at a fixed bar is only meaningful where the oracle itself separates.** In the Legendre basis the oracle ratio is 0.497 < 3. **No estimator can pass a 3× separation test in Legendre, at any n.** The asymptotic pass probability is 0.0000 at every n from 2000 to 32000. Legendre is therefore used only for a consistency test (T4v2).
- **The basis is stated up front: `reference` with e1 = z, which is TRACE's default reference.** Both hermite (7.4×) and reference (167×) separate at the oracle. Reference was chosen because it has the larger oracle margin and the higher finite-sample power at every n (table below).
- **Primary claim:** (b) separation, `E(x_shape,3) ≥ 3·max_{j≠x_shape} E(j,3)`, evaluated across seeds.
- **Secondary claim:** (a) consistency, meaning that energies fall inside finite-sample bands centred on the reference estimator's sampling distribution.

Consistency is not tested as "within X% of the oracle". At finite n the energies are biased: noise adds energy, and the pinball estimator shrinks the tail coefficient. At n = 16000 the mean x_shape energy from the reference estimator is 1.92e-3, against the oracle's 2.03e-3.

## 3. Power and size analysis (not based on TRACE results)

**Asymptotic law.** θ̂ ≈ N(θ*, H⁻¹SH⁻¹/n), where H and S are the population Hessian and score covariance at the oracle for the full 8-feature model. Each cell uses 4000 draws. Each entry gives the per-seed probability that the 3× separation passes:

| n | legendre (alt / null) | hermite (alt / null) | reference (alt / null) |
|---:|---|---|---|
| 2000 | 0.000 / 0.000 | 0.009 / 0.000 | 0.017 / 0.000 |
| 4000 | 0.000 / 0.000 | 0.083 / 0.000 | 0.146 / 0.000 |
| 8000 | 0.000 / 0.000 | 0.415 / 0.000 | 0.683 / 0.000 |
| 16000 | 0.000 / 0.000 | 0.796 / 0.000 | **0.992 / 0.000** |
| 32000 | 0.000 / 0.000 | 0.964 / 0.000 | 1.000 / 0.000 |

At the original n = 2000, even the population-correct estimator passes 3× separation in about 1–2% of seeds. The v1 S1/T3 design was underpowered regardless of the oracle question. **The chosen n is 16000.**

**Finite-sample check.** The reference estimator is an exact smoothed-pinball Newton fit in the same span (bandwidth 0.05 → 0.02 → 0.01). At n = 2000, seed 11, its channel-3 energies are within about 10% of the exact LP minimiser (LP: 412 s per fit). Results:
- **Alt** (20 reps, seeds 910000+): 19/20 pass separation. The minimum ratio is 2.35 and the median 4.90.
- **Null twin** (10 reps, seeds 920000+): 0/10 false positives. The maximum ratio is 1.06; the asymptotic 99th percentile is 1.39.
- Mean channel-3 energies, asymptotic simulation vs MC: x_loc 2.13e-4 vs 2.01e-4; x_scale 3.00e-4 vs 3.00e-4; noise 2.13e-4 vs 2.10e-4. x_shape is 2.27e-3 vs 1.92e-3; the asymptotic approximation overstates the tail energy by about 18%.

The MC is small, with 20 and 10 reps. Both runs were cut short for time, and this is why the separation bar below is conservative.

## 4. Replacement tests (frozen)

Common setup for all three tests:
- `TraceV1(k_fit=8, k_report=4, n_u=0, max_bins=16, mono="none", max_rounds=200, tol=1e-10)`.
- Data are drawn as in `run_v1._s1_period2`: x = rng.random((n,p)), then z = Φ⁻¹(rng.random(n)).
- Energies come from `channel_energy`. Twenty seeds per test.
- Each bar is set so that an estimator behaving like the reference estimator fails it with probability ≤ 0.01.
- Optimisation shortfall in TRACE counts against TRACE.

**T3v2 — tail separation (primary).**
- Design: S1 period 2, p = 8, **n = 16000**, `reporting="reference"` (default reference z).
- Seeds: 11, 2101–2119.
- Per seed: pass if `E(x_shape,3) ≥ 3 · max_{j≠x_shape} E(j,3)`.
- **Pass if ≥ 9 of 20 seeds pass.** The per-seed rate used is min(asymptotic 0.9895, MC 99% Wilson lower bound 0.682) = 0.682, and P(fail) at that rate is 0.0087. At the asymptotic rate, P(fail) is 2e-19.
- Secondary consistency, reported but not a pass condition: per seed, all of x_loc, x_scale, x_shape channel-3 energies lie in [−5.6e-6, 4.07e-4], [−1.20e-4, 7.20e-4] and [7.29e-4, 3.11e-3] respectively, with ≥ 18/20 seeds. These bands are the MC mean ± 3 SD; negative lower ends are vacuous.

**T3v2-null — false-positive twin.**
- Same as T3v2 with the shape multiplier set to 0. Seeds 3101–3120.
- **Pass if false positives ≤ 1 of 20.** P(fail) is about 0 at the asymptotic rate (0/20000) and 0.264 at a hypothetical 5% per-seed rate.

**T4v2 — Legendre consistency.** This replaces the v1 separation-style T4, which the oracle itself fails.
- Same data and seeds as T3v2, with `reporting="legendre"`.
- Per seed: all three channel-3 energies lie in their bands, x_loc [−2.02e-5, 5.86e-4], x_scale [1.68e-3, 4.83e-3], x_shape [4.53e-4, 3.05e-3].
- **Pass if ≥ 18/20 seeds.** P(fail) is 0.0005, using a per-seed rate of 0.992 (normal approximation with a Bonferroni correction).
- The report must also state that 3× separation is impossible in Legendre (oracle ratio 0.497).

**T6v2 — scale leakage consistency.**
- Design: `y = (0.5+x)z`, p = 1, n = 2000, `reporting="reference"`. Seeds 6, 6101–6119.
- Per seed, both must hold: channel-3 energy in [−3.49e-4, 3.75e-3], and channel-3/channel-1 ratio ≤ 4.02e-2. These bars are from 40 MC reps.
- **Pass if ≥ 19/20 seeds.** P(fail) is 0.0052.
- The oracle leak ratio is 1.3e-4. The v1 bar "≤ 0.05 × v0 energy" is dropped because it compared two different estimators rather than an estimator against its own oracle.

**Rationale.** The v1 tests asked a pinball fitter to reproduce L2 projections, and asked for 3× tail separation in a basis where the population optimum of the fitter's own objective gives 0.497×. They also used n = 2000, where even a perfect estimator passes separation in fewer than 2% of seeds. v2 does four things:
- it computes the oracle from the fitter's own objective, span, nodes and binning;
- it tests separation only in a basis where that oracle separates (reference e1 = z, 167×);
- it sets n from a power analysis (16000);
- it adds a null twin and replaces "within X% of an oracle" with finite-sample bands.

## 5. Reproduction and provenance

- `.venv/bin/python verify/prereg_tail_v2.py` (about 3 min) reads the cached MC files. Regenerate them with `prereg_validate.py` and `prereg_validate_null.py`, which take hours; both were stopped early and the script uses the first 20 alt and 10 null reps.
- The exact-LP comparison (`fit_lp`) is in `prereg_core.py`. It was run once at n = 2000, seed 11, and is not part of the script.
- The sha256 of this file is recorded in `verify/PREREG_TAIL_v2.sha256` after freezing.
