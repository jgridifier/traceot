# Changelog

All notable changes to this project are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Status: research, unvalidated.

## [Unreleased]

### Added (2026-11-12)
- Hold record for the V5.1 coverage pre-registration: on 2026-11-11 at 7:31 AM ET the owner answered a card asking "Run the V5.1 coverage study?" with, quoted verbatim, "Not yet, hold the coverage study." Status in `docs/research/PREREG_V5_1_COVERAGE.md` is now **PRE-REGISTERED · NOT APPROVED · NOT RUN · HELD**. This is a hold, not approval to run. Record: `docs/research/HOLD_V5_1_COVERAGE.md` (also proposes, but does not register or run, a narrower sqrt(n) sampling-variability follow-up on the already-committed gate replications if the coverage study is never approved). Learn page `docs/learn/v5_1_coverage_prereg.html` updated. Indices 10400–10799 unused; 9600–9619 still sealed; no runner written or executed.
- Working paper `docs/paper/V5_1_working_paper.md`: final abstract; §3 per-step paragraphs and the v1–v3 records; §4 simulator-family and truth-file prose; §5 figure of paired d, drawn from the committed `benchmarks/results/v5_1_gate_shape.jsonl` by the deterministic script `docs/paper/fig_v5_1_paired_d.py` (same `graded_err` as the gate summariser; no TRACE fit; no seed touched; mean of the plotted points checked against the published mean d to 1e-15). Learn page `docs/learn/v5_1_paper.html` and `docs/index.html` updated to match. No new numbers from a new run.

### Added (2026-11-10)
- Report-only residual-vs-fit check of the location fit in `explain_v5`: `ExplainV5Report::linear_mean_check_a` / `linear_mean_check_b` (`LinearMeanCheck`: correlation of residuals with the fit and with the squared centred fit, mean fit and mean residual in `min(10, n)` equal-count bins of the fit, max |binned residual| / sd) The helper is public as `traceot_core::trace::linear_mean_check`. Python: `result["linear_mean_check"]["a" | "b"]` (with `diagnostic_only: True`). **Diagnostic only:** no validated claim, no threshold, gates nothing. It is computed after every other output and changes no existing value (checked bit-for-bit against `main` on all three location models); V5 gate result files are unchanged. Tests: Rust `linear_mean_check_flags_curvature_only`, `linear_mean_check_matches_direct_recompute`; Python `test_explain_v5_linear_mean_check_report_only`.
- Working paper `docs/paper/V5_1_working_paper.md`: §1 (problem, notation, relation to DFL and CFM, why L-moments), the full two-player Shapley derivation with the budget identity and grid/quadrature details in §2, and the §5 discussion, written from existing results and code only (no new numbers). Learn pages `v5_1_paper.html` and `v5_1_app.html` updated to match.

### Changed (2026-11-09)
- `explain_v5` now defaults to the TRACE_V5_1 location model (`mean_model="ols"`, `LocationModel::Ols`, `ExplainV5Config::default()`) in the Rust core and the Python binding. The v5 booster stays selectable with `mean_model="booster"` / `LocationModel::Booster`. Tests and V5 gate scripts that relied on the old default now pass `"booster"` explicitly. No results change; seeds 9600–9619 untouched.

### Fixed
- Bandwidth cross-validation now scales `h_j` by the range of the training rows of each fold. The previous version included the held-out fold's covariates in that range. Held-out responses were never used in the fit. See `docs/ADOPT_AUDIT.md`.

### Changed
- TRACE underfitting: centered feature updates no longer discard a common scale or shape step; that average is added once to `Q0`. Defaults (`learning_rate=1`, `max_rounds=40`, `patience=8`, `min_rounds=20`) come from a separate tuning draw, recorded in `benchmarks/results/trace_tune_grid.json`.
- TRACE feature screen tests the shape channel on its own with tail-weighted pinball, and cannot drop a feature before `min_rounds`.
- TRACE pair tables are purified on both margins into the main effects.
- When `n_bags > 1` and `bag_frac < 1`, validation rows are the bag's complement.
- `Q0` uses the lower pinball endpoint, `ceil(u n) - 1`.
- Optional split-conformal offsets on TRACE quantiles (`calibrate`), scored separately from the raw fit.

### Added
- Optional ADOPT controls, paper behaviour by default: averaged-order and tangent-space composition, pushforward centering, damped backfitting with an in-sample descent certificate, isotonic projection of the composed prediction, equal-width binning, and endpoint reflection. Numpy references and parity tests cover each path.
- Audit of leakage, noise, the MISE grid and seeds (`tests/python/test_audit.py`) and method-level experiments (`benchmarks/adopt_audit/`).
- TRACE v0 (`TraceRegressor`): binned cyclic boosting of location, scale and shape channels of the conditional quantile function, with a final isotonic projection, bagging, a residual-gradient pair screen, held-out feature selection, leakage screens and conformal oddness scores. Numpy reference and parity tests. Design in `docs/TRACE_DESIGN.md`, simulations in `docs/TRACE_SIMS.md`, learning page `docs/learn/trace.html`. ADOPT is unchanged and remains a reproduction only.

## [0.1.0] - 2026-10-11

### Architecture
- All numerical methods live in the Rust core. The Python package is a thin PyO3
  binding layer (validation, numpy conversion, sklearn-style API). The numpy reference
  is test-only (`tests/python/reference/`). Enforced by `tests/python/test_architecture.py`.

### Added
- Test-only pure-numpy reference implementation (`tests/python/reference/`) of every numerical routine:
  quantile-grid W2 geometry (PAVA projection, monotone transport maps with tie handling),
  local-linear and global Fréchet weights, ADOPT transport backfitting (W2), additive
  backfitting in flat coordinates, log-Cholesky coordinates.
- Rust core crate `traceot-core` porting all of the above, with no dependencies on the ADOPT path.
- Rust estimator layer: SplitMix64 K-fold split, rayon-parallel bandwidth
  cross-validation, model fit/predict/component/transport, global Fréchet regression, Fréchet R².
- PyO3/maturin extension `traceot._core`; low-level functions in `traceot.primitives`.
- Scikit-learn-style estimators: `ADOPTRegressor` (W2 and log-Cholesky spaces, 5-fold CV
  bandwidth), `GlobalFrechetRegressor`, `FrechetMeanRegressor`, `L2AdditiveQuantileRegressor`.
- Scalable primitives for the future TRACE core (separate from ADOPT): mergeable
  deterministic quantile sketch (chunk-parallel update), u8/u16 histogram binning
  (columnar), rayon-parallel gradient histograms, pinball gradients for
  location-scale quantile models; zero-copy numpy inputs.
- Simulation designs of the ADOPT paper (Section 5, `benchmarks/adopt_paper/dgp.py`) and a reproduction driver with
  fixed seeds; verbatim paper values in `benchmarks/adopt_paper/paper_values.json`.
- Criterion benchmarks and a Python scaling benchmark (1e4 to 1e7 rows, 10 and 100 features).
- Docs: learning page `docs/learn/adopt.html` (self-contained), landing page,
  reproduction notes, results table and benchmarks.
- CI: cargo fmt/clippy/test (with and without rayon), maturin build, pytest, ruff.
