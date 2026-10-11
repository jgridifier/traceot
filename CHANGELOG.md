# Changelog

All notable changes to this project are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Status: research, unvalidated.

## [Unreleased]

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
