# Changelog

All notable changes to this project are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Status: research, unvalidated.

## [Unreleased]

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
