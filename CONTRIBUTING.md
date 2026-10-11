# Contributing to traceot

traceot is a personal research project (status: research, unvalidated). Issues and
pull requests are welcome. Please read the rules below first. They are binding for
every change.

## Architecture rule: all numerical methods live in Rust

* **Everything numerical is in the Rust core** (`crates/core`, crate `traceot-core`).
  That includes fitting, binning, transport and quantile maths, the ADOPT and TRACE
  algorithms, cross-validation, checks and scoring.
* **The Python package is a thin PyO3 binding layer only** (`crates/py` → `traceot._core`,
  plus `python/traceot`). It handles input validation, numpy conversion and a scikit-learn-style
  API. It contains **no numerical logic**. If a change needs arithmetic on data, write
  it in Rust and expose it.
* **The pure-numpy reference implementation lives only under `tests/python/reference/`.**
  It is written for readability, never imported by the runtime package, and used
  solely to verify the Rust results.
* Guard rails: `tests/python/test_architecture.py` fails if numerical numpy calls (for example
  `np.mean`, `np.linalg`, `@`) appear in `python/traceot`, or if the package imports the
  reference.

## Correctness: a reference first, then Rust that matches it

1. Write (or extend) the numpy reference in `tests/python/reference/`, with tests of its
   mathematical properties (`tests/python/test_reference_*.py`).
2. Implement the routine in `crates/core` with Rust unit tests.
3. Add a parity test (`tests/python/test_parity.py`, `test_scalable.py`,
   `test_estimators.py`) showing that Rust and the reference agree to a tight tolerance
   (typically `atol=1e-10`, or exact equality for integer outputs such as bin codes).

## Scale and speed rank equally with correctness

The TRACE path targets millions of rows and about 100 features:

* histogram binning (`u8`/`u16` codes, columnar `(p, n)` layout), O(n) passes per
  boosting round, and no O(n²) work in the main path;
* streaming, mergeable quantile sketches instead of full sorts;
* rayon parallelism behind the default `parallel` feature (the crate must still build
  with `--no-default-features`, for WASM later);
* zero-copy numpy inputs where possible, with the GIL released during computation.

**Profile before optimising.** Add or extend a criterion benchmark (`crates/core/benches`)
and report wall time and peak memory in `docs/BENCHMARKS.md`
(`benchmarks/scale_bench.py`). ADOPT is O(n²·p·q) by design and stays separate from
the scalable core.

## Reproductions

Reproductions of published methods are written from the paper only. Paper numbers are copied
verbatim and labelled. Every ambiguity, and the choice made, goes into
`docs/REPRODUCTION_NOTES.md`. Never tune towards a paper's numbers.

## Positioning and data

Describe the project generically ("interpretable distributional modelling for large
tabular data"). Examples, tests and docs use neutral simulated data only.

## Checks (all run in CI)

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p traceot-core && cargo test -p traceot-core --no-default-features
maturin develop --release && pytest
ruff check python tests benchmarks examples docs
ruff format --check python tests benchmarks examples docs
```

Commit in small, meaningful steps, and update `CHANGELOG.md`.
