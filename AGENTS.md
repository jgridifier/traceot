# Agent notes for traceot

## Cursor Cloud specific instructions

- **Architecture:** All numerical methods live in the Rust core (`crates/core`). The Python package is a thin PyO3 / maturin binding layer only (`crates/py`, `python/traceot`). Do not put arithmetic on data in Python.
- **No neural nets:** Do not introduce neural networks, deep learning frameworks, or learned black-box models. Prefer glass-box / interpretable methods already in the design docs.
- **Positioning:** Describe the work generically (interpretable distributional modelling for large tabular data). Never mention banks or any industry. Use neutral simulated data only in examples, tests, and docs.
- **v4 / research order:** When implementing or extending v4, follow `docs/theory/V4_IMPL_SPEC.md` in order. Freeze and commit preregistration (`docs/benchmark/PREREG_*.md`) before any results document or scored runs.
- **CI commands** (all should stay green):

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p traceot-core
cargo test -p traceot-core --no-default-features
maturin develop --release   # or: maturin build --release
pytest
ruff check python tests benchmarks examples docs
ruff format --check python tests benchmarks examples docs
```

- **Cloud env:** `.cursor/environment.json` prepares Rust stable (rustfmt, clippy), Python 3.11+, maturin, project deps, and warms `cargo` / maturin caches. Do not start long-running services from that install script.
