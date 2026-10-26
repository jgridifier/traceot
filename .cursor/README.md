# Cursor Cloud Agent environment (traceot)

`environment.json` points Cloud Agent Builds at `install.sh`, which:

1. Installs Rust stable with clippy + rustfmt (via rustup if needed).
2. Creates `.venv`, installs maturin / pytest / scipy / numpy / ruff.
3. Runs `maturin develop --release` so `import traceot` works.
4. Warms `cargo test -p traceot-core` and compiles benches (`--no-run`).

Local box: `bash .cursor/install.sh` from the repo root (or let the Cloud Agent run
it on boot). After install: `cargo test -p traceot-core && pytest -q`.

Does not touch `docs/learn/nightly*`. No secrets. Neutral research tooling only.
