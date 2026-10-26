#!/usr/bin/env bash
# Cursor Cloud Agent Builds / box bootstrap for traceot (TRACE v4).
# Idempotent. No application-domain setup — toolchain only.
set -euo pipefail
cd "$(dirname "$0")/.."

export CARGO_TERM_COLOR=always
export RUSTFLAGS="${RUSTFLAGS:-}"

echo "[traceot] install: Rust toolchain (stable + clippy + rustfmt)"
if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi
rustup toolchain install stable --profile minimal --component clippy,rustfmt
rustup default stable

echo "[traceot] install: Python venv + maturin + test deps"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="${TRACEOT_VENV:-.venv}"
if [[ ! -d "$VENV" ]]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install maturin pytest scipy numpy ruff

echo "[traceot] install: build release extension into the venv"
maturin develop --release

echo "[traceot] install: warm cargo test (parallel feature)"
cargo test -p traceot-core --tests -- --list >/dev/null || cargo test -p traceot-core -q || true
# Compile benches so later criterion runs do not cold-start the world
cargo bench -p traceot-core --no-run -q || true

echo "[traceot] install: done"
rustc --version
cargo --version
python -V
maturin --version
