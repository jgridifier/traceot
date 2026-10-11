# traceot

[![CI](https://github.com/jgridifier/traceot/actions/workflows/ci.yml/badge.svg)](https://github.com/jgridifier/traceot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-informational)](https://jgridifier.github.io/traceot/)
![Status: research, unvalidated](https://img.shields.io/badge/status-research%2C%20unvalidated-orange)

**Interpretable distributional modelling for large tabular data, built on optimal transport.**
All numerics in a Rust core, thin scikit-learn-style Python API, test-only numpy reference.

> **Status: research, unvalidated.** This is a personal research project. APIs will change,
> and nothing here has been validated beyond the simulations documented in `docs/`.
> Do not rely on it for decisions.

## What is here

| Part | What it is | State |
|---|---|---|
| **ADOPT** | Faithful re-implementation of *ADOPT: Additive Optimal Transport Regression* (Song & Müller, AISTATS 2026, [PMLR 300:1558–1566](https://proceedings.mlr.press/v300/song26c.html)), written from the paper alone (no public code). Wasserstein-2 and log-Cholesky (SPD) responses. | Milestone 1: done; simulation study reproduced, partly matching (see below) |
| **TRACE** | Our own **unpublished** method: a binned additive model of the conditional quantile function (location, scale and shape channels), with oddness scores and leakage/placebo checks. Scales to millions of rows. | v0 in `TraceRegressor`. Design: [docs/TRACE_DESIGN.md](docs/TRACE_DESIGN.md). Simulations: [docs/TRACE_SIMS.md](docs/TRACE_SIMS.md) |

Learn the ideas:

- **[docs/learn/adopt.html](https://jgridifier.github.io/traceot/learn/adopt.html)**: OT regression, additive transport, ADOPT's components, our reproduction.
- **[docs/learn/trace.html](https://jgridifier.github.io/traceot/learn/trace.html)**: EBMs, TRACE versus a mean model, how the fit works, checks and oddness.

## Reproduction at a glance

Details: [docs/ADOPT_RESULTS.md](docs/ADOPT_RESULTS.md) (paper numbers verbatim, next to ours) and
[docs/REPRODUCTION_NOTES.md](docs/REPRODUCTION_NOTES.md) (every ambiguity and choice).

Simulation study of the paper, re-run with B = 200 fixed-seed replicates per setting:

| | Qualitative claim (ADOPT < GF < FM, and ADOPT improves with n) | Numbers |
|---|---|---|
| Table 1, Case I (W2) | reproduced | close: ADOPT n=300 0.179 vs paper 0.181 (×10⁻³); n=100 and the baselines within 5–25% |
| Table 1, Case II (W2) | reproduced | GF and FM within 1–11%; **our ADOPT is 3–6× lower than the paper's** |
| Table 2 (SPD, log-Cholesky) | reproduced | **levels do not match** (ours 10–50× lower for ADOPT/GF; FM matches only for Case I, m=10) |

Likely causes (unstated bandwidth details, response noise, SPD design reading) are discussed in the notes.
Nothing was tuned towards the paper's numbers.

## Install (from source)

Requires Rust ≥ 1.80 and Python ≥ 3.10.

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy
maturin develop --release          # builds the Rust extension traceot._core
pip install -e ".[dev,sim,docs]"   # optional extras
```

```python
import numpy as np
from traceot import ADOPTRegressor
from traceot import quantile_grid

# X: (n, p) inputs in [0, 1]; Y: (n, q) response quantile functions on quantile_grid(q)
model = ADOPTRegressor(space="wasserstein", bandwidth="cv", random_state=0).fit(X, Y)
Q_pred = model.predict(X_new)          # (m, q) predicted quantile functions
model.component(j=0, x=grid)           # glass-box: effect of input 0 alone
```

See [examples/adopt_quickstart.py](examples/adopt_quickstart.py) and [examples/trace_quickstart.py](examples/trace_quickstart.py) (neutral simulated data).

```python
from traceot import TraceRegressor

# y is a scalar outcome. predict() returns quantile functions, shape (n, u_levels).
trace = TraceRegressor(max_rounds=40, seed=0).fit(X, y)
Q = trace.predict(X_new)
trace.effect(0)   # location, scale and shape steps for feature 0
```

## Layout

```
crates/core/            Rust core (traceot-core): ALL numerics. w2, smoother, adopt, additive,
                        logchol, estimators (CV, scoring), scalable/, trace/ (TRACE v0)
crates/py/              PyO3 bindings -> traceot._core
python/traceot/         thin Python layer: validation, numpy conversion, sklearn-style estimators
tests/python/           pytest: reference properties, Rust/reference parity, estimators, architecture
tests/python/reference/ pure-numpy reference (test-only, used to verify Rust)
benchmarks/             ADOPT reproduction, scaling benchmark, benchmarks/trace_sims/
docs/               GitHub Pages: index.html, learn/adopt.html, learn/trace.html, reports
examples/           runnable examples on simulated data
```

## Architecture rule

**All core methods live in Rust.** Fitting, binning, transport and quantile maths, the ADOPT and
TRACE algorithms, checks and scoring are all in `crates/core`. The Python package is a
**thin PyO3 binding layer only**: input validation, numpy conversion and a scikit-learn-style
API, with no numerical logic. The pure-numpy reference implementation lives **only under
`tests/python/reference/`**. It is never imported at runtime and is used solely to verify the
Rust results. A guard test (`tests/python/test_architecture.py`) enforces this. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## Design principles

* **A reference implementation comes first.** Every numerical routine is first written in readable numpy
  (test-only, `tests/python/reference/`). The Rust core must match it, and parity tests enforce that.
* **Scale ranks with correctness.** The TRACE path uses histogram binning (u8/u16, columnar), O(n) passes,
  rayon parallelism, streaming mergeable quantile sketches and zero-copy numpy inputs.
  ADOPT is O(n²·p·q) per backfitting cycle and is kept separate as a baseline.
  Numbers: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).
* **Honest reporting.** Paper numbers are shown verbatim. Mismatches are explained, not tuned away.

## Development

```bash
cargo test --workspace                    # Rust unit tests (also: --no-default-features, without rayon)
cargo clippy --workspace -- -D warnings
cargo bench -p traceot-core               # criterion
pytest                                    # Python tests (needs the built extension)
ruff check python tests benchmarks examples docs
python benchmarks/adopt_paper/reproduce.py --reps 200   # paper simulations (~30 min on 7 cores)
python benchmarks/scale_bench.py          # 1e4..1e7 rows x 10/100 features
python benchmarks/trace_sims/run_sims.py  # TRACE S1-S4, criteria in docs/TRACE_SIMS.md
python benchmarks/trace_sims/compare_ebm.py  # optional; needs the ebm extra
python docs/build_docs.py                 # regenerate figures, tables and docs/learn/adopt.html
python docs/build_trace_learn.py          # regenerate docs/learn/trace.html
```

## Citation

If you use the ADOPT part, cite the original paper:

> Song, W. & Müller, H.-G. (2026). ADOPT: Additive Optimal Transport Regression.
> *Proceedings of AISTATS 2026*, PMLR 300:1558–1566. https://proceedings.mlr.press/v300/song26c.html

## License

MIT. See [LICENSE](LICENSE).
