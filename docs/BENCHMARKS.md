# Benchmarks

Status: research, unvalidated. Every number here comes from a shared cloud VM (8 vCPU, x86_64,
about 15 GB RAM, other jobs possibly running). Treat them as orders of magnitude, not as
precise measurements. Regenerate with the commands at the end.

**Targets.** The TRACE core is meant for millions of rows and about 100 features. Speed and memory
are treated as first-class requirements, alongside correctness. Every Rust routine must still match
the numpy reference (`tests/python/test_parity.py`, `tests/python/test_scalable.py`).

## Complexity: ADOPT versus the TRACE core

| Path | Per iteration | Memory | Use |
|---|---|---|---|
| ADOPT (paper method) | `O(n² · p · q)` (paper Supplement C) | `O(n·p·q)` for the component curves, plus an `n × n` weight matrix per predictor at a time | faithful baseline. Fine up to a few thousand distributions |
| TRACE primitives (this repo) | `O(n · p)` per boosting round (histogram pass), plus `O(n · K)` pinball gradients | `n · p` bytes of u8 codes (u16 when more than 254 bins), `n · M` floats of gradients, never `n × K` | the scalable core for unit-level data |

Design of the TRACE core (`crates/core/src/scalable/`):

* **Streaming, mergeable quantile sketch** (deterministic compactor, KLL/MRL style) for bin edges.
  One pass, bounded memory, mergeable across chunks or threads (`update_chunked`).
* **Histogram binning** to `u8`/`u16` codes, **columnar** `(p, n)` layout, code 0 reserved for NaN.
* **Gradient histograms** run in parallel over features with rayon (feature `parallel`, on by
  default; build with `--no-default-features` for single-threaded or WASM targets).
* **Pinball gradients** for `Q(u|x) = loc(x) + scale(x)·z(u)`, accumulated directly into the
  `M` basis channels: `O(n·K)` time, `O(n·M)` memory.
* **Zero-copy numpy inputs**: contiguous arrays are borrowed as slices, and outputs are moved into numpy without copying.
  The GIL is released during computation.

## Profile first: criterion micro-benchmarks (Rust, `cargo bench -p traceot-core`)

| Benchmark | Result |
|---|---|
| `w2/pava` q = 100 | 387 ns |
| `w2/transport_apply` q = 100 | 932 ns |
| `adopt/fit_p3_q100` n = 100 | 12.1 ms |
| `adopt/fit_p3_q100` n = 300 | 45.4 ms |
| `adopt/fit_p3_q100` n = 1000 | 299 ms (≈ n² growth) |
| `scalable/sketch_update` (sequential) | ≈ 47 M values/s per core |
| `scalable/bin_column_u8` | 231 M values/s at 1e5, 822 M values/s at 1e6 |
| `scalable/histograms` 10 features × 2 channels | ≈ 2.5 G elements/s |
| `scalable/pinball` K = 19, M = 2 | 20.7 M rows/s at 1e5, 42 M rows/s at 1e6 |

What profiling changed: the sequential sketch (≈ 47 M values/s) was the slowest step in a full
pass. We therefore added the chunk-parallel `update_chunked`, which sketches chunks in parallel and
then merges them. It has an exact-parity reference in numpy. We have not yet optimised anything that
the profile did not point to.

## End-to-end scaling (Python → Rust, `benchmarks/scale_bench.py`)

Each configuration runs in a fresh process, so the peak resident memory (`ru_maxrss`) belongs to it alone.
*Setup* = sketch + bin all features (done once per fit). *Round* = pinball gradients + gradient
histograms for all features (done once per boosting round). Data are neutral simulated columns,
generated one at a time (the script never builds an `n × p` float matrix). Times are wall-clock seconds.

<!-- SCALE_TABLE -->
| rows | features | setup: sketch | setup: bin | round: grad | round: hist | round total | peak RSS (MB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e+04 | 10 | 0.004 | 0.001 | 0.001 | 0.000 | 0.001 | 37 |
| 1e+04 | 100 | 0.030 | 0.008 | 0.001 | 0.001 | 0.002 | 39 |
| 1e+05 | 10 | 0.028 | 0.007 | 0.009 | 0.001 | 0.010 | 45 |
| 1e+05 | 100 | 0.244 | 0.067 | 0.008 | 0.004 | 0.012 | 53 |
| 1e+06 | 10 | 0.232 | 0.024 | 0.048 | 0.005 | 0.052 | 114 |
| 1e+06 | 100 | 2.459 | 0.258 | 0.053 | 0.035 | 0.087 | 201 |
| 1e+07 | 10 | 0.609 | 0.161 | 0.542 | 0.055 | 0.597 | 595 |
| 1e+07 | 100 | 6.760 | 1.879 | 0.423 | 0.326 | 0.749 | 1475 |

Machine: 8 logical CPUs, x86_64, Python 3.13.5, rayon threads: default (all cores), run 2026-10-11.
<!-- /SCALE_TABLE -->

**Reading the table.**

* At 10⁷ rows × 100 features, one full boosting round (gradients plus 100 gradient histograms)
  takes **0.75 s**. Peak memory is **1.5 GB**, about 1 GB of which is the `u8` code matrix
  (`n · p` bytes). The one-off setup (sketch plus binning for all features) takes 8.6 s.
* Round time grows linearly with rows, from 10⁵ to 10⁷.
  ADOPT, by contrast, grows quadratically (12 ms at n = 100, 299 ms at n = 1000; see the criterion table above).
* **Next profile-driven fix:** at 10⁶ rows the sketch step runs at roughly sequential speed
  (≈ 43 M values/s per feature), because the binding's default chunk (2²⁰ values) gives
  one chunk per column. At 10⁷ rows it parallelises (≈ 160 M values/s). Sketching several
  features in parallel, or using a smaller chunk, should cut setup time at 10⁶ rows. This is not done yet.


## ADOPT audit paths (release build)

Measured by `python benchmarks/adopt_audit/run_audit.py` on this machine: 4 CPUs, x86_64, Python 3.12.3, release wheel, 11 October 2026. The task is one backfitting fit, p = 2, q = 12, 3 iterations, fixed bandwidth, smooth location truth. `bins=32` is the equal-width histogram smoother. Holdout W2² on a 5×5 midpoint grid stays within about 5% of the dense fit at every n below (both near 2.7×10⁻⁴).

| n | dense seconds | binned seconds | dense / binned |
|---:|---:|---:|---:|
| 200 | 0.00288 | 0.00146 | 2.0 |
| 400 | 0.00883 | 0.00213 | 4.1 |
| 800 | 0.0257 | 0.00363 | 7.1 |
| 1600 | 0.0987 | 0.00739 | 13.4 |

From n = 200 to n = 1600 (8× rows) dense time grew 34× and binned time grew 5.1×. Step to step, dense growth from 800 to 1600 was 3.83 (quadratic scaling is 4) and binned growth was 2.03 (linear scaling is 2). Full JSON: `benchmarks/results/adopt_audit.json`.

## Re-running

```bash
cargo bench -p traceot-core                  # criterion; HTML report in target/criterion
python benchmarks/scale_bench.py             # 1e4..1e7 rows x 10/100 features, about 1.6 GB peak
python docs/build_docs.py                    # refreshes the table above and the scaling figure
python benchmarks/adopt_audit/run_audit.py   # order, centering, convergence, monotone, scaling, boundary
```
