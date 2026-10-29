//! Criterion benchmarks for traceot-core.
//!
//! Two families, kept apart on purpose:
//! * `adopt/*`  - faithful ADOPT reproduction path, O(n^2 p q) per backfitting cycle;
//! * `scalable/*` - TRACE primitives (binning, histograms, pinball gradients), O(n) passes.
//!
//! Run: `cargo bench -p traceot-core` (results in target/criterion).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use traceot_core::adopt::{adopt_w2_fit, BackfitConfig};
use traceot_core::scalable::{binning, histogram, pinball, sketch::QuantileSketch};
use traceot_core::trace::{
    explain_v5, fit_trace, mixture_cdf_location, support_grid, ExplainV5Config, TraceConfig,
};
use traceot_core::w2::{pava, quantile_grid, transport_apply};
use traceot_core::Mat;

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 11) as f64) / ((1u64 << 53) as f64)
}

fn normals(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            let (a, b) = (lcg(&mut s).max(1e-300), lcg(&mut s));
            (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
        })
        .collect()
}

fn bench_w2(c: &mut Criterion) {
    let q = 100;
    let y = normals(q, 1);
    c.bench_function("w2/pava_q100", |b| {
        b.iter(|| pava(black_box(&y), None, None))
    });
    let mut a = normals(q, 2);
    a.sort_by(f64::total_cmp);
    let mut t = normals(q, 3);
    t.sort_by(f64::total_cmp);
    c.bench_function("w2/transport_apply_q100", |b| {
        b.iter(|| transport_apply(black_box(&a), black_box(&t), black_box(&a)))
    });
}

fn bench_adopt(c: &mut Criterion) {
    let mut g = c.benchmark_group("adopt");
    g.sample_size(10);
    let q = 100;
    let u = quantile_grid(q);
    for &n in &[100usize, 300, 1000] {
        let mut s = 11u64;
        let mut x = Mat::zeros(n, 3);
        let mut y = Mat::zeros(n, q);
        for i in 0..n {
            let r = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
            x.row_mut(i).copy_from_slice(&r);
            for (k, v) in y.row_mut(i).iter_mut().enumerate() {
                *v = (r[0] - 0.5) + (1.0 + r[1]) * (u[k] - 0.5) + 0.2 * r[2] * u[k] * u[k];
            }
        }
        g.bench_with_input(BenchmarkId::new("fit_p3_q100", n), &n, |b, _| {
            b.iter(|| adopt_w2_fit(&x, &y, &[0.1, 0.1, 0.1], &BackfitConfig::default()).unwrap())
        });
    }
    g.finish();
}

fn bench_scalable(c: &mut Criterion) {
    let mut g = c.benchmark_group("scalable");
    g.sample_size(10);
    for &n in &[100_000usize, 1_000_000] {
        let x = normals(n, 5);
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("sketch_update_k2048", n), &n, |b, _| {
            b.iter(|| {
                let mut s = QuantileSketch::new(2048).unwrap();
                s.update(black_box(&x));
                s
            })
        });
        let mut s = QuantileSketch::new(2048).unwrap();
        s.update(&x);
        let edges = s.bin_edges(255).unwrap();
        g.bench_with_input(BenchmarkId::new("bin_column_u8", n), &n, |b, _| {
            b.iter(|| binning::bin_column::<u8>(black_box(&x), &edges).unwrap())
        });
        let codes: Vec<u8> = binning::bin_column(&x, &edges).unwrap();
        let g0 = normals(n, 6);
        let g1 = normals(n, 7);
        let cols: Vec<&[u8]> = (0..10).map(|_| &codes[..]).collect();
        g.throughput(Throughput::Elements((n * 10) as u64));
        g.bench_with_input(BenchmarkId::new("histograms_10feat_2ch", n), &n, |b, _| {
            b.iter(|| histogram::histograms(black_box(&cols), &[&g0, &g1], 256).unwrap())
        });
        let k = 19;
        let uu: Vec<f64> = (1..=k).map(|i| i as f64 / 20.0).collect();
        let z = normals(k, 8);
        let basis: Vec<f64> = z.iter().flat_map(|&zz| [1.0, zz]).collect();
        let ones = vec![1.0; n];
        let zeros = vec![0.0; n];
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("pinball_grad_k19_m2", n), &n, |b, _| {
            b.iter(|| {
                pinball::pinball_grad_location_scale(&x, &zeros, &ones, &z, &uu, &basis, 2).unwrap()
            })
        });
    }
    g.finish();
}

fn bench_trace(c: &mut Criterion) {
    let mut g = c.benchmark_group("trace");
    g.sample_size(10);
    let n = 20_000usize;
    let p = 8usize;
    let mut s = 19u64;
    let mut x = vec![0.0; n * p];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut row = [0.0; 8];
        for v in &mut row {
            *v = lcg(&mut s);
        }
        x[i * p..(i + 1) * p].copy_from_slice(&row);
        let z = {
            let (a, b) = (lcg(&mut s).max(1e-300), lcg(&mut s));
            (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
        };
        y[i] = 1.2 * row[0] + (0.4 + row[1]) * z;
    }
    let cfg = TraceConfig {
        max_rounds: 3,
        patience: 0,
        val_frac: 0.0,
        max_interactions: 0,
        max_bins: 32,
        u_levels: 19,
        n_channels: 3,
        ..TraceConfig::default()
    };
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function("fit_n20k_p8_r3", |b| {
        b.iter(|| fit_trace(black_box(&x), n, p, black_box(&y), &cfg).unwrap())
    });
    g.finish();
}

fn bench_v5(c: &mut Criterion) {
    let mut g = c.benchmark_group("v5");
    g.sample_size(10);
    for &n in &[2_000usize, 20_000, 200_000] {
        let mu = normals(n, 1);
        let r = normals(n, 2);
        let grid = support_grid(-8.0, 8.0, 256).unwrap();
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("mixture_cdf_location", n), &n, |b, _| {
            b.iter(|| mixture_cdf_location(black_box(&mu), black_box(&r), &grid).unwrap())
        });
    }
    for &n in &[2_000usize, 20_000] {
        let p = 5;
        let xa = normals(n * p, 3);
        let xb: Vec<f64> = normals(n * p, 4).iter().map(|v| v + 0.3).collect();
        let e = normals(2 * n, 5);
        let ya: Vec<f64> = (0..n).map(|i| xa[i * p] + e[i]).collect();
        let yb: Vec<f64> = (0..n).map(|i| xb[i * p] + 1.2 * e[n + i]).collect();
        let cfg = ExplainV5Config::default();
        g.bench_with_input(BenchmarkId::new("explain_v5_p5", n), &n, |b, _| {
            b.iter(|| explain_v5(&xa, n, &ya, &xb, n, &yb, p, &cfg).unwrap())
        });
    }
    g.finish();
}

criterion_group!(
    benches,
    bench_w2,
    bench_adopt,
    bench_scalable,
    bench_trace,
    bench_v5
);
criterion_main!(benches);
