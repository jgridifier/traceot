//! Model mixture law F_[s,t](y) = E_{x~F_s} F_t(y|x) from monotone conditional quantiles.
//!
//! Spec: V4_IMPL_SPEC § Step 4 `mixture.rs`. Evaluate the average CDF on a y-grid by
//! inverting each row's monotone Q via interpolation (chunked, optionally parallel),
//! then invert the mixture CDF to Q_[s,t] on Gauss–Legendre nodes and project to K
//! Legendre channels. Hybrids (s,t) reuse the same path with period-s covariates and
//! period-t conditional curves.

#![allow(dead_code, clippy::too_many_arguments)]

use crate::trace::legendre::{gauss_legendre_01, pstar_all};
use crate::trace::shift::SplitMix64;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Rows per chunk for the parallel CDF average (deterministic merge order).
const CHUNK: usize = 4096;

/// Configuration for mixture evaluation.
#[derive(Clone, Debug)]
pub struct MixtureConfig {
    /// Number of y-grid points for F_M (default 256).
    pub n_y: usize,
    /// Subsample cap on rows (deterministic); None = all.
    pub max_rows: Option<usize>,
    /// Number of Legendre channels to project onto.
    pub k_channels: usize,
    /// GL nodes for the reported Q_[s,t].
    pub n_quad: usize,
    /// Seed for deterministic subsample when n > max_rows.
    pub subsample_seed: u64,
}

impl Default for MixtureConfig {
    fn default() -> Self {
        Self {
            n_y: 256,
            max_rows: Some(200_000),
            k_channels: 6,
            n_quad: 64,
            subsample_seed: 0,
        }
    }
}

/// Model mixture law: quantile on GL nodes + channel coefficients.
#[derive(Clone, Debug)]
pub struct MixtureLaw {
    pub u: Vec<f64>,
    pub q: Vec<f64>,
    pub channels: Vec<f64>,
    pub y_grid: Vec<f64>,
    pub f_grid: Vec<f64>,
    /// Number of rows actually averaged (after subsample).
    pub n_used: usize,
}

/// Linear interpolate Q(u) given monotone nodes.
pub fn interp_q(u_nodes: &[f64], q_nodes: &[f64], u: f64) -> f64 {
    let n = u_nodes.len();
    if n == 0 {
        return 0.0;
    }
    if u <= u_nodes[0] {
        return q_nodes[0];
    }
    if u >= u_nodes[n - 1] {
        return q_nodes[n - 1];
    }
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if u_nodes[mid] <= u {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = (u - u_nodes[lo]) / (u_nodes[hi] - u_nodes[lo]).max(1e-18);
    q_nodes[lo] + t * (q_nodes[hi] - q_nodes[lo])
}

/// Invert monotone Q at level y → u = F(y).
pub fn cdf_from_q(u_nodes: &[f64], q_nodes: &[f64], y: f64) -> f64 {
    let n = u_nodes.len();
    if n == 0 {
        return 0.0;
    }
    if y <= q_nodes[0] {
        return 0.0;
    }
    if y >= q_nodes[n - 1] {
        return 1.0;
    }
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if q_nodes[mid] <= y {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = (y - q_nodes[lo]) / (q_nodes[hi] - q_nodes[lo]).max(1e-18);
    u_nodes[lo] + t * (u_nodes[hi] - u_nodes[lo])
}

fn invert_cdf(y_grid: &[f64], f_grid: &[f64], u: f64) -> f64 {
    let n = y_grid.len();
    if u <= f_grid[0] {
        return y_grid[0];
    }
    if u >= f_grid[n - 1] {
        return y_grid[n - 1];
    }
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if f_grid[mid] <= u {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = (u - f_grid[lo]) / (f_grid[hi] - f_grid[lo]).max(1e-18);
    y_grid[lo] + t * (y_grid[hi] - y_grid[lo])
}

/// Deterministic subsample of row indices when n exceeds the cap (SplitMix64).
pub fn subsample_indices(n: usize, max_rows: usize, seed: u64) -> Vec<usize> {
    if n <= max_rows {
        return (0..n).collect();
    }
    let mut rng = SplitMix64::new(seed ^ MIX_SUB);
    let mut idx: Vec<usize> = (0..n).collect();
    // Partial Fisher–Yates for the first max_rows
    for i in 0..max_rows {
        let j = i + rng.index(n - i);
        idx.swap(i, j);
    }
    idx.truncate(max_rows);
    idx.sort_unstable(); // stable merge order across machines
    idx
}

const MIX_SUB: u64 = 0x4D49_585F_5355_4231; // "MIX_SUB1"

fn accumulate_chunk(
    u_nodes: &[f64],
    q_rows: &[f64],
    q: usize,
    row_idx: &[usize],
    start: usize,
    end: usize,
    y_grid: &[f64],
    row_weights: Option<&[f64]>,
) -> (Vec<f64>, f64) {
    let n_y = y_grid.len();
    let mut local = vec![0.0; n_y];
    let mut wsum = 0.0;
    for &i in &row_idx[start..end] {
        let wi = row_weights.map(|w| w[i]).unwrap_or(1.0);
        if !(wi.is_finite() && wi >= 0.0) {
            continue;
        }
        wsum += wi;
        let row = &q_rows[i * q..(i + 1) * q];
        for (g, &y) in y_grid.iter().enumerate() {
            local[g] += wi * cdf_from_q(u_nodes, row, y);
        }
    }
    (local, wsum)
}

fn average_cdf(
    u_nodes: &[f64],
    q_rows: &[f64],
    n: usize,
    row_weights: Option<&[f64]>,
    y_grid: &[f64],
    cfg: &MixtureConfig,
) -> (Vec<f64>, usize) {
    let q = u_nodes.len();
    let max_rows = cfg.max_rows.unwrap_or(n);
    let row_idx = subsample_indices(n, max_rows, cfg.subsample_seed);
    let n_use = row_idx.len();
    let n_y = y_grid.len();

    let starts: Vec<usize> = (0..n_use).step_by(CHUNK).collect();

    #[cfg(feature = "parallel")]
    let partials: Vec<(Vec<f64>, f64)> = starts
        .par_iter()
        .map(|&s| {
            let e = (s + CHUNK).min(n_use);
            accumulate_chunk(u_nodes, q_rows, q, &row_idx, s, e, y_grid, row_weights)
        })
        .collect();

    #[cfg(not(feature = "parallel"))]
    let partials: Vec<(Vec<f64>, f64)> = starts
        .iter()
        .map(|&s| {
            let e = (s + CHUNK).min(n_use);
            accumulate_chunk(u_nodes, q_rows, q, &row_idx, s, e, y_grid, row_weights)
        })
        .collect();

    let mut f_grid = vec![0.0; n_y];
    let mut wsum = 0.0;
    for (local, ws) in partials {
        wsum += ws;
        for g in 0..n_y {
            f_grid[g] += local[g];
        }
    }
    let inv = 1.0 / wsum.max(1e-18);
    for f in &mut f_grid {
        *f *= inv;
    }
    for g in 1..n_y {
        if f_grid[g] < f_grid[g - 1] {
            f_grid[g] = f_grid[g - 1];
        }
    }
    if let Some(f0) = f_grid.first_mut() {
        *f0 = f0.clamp(0.0, 1.0);
    }
    if let Some(fl) = f_grid.last_mut() {
        *fl = fl.clamp(0.0, 1.0);
    }
    (f_grid, n_use)
}

fn project_channels(u: &[f64], q_mix: &[f64], weights: &[f64], k: usize) -> Vec<f64> {
    let mut channels = vec![0.0; k];
    for i in 0..u.len() {
        let (pk, _) = pstar_all(u[i], k);
        for j in 0..k {
            channels[j] += weights[i] * q_mix[i] * pk[j];
        }
    }
    for (j, c) in channels.iter_mut().enumerate() {
        *c *= (2 * j + 1) as f64;
    }
    channels
}

/// Build the model mixture from per-row conditional quantile curves.
pub fn mixture_from_conditionals(
    u_nodes: &[f64],
    q_rows: &[f64],
    n: usize,
    y_lo: f64,
    y_hi: f64,
    cfg: &MixtureConfig,
) -> MixtureLaw {
    mixture_from_conditionals_weighted(u_nodes, q_rows, n, None, y_lo, y_hi, cfg)
}

/// Like [`mixture_from_conditionals`] with optional non-negative row weights.
pub fn mixture_from_conditionals_weighted(
    u_nodes: &[f64],
    q_rows: &[f64],
    n: usize,
    row_weights: Option<&[f64]>,
    y_lo: f64,
    y_hi: f64,
    cfg: &MixtureConfig,
) -> MixtureLaw {
    assert_eq!(u_nodes.len() * n, q_rows.len());
    if let Some(w) = row_weights {
        assert_eq!(w.len(), n);
    }
    let n_y = cfg.n_y.max(8);
    let mut y_grid = Vec::with_capacity(n_y);
    for i in 0..n_y {
        let t = i as f64 / (n_y - 1) as f64;
        y_grid.push(y_lo + t * (y_hi - y_lo));
    }
    let (f_grid, n_used) = average_cdf(u_nodes, q_rows, n, row_weights, &y_grid, cfg);

    let gl = gauss_legendre_01(cfg.n_quad.max(8)).expect("GL order > 0");
    let mut q_mix = Vec::with_capacity(gl.nodes.len());
    for &u in &gl.nodes {
        q_mix.push(invert_cdf(&y_grid, &f_grid, u));
    }
    let k = cfg.k_channels.max(1);
    let channels = project_channels(&gl.nodes, &q_mix, &gl.weights, k);

    MixtureLaw {
        u: gl.nodes,
        q: q_mix,
        channels,
        y_grid,
        f_grid,
        n_used,
    }
}

/// Convenience: mixture when every row shares the same Q (degenerate X).
pub fn mixture_identical_rows(
    u_nodes: &[f64],
    q_row: &[f64],
    y_lo: f64,
    y_hi: f64,
    cfg: &MixtureConfig,
) -> MixtureLaw {
    mixture_from_conditionals(u_nodes, q_row, 1, y_lo, y_hi, cfg)
}

/// Hybrid mixture: covariates from period `s` (row weights / subsample on `q_rows_s`
/// indexing), structure from period `t` curves. `q_rows` must be Q_t(u|x_i) for
/// x_i drawn from F_s (already evaluated on the s-sample).
pub fn mixture_hybrid(
    u_nodes: &[f64],
    q_rows_on_s: &[f64],
    n_s: usize,
    y_lo: f64,
    y_hi: f64,
    cfg: &MixtureConfig,
) -> MixtureLaw {
    mixture_from_conditionals(u_nodes, q_rows_on_s, n_s, y_lo, y_hi, cfg)
}

/// Barycenter (Vincent average of quantile functions) on the same GL nodes.
pub fn barycenter_from_conditionals(
    u_nodes: &[f64],
    q_rows: &[f64],
    n: usize,
    cfg: &MixtureConfig,
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(u_nodes.len() * n, q_rows.len());
    let q = u_nodes.len();
    let max_rows = cfg.max_rows.unwrap_or(n);
    let row_idx = subsample_indices(n, max_rows, cfg.subsample_seed ^ 0xBA51_0001);
    let n_use = row_idx.len().max(1);
    let gl = gauss_legendre_01(cfg.n_quad.max(8)).expect("GL");
    let mut qbar = vec![0.0; gl.nodes.len()];
    for &i in &row_idx {
        let row = &q_rows[i * q..(i + 1) * q];
        for (g, &u) in gl.nodes.iter().enumerate() {
            qbar[g] += interp_q(u_nodes, row, u);
        }
    }
    let inv = 1.0 / n_use as f64;
    for v in &mut qbar {
        *v *= inv;
    }
    let channels = project_channels(&gl.nodes, &qbar, &gl.weights, cfg.k_channels.max(1));
    (qbar, channels)
}

/// Mixing term M = Q_mixture − Q_barycenter on GL nodes (same cfg / subsample seed).
pub fn mixing_term(
    u_nodes: &[f64],
    q_rows: &[f64],
    n: usize,
    y_lo: f64,
    y_hi: f64,
    cfg: &MixtureConfig,
) -> (MixtureLaw, Vec<f64>, Vec<f64>) {
    let mix = mixture_from_conditionals(u_nodes, q_rows, n, y_lo, y_hi, cfg);
    let (qbar, cbar) = barycenter_from_conditionals(u_nodes, q_rows, n, cfg);
    let m: Vec<f64> = mix.q.iter().zip(&qbar).map(|(a, b)| a - b).collect();
    let cm: Vec<f64> = mix.channels.iter().zip(&cbar).map(|(a, b)| a - b).collect();
    (mix, m, cm)
}

/// Pooled y-range from empirical quantiles of stacked outcomes (0.1%–99.9%).
pub fn pooled_y_range(y_a: &[f64], y_b: &[f64]) -> (f64, f64) {
    let mut v: Vec<f64> = y_a
        .iter()
        .chain(y_b.iter())
        .copied()
        .filter(|z| z.is_finite())
        .collect();
    if v.is_empty() {
        return (-1.0, 1.0);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    let lo = v[((0.001 * (n - 1) as f64).round() as usize).min(n - 1)];
    let hi = v[((0.999 * (n - 1) as f64).round() as usize).min(n - 1)];
    if hi > lo {
        (lo, hi)
    } else {
        (lo - 1.0, hi + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm_ppf(u: f64) -> f64 {
        let a = [
            -3.969683028665376e+01,
            2.209460984245205e+02,
            -2.759285104469687e+02,
            1.383577459334128e+02,
            -3.066479806614716e+01,
            2.506628277459239e+00,
        ];
        let b = [
            -5.447609879822406e+01,
            1.615858368580409e+02,
            -1.556989798598866e+02,
            6.680131188771972e+01,
            -1.328068155288572e+01,
        ];
        let c = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e+00,
            -2.549732539343734e+00,
            4.374664141464968e+00,
            2.938163982698783e+00,
        ];
        let d = [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e+00,
            3.754408661907416e+00,
        ];
        let plow = 0.02425;
        let phigh = 1.0 - plow;
        if u < plow {
            let q = (-2.0 * u.ln()).sqrt();
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
        }
        if u > phigh {
            let q = (-2.0 * (1.0 - u).ln()).sqrt();
            return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
        }
        let q = u - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    }

    #[test]
    fn identity_when_x_degenerate() {
        let u: Vec<f64> = (0..128).map(|i| (i as f64 + 0.5) / 128.0).collect();
        let q: Vec<f64> = u.iter().map(|&uu| 2.0 + 1.5 * norm_ppf(uu)).collect();
        let qmin = q.iter().cloned().fold(f64::INFINITY, f64::min);
        let qmax = q.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let pad = 0.5 * (qmax - qmin);
        let cfg = MixtureConfig {
            n_y: 4096,
            max_rows: None,
            k_channels: 4,
            n_quad: 64,
            subsample_seed: 0,
        };
        let law = mixture_identical_rows(&u, &q, qmin - pad, qmax + pad, &cfg);
        let mut max_err = 0.0f64;
        for i in 0..law.u.len() {
            let uu = law.u[i];
            if !(0.02..=0.98).contains(&uu) {
                continue;
            }
            let truth = interp_q(&u, &q, uu);
            max_err = max_err.max((law.q[i] - truth).abs());
        }
        assert!(max_err < 1e-3, "max_err={max_err}");
    }

    #[test]
    fn gaussian_location_mixture_closed_form() {
        let sigma = 1.0_f64;
        let v = 0.25_f64;
        let s_mu = v.sqrt();
        let u: Vec<f64> = (0..128).map(|i| (i as f64 + 0.5) / 128.0).collect();
        let n_mu = 801usize;
        let mut rows = Vec::new();
        let mut weights = Vec::new();
        let z_lo = -5.0_f64;
        let z_hi = 5.0_f64;
        let dz = (z_hi - z_lo) / (n_mu - 1) as f64;
        for i in 0..n_mu {
            let z = z_lo + dz * i as f64;
            let mu = s_mu * z;
            let dens = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
            weights.push(dens * dz);
            let q_row: Vec<f64> = u.iter().map(|&uu| mu + sigma * norm_ppf(uu)).collect();
            rows.extend_from_slice(&q_row);
        }
        let cfg = MixtureConfig {
            n_y: 4096,
            max_rows: None,
            k_channels: 4,
            n_quad: 64,
            subsample_seed: 0,
        };
        let s_eff = (sigma * sigma + v).sqrt();
        let law = mixture_from_conditionals_weighted(
            &u,
            &rows,
            n_mu,
            Some(&weights),
            -6.0 * s_eff,
            6.0 * s_eff,
            &cfg,
        );
        let mut max_err = 0.0f64;
        for i in 0..law.u.len() {
            let uu = law.u[i];
            if !(0.05..=0.95).contains(&uu) {
                continue;
            }
            let truth = s_eff * norm_ppf(uu);
            max_err = max_err.max((law.q[i] - truth).abs());
        }
        assert!(max_err < 1e-3, "max_err={max_err}");
    }

    #[test]
    fn subsample_is_deterministic_and_sorted() {
        let a = subsample_indices(10_000, 100, 42);
        let b = subsample_indices(10_000, 100, 42);
        assert_eq!(a, b);
        assert_eq!(a.len(), 100);
        assert!(a.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn mixing_vanishes_for_degenerate_x() {
        let u: Vec<f64> = (0..64).map(|i| (i as f64 + 0.5) / 64.0).collect();
        let q: Vec<f64> = u.iter().map(|&uu| norm_ppf(uu)).collect();
        let cfg = MixtureConfig {
            n_y: 2048,
            max_rows: None,
            k_channels: 4,
            n_quad: 32,
            subsample_seed: 1,
        };
        let (_mix, m, cm) = mixing_term(&u, &q, 1, -5.0, 5.0, &cfg);
        let max_m = m.iter().map(|z| z.abs()).fold(0.0, f64::max);
        let max_c = cm.iter().map(|z| z.abs()).fold(0.0, f64::max);
        assert!(max_m < 5e-3, "M should be ~0; max={max_m}");
        assert!(max_c < 5e-3, "channel M should be ~0; max={max_c}");
    }
}
