//! Model mixture law F_[s,t](y) = E_{x~F_s} F_t(y|x) from monotone conditional quantiles.
//!
//! Spec: V4_IMPL_SPEC § Step 4 `mixture.rs`. Evaluate the average CDF on a y-grid by
//! inverting each row's monotone Q via interpolation, then invert the mixture CDF to
//! Q_[s,t] on Gauss–Legendre nodes and project to K Legendre channels.

#![allow(dead_code)]
use crate::trace::legendre::{gauss_legendre_01, pstar_all};

/// Configuration for mixture evaluation.
#[derive(Clone, Debug)]
pub struct MixtureConfig {
    /// Number of y-grid points for F_M (default 256).
    pub n_y: usize,
    /// Subsample cap on rows (deterministic prefix); None = all.
    pub max_rows: Option<usize>,
    /// Number of Legendre channels to project onto.
    pub k_channels: usize,
}

impl Default for MixtureConfig {
    fn default() -> Self {
        Self {
            n_y: 256,
            max_rows: Some(200_000),
            k_channels: 6,
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
}

/// Linear interpolate Q(u) given monotone nodes (u_nodes ascending, q_nodes monotone).
fn interp_q(u_nodes: &[f64], q_nodes: &[f64], u: f64) -> f64 {
    let n = u_nodes.len();
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

/// Invert monotone Q at level y → u = F(y) by root search on Q(u) - y.
fn cdf_from_q(u_nodes: &[f64], q_nodes: &[f64], y: f64) -> f64 {
    let n = u_nodes.len();
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

/// Invert a mixture CDF tabulated on `y_grid` / `f_grid` at probability `u`.
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

/// Build the model mixture from per-row conditional quantile curves.
///
/// `u_nodes`: shared u-grid for every row (length q).
/// `q_rows`: row-major (n × q) monotone conditional quantiles Q_t(u|x_i).
/// `y_lo`/`y_hi`: range for the F_M grid (typically pooled 0.1%–99.9%).
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

/// Like [`mixture_from_conditionals`] but with optional non-negative row weights
/// (e.g. a quadrature rule over the covariate law). Weights are renormalized.
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
    let q = u_nodes.len();
    let n_use = cfg.max_rows.map(|m| m.min(n)).unwrap_or(n).max(1);
    let n_y = cfg.n_y.max(8);
    let mut y_grid = Vec::with_capacity(n_y);
    for i in 0..n_y {
        let t = i as f64 / (n_y - 1) as f64;
        y_grid.push(y_lo + t * (y_hi - y_lo));
    }
    let mut wsum = 0.0;
    let mut f_grid = vec![0.0; n_y];
    for i in 0..n_use {
        let wi = row_weights.map(|w| w[i]).unwrap_or(1.0);
        wsum += wi;
        let row = &q_rows[i * q..(i + 1) * q];
        for (g, &y) in y_grid.iter().enumerate() {
            f_grid[g] += wi * cdf_from_q(u_nodes, row, y);
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
    f_grid[0] = f_grid[0].clamp(0.0, 1.0);
    f_grid[n_y - 1] = f_grid[n_y - 1].clamp(0.0, 1.0);

    let gl = gauss_legendre_01(64).expect("GL order > 0");
    let mut q_mix = Vec::with_capacity(gl.nodes.len());
    for &u in &gl.nodes {
        q_mix.push(invert_cdf(&y_grid, &f_grid, u));
    }

    let k = cfg.k_channels.max(1);
    let mut channels = vec![0.0; k];
    for i in 0..gl.nodes.len() {
        let (pk, _) = pstar_all(gl.nodes[i], k);
        for j in 0..k {
            channels[j] += gl.weights[i] * q_mix[i] * pk[j];
        }
    }
    for j in 0..k {
        channels[j] *= (2 * j + 1) as f64;
    }

    MixtureLaw {
        u: gl.nodes,
        q: q_mix,
        channels,
        y_grid,
        f_grid,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn norm_ppf(u: f64) -> f64 {
        // Beasley-Springer-Moro approximation is overkill; use erfinv via libm if available.
        // Rational approximation (Acklam):
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
        assert!(
            max_err < 1e-3,
            "degenerate-X mixture should recover Q; max_err={max_err}"
        );
    }

    #[test]
    fn gaussian_location_mixture_closed_form() {
        // Exact: mu ~ N(0, v), eps ~ N(0,1) independent ⇒ Y ~ N(0, sigma^2+v).
        // Integrate over mu with a trapezoid rule on the normal density (row weights).
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
        assert!(
            max_err < 1e-3,
            "Gaussian location mixture vs closed form; max_err={max_err}"
        );
    }
}
