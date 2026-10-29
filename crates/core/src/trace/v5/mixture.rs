//! Mixture counterfactual CDF, its inversion, and exact L-moments of the result.
//!
//! All functions are pure and allocation-light. Grid evaluation is parallel over grid
//! points when the `parallel` feature is on; results are bit-identical either way.

use crate::error::{Result, TraceError};
use crate::trace::v4::mixture::cdf_from_q;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// A mixture law tabulated on a y-grid.
#[derive(Clone, Debug)]
pub struct MixtureLawV5 {
    /// Increasing y-grid.
    pub y_grid: Vec<f64>,
    /// `F(y_grid[g])`, non-decreasing in `[0, 1]`.
    pub f_grid: Vec<f64>,
    /// L-moments `λ_1..λ_K` of the law (piecewise-linear CDF between grid points).
    pub lmoments: Vec<f64>,
}

/// Evenly spaced grid of `n_y` points on `[lo, hi]`.
pub fn support_grid(lo: f64, hi: f64, n_y: usize) -> Result<Vec<f64>> {
    if n_y < 2 || !(lo.is_finite() && hi.is_finite()) || hi <= lo {
        return Err(TraceError::InvalidArgument(
            "support_grid needs n_y >= 2 and finite lo < hi".into(),
        ));
    }
    let h = (hi - lo) / (n_y - 1) as f64;
    Ok((0..n_y).map(|g| lo + h * g as f64).collect())
}

fn sorted_finite(v: &[f64]) -> Vec<f64> {
    let mut s: Vec<f64> = v.iter().copied().filter(|z| z.is_finite()).collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s
}

/// Count pairs `(i, j)` with `a[i] + b[j] <= y`, both sorted ascending. O(n + m).
fn count_pairs_le(a: &[f64], b: &[f64], y: f64) -> u64 {
    let mut j = b.len();
    let mut c: u64 = 0;
    for &ai in a {
        while j > 0 && ai + b[j - 1] > y {
            j -= 1;
        }
        if j == 0 {
            break;
        }
        c += j as u64;
    }
    c
}

/// Mixture CDF for a location + residual conditional `Y | x = μ_t(x) + ε`, `ε ~ G_t`
/// (empirical residuals of period `t`).
///
/// `mu` holds `μ_t(x_i)` for the covariate rows `x_i ~ F_s`; `resid` is the residual
/// sample defining `G_t`. Returns `F_[s,t](y) = (1/n) Σ_i G_t(y - μ_i)` at each grid
/// point, computed exactly by pair counting (no smoothing, no subsampling).
pub fn mixture_cdf_location(mu: &[f64], resid: &[f64], y_grid: &[f64]) -> Result<Vec<f64>> {
    let a = sorted_finite(mu);
    let b = sorted_finite(resid);
    if a.is_empty() || b.is_empty() {
        return Err(TraceError::InvalidArgument(
            "mixture_cdf_location needs non-empty finite mu and resid".into(),
        ));
    }
    let denom = (a.len() as f64) * (b.len() as f64);
    let f = |y: &f64| count_pairs_le(&a, &b, *y) as f64 / denom;
    #[cfg(feature = "parallel")]
    let out: Vec<f64> = y_grid.par_iter().map(f).collect();
    #[cfg(not(feature = "parallel"))]
    let out: Vec<f64> = y_grid.iter().map(f).collect();
    Ok(out)
}

/// Generic mixture CDF from per-row conditional quantile curves.
///
/// `q_rows` is row-major `(n_rows × u_nodes.len())`, each row a non-decreasing
/// `Q_t(u | x_i)` on `u_nodes` (increasing in `(0, 1)`). Optional non-negative
/// `weights` turn the plain average into a weighted one. Returns
/// `F(y) = Σ_i w_i F_t(y | x_i) / Σ_i w_i` on `y_grid`.
pub fn mixture_cdf_quantile_rows(
    u_nodes: &[f64],
    q_rows: &[f64],
    n_rows: usize,
    weights: Option<&[f64]>,
    y_grid: &[f64],
) -> Result<Vec<f64>> {
    let q = u_nodes.len();
    if q < 2 || n_rows == 0 || q_rows.len() != n_rows * q {
        return Err(TraceError::InvalidArgument(
            "q_rows must be n_rows × len(u_nodes) with len(u_nodes) >= 2".into(),
        ));
    }
    if let Some(w) = weights {
        if w.len() != n_rows || w.iter().any(|v| !(v.is_finite() && *v >= 0.0)) {
            return Err(TraceError::InvalidArgument(
                "weights must be n_rows finite non-negative values".into(),
            ));
        }
    }
    let wsum: f64 = weights.map(|w| w.iter().sum()).unwrap_or(n_rows as f64);
    if wsum <= 0.0 {
        return Err(TraceError::InvalidArgument("weights sum to zero".into()));
    }
    let f = |y: &f64| {
        let mut s = 0.0;
        for i in 0..n_rows {
            let wi = weights.map(|w| w[i]).unwrap_or(1.0);
            if wi > 0.0 {
                s += wi * cdf_from_q(u_nodes, &q_rows[i * q..(i + 1) * q], *y);
            }
        }
        s / wsum
    };
    #[cfg(feature = "parallel")]
    let out: Vec<f64> = y_grid.par_iter().map(f).collect();
    #[cfg(not(feature = "parallel"))]
    let out: Vec<f64> = y_grid.iter().map(f).collect();
    Ok(out)
}

/// Left-inverse of a tabulated CDF (linear between grid points) at levels `u`.
pub fn quantile_from_cdf(y_grid: &[f64], f_grid: &[f64], u: &[f64]) -> Vec<f64> {
    let n = y_grid.len();
    u.iter()
        .map(|&uu| {
            if n == 0 {
                return f64::NAN;
            }
            // first g with F[g] >= u
            let g = f_grid.partition_point(|&fv| fv < uu);
            if g == 0 {
                return y_grid[0];
            }
            if g >= n {
                return y_grid[n - 1];
            }
            let (f0, f1) = (f_grid[g - 1], f_grid[g]);
            let t = if f1 > f0 { (uu - f0) / (f1 - f0) } else { 1.0 };
            y_grid[g - 1] + t * (y_grid[g] - y_grid[g - 1])
        })
        .collect()
}

/// Shifted Legendre `P*_0..P*_{k-1}` at `u`.
fn pstar(u: f64, k: usize) -> Vec<f64> {
    crate::trace::legendre::pstar_all(u, k).0
}

/// Exact L-moments `λ_1..λ_k` (`λ_{r+1} = ∫ Q(u) P*_r(u) du`) of the law whose CDF is
/// linear between grid points, with atoms at `y_grid[0]` (mass `F[0]`) and
/// `y_grid[last]` (mass `1 - F[last]`).
pub fn lmoments_from_cdf_grid(y_grid: &[f64], f_grid: &[f64], k: usize) -> Vec<f64> {
    // 4-point Gauss–Legendre on [0,1] is exact for degree <= 7 (Q linear × P*_r, r <= 6).
    const GN: [f64; 4] = [
        0.069_431_844_202_973_71,
        0.330_009_478_207_571_9,
        0.669_990_521_792_428_1,
        0.930_568_155_797_026_3,
    ];
    const GW: [f64; 4] = [
        0.173_927_422_568_726_9,
        0.326_072_577_431_273_1,
        0.326_072_577_431_273_1,
        0.173_927_422_568_726_9,
    ];
    let n = y_grid.len();
    let mut lam = vec![0.0; k];
    if n == 0 || k == 0 {
        return lam;
    }
    // ∫_a^b P*_r(u) du for the flat end pieces via exact antiderivative quadrature.
    let flat = |a: f64, b: f64, y: f64, lam: &mut [f64]| {
        if b <= a {
            return;
        }
        for (gn, gw) in GN.iter().zip(GW.iter()) {
            let u = a + (b - a) * gn;
            let p = pstar(u, k);
            for r in 0..k {
                lam[r] += gw * (b - a) * y * p[r];
            }
        }
    };
    let f0 = f_grid[0].clamp(0.0, 1.0);
    flat(0.0, f0, y_grid[0], &mut lam);
    let mut prev_f = f0;
    for g in 1..n {
        let f1 = f_grid[g].clamp(prev_f, 1.0);
        if f1 > prev_f {
            let (ya, yb) = (y_grid[g - 1], y_grid[g]);
            for (gn, gw) in GN.iter().zip(GW.iter()) {
                let u = prev_f + (f1 - prev_f) * gn;
                let y = ya + (yb - ya) * gn;
                let p = pstar(u, k);
                for r in 0..k {
                    lam[r] += gw * (f1 - prev_f) * y * p[r];
                }
            }
        }
        prev_f = f1;
    }
    flat(prev_f, 1.0, y_grid[n - 1], &mut lam);
    lam
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_count_matches_brute_force() {
        let mu = [0.3, -1.0, 2.0, 0.0];
        let r = [-0.5, 0.1, 0.7];
        let grid = support_grid(-3.0, 4.0, 57).unwrap();
        let f = mixture_cdf_location(&mu, &r, &grid).unwrap();
        for (g, y) in grid.iter().enumerate() {
            let mut c = 0;
            for a in mu {
                for b in r {
                    if a + b <= *y {
                        c += 1;
                    }
                }
            }
            assert!((f[g] - c as f64 / 12.0).abs() < 1e-15);
        }
    }

    #[test]
    fn lmoments_of_uniform_are_exact() {
        // U(0,1): λ1 = 1/2, λ2 = 1/6, λ3 = λ4 = 0.
        let grid = support_grid(0.0, 1.0, 11).unwrap();
        let lam = lmoments_from_cdf_grid(&grid, &grid, 4);
        assert!((lam[0] - 0.5).abs() < 1e-12);
        assert!((lam[1] - 1.0 / 6.0).abs() < 1e-12);
        assert!(lam[2].abs() < 1e-12 && lam[3].abs() < 1e-12);
    }

    #[test]
    fn quantile_inverts_cdf() {
        let grid = support_grid(-2.0, 2.0, 101).unwrap();
        let f: Vec<f64> = grid.iter().map(|y| (y + 2.0) / 4.0).collect();
        let q = quantile_from_cdf(&grid, &f, &[0.1, 0.5, 0.9]);
        assert!((q[0] + 1.6).abs() < 1e-12 && q[1].abs() < 1e-12 && (q[2] - 1.6).abs() < 1e-12);
    }
}
