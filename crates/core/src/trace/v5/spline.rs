//! TRACE_V5_2 location model (PREREG_V5_2 §2): fixed additive restricted (natural)
//! cubic spline basis fitted per period by unpenalised least squares.
//!
//! * Basis: for each input column, Harrell's (2015, §2.4.5) truncated-power restricted
//!   cubic spline with `K = 5` knots at the 5%, 27.5%, 50%, 72.5%, 95% quantiles of
//!   that column on the **pooled** covariates (both periods share one basis). Each
//!   column gives `K - 1 = 4` functions (linear term + 3 non-linear terms); additive,
//!   no cross terms. Design: intercept + `4p` columns.
//! * Fit: ordinary least squares by Householder QR with column pivoting. A column whose
//!   pivot `|R_kk| <= RANK_TOL · |R_00|` is dropped (coefficient 0) and recorded.
//!   That is a fixed rule, not a tuning step.

use crate::error::{Result, TraceError};

/// Knot quantile levels (PREREG_V5_2 §2, Harrell 2015 §2.4.6 for K = 5).
pub const KNOT_PROBS: [f64; 5] = [0.05, 0.275, 0.5, 0.725, 0.95];
/// Relative pivot tolerance of the pivoted QR rank rule.
pub const RANK_TOL: f64 = 1e-10;

/// Shared additive spline basis (knots per column).
#[derive(Clone, Debug)]
pub struct SplineBasis {
    /// `knots[j]` = the 5 knots of column `j`.
    pub knots: Vec<[f64; 5]>,
}

/// Fitted per-period spline location model.
#[derive(Clone, Debug)]
pub struct SplineFit {
    /// Coefficients on the `1 + 4p` design (dropped columns are 0).
    pub coef: Vec<f64>,
    /// Indices (into the `1 + 4p` design; 0 = intercept, then 4 per column) dropped by
    /// the pivoted-QR rank rule.
    pub dropped: Vec<usize>,
}

/// Quantile with linear interpolation between order statistics (Hyndman–Fan type 7).
fn quantile_sorted(s: &[f64], prob: f64) -> f64 {
    let h = (s.len() - 1) as f64 * prob;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(s.len() - 1);
    s[lo] + (h - lo as f64) * (s[hi] - s[lo])
}

impl SplineBasis {
    /// Knots from the pooled covariates `x_a ∪ x_b` (row-major, `p` columns).
    pub fn from_pooled(x_a: &[f64], n_a: usize, x_b: &[f64], n_b: usize, p: usize) -> Self {
        let knots = (0..p)
            .map(|j| {
                let mut col: Vec<f64> = (0..n_a)
                    .map(|i| x_a[i * p + j])
                    .chain((0..n_b).map(|i| x_b[i * p + j]))
                    .collect();
                col.sort_by(|a, b| a.total_cmp(b));
                let mut k = [0.0; 5];
                for (kk, pr) in k.iter_mut().zip(KNOT_PROBS) {
                    *kk = quantile_sorted(&col, pr);
                }
                k
            })
            .collect();
        Self { knots }
    }

    pub fn p(&self) -> usize {
        self.knots.len()
    }

    /// Number of design columns (`1 + 4p`).
    pub fn n_cols(&self) -> usize {
        1 + 4 * self.p()
    }

    /// The 4 basis values of column `j` at `v`: `[v, s_1, s_2, s_3]` with
    /// `s_l = [(v-t_l)_+^3 - (v-t_4)_+^3 (t_5-t_l)/(t_5-t_4) + (v-t_5)_+^3 (t_4-t_l)/(t_5-t_4)] / (t_5-t_1)^2`.
    pub fn eval_col(&self, j: usize, v: f64) -> [f64; 4] {
        let t = &self.knots[j];
        let c = |a: f64| {
            let d = v - a;
            if d > 0.0 {
                d * d * d
            } else {
                0.0
            }
        };
        let den = t[4] - t[3];
        let norm = (t[4] - t[0]).powi(2);
        let mut out = [v, 0.0, 0.0, 0.0];
        if den > 0.0 && norm > 0.0 {
            for l in 0..3 {
                out[l + 1] = (c(t[l]) - c(t[3]) * (t[4] - t[l]) / den
                    + c(t[4]) * (t[3] - t[l]) / den)
                    / norm;
            }
        }
        // tied boundary knots: non-linear terms are identically 0 -> dropped by the QR rule
        out
    }

    /// Column-major design (`n × (1 + 4p)`).
    pub fn design(&self, x: &[f64], n: usize) -> Vec<f64> {
        let p = self.p();
        let m = self.n_cols();
        let mut a = vec![0.0f64; n * m];
        a[..n].fill(1.0);
        for i in 0..n {
            for j in 0..p {
                let b = self.eval_col(j, x[i * p + j]);
                for (l, bv) in b.iter().enumerate() {
                    a[(1 + 4 * j + l) * n + i] = *bv;
                }
            }
        }
        a
    }

    /// Per-period OLS on the spline design (pivoted Householder QR).
    pub fn fit(&self, x: &[f64], n: usize, y: &[f64]) -> Result<SplineFit> {
        let p = self.p();
        let m = self.n_cols();
        if x.len() != n * p || y.len() != n || n < m {
            return Err(TraceError::InvalidArgument(
                "spline location fit needs n >= 1 + 4p and matching shapes".into(),
            ));
        }
        let a = self.design(x, n);
        let (coef, dropped) = lstsq_pivoted(a, n, m, y)?;
        Ok(SplineFit { coef, dropped })
    }

    /// Fitted mean at each row of `x`.
    pub fn predict(&self, fit: &SplineFit, x: &[f64], n: usize) -> Vec<f64> {
        let p = self.p();
        (0..n)
            .map(|i| {
                let mut s = fit.coef[0];
                for j in 0..p {
                    let b = self.eval_col(j, x[i * p + j]);
                    for (l, bv) in b.iter().enumerate() {
                        s += fit.coef[1 + 4 * j + l] * bv;
                    }
                }
                s
            })
            .collect()
    }
}

/// Least squares `min ||A c - y||` with `A` column-major `n × m`, by Householder QR with
/// column pivoting (largest remaining norm first). Returns coefficients (0 for dropped
/// columns) and the dropped column indices.
pub fn lstsq_pivoted(
    mut a: Vec<f64>,
    n: usize,
    m: usize,
    y: &[f64],
) -> Result<(Vec<f64>, Vec<usize>)> {
    let mut b = y.to_vec();
    let mut perm: Vec<usize> = (0..m).collect();
    let mut norms: Vec<f64> = (0..m)
        .map(|k| a[k * n..(k + 1) * n].iter().map(|v| v * v).sum())
        .collect();
    let kmax = m.min(n);
    let mut rdiag = vec![0.0f64; kmax];
    let mut rank = 0usize;
    let mut r00 = 0.0f64;
    for k in 0..kmax {
        // pivot: remaining column with the largest residual norm (recomputed exactly)
        for jj in k..m {
            let c = &a[jj * n + k..(jj + 1) * n];
            norms[jj] = c.iter().map(|v| v * v).sum();
        }
        let (piv, _) =
            (k..m)
                .map(|jj| (jj, norms[jj]))
                .fold((k, -1.0), |acc, t| if t.1 > acc.1 { t } else { acc });
        if piv != k {
            for i in 0..n {
                a.swap(k * n + i, piv * n + i);
            }
            perm.swap(k, piv);
            norms.swap(k, piv);
        }
        let norm = norms[k].sqrt();
        if k == 0 {
            r00 = norm;
        }
        if norm <= RANK_TOL * r00 || norm == 0.0 {
            break;
        }
        let col = &mut a[k * n..(k + 1) * n];
        let alpha = if col[k] > 0.0 { -norm } else { norm };
        col[k] -= alpha;
        let vnorm2: f64 = col[k..].iter().map(|v| v * v).sum();
        rdiag[k] = alpha;
        let v: Vec<f64> = col[k..].to_vec();
        for jj in (k + 1)..m {
            let cj = &mut a[jj * n + k..(jj + 1) * n];
            let dot: f64 = v.iter().zip(cj.iter()).map(|(p, q)| p * q).sum();
            let f = 2.0 * dot / vnorm2;
            for (c, vv) in cj.iter_mut().zip(&v) {
                *c -= f * vv;
            }
        }
        let dot: f64 = v.iter().zip(&b[k..]).map(|(p, q)| p * q).sum();
        let f = 2.0 * dot / vnorm2;
        for (c, vv) in b[k..].iter_mut().zip(&v) {
            *c -= f * vv;
        }
        rank += 1;
    }
    if rank == 0 {
        return Err(TraceError::InvalidArgument(
            "spline location fit: design has rank 0".into(),
        ));
    }
    // back-substitution on the leading rank × rank block (R_kj stored at a[j*n + k])
    let mut c_perm = vec![0.0f64; m];
    for k in (0..rank).rev() {
        let mut s = b[k];
        for jj in (k + 1)..rank {
            s -= a[jj * n + k] * c_perm[jj];
        }
        c_perm[k] = s / rdiag[k];
    }
    let mut coef = vec![0.0f64; m];
    for (k, &orig) in perm.iter().enumerate() {
        coef[orig] = c_perm[k];
    }
    let mut dropped: Vec<usize> = perm[rank..].to_vec();
    dropped.sort_unstable();
    Ok((coef, dropped))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::shift::SplitMix64;

    fn normal(rng: &mut SplitMix64) -> f64 {
        let u1 = rng.next_f64().max(1e-300);
        let u2 = rng.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    #[test]
    fn knots_are_pooled_type7_quantiles() {
        let xa: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let xb: Vec<f64> = (50..101).map(|i| i as f64).collect();
        let b = SplineBasis::from_pooled(&xa, 50, &xb, 51, 1);
        for (k, want) in b.knots[0].iter().zip([5.0, 27.5, 50.0, 72.5, 95.0]) {
            assert!((k - want).abs() < 1e-12, "{:?}", b.knots);
        }
    }

    #[test]
    fn basis_is_linear_beyond_boundary_knots() {
        let b = SplineBasis {
            knots: vec![[-1.6, -0.6, 0.0, 0.6, 1.6]],
        };
        for side in [(-5.0, -4.0, -3.0), (3.0, 4.0, 5.0)] {
            let (f0, f1, f2) = (
                b.eval_col(0, side.0),
                b.eval_col(0, side.1),
                b.eval_col(0, side.2),
            );
            for l in 0..4 {
                assert!((f2[l] - 2.0 * f1[l] + f0[l]).abs() < 1e-10, "{l} {side:?}");
            }
        }
        // zero below the first knot for the non-linear terms
        assert!(b.eval_col(0, -2.0)[1..].iter().all(|v| *v == 0.0));
    }

    #[test]
    fn spline_fits_smooth_additive_curve_and_nests_linear() {
        let mut rng = SplitMix64::new(61);
        let (n, p) = (4000, 2);
        let x: Vec<f64> = (0..n * p).map(|_| normal(&mut rng)).collect();
        let f = |i: usize| 1.0 + x[i * p] - 0.5 * x[i * p + 1] + (x[i * p + 1]).sin();
        let y: Vec<f64> = (0..n).map(f).collect();
        let b = SplineBasis::from_pooled(&x, n, &x[..0], 0, p);
        let fit = b.fit(&x, n, &y).unwrap();
        assert!(fit.dropped.is_empty());
        let yh = b.predict(&fit, &x, n);
        let rmse = (yh.iter().zip(&y).map(|(a, c)| (a - c).powi(2)).sum::<f64>() / n as f64).sqrt();
        // natural spline is linear beyond the 5%/95% knots, so tails carry some bias
        assert!(rmse < 0.1, "{rmse}");
        // exact recovery of a linear mean (non-linear coefficients 0)
        let yl: Vec<f64> = (0..n)
            .map(|i| 0.3 + 2.0 * x[i * p] - x[i * p + 1])
            .collect();
        let fit = b.fit(&x, n, &yl).unwrap();
        for (k, want) in [(0, 0.3), (1, 2.0), (5, -1.0)] {
            assert!((fit.coef[k] - want).abs() < 1e-8, "{:?}", fit.coef);
        }
        for k in [2, 3, 4, 6, 7, 8] {
            assert!(fit.coef[k].abs() < 1e-8);
        }
    }

    #[test]
    fn tied_quantiles_drop_columns_and_record_them() {
        let mut rng = SplitMix64::new(62);
        let (n, p) = (2000, 2);
        // column 1 is binary: knots tie, non-linear terms vanish
        let x: Vec<f64> = (0..n)
            .flat_map(|i| [normal(&mut rng), (i % 2) as f64])
            .collect();
        let y: Vec<f64> = (0..n).map(|i| x[i * p] + 2.0 * x[i * p + 1]).collect();
        let b = SplineBasis::from_pooled(&x, n, &x[..0], 0, p);
        let fit = b.fit(&x, n, &y).unwrap();
        assert_eq!(fit.dropped, vec![6, 7, 8]);
        assert!((fit.coef[5] - 2.0).abs() < 1e-8);
        let yh = b.predict(&fit, &x, n);
        assert!(yh.iter().zip(&y).all(|(a, c)| (a - c).abs() < 1e-8));
    }
}
