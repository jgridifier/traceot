//! TRACE_V5_2_SPECIFIED location model (PREREG_V5_2_SPECIFIED §2).
//!
//! Per-period ordinary least squares on an intercept, the same `p` raw columns as
//! V5.1, plus the two known non-linear terms that define the true mean:
//! `x₃²` and `sin(x₅)`, where `x₃` and `x₅` are the raw (uncentred) third and fifth
//! input columns (1-indexed; code columns 2 and 4). `sin` is `f64::sin`. Design
//! width: intercept + `p` + 2 (8 at `p = 5`, 23 at `p = 20`). No penalty, no
//! smoothing parameter, no feature selection, no data-driven choice of terms, and
//! no column centering: the design is computed on the raw draws.
//!
//! Solved by the same pivoted Householder QR as the spline model
//! ([`super::spline::lstsq_pivoted`]), with the fixed drop rule
//! `|R_{kk}| ≤ 10⁻¹⁰ |R_{00}|`. Dropped columns are recorded. That is a fixed rule,
//! not a tuning step.

use super::spline::lstsq_pivoted;
use crate::error::{Result, TraceError};

/// Code column of `x₃` (1-indexed 3).
pub const COL_X3: usize = 2;
/// Code column of `x₅` (1-indexed 5).
pub const COL_X5: usize = 4;

/// Number of design columns: intercept + `p` + `x₃²` + `sin(x₅)`.
pub fn n_cols(p: usize) -> usize {
    1 + p + 2
}

/// Fitted per-period specified-family location model.
#[derive(Clone, Debug)]
pub struct SpecifiedFit {
    /// Coefficients on the `1 + p + 2` design (dropped columns are 0).
    pub coef: Vec<f64>,
    /// Indices into the design (0 = intercept, 1..p = raw columns, `p+1` = `x₃²`,
    /// `p+2` = `sin(x₅)`) dropped by the pivoted-QR rank rule.
    pub dropped: Vec<usize>,
    pub p: usize,
}

/// Column-major design (`n × (1 + p + 2)`), computed on raw (uncentred) `x`.
pub fn design(x: &[f64], n: usize, p: usize) -> Result<Vec<f64>> {
    if p < 5 {
        return Err(TraceError::InvalidArgument(
            "specified location fit needs p >= 5 (x_3 and x_5)".into(),
        ));
    }
    if x.len() != n * p {
        return Err(TraceError::InvalidArgument(
            "specified location fit: x must be n×p row-major".into(),
        ));
    }
    let m = n_cols(p);
    let mut a = vec![0.0f64; n * m];
    a[..n].fill(1.0);
    for i in 0..n {
        for j in 0..p {
            a[(1 + j) * n + i] = x[i * p + j];
        }
        let x3 = x[i * p + COL_X3];
        let x5 = x[i * p + COL_X5];
        a[(1 + p) * n + i] = x3 * x3;
        a[(2 + p) * n + i] = x5.sin();
    }
    Ok(a)
}

/// Per-period OLS on the specified design (pivoted Householder QR).
pub fn fit(x: &[f64], n: usize, p: usize, y: &[f64]) -> Result<SpecifiedFit> {
    let m = n_cols(p);
    if n < m || y.len() != n {
        return Err(TraceError::InvalidArgument(
            "specified location fit needs n >= 1 + p + 2 and matching shapes".into(),
        ));
    }
    let a = design(x, n, p)?;
    let (coef, dropped) = lstsq_pivoted(a, n, m, y)?;
    Ok(SpecifiedFit { coef, dropped, p })
}

/// Fitted mean at each row of raw `x`.
pub fn predict(fit: &SpecifiedFit, x: &[f64], n: usize, p: usize) -> Vec<f64> {
    debug_assert_eq!(fit.p, p);
    (0..n)
        .map(|i| {
            let mut s = fit.coef[0];
            for j in 0..p {
                s += fit.coef[1 + j] * x[i * p + j];
            }
            let x3 = x[i * p + COL_X3];
            let x5 = x[i * p + COL_X5];
            s += fit.coef[1 + p] * x3 * x3;
            s += fit.coef[2 + p] * x5.sin();
            s
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Frozen β of the H7 / V5.1 simulator (PREREG_V5_2_SPECIFIED §3).
    const BETA: [f64; 5] = [1.0, -0.8, 0.6, 0.4, -0.3];

    fn true_mu(x: &[f64], n: usize, p: usize) -> Vec<f64> {
        // κ = 1: 1 + X_{1:5}^T β + (1/√2)(X₃² − 1) + √2 sin(X₅)
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let sqrt2 = 2.0_f64.sqrt();
        (0..n)
            .map(|i| {
                let mut s = 1.0;
                for j in 0..5 {
                    s += BETA[j] * x[i * p + j];
                }
                let x3 = x[i * p + COL_X3];
                let x5 = x[i * p + COL_X5];
                s + inv_sqrt2 * (x3 * x3 - 1.0) + sqrt2 * x5.sin()
            })
            .collect()
    }

    fn draw(n: usize, p: usize, seed: u64) -> Vec<f64> {
        // SplitMix64-ish, independent of any study seed stream.
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut next = || {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        // Box–Muller from those bits. Seed 17, 23, … are below 1000 and are not
        // gate indices (PREREG_V5_2_SPECIFIED §9: non-gate seeds only).
        let mut x = vec![0.0f64; n * p];
        for i in 0..(n * p).div_ceil(2) {
            let u1 = ((next() >> 11) as f64) / ((1u64 << 53) as f64);
            let u2 = ((next() >> 11) as f64) / ((1u64 << 53) as f64);
            let r = (-2.0 * u1.max(1e-16).ln()).sqrt();
            let th = 2.0 * std::f64::consts::PI * u2;
            let a = r * th.cos();
            let b = r * th.sin();
            if 2 * i < n * p {
                x[2 * i] = a;
            }
            if 2 * i + 1 < n * p {
                x[2 * i + 1] = b;
            }
        }
        x
    }

    #[test]
    fn recovers_true_mean_exactly_on_noiseless_draws() {
        // Seed 17: non-gate, below 1000.
        let n = 400;
        let p = 5;
        let x = draw(n, p, 17);
        let y = true_mu(&x, n, p);
        let f = fit(&x, n, p, &y).unwrap();
        assert!(f.dropped.is_empty(), "dropped {:?}", f.dropped);
        let hat = predict(&f, &x, n, p);
        let max_err = y
            .iter()
            .zip(&hat)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(max_err < 1e-10, "max |μ̂ − μ| = {max_err}");
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let sqrt2 = 2.0_f64.sqrt();
        assert!((f.coef[0] - (1.0 - inv_sqrt2)).abs() < 1e-10);
        for j in 0..5 {
            assert!((f.coef[1 + j] - BETA[j]).abs() < 1e-10, "β_{j}");
        }
        assert!((f.coef[1 + p] - inv_sqrt2).abs() < 1e-10);
        assert!((f.coef[2 + p] - sqrt2).abs() < 1e-10);
    }

    #[test]
    fn p20_extra_columns_have_near_zero_coef_on_noiseless_mean() {
        let n = 500;
        let p = 20;
        let x = draw(n, p, 23);
        let y = true_mu(&x, n, p);
        let f = fit(&x, n, p, &y).unwrap();
        assert!(f.dropped.is_empty());
        for j in 5..p {
            assert!(
                f.coef[1 + j].abs() < 1e-10,
                "extra col {j} coef {}",
                f.coef[1 + j]
            );
        }
        let hat = predict(&f, &x, n, p);
        let max_err = y
            .iter()
            .zip(&hat)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(max_err < 1e-10, "max |μ̂ − μ| = {max_err}");
    }

    #[test]
    fn drop_rule_records_a_zero_column() {
        // Seed 41. Make column 0 identically 0 so it is dropped.
        let n = 80;
        let p = 5;
        let mut x = draw(n, p, 41);
        for i in 0..n {
            x[i * p] = 0.0;
        }
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let x3 = x[i * p + COL_X3];
                1.0 + x[i * p + 1] + x3 * x3 + x[i * p + COL_X5].sin()
            })
            .collect();
        let f = fit(&x, n, p, &y).unwrap();
        assert!(
            f.dropped.contains(&1),
            "expected raw column 0 (design index 1) dropped, got {:?}",
            f.dropped
        );
        assert_eq!(f.coef[1], 0.0);
    }

    #[test]
    fn rejects_p_below_5_and_short_n() {
        let x = vec![0.0; 8];
        let y = vec![0.0; 2];
        assert!(fit(&x, 2, 4, &y).is_err());
        let x5 = vec![0.0; 5 * 5];
        let y5 = vec![0.0; 5];
        // n = 5 < 1+5+2 = 8
        assert!(fit(&x5, 5, 5, &y5).is_err());
    }
}
