//! Quantile basis for TRACE: location, scale and shape channels.
//!
//! `z(u) = Phi^{-1}(u)` uses Peter J. Acklam's rational approximation
//! (relative error about 1e-9). Channels, before root-mean-square normalisation
//! on the grid:
//!
//! * 0 location: `1`
//! * 1 scale: `z`
//! * 2 shape (tail): `z^3 - 3z` (probabilists' Hermite He_3), when `M >= 3`
//! * 3 even shape: `z^2 - 1` (He_2), when `M = 4`

use crate::error::{Result, TraceError};
use crate::w2::quantile_grid;

/// Acklam's inverse standard-normal approximation. `p` must lie in `(0, 1)`.
#[allow(clippy::excessive_precision)]
pub fn inv_norm(p: f64) -> Result<f64> {
    if !(p > 0.0 && p < 1.0) {
        return Err(TraceError::InvalidArgument(
            "inv_norm requires p in (0, 1)".into(),
        ));
    }
    // Published coefficients (Acklam). Kept as literals so the numpy reference
    // can use the same operations in the same order.
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
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
    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        let num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5];
        let den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0;
        return Ok(num / den);
    }
    if p > 1.0 - plow {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        let num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5];
        let den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0;
        return Ok(-(num / den));
    }
    let q = p - 0.5;
    let r = q * q;
    let num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q;
    let den = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0;
    Ok(num / den)
}

/// Midpoint grid and the `(K, M)` basis, row-major, each column RMS-normalised.
/// A column that is numerically zero (for example the scale channel on `u = 0.5`
/// alone) is left as zeros.
pub fn quantile_basis(n_levels: usize, n_channels: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if !(2..=4).contains(&n_channels) {
        return Err(TraceError::InvalidArgument(
            "n_channels must be 2, 3 or 4".into(),
        ));
    }
    if !(3..=401).contains(&n_levels) {
        return Err(TraceError::InvalidArgument(
            "u_levels must be in [3, 401]".into(),
        ));
    }
    let u = quantile_grid(n_levels);
    let z: Vec<f64> = u.iter().map(|&ui| inv_norm(ui)).collect::<Result<_>>()?;
    let mut basis = vec![0.0; n_levels * n_channels];
    for (k, &zk) in z.iter().enumerate() {
        basis[k * n_channels] = 1.0;
        basis[k * n_channels + 1] = zk;
        if n_channels >= 3 {
            basis[k * n_channels + 2] = zk * zk * zk - 3.0 * zk;
        }
        if n_channels == 4 {
            basis[k * n_channels + 3] = zk * zk - 1.0;
        }
    }
    for m in 0..n_channels {
        let mut ss = 0.0;
        for k in 0..n_levels {
            let v = basis[k * n_channels + m];
            ss += v * v;
        }
        let rms = (ss / n_levels as f64).sqrt();
        if rms > 1e-15 {
            for k in 0..n_levels {
                basis[k * n_channels + m] /= rms;
            }
        }
    }
    Ok((u, basis))
}

/// Order statistics of finite `y` on the probability grid `u`.
pub fn empirical_quantile(y: &[f64], u: &[f64]) -> Result<Vec<f64>> {
    if y.is_empty() || u.is_empty() {
        return Err(TraceError::InvalidArgument(
            "empirical_quantile needs data and a grid".into(),
        ));
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(TraceError::InvalidArgument(
            "y must be finite to form the marginal quantile".into(),
        ));
    }
    let mut ys = y.to_vec();
    ys.sort_by(f64::total_cmp);
    let n = ys.len();
    Ok(u.iter()
        .map(|&ui| {
            let r = (ui * n as f64).floor() as usize;
            ys[r.min(n - 1)]
        })
        .collect())
}

/// Lexicographic column order under `f64::total_cmp`. A permutation of the columns
/// produces the same sequence of columns.
pub fn column_order(x: &[f64], n: usize, p: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..p).collect();
    idx.sort_by(|&a, &b| {
        for i in 0..n {
            let c = x[i * p + a].total_cmp(&x[i * p + b]);
            if c != std::cmp::Ordering::Equal {
                return c;
            }
        }
        std::cmp::Ordering::Equal
    });
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acklam_matches_known_tail() {
        let z = inv_norm(0.975).unwrap();
        assert!((z - 1.959963984540054).abs() < 1e-8, "{z}");
        assert!(inv_norm(0.5).unwrap().abs() < 1e-15);
        assert!((inv_norm(0.025).unwrap() + z).abs() < 1e-12);
        assert!(inv_norm(0.0).is_err());
    }

    #[test]
    fn basis_is_rms_one_and_ordered() {
        let (_u, b) = quantile_basis(19, 3).unwrap();
        for m in 0..3 {
            let ss: f64 = (0..19).map(|k| b[k * 3 + m].powi(2)).sum();
            assert!((ss / 19.0 - 1.0).abs() < 1e-12);
        }
        // location channel is constant
        let loc0 = b[0];
        assert!((0..19).all(|k| (b[k * 3] - loc0).abs() < 1e-15));
        assert!(quantile_basis(19, 5).is_err());
    }
}
