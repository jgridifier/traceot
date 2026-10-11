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

/// Lower endpoint of the pinball minimiser on a finite sample.
///
/// For a level `u`, the check loss is minimised by every `q` in the closed interval
/// between two order statistics when `u * n` is an integer, and by a single order
/// statistic otherwise. The unique minimiser (non-integer case) is the order
/// statistic `y_(ceil(u*n))` in 1-based ranking, which is `sorted[ceil(u*n) - 1]`.
/// When `u * n` is an integer the minimisers are `[y_(u*n), y_(u*n+1)]` (1-based):
/// the lower endpoint is `sorted[u*n - 1]` and the upper endpoint is `sorted[u*n]`
/// (the same index `floor(u*n)` produces). TRACE uses the lower endpoint,
/// `clamp(ceil(u*n) - 1, 0, n-1)`. The two conventions agree whenever `u*n` is not
/// an integer.
pub fn pinball_quantile_index(u: f64, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    if u <= 0.0 {
        return 0;
    }
    if u >= 1.0 {
        return n - 1;
    }
    let ceil_un = (u * n as f64).ceil();
    let idx = (ceil_un - 1.0).max(0.0);
    (idx as usize).min(n - 1)
}

/// Order statistics of finite `y` on the probability grid `u` (lower pinball convention).
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
        .map(|&ui| ys[pinball_quantile_index(ui, n)])
        .collect())
}

/// Split-conformal residual quantile. Index `ceil((n+1) * u) - 1`, clamped to the
/// sample. This is the upper finite-sample rank used for one-sided coverage at
/// level `u`, and it is not the same index as [`empirical_quantile`]: the `+1`
/// in `n+1` is the conformal correction. When the rank would pass `n`, the largest
/// residual is used.
pub fn conformal_quantile(y: &[f64], u: f64) -> Result<f64> {
    if y.is_empty() || !u.is_finite() {
        return Err(TraceError::InvalidArgument(
            "conformal_quantile needs residuals and a finite level".into(),
        ));
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(TraceError::InvalidArgument(
            "residuals must be finite".into(),
        ));
    }
    let mut ys = y.to_vec();
    ys.sort_by(f64::total_cmp);
    let n = ys.len();
    let rank = ((n as f64 + 1.0) * u).ceil();
    let idx = (rank - 1.0).clamp(0.0, (n - 1) as f64) as usize;
    Ok(ys[idx])
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

    #[test]
    fn lower_pinball_endpoint_when_level_times_n_is_integral() {
        let y = [1.0, 2.0, 3.0, 4.0];
        // u*n = 2. Lower endpoint is y_(2) = 2. The upper endpoint would be y_(3) = 3.
        let q = empirical_quantile(&y, &[0.5]).unwrap();
        assert_eq!(q[0], 2.0);
        let q = empirical_quantile(&y, &[0.25]).unwrap();
        assert_eq!(q[0], 1.0);
        // u*n = 1.6 is not integral, so lower and upper indices agree at 1 (value 2).
        let q = empirical_quantile(&y, &[0.4]).unwrap();
        assert_eq!(q[0], 2.0);
        let upper = {
            let n: f64 = 4.0;
            let r = (0.5 * n).floor() as usize;
            y[r.min(3)]
        };
        assert_eq!(upper, 3.0);
    }

    #[test]
    fn conformal_rank_uses_n_plus_one() {
        let y = [1.0, 2.0, 3.0];
        // ceil((3+1)*0.5) - 1 = 1 → 2. The plain lower pinball index would be ceil(1.5)-1 = 1 as well.
        assert_eq!(conformal_quantile(&y, 0.5).unwrap(), 2.0);
        // ceil(4*0.9)-1 = 3, clamped to the last residual.
        assert_eq!(conformal_quantile(&y, 0.9).unwrap(), 3.0);
        // ceil(4*0.25)-1 = 0 → 1, while floor(0.25*3) = 0 as well. A split case:
        // u = 2/4 = 0.5 already checked. u = 1/(n+1) = 0.25.
        assert_eq!(conformal_quantile(&y, 0.2).unwrap(), 1.0);
    }
}
