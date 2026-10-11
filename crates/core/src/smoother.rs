//! Local linear (local Fréchet) and global Fréchet regression weights.
//!
//! Mirrors `tests/python/reference/smoother.py`.

use crate::error::{Result, TraceError};
use crate::linalg::{solve_spd, Mat};

/// Below this value of `1 - u1^2/(u0 u2)` a local linear window is degenerate and
/// the smoother falls back to Nadaraya-Watson weights.
pub const DEGENERACY_RTOL: f64 = 1e-10;

/// Smoothing kernel (unnormalised; constants cancel).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    Gaussian,
    Epanechnikov,
}

impl Kernel {
    /// Parse `"gaussian"` / `"epanechnikov"`.
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "gaussian" => Ok(Kernel::Gaussian),
            "epanechnikov" => Ok(Kernel::Epanechnikov),
            other => Err(TraceError::InvalidArgument(format!(
                "unknown kernel {other:?}"
            ))),
        }
    }

    /// `K(z)`.
    #[inline]
    pub fn value(self, z: f64) -> f64 {
        match self {
            Kernel::Gaussian => (-0.5 * z * z).exp(),
            Kernel::Epanechnikov => {
                if z.abs() < 1.0 {
                    1.0 - z * z
                } else {
                    0.0
                }
            }
        }
    }
}

/// Local linear weights (Petersen & Müller 2019, eq. (3) of the ADOPT paper).
/// Returns an `x_eval.len() x x_train.len()` matrix whose rows sum to one.
pub fn local_linear_weights(
    x_train: &[f64],
    x_eval: &[f64],
    h: f64,
    kernel: Kernel,
) -> Result<Mat> {
    if !(h > 0.0) {
        return Err(TraceError::InvalidArgument(
            "bandwidth h must be > 0".into(),
        ));
    }
    let n = x_train.len();
    let mut w = Mat::zeros(x_eval.len(), n);
    let mut kv = vec![0.0; n];
    let mut dv = vec![0.0; n];
    for (r, &x0) in x_eval.iter().enumerate() {
        let (mut u0, mut u1, mut u2) = (0.0, 0.0, 0.0);
        for i in 0..n {
            let d = x_train[i] - x0;
            let k = kernel.value(d / h);
            dv[i] = d;
            kv[i] = k;
            u0 += k;
            u1 += k * d;
            u2 += k * d * d;
        }
        let row = w.row_mut(r);
        let ok = u0 > 0.0 && u2 > 0.0 && (1.0 - (u1 * u1) / (u0 * u2)) > DEGENERACY_RTOL;
        if ok {
            let mut tot = 0.0;
            for i in 0..n {
                let s = kv[i] * (u2 - u1 * dv[i]);
                row[i] = s;
                tot += s;
            }
            row.iter_mut().for_each(|v| *v /= tot);
        } else if u0 > 0.0 {
            for i in 0..n {
                row[i] = kv[i] / u0;
            }
        } else {
            let mut best = 0;
            for i in 1..n {
                if dv[i].abs() < dv[best].abs() {
                    best = i;
                }
            }
            row[best] = 1.0;
        }
    }
    Ok(w)
}

/// How the local-linear window treats the ends of the observed predictor range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Boundary {
    /// Paper local-linear weights on the observed sample.
    #[default]
    None,
    /// Reflect each training point through both ends of the observed range and
    /// pool the ghost contributions into that point's weight. This is the
    /// classical reflection correction for kernel bias at the boundary.
    Reflect,
}

impl Boundary {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "none" => Ok(Boundary::None),
            "reflect" => Ok(Boundary::Reflect),
            other => Err(TraceError::InvalidArgument(format!(
                "unknown boundary {other:?} (expected 'none' or 'reflect')"
            ))),
        }
    }
}

/// Local linear weights with an optional reflection correction.
///
/// `Boundary::None` is exactly [`local_linear_weights`].
pub fn local_linear_weights_boundary(
    x_train: &[f64],
    x_eval: &[f64],
    h: f64,
    kernel: Kernel,
    boundary: Boundary,
) -> Result<Mat> {
    match boundary {
        Boundary::None => local_linear_weights(x_train, x_eval, h, kernel),
        Boundary::Reflect => reflected_local_linear_weights(x_train, x_eval, h, kernel),
    }
}

/// Local linear weights in which each training point also contributes its two
/// reflections through `min(x)` and `max(x)`.
///
/// The returned matrix has one column per original training point. A reflected
/// location uses the same response as its parent, so the column is the sum of
/// the three un-normalised contributions.
pub fn reflected_local_linear_weights(
    x_train: &[f64],
    x_eval: &[f64],
    h: f64,
    kernel: Kernel,
) -> Result<Mat> {
    if !(h > 0.0) {
        return Err(TraceError::InvalidArgument(
            "bandwidth h must be > 0".into(),
        ));
    }
    let n = x_train.len();
    if n == 0 {
        return Err(TraceError::InvalidArgument("x_train is empty".into()));
    }
    let lo = x_train.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = x_train.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !(hi > lo) {
        return local_linear_weights(x_train, x_eval, h, kernel);
    }
    // Three locations per training point: itself and the two reflections.
    let mut locs = vec![0.0; 3 * n];
    for i in 0..n {
        locs[i] = x_train[i];
        locs[n + i] = 2.0 * lo - x_train[i];
        locs[2 * n + i] = 2.0 * hi - x_train[i];
    }
    let m = 3 * n;
    let mut w = Mat::zeros(x_eval.len(), n);
    let mut kv = vec![0.0; m];
    let mut dv = vec![0.0; m];
    for (r, &x0) in x_eval.iter().enumerate() {
        let (mut u0, mut u1, mut u2) = (0.0, 0.0, 0.0);
        for i in 0..m {
            let d = locs[i] - x0;
            let k = kernel.value(d / h);
            dv[i] = d;
            kv[i] = k;
            u0 += k;
            u1 += k * d;
            u2 += k * d * d;
        }
        let row = w.row_mut(r);
        let ok = u0 > 0.0 && u2 > 0.0 && (1.0 - (u1 * u1) / (u0 * u2)) > DEGENERACY_RTOL;
        if ok {
            let mut tot = 0.0;
            for i in 0..m {
                let s = kv[i] * (u2 - u1 * dv[i]);
                row[i % n] += s;
                tot += s;
            }
            if tot != 0.0 {
                row.iter_mut().for_each(|v| *v /= tot);
            }
        } else if u0 > 0.0 {
            for i in 0..m {
                row[i % n] += kv[i] / u0;
            }
        } else {
            let mut best = 0;
            let mut best_d = f64::INFINITY;
            for i in 0..n {
                let d = (x_train[i] - x0).abs();
                if d < best_d {
                    best_d = d;
                    best = i;
                }
            }
            row[best] = 1.0;
        }
    }
    Ok(w)
}

/// Local linear weights for points that carry multiplicities (bin counts).
///
/// `weight[i] == 1` for every i reproduces [`local_linear_weights`]. A point
/// with weight 0 is ignored. Rows of the result sum to one.
pub fn local_linear_weights_weighted(
    x_train: &[f64],
    multiplicity: &[f64],
    x_eval: &[f64],
    h: f64,
    kernel: Kernel,
) -> Result<Mat> {
    if !(h > 0.0) {
        return Err(TraceError::InvalidArgument(
            "bandwidth h must be > 0".into(),
        ));
    }
    if x_train.len() != multiplicity.len() {
        return Err(TraceError::Shape(
            "multiplicity must have one entry per training point".into(),
        ));
    }
    let n = x_train.len();
    let mut w = Mat::zeros(x_eval.len(), n);
    let mut kv = vec![0.0; n];
    let mut dv = vec![0.0; n];
    for (r, &x0) in x_eval.iter().enumerate() {
        let (mut u0, mut u1, mut u2) = (0.0, 0.0, 0.0);
        for i in 0..n {
            let c = multiplicity[i];
            if !(c > 0.0) {
                kv[i] = 0.0;
                dv[i] = x_train[i] - x0;
                continue;
            }
            let d = x_train[i] - x0;
            let k = c * kernel.value(d / h);
            dv[i] = d;
            kv[i] = k;
            u0 += k;
            u1 += k * d;
            u2 += k * d * d;
        }
        let row = w.row_mut(r);
        let ok = u0 > 0.0 && u2 > 0.0 && (1.0 - (u1 * u1) / (u0 * u2)) > DEGENERACY_RTOL;
        if ok {
            let mut tot = 0.0;
            for i in 0..n {
                let s = kv[i] * (u2 - u1 * dv[i]);
                row[i] = s;
                tot += s;
            }
            if tot != 0.0 {
                row.iter_mut().for_each(|v| *v /= tot);
            }
        } else if u0 > 0.0 {
            for i in 0..n {
                row[i] = kv[i] / u0;
            }
        } else {
            let mut best = 0;
            let mut best_d = f64::INFINITY;
            for i in 0..n {
                if !(multiplicity[i] > 0.0) {
                    continue;
                }
                let d = dv[i].abs();
                if d < best_d {
                    best_d = d;
                    best = i;
                }
            }
            row[best] = 1.0;
        }
    }
    Ok(w)
}

/// Assign each `x` to an equal-width bin on `[min x, max x]`. The right endpoint
/// belongs to the last bin. Returns `(lo, hi, bin_of)`.
pub fn equal_width_bins(x: &[f64], n_bins: usize) -> Result<(f64, f64, Vec<usize>)> {
    if n_bins < 1 {
        return Err(TraceError::InvalidArgument("n_bins must be >= 1".into()));
    }
    if x.is_empty() {
        return Err(TraceError::InvalidArgument("x is empty".into()));
    }
    let lo = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut bin_of = vec![0usize; x.len()];
    if !(hi > lo) {
        return Ok((lo, hi, bin_of));
    }
    let width = (hi - lo) / n_bins as f64;
    for (i, &v) in x.iter().enumerate() {
        let mut b = ((v - lo) / width).floor() as isize;
        if b < 0 {
            b = 0;
        }
        if b >= n_bins as isize {
            b = n_bins as isize - 1;
        }
        bin_of[i] = b as usize;
    }
    Ok((lo, hi, bin_of))
}

/// Histogram smoother: bin `x_train`, average `y` inside each occupied bin, then
/// local-linear smooth the bin centres (with multiplicity equal to the bin count).
///
/// Cost is `O(n · d + n_eval · B · d)` rather than `O(n · n_eval · d)`.
pub fn binned_smooth(
    x_train: &[f64],
    y: &Mat,
    h: f64,
    x_eval: &[f64],
    n_bins: usize,
    kernel: Kernel,
) -> Result<Mat> {
    if y.rows != x_train.len() {
        return Err(TraceError::Shape(
            "y must have one row per training point".into(),
        ));
    }
    if n_bins < 1 {
        return Err(TraceError::InvalidArgument("n_bins must be >= 1".into()));
    }
    let (lo, hi, bin_of) = equal_width_bins(x_train, n_bins)?;
    let d = y.cols;
    let mut counts = vec![0.0; n_bins];
    let mut sums = vec![0.0; n_bins * d];
    for (i, &b) in bin_of.iter().enumerate() {
        counts[b] += 1.0;
        let row = y.row(i);
        for k in 0..d {
            sums[b * d + k] += row[k];
        }
    }
    let mut centers = Vec::new();
    let mut mult = Vec::new();
    let mut means_data = Vec::new();
    let width = if hi > lo {
        (hi - lo) / n_bins as f64
    } else {
        1.0
    };
    for b in 0..n_bins {
        if counts[b] > 0.0 {
            centers.push(lo + (b as f64 + 0.5) * width);
            mult.push(counts[b]);
            for k in 0..d {
                means_data.push(sums[b * d + k] / counts[b]);
            }
        }
    }
    if centers.is_empty() {
        return Err(TraceError::InvalidArgument("no occupied bins".into()));
    }
    let means = Mat::from_vec(centers.len(), d, means_data)?;
    // A single occupied location has no local-linear slope; the weighted call
    // falls back to putting mass 1 on that bin, which is the bin mean.
    let weights = if centers.len() == 1 {
        let mut w = Mat::zeros(x_eval.len(), 1);
        for r in 0..x_eval.len() {
            w.row_mut(r)[0] = 1.0;
        }
        w
    } else {
        local_linear_weights_weighted(&centers, &mult, x_eval, h, kernel)?
    };
    weights.matmul(&means)
}

/// Global Fréchet regression weights; rows sum to one.
pub fn global_frechet_weights(x_train: &Mat, x_eval: &Mat) -> Result<Mat> {
    let (n, p) = (x_train.rows, x_train.cols);
    if x_eval.cols != p {
        return Err(TraceError::Shape(
            "x_eval has wrong number of columns".into(),
        ));
    }
    let xbar = x_train.col_means();
    let mut s = vec![0.0; p * p];
    for i in 0..n {
        let r = x_train.row(i);
        for a in 0..p {
            for b in 0..p {
                s[a * p + b] += (r[a] - xbar[a]) * (r[b] - xbar[b]);
            }
        }
    }
    s.iter_mut().for_each(|v| *v /= n as f64);
    let mut w = Mat::zeros(x_eval.rows, n);
    for e in 0..x_eval.rows {
        let dx: Vec<f64> = x_eval
            .row(e)
            .iter()
            .zip(&xbar)
            .map(|(x, m)| x - m)
            .collect();
        let a = solve_spd(&s, p, &dx)?;
        let row = w.row_mut(e);
        for i in 0..n {
            let r = x_train.row(i);
            let mut c = 0.0;
            for t in 0..p {
                c += (r[t] - xbar[t]) * a[t];
            }
            row[i] = (1.0 + c) / n as f64;
        }
    }
    Ok(w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_linear_reproduces_lines() {
        let x: Vec<f64> = (0..30).map(|i| (i as f64 / 29.0).powf(1.3)).collect();
        let y: Vec<f64> = x.iter().map(|v| 2.0 - 3.0 * v).collect();
        let xe = [0.0, 0.33, 0.9, 1.2];
        for k in [Kernel::Gaussian, Kernel::Epanechnikov] {
            let w = local_linear_weights(&x, &xe, 0.3, k).unwrap();
            for (r, x0) in xe.iter().enumerate() {
                let row = w.row(r);
                let s: f64 = row.iter().sum();
                let fit: f64 = row.iter().zip(&y).map(|(a, b)| a * b).sum();
                assert!((s - 1.0).abs() < 1e-12);
                assert!((fit - (2.0 - 3.0 * x0)).abs() < 1e-9, "{k:?} {x0} {fit}");
            }
        }
    }

    #[test]
    fn degenerate_window_falls_back() {
        let x = [0.0, 1.0, 2.0];
        let w = local_linear_weights(&x, &[10.0], 0.1, Kernel::Epanechnikov).unwrap();
        assert_eq!(w.row(0), &[0.0, 0.0, 1.0]);
        assert!(local_linear_weights(&x, &[0.0], 0.0, Kernel::Gaussian).is_err());
        assert!(Kernel::parse("box").is_err());
    }

    #[test]
    fn weighted_weights_match_unit_multiplicity() {
        let x: Vec<f64> = (0..12).map(|i| i as f64 / 11.0).collect();
        let mult = vec![1.0; x.len()];
        let xe = [0.0, 0.2, 0.8, 1.0];
        let a = local_linear_weights(&x, &xe, 0.3, Kernel::Gaussian).unwrap();
        let b = local_linear_weights_weighted(&x, &mult, &xe, 0.3, Kernel::Gaussian).unwrap();
        for (u, v) in a.data.iter().zip(&b.data) {
            assert!((u - v).abs() < 1e-12);
        }
    }

    #[test]
    fn reflected_rows_sum_to_one() {
        let x = [0.0, 0.1, 0.4, 0.9, 1.0];
        let w =
            reflected_local_linear_weights(&x, &[0.0, 0.5, 1.0], 0.2, Kernel::Gaussian).unwrap();
        for r in 0..3 {
            let s: f64 = w.row(r).iter().sum();
            assert!((s - 1.0).abs() < 1e-12, "{s}");
        }
        assert!(Boundary::parse("none").unwrap() == Boundary::None);
        assert!(Boundary::parse("box").is_err());
    }

    #[test]
    fn binned_smooth_is_linear_in_y() {
        let x: Vec<f64> = (0..40).map(|i| (i as f64) / 39.0).collect();
        let mut y = Mat::zeros(40, 1);
        for i in 0..40 {
            y.row_mut(i)[0] = x[i];
        }
        let xe: Vec<f64> = (0..5).map(|i| i as f64 / 4.0).collect();
        let s = binned_smooth(&x, &y, 0.25, &xe, 8, Kernel::Gaussian).unwrap();
        assert_eq!(s.rows, 5);
        assert_eq!(s.cols, 1);
        // a line is reproduced up to binning error, which is small at h = 0.25
        for (i, &x0) in xe.iter().enumerate() {
            assert!((s.row(i)[0] - x0).abs() < 0.05, "{} {}", x0, s.row(i)[0]);
        }
    }

    #[test]
    fn global_weights_reproduce_linear() {
        let xt = Mat::from_vec(5, 2, vec![0., 0., 1., 0., 0., 1., 1., 1., 0.5, 0.2]).unwrap();
        let xe = Mat::from_vec(1, 2, vec![0.3, 0.7]).unwrap();
        let w = global_frechet_weights(&xt, &xe).unwrap();
        let y: Vec<f64> = (0..5)
            .map(|i| 1.0 + 2.0 * xt.get(i, 0) - xt.get(i, 1))
            .collect();
        let fit: f64 = w.row(0).iter().zip(&y).map(|(a, b)| a * b).sum();
        assert!((fit - (1.0 + 0.6 - 0.7)).abs() < 1e-12);
    }
}
