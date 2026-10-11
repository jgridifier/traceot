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
