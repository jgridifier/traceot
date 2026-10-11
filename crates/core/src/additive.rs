//! Additive backfitting in flat (Euclidean) coordinates.
//!
//! Used for ADOPT under the log-Cholesky metric (where transports are translations
//! of log-Cholesky coordinates) and for the extrinsic L2-additive baseline on
//! quantile functions. Mirrors `python/traceot/reference/additive.py`.

use crate::adopt::{BackfitConfig, BackfitResult, ABS_TOL_FLOOR};
use crate::error::{Result, TraceError};
use crate::linalg::Mat;
use crate::smoother::local_linear_weights;

fn mean_sq_rows(a: &Mat, b: &Mat) -> f64 {
    let mut tot = 0.0;
    for i in 0..a.rows {
        let s: f64 = a
            .row(i)
            .iter()
            .zip(b.row(i))
            .map(|(x, y)| (x - y) * (x - y))
            .sum();
        tot += s / a.cols as f64;
    }
    tot / a.rows as f64
}

/// Backfit `Z_i ≈ mu + sum_j (g_j(X_ij) - M_j)`.
pub fn additive_euclid_fit(
    x: &Mat,
    z: &Mat,
    h: &[f64],
    cfg: &BackfitConfig,
) -> Result<BackfitResult> {
    if x.rows != z.rows || h.len() != x.cols {
        return Err(TraceError::Shape("inconsistent X / Z / h shapes".into()));
    }
    if h.iter().any(|v| !(*v > 0.0)) {
        return Err(TraceError::InvalidArgument("bandwidths must be > 0".into()));
    }
    let (n, p, d) = (x.rows, x.cols, z.cols);
    let mu = z.col_means();
    let mut mu_mat = Mat::zeros(n, d);
    for i in 0..n {
        mu_mat.row_mut(i).copy_from_slice(&mu);
    }
    let v = mean_sq_rows(z, &mu_mat);
    let mut smoothers = Vec::with_capacity(p);
    for (j, &hj) in h.iter().enumerate() {
        let xj = x.col(j);
        smoothers.push(local_linear_weights(&xj, &xj, hj, cfg.kernel)?);
    }
    let mut g: Vec<Mat> = (0..p).map(|_| Mat::zeros(n, d)).collect();
    let mut m: Vec<Vec<f64>> = vec![vec![0.0; d]; p];
    let mut resid: Vec<Mat> = (0..p).map(|_| Mat::zeros(n, d)).collect();
    let mut prev = mu_mat.clone();
    let mut history = Vec::new();
    let mut converged = false;
    let mut it = 0;
    let thresh = (cfg.tol * v).max(ABS_TOL_FLOOR);
    for t in 1..=cfg.max_iter {
        it = t;
        for j in 0..p {
            let mut others = Mat::zeros(n, d);
            for k in 0..p {
                if k != j {
                    for i in 0..n {
                        let orow = others.row_mut(i);
                        for ((o, gv), mv) in orow.iter_mut().zip(g[k].row(i)).zip(&m[k]) {
                            *o += gv - mv;
                        }
                    }
                }
            }
            let mut pj = z.clone();
            for (a, b) in pj.data.iter_mut().zip(&others.data) {
                *a -= b;
            }
            g[j] = smoothers[j].matmul(&pj)?;
            m[j] = g[j].col_means();
            resid[j] = pj;
        }
        let mut fitted = mu_mat.clone();
        for k in 0..p {
            for i in 0..n {
                let frow = fitted.row_mut(i);
                for ((f, gv), mv) in frow.iter_mut().zip(g[k].row(i)).zip(&m[k]) {
                    *f += gv - mv;
                }
            }
        }
        let crit = mean_sq_rows(&fitted, &prev);
        history.push(crit);
        prev = fitted;
        if crit <= thresh {
            converged = true;
            break;
        }
    }
    Ok(BackfitResult {
        mu,
        g,
        m,
        resid,
        fitted: prev,
        n_iter: it,
        converged,
        history,
    })
}

/// Predict `mu + sum_j (g_j(x_j) - M_j)` at the rows of `x_new`.
pub fn additive_euclid_predict(
    x_train: &Mat,
    resid: &[Mat],
    m: &[Vec<f64>],
    mu: &[f64],
    h: &[f64],
    x_new: &Mat,
    cfg: &BackfitConfig,
) -> Result<Mat> {
    let p = x_train.cols;
    if x_new.cols != p || resid.len() != p || m.len() != p || h.len() != p {
        return Err(TraceError::Shape(
            "inconsistent number of predictors".into(),
        ));
    }
    let d = mu.len();
    let mut out = Mat::zeros(x_new.rows, d);
    for i in 0..x_new.rows {
        out.row_mut(i).copy_from_slice(mu);
    }
    for j in 0..p {
        let w = local_linear_weights(&x_train.col(j), &x_new.col(j), h[j], cfg.kernel)?;
        let gj = w.matmul(&resid[j])?;
        for i in 0..x_new.rows {
            let orow = out.row_mut(i);
            for ((o, gv), mv) in orow.iter_mut().zip(gj.row(i)).zip(&m[j]) {
                *o += gv - mv;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_additive_signal() {
        let n = 80;
        let mut x = Mat::zeros(n, 2);
        let mut z = Mat::zeros(n, 2);
        for i in 0..n {
            let a = (i as f64 * 0.618_033_988_7).fract();
            let b = (i as f64 * 0.414_213_562_3).fract();
            x.row_mut(i).copy_from_slice(&[a, b]);
            z.row_mut(i).copy_from_slice(&[a * a + b, (3.0 * b).sin()]);
        }
        let cfg = BackfitConfig::default();
        let res = additive_euclid_fit(&x, &z, &[0.1, 0.1], &cfg).unwrap();
        assert!(res.converged);
        let err = mean_sq_rows(&res.fitted, &z);
        assert!(err < 1e-3, "err {err}");
        let pred = additive_euclid_predict(&x, &res.resid, &res.m, &res.mu, &[0.1, 0.1], &x, &cfg)
            .unwrap();
        for (a, b) in pred.data.iter().zip(&res.fitted.data) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
