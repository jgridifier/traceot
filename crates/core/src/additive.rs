//! Additive backfitting in flat (Euclidean) coordinates.
//!
//! Used for ADOPT under the log-Cholesky metric (where transports are translations
//! of log-Cholesky coordinates) and for the extrinsic L2-additive baseline on
//! quantile functions. Mirrors `tests/python/reference/additive.py`.

use crate::adopt::{BackfitConfig, BackfitResult, ABS_TOL_FLOOR};
use crate::error::{Result, TraceError};
use crate::linalg::Mat;
use crate::smoother::{binned_smooth, local_linear_weights, local_linear_weights_boundary};

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
    cfg.validate()?;
    let v = mean_sq_rows(z, &mu_mat);
    // Paper path: keep the original accumulation order so results stay
    // identical to the dense undamped fit.
    if cfg.damping == 1.0 && !cfg.certificate && cfg.dense_smoother() {
        return additive_euclid_fit_paper(x, z, h, cfg, mu, mu_mat, v);
    }
    let mut smoothers = Vec::with_capacity(p);
    for (j, &hj) in h.iter().enumerate() {
        if cfg.dense_smoother() {
            let xj = x.col(j);
            smoothers.push(Some(local_linear_weights(&xj, &xj, hj, cfg.kernel)?));
        } else {
            smoothers.push(None);
        }
    }
    let mut g: Vec<Mat> = (0..p).map(|_| Mat::zeros(n, d)).collect();
    let mut m: Vec<Vec<f64>> = vec![vec![0.0; d]; p];
    let mut resid: Vec<Mat> = (0..p).map(|_| Mat::zeros(n, d)).collect();
    let mut prev = mu_mat.clone();
    let mut loss_prev = mean_sq_rows(z, &mu_mat);
    let mut history = Vec::new();
    let mut converged = false;
    let mut it = 0usize;
    let mut n_rejected = 0usize;
    let thresh = (cfg.tol * v).max(ABS_TOL_FLOOR);
    for t in 1..=cfg.max_iter {
        it = t;
        let snap_g = g.clone();
        let snap_m = m.clone();
        let snap_resid = resid.clone();
        let mut lambda = cfg.damping;
        let mut accepted = false;
        loop {
            g.clone_from(&snap_g);
            m.clone_from(&snap_m);
            resid.clone_from(&snap_resid);
            for j in 0..p {
                let pj = partial_euclid(z, &g, &m, j);
                let raw = smooth_one(&x.col(j), &pj, h[j], smoothers[j].as_ref(), cfg)?;
                let g_new = if lambda == 1.0 {
                    raw
                } else {
                    blend(&g[j], &raw, lambda)
                };
                resid[j] = pj;
                g[j] = g_new;
                m[j] = g[j].col_means();
            }
            let fitted = compose_euclid(&mu, &g, &m);
            if !cfg.certificate {
                prev = fitted;
                accepted = true;
                break;
            }
            let loss = mean_sq_rows(z, &fitted);
            if loss <= loss_prev + 1e-12 {
                loss_prev = loss;
                prev = fitted;
                accepted = true;
                break;
            }
            n_rejected += 1;
            lambda *= 0.5;
            if lambda < 1e-8 {
                g.clone_from(&snap_g);
                m.clone_from(&snap_m);
                resid.clone_from(&snap_resid);
                converged = true;
                it = t - 1;
                break;
            }
        }
        if !accepted {
            break;
        }
        let crit = mean_sq_rows(&prev, &{
            // `prev` was just overwritten with the new fitted value. Compare
            // against the snapshot composition instead.
            compose_euclid(&mu, &snap_g, &snap_m)
        });
        history.push(crit);
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
        n_rejected,
        orders: Vec::new(),
    })
}

#[allow(clippy::too_many_arguments)]
fn additive_euclid_fit_paper(
    x: &Mat,
    z: &Mat,
    h: &[f64],
    cfg: &BackfitConfig,
    mu: Vec<f64>,
    mu_mat: Mat,
    v: f64,
) -> Result<BackfitResult> {
    let (n, p, d) = (x.rows, x.cols, z.cols);
    let mut smoothers = Vec::with_capacity(p);
    for (j, &hj) in h.iter().enumerate() {
        let xj = x.col(j);
        smoothers.push(local_linear_weights(&xj, &xj, hj, cfg.kernel)?);
    }
    let mut g: Vec<Mat> = (0..p).map(|_| Mat::zeros(n, d)).collect();
    let mut m: Vec<Vec<f64>> = vec![vec![0.0; d]; p];
    let mut resid: Vec<Mat> = (0..p).map(|_| Mat::zeros(n, d)).collect();
    let mut prev = mu_mat;
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
        let mut fitted = Mat::zeros(n, d);
        for i in 0..n {
            fitted.row_mut(i).copy_from_slice(&mu);
        }
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
        n_rejected: 0,
        orders: Vec::new(),
    })
}

fn partial_euclid(z: &Mat, g: &[Mat], m: &[Vec<f64>], j: usize) -> Mat {
    let mut pj = z.clone();
    for k in 0..g.len() {
        if k == j {
            continue;
        }
        for i in 0..z.rows {
            let row = pj.row_mut(i);
            for ((a, gv), mv) in row.iter_mut().zip(g[k].row(i)).zip(&m[k]) {
                *a -= gv - mv;
            }
        }
    }
    pj
}

fn compose_euclid(mu: &[f64], g: &[Mat], m: &[Vec<f64>]) -> Mat {
    let n = g[0].rows;
    let mut fitted = Mat::zeros(n, mu.len());
    for i in 0..n {
        fitted.row_mut(i).copy_from_slice(mu);
    }
    for k in 0..g.len() {
        for i in 0..n {
            let row = fitted.row_mut(i);
            for ((f, gv), mv) in row.iter_mut().zip(g[k].row(i)).zip(&m[k]) {
                *f += gv - mv;
            }
        }
    }
    fitted
}

fn blend(old: &Mat, new: &Mat, lambda: f64) -> Mat {
    let mut out = new.clone();
    let keep = 1.0 - lambda;
    for (o, a) in out.data.iter_mut().zip(&old.data) {
        *o = keep * *a + lambda * *o;
    }
    out
}

fn smooth_one(
    x: &[f64],
    resid: &Mat,
    h: f64,
    prepared: Option<&Mat>,
    cfg: &BackfitConfig,
) -> Result<Mat> {
    if let Some(w) = prepared {
        return w.matmul(resid);
    }
    if let Some(bins) = cfg.bins {
        binned_smooth(x, resid, h, x, bins, cfg.kernel)
    } else {
        local_linear_weights_boundary(x, x, h, cfg.kernel, cfg.boundary)?.matmul(resid)
    }
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
        let gj = if let Some(bins) = cfg.bins {
            binned_smooth(
                &x_train.col(j),
                &resid[j],
                h[j],
                &x_new.col(j),
                bins,
                cfg.kernel,
            )?
        } else if cfg.dense_smoother() {
            local_linear_weights(&x_train.col(j), &x_new.col(j), h[j], cfg.kernel)?
                .matmul(&resid[j])?
        } else {
            local_linear_weights_boundary(
                &x_train.col(j),
                &x_new.col(j),
                h[j],
                cfg.kernel,
                cfg.boundary,
            )?
            .matmul(&resid[j])?
        };
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
