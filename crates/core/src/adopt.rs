//! ADOPT transport backfitting for distributional responses in W2.
//!
//! Algorithm 1 of Song & Müller, "ADOPT: Additive Optimal Transport Regression",
//! AISTATS 2026 (PMLR v300). Mirrors `python/traceot/reference/adopt.py`; see that
//! file and `docs/REPRODUCTION_NOTES.md` for the conventions:
//!
//! * `T_j(x) = T_{M_j, g_j(x)}` is the monotone map with knots `M_j -> g_j(x)`;
//! * `A ⊕ B` = "apply A, then B", so `⊕_j T_j` applies `T_1` first;
//! * partial residual (eq. 10): apply `T_{j-1}^{-1}, ..., T_1^{-1}`, then
//!   `T_{µ̂,Y_i}`, then `T_p^{-1}, ..., T_{j+1}^{-1}` to `µ̂`.

use crate::error::{Result, TraceError};
use crate::linalg::Mat;
use crate::smoother::{local_linear_weights, Kernel};
use crate::w2::{pava, w2_sq, TransportMap};

/// Absolute floor of the stopping threshold (handles a constant response, V = 0).
pub const ABS_TOL_FLOOR: f64 = 1e-30;

/// Fitting options shared by the backfitting routines.
#[derive(Debug, Clone, Copy)]
pub struct BackfitConfig {
    pub kernel: Kernel,
    /// Relative tolerance: stop when `mean_i d²(Ŷ_i^(t), Ŷ_i^(t-1)) < tol * V`.
    pub tol: f64,
    pub max_iter: usize,
    /// Optional support bounds for the isotonic projection (W2 only).
    pub lower: Option<f64>,
    pub upper: Option<f64>,
}

impl Default for BackfitConfig {
    fn default() -> Self {
        BackfitConfig {
            kernel: Kernel::Gaussian,
            tol: 1e-10,
            max_iter: 200,
            lower: None,
            upper: None,
        }
    }
}

/// Output of a backfitting run.
#[derive(Debug, Clone)]
pub struct BackfitResult {
    /// Sample Fréchet mean of the responses (length q).
    pub mu: Vec<f64>,
    /// `g_j(X_ij)` for each predictor j (p matrices, n x q).
    pub g: Vec<Mat>,
    /// Fréchet means `Ê⊕ g_j(X_j)` (p vectors of length q).
    pub m: Vec<Vec<f64>>,
    /// Final partial residuals (p matrices, n x q).
    pub resid: Vec<Mat>,
    /// Fitted responses (n x q).
    pub fitted: Mat,
    pub n_iter: usize,
    pub converged: bool,
    /// Convergence criterion per iteration.
    pub history: Vec<f64>,
}

fn check_inputs(x: &Mat, y: &Mat, h: &[f64]) -> Result<()> {
    if x.rows != y.rows {
        return Err(TraceError::Shape(format!(
            "X has {} rows, Y has {}",
            x.rows, y.rows
        )));
    }
    if h.len() != x.cols {
        return Err(TraceError::Shape(format!(
            "{} bandwidths for {} predictors",
            h.len(),
            x.cols
        )));
    }
    if x.rows < 2 || y.cols < 1 {
        return Err(TraceError::InvalidArgument("need n >= 2 and q >= 1".into()));
    }
    if h.iter().any(|v| !(*v > 0.0)) {
        return Err(TraceError::InvalidArgument("bandwidths must be > 0".into()));
    }
    Ok(())
}

/// Composition `[T_1(x_1) ⊕ ... ⊕ T_p(x_p)](µ)` for every row `i` of the stacked `g`.
fn compose_forward(g: &[Mat], m: &[Vec<f64>], mu: &[f64], active: &[bool]) -> Mat {
    let n = g[0].rows;
    let q = mu.len();
    let mut out = Mat::zeros(n, q);
    for i in 0..n {
        let z = out.row_mut(i);
        z.copy_from_slice(mu);
        for k in 0..g.len() {
            if active[k] {
                TransportMap::new(&m[k], g[k].row(i)).apply_inplace(z);
            }
        }
    }
    out
}

/// Partial transport residuals for predictor `j` (eq. 10).
fn partial_residuals(
    y: &Mat,
    g: &[Mat],
    m: &[Vec<f64>],
    mu: &[f64],
    active: &[bool],
    j: usize,
) -> Mat {
    let n = y.rows;
    let p = g.len();
    let mut out = Mat::zeros(n, mu.len());
    for i in 0..n {
        let z = out.row_mut(i);
        z.copy_from_slice(mu);
        for k in (0..j).rev() {
            if active[k] {
                TransportMap::new(g[k].row(i), &m[k]).apply_inplace(z);
            }
        }
        TransportMap::new(mu, y.row(i)).apply_inplace(z);
        for k in (j + 1..p).rev() {
            if active[k] {
                TransportMap::new(g[k].row(i), &m[k]).apply_inplace(z);
            }
        }
    }
    out
}

fn pava_rows(a: &Mat, lower: Option<f64>, upper: Option<f64>) -> Mat {
    let mut out = Mat::zeros(a.rows, a.cols);
    for i in 0..a.rows {
        out.row_mut(i)
            .copy_from_slice(&pava(a.row(i), lower, upper));
    }
    out
}

/// Fit ADOPT on predictors `x` (n x p) and response quantile functions `y` (n x q).
pub fn adopt_w2_fit(x: &Mat, y: &Mat, h: &[f64], cfg: &BackfitConfig) -> Result<BackfitResult> {
    check_inputs(x, y, h)?;
    let (n, p, q) = (x.rows, x.cols, y.cols);
    let mu = y.col_means();
    let v: f64 = (0..n).map(|i| w2_sq(y.row(i), &mu)).sum::<f64>() / n as f64;
    let mut smoothers = Vec::with_capacity(p);
    for (j, &hj) in h.iter().enumerate() {
        let xj = x.col(j);
        smoothers.push(local_linear_weights(&xj, &xj, hj, cfg.kernel)?);
    }
    let mut g: Vec<Mat> = (0..p).map(|_| Mat::zeros(n, q)).collect();
    let mut m: Vec<Vec<f64>> = vec![vec![0.0; q]; p];
    let mut resid: Vec<Mat> = (0..p).map(|_| Mat::zeros(n, q)).collect();
    let mut active = vec![false; p];
    let mut prev = Mat::zeros(n, q);
    for i in 0..n {
        prev.row_mut(i).copy_from_slice(&mu);
    }
    let mut history = Vec::new();
    let mut converged = false;
    let mut it = 0;
    let thresh = (cfg.tol * v).max(ABS_TOL_FLOOR);
    for t in 1..=cfg.max_iter {
        it = t;
        for j in 0..p {
            resid[j] = partial_residuals(y, &g, &m, &mu, &active, j);
            let raw = smoothers[j].matmul(&resid[j])?;
            g[j] = pava_rows(&raw, cfg.lower, cfg.upper);
            m[j] = g[j].col_means();
            active[j] = true;
        }
        let fitted = compose_forward(&g, &m, &mu, &active);
        let crit: f64 = (0..n)
            .map(|i| w2_sq(fitted.row(i), prev.row(i)))
            .sum::<f64>()
            / n as f64;
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

/// `g_j(x)` at new points from the stored residuals of predictor `j`.
pub fn adopt_w2_component(
    x_train_j: &[f64],
    resid_j: &Mat,
    h_j: f64,
    x_eval: &[f64],
    cfg: &BackfitConfig,
) -> Result<Mat> {
    let w = local_linear_weights(x_train_j, x_eval, h_j, cfg.kernel)?;
    Ok(pava_rows(&w.matmul(resid_j)?, cfg.lower, cfg.upper))
}

/// Predict `[T_1(x_1) ⊕ ... ⊕ T_p(x_p)](µ̂)` for each row of `x_new`.
pub fn adopt_w2_predict(
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
    let mut gnew = Vec::with_capacity(p);
    for j in 0..p {
        gnew.push(adopt_w2_component(
            &x_train.col(j),
            &resid[j],
            h[j],
            &x_new.col(j),
            cfg,
        )?);
    }
    Ok(compose_forward(&gnew, m, mu, &vec![true; p]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random numbers in [0, 1) (no external crates).
    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn toy(n: usize, q: usize) -> (Mat, Mat) {
        let mut s = 7u64;
        let u = crate::w2::quantile_grid(q);
        let mut x = Mat::zeros(n, 2);
        let mut y = Mat::zeros(n, q);
        for i in 0..n {
            let (a, b) = (lcg(&mut s), lcg(&mut s));
            x.row_mut(i).copy_from_slice(&[a, b]);
            let row = y.row_mut(i);
            for k in 0..q {
                // location shift in x1 and scale change in x2
                row[k] = (a - 0.5) + (1.0 + 0.5 * b) * (u[k] - 0.5);
            }
        }
        (x, y)
    }

    #[test]
    fn fit_converges_and_fits_well() {
        let (x, y) = toy(60, 20);
        let res = adopt_w2_fit(&x, &y, &[0.2, 0.2], &BackfitConfig::default()).unwrap();
        assert!(res.converged, "history {:?}", res.history);
        let err: f64 = (0..60)
            .map(|i| w2_sq(res.fitted.row(i), y.row(i)))
            .sum::<f64>()
            / 60.0;
        let v: f64 = (0..60).map(|i| w2_sq(y.row(i), &res.mu)).sum::<f64>() / 60.0;
        assert!(err < 0.02 * v, "err {err} v {v}");
        // fitted values are valid quantile functions
        for i in 0..60 {
            assert!(res.fitted.row(i).windows(2).all(|w| w[0] <= w[1] + 1e-12));
        }
        // predicting at the training points reproduces the fitted values
        let pred = adopt_w2_predict(
            &x,
            &res.resid,
            &res.m,
            &res.mu,
            &[0.2, 0.2],
            &x,
            &BackfitConfig::default(),
        )
        .unwrap();
        for (a, b) in pred.data.iter().zip(&res.fitted.data) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn shape_errors() {
        let (x, y) = toy(10, 5);
        assert!(adopt_w2_fit(&x, &y, &[0.2], &BackfitConfig::default()).is_err());
        assert!(adopt_w2_fit(&x, &y, &[0.2, -1.0], &BackfitConfig::default()).is_err());
    }
}
