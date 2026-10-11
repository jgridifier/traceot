//! ADOPT transport backfitting for distributional responses in W2.
//!
//! Algorithm 1 of Song & Müller, "ADOPT: Additive Optimal Transport Regression",
//! AISTATS 2026 (PMLR v300). Mirrors `tests/python/reference/adopt.py`; see that
//! file and `docs/REPRODUCTION_NOTES.md` for the conventions:
//!
//! * `T_j(x) = T_{M_j, g_j(x)}` is the monotone map with knots `M_j -> g_j(x)`;
//! * `A ⊕ B` = "apply A, then B", so `⊕_j T_j` applies `T_1` first;
//! * partial residual (eq. 10): apply `T_{j-1}^{-1}, ..., T_1^{-1}`, then
//!   `T_{µ̂,Y_i}`, then `T_p^{-1}, ..., T_{j+1}^{-1}` to `µ̂`.

use crate::error::{Result, TraceError};
use crate::linalg::Mat;
use crate::smoother::{
    binned_smooth, local_linear_weights, local_linear_weights_boundary, Boundary, Kernel,
};
use crate::w2::{pava, w2_sq, TransportMap};

/// Order in which component transports are fitted and composed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Composition {
    /// Paper order: features `1..p` in the order given (Gauss–Seidel, Algorithm 1).
    #[default]
    Given,
    /// Fit every permutation and average the composed predictions.
    Average,
    /// Fit additive components in the L2 tangent space at the barycenter, then
    /// map back (add the mean quantile) and project onto monotone functions.
    Tangent,
}

impl Composition {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "given" => Ok(Composition::Given),
            "average" => Ok(Composition::Average),
            "tangent" => Ok(Composition::Tangent),
            other => Err(TraceError::InvalidArgument(format!(
                "unknown composition {other:?} (expected 'given', 'average' or 'tangent')"
            ))),
        }
    }
}

/// How component transports are anchored after each smoothing step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Centering {
    /// Paper eq. (11): the anchor is the sample Fréchet mean of `g_j(X_j)`.
    #[default]
    Frechet,
    /// Push each component through `T_{M_j, g_j}` applied to the global barycenter,
    /// subtract the average displacement, and re-project. The average pushforward
    /// of the barycenter is then the barycenter (up to the final isotonic step).
    Pushforward,
}

impl Centering {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "frechet" => Ok(Centering::Frechet),
            "pushforward" => Ok(Centering::Pushforward),
            other => Err(TraceError::InvalidArgument(format!(
                "unknown centering {other:?} (expected 'frechet' or 'pushforward')"
            ))),
        }
    }
}

/// Absolute floor of the stopping threshold (handles a constant response, V = 0).
pub const ABS_TOL_FLOOR: f64 = 1e-30;

/// Fitting options shared by the backfitting routines.
///
/// The defaults are the paper's Algorithm 1: undamped Gauss–Seidel in the given
/// feature order, Fréchet centering, dense local-linear weights, no extra
/// projection of the composed prediction.
#[derive(Debug, Clone, Copy)]
pub struct BackfitConfig {
    pub kernel: Kernel,
    /// Relative tolerance: stop when `mean_i d²(Ŷ_i^(t), Ŷ_i^(t-1)) < tol * V`.
    pub tol: f64,
    pub max_iter: usize,
    /// Optional support bounds for the isotonic projection (W2 only).
    pub lower: Option<f64>,
    pub upper: Option<f64>,
    /// Step length in `(0, 1]`. `1` is the paper's full Gauss–Seidel step.
    pub damping: f64,
    /// If set, accept a cycle only when the in-sample mean `d²(Y, Ŷ)` does not
    /// increase; otherwise halve `damping` and retry. Off by default.
    pub certificate: bool,
    pub composition: Composition,
    pub centering: Centering,
    /// Project the composed prediction onto monotone quantile functions.
    pub project_prediction: bool,
    /// Equal-width bins per predictor. `None` is the paper's dense `O(n²)` smoother.
    pub bins: Option<usize>,
    pub boundary: Boundary,
}

impl Default for BackfitConfig {
    fn default() -> Self {
        BackfitConfig {
            kernel: Kernel::Gaussian,
            tol: 1e-10,
            max_iter: 200,
            lower: None,
            upper: None,
            damping: 1.0,
            certificate: false,
            composition: Composition::Given,
            centering: Centering::Frechet,
            project_prediction: false,
            bins: None,
            boundary: Boundary::None,
        }
    }
}

impl BackfitConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if !(self.damping > 0.0 && self.damping <= 1.0) {
            return Err(TraceError::InvalidArgument(
                "damping must lie in (0, 1]".into(),
            ));
        }
        if let Some(b) = self.bins {
            if b < 1 {
                return Err(TraceError::InvalidArgument(
                    "bins must be >= 1 when set".into(),
                ));
            }
        }
        if !(self.tol >= 0.0) || self.max_iter < 1 {
            return Err(TraceError::InvalidArgument(
                "tol must be >= 0 and max_iter >= 1".into(),
            ));
        }
        Ok(())
    }

    /// Paper path: dense local-linear weights, no reflection and no binning.
    pub(crate) fn dense_smoother(&self) -> bool {
        self.bins.is_none() && self.boundary == Boundary::None
    }
}

/// One permutation's components, kept so averaged-order prediction can
/// re-smooth each order at new inputs.
#[derive(Debug, Clone)]
pub struct OrderFit {
    /// `perm[k]` is the original feature index sitting in position `k`.
    pub perm: Vec<usize>,
    pub g: Vec<Mat>,
    pub m: Vec<Vec<f64>>,
    pub resid: Vec<Mat>,
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
    /// Cycles rejected by the convergence certificate.
    pub n_rejected: usize,
    /// Empty unless [`Composition::Average`].
    pub orders: Vec<OrderFit>,
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

fn blend_toward(old: &Mat, new: &Mat, lambda: f64) -> Mat {
    let mut out = new.clone();
    let keep = 1.0 - lambda;
    for (o, a) in out.data.iter_mut().zip(&old.data) {
        *o = keep * *a + lambda * *o;
    }
    out
}

fn mean_w2(a: &Mat, b: &Mat) -> f64 {
    let n = a.rows as f64;
    (0..a.rows).map(|i| w2_sq(a.row(i), b.row(i))).sum::<f64>() / n
}

/// Smooth `resid` at the training locations of column `x`.
fn smooth_train(
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

/// Center so the average pushforward of `mu` is `mu`, then restore monotonicity.
fn pushforward_center(g: &Mat, mu: &[f64], cfg: &BackfitConfig) -> (Mat, Vec<f64>) {
    let (n, q) = (g.rows, g.cols);
    let anchor = g.col_means();
    let mut pushed = Mat::zeros(n, q);
    for i in 0..n {
        let z = pushed.row_mut(i);
        z.copy_from_slice(mu);
        TransportMap::new(&anchor, g.row(i)).apply_inplace(z);
    }
    let qbar = pushed.col_means();
    let mut shifted = Mat::zeros(n, q);
    for i in 0..n {
        let src = pushed.row(i);
        let dst = shifted.row_mut(i);
        for k in 0..q {
            dst[k] = src[k] - qbar[k] + mu[k];
        }
    }
    let mut out = pava_rows(&shifted, cfg.lower, cfg.upper);
    // Column-center back onto mu. A level-dependent shift can break monotonicity,
    // so project once more. PAVA preserves each row sum, not the column means;
    // the column means of the result are the anchors we store.
    let col = out.col_means();
    for i in 0..n {
        let row = out.row_mut(i);
        for k in 0..q {
            row[k] += mu[k] - col[k];
        }
    }
    out = pava_rows(&out, cfg.lower, cfg.upper);
    let center = out.col_means();
    (out, center)
}

fn update_component(
    g_old: &Mat,
    raw: &Mat,
    mu: &[f64],
    lambda: f64,
    cfg: &BackfitConfig,
) -> (Mat, Vec<f64>) {
    let mut g_new = pava_rows(raw, cfg.lower, cfg.upper);
    let mut m_new = g_new.col_means();
    if cfg.centering == Centering::Pushforward {
        let (gc, mc) = pushforward_center(&g_new, mu, cfg);
        g_new = gc;
        m_new = mc;
    }
    if lambda != 1.0 {
        g_new = blend_toward(g_old, &g_new, lambda);
        if cfg.centering == Centering::Pushforward {
            let (gc, mc) = pushforward_center(&g_new, mu, cfg);
            g_new = gc;
            m_new = mc;
        } else {
            m_new = g_new.col_means();
        }
    }
    (g_new, m_new)
}

fn finish_prediction(fitted: Mat, cfg: &BackfitConfig) -> Mat {
    if cfg.project_prediction || cfg.composition == Composition::Tangent {
        pava_rows(&fitted, cfg.lower, cfg.upper)
    } else {
        fitted
    }
}

#[derive(Clone)]
struct Sweep {
    g: Vec<Mat>,
    m: Vec<Vec<f64>>,
    resid: Vec<Mat>,
    active: Vec<bool>,
    fitted: Mat,
}

#[allow(clippy::too_many_arguments)]
fn gauss_seidel_cycle(
    x: &Mat,
    y: &Mat,
    h: &[f64],
    mu: &[f64],
    smoothers: &[Option<Mat>],
    state: &Sweep,
    lambda: f64,
    cfg: &BackfitConfig,
) -> Result<Sweep> {
    let p = x.cols;
    let mut g = state.g.clone();
    let mut m = state.m.clone();
    let mut resid = state.resid.clone();
    let mut active = state.active.clone();
    for j in 0..p {
        resid[j] = partial_residuals(y, &g, &m, mu, &active, j);
        let raw = smooth_train(&x.col(j), &resid[j], h[j], smoothers[j].as_ref(), cfg)?;
        let (g_new, m_new) = update_component(&g[j], &raw, mu, lambda, cfg);
        g[j] = g_new;
        m[j] = m_new;
        active[j] = true;
    }
    let fitted = compose_forward(&g, &m, mu, &active);
    Ok(Sweep {
        g,
        m,
        resid,
        active,
        fitted,
    })
}

fn empty_result(mu: Vec<f64>, fitted: Mat) -> BackfitResult {
    BackfitResult {
        mu,
        g: Vec::new(),
        m: Vec::new(),
        resid: Vec::new(),
        fitted,
        n_iter: 0,
        converged: false,
        history: Vec::new(),
        n_rejected: 0,
        orders: Vec::new(),
    }
}

/// All permutations of `0..p`, in lexicographic order.
fn permutations(p: usize) -> Vec<Vec<usize>> {
    let mut idx: Vec<usize> = (0..p).collect();
    let mut out = Vec::new();
    fn rec(idx: &mut Vec<usize>, start: usize, out: &mut Vec<Vec<usize>>) {
        if start == idx.len() {
            out.push(idx.clone());
            return;
        }
        for i in start..idx.len() {
            idx.swap(start, i);
            rec(idx, start + 1, out);
            idx.swap(start, i);
        }
    }
    rec(&mut idx, 0, &mut out);
    out.sort();
    out
}

fn permute_cols(x: &Mat, perm: &[usize]) -> Mat {
    let mut out = Mat::zeros(x.rows, x.cols);
    for i in 0..x.rows {
        let src = x.row(i);
        let dst = out.row_mut(i);
        for (k, &j) in perm.iter().enumerate() {
            dst[k] = src[j];
        }
    }
    out
}

/// Fit ADOPT on predictors `x` (n x p) and response quantile functions `y` (n x q).
pub fn adopt_w2_fit(x: &Mat, y: &Mat, h: &[f64], cfg: &BackfitConfig) -> Result<BackfitResult> {
    check_inputs(x, y, h)?;
    cfg.validate()?;
    match cfg.composition {
        Composition::Average => adopt_w2_fit_average(x, y, h, cfg),
        Composition::Tangent => adopt_w2_fit_tangent(x, y, h, cfg),
        Composition::Given => adopt_w2_fit_given(x, y, h, cfg),
    }
}

fn adopt_w2_fit_given(x: &Mat, y: &Mat, h: &[f64], cfg: &BackfitConfig) -> Result<BackfitResult> {
    let (n, p, q) = (x.rows, x.cols, y.cols);
    let mu = y.col_means();
    let v: f64 = (0..n).map(|i| w2_sq(y.row(i), &mu)).sum::<f64>() / n as f64;
    // The paper path precomputes the same n×n weight matrices as before.
    // Optional smoothers are applied on the fly and never form that matrix.
    let mut smoothers: Vec<Option<Mat>> = Vec::with_capacity(p);
    for (j, &hj) in h.iter().enumerate() {
        if cfg.dense_smoother() {
            let xj = x.col(j);
            smoothers.push(Some(local_linear_weights(&xj, &xj, hj, cfg.kernel)?));
        } else {
            smoothers.push(None);
        }
    }
    let mut state = Sweep {
        g: (0..p).map(|_| Mat::zeros(n, q)).collect(),
        m: vec![vec![0.0; q]; p],
        resid: (0..p).map(|_| Mat::zeros(n, q)).collect(),
        active: vec![false; p],
        fitted: Mat::zeros(n, q),
    };
    for i in 0..n {
        state.fitted.row_mut(i).copy_from_slice(&mu);
    }
    let mut prev = state.fitted.clone();
    let mut loss_prev = mean_w2(&state.fitted, y);
    let mut history = Vec::new();
    let mut converged = false;
    let mut it = 0usize;
    let mut n_rejected = 0usize;
    let thresh = (cfg.tol * v).max(ABS_TOL_FLOOR);
    for t in 1..=cfg.max_iter {
        it = t;
        let mut lambda = cfg.damping;
        let mut accepted: Option<Sweep> = None;
        loop {
            let sweep = gauss_seidel_cycle(x, y, h, &mu, &smoothers, &state, lambda, cfg)?;
            if !cfg.certificate {
                accepted = Some(sweep);
                break;
            }
            let loss = mean_w2(&sweep.fitted, y);
            if loss <= loss_prev + 1e-12 {
                loss_prev = loss;
                accepted = Some(sweep);
                break;
            }
            n_rejected += 1;
            lambda *= 0.5;
            if lambda < 1e-8 {
                // No damped step improves the in-sample loss. Keep the last
                // accepted cycle and stop: the certificate says we are done.
                converged = true;
                it = t - 1;
                break;
            }
        }
        let Some(sweep) = accepted else {
            break;
        };
        state = sweep;
        let crit = mean_w2(&state.fitted, &prev);
        history.push(crit);
        prev = state.fitted.clone();
        if crit <= thresh {
            converged = true;
            break;
        }
    }
    let fitted = finish_prediction(state.fitted, cfg);
    Ok(BackfitResult {
        mu,
        g: state.g,
        m: state.m,
        resid: state.resid,
        fitted,
        n_iter: it,
        converged,
        history,
        n_rejected,
        orders: Vec::new(),
    })
}

fn adopt_w2_fit_average(x: &Mat, y: &Mat, h: &[f64], cfg: &BackfitConfig) -> Result<BackfitResult> {
    let p = x.cols;
    if p > 5 {
        return Err(TraceError::InvalidArgument(
            "averaged-order composition supports at most 5 features".into(),
        ));
    }
    let mut inner = *cfg;
    inner.composition = Composition::Given;
    inner.project_prediction = false;
    let perms = permutations(p);
    let mut orders = Vec::with_capacity(perms.len());
    let mut n_iter = 0usize;
    let mut converged = true;
    let mut n_rejected = 0usize;
    let mu = y.col_means();
    for perm in &perms {
        let xp = permute_cols(x, perm);
        let hp: Vec<f64> = perm.iter().map(|&j| h[j]).collect();
        let fit = adopt_w2_fit_given(&xp, y, &hp, &inner)?;
        n_iter = n_iter.max(fit.n_iter);
        converged &= fit.converged;
        n_rejected += fit.n_rejected;
        orders.push(OrderFit {
            perm: perm.clone(),
            g: fit.g,
            m: fit.m,
            resid: fit.resid,
        });
    }
    // Training prediction is the average of the composed predictions.
    let mut acc = Mat::zeros(x.rows, y.cols);
    for order in &orders {
        let active = vec![true; p];
        let fitted = compose_forward(&order.g, &order.m, &mu, &active);
        for (a, b) in acc.data.iter_mut().zip(&fitted.data) {
            *a += *b;
        }
    }
    let scale = 1.0 / orders.len() as f64;
    acc.data.iter_mut().for_each(|v| *v *= scale);
    // Representative components: average of each feature's curve across orders,
    // used for `component` / inspection. Prediction itself averages compositions.
    let (n, q) = (x.rows, y.cols);
    let mut g: Vec<Mat> = (0..p).map(|_| Mat::zeros(n, q)).collect();
    let mut m = vec![vec![0.0; q]; p];
    let mut resid: Vec<Mat> = (0..p).map(|_| Mat::zeros(n, q)).collect();
    for order in &orders {
        for (k, &j) in order.perm.iter().enumerate() {
            for (a, b) in g[j].data.iter_mut().zip(&order.g[k].data) {
                *a += *b * scale;
            }
            for (a, b) in resid[j].data.iter_mut().zip(&order.resid[k].data) {
                *a += *b * scale;
            }
            for (a, b) in m[j].iter_mut().zip(&order.m[k]) {
                *a += *b * scale;
            }
        }
    }
    let fitted = finish_prediction(acc, cfg);
    let mut out = empty_result(mu, fitted);
    out.g = g;
    out.m = m;
    out.resid = resid;
    out.n_iter = n_iter;
    out.converged = converged;
    out.n_rejected = n_rejected;
    out.orders = orders;
    Ok(out)
}

fn adopt_w2_fit_tangent(x: &Mat, y: &Mat, h: &[f64], cfg: &BackfitConfig) -> Result<BackfitResult> {
    // Tangent-space fit is Euclidean additive backfitting. The caller (estimators)
    // dispatches that path; this entry point is used when the W2 fitter is asked
    // for the tangent option directly.
    let eucl = crate::additive::additive_euclid_fit(x, y, h, cfg)?;
    let mut fitted = Mat::zeros(eucl.fitted.rows, eucl.fitted.cols);
    for i in 0..fitted.rows {
        fitted.row_mut(i).copy_from_slice(&eucl.mu);
    }
    for j in 0..eucl.g.len() {
        for i in 0..fitted.rows {
            let row = fitted.row_mut(i);
            for ((r, gv), mv) in row.iter_mut().zip(eucl.g[j].row(i)).zip(&eucl.m[j]) {
                *r += gv - mv;
            }
        }
    }
    fitted = pava_rows(&fitted, cfg.lower, cfg.upper);
    Ok(BackfitResult {
        fitted,
        orders: Vec::new(),
        ..eucl
    })
}

/// Smooth residual rows at `x_eval`.
fn smooth_at(
    x_train: &[f64],
    resid: &Mat,
    h: f64,
    x_eval: &[f64],
    cfg: &BackfitConfig,
) -> Result<Mat> {
    if let Some(bins) = cfg.bins {
        return binned_smooth(x_train, resid, h, x_eval, bins, cfg.kernel);
    }
    if cfg.boundary == Boundary::None {
        return local_linear_weights(x_train, x_eval, h, cfg.kernel)?.matmul(resid);
    }
    local_linear_weights_boundary(x_train, x_eval, h, cfg.kernel, cfg.boundary)?.matmul(resid)
}

/// `g_j(x)` at new points from the stored residuals of predictor `j`.
pub fn adopt_w2_component(
    x_train_j: &[f64],
    resid_j: &Mat,
    h_j: f64,
    x_eval: &[f64],
    cfg: &BackfitConfig,
) -> Result<Mat> {
    let raw = smooth_at(x_train_j, resid_j, h_j, x_eval, cfg)?;
    Ok(pava_rows(&raw, cfg.lower, cfg.upper))
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
    let fitted = compose_forward(&gnew, m, mu, &vec![true; p]);
    Ok(finish_prediction(fitted, cfg))
}

/// Predict by averaging the composed map over every stored feature order.
///
/// `h` is in the original feature order. Each stored order was fitted on columns
/// permuted by `order.perm`, so new inputs are permuted the same way before
/// that order's map is applied.
pub fn adopt_w2_predict_averaged(
    x_train: &Mat,
    orders: &[OrderFit],
    mu: &[f64],
    h: &[f64],
    x_new: &Mat,
    cfg: &BackfitConfig,
) -> Result<Mat> {
    if orders.is_empty() {
        return Err(TraceError::InvalidArgument(
            "averaged prediction needs at least one stored order".into(),
        ));
    }
    let mut acc = Mat::zeros(x_new.rows, mu.len());
    let mut cfg_one = *cfg;
    cfg_one.composition = Composition::Given;
    cfg_one.project_prediction = false;
    for order in orders {
        let xp = permute_cols(x_train, &order.perm);
        let xn = permute_cols(x_new, &order.perm);
        let hp: Vec<f64> = order.perm.iter().map(|&j| h[j]).collect();
        let pred = adopt_w2_predict(&xp, &order.resid, &order.m, mu, &hp, &xn, &cfg_one)?;
        for (a, b) in acc.data.iter_mut().zip(&pred.data) {
            *a += *b;
        }
    }
    let scale = 1.0 / orders.len() as f64;
    acc.data.iter_mut().for_each(|v| *v *= scale);
    Ok(finish_prediction(acc, cfg))
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
