//! Wasserstein-robust pinball risk on the histogram grid.
//!
//! The exact dual (Theorem 4.1) is the objective. The max-jump formula is
//! returned only as a labelled upper bound. Outcome-only W1 adds the constant
//! `ε ρ̄ / κ_y` and is not a regulariser.

#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use crate::error::{Result, TraceError};
use crate::scalable::binning::{bin_column_into, edges_from_sorted};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ball {
    W1,
    W2,
    WInf,
}

#[derive(Clone, Debug)]
pub struct RobustModel {
    pub levels: Vec<f64>,
    pub weights: Vec<f64>,
    pub edges: Vec<Vec<f64>>,
    pub n_bins: Vec<usize>,
    /// `intercept[level]`
    pub intercept: Vec<f64>,
    /// `effects[level][feature][bin]`
    pub effects: Vec<Vec<Vec<f64>>>,
    pub kappa: Vec<f64>,
    pub eps: f64,
    pub ball: Ball,
    /// `Some(κ_y)` for a joint cost. `None` is covariate-only.
    pub kappa_y: Option<f64>,
    pub train_loss: f64,
    pub robust_value: f64,
}

#[derive(Clone, Debug)]
pub struct RobustConfig {
    pub levels: Vec<f64>,
    pub max_bins: usize,
    pub rounds: usize,
    pub step: f64,
    pub eps: f64,
    pub kappa: Vec<f64>,
    pub ball: Ball,
    pub kappa_y: Option<f64>,
}

impl Default for RobustConfig {
    fn default() -> Self {
        Self {
            levels: vec![0.1, 0.3, 0.5, 0.7, 0.9],
            max_bins: 8,
            rounds: 60,
            step: 0.6,
            eps: 0.0,
            kappa: Vec::new(),
            ball: Ball::W1,
            kappa_y: None,
        }
    }
}

pub fn fit_robust(
    x: &[f64],
    n: usize,
    p: usize,
    y: &[f64],
    cfg: &RobustConfig,
) -> Result<RobustModel> {
    if n < 2 || p < 1 || cfg.levels.is_empty() {
        return Err(TraceError::InvalidArgument(
            "robust fit needs rows, features and quantile levels".into(),
        ));
    }
    if x.len() != n * p || y.len() != n || y.iter().any(|v| !v.is_finite()) {
        return Err(TraceError::InvalidArgument(
            "X and y must be finite and aligned".into(),
        ));
    }
    let (codes, edges, n_bins) = bin_matrix(x, n, p, cfg.max_bins)?;
    let kappa = if cfg.kappa.is_empty() {
        default_kappa(&edges)
    } else if cfg.kappa.len() == p {
        cfg.kappa.clone()
    } else {
        return Err(TraceError::InvalidArgument(
            "kappa must be empty or have one entry per feature".into(),
        ));
    };
    let mut intercept: Vec<f64> = cfg
        .levels
        .iter()
        .map(|&u| empirical_quantile(y, u))
        .collect();
    let mut effects: Vec<Vec<Vec<f64>>> = cfg
        .levels
        .iter()
        .map(|_| (0..p).map(|j| vec![0.0; n_bins[j]]).collect())
        .collect();
    let mut weights_u = vec![1.0 / cfg.levels.len() as f64; cfg.levels.len()];
    if cfg.levels.len() > 1 {
        // Trapezoid-like equal weights already sum to 1. Keep them equal.
        let _ = &mut weights_u;
    }
    for level in 0..cfg.levels.len() {
        let u = cfg.levels[level];
        for round in 0..cfg.rounds {
            let (g0, ge) = subgradient(
                &intercept[level],
                &effects[level],
                &codes,
                y,
                u,
                &kappa,
                cfg.eps,
                cfg.ball,
                cfg.kappa_y,
            )?;
            let lr = cfg.step / (1.0 + round as f64).sqrt();
            intercept[level] -= lr * g0;
            for j in 0..p {
                for b in 0..n_bins[j] {
                    effects[level][j][b] -= lr * ge[j][b];
                }
            }
            center_level(&mut intercept[level], &mut effects[level], &codes);
        }
    }
    let train_loss = integrated_pinball(&intercept, &effects, &codes, y, &cfg.levels, &weights_u);
    let robust_value = integrated_robust(
        &intercept,
        &effects,
        &codes,
        y,
        &cfg.levels,
        &weights_u,
        &kappa,
        cfg.eps,
        cfg.ball,
        cfg.kappa_y,
    )?;
    Ok(RobustModel {
        levels: cfg.levels.clone(),
        weights: weights_u,
        edges,
        n_bins,
        intercept,
        effects,
        kappa,
        eps: cfg.eps,
        ball: cfg.ball,
        kappa_y: cfg.kappa_y,
        train_loss,
        robust_value,
    })
}

fn center_level(intercept: &mut f64, effects: &mut [Vec<f64>], codes: &[Vec<u16>]) {
    let n = codes[0].len() as f64;
    for (j, col) in codes.iter().enumerate() {
        let nb = effects[j].len();
        let mut cnt = vec![0.0; nb];
        for &c in col {
            cnt[c as usize] += 1.0;
        }
        let mut mean = 0.0;
        for b in 0..nb {
            mean += cnt[b] * effects[j][b];
        }
        mean /= n;
        *intercept += mean;
        for b in 0..nb {
            effects[j][b] -= mean;
        }
    }
}

fn subgradient(
    intercept: &f64,
    effects: &[Vec<f64>],
    codes: &[Vec<u16>],
    y: &[f64],
    u: f64,
    kappa: &[f64],
    eps: f64,
    ball: Ball,
    kappa_y: Option<f64>,
) -> Result<(f64, Vec<Vec<f64>>)> {
    let n = y.len();
    let p = effects.len();
    let bins: Vec<Vec<usize>> = (0..n)
        .map(|i| (0..p).map(|j| codes[j][i] as usize).collect())
        .collect();
    let f = effects;
    let (lam, _) = best_lambda(intercept, f, &bins, y, u, kappa, eps, ball, kappa_y)?;
    let mut g0 = 0.0;
    let mut ge: Vec<Vec<f64>> = f.iter().map(|col| vec![0.0; col.len()]).collect();
    for i in 0..n {
        let (_, chosen, _, slope) =
            inner_choice(*intercept, f, &bins[i], y[i], u, kappa, lam, ball);
        // Pinball's slope in the quantile is −s, with s the active check weight.
        let scale = 1.0 / n as f64;
        g0 += scale * (-slope);
        for j in 0..p {
            ge[j][chosen[j]] += scale * (-slope);
        }
    }
    Ok((g0, ge))
}

fn best_lambda(
    intercept: &f64,
    f: &[Vec<f64>],
    bins: &[Vec<usize>],
    y: &[f64],
    u: f64,
    kappa: &[f64],
    eps: f64,
    ball: Ball,
    kappa_y: Option<f64>,
) -> Result<(f64, f64)> {
    let rbar = u.max(1.0 - u);
    let lam_min = match kappa_y {
        Some(ky) if ky > 0.0 => rbar / ky,
        _ => 0.0,
    };
    if eps == 0.0 && lam_min == 0.0 {
        let val = y
            .iter()
            .enumerate()
            .map(|(i, &yi)| pinball(yi - predict_at(*intercept, f, &bins[i]), u))
            .sum::<f64>()
            / y.len() as f64;
        return Ok((f64::INFINITY, val));
    }
    let mut hi = (1.0_f64).max(lam_min);
    while hi < 1e8 && mean_dist(*intercept, f, bins, y, u, kappa, hi, ball) > eps {
        hi *= 2.0;
    }
    let mut lo = lam_min;
    if hi < lo {
        hi = lo;
    }
    for _ in 0..50 {
        let mid = 0.5 * (lo + hi);
        if mean_dist(*intercept, f, bins, y, u, kappa, mid, ball) > eps {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let lam = 0.5 * (lo + hi);
    let val = lam * eps + mean_inner(*intercept, f, bins, y, u, kappa, lam, ball);
    Ok((lam, val))
}

fn mean_dist(
    intercept: f64,
    f: &[Vec<f64>],
    bins: &[Vec<usize>],
    y: &[f64],
    u: f64,
    kappa: &[f64],
    lam: f64,
    ball: Ball,
) -> f64 {
    let n = y.len() as f64;
    bins.iter()
        .zip(y)
        .map(|(b, &yi)| inner_choice(intercept, f, b, yi, u, kappa, lam, ball).2)
        .sum::<f64>()
        / n
}

fn mean_inner(
    intercept: f64,
    f: &[Vec<f64>],
    bins: &[Vec<usize>],
    y: &[f64],
    u: f64,
    kappa: &[f64],
    lam: f64,
    ball: Ball,
) -> f64 {
    let n = y.len() as f64;
    bins.iter()
        .zip(y)
        .map(|(b, &yi)| inner_choice(intercept, f, b, yi, u, kappa, lam, ball).0)
        .sum::<f64>()
        / n
}

/// Returns `(loss − λ d, chosen bins, ground cost, active check weight s)`.
/// The pinball derivative in the quantile is −s.
fn inner_choice(
    intercept: f64,
    f: &[Vec<f64>],
    b: &[usize],
    y: f64,
    u: f64,
    kappa: &[f64],
    lam: f64,
    ball: Ball,
) -> (f64, Vec<usize>, f64, f64) {
    let p = f.len();
    if ball == Ball::W1 && lam.is_finite() {
        let mut best = f64::NEG_INFINITY;
        let mut best_c = b.to_vec();
        let mut best_d = 0.0;
        let mut best_s = u;
        for &s in &[u, u - 1.0] {
            let mut val = s * y;
            let mut dist = 0.0;
            let mut chosen = vec![0usize; p];
            for j in 0..p {
                let mut local = f64::NEG_INFINITY;
                let mut cj = b[j];
                let mut dj = 0.0;
                for c in 0..f[j].len() {
                    let step = (c as f64 - b[j] as f64).abs();
                    let score = -s * f[j][c] - lam * kappa[j] * step;
                    if score > local {
                        local = score;
                        cj = c;
                        dj = kappa[j] * step;
                    }
                }
                val += local;
                dist += dj;
                chosen[j] = cj;
            }
            // s * (y - intercept - sum f) = s*y - s*intercept + sum (-s f)
            val -= s * intercept;
            if val > best {
                best = val;
                best_c = chosen;
                best_d = dist;
                best_s = s;
            }
        }
        return (best, best_c, best_d, best_s);
    }
    // Enumeration, including W2 where the cost is the square of the additive distance.
    let sizes: Vec<usize> = f.iter().map(|col| col.len()).collect();
    let mut idx = vec![0usize; p];
    let mut best = f64::NEG_INFINITY;
    let mut best_c = b.to_vec();
    let mut best_d = 0.0;
    let mut best_s = u;
    loop {
        let q = predict_bins(intercept, f, &idx);
        let d_w1: f64 = idx
            .iter()
            .zip(b)
            .enumerate()
            .map(|(j, (&c, &bi))| kappa[j] * (c as f64 - bi as f64).abs())
            .sum();
        let d = match ball {
            Ball::W2 => d_w1 * d_w1,
            _ => d_w1,
        };
        let loss = pinball(y - q, u);
        let slope = if y >= q { u } else { u - 1.0 };
        let val = if lam.is_finite() {
            loss - lam * d
        } else if d == 0.0 {
            loss
        } else {
            f64::NEG_INFINITY
        };
        if val > best {
            best = val;
            best_c = idx.clone();
            best_d = d;
            best_s = slope;
        }
        if !bump(&mut idx, &sizes) {
            break;
        }
    }
    (best, best_c, best_d, best_s)
}

fn bump(idx: &mut [usize], sizes: &[usize]) -> bool {
    for j in 0..idx.len() {
        idx[j] += 1;
        if idx[j] < sizes[j] {
            return true;
        }
        idx[j] = 0;
    }
    false
}

fn predict_at(intercept: f64, f: &[Vec<f64>], b: &[usize]) -> f64 {
    predict_bins(intercept, f, b)
}

fn predict_bins(intercept: f64, f: &[Vec<f64>], b: &[usize]) -> f64 {
    let mut q = intercept;
    for j in 0..f.len() {
        q += f[j][b[j]];
    }
    q
}

pub fn pinball(r: f64, u: f64) -> f64 {
    if r >= 0.0 {
        u * r
    } else {
        (u - 1.0) * r
    }
}

/// Exact dual value of one quantile level. `f[j][b]` excludes the intercept,
/// which is passed separately. `bins[i][j]` are the occupied codes.
pub fn robust_risk_level(
    intercept: f64,
    f: &[Vec<f64>],
    bins: &[Vec<usize>],
    y: &[f64],
    u: f64,
    kappa: &[f64],
    eps: f64,
    ball: Ball,
    kappa_y: Option<f64>,
) -> Result<f64> {
    Ok(best_lambda(&intercept, f, bins, y, u, kappa, eps, ball, kappa_y)?.1)
}

/// Brute-force dual: evaluate the piecewise-linear dual at every breakpoint
/// generated by pairs of configurations. Feasible when `Π B_j` is a few hundred.
pub fn robust_risk_breakpoints(
    intercept: f64,
    f: &[Vec<f64>],
    bins: &[Vec<usize>],
    y: &[f64],
    u: f64,
    kappa: &[f64],
    eps: f64,
    ball: Ball,
    kappa_y: Option<f64>,
) -> Result<f64> {
    let rbar = u.max(1.0 - u);
    let lam_min = match kappa_y {
        Some(ky) if ky > 0.0 => rbar / ky,
        _ => 0.0,
    };
    let n = y.len();
    let mut cand = vec![lam_min];
    let mut loss = Vec::with_capacity(n);
    let mut dist = Vec::with_capacity(n);
    for i in 0..n {
        let (ls, ds) = config_table(intercept, f, &bins[i], y[i], u, kappa, ball);
        for a in 0..ls.len() {
            for b in 0..ls.len() {
                let dd = ds[a] - ds[b];
                if dd > 1e-15 {
                    let slope = (ls[a] - ls[b]) / dd;
                    if slope >= lam_min && slope.is_finite() {
                        cand.push(slope);
                    }
                }
            }
        }
        loss.push(ls);
        dist.push(ds);
    }
    cand.sort_by(f64::total_cmp);
    cand.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    let mut best = f64::INFINITY;
    for &lam in &cand {
        let mut mean = 0.0;
        for i in 0..n {
            let mut mx = f64::NEG_INFINITY;
            for c in 0..loss[i].len() {
                mx = mx.max(loss[i][c] - lam * dist[i][c]);
            }
            mean += mx;
        }
        mean /= n as f64;
        best = best.min(lam * eps + mean);
    }
    Ok(best)
}

fn config_table(
    intercept: f64,
    f: &[Vec<f64>],
    b: &[usize],
    y: f64,
    u: f64,
    kappa: &[f64],
    ball: Ball,
) -> (Vec<f64>, Vec<f64>) {
    let p = f.len();
    let sizes: Vec<usize> = f.iter().map(|col| col.len()).collect();
    let mut idx = vec![0usize; p];
    let mut loss = Vec::new();
    let mut dist = Vec::new();
    loop {
        let q = predict_bins(intercept, f, &idx);
        let d_w1: f64 = (0..p)
            .map(|j| kappa[j] * (idx[j] as f64 - b[j] as f64).abs())
            .sum();
        let d = match ball {
            Ball::W2 => d_w1 * d_w1,
            _ => d_w1,
        };
        loss.push(pinball(y - q, u));
        dist.push(d);
        if !bump(&mut idx, &sizes) {
            break;
        }
    }
    (loss, dist)
}

/// Upper bound `emp + ε ρ̄ max_j G_j/κ_j`. Not the robust objective.
pub fn max_jump_bound(f: &[Vec<f64>], kappa: &[f64], emp: f64, eps: f64, u: f64) -> f64 {
    let rbar = u.max(1.0 - u);
    let mut lip = 0.0_f64;
    for (j, col) in f.iter().enumerate() {
        let g = col
            .windows(2)
            .map(|w| (w[0] - w[1]).abs())
            .fold(0.0, f64::max);
        let kj = kappa[j].max(1e-15);
        lip = lip.max(g / kj);
    }
    emp + eps * rbar * lip
}

/// Outcome-only W1 gap. It does not depend on the fitted curve.
pub fn outcome_only_w1_gap(eps: f64, u: f64, kappa_y: f64) -> f64 {
    eps * u.max(1.0 - u) / kappa_y
}

/// Exact W∞ worst case and the window total-variation upper bound, averaged over rows.
pub fn winf_risk(
    intercept: f64,
    f: &[Vec<f64>],
    bins: &[Vec<usize>],
    y: &[f64],
    u: f64,
    radius: usize,
    delta: f64,
) -> (f64, f64) {
    let n = y.len() as f64;
    let rbar = u.max(1.0 - u);
    let mut exact = 0.0;
    let mut bound = 0.0;
    for (i, b) in bins.iter().enumerate() {
        let mut qmin = intercept;
        let mut qmax = intercept;
        let mut tv = 0.0;
        for j in 0..f.len() {
            let lo = b[j].saturating_sub(radius);
            let hi = (b[j] + radius + 1).min(f[j].len());
            let window = &f[j][lo..hi];
            qmin += window.iter().cloned().fold(f64::INFINITY, f64::min);
            qmax += window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            tv += window.windows(2).map(|w| (w[0] - w[1]).abs()).sum::<f64>();
        }
        let q = predict_at(intercept, f, b);
        exact += pinball(y[i] + delta - qmin, u).max(pinball(y[i] - delta - qmax, u));
        bound += pinball(y[i] - q, u) + rbar * (delta + tv);
    }
    (exact / n, bound / n)
}

pub fn predict_robust(model: &RobustModel, x: &[f64], n: usize) -> Result<Vec<f64>> {
    let p = model.edges.len();
    if x.len() != n * p {
        return Err(TraceError::Shape("X has the wrong shape".into()));
    }
    let codes = bin_with_edges(x, n, &model.edges)?;
    let n_u = model.levels.len();
    let mut out = vec![0.0; n * n_u];
    for (t, _u) in model.levels.iter().enumerate() {
        for i in 0..n {
            let mut q = model.intercept[t];
            for j in 0..p {
                q += model.effects[t][j][codes[j][i] as usize];
            }
            out[i * n_u + t] = q;
        }
    }
    Ok(out)
}

pub fn integrated_pinball_on(model: &RobustModel, x: &[f64], y: &[f64]) -> Result<f64> {
    let n = y.len();
    let pred = predict_robust(model, x, n)?;
    let nu = model.levels.len();
    let mut loss = 0.0;
    for i in 0..n {
        for t in 0..nu {
            loss += model.weights[t] * pinball(y[i] - pred[i * nu + t], model.levels[t]);
        }
    }
    Ok(loss / n as f64)
}

fn integrated_pinball(
    intercept: &[f64],
    effects: &[Vec<Vec<f64>>],
    codes: &[Vec<u16>],
    y: &[f64],
    levels: &[f64],
    weights: &[f64],
) -> f64 {
    let n = y.len();
    let p = codes.len();
    let mut loss = 0.0;
    for (t, &u) in levels.iter().enumerate() {
        for i in 0..n {
            let mut q = intercept[t];
            for j in 0..p {
                q += effects[t][j][codes[j][i] as usize];
            }
            loss += weights[t] * pinball(y[i] - q, u);
        }
    }
    loss / n as f64
}

fn integrated_robust(
    intercept: &[f64],
    effects: &[Vec<Vec<f64>>],
    codes: &[Vec<u16>],
    y: &[f64],
    levels: &[f64],
    weights: &[f64],
    kappa: &[f64],
    eps: f64,
    ball: Ball,
    kappa_y: Option<f64>,
) -> Result<f64> {
    let n = y.len();
    let p = codes.len();
    let bins: Vec<Vec<usize>> = (0..n)
        .map(|i| (0..p).map(|j| codes[j][i] as usize).collect())
        .collect();
    let mut total = 0.0;
    for (t, &u) in levels.iter().enumerate() {
        total += weights[t]
            * robust_risk_level(
                intercept[t],
                &effects[t],
                &bins,
                y,
                u,
                kappa,
                eps,
                ball,
                kappa_y,
            )?;
    }
    Ok(total)
}

/// Matched-row coupling cost: an upper bound on joint W1 when rows are paired.
pub fn coupling_w1(bins_a: &[Vec<usize>], bins_b: &[Vec<usize>], kappa: &[f64]) -> f64 {
    let n = bins_a.len() as f64;
    bins_a
        .iter()
        .zip(bins_b)
        .map(|(a, b)| {
            a.iter()
                .zip(b)
                .enumerate()
                .map(|(j, (&ba, &bb))| kappa[j] * (ba as f64 - bb as f64).abs())
                .sum::<f64>()
        })
        .sum::<f64>()
        / n
}

fn default_kappa(edges: &[Vec<f64>]) -> Vec<f64> {
    edges
        .iter()
        .map(|e| {
            if e.len() >= 2 {
                (e[e.len() - 1] - e[0]) / e.len() as f64
            } else {
                1.0
            }
        })
        .collect()
}

fn empirical_quantile(y: &[f64], u: f64) -> f64 {
    let mut z = y.to_vec();
    z.sort_by(f64::total_cmp);
    let n = z.len();
    let idx = ((u * n as f64).ceil() as isize - 1).clamp(0, n as isize - 1) as usize;
    z[idx]
}

fn bin_matrix(
    x: &[f64],
    n: usize,
    p: usize,
    max_bins: usize,
) -> Result<(Vec<Vec<u16>>, Vec<Vec<f64>>, Vec<usize>)> {
    bin_with_edges_new(x, n, p, max_bins)
}

fn bin_with_edges_new(
    x: &[f64],
    n: usize,
    p: usize,
    max_bins: usize,
) -> Result<(Vec<Vec<u16>>, Vec<Vec<f64>>, Vec<usize>)> {
    let mut codes = Vec::with_capacity(p);
    let mut edges = Vec::with_capacity(p);
    let mut n_bins = Vec::with_capacity(p);
    let mut col = vec![0.0; n];
    for j in 0..p {
        for i in 0..n {
            col[i] = x[i * p + j];
        }
        let mut finite: Vec<f64> = col.iter().copied().filter(|v| v.is_finite()).collect();
        finite.sort_by(f64::total_cmp);
        let e = edges_from_sorted(&finite, max_bins)?;
        let mut c = vec![0u16; n];
        bin_column_into(&col, &e, &mut c)?;
        n_bins.push(e.len() + 2);
        edges.push(e);
        codes.push(c);
    }
    Ok((codes, edges, n_bins))
}

fn bin_with_edges(x: &[f64], n: usize, edges: &[Vec<f64>]) -> Result<Vec<Vec<u16>>> {
    let p = edges.len();
    let mut codes = Vec::with_capacity(p);
    let mut col = vec![0.0; n];
    for j in 0..p {
        for i in 0..n {
            col[i] = x[i * p + j];
        }
        let mut c = vec![0u16; n];
        bin_column_into(&col, &edges[j], &mut c)?;
        codes.push(c);
    }
    Ok(codes)
}

pub fn codes_for_robust(model: &RobustModel, x: &[f64], n: usize) -> Result<Vec<Vec<u16>>> {
    bin_with_edges(x, n, &model.edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dual_matches_breakpoints_and_bound() {
        let f = vec![vec![0.0, 0.4, 0.1, 0.9], vec![0.2, -0.3, 0.5, 0.0]];
        let bins = vec![vec![0, 1], vec![2, 0], vec![1, 3], vec![3, 2], vec![0, 0]];
        let y = vec![0.2, -0.4, 1.1, 0.0, 0.5];
        let kappa = vec![0.25, 0.25];
        let u = 0.8;
        for eps in [0.01, 0.1, 0.5] {
            let a = robust_risk_level(0.0, &f, &bins, &y, u, &kappa, eps, Ball::W1, None).unwrap();
            let b = robust_risk_breakpoints(0.0, &f, &bins, &y, u, &kappa, eps, Ball::W1, None)
                .unwrap();
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
            let emp = y
                .iter()
                .enumerate()
                .map(|(i, &yi)| pinball(yi - predict_at(0.0, &f, &bins[i]), u))
                .sum::<f64>()
                / y.len() as f64;
            let ub = max_jump_bound(&f, &kappa, emp, eps, u);
            assert!(a <= ub + 1e-8);
        }
    }

    #[test]
    fn outcome_gap_is_constant() {
        let g = outcome_only_w1_gap(0.2, 0.7, 0.5);
        assert!((g - 0.2 * 0.7 / 0.5).abs() < 1e-15);
    }
}
