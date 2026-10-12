//! Fréchet regression for distribution-valued rows.
//!
//! Targets are Hosking's unbiased sample L-moments, converted to orthonormal
//! `a` units `t_k = √(2k+1) ℓ_{k+1}`. Without cross-channel constraints each
//! channel is a separate weighted least-squares fit. Plug-in quantile targets
//! are not used.

#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use super::legendre::{hosking_lmoments, lambda_to_a};
use crate::error::{Result, TraceError};
use crate::scalable::binning::{bin_column_into, edges_from_sorted};

#[derive(Clone, Debug)]
pub struct FrechetConfig {
    pub k_fit: usize,
    pub max_bins: usize,
    pub max_rounds: usize,
    pub tol: f64,
    /// When true, a first equal-weight pass estimates `τ² + v/m` and a second
    /// pass refits. When false, `weights` (or equal weights) are used as given.
    pub estimate_weights: bool,
}

impl Default for FrechetConfig {
    fn default() -> Self {
        Self {
            k_fit: 4,
            max_bins: 16,
            max_rounds: 40,
            tol: 1e-12,
            estimate_weights: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FrechetModel {
    pub k_fit: usize,
    pub edges: Vec<Vec<f64>>,
    pub n_bins: Vec<usize>,
    pub intercept: Vec<f64>,
    pub effects: Vec<Vec<f64>>,
    pub train_counts: Vec<Vec<u64>>,
    /// Per channel: `(τ², v)` in the `a`-unit target.
    pub variance: Vec<(f64, f64)>,
    /// Row weights used in the final pass, length `n`. Shared across channels
    /// only when `estimate_weights` is false; otherwise this is channel 0's weights
    /// and `channel_weights` holds the rest.
    pub weights: Vec<f64>,
    pub channel_weights: Vec<Vec<f64>>,
}

/// `samples` is row-major `n × m_max`. `m[i]` is the number of leading finite
/// entries used for row `i`.
pub fn fit_frechet(
    x: &[f64],
    n: usize,
    p: usize,
    samples: &[f64],
    m_max: usize,
    m: &[usize],
    cfg: &FrechetConfig,
    weights: Option<&[f64]>,
) -> Result<FrechetModel> {
    if n < 2 || p < 1 || cfg.k_fit < 1 {
        return Err(TraceError::InvalidArgument(
            "Fréchet fit needs at least 2 rows, 1 feature and one channel".into(),
        ));
    }
    if x.len() != n * p || samples.len() != n * m_max || m.len() != n {
        return Err(TraceError::Shape("X, samples and lengths disagree".into()));
    }
    let mut targets = vec![0.0; n * cfg.k_fit];
    for i in 0..n {
        if m[i] < cfg.k_fit || m[i] > m_max {
            return Err(TraceError::InvalidArgument(
                "each row needs at least K_fit samples".into(),
            ));
        }
        let row = &samples[i * m_max..i * m_max + m[i]];
        let lam = hosking_lmoments(row, cfg.k_fit)?;
        let a = lambda_to_a(&lam);
        targets[i * cfg.k_fit..(i + 1) * cfg.k_fit].copy_from_slice(&a);
    }
    let (codes, edges, n_bins) = bin_matrix(x, n, p, cfg.max_bins)?;
    let base_w = match weights {
        Some(w) if w.len() == n => w.to_vec(),
        Some(_) => {
            return Err(TraceError::Shape(
                "weights must have one entry per row".into(),
            ))
        }
        None => vec![1.0; n],
    };
    let mut variance = vec![(0.0, 0.0); cfg.k_fit];
    let mut channel_weights = vec![base_w.clone(); cfg.k_fit];
    let mut intercept = vec![0.0; cfg.k_fit];
    let mut effects = Vec::new();
    let mut train_counts = Vec::new();
    for j in 0..p {
        effects.push(vec![0.0; n_bins[j] * cfg.k_fit]);
        let mut cnt = vec![0u64; n_bins[j]];
        for i in 0..n {
            cnt[codes[j][i] as usize] += 1;
        }
        train_counts.push(cnt);
    }
    for c in 0..cfg.k_fit {
        let col: Vec<f64> = (0..n).map(|i| targets[i * cfg.k_fit + c]).collect();
        let (icept, eff) = wls_channel(&col, &codes, &base_w, cfg)?;
        intercept[c] = icept;
        for j in 0..p {
            for b in 0..n_bins[j] {
                effects[j][b * cfg.k_fit + c] = eff[j][b];
            }
        }
        if cfg.estimate_weights {
            let pred = predict_channel(&intercept, &effects, &codes, c, cfg.k_fit);
            let resid: Vec<f64> = pred.iter().zip(&col).map(|(p, t)| t - p).collect();
            let (tau2, v) = moment_variance(&resid, m);
            variance[c] = (tau2, v);
            let w: Vec<f64> = (0..n)
                .map(|i| 1.0 / (tau2 + v / m[i] as f64).max(1e-12))
                .collect();
            let (icept2, eff2) = wls_channel(&col, &codes, &w, cfg)?;
            intercept[c] = icept2;
            for j in 0..p {
                for b in 0..n_bins[j] {
                    effects[j][b * cfg.k_fit + c] = eff2[j][b];
                }
            }
            channel_weights[c] = w;
        }
    }
    Ok(FrechetModel {
        k_fit: cfg.k_fit,
        edges,
        n_bins,
        intercept,
        effects,
        train_counts,
        variance,
        weights: channel_weights[0].clone(),
        channel_weights,
    })
}

fn moment_variance(resid: &[f64], m: &[usize]) -> (f64, f64) {
    let n = resid.len() as f64;
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for (r, &mi) in resid.iter().zip(m) {
        let x = 1.0 / mi as f64;
        let y = r * r;
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    let den = n * sxx - sx * sx;
    if den.abs() < 1e-18 {
        let mean = sy / n;
        return (mean.max(0.0), 0.0);
    }
    let v = (n * sxy - sx * sy) / den;
    let tau2 = (sy - v * sx) / n;
    (tau2.max(0.0), v.max(0.0))
}

fn wls_channel(
    target: &[f64],
    codes: &[Vec<u16>],
    w: &[f64],
    cfg: &FrechetConfig,
) -> Result<(f64, Vec<Vec<f64>>)> {
    let n = target.len();
    let p = codes.len();
    let mut intercept = weighted_mean(target, w);
    let mut effects: Vec<Vec<f64>> = codes
        .iter()
        .map(|c| vec![0.0; c.iter().copied().max().unwrap_or(0) as usize + 1])
        .collect();
    // Size effects from the maximum code actually, using a first scan.
    for j in 0..p {
        let nb = codes[j].iter().copied().max().unwrap_or(0) as usize + 1;
        effects[j] = vec![0.0; nb];
    }
    let mut prev = f64::INFINITY;
    for _ in 0..cfg.max_rounds {
        for j in 0..p {
            let nb = effects[j].len();
            let mut sw = vec![0.0; nb];
            let mut sy = vec![0.0; nb];
            for i in 0..n {
                let b = codes[j][i] as usize;
                let mut resid = target[i] - intercept;
                for t in 0..p {
                    if t != j {
                        resid -= effects[t][codes[t][i] as usize];
                    }
                }
                sw[b] += w[i];
                sy[b] += w[i] * resid;
            }
            for b in 0..nb {
                effects[j][b] = if sw[b] > 0.0 { sy[b] / sw[b] } else { 0.0 };
            }
            let mut mean = 0.0;
            let mut mass = 0.0;
            for b in 0..nb {
                mean += sw[b] * effects[j][b];
                mass += sw[b];
            }
            if mass > 0.0 {
                mean /= mass;
            }
            for b in 0..nb {
                effects[j][b] -= mean;
            }
            intercept += mean;
        }
        let pred = predict_from(intercept, &effects, codes);
        let loss = target
            .iter()
            .zip(&pred)
            .zip(w)
            .map(|((t, p), wi)| wi * (t - p) * (t - p))
            .sum::<f64>();
        if (prev - loss).abs() <= cfg.tol * (prev.abs() + 1.0) {
            break;
        }
        prev = loss;
    }
    Ok((intercept, effects))
}

fn predict_from(intercept: f64, effects: &[Vec<f64>], codes: &[Vec<u16>]) -> Vec<f64> {
    let n = codes[0].len();
    let mut out = vec![intercept; n];
    for (j, col) in codes.iter().enumerate() {
        for i in 0..n {
            out[i] += effects[j][col[i] as usize];
        }
    }
    out
}

fn predict_channel(
    intercept: &[f64],
    effects: &[Vec<f64>],
    codes: &[Vec<u16>],
    channel: usize,
    k: usize,
) -> Vec<f64> {
    let n = codes[0].len();
    let mut out = vec![intercept[channel]; n];
    for (j, col) in codes.iter().enumerate() {
        for i in 0..n {
            out[i] += effects[j][col[i] as usize * k + channel];
        }
    }
    out
}

fn weighted_mean(y: &[f64], w: &[f64]) -> f64 {
    let sw = w.iter().sum::<f64>();
    if sw == 0.0 {
        return 0.0;
    }
    y.iter().zip(w).map(|(yi, wi)| wi * yi).sum::<f64>() / sw
}

fn bin_matrix(
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

/// Predicted orthonormal coefficients, `n × k_fit`.
pub fn predict_frechet(model: &FrechetModel, x: &[f64], n: usize) -> Result<Vec<f64>> {
    let p = model.edges.len();
    if x.len() != n * p {
        return Err(TraceError::Shape("X has the wrong shape".into()));
    }
    let mut codes = Vec::with_capacity(p);
    let mut col = vec![0.0; n];
    for j in 0..p {
        for i in 0..n {
            col[i] = x[i * p + j];
        }
        let mut c = vec![0u16; n];
        bin_column_into(&col, &model.edges[j], &mut c)?;
        codes.push(c);
    }
    let k = model.k_fit;
    let mut out = vec![0.0; n * k];
    for i in 0..n {
        for c in 0..k {
            out[i * k + c] = model.intercept[c];
        }
        for j in 0..p {
            let b = codes[j][i] as usize;
            for c in 0..k {
                out[i * k + c] += model.effects[j][b * k + c];
            }
        }
    }
    Ok(out)
}

/// Ordinary least-squares slope of `y` on `x` with weights `w`.
pub fn weighted_slope(x: &[f64], y: &[f64], w: &[f64]) -> Result<f64> {
    if x.len() != y.len() || x.len() != w.len() || x.len() < 2 {
        return Err(TraceError::Shape(
            "weighted slope needs matching columns of length at least 2".into(),
        ));
    }
    let sw = w.iter().sum::<f64>();
    let mx = x.iter().zip(w).map(|(v, wi)| wi * v).sum::<f64>() / sw;
    let my = y.iter().zip(w).map(|(v, wi)| wi * v).sum::<f64>() / sw;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..x.len() {
        num += w[i] * (x[i] - mx) * (y[i] - my);
        den += w[i] * (x[i] - mx) * (x[i] - mx);
    }
    if den == 0.0 {
        return Err(TraceError::Numerical(
            "covariate has no weighted variation".into(),
        ));
    }
    Ok(num / den)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_a_discrete_location() {
        let n = 60;
        let mut x = Vec::new();
        let mut samples = Vec::new();
        let mut m = Vec::new();
        for i in 0..n {
            let b = (i % 3) as f64;
            x.push(b);
            let shift = 1.0 + 0.5 * b;
            for s in 0..8 {
                samples.push(shift + (s as f64 - 3.5) * 0.01);
            }
            m.push(8);
        }
        let cfg = FrechetConfig {
            k_fit: 2,
            max_bins: 3,
            estimate_weights: false,
            ..FrechetConfig::default()
        };
        let model = fit_frechet(&x, n, 1, &samples, 8, &m, &cfg, None).unwrap();
        let pred = predict_frechet(&model, &x, n).unwrap();
        let mut err = 0.0;
        for i in 0..n {
            let truth = 1.0 + 0.5 * x[i];
            err += (pred[i * 2] - truth).abs();
        }
        assert!(err / (n as f64) < 0.05);
    }
}
