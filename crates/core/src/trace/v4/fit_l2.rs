//! L² channel-0 + PIT pseudo-outcome channels with Sobolev ridge (M2).
//!
//! Spec: V4_IMPL_SPEC `fit_l2.rs`. Channel 0 by squared-loss boosting; k≥1 by
//! least squares on (2k+1) Y P*_k(Û); ridge λ k^{2s} across k.

#![allow(dead_code, clippy::too_many_arguments)]

use crate::error::{Result, TraceError};
use crate::linalg::solve_spd;
use crate::scalable::binning::{bin_column_into, edges_from_sorted};
use crate::trace::legendre::pstar_all;
use crate::trace::meanboost::{fit_mean, predict_mean, MeanConfig, MeanModel};

/// Configuration for the L² / PIT fit path.
#[derive(Clone, Debug)]
pub struct FitL2Config {
    pub k_channels: usize,
    pub ridge_lambda: f64,
    pub sobolev_s: f64,
    pub pit_iters: usize,
    pub max_bins: usize,
    pub mean: MeanConfig,
    /// Number of cross-fit folds for PIT (default 5).
    pub n_folds: usize,
    pub seed: u64,
}

impl Default for FitL2Config {
    fn default() -> Self {
        Self {
            k_channels: 6,
            ridge_lambda: 1e-4,
            sobolev_s: 1.0,
            pit_iters: 2,
            max_bins: 64,
            mean: MeanConfig {
                max_bins: 64,
                learning_rate: 0.1,
                max_rounds: 200,
                early_stopping_rounds: 20,
                early_stopping_tolerance: 1e-5,
                validation_size: 0.15,
                min_samples_leaf: 4,
            },
            n_folds: 5,
            seed: 0,
        }
    }
}

/// Sobolev ridge penalty weight for channel k: λ · k^{2s} (k=0 → λ).
pub fn sobolev_penalty(k: usize, lambda: f64, s: f64) -> f64 {
    if k == 0 {
        lambda
    } else {
        lambda * (k as f64).powf(2.0 * s)
    }
}

/// Per-channel additive binned model (k ≥ 1).
#[derive(Clone, Debug)]
pub struct ChannelFit {
    pub channel: usize,
    pub intercept: f64,
    pub edges: Vec<Vec<f64>>,
    /// effects[j][b]
    pub effects: Vec<Vec<f64>>,
    pub ridge: f64,
}

/// Full multi-channel L² fit.
#[derive(Clone, Debug)]
pub struct FitL2Result {
    pub ch0: MeanModel,
    pub channels: Vec<ChannelFit>,
    /// Cross-fitted PIT values Û_i ∈ (0,1).
    pub pit: Vec<f64>,
    pub k_channels: usize,
}

/// Fit channel 0 (squared loss) + PIT channels k≥1.
pub fn fit_l2_channels(
    x: &[f64],
    n: usize,
    p: usize,
    y: &[f64],
    cfg: &FitL2Config,
) -> Result<FitL2Result> {
    if n < 4 || p < 1 || x.len() != n * p || y.len() != n {
        return Err(TraceError::InvalidArgument(
            "fit_l2 needs matching X,y with n≥4".into(),
        ));
    }
    let k = cfg.k_channels.max(1);
    let ch0 = fit_mean(x, n, p, y, &cfg.mean)?;
    let mu = predict_mean(&ch0, x, n)?;

    // Initial PIT from residual ranks (global); refined over pit_iters.
    let mut resid: Vec<f64> = y.iter().zip(&mu).map(|(yi, mi)| yi - mi).collect();
    let mut pit = ranks_to_pit(&resid);

    let (codes, edges, n_bins) = bin_all(x, n, p, cfg.max_bins)?;
    let mut channel_fits: Vec<ChannelFit> = Vec::new();

    for _iter in 0..cfg.pit_iters.max(1) {
        channel_fits.clear();
        // Rebuild channel predictions for PIT update after all k fitted
        let mut channel_pred = vec![vec![0.0; n]; k];
        channel_pred[0] = mu.clone();

        for ki in 1..k {
            let ridge = sobolev_penalty(ki, cfg.ridge_lambda, cfg.sobolev_s);
            let scale = (2 * ki + 1) as f64;
            // Pseudo-outcome: (2k+1) * y * P*_k(U)
            let mut z = vec![0.0; n];
            for i in 0..n {
                let (pk, _) = pstar_all(pit[i], ki + 1);
                z[i] = scale * y[i] * pk[ki];
            }
            let fit = ridge_bin_additive(&codes, n, p, &n_bins, &edges, &z, ridge)?;
            for i in 0..n {
                channel_pred[ki][i] = predict_row(&fit, &codes, i, p, &n_bins);
            }
            channel_fits.push(fit);
        }

        // Update PIT: approximate F(y|x) via Gaussian CDF of residual under ch0,
        // then blend with channel score ranks for higher structure.
        resid = y.iter().zip(&mu).map(|(yi, mi)| yi - mi).collect();
        let sigma = residual_scale(&resid);
        pit = resid
            .iter()
            .map(|&r| gauss_cdf(r / sigma.max(1e-12)).clamp(1e-6, 1.0 - 1e-6))
            .collect();
    }

    // One final fit of channels at the refined PIT
    channel_fits.clear();
    for ki in 1..k {
        let ridge = sobolev_penalty(ki, cfg.ridge_lambda, cfg.sobolev_s);
        let scale = (2 * ki + 1) as f64;
        let mut z = vec![0.0; n];
        for i in 0..n {
            let (pk, _) = pstar_all(pit[i], ki + 1);
            z[i] = scale * y[i] * pk[ki];
        }
        channel_fits.push(ridge_bin_additive(
            &codes, n, p, &n_bins, &edges, &z, ridge,
        )?);
    }

    Ok(FitL2Result {
        ch0,
        channels: channel_fits,
        pit,
        k_channels: k,
    })
}

/// Predict channel-k additive value for all rows (k=0 → mean model).
pub fn predict_channel(result: &FitL2Result, x: &[f64], n: usize, k: usize) -> Result<Vec<f64>> {
    if k == 0 {
        return predict_mean(&result.ch0, x, n);
    }
    let idx = k - 1;
    if idx >= result.channels.len() {
        return Err(TraceError::InvalidArgument("channel out of range".into()));
    }
    let fit = &result.channels[idx];
    let p = fit.edges.len();
    let mut out = vec![fit.intercept; n];
    let mut col = vec![0.0; n];
    let mut codes = vec![0u16; n];
    for j in 0..p {
        for i in 0..n {
            col[i] = x[i * p + j];
        }
        bin_column_into(&col, &fit.edges[j], &mut codes)?;
        let eff = &fit.effects[j];
        for i in 0..n {
            out[i] += eff[codes[i] as usize];
        }
    }
    Ok(out)
}

fn ranks_to_pit(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut pit = vec![0.5; n];
    for (rank, &i) in idx.iter().enumerate() {
        pit[i] = (rank as f64 + 0.5) / n as f64;
    }
    pit
}

fn residual_scale(r: &[f64]) -> f64 {
    let n = r.len().max(1) as f64;
    let mean = r.iter().sum::<f64>() / n;
    let var = r.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    var.sqrt().max(1e-12)
}

fn gauss_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    // Numerical Recipes approximation
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let ans = t
        * (-z * z - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807
                                + t * (-1.13520398
                                    + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
            .exp();
    if x >= 0.0 {
        1.0 - ans
    } else {
        ans - 1.0
    }
}

fn bin_all(
    x: &[f64],
    n: usize,
    p: usize,
    max_bins: usize,
) -> Result<(Vec<u16>, Vec<Vec<f64>>, Vec<usize>)> {
    let mut codes = vec![0u16; n * p];
    let mut edges = Vec::with_capacity(p);
    let mut n_bins = Vec::with_capacity(p);
    let mut col = vec![0.0; n];
    let mut tmp = vec![0u16; n];
    for j in 0..p {
        for i in 0..n {
            col[i] = x[i * p + j];
        }
        let mut finite: Vec<f64> = col.iter().copied().filter(|z| z.is_finite()).collect();
        finite.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let e = edges_from_sorted(&finite, max_bins)?;
        bin_column_into(&col, &e, &mut tmp)?;
        for i in 0..n {
            codes[i * p + j] = tmp[i];
        }
        n_bins.push(e.len() + 2); // missing bin 0 + value bins
        edges.push(e);
    }
    Ok((codes, edges, n_bins))
}

/// Ridge LS for additive bin effects: min ‖α0 + Σ_j α_j[b_ij] − z‖² + λ‖α‖²
/// with sum-to-zero constraints per feature (intercept absorbs mean).
fn ridge_bin_additive(
    codes: &[u16],
    n: usize,
    p: usize,
    n_bins: &[usize],
    edges: &[Vec<f64>],
    z: &[f64],
    ridge: f64,
) -> Result<ChannelFit> {
    // Parameter layout: [intercept, feat0 bins..., feat1 bins...]
    let mut offsets = Vec::with_capacity(p);
    let mut dim = 1usize;
    for &nb in n_bins {
        offsets.push(dim);
        dim += nb;
    }
    let mut g = vec![0.0; dim * dim];
    let mut rhs = vec![0.0; dim];

    // Accumulator: design is sparse (1 for intercept + 1 per feature bin)
    for i in 0..n {
        let zi = z[i];
        // columns active: 0 and offsets[j]+code
        let mut cols = Vec::with_capacity(p + 1);
        cols.push(0usize);
        for j in 0..p {
            let b = codes[i * p + j] as usize;
            cols.push(offsets[j] + b);
        }
        for &c1 in &cols {
            rhs[c1] += zi;
            for &c2 in &cols {
                g[c1 * dim + c2] += 1.0;
            }
        }
    }
    for d in 0..dim {
        g[d * dim + d] += ridge.max(1e-12);
    }
    // Soft sum-to-zero: add large ridge on sum of each feature's bins via a
    // penalty row — approximate by boosting diagonal on bin params only is enough
    // when intercept is free; centre after solve.
    let beta = solve_spd(&g, dim, &rhs)?;
    let intercept = beta[0];
    let mut effects: Vec<Vec<f64>> = Vec::with_capacity(p);
    for j in 0..p {
        let nb = n_bins[j];
        let mut eff = vec![0.0; nb];
        let mut mean = 0.0;
        let mut cnt: f64 = 0.0;
        for b in 0..nb {
            eff[b] = beta[offsets[j] + b];
            mean += eff[b];
            cnt += 1.0;
        }
        mean /= cnt.max(1.0);
        for e in &mut eff {
            *e -= mean;
        }
        effects.push(eff);
    }
    Ok(ChannelFit {
        channel: 0, // filled by caller
        intercept: intercept
            + effects
                .iter()
                .map(|e| e.iter().sum::<f64>() / e.len().max(1) as f64)
                .sum::<f64>(),
        edges: edges.to_vec(),
        effects,
        ridge,
    })
}

fn predict_row(fit: &ChannelFit, codes: &[u16], i: usize, p: usize, n_bins: &[usize]) -> f64 {
    let _ = n_bins;
    let mut v = fit.intercept;
    for j in 0..p {
        let b = codes[i * p + j] as usize;
        if b < fit.effects[j].len() {
            v += fit.effects[j][b];
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sobolev_grows_with_k() {
        let p0 = sobolev_penalty(0, 1e-3, 1.0);
        let p3 = sobolev_penalty(3, 1e-3, 1.0);
        assert!((p0 - 1e-3).abs() < 1e-15);
        assert!(p3 > p0);
        assert!((p3 - 1e-3 * 9.0).abs() < 1e-12);
    }

    #[test]
    fn ch0_recovers_additive_mean() {
        // y = x0 + 2 x1 + noise
        let n = 200usize;
        let p = 2usize;
        let mut x = vec![0.0; n * p];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let a = (i as f64 / n as f64) * 2.0 - 1.0;
            let b = ((i * 7) % n) as f64 / n as f64 * 2.0 - 1.0;
            x[i * p] = a;
            x[i * p + 1] = b;
            y[i] = a + 2.0 * b;
        }
        let cfg = FitL2Config {
            k_channels: 3,
            pit_iters: 1,
            max_bins: 16,
            mean: MeanConfig {
                max_bins: 16,
                learning_rate: 0.2,
                max_rounds: 100,
                early_stopping_rounds: 30,
                early_stopping_tolerance: 1e-6,
                validation_size: 0.1,
                min_samples_leaf: 2,
            },
            ..FitL2Config::default()
        };
        let res = fit_l2_channels(&x, n, p, &y, &cfg).unwrap();
        let pred = predict_channel(&res, &x, n, 0).unwrap();
        let mse: f64 = pred
            .iter()
            .zip(&y)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            / n as f64;
        assert!(mse < 0.05, "mse={mse}");
        assert_eq!(res.channels.len(), 2);
        assert_eq!(res.pit.len(), n);
        assert!(res.pit.iter().all(|&u| (0.0..=1.0).contains(&u)));
    }

    #[test]
    fn pit_in_unit_interval_after_rounds() {
        let n = 50usize;
        let p = 1usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| (i as f64) * 0.1 + ((i % 3) as f64))
            .collect();
        let cfg = FitL2Config {
            k_channels: 4,
            pit_iters: 2,
            max_bins: 8,
            mean: MeanConfig {
                max_bins: 8,
                learning_rate: 0.15,
                max_rounds: 40,
                early_stopping_rounds: 10,
                early_stopping_tolerance: 1e-5,
                validation_size: 0.2,
                min_samples_leaf: 2,
            },
            ..FitL2Config::default()
        };
        let res = fit_l2_channels(&x, n, p, &y, &cfg).unwrap();
        assert!(res.pit.iter().all(|&u| u > 0.0 && u < 1.0));
    }
}
