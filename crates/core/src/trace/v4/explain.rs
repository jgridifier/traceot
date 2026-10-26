//! End-to-end TRACE v4 explain path: L² fit → mixture/bary → game → bands.
//!
//! Spec: V4_IMPL_SPEC § Step 4 Python entry. Numerics stay here; Python only
//! forwards arrays.

#![allow(dead_code, clippy::too_many_arguments, clippy::type_complexity)]

use std::time::Instant;

use crate::error::{Result, TraceError};
use crate::trace::meanboost::predict_mean;

use super::fit_l2::{fit_l2_channels, predict_channel, FitL2Config, FitL2Result};
use super::game::{play_mixing_game, MixingGameConfig, MixingGameResult, PlayerShare};
use super::infer::{bootstrap_multiplier, InferConfig};
use super::mixture::{
    barycenter_from_conditionals, mixture_from_conditionals, pooled_y_range, MixtureConfig,
};

/// Knobs for `explain_v4`.
#[derive(Clone, Debug)]
pub struct ExplainV4Config {
    pub fit: FitL2Config,
    pub mixture: MixtureConfig,
    pub infer: InferConfig,
    /// Quantile grid size for building row-wise Q from channel-0 + residual scale.
    pub n_u: usize,
}

impl Default for ExplainV4Config {
    fn default() -> Self {
        Self {
            fit: FitL2Config::default(),
            mixture: MixtureConfig::default(),
            infer: InferConfig::default(),
            n_u: 64,
        }
    }
}

/// Per-role summary on the mean channel.
#[derive(Clone, Debug, Default)]
pub struct MeanSplit {
    pub comp: f64,
    pub structure: f64,
    pub mix_comp: f64,
    pub mix_structure: f64,
}

/// Full v4 explain report.
#[derive(Clone, Debug)]
pub struct ExplainV4Report {
    pub k_channels: usize,
    pub p: usize,
    pub mean: MeanSplit,
    pub feat_comp: Vec<f64>,
    pub feat_struct: Vec<f64>,
    pub feat_mix_comp: Vec<f64>,
    pub feat_mix_struct: Vec<f64>,
    pub players: Vec<PlayerShare>,
    pub budget: f64,
    pub interaction_mass: f64,
    pub mixing_residual: f64,
    /// Multiplier SE / CI for the total budget.
    pub budget_se: f64,
    pub budget_ci_lo: f64,
    pub budget_ci_hi: f64,
    pub seconds_fit: f64,
    pub seconds_game: f64,
    pub seconds_infer: f64,
    pub seconds_total: f64,
}

fn u_grid(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect()
}

/// Build a crude conditional Q row from mean prediction + residual scale:
/// Q(u|x) ≈ μ(x) + σ Φ⁻¹(u). Enough to drive mixture/bary helpers.
fn q_rows_location_scale(mu: &[f64], sigma: f64, u: &[f64]) -> Vec<f64> {
    let n = mu.len();
    let qn = u.len();
    let z: Vec<f64> = u
        .iter()
        .map(|&uu| norm_ppf(uu.clamp(1e-6, 1.0 - 1e-6)))
        .collect();
    let mut out = vec![0.0; n * qn];
    for i in 0..n {
        for (j, &zj) in z.iter().enumerate() {
            out[i * qn + j] = mu[i] + sigma * zj;
        }
    }
    out
}

fn residual_sigma(y: &[f64], mu: &[f64]) -> f64 {
    let n = y.len().max(1) as f64;
    let mut s = 0.0;
    for (yi, mi) in y.iter().zip(mu) {
        let r = yi - mi;
        s += r * r;
    }
    (s / n).sqrt().max(1e-8)
}

fn norm_ppf(u: f64) -> f64 {
    // Beasley–Springer–Moro approximation
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577459334128e+02,
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
    let phigh = 1.0 - plow;
    if u < plow {
        let q = (-2.0 * u.ln()).sqrt();
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if u > phigh {
        let q = (-2.0 * (1.0 - u).ln()).sqrt();
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    let q = u - 0.5;
    let r = q * q;
    (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
}

fn channel0_effects_cov(fit: &FitL2Result, x: &[f64], n: usize, p: usize) -> Result<Vec<f64>> {
    // Cov(f_j, μ) ≈ mean( effect_j[bin] * (μ − mean_μ) ) using ch0 effects
    let mu = predict_mean(&fit.ch0, x, n)?;
    let mean_mu = mu.iter().sum::<f64>() / n.max(1) as f64;
    let mut cov = vec![0.0; p];
    let mut col = vec![0.0; n];
    let mut codes = vec![0u16; n];
    use crate::scalable::binning::bin_column_into;
    for j in 0..p {
        for i in 0..n {
            col[i] = x[i * p + j];
        }
        bin_column_into(&col, &fit.ch0.edges[j], &mut codes)?;
        let eff = &fit.ch0.effects[j];
        let mut c = 0.0;
        for i in 0..n {
            let fj = eff[codes[i] as usize];
            c += fj * (mu[i] - mean_mu);
        }
        cov[j] = c / n as f64;
    }
    Ok(cov)
}

/// Explain a two-period shift under the v4 mixing law + game.
pub fn explain_v4(
    x_a: &[f64],
    y_a: &[f64],
    n_a: usize,
    x_b: &[f64],
    y_b: &[f64],
    n_b: usize,
    p: usize,
    cfg: &ExplainV4Config,
) -> Result<ExplainV4Report> {
    if x_a.len() != n_a * p || x_b.len() != n_b * p || y_a.len() != n_a || y_b.len() != n_b {
        return Err(TraceError::Shape("explain_v4 shape mismatch".into()));
    }
    let t0 = Instant::now();
    let mut fit_cfg = cfg.fit.clone();
    fit_cfg.seed = cfg.infer.seed;

    let fit_a = fit_l2_channels(x_a, n_a, p, y_a, &fit_cfg)?;
    let fit_b = fit_l2_channels(x_b, n_b, p, y_b, &fit_cfg)?;
    let seconds_fit = t0.elapsed().as_secs_f64();

    let mu_a = predict_mean(&fit_a.ch0, x_a, n_a)?;
    let mu_b = predict_mean(&fit_b.ch0, x_b, n_b)?;
    // Hybrids: structure B on covariates A, structure A on covariates B
    let mu_ab = predict_mean(&fit_b.ch0, x_a, n_a)?; // cov A, struct B
    let mu_ba = predict_mean(&fit_a.ch0, x_b, n_b)?; // cov B, struct A

    let k = fit_cfg.k_channels.max(1);
    let mean_a = mu_a.iter().sum::<f64>() / n_a as f64;
    let mean_b = mu_b.iter().sum::<f64>() / n_b as f64;
    let mean_ab = mu_ab.iter().sum::<f64>() / n_a as f64;
    let mean_ba = mu_ba.iter().sum::<f64>() / n_b as f64;

    // Channel-0 G-table on the mean functional
    let g0 = [mean_a, mean_ba, mean_ab, mean_b];
    // Higher channels: use mean of predict_channel if available, else 0
    let mut g_bary = vec![[0.0; 4]; k];
    g_bary[0] = g0;
    for ki in 1..k {
        let ca = predict_channel(&fit_a, x_a, n_a, ki).unwrap_or_else(|_| vec![0.0; n_a]);
        let cb = predict_channel(&fit_b, x_b, n_b, ki).unwrap_or_else(|_| vec![0.0; n_b]);
        let cab = predict_channel(&fit_b, x_a, n_a, ki).unwrap_or_else(|_| vec![0.0; n_a]);
        let cba = predict_channel(&fit_a, x_b, n_b, ki).unwrap_or_else(|_| vec![0.0; n_b]);
        g_bary[ki] = [
            ca.iter().sum::<f64>() / n_a as f64,
            cba.iter().sum::<f64>() / n_b as f64,
            cab.iter().sum::<f64>() / n_a as f64,
            cb.iter().sum::<f64>() / n_b as f64,
        ];
    }

    // Mixing G-table from mixture vs barycenter on period-B structure / A covariates
    let u = u_grid(cfg.n_u);
    let sig_b = residual_sigma(y_b, &mu_b);
    let q_on_a = q_rows_location_scale(&mu_ab, sig_b, &u);
    let (ylo, yhi) = pooled_y_range(y_a, y_b);
    let mut mix_cfg = cfg.mixture.clone();
    mix_cfg.k_channels = k;
    mix_cfg.n_quad = cfg.n_u;
    mix_cfg.subsample_seed = cfg.infer.seed;
    let mix = mixture_from_conditionals(&u, &q_on_a, n_a, ylo, yhi, &mix_cfg);
    let (qbar, cbar) = barycenter_from_conditionals(&u, &q_on_a, n_a, &mix_cfg);
    let _ = qbar;
    let mut g_mix = vec![[0.0; 4]; k];
    for ki in 0..k.min(mix.channels.len()).min(cbar.len()) {
        let m = mix.channels[ki] - cbar[ki];
        // Put mixing mass on the AB/BB corner difference (structure B side)
        g_mix[ki] = [0.0, 0.0, m, m];
    }

    let delta_c: Vec<f64> = (0..k).map(|ki| g_bary[ki][3] - g_bary[ki][0]).collect();
    // Equal feature shares of Δ within channel (refined by cov for mixing only)
    let mut a_comp = vec![0.0; k * p];
    let mut a_struct = vec![0.0; k * p];
    let cov = channel0_effects_cov(&fit_a, x_a, n_a, p)?;
    let cov_sum: f64 = cov.iter().map(|c| c.max(0.0)).sum::<f64>().max(1e-18);
    for ki in 0..k {
        for j in 0..p {
            let share = cov[j].max(0.0) / cov_sum;
            a_comp[ki * p + j] = 0.5 * share;
            a_struct[ki * p + j] = 0.5 * share;
        }
    }

    let t1 = Instant::now();
    let game_cfg = MixingGameConfig::from_k(k);
    let game: MixingGameResult = play_mixing_game(
        &g_bary, &g_mix, &delta_c, &a_comp, &a_struct, &cov, &game_cfg,
    );
    let seconds_game = t1.elapsed().as_secs_f64();

    let mut feat_comp = vec![0.0; p];
    let mut feat_struct = vec![0.0; p];
    let mut feat_mix_comp = vec![0.0; p];
    let mut feat_mix_struct = vec![0.0; p];
    let mut mean = MeanSplit::default();
    for pl in &game.players {
        if pl.channel != 0 {
            continue;
        }
        match pl.role {
            0 => {
                feat_comp[pl.feature] += pl.value;
                mean.comp += pl.value;
            }
            1 => {
                feat_struct[pl.feature] += pl.value;
                mean.structure += pl.value;
            }
            2 => {
                feat_mix_comp[pl.feature] += pl.value;
                mean.mix_comp += pl.value;
            }
            3 => {
                feat_mix_struct[pl.feature] += pl.value;
                mean.mix_structure += pl.value;
            }
            _ => {}
        }
    }

    let t2 = Instant::now();
    let contrib: Vec<f64> = game.players.iter().map(|p| p.value).collect();
    let draws = bootstrap_multiplier(&contrib, &cfg.infer);
    let se = {
        let n = draws.len().max(2);
        let m = draws.iter().sum::<f64>() / n as f64;
        let v = draws.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1) as f64;
        v.sqrt()
    };
    let mut sorted = draws;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let alpha = cfg.infer.alpha;
    let lo_i = ((alpha / 2.0) * (sorted.len() - 1) as f64).round() as usize;
    let hi_i = ((1.0 - alpha / 2.0) * (sorted.len() - 1) as f64).round() as usize;
    let ci_lo = game.budget + sorted[lo_i.min(sorted.len() - 1)];
    let ci_hi = game.budget + sorted[hi_i.min(sorted.len() - 1)];
    let seconds_infer = t2.elapsed().as_secs_f64();
    let seconds_total = t0.elapsed().as_secs_f64();

    Ok(ExplainV4Report {
        k_channels: k,
        p,
        mean,
        feat_comp,
        feat_struct,
        feat_mix_comp,
        feat_mix_struct,
        players: game.players,
        budget: game.budget,
        interaction_mass: game.interaction_mass,
        mixing_residual: game.mixing_residual,
        budget_se: se,
        budget_ci_lo: ci_lo,
        budget_ci_hi: ci_hi,
        seconds_fit,
        seconds_game,
        seconds_infer,
        seconds_total,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explain_v4_smoke_two_period_shift() {
        let n = 80usize;
        let p = 2usize;
        let mut xa = vec![0.0; n * p];
        let mut xb = vec![0.0; n * p];
        let mut ya = vec![0.0; n];
        let mut yb = vec![0.0; n];
        for i in 0..n {
            let a = (i as f64 / n as f64) * 2.0 - 1.0;
            let b = (((i * 5) % n) as f64 / n as f64) * 2.0 - 1.0;
            xa[i * p] = a;
            xa[i * p + 1] = b;
            xb[i * p] = a + 0.1;
            xb[i * p + 1] = b;
            ya[i] = a + 0.5 * b;
            yb[i] = a + 0.5 * b + 1.0; // pure structure location shift
        }
        let cfg = ExplainV4Config {
            fit: FitL2Config {
                k_channels: 3,
                pit_iters: 1,
                max_bins: 8,
                mean: crate::trace::meanboost::MeanConfig {
                    max_bins: 8,
                    learning_rate: 0.2,
                    max_rounds: 40,
                    early_stopping_rounds: 10,
                    early_stopping_tolerance: 1e-5,
                    validation_size: 0.15,
                    min_samples_leaf: 2,
                },
                ..FitL2Config::default()
            },
            mixture: MixtureConfig {
                n_y: 128,
                max_rows: Some(80),
                k_channels: 3,
                n_quad: 32,
                subsample_seed: 0,
            },
            infer: InferConfig {
                n_boot: 200,
                n_perm: 0,
                seed: 11,
                alpha: 0.05,
            },
            n_u: 32,
        };
        let rep = explain_v4(&xa, &ya, n, &xb, &yb, n, p, &cfg).unwrap();
        assert_eq!(rep.feat_comp.len(), p);
        assert!(rep.seconds_total > 0.0);
        assert!(rep.budget.is_finite());
        assert!(rep.budget_ci_lo <= rep.budget_ci_hi);
    }
}
