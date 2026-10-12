//! Location-anchored fit: squared-loss mean, then CRPS on the residual.
//!
//! Channel 0 is the mean booster. Channels k ≥ 1 are a locked-mean Newton fit of
//! the residual y − m̂(x) under pinball on the Gauss–Legendre nodes. A requested
//! monotone curve is the mean plus a weighted isotonic projection of the residual
//! curve, recentred so the quadrature mean stays m̂(x).

use super::legendre::{eval_a, weighted_pav};
use super::meanboost::{fit_mean, predict_mean, MeanConfig, MeanModel};
use super::v1fit::{
    channel_energy, curve_coefficients, feature_report_table, fit_v1, player_coefficients,
    V1Config, V1Model,
};
use crate::error::{Result, TraceError};

#[derive(Clone, Debug)]
pub struct HybridModel {
    pub mean: MeanModel,
    pub residual: V1Model,
}

pub fn fit_hybrid(
    x: &[f64],
    n: usize,
    p: usize,
    y: &[f64],
    mean_cfg: &MeanConfig,
    residual_cfg: &V1Config,
) -> Result<HybridModel> {
    let mean = fit_mean(x, n, p, y, mean_cfg)?;
    let fitted = predict_mean(&mean, x, n)?;
    let resid: Vec<f64> = y.iter().zip(&fitted).map(|(yi, mi)| yi - mi).collect();
    let mut cfg = residual_cfg.clone();
    cfg.lock_mean = true;
    let residual = fit_v1(x, n, p, &resid, &cfg)?;
    Ok(HybridModel { mean, residual })
}

pub fn predict_mean_channel(model: &HybridModel, x: &[f64], n: usize) -> Result<Vec<f64>> {
    predict_mean(&model.mean, x, n)
}

/// Quantiles at `u`. `project` applies the mean-preserving isotonic map.
pub fn predict_hybrid(
    model: &HybridModel,
    x: &[f64],
    n: usize,
    u: &[f64],
    project: bool,
) -> Result<Vec<f64>> {
    if u.is_empty() {
        return Err(TraceError::InvalidArgument("u is empty".into()));
    }
    let mean = predict_mean(&model.mean, x, n)?;
    let p = model.residual.edges.len();
    let k = model.residual.k_fit;
    let players = player_coefficients(&model.residual, x, n)?;
    let curves = curve_coefficients(&players, n, p, k);
    let nodes = &model.residual.basis.nodes;
    let weights = &model.residual.basis.weights;
    let nu = u.len();
    let mut out = vec![0.0; n * nu];
    for i in 0..n {
        let a = &curves[i * k..(i + 1) * k];
        let mut on_nodes: Vec<f64> = nodes.iter().map(|&ui| eval_a(a, ui)).collect();
        if project {
            if let Ok(z) = weighted_pav(&on_nodes, weights) {
                on_nodes = z;
            }
            let shift: f64 = on_nodes.iter().zip(weights).map(|(v, w)| w * v).sum();
            for v in &mut on_nodes {
                *v -= shift;
            }
        }
        for (t, &ui) in u.iter().enumerate() {
            out[i * nu + t] = mean[i] + interp(nodes, &on_nodes, ui);
        }
    }
    Ok(out)
}

pub fn hybrid_channel_energy(model: &HybridModel, feature: usize, channel: usize) -> Result<f64> {
    let tables = feature_report_table(&model.residual);
    if feature >= tables.len() {
        return Err(TraceError::InvalidArgument(
            "feature index out of range".into(),
        ));
    }
    Ok(channel_energy(
        &tables[feature],
        model.residual.k_report,
        channel,
    ))
}

fn interp(nodes: &[f64], values: &[f64], u: f64) -> f64 {
    let k = nodes.len();
    if k == 0 {
        return 0.0;
    }
    if u <= nodes[0] {
        return values[0];
    }
    if u >= nodes[k - 1] {
        return values[k - 1];
    }
    let j = nodes.partition_point(|&v| v <= u) - 1;
    let span = nodes[j + 1] - nodes[j];
    if span.abs() < 1e-15 {
        return values[j];
    }
    let t = (u - nodes[j]) / span;
    values[j] * (1.0 - t) + values[j + 1] * t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::meanboost::MeanConfig;
    use crate::trace::V1Config;

    #[test]
    fn locked_residual_has_zero_mean_channel() {
        let n = 60;
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..n {
            let v = i as f64 / (n - 1) as f64;
            x.push(v);
            let z = (i as f64 * 0.37).sin();
            y.push(1.0 + 2.0 * v + 0.3 * z);
        }
        let mean = MeanConfig {
            max_bins: 6,
            learning_rate: 0.4,
            max_rounds: 25,
            early_stopping_rounds: 8,
            early_stopping_tolerance: 1e-8,
            validation_size: 0.0,
            min_samples_leaf: 1,
        };
        let residual = V1Config {
            k_fit: 3,
            k_report: 3,
            n_u: 6,
            max_bins: 4,
            max_rounds: 8,
            tol: 1e-8,
            mono: crate::trace::MonoMode::None,
            ..V1Config::default()
        };
        let model = fit_hybrid(&x, n, 1, &y, &mean, &residual).unwrap();
        assert!(model.residual.intercept[0].abs() < 1e-12);
        for b in 0..model.residual.n_bins[0] {
            assert!(model.residual.effects[0][b * 3].abs() < 1e-8);
        }
        let m = predict_mean_channel(&model, &x, n).unwrap();
        let mut sse = 0.0;
        for i in 0..n {
            let d = m[i] - (1.0 + 2.0 * x[i]);
            sse += d * d;
        }
        assert!(sse / (n as f64) < 0.05, "mean mse {}", sse / (n as f64));
    }
}
