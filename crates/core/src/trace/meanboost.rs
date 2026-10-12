//! Cyclic squared-loss booster for the mean channel.
//!
//! One feature per step, histogram bins, a count-weighted centre into the intercept.
//! Early stopping holds out the last `validation_size` fraction of rows. The bins,
//! learning rate, round cap and patience match the additive booster used as the
//! mean baseline (max bins 1024, learning rate 0.04, 50_000 rounds, patience 100,
//! tolerance 1e-5, validation fraction 0.15, at least 4 rows in a bin). Bagging and
//! interaction terms are not part of this channel.

#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use crate::error::{Result, TraceError};
use crate::scalable::binning::{bin_column_into, edges_from_sorted};

#[derive(Clone, Debug)]
pub struct MeanConfig {
    pub max_bins: usize,
    pub learning_rate: f64,
    pub max_rounds: usize,
    pub early_stopping_rounds: usize,
    pub early_stopping_tolerance: f64,
    pub validation_size: f64,
    pub min_samples_leaf: u64,
}

impl Default for MeanConfig {
    fn default() -> Self {
        Self {
            max_bins: 1024,
            learning_rate: 0.04,
            max_rounds: 50_000,
            early_stopping_rounds: 100,
            early_stopping_tolerance: 1e-5,
            validation_size: 0.15,
            min_samples_leaf: 4,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MeanModel {
    pub intercept: f64,
    pub edges: Vec<Vec<f64>>,
    /// `effects[j][b]`, bin 0 is missing.
    pub effects: Vec<Vec<f64>>,
    pub rounds: usize,
    pub best_round: usize,
    pub train_mse: f64,
    pub val_mse: f64,
}

pub fn fit_mean(x: &[f64], n: usize, p: usize, y: &[f64], cfg: &MeanConfig) -> Result<MeanModel> {
    if n < 2 || p < 1 || x.len() != n * p || y.len() != n {
        return Err(TraceError::InvalidArgument(
            "mean booster needs matching X and y with at least 2 rows".into(),
        ));
    }
    if !(cfg.learning_rate.is_finite() && cfg.learning_rate > 0.0)
        || !(0.0..1.0).contains(&cfg.validation_size)
    {
        return Err(TraceError::InvalidArgument(
            "learning_rate must be positive and validation_size in [0, 1)".into(),
        ));
    }
    let n_val = ((cfg.validation_size * n as f64).floor() as usize).min(n - 1);
    let train_end = n - n_val;
    // Quantile cuts come from the training rows only. The validation slice is
    // scored with those cuts, and it does not move them.
    let (codes, edges, n_bins) = bin_columns(x, n, p, train_end, cfg.max_bins)?;
    let mut intercept = 0.0;
    for i in 0..train_end {
        intercept += y[i];
    }
    intercept /= train_end as f64;
    let mut effects: Vec<Vec<f64>> = n_bins.iter().map(|&nb| vec![0.0; nb]).collect();
    let mut pred = vec![intercept; n];
    let mut best_intercept = intercept;
    let mut best_effects = effects.clone();
    let mut best_mse = mse(&pred, y, train_end, n);
    let mut best_round = 0usize;
    let mut patience = 0usize;
    let mut executed = 0usize;
    for round in 0..cfg.max_rounds {
        for j in 0..p {
            step_feature(
                &mut pred,
                &mut intercept,
                &mut effects[j],
                &codes[j],
                y,
                train_end,
                cfg.learning_rate,
                cfg.min_samples_leaf,
            );
        }
        executed = round + 1;
        if n_val == 0 {
            best_round = executed;
            continue;
        }
        let cur = mse(&pred, y, train_end, n);
        let improve = best_mse - cur;
        let bar = cfg.early_stopping_tolerance * best_mse.abs().max(1e-12);
        if improve > bar {
            best_mse = cur;
            best_round = executed;
            patience = 0;
            best_intercept = intercept;
            best_effects = effects.clone();
        } else {
            patience += 1;
            if patience >= cfg.early_stopping_rounds {
                break;
            }
        }
    }
    if n_val > 0 {
        intercept = best_intercept;
        effects = best_effects;
    }
    let fitted = predict_mean_with(&intercept, &effects, &edges, x, n, p)?;
    let train_mse = mse_prefix(&fitted, y, train_end);
    Ok(MeanModel {
        intercept,
        edges,
        effects,
        rounds: executed,
        best_round,
        train_mse,
        val_mse: if n_val > 0 { best_mse } else { train_mse },
    })
}

pub fn predict_mean(model: &MeanModel, x: &[f64], n: usize) -> Result<Vec<f64>> {
    let p = model.edges.len();
    predict_mean_with(&model.intercept, &model.effects, &model.edges, x, n, p)
}

fn predict_mean_with(
    intercept: &f64,
    effects: &[Vec<f64>],
    edges: &[Vec<f64>],
    x: &[f64],
    n: usize,
    p: usize,
) -> Result<Vec<f64>> {
    if x.len() != n * p || effects.len() != p {
        return Err(TraceError::Shape("mean predict shape".into()));
    }
    let mut out = vec![*intercept; n];
    let mut col = vec![0.0; n];
    let mut codes = vec![0u16; n];
    for j in 0..p {
        for i in 0..n {
            col[i] = x[i * p + j];
        }
        bin_column_into(&col, &edges[j], &mut codes)?;
        let eff = &effects[j];
        for i in 0..n {
            out[i] += eff[codes[i] as usize];
        }
    }
    Ok(out)
}

fn step_feature(
    pred: &mut [f64],
    intercept: &mut f64,
    effect: &mut [f64],
    codes: &[u16],
    y: &[f64],
    train_end: usize,
    lr: f64,
    min_leaf: u64,
) {
    let nb = effect.len();
    let mut sum = vec![0.0; nb];
    let mut cnt = vec![0u64; nb];
    for i in 0..train_end {
        let b = codes[i] as usize;
        sum[b] += y[i] - pred[i];
        cnt[b] += 1;
    }
    // Adjacent bins are pooled until each pool has `min_leaf` rows. A hard
    // per-bin gate would leave every bin idle when max_bins is finer than the
    // sample, which is the usual case at 1024 bins.
    let raw = pooled_steps(&sum, &cnt, lr, min_leaf);
    let mut total = 0.0;
    let mut moment = 0.0;
    for b in 0..nb {
        total += cnt[b] as f64;
        moment += cnt[b] as f64 * raw[b];
    }
    let mean = if total > 0.0 { moment / total } else { 0.0 };
    *intercept += mean;
    for b in 0..nb {
        if cnt[b] > 0 {
            effect[b] += raw[b] - mean;
        }
    }
    for (i, &b) in codes.iter().enumerate() {
        pred[i] += raw[b as usize];
    }
}

fn pooled_steps(sum: &[f64], cnt: &[u64], lr: f64, min_leaf: u64) -> Vec<f64> {
    let nb = sum.len();
    let mut raw = vec![0.0; nb];
    let mut pools: Vec<(usize, usize, u64, f64)> = Vec::new();
    let mut i = 0;
    while i < nb {
        if cnt[i] == 0 {
            i += 1;
            continue;
        }
        let start = i;
        let mut c = 0u64;
        let mut s = 0.0;
        while i < nb && c < min_leaf.max(1) {
            c += cnt[i];
            s += sum[i];
            i += 1;
        }
        pools.push((start, i, c, s));
    }
    if pools.len() >= 2 {
        let last = pools.len() - 1;
        if pools[last].2 < min_leaf.max(1) {
            let tail = pools.pop().unwrap();
            let prev = pools.last_mut().unwrap();
            prev.1 = tail.1;
            prev.2 += tail.2;
            prev.3 += tail.3;
        }
    }
    for (start, end, c, s) in pools {
        if c == 0 {
            continue;
        }
        let delta = lr * s / c as f64;
        for b in start..end {
            if cnt[b] > 0 {
                raw[b] = delta;
            }
        }
    }
    raw
}

fn mse(pred: &[f64], y: &[f64], start: usize, end: usize) -> f64 {
    if end <= start {
        return 0.0;
    }
    let mut s = 0.0;
    for i in start..end {
        let d = pred[i] - y[i];
        s += d * d;
    }
    s / (end - start) as f64
}

fn mse_prefix(pred: &[f64], y: &[f64], n: usize) -> f64 {
    mse(pred, y, 0, n)
}

fn bin_columns(
    x: &[f64],
    n: usize,
    p: usize,
    train_end: usize,
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
        let mut finite: Vec<f64> = col[..train_end]
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        finite.sort_by(f64::total_cmp);
        let e = edges_from_sorted(&finite, max_bins)?;
        let mut c = vec![0u16; n];
        bin_column_into(&col, &e, &mut c)?;
        n_bins.push(e.len() + 2);
        codes.push(c);
        edges.push(e);
    }
    Ok((codes, edges, n_bins))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_mean_is_recovered() {
        let n = 80;
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..n {
            let v = i as f64 / (n - 1) as f64;
            x.push(v);
            y.push(0.2 + 1.7 * v);
        }
        let cfg = MeanConfig {
            max_bins: 8,
            learning_rate: 0.5,
            max_rounds: 40,
            early_stopping_rounds: 10,
            early_stopping_tolerance: 1e-8,
            validation_size: 0.0,
            min_samples_leaf: 1,
        };
        let model = fit_mean(&x, n, 1, &y, &cfg).unwrap();
        let pred = predict_mean(&model, &x, n).unwrap();
        let mut sse = 0.0;
        for i in 0..n {
            let d = pred[i] - y[i];
            sse += d * d;
        }
        let mse = sse / (n as f64);
        // Eight bins cannot interpolate a line. The null variance of y is about 0.24.
        assert!(mse < 0.01, "mse {mse}");
        assert!(
            pred[n - 1] - pred[0] > 1.4,
            "span {}",
            pred[n - 1] - pred[0]
        );
    }

    #[test]
    fn one_step_is_learning_rate_times_residual() {
        // Two occupied bins. The squared-loss Newton step in a bin is the mean
        // residual, multiplied by the learning rate. Centering puts the
        // count-weighted average of that step into the intercept and must not
        // change the prediction.
        let x = vec![0.0, 0.0, 1.0, 1.0];
        let y = vec![0.0, 0.0, 4.0, 4.0];
        let lr = 0.5;
        let cfg = MeanConfig {
            max_bins: 2,
            learning_rate: lr,
            max_rounds: 1,
            early_stopping_rounds: 5,
            early_stopping_tolerance: 1e-12,
            validation_size: 0.0,
            min_samples_leaf: 1,
        };
        let model = fit_mean(&x, 4, 1, &y, &cfg).unwrap();
        assert_eq!(model.rounds, 1);
        let pred = predict_mean(&model, &x, 4).unwrap();
        // Training mean is 2. Bin residuals are -2 and +2. One step moves them by lr.
        assert!((pred[0] - (2.0 - 2.0 * lr)).abs() < 1e-12, "{}", pred[0]);
        assert!((pred[1] - (2.0 - 2.0 * lr)).abs() < 1e-12, "{}", pred[1]);
        assert!((pred[2] - (2.0 + 2.0 * lr)).abs() < 1e-12, "{}", pred[2]);
        assert!((pred[3] - (2.0 + 2.0 * lr)).abs() < 1e-12, "{}", pred[3]);
        let nonzero: Vec<f64> = model.effects[0]
            .iter()
            .copied()
            .filter(|v| v.abs() > 1e-15)
            .collect();
        assert_eq!(nonzero.len(), 2, "effects {:?}", model.effects[0]);
        assert!((nonzero[0] + 2.0 * lr).abs() < 1e-12, "{nonzero:?}");
        assert!((nonzero[1] - 2.0 * lr).abs() < 1e-12, "{nonzero:?}");
        // The two steps cancel, so the intercept stays at the training mean.
        assert!((model.intercept - 2.0).abs() < 1e-12, "{}", model.intercept);
    }

    #[test]
    fn validation_rows_do_not_move_bin_edges() {
        // Training values lie in [0, 1]. The held-out tail sits at 100. A cut
        // computed on every row would place an edge on that tail.
        let n = 20;
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..16 {
            x[i] = i as f64 / 15.0;
            y[i] = 2.0 * x[i];
        }
        for i in 16..n {
            x[i] = 100.0;
            y[i] = 0.0;
        }
        let cfg = MeanConfig {
            max_bins: 8,
            learning_rate: 0.4,
            max_rounds: 2,
            early_stopping_rounds: 2,
            early_stopping_tolerance: 1e-8,
            validation_size: 0.2,
            min_samples_leaf: 1,
        };
        let model = fit_mean(&x, n, 1, &y, &cfg).unwrap();
        assert!(!model.edges[0].is_empty());
        for edge in &model.edges[0] {
            assert!(
                (0.0..=1.0).contains(edge),
                "validation row moved an edge to {edge}"
            );
        }
    }
}
