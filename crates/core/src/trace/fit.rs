#![allow(clippy::too_many_arguments, clippy::type_complexity)]
//! Cyclic histogram boosting of location, scale and shape channels.
//!
//! Each round evaluates the pinball subgradient once, then updates every feature
//! from that snapshot (Jacobi). Each feature update is centered, and the average
//! of those removed channel means is added once to the quantile level. That keeps
//! a common scale or shape correction (which every bin wants when the marginal
//! quantile is too wide) without letting `p` features add it `p` times. Features
//! of a dense `fit_trace` are accumulated in the canonical column order from
//! [`super::basis::column_order`], which makes a permutation of the columns run
//! the same additions.

use super::basis::{column_order, conformal_quantile, empirical_quantile, quantile_basis};
use super::checks::{pinball_scores, score_quantiles, DistScore};
use crate::error::{Result, TraceError};
use crate::estimators::SplitMix64;
use crate::scalable::binning::{bin_column, bin_column_into, edges_from_sorted};
use crate::scalable::pinball::pinball_grad_into;
use crate::scalable::BinCode;
use crate::w2::pava;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Hyper-parameters of a TRACE fit. Defaults are a single full-sample bag.
#[derive(Debug, Clone)]
pub struct TraceConfig {
    pub n_channels: usize,
    pub max_bins: usize,
    pub learning_rate: f64,
    pub max_rounds: usize,
    pub patience: usize,
    pub tol: f64,
    pub val_frac: f64,
    pub n_bags: usize,
    pub bag_frac: f64,
    pub gain_threshold: f64,
    pub select_frac: f64,
    pub max_interactions: usize,
    pub interaction_rounds: usize,
    pub screen_rows: usize,
    pub min_count: u64,
    pub u_levels: usize,
    pub seed: u64,
    /// Rounds that must run before early stopping can keep an earlier checkpoint,
    /// and before the gain screen is allowed to drop a feature.
    pub min_rounds: usize,
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            n_channels: 3,
            max_bins: 32,
            learning_rate: 1.0,
            max_rounds: 40,
            patience: 8,
            tol: 0.0,
            val_frac: 0.2,
            n_bags: 1,
            bag_frac: 1.0,
            gain_threshold: 0.0,
            select_frac: 0.7,
            max_interactions: 0,
            interaction_rounds: 15,
            screen_rows: 4096,
            min_count: 1,
            u_levels: 19,
            seed: 0,
            min_rounds: 20,
        }
    }
}

impl TraceConfig {
    fn validate(&self) -> Result<()> {
        if !(2..=4).contains(&self.n_channels) {
            return Err(TraceError::InvalidArgument(
                "n_channels must be 2, 3 or 4".into(),
            ));
        }
        if !(2..=65535).contains(&self.max_bins) {
            return Err(TraceError::InvalidArgument(
                "max_bins must be in [2, 65535]".into(),
            ));
        }
        if !(self.learning_rate.is_finite()
            && self.learning_rate > 0.0
            && self.learning_rate <= 5.0)
        {
            return Err(TraceError::InvalidArgument(
                "learning_rate must be in (0, 5]".into(),
            ));
        }
        if self.max_rounds > 100_000 || self.interaction_rounds > 100_000 {
            return Err(TraceError::InvalidArgument("too many rounds".into()));
        }
        if !(self.tol.is_finite() && self.tol >= 0.0) {
            return Err(TraceError::InvalidArgument("tol must be >= 0".into()));
        }
        if !(self.val_frac >= 0.0 && self.val_frac <= 0.5) {
            return Err(TraceError::InvalidArgument(
                "val_frac must be in [0, 0.5]".into(),
            ));
        }
        if !(1..=200).contains(&self.n_bags) {
            return Err(TraceError::InvalidArgument(
                "n_bags must be in [1, 200]".into(),
            ));
        }
        if !(self.bag_frac > 0.0 && self.bag_frac <= 1.0) {
            return Err(TraceError::InvalidArgument(
                "bag_frac must be in (0, 1]".into(),
            ));
        }
        if !self.gain_threshold.is_finite() {
            return Err(TraceError::InvalidArgument(
                "gain_threshold must be finite".into(),
            ));
        }
        if !(self.select_frac >= 0.0 && self.select_frac <= 1.0) {
            return Err(TraceError::InvalidArgument(
                "select_frac must be in [0, 1]".into(),
            ));
        }
        if self.max_interactions > 1000 || self.screen_rows == 0 {
            return Err(TraceError::InvalidArgument(
                "interaction settings out of range".into(),
            ));
        }
        if self.min_count == 0 {
            return Err(TraceError::InvalidArgument("min_count must be >= 1".into()));
        }
        if !(3..=401).contains(&self.u_levels) {
            return Err(TraceError::InvalidArgument(
                "u_levels must be in [3, 401]".into(),
            ));
        }
        if self.min_rounds > 100_000 {
            return Err(TraceError::InvalidArgument(
                "min_rounds is too large".into(),
            ));
        }
        Ok(())
    }
}

/// One selected pairwise term. `coef` is row-major `(n_bins_left, n_bins_right, M)`.
#[derive(Debug, Clone)]
pub struct PairTerm {
    pub left: usize,
    pub right: usize,
    pub n_bins_left: usize,
    pub n_bins_right: usize,
    pub coef: Vec<f64>,
    /// FAST-style residual-gradient energy on the screening subsample.
    pub score: f64,
}

/// A fitted TRACE model. Coefficients are indexed by the original column.
#[derive(Debug, Clone)]
pub struct TraceModel {
    pub u: Vec<f64>,
    pub basis: Vec<f64>,
    pub n_channels: usize,
    pub q0: Vec<f64>,
    pub edges: Vec<Vec<f64>>,
    pub n_bins: Vec<usize>,
    /// Canonical accumulation order (original column indices).
    pub order: Vec<usize>,
    /// Per feature, flat `(n_bins, M)` coefficients. Dropped features are zeros.
    pub coef: Vec<Vec<f64>>,
    /// Ensemble standard deviation of the bag coefficients, before dropping.
    pub std: Vec<Vec<f64>>,
    pub selected: Vec<bool>,
    /// Mean held-out pinball gain across bags that had a validation split.
    pub gains: Vec<f64>,
    /// Mean held-out gain of the shape channel alone, under tail-weighted pinball.
    pub shape_gains: Vec<f64>,
    pub select_counts: Vec<u32>,
    pub pairs: Vec<PairTerm>,
    pub train_loss: f64,
    pub val_loss: f64,
    pub bag_rounds: Vec<usize>,
    pub interaction_rounds: usize,
    /// Pinball on the training rows of bag 0 before round 1 and after every round.
    pub round_train: Vec<f64>,
    /// Same rounds, on the validation rows of bag 0.
    pub round_val: Vec<f64>,
    /// Mean absolute location-channel step in bag 0, one entry per round.
    pub round_step: Vec<f64>,
    /// Fraction of the location-step L2 that centering kept, bag 0, per round.
    pub round_center_kept: Vec<f64>,
    /// Per-level split-conformal offsets. Empty until [`TraceModel::calibrate`].
    pub calibration: Vec<f64>,
}

struct BagFit {
    coef: Vec<Vec<f64>>,
    q0: Vec<f64>,
    gains: Vec<f64>,
    shape_gains: Vec<f64>,
    selected: Vec<bool>,
    had_val: bool,
    rounds: usize,
}

fn fisher_yates<T>(a: &mut [T], rng: &mut SplitMix64) {
    let n = a.len();
    if n < 2 {
        return;
    }
    for i in (1..n).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        a.swap(i, j);
    }
}

fn split_mask(n: usize, seed: u64, val_frac: f64) -> Vec<u8> {
    let mut idx: Vec<usize> = (0..n).collect();
    fisher_yates(&mut idx, &mut SplitMix64(seed));
    let n_val = if val_frac <= 0.0 || n < 4 {
        0
    } else {
        ((val_frac * n as f64).round() as usize).clamp(1, n - 1)
    };
    let mut mask = vec![1u8; n];
    for &i in idx.iter().take(n_val) {
        mask[i] = 2;
    }
    mask
}

fn oob_mode(cfg: &TraceConfig) -> bool {
    cfg.n_bags > 1 && cfg.bag_frac < 1.0 - 1e-15
}

fn bag_mask(n: usize, cfg: &TraceConfig, bag: usize) -> Vec<u8> {
    let full = cfg.n_bags == 1 && cfg.bag_frac >= 1.0 - 1e-15;
    if full {
        return split_mask(
            n,
            cfg.seed.wrapping_add(20_011 * (bag as u64 + 1)),
            cfg.val_frac,
        );
    }
    if oob_mode(cfg) {
        // In-bag rows train. The complement is validation, so early stopping and
        // the gain screen see rows the bag did not fit. bag_frac = 1 has no
        // complement and stays on the path below.
        let mut perm: Vec<usize> = (0..n).collect();
        fisher_yates(
            &mut perm,
            &mut SplitMix64(cfg.seed.wrapping_add(10_007 * (bag as u64 + 1))),
        );
        let mut m = ((cfg.bag_frac * n as f64).round() as usize).clamp(2, n);
        if n >= 4 {
            m = m.min(n - 1);
        }
        perm.truncate(m);
        let mut mask = vec![2u8; n];
        for &i in &perm {
            mask[i] = 1;
        }
        return mask;
    }
    let mut perm: Vec<usize> = (0..n).collect();
    fisher_yates(
        &mut perm,
        &mut SplitMix64(cfg.seed.wrapping_add(10_007 * (bag as u64 + 1))),
    );
    let m = ((cfg.bag_frac * n as f64).round() as usize).clamp(2, n);
    perm.truncate(m);
    fisher_yates(
        &mut perm,
        &mut SplitMix64(cfg.seed.wrapping_add(20_011 * (bag as u64 + 1))),
    );
    let n_val = if cfg.val_frac <= 0.0 || perm.len() < 4 {
        0
    } else {
        ((cfg.val_frac * perm.len() as f64).round() as usize).clamp(1, perm.len() - 1)
    };
    let mut mask = vec![0u8; n];
    for (t, &i) in perm.iter().enumerate() {
        mask[i] = if t < n_val { 2 } else { 1 };
    }
    mask
}

fn masked_means(row_loss: &[f64], mask: &[u8]) -> (f64, f64, usize, usize) {
    let mut train_sum = 0.0;
    let mut val_sum = 0.0;
    let mut n_train = 0usize;
    let mut n_val = 0usize;
    for (i, &m) in mask.iter().enumerate() {
        if m == 1 {
            train_sum += row_loss[i];
            n_train += 1;
        } else if m == 2 {
            val_sum += row_loss[i];
            n_val += 1;
        }
    }
    let train = if n_train == 0 {
        f64::NAN
    } else {
        train_sum / n_train as f64
    };
    let val = if n_val == 0 {
        train
    } else {
        val_sum / n_val as f64
    };
    (train, val, n_train, n_val)
}

fn feature_delta<T: BinCode>(
    codes: &[T],
    grad: &[f64],
    mask: &[u8],
    n_bins: usize,
    m_ch: usize,
    k: usize,
    lr: f64,
    min_count: u64,
) -> (Vec<f64>, f64, Vec<f64>, bool) {
    let mut sums = vec![0.0; n_bins * m_ch];
    let mut counts = vec![0u64; n_bins];
    for i in 0..codes.len() {
        if mask[i] != 1 {
            continue;
        }
        let b: usize = codes[i].into();
        counts[b] += 1;
        for m in 0..m_ch {
            sums[b * m_ch + m] += grad[i * m_ch + m];
        }
    }
    let mut delta = vec![0.0; n_bins * m_ch];
    let mut weight = 0.0;
    let scale = lr / k as f64;
    for b in 0..n_bins {
        if counts[b] >= min_count {
            weight += counts[b] as f64;
            let inv = scale / counts[b] as f64;
            for m in 0..m_ch {
                delta[b * m_ch + m] = sums[b * m_ch + m] * inv;
            }
        }
    }
    let active = weight > 0.0;
    let (kept, means) = center_delta(&mut delta, &counts, m_ch, min_count, weight);
    (delta, kept, means, active)
}

/// Subtract the count-weighted mean of each channel.
///
/// Returns the fraction of the location-channel L2 that remains, and the means
/// that were removed. Callers add the average of those means back once, into the
/// shared quantile level, so a common scale correction is not discarded and is
/// not added once per feature.
fn center_delta(
    delta: &mut [f64],
    counts: &[u64],
    m_ch: usize,
    min_count: u64,
    weight: f64,
) -> (f64, Vec<f64>) {
    let mut means = vec![0.0; m_ch];
    if weight <= 0.0 {
        return (1.0, means);
    }
    let mut before = 0.0;
    for b in 0..counts.len() {
        if counts[b] >= min_count {
            let v = delta[b * m_ch];
            before += v * v;
        }
    }
    for m in 0..m_ch {
        let mut mean = 0.0;
        for b in 0..counts.len() {
            if counts[b] >= min_count {
                mean += delta[b * m_ch + m] * counts[b] as f64;
            }
        }
        means[m] = mean / weight;
        for b in 0..counts.len() {
            if counts[b] >= min_count {
                delta[b * m_ch + m] -= means[m];
            }
        }
    }
    let mut after = 0.0;
    for b in 0..counts.len() {
        if counts[b] >= min_count {
            let v = delta[b * m_ch];
            after += v * v;
        }
    }
    let kept = if before <= 0.0 {
        1.0
    } else {
        (after / before).sqrt()
    };
    (kept, means)
}

fn add_channel_step(level: &mut [f64], basis: &[f64], m_ch: usize, step: &[f64]) {
    let k = level.len();
    for kk in 0..k {
        for m in 0..m_ch {
            level[kk] += step[m] * basis[kk * m_ch + m];
        }
    }
}

fn apply_delta<T: BinCode>(
    theta: &mut [f64],
    coef: &mut [f64],
    codes: &[T],
    delta: &[f64],
    m_ch: usize,
) {
    let n_bins = delta.len() / m_ch;
    for b in 0..n_bins {
        for m in 0..m_ch {
            coef[b * m_ch + m] += delta[b * m_ch + m];
        }
    }
    #[cfg(feature = "parallel")]
    {
        theta.par_chunks_mut(m_ch).enumerate().for_each(|(i, row)| {
            let b: usize = codes[i].into();
            let d = &delta[b * m_ch..(b + 1) * m_ch];
            for m in 0..m_ch {
                row[m] += d[m];
            }
        });
    }
    #[cfg(not(feature = "parallel"))]
    {
        for (i, row) in theta.chunks_mut(m_ch).enumerate() {
            let b: usize = codes[i].into();
            for m in 0..m_ch {
                row[m] += delta[b * m_ch + m];
            }
        }
    }
}

fn rebuild_theta<T: BinCode>(
    theta: &mut [f64],
    codes: &[T],
    n: usize,
    order: &[usize],
    coef: &[Vec<f64>],
    pairs: &[PairTerm],
    m_ch: usize,
) {
    theta.fill(0.0);
    for &j in order {
        let col = &codes[j * n..(j + 1) * n];
        let cf = &coef[j];
        for (i, row) in theta.chunks_mut(m_ch).enumerate() {
            let b: usize = col[i].into();
            for m in 0..m_ch {
                row[m] += cf[b * m_ch + m];
            }
        }
    }
    for pair in pairs {
        let cl = &codes[pair.left * n..(pair.left + 1) * n];
        let cr = &codes[pair.right * n..(pair.right + 1) * n];
        let nb_r = pair.n_bins_right;
        for (i, row) in theta.chunks_mut(m_ch).enumerate() {
            let bj: usize = cl[i].into();
            let bk: usize = cr[i].into();
            let base = (bj * nb_r + bk) * m_ch;
            for m in 0..m_ch {
                row[m] += pair.coef[base + m];
            }
        }
    }
}

fn tail_weights(u: &[f64]) -> Vec<f64> {
    let mut w = vec![0.0; u.len()];
    let mut s = 0.0;
    for (k, &ui) in u.iter().enumerate() {
        let d = ui - 0.5;
        w[k] = d * d;
        s += w[k];
    }
    if s > 0.0 {
        for v in &mut w {
            *v /= s;
        }
    }
    w
}

fn loss_on_rows(
    y: &[f64],
    theta: &[f64],
    q0: &[f64],
    basis: &[f64],
    u: &[f64],
    rows: &[usize],
    m_ch: usize,
    weights: Option<&[f64]>,
) -> f64 {
    if rows.is_empty() {
        return f64::NAN;
    }
    let k = u.len();
    let mut total = 0.0;
    for &i in rows {
        let th = &theta[i * m_ch..(i + 1) * m_ch];
        let mut acc = 0.0;
        for kk in 0..k {
            let b = &basis[kk * m_ch..(kk + 1) * m_ch];
            let mut q = q0[kk];
            for m in 0..m_ch {
                q += th[m] * b[m];
            }
            let res = y[i] - q;
            let r = if res < 0.0 { u[kk] - 1.0 } else { u[kk] };
            let pin = res * r;
            acc += match weights {
                Some(w) => pin * w[kk],
                None => pin / k as f64,
            };
        }
        total += acc;
    }
    total / rows.len() as f64
}

fn feature_gains<T: BinCode>(
    y: &[f64],
    theta: &mut [f64],
    q0: &[f64],
    basis: &[f64],
    u: &[f64],
    codes: &[T],
    n: usize,
    coef: &[Vec<f64>],
    val_rows: &[usize],
    threshold: f64,
    m_ch: usize,
    kept_round: usize,
    min_rounds: usize,
) -> (Vec<f64>, Vec<f64>, Vec<bool>) {
    let p = coef.len();
    let base = loss_on_rows(y, theta, q0, basis, u, val_rows, m_ch, None);
    let weights = tail_weights(u);
    let base_shape = if m_ch >= 3 {
        loss_on_rows(y, theta, q0, basis, u, val_rows, m_ch, Some(&weights))
    } else {
        0.0
    };
    let mut gains = vec![0.0; p];
    let mut shape_gains = vec![0.0; p];
    let mut selected = vec![false; p];
    for j in 0..p {
        let col = &codes[j * n..(j + 1) * n];
        let cf = &coef[j];
        for &i in val_rows {
            let b: usize = col[i].into();
            for m in 0..m_ch {
                theta[i * m_ch + m] -= cf[b * m_ch + m];
            }
        }
        let loss_j = loss_on_rows(y, theta, q0, basis, u, val_rows, m_ch, None);
        gains[j] = loss_j - base;
        for &i in val_rows {
            let b: usize = col[i].into();
            for m in 0..m_ch {
                theta[i * m_ch + m] += cf[b * m_ch + m];
            }
        }
        if m_ch >= 3 {
            for &i in val_rows {
                let b: usize = col[i].into();
                theta[i * m_ch + 2] -= cf[b * m_ch + 2];
            }
            let loss_s = loss_on_rows(y, theta, q0, basis, u, val_rows, m_ch, Some(&weights));
            shape_gains[j] = loss_s - base_shape;
            for &i in val_rows {
                let b: usize = col[i].into();
                theta[i * m_ch + 2] += cf[b * m_ch + 2];
            }
        }
        selected[j] = gains[j] > threshold || shape_gains[j] > threshold;
    }
    if kept_round < min_rounds {
        selected.fill(true);
    }
    (gains, shape_gains, selected)
}

fn eval_grad(
    y: &[f64],
    theta: &[f64],
    q0: &[f64],
    basis: &[f64],
    u: &[f64],
    m_ch: usize,
    grad: &mut [f64],
    row_loss: &mut [f64],
    mask: &[u8],
) -> (f64, f64, usize, usize) {
    pinball_grad_into(y, theta, q0, basis, u, m_ch, grad, row_loss);
    masked_means(row_loss, mask)
}

struct RoundTrace {
    train: Vec<f64>,
    val: Vec<f64>,
    step: Vec<f64>,
    center_kept: Vec<f64>,
}

fn boost_main<T: BinCode>(
    codes: &[T],
    n: usize,
    order: &[usize],
    n_bins: &[usize],
    y: &[f64],
    mask: &[u8],
    q0: &[f64],
    basis: &[f64],
    u: &[f64],
    cfg: &TraceConfig,
    log_rounds: bool,
) -> (Vec<Vec<f64>>, usize, Vec<f64>, usize, RoundTrace) {
    let p = n_bins.len();
    let m_ch = cfg.n_channels;
    let k = u.len();
    let mut level = q0.to_vec();
    let mut coef: Vec<Vec<f64>> = (0..p).map(|j| vec![0.0; n_bins[j] * m_ch]).collect();
    let mut theta = vec![0.0; n * m_ch];
    let mut grad = vec![0.0; n * m_ch];
    let mut row_loss = vec![0.0; n];
    let (train0, val0, _, n_val) = eval_grad(
        y,
        &theta,
        &level,
        basis,
        u,
        m_ch,
        &mut grad,
        &mut row_loss,
        mask,
    );
    let mut trace = RoundTrace {
        train: Vec::new(),
        val: Vec::new(),
        step: Vec::new(),
        center_kept: Vec::new(),
    };
    if log_rounds {
        trace.train.push(train0);
        trace.val.push(val0);
    }
    let use_es = cfg.patience > 0 && n_val > 0 && (cfg.val_frac > 0.0 || oob_mode(cfg));
    let mut best_val = val0;
    let mut best_coef = coef.clone();
    let mut best_level = level.clone();
    let mut best_round = 0usize;
    let mut stale = 0usize;
    let mut rounds = 0usize;
    for _r in 0..cfg.max_rounds {
        let mut abs_step = 0.0;
        let mut step_bins = 0.0;
        let mut kept_sum = 0.0;
        let mut kept_n = 0.0;
        let mut mean_sum = vec![0.0; m_ch];
        let mut n_used = 0.0;
        for &j in order {
            let (delta, kept, means, active) = feature_delta(
                &codes[j * n..(j + 1) * n],
                &grad,
                mask,
                n_bins[j],
                m_ch,
                k,
                cfg.learning_rate,
                cfg.min_count,
            );
            if active {
                for m in 0..m_ch {
                    mean_sum[m] += means[m];
                }
                n_used += 1.0;
            }
            if log_rounds {
                for b in 0..n_bins[j] {
                    let v = delta[b * m_ch];
                    if v != 0.0 {
                        abs_step += v.abs();
                        step_bins += 1.0;
                    }
                }
                kept_sum += kept;
                kept_n += 1.0;
            }
            apply_delta(
                &mut theta,
                &mut coef[j],
                &codes[j * n..(j + 1) * n],
                &delta,
                m_ch,
            );
        }
        if n_used > 0.0 {
            let step: Vec<f64> = mean_sum.iter().map(|v| v / n_used).collect();
            add_channel_step(&mut level, basis, m_ch, &step);
        }
        let (train_loss, val_loss, _, _) = eval_grad(
            y,
            &theta,
            &level,
            basis,
            u,
            m_ch,
            &mut grad,
            &mut row_loss,
            mask,
        );
        rounds += 1;
        if log_rounds {
            trace.step.push(if step_bins > 0.0 {
                abs_step / step_bins
            } else {
                0.0
            });
            trace
                .center_kept
                .push(if kept_n > 0.0 { kept_sum / kept_n } else { 1.0 });
            trace.train.push(train_loss);
            trace.val.push(val_loss);
        }
        if use_es && cfg.min_rounds > 0 && rounds <= cfg.min_rounds {
            // The checkpoint has to contain these updates before the screen may drop.
            best_val = val_loss;
            best_coef.clone_from(&coef);
            best_level.clone_from(&level);
            best_round = rounds;
            stale = 0;
        } else if use_es {
            if val_loss < best_val - cfg.tol {
                best_val = val_loss;
                best_coef.clone_from(&coef);
                best_level.clone_from(&level);
                best_round = rounds;
                stale = 0;
            } else {
                stale += 1;
                if stale >= cfg.patience {
                    break;
                }
            }
        }
    }
    if use_es {
        coef = best_coef;
        level = best_level;
    } else {
        best_round = rounds;
    }
    (coef, rounds, level, best_round, trace)
}

fn pair_delta<T: BinCode>(
    left: &[T],
    right: &[T],
    grad: &[f64],
    mask: &[u8],
    n_bins_l: usize,
    n_bins_r: usize,
    m_ch: usize,
    k: usize,
    lr: f64,
    min_count: u64,
) -> (Vec<f64>, Vec<f64>, bool) {
    let n_cells = n_bins_l * n_bins_r;
    let mut sums = vec![0.0; n_cells * m_ch];
    let mut counts = vec![0u64; n_cells];
    for i in 0..left.len() {
        if mask[i] != 1 {
            continue;
        }
        let cell = left[i].into() * n_bins_r + right[i].into();
        counts[cell] += 1;
        for m in 0..m_ch {
            sums[cell * m_ch + m] += grad[i * m_ch + m];
        }
    }
    let mut delta = vec![0.0; n_cells * m_ch];
    let mut weight = 0.0;
    let scale = lr / k as f64;
    for c in 0..n_cells {
        if counts[c] >= min_count {
            weight += counts[c] as f64;
            let inv = scale / counts[c] as f64;
            for m in 0..m_ch {
                delta[c * m_ch + m] = sums[c * m_ch + m] * inv;
            }
        }
    }
    let (kept, means) = center_delta(&mut delta, &counts, m_ch, min_count, weight);
    let _ = kept;
    (delta, means, weight > 0.0)
}

fn pair_score<T: BinCode>(
    left: &[T],
    right: &[T],
    grad: &[f64],
    rows: &[usize],
    n_bins_l: usize,
    n_bins_r: usize,
    m_ch: usize,
    k: usize,
) -> f64 {
    let n_cells = n_bins_l * n_bins_r;
    let mut sums = vec![0.0; n_cells * m_ch];
    let mut counts = vec![0u64; n_cells];
    for &i in rows {
        let cell: usize = left[i].into() * n_bins_r + right[i].into();
        counts[cell] += 1;
        for m in 0..m_ch {
            sums[cell * m_ch + m] += grad[i * m_ch + m];
        }
    }
    let mut score = 0.0;
    let inv_k = 1.0 / k as f64;
    for c in 0..n_cells {
        if counts[c] == 0 {
            continue;
        }
        let mut energy = 0.0;
        for m in 0..m_ch {
            let mean = sums[c * m_ch + m] / counts[c] as f64 * inv_k;
            energy += mean * mean;
        }
        score += counts[c] as f64 * energy;
    }
    score
}

fn boost_pairs<T: BinCode>(
    codes: &[T],
    n: usize,
    order: &[usize],
    n_bins: &[usize],
    y: &[f64],
    mask: &[u8],
    q0: &mut [f64],
    basis: &[f64],
    u: &[f64],
    coef: &mut [Vec<f64>],
    selected: &[bool],
    cfg: &TraceConfig,
) -> (Vec<PairTerm>, usize) {
    if cfg.max_interactions == 0 || cfg.interaction_rounds == 0 {
        return (Vec::new(), 0);
    }
    let eligible: Vec<usize> = order.iter().copied().filter(|&j| selected[j]).collect();
    if eligible.len() < 2 {
        return (Vec::new(), 0);
    }
    let p = n_bins.len();
    let m_ch = cfg.n_channels;
    let k = u.len();
    let mut pos = vec![0usize; p];
    for (rank, &j) in order.iter().enumerate() {
        pos[j] = rank;
    }
    let mut level = q0.to_vec();
    let mut theta = vec![0.0; n * m_ch];
    rebuild_theta(&mut theta, codes, n, order, coef, &[], m_ch);
    let mut grad = vec![0.0; n * m_ch];
    let mut row_loss = vec![0.0; n];
    let (_, val0, _, n_val) = eval_grad(
        y,
        &theta,
        &level,
        basis,
        u,
        m_ch,
        &mut grad,
        &mut row_loss,
        mask,
    );
    let mut rows: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter(|(_, m)| **m == 1)
        .map(|(i, _)| i)
        .collect();
    fisher_yates(&mut rows, &mut SplitMix64(cfg.seed.wrapping_add(42_042)));
    let take = rows.len().min(cfg.screen_rows);
    rows.truncate(take);
    rows.sort_unstable();
    let mut cands: Vec<(f64, usize, usize, usize, usize)> = Vec::new();
    for a in 0..eligible.len() {
        for b in (a + 1)..eligible.len() {
            let j = eligible[a];
            let kft = eligible[b];
            let score = pair_score(
                &codes[j * n..(j + 1) * n],
                &codes[kft * n..(kft + 1) * n],
                &grad,
                &rows,
                n_bins[j],
                n_bins[kft],
                m_ch,
                k,
            );
            if score > 0.0 {
                cands.push((score, pos[j], pos[kft], j, kft));
            }
        }
    }
    cands.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    cands.truncate(cfg.max_interactions);
    if cands.is_empty() {
        return (Vec::new(), 0);
    }
    let mut pairs: Vec<PairTerm> = cands
        .into_iter()
        .map(|(score, _, _, j, kft)| PairTerm {
            left: j,
            right: kft,
            n_bins_left: n_bins[j],
            n_bins_right: n_bins[kft],
            coef: vec![0.0; n_bins[j] * n_bins[kft] * m_ch],
            score,
        })
        .collect();
    let use_es = cfg.patience > 0 && n_val > 0 && (cfg.val_frac > 0.0 || oob_mode(cfg));
    let mut best_val = val0;
    let mut best_pairs = pairs.clone();
    let mut best_level = level.clone();
    let mut stale = 0usize;
    let mut rounds = 0usize;
    for _r in 0..cfg.interaction_rounds {
        let mut deltas = Vec::with_capacity(pairs.len());
        let mut mean_sum = vec![0.0; m_ch];
        let mut n_used = 0.0;
        for pair in &pairs {
            let (delta, means, active) = pair_delta(
                &codes[pair.left * n..(pair.left + 1) * n],
                &codes[pair.right * n..(pair.right + 1) * n],
                &grad,
                mask,
                pair.n_bins_left,
                pair.n_bins_right,
                m_ch,
                k,
                cfg.learning_rate,
                cfg.min_count,
            );
            if active {
                for m in 0..m_ch {
                    mean_sum[m] += means[m];
                }
                n_used += 1.0;
            }
            deltas.push(delta);
        }
        for (pair, delta) in pairs.iter_mut().zip(&deltas) {
            for (c, d) in pair.coef.iter_mut().zip(delta) {
                *c += *d;
            }
            let cl = &codes[pair.left * n..(pair.left + 1) * n];
            let cr = &codes[pair.right * n..(pair.right + 1) * n];
            let nb_r = pair.n_bins_right;
            for (i, row) in theta.chunks_mut(m_ch).enumerate() {
                let bj: usize = cl[i].into();
                let bk: usize = cr[i].into();
                let base = (bj * nb_r + bk) * m_ch;
                for m in 0..m_ch {
                    row[m] += delta[base + m];
                }
            }
        }
        if n_used > 0.0 {
            let step: Vec<f64> = mean_sum.iter().map(|v| v / n_used).collect();
            add_channel_step(&mut level, basis, m_ch, &step);
        }
        let (_, val_loss, _, _) = eval_grad(
            y,
            &theta,
            &level,
            basis,
            u,
            m_ch,
            &mut grad,
            &mut row_loss,
            mask,
        );
        rounds += 1;
        if use_es {
            if val_loss < best_val - cfg.tol {
                best_val = val_loss;
                best_pairs.clone_from(&pairs);
                best_level.clone_from(&level);
                stale = 0;
            } else {
                stale += 1;
                if stale >= cfg.patience {
                    break;
                }
            }
        }
    }
    if use_es {
        pairs = best_pairs;
        level = best_level;
    }
    purify_pairs(&mut pairs, coef, codes, n, mask, m_ch);
    q0.copy_from_slice(&level);
    (pairs, rounds)
}

/// Move each pair's row and column margins into the main effects.
///
/// Histogram weights are not an outer product, so one subtraction of both
/// margins does not zero them. Alternating weighted centering does: each removed
/// row mean is added to the left feature and each removed column mean to the
/// right feature, in pair order. The level is left unchanged because the grand
/// mean travels with those margins. Predictions are unchanged.
fn purify_pairs<T: BinCode>(
    pairs: &mut [PairTerm],
    coef: &mut [Vec<f64>],
    codes: &[T],
    n: usize,
    mask: &[u8],
    m_ch: usize,
) {
    for pair in pairs.iter_mut() {
        let nb_l = pair.n_bins_left;
        let nb_r = pair.n_bins_right;
        let left = &codes[pair.left * n..(pair.left + 1) * n];
        let right = &codes[pair.right * n..(pair.right + 1) * n];
        let mut counts = vec![0u64; nb_l * nb_r];
        for i in 0..n {
            if mask[i] != 1 {
                continue;
            }
            counts[left[i].into() * nb_r + right[i].into()] += 1;
        }
        let mut row_w = vec![0.0; nb_l];
        let mut col_w = vec![0.0; nb_r];
        for i in 0..nb_l {
            for j in 0..nb_r {
                let c = counts[i * nb_r + j] as f64;
                row_w[i] += c;
                col_w[j] += c;
            }
        }
        for _pass in 0..24 {
            let mut moved = 0.0f64;
            for m in 0..m_ch {
                for i in 0..nb_l {
                    if row_w[i] <= 0.0 {
                        continue;
                    }
                    let mut sum = 0.0;
                    for j in 0..nb_r {
                        let c = counts[i * nb_r + j] as f64;
                        sum += pair.coef[(i * nb_r + j) * m_ch + m] * c;
                    }
                    let mean = sum / row_w[i];
                    moved = moved.max(mean.abs());
                    if mean == 0.0 {
                        continue;
                    }
                    for j in 0..nb_r {
                        if counts[i * nb_r + j] == 0 {
                            continue;
                        }
                        pair.coef[(i * nb_r + j) * m_ch + m] -= mean;
                    }
                    coef[pair.left][i * m_ch + m] += mean;
                }
                for j in 0..nb_r {
                    if col_w[j] <= 0.0 {
                        continue;
                    }
                    let mut sum = 0.0;
                    for i in 0..nb_l {
                        let c = counts[i * nb_r + j] as f64;
                        sum += pair.coef[(i * nb_r + j) * m_ch + m] * c;
                    }
                    let mean = sum / col_w[j];
                    moved = moved.max(mean.abs());
                    if mean == 0.0 {
                        continue;
                    }
                    for i in 0..nb_l {
                        if counts[i * nb_r + j] == 0 {
                            continue;
                        }
                        pair.coef[(i * nb_r + j) * m_ch + m] -= mean;
                    }
                    coef[pair.right][j * m_ch + m] += mean;
                }
            }
            if moved < 1e-12 {
                break;
            }
        }
    }
}

fn check_codes<T: BinCode>(codes: &[T], n: usize, p: usize, n_bins: &[usize]) -> Result<()> {
    for j in 0..p {
        for &c in &codes[j * n..(j + 1) * n] {
            if c.into() >= n_bins[j] {
                return Err(TraceError::InvalidArgument(
                    "bin code is outside the edge set".into(),
                ));
            }
        }
    }
    Ok(())
}

fn frequency_keeps(count: u32, votes: u32, frac: f64) -> bool {
    if votes == 0 {
        return true;
    }
    count as f64 + 1e-12 >= frac * votes as f64
}

fn assemble<T: BinCode>(
    codes: &[T],
    n: usize,
    p: usize,
    edges: Vec<Vec<f64>>,
    n_bins: Vec<usize>,
    order: Vec<usize>,
    y: &[f64],
    cfg: &TraceConfig,
) -> Result<TraceModel> {
    check_codes(codes, n, p, &n_bins)?;
    let (u, basis) = quantile_basis(cfg.u_levels, cfg.n_channels)?;
    let m_ch = cfg.n_channels;
    let mut bags = Vec::with_capacity(cfg.n_bags);
    let mut round_train = Vec::new();
    let mut round_val = Vec::new();
    let mut round_step = Vec::new();
    let mut round_center_kept = Vec::new();
    for b in 0..cfg.n_bags {
        let mask = bag_mask(n, cfg, b);
        let train_y: Vec<f64> = y
            .iter()
            .zip(&mask)
            .filter(|(_, m)| **m == 1)
            .map(|(v, _)| *v)
            .collect();
        if train_y.is_empty() {
            return Err(TraceError::InvalidArgument(
                "a bag has no training rows".into(),
            ));
        }
        let q0_emp = empirical_quantile(&train_y, &u)?;
        let (coef, rounds, q0, kept_round, trace) = boost_main(
            codes,
            n,
            &order,
            &n_bins,
            y,
            &mask,
            &q0_emp,
            &basis,
            &u,
            cfg,
            b == 0,
        );
        if b == 0 {
            round_train = trace.train;
            round_val = trace.val;
            round_step = trace.step;
            round_center_kept = trace.center_kept;
        }
        let mut theta = vec![0.0; n * m_ch];
        rebuild_theta(&mut theta, codes, n, &order, &coef, &[], m_ch);
        let val_rows: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter(|(_, m)| **m == 2)
            .map(|(i, _)| i)
            .collect();
        let had_val = !val_rows.is_empty();
        let (gains, shape_gains, selected) = if had_val {
            feature_gains(
                y,
                &mut theta,
                &q0,
                &basis,
                &u,
                codes,
                n,
                &coef,
                &val_rows,
                cfg.gain_threshold,
                m_ch,
                kept_round,
                cfg.min_rounds,
            )
        } else {
            (vec![0.0; p], vec![0.0; p], vec![true; p])
        };
        bags.push(BagFit {
            coef,
            q0,
            gains,
            shape_gains,
            selected,
            had_val,
            rounds,
        });
    }
    let votes = bags.iter().filter(|b| b.had_val).count() as u32;
    let mut select_counts = vec![0u32; p];
    let mut gain_sum = vec![0.0; p];
    let mut shape_sum = vec![0.0; p];
    for bag in &bags {
        if bag.had_val {
            for j in 0..p {
                if bag.selected[j] {
                    select_counts[j] += 1;
                }
                gain_sum[j] += bag.gains[j];
                shape_sum[j] += bag.shape_gains[j];
            }
        }
    }
    let selected: Vec<bool> = (0..p)
        .map(|j| frequency_keeps(select_counts[j], votes, cfg.select_frac))
        .collect();
    let gains = if votes == 0 {
        vec![0.0; p]
    } else {
        gain_sum.iter().map(|g| g / votes as f64).collect()
    };
    let shape_gains = if votes == 0 {
        vec![0.0; p]
    } else {
        shape_sum.iter().map(|g| g / votes as f64).collect()
    };
    let mut coef: Vec<Vec<f64>> = (0..p).map(|j| vec![0.0; n_bins[j] * m_ch]).collect();
    let mut std = coef.clone();
    let n_bags_f = cfg.n_bags as f64;
    for j in 0..p {
        let len = coef[j].len();
        for t in 0..len {
            let mut mean = 0.0;
            for bag in &bags {
                mean += bag.coef[j][t];
            }
            mean /= n_bags_f;
            let mut var = 0.0;
            if cfg.n_bags > 1 {
                for bag in &bags {
                    let d = bag.coef[j][t] - mean;
                    var += d * d;
                }
                var /= (cfg.n_bags - 1) as f64;
            }
            std[j][t] = var.sqrt();
            coef[j][t] = if selected[j] { mean } else { 0.0 };
        }
    }
    let mut q0 = vec![0.0; u.len()];
    for bag in &bags {
        for (a, bq) in q0.iter_mut().zip(&bag.q0) {
            *a += *bq;
        }
    }
    for v in &mut q0 {
        *v /= n_bags_f;
    }
    let report = if oob_mode(cfg) {
        bag_mask(n, cfg, 0)
    } else {
        split_mask(n, cfg.seed.wrapping_add(20_011), cfg.val_frac)
    };
    let (pairs, interaction_rounds) = boost_pairs(
        codes, n, &order, &n_bins, y, &report, &mut q0, &basis, &u, &mut coef, &selected, cfg,
    );
    let mut theta = vec![0.0; n * m_ch];
    rebuild_theta(&mut theta, codes, n, &order, &coef, &pairs, m_ch);
    let mut grad = vec![0.0; n * m_ch];
    let mut row_loss = vec![0.0; n];
    let (train_loss, val_loss, _, _) = eval_grad(
        y,
        &theta,
        &q0,
        &basis,
        &u,
        m_ch,
        &mut grad,
        &mut row_loss,
        &report,
    );
    Ok(TraceModel {
        u,
        basis,
        n_channels: m_ch,
        q0,
        edges,
        n_bins,
        order,
        coef,
        std,
        selected,
        gains,
        shape_gains,
        select_counts,
        pairs,
        train_loss,
        val_loss,
        bag_rounds: bags.iter().map(|b| b.rounds).collect(),
        interaction_rounds,
        round_train,
        round_val,
        round_step,
        round_center_kept,
        calibration: Vec::new(),
    })
}

fn bin_matrix<T: BinCode + Default>(
    x: &[f64],
    n: usize,
    p: usize,
    max_bins: usize,
) -> Result<(Vec<T>, Vec<Vec<f64>>, Vec<usize>)> {
    let mut codes = vec![T::from_usize(0); p * n];
    let mut edges = Vec::with_capacity(p);
    let mut n_bins = Vec::with_capacity(p);
    let mut col = vec![0.0; n];
    let mut finite = Vec::with_capacity(n);
    for j in 0..p {
        for i in 0..n {
            col[i] = x[i * p + j];
        }
        finite.clear();
        finite.extend(col.iter().copied().filter(|v| v.is_finite()));
        finite.sort_by(f64::total_cmp);
        let e = edges_from_sorted(&finite, max_bins)?;
        let nb = e.len() + 2;
        bin_column_into(&col, &e, &mut codes[j * n..(j + 1) * n])?;
        edges.push(e);
        n_bins.push(nb);
    }
    Ok((codes, edges, n_bins))
}

fn prepare_y(y: &[f64], n: usize, p: usize) -> Result<()> {
    if n < 2 || p < 1 {
        return Err(TraceError::InvalidArgument(
            "need at least 2 rows and 1 column".into(),
        ));
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(TraceError::InvalidArgument("y must be finite".into()));
    }
    Ok(())
}

/// Fit from a dense row-major matrix. Column order follows [`column_order`].
pub fn fit_trace(
    x: &[f64],
    n: usize,
    p: usize,
    y: &[f64],
    cfg: &TraceConfig,
) -> Result<TraceModel> {
    cfg.validate()?;
    if x.len() != n * p || y.len() != n {
        return Err(TraceError::Shape("X and y shapes disagree".into()));
    }
    prepare_y(y, n, p)?;
    let order = column_order(x, n, p);
    if cfg.max_bins <= 255 {
        let (codes, edges, n_bins) = bin_matrix::<u8>(x, n, p, cfg.max_bins)?;
        assemble(&codes, n, p, edges, n_bins, order, y, cfg)
    } else {
        let (codes, edges, n_bins) = bin_matrix::<u16>(x, n, p, cfg.max_bins)?;
        assemble(&codes, n, p, edges, n_bins, order, y, cfg)
    }
}

fn fit_codes<T: BinCode>(
    codes: &[T],
    n: usize,
    p: usize,
    edges: Vec<Vec<f64>>,
    y: &[f64],
    cfg: &TraceConfig,
) -> Result<TraceModel> {
    cfg.validate()?;
    if codes.len() != n * p || y.len() != n || edges.len() != p {
        return Err(TraceError::Shape("codes, edges and y disagree".into()));
    }
    prepare_y(y, n, p)?;
    let n_bins: Vec<usize> = edges.iter().map(|e| e.len() + 2).collect();
    // Pre-binned fits keep the caller's column order. Dense fits canonicalise.
    let order: Vec<usize> = (0..p).collect();
    assemble(codes, n, p, edges, n_bins, order, y, cfg)
}

/// Fit from columnar `u8` codes, shape `(p, n)`, and the edges that produced them.
pub fn fit_codes_u8(
    codes: &[u8],
    n: usize,
    p: usize,
    edges: Vec<Vec<f64>>,
    y: &[f64],
    cfg: &TraceConfig,
) -> Result<TraceModel> {
    fit_codes(codes, n, p, edges, y, cfg)
}

/// Fit from columnar `u16` codes, shape `(p, n)`.
pub fn fit_codes_u16(
    codes: &[u16],
    n: usize,
    p: usize,
    edges: Vec<Vec<f64>>,
    y: &[f64],
    cfg: &TraceConfig,
) -> Result<TraceModel> {
    fit_codes(codes, n, p, edges, y, cfg)
}

impl TraceModel {
    fn theta_from_codes(&self, codes: &[Vec<u16>]) -> Vec<f64> {
        let n = codes[0].len();
        let m_ch = self.n_channels;
        let mut theta = vec![0.0; n * m_ch];
        for &j in &self.order {
            let cf = &self.coef[j];
            for (i, row) in theta.chunks_mut(m_ch).enumerate() {
                let b = codes[j][i] as usize;
                for m in 0..m_ch {
                    row[m] += cf[b * m_ch + m];
                }
            }
        }
        for pair in &self.pairs {
            let nb_r = pair.n_bins_right;
            for (i, row) in theta.chunks_mut(m_ch).enumerate() {
                let bj = codes[pair.left][i] as usize;
                let bk = codes[pair.right][i] as usize;
                let base = (bj * nb_r + bk) * m_ch;
                for m in 0..m_ch {
                    row[m] += pair.coef[base + m];
                }
            }
        }
        theta
    }

    fn quantiles_from_theta(&self, theta: &[f64], project: bool, calibrated: bool) -> Vec<f64> {
        let m_ch = self.n_channels;
        let k = self.u.len();
        let n = theta.len() / m_ch;
        let apply_cal = calibrated && self.calibration.len() == k;
        let mut out = vec![0.0; n * k];
        for i in 0..n {
            let th = &theta[i * m_ch..(i + 1) * m_ch];
            let row = &mut out[i * k..(i + 1) * k];
            for kk in 0..k {
                let b = &self.basis[kk * m_ch..(kk + 1) * m_ch];
                let mut q = self.q0[kk];
                for m in 0..m_ch {
                    q += th[m] * b[m];
                }
                if apply_cal {
                    q += self.calibration[kk];
                }
                row[kk] = q;
            }
            if project {
                let projected = pava(row, None, None);
                row.copy_from_slice(&projected);
            }
        }
        out
    }

    fn codes_of(&self, x: &[f64], n: usize) -> Result<Vec<Vec<u16>>> {
        let p = self.edges.len();
        if x.len() != n * p {
            return Err(TraceError::Shape("predict shape".into()));
        }
        let mut cols = Vec::with_capacity(p);
        let mut col = vec![0.0; n];
        for j in 0..p {
            for i in 0..n {
                col[i] = x[i * p + j];
            }
            cols.push(bin_column::<u16>(&col, &self.edges[j])?);
        }
        Ok(cols)
    }

    /// Predicted quantile curves, row-major `(n, K)`. `project` applies PAVA in `u`.
    pub fn predict(&self, x: &[f64], n: usize, project: bool) -> Result<Vec<f64>> {
        self.predict_with(x, n, project, false)
    }

    /// Same as [`predict`](Self::predict), adding split-conformal offsets before PAVA.
    pub fn predict_calibrated(&self, x: &[f64], n: usize, project: bool) -> Result<Vec<f64>> {
        self.predict_with(x, n, project, true)
    }

    fn predict_with(
        &self,
        x: &[f64],
        n: usize,
        project: bool,
        calibrated: bool,
    ) -> Result<Vec<f64>> {
        let codes = self.codes_of(x, n)?;
        let theta = self.theta_from_codes(&codes);
        Ok(self.quantiles_from_theta(&theta, project, calibrated))
    }

    /// Per-level split-conformal offsets from a held-out calibration sample.
    ///
    /// Residuals are `y - Q(u|x)` on the projected raw curve. The offset at level
    /// `u` is [`conformal_quantile`] of those residuals. Prediction adds the offsets
    /// and runs PAVA again, so the calibrated curve stays monotone.
    pub fn calibrate(&mut self, x: &[f64], n: usize, y: &[f64]) -> Result<()> {
        if y.len() != n {
            return Err(TraceError::Shape("calibration shape".into()));
        }
        if y.iter().any(|v| !v.is_finite()) {
            return Err(TraceError::InvalidArgument(
                "calibration y must be finite".into(),
            ));
        }
        self.calibration.clear();
        let q = self.predict(x, n, true)?;
        let k = self.u.len();
        let mut resid = vec![0.0; n];
        let mut delta = vec![0.0; k];
        for kk in 0..k {
            for i in 0..n {
                resid[i] = y[i] - q[i * k + kk];
            }
            delta[kk] = conformal_quantile(&resid, self.u[kk])?;
        }
        self.calibration = delta;
        Ok(())
    }

    /// Increase in mean pinball when each column is shuffled on this sample.
    pub fn permutation_importance(
        &self,
        x: &[f64],
        n: usize,
        y: &[f64],
        repeats: usize,
        seed: u64,
    ) -> Result<Vec<f64>> {
        if y.len() != n || repeats == 0 {
            return Err(TraceError::InvalidArgument(
                "importance needs y aligned with X and repeats >= 1".into(),
            ));
        }
        let mut codes = self.codes_of(x, n)?;
        let base_q = self.quantiles_from_theta(&self.theta_from_codes(&codes), true, false);
        let base = pinball_scores(y, &base_q, &self.u)?.iter().sum::<f64>() / n as f64;
        let p = codes.len();
        let mut rng = SplitMix64(seed);
        let mut out = vec![0.0; p];
        for j in 0..p {
            let original = codes[j].clone();
            let mut acc = 0.0;
            for _ in 0..repeats {
                codes[j].copy_from_slice(&original);
                fisher_yates(&mut codes[j], &mut rng);
                let q = self.quantiles_from_theta(&self.theta_from_codes(&codes), true, false);
                let loss = pinball_scores(y, &q, &self.u)?.iter().sum::<f64>() / n as f64;
                acc += loss - base;
            }
            codes[j].copy_from_slice(&original);
            out[j] = acc / repeats as f64;
        }
        Ok(out)
    }

    /// Distributional scores of `predict(x, project = true)` against `y`.
    pub fn score(&self, x: &[f64], n: usize, y: &[f64]) -> Result<DistScore> {
        self.score_mode(x, n, y, false)
    }

    /// Scores of the calibrated curves. Empty calibration matches [`score`](Self::score).
    pub fn score_calibrated(&self, x: &[f64], n: usize, y: &[f64]) -> Result<DistScore> {
        self.score_mode(x, n, y, true)
    }

    fn score_mode(&self, x: &[f64], n: usize, y: &[f64], calibrated: bool) -> Result<DistScore> {
        if y.len() != n {
            return Err(TraceError::Shape("score shape".into()));
        }
        let q = self.predict_with(x, n, true, calibrated)?;
        score_quantiles(y, &q, &self.u)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimators::SplitMix64;
    use crate::trace::{empirical_quantile, inv_norm};

    fn cfg_small() -> TraceConfig {
        TraceConfig {
            n_channels: 2,
            max_bins: 4,
            learning_rate: 0.5,
            max_rounds: 15,
            patience: 0,
            val_frac: 0.0,
            u_levels: 5,
            max_interactions: 0,
            ..TraceConfig::default()
        }
    }

    #[test]
    fn pure_location_is_monotone_in_the_bin() {
        let n = 80;
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            x[i] = i as f64 / (n - 1) as f64;
            y[i] = 3.0 * x[i];
        }
        let model = fit_trace(&x, n, 1, &y, &cfg_small()).unwrap();
        let loc: Vec<f64> = model.coef[0]
            .chunks(model.n_channels)
            .skip(1)
            .map(|c| c[0])
            .collect();
        assert!(loc.len() >= 2);
        assert!(
            loc[loc.len() - 1] > loc[0],
            "location channel should rise with x, got {loc:?}"
        );
        let q = model.predict(&x, n, true).unwrap();
        let k = model.u.len();
        for row in q.chunks(k) {
            for w in row.windows(2) {
                assert!(w[1] + 1e-12 >= w[0]);
            }
        }
    }

    #[test]
    fn permuting_columns_does_not_change_predictions() {
        let n = 30;
        let p = 3;
        let mut x = vec![0.0; n * p];
        let mut y = vec![0.0; n];
        let mut s = 7u64;
        for i in 0..n * p {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            x[i] = (s >> 11) as f64 / (1u64 << 53) as f64;
        }
        for i in 0..n {
            y[i] = 0.8 * x[i * p] - 0.3 * x[i * p + 1] + 0.1 * x[i * p + 2];
        }
        let mut cfg = cfg_small();
        cfg.n_channels = 3;
        cfg.max_rounds = 6;
        cfg.learning_rate = 0.3;
        let a = fit_trace(&x, n, p, &y, &cfg).unwrap();
        let perm = [2usize, 0, 1];
        let mut xp = vec![0.0; n * p];
        for i in 0..n {
            for (t, &j) in perm.iter().enumerate() {
                xp[i * p + t] = x[i * p + j];
            }
        }
        let b = fit_trace(&xp, n, p, &y, &cfg).unwrap();
        let pa = a.predict(&x, n, true).unwrap();
        let pb = b.predict(&xp, n, true).unwrap();
        for (u, v) in pa.iter().zip(&pb) {
            assert!((u - v).abs() < 1e-8, "{} vs {}", u, v);
        }
    }

    fn unit(rng: &mut SplitMix64) -> f64 {
        ((rng.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn at_level(row: &[f64], u: &[f64], level: f64) -> f64 {
        if level <= u[0] {
            return row[0];
        }
        if level >= u[u.len() - 1] {
            return row[u.len() - 1];
        }
        let j = u.partition_point(|v| *v <= level) - 1;
        let t = (level - u[j]) / (u[j + 1] - u[j]);
        row[j] * (1.0 - t) + row[j + 1] * t
    }

    #[test]
    fn common_scale_correction_narrows_homoscedastic_bands() {
        let n = 500;
        let mut rng = SplitMix64(7);
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let x0 = unit(&mut rng);
            let z = inv_norm((unit(&mut rng) * 0.998 + 0.001).clamp(1e-4, 1.0 - 1e-4)).unwrap();
            x[i * 2] = x0;
            x[i * 2 + 1] = unit(&mut rng);
            y[i] = x0 + 0.08 * z;
        }
        let cfg = TraceConfig {
            max_rounds: 30,
            patience: 0,
            val_frac: 0.0,
            min_rounds: 0,
            learning_rate: 1.0,
            max_bins: 8,
            u_levels: 19,
            min_count: 1,
            ..TraceConfig::default()
        };
        let model = fit_trace(&x, n, 2, &y, &cfg).unwrap();
        let q = model.predict(&x, n, true).unwrap();
        let k = model.u.len();
        let mut spread = 0.0;
        for i in 0..n {
            let row = &q[i * k..(i + 1) * k];
            spread += at_level(row, &model.u, 0.9) - at_level(row, &model.u, 0.1);
        }
        spread /= n as f64;
        let marginal = {
            let q0 = empirical_quantile(&y, &[0.1, 0.9]).unwrap();
            q0[1] - q0[0]
        };
        let conditional = 2.0 * 1.2815515655446004 * 0.08;
        assert!(
            spread < 0.55 * marginal,
            "pred {spread} marginal {marginal} conditional {conditional}"
        );
        assert!(
            (spread - conditional).abs() < (marginal - conditional).abs(),
            "pred {spread} should sit closer to {conditional} than to {marginal}"
        );
    }

    #[test]
    fn out_of_bag_rows_partition_the_sample() {
        let cfg = TraceConfig {
            n_bags: 3,
            bag_frac: 0.6,
            val_frac: 0.0,
            ..TraceConfig::default()
        };
        for bag in 0..3 {
            let mask = bag_mask(30, &cfg, bag);
            let train = mask.iter().filter(|m| **m == 1).count();
            let val = mask.iter().filter(|m| **m == 2).count();
            assert_eq!(train + val, 30);
            assert!(val >= 1 && train >= 2);
            assert!(mask.iter().all(|m| *m == 1 || *m == 2));
        }
        let full = TraceConfig {
            n_bags: 1,
            bag_frac: 1.0,
            val_frac: 0.25,
            ..TraceConfig::default()
        };
        let mask = bag_mask(20, &full, 0);
        assert!(mask.iter().filter(|m| **m == 2).count() >= 1);
        assert!(mask.iter().all(|m| *m != 0));
    }

    #[test]
    fn screen_keeps_every_feature_before_min_rounds() {
        let n = 40;
        let mut x = vec![0.0; n * 3];
        let mut y = vec![0.0; n];
        let mut rng = SplitMix64(3);
        for i in 0..n {
            for j in 0..3 {
                x[i * 3 + j] = unit(&mut rng);
            }
            y[i] = unit(&mut rng) - 0.5;
        }
        let cfg = TraceConfig {
            max_rounds: 3,
            min_rounds: 10,
            patience: 2,
            val_frac: 0.25,
            learning_rate: 0.5,
            max_bins: 4,
            u_levels: 7,
            n_bags: 1,
            max_interactions: 0,
            ..TraceConfig::default()
        };
        let model = fit_trace(&x, n, 3, &y, &cfg).unwrap();
        assert!(model.selected.iter().all(|s| *s));
        assert!(model.bag_rounds[0] < cfg.min_rounds);
    }

    #[test]
    fn shape_gain_is_larger_on_the_tail_feature() {
        let n = 360;
        let mut rng = SplitMix64(11);
        let mut x = vec![0.0; n * 3];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let x0 = unit(&mut rng);
            let z = inv_norm((unit(&mut rng) * 0.996 + 0.002).clamp(1e-3, 1.0 - 1e-3)).unwrap();
            let he3 = z * z * z - 3.0 * z;
            x[i * 3] = x0;
            x[i * 3 + 1] = unit(&mut rng);
            x[i * 3 + 2] = unit(&mut rng);
            y[i] = z + 0.22 * x0 * he3;
        }
        let cfg = TraceConfig {
            max_rounds: 40,
            min_rounds: 0,
            patience: 0,
            val_frac: 0.2,
            learning_rate: 0.8,
            max_bins: 6,
            u_levels: 31,
            min_count: 5,
            n_bags: 1,
            max_interactions: 0,
            ..TraceConfig::default()
        };
        let model = fit_trace(&x, n, 3, &y, &cfg).unwrap();
        assert!(
            model.shape_gains[0] > model.shape_gains[1]
                && model.shape_gains[0] > model.shape_gains[2],
            "shape gains {:?}",
            model.shape_gains
        );
        assert!(model.selected[0]);
    }

    #[test]
    fn purified_pairs_have_zero_margins() {
        let n = 90;
        let mut rng = SplitMix64(5);
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let a = unit(&mut rng);
            let b = unit(&mut rng);
            x[i * 2] = a;
            x[i * 2 + 1] = b;
            y[i] = 0.8 * a + 0.5 * b + 1.2 * a * b;
        }
        let cfg = TraceConfig {
            max_rounds: 6,
            interaction_rounds: 4,
            max_interactions: 1,
            patience: 0,
            val_frac: 0.0,
            min_rounds: 0,
            learning_rate: 0.6,
            max_bins: 4,
            u_levels: 7,
            min_count: 1,
            n_bags: 1,
            ..TraceConfig::default()
        };
        let model = fit_trace(&x, n, 2, &y, &cfg).unwrap();
        assert_eq!(model.pairs.len(), 1);
        let pair = &model.pairs[0];
        let mut col = vec![0.0; n];
        for i in 0..n {
            col[i] = x[i * 2 + pair.left];
        }
        let left =
            crate::scalable::binning::bin_column::<u16>(&col, &model.edges[pair.left]).unwrap();
        for i in 0..n {
            col[i] = x[i * 2 + pair.right];
        }
        let right =
            crate::scalable::binning::bin_column::<u16>(&col, &model.edges[pair.right]).unwrap();
        let nb_r = pair.n_bins_right;
        let m_ch = model.n_channels;
        let mut counts = vec![0u64; pair.n_bins_left * nb_r];
        for i in 0..n {
            counts[left[i] as usize * nb_r + right[i] as usize] += 1;
        }
        for m in 0..m_ch {
            let mut row_sum = vec![0.0; pair.n_bins_left];
            let mut row_w = vec![0.0; pair.n_bins_left];
            let mut col_sum = vec![0.0; nb_r];
            let mut col_w = vec![0.0; nb_r];
            for i in 0..pair.n_bins_left {
                for j in 0..nb_r {
                    let c = counts[i * nb_r + j] as f64;
                    let v = pair.coef[(i * nb_r + j) * m_ch + m];
                    row_sum[i] += v * c;
                    row_w[i] += c;
                    col_sum[j] += v * c;
                    col_w[j] += c;
                }
            }
            for i in 0..pair.n_bins_left {
                if row_w[i] > 0.0 {
                    let mean = row_sum[i] / row_w[i];
                    assert!(mean.abs() < 1e-8, "row {i} channel {m} mean {mean}");
                }
            }
            for j in 0..nb_r {
                if col_w[j] > 0.0 {
                    let mean = col_sum[j] / col_w[j];
                    assert!(mean.abs() < 1e-8, "col {j} channel {m} mean {mean}");
                }
            }
        }
    }

    #[test]
    fn calibration_offsets_are_conformal_residual_quantiles() {
        let n = 80;
        let mut rng = SplitMix64(9);
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let a = unit(&mut rng);
            x[i * 2] = a;
            x[i * 2 + 1] = unit(&mut rng);
            y[i] = 1.5 * a + 0.1 * (unit(&mut rng) - 0.5);
        }
        let cfg = TraceConfig {
            max_rounds: 8,
            patience: 0,
            val_frac: 0.0,
            min_rounds: 0,
            learning_rate: 0.8,
            max_bins: 4,
            u_levels: 7,
            n_bags: 1,
            max_interactions: 0,
            ..TraceConfig::default()
        };
        let mut model = fit_trace(&x[..40 * 2], 40, 2, &y[..40], &cfg).unwrap();
        let x_cal = &x[40 * 2..];
        let y_cal = &y[40..];
        let raw = model.predict(x_cal, 40, true).unwrap();
        model.calibrate(x_cal, 40, y_cal).unwrap();
        let k = model.u.len();
        assert_eq!(model.calibration.len(), k);
        for kk in 0..k {
            let mut resid = vec![0.0; 40];
            for i in 0..40 {
                resid[i] = y_cal[i] - raw[i * k + kk];
            }
            let expect = crate::trace::conformal_quantile(&resid, model.u[kk]).unwrap();
            assert!((model.calibration[kk] - expect).abs() < 1e-12);
        }
        let cal = model.predict_calibrated(x_cal, 40, true).unwrap();
        for row in cal.chunks(k) {
            for w in row.windows(2) {
                assert!(w[1] + 1e-12 >= w[0]);
            }
        }
    }
}
