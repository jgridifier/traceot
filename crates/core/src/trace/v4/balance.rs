//! Bin-marginal raking / entropy balancing and Riesz ridge weights (M4).
//!
//! Spec: V4_IMPL_SPEC `balance.rs`. Entropy balancing = Newton on the dual of
//! min KL(w ‖ 1) s.t. bin-marginal moment constraints; IPF as fallback. Riesz
//! ridge = closed-form LS of the density ratio on a bin+pairwise dictionary.

#![allow(dead_code, clippy::too_many_arguments)]

use crate::error::{Result, TraceError};
use crate::linalg::solve_spd;

/// Knobs for raking / entropy balancing / Riesz.
#[derive(Clone, Debug)]
pub struct BalanceConfig {
    pub max_iter: usize,
    pub tol: f64,
    /// Ridge on the Newton Hessian / Riesz Gram (absolute, not scaled).
    pub ridge: f64,
    /// Cap on individual weights relative to mean (overlap trim).
    pub max_weight_ratio: f64,
}

impl Default for BalanceConfig {
    fn default() -> Self {
        Self {
            max_iter: 80,
            tol: 1e-10,
            ridge: 1e-8,
            max_weight_ratio: 50.0,
        }
    }
}

impl BalanceConfig {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Diagnostics from a balancing call.
#[derive(Clone, Debug)]
pub struct BalanceDiagnostics {
    pub max_weight: f64,
    pub median_weight: f64,
    pub ess: f64,
    pub iters: usize,
    pub method: &'static str,
    pub residual: f64,
}

/// Result of balancing: weights + diagnostics.
#[derive(Clone, Debug)]
pub struct BalanceResult {
    pub weights: Vec<f64>,
    pub diag: BalanceDiagnostics,
}

fn normalize_to_n(w: &mut [f64], n: usize) {
    let s: f64 = w.iter().sum();
    let scale = n as f64 / s.max(1e-18);
    for wi in w {
        *wi *= scale;
    }
}

fn trim_weights(w: &mut [f64], max_ratio: f64) {
    if !(max_ratio.is_finite() && max_ratio > 1.0) {
        return;
    }
    let mean = w.iter().sum::<f64>() / w.len().max(1) as f64;
    let cap = max_ratio * mean.max(1e-18);
    for wi in w.iter_mut() {
        if *wi > cap {
            *wi = cap;
        }
    }
}

fn weight_diagnostics(
    w: &[f64],
    iters: usize,
    method: &'static str,
    residual: f64,
) -> BalanceDiagnostics {
    let mut sorted = w.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];
    let max_w = w.iter().cloned().fold(0.0_f64, f64::max);
    let sum: f64 = w.iter().sum();
    let sum_sq: f64 = w.iter().map(|x| x * x).sum();
    let ess = if sum_sq > 0.0 {
        (sum * sum) / sum_sq
    } else {
        0.0
    };
    BalanceDiagnostics {
        max_weight: max_w,
        median_weight: median,
        ess,
        iters,
        method,
        residual,
    }
}

/// IPF / raking weights so each feature's bin masses match `target_marginals`.
///
/// `codes`: row-major (n × p) bin codes; `n_bins[j]` bins for feature j;
/// `target_marginals`: concatenated length-sum(n_bins) target counts (sum to n).
pub fn rake_bin_marginals(
    codes: &[u16],
    n: usize,
    p: usize,
    n_bins: &[usize],
    target_marginals: &[f64],
    cfg: &BalanceConfig,
) -> Vec<f64> {
    assert_eq!(codes.len(), n * p);
    let mut w = vec![1.0; n];
    for _ in 0..cfg.max_iter {
        let mut offset = 0usize;
        for j in 0..p {
            let bj = n_bins[j];
            let mut cur: Vec<f64> = vec![0.0; bj];
            for i in 0..n {
                let b = codes[i * p + j] as usize;
                cur[b] += w[i];
            }
            for b in 0..bj {
                let t = target_marginals[offset + b];
                let factor = t / cur[b].max(1e-18_f64);
                for i in 0..n {
                    if codes[i * p + j] as usize == b {
                        w[i] *= factor;
                    }
                }
            }
            offset += bj;
        }
        normalize_to_n(&mut w, n);
    }
    trim_weights(&mut w, cfg.max_weight_ratio);
    normalize_to_n(&mut w, n);
    w
}

/// Build the moment matrix Φ (n × m) of one-hot bin indicators across features.
/// Drops the last bin of each feature to avoid the sum-to-n singularity (m = sum(n_bins)-p).
fn bin_design(codes: &[u16], n: usize, p: usize, n_bins: &[usize]) -> (Vec<f64>, usize) {
    let mut m = 0usize;
    for &bj in n_bins {
        m += bj.saturating_sub(1);
    }
    let mut phi = vec![0.0; n * m];
    let mut col = 0usize;
    for j in 0..p {
        let bj = n_bins[j];
        let free = bj.saturating_sub(1);
        for b in 0..free {
            for i in 0..n {
                if codes[i * p + j] as usize == b {
                    phi[i * m + col] = 1.0;
                }
            }
            col += 1;
        }
    }
    (phi, m)
}

fn free_targets(target_marginals: &[f64], n_bins: &[usize]) -> Vec<f64> {
    let mut out = Vec::new();
    let mut offset = 0usize;
    for &bj in n_bins {
        for b in 0..bj.saturating_sub(1) {
            out.push(target_marginals[offset + b]);
        }
        offset += bj;
    }
    out
}

fn moment_residual(phi: &[f64], n: usize, m: usize, w: &[f64], target: &[f64]) -> (Vec<f64>, f64) {
    let mut cur = vec![0.0; m];
    for i in 0..n {
        let base = i * m;
        for j in 0..m {
            cur[j] += w[i] * phi[base + j];
        }
    }
    let mut r = vec![0.0; m];
    let mut max_abs = 0.0_f64;
    for j in 0..m {
        r[j] = cur[j] - target[j];
        max_abs = max_abs.max(r[j].abs());
    }
    (r, max_abs)
}

/// Entropy balancing: min_w KL(w ‖ 1) s.t. Φᵀ w = m_target, 1ᵀ w = n.
/// Newton on dual λ; falls back to IPF if Newton fails to converge.
pub fn entropy_balance(
    codes: &[u16],
    n: usize,
    p: usize,
    n_bins: &[usize],
    target_marginals: &[f64],
    cfg: &BalanceConfig,
) -> BalanceResult {
    assert_eq!(codes.len(), n * p);
    let (phi, m) = bin_design(codes, n, p, n_bins);
    if m == 0 {
        let w = vec![1.0; n];
        return BalanceResult {
            weights: w.clone(),
            diag: weight_diagnostics(&w, 0, "identity", 0.0),
        };
    }
    let target = free_targets(target_marginals, n_bins);
    assert_eq!(target.len(), m);

    let mut lambda = vec![0.0; m];
    let mut w = vec![1.0; n];
    let mut last_res = f64::INFINITY;
    let mut iters = 0usize;
    let mut ok = false;

    for it in 0..cfg.max_iter {
        iters = it + 1;
        // w_i = exp(λ · φ_i)
        for i in 0..n {
            let mut dot = 0.0;
            let base = i * m;
            for j in 0..m {
                dot += lambda[j] * phi[base + j];
            }
            // Soft clip for overflow
            w[i] = dot.clamp(-40.0, 40.0).exp();
        }
        normalize_to_n(&mut w, n);
        let (r, max_abs) = moment_residual(&phi, n, m, &w, &target);
        last_res = max_abs;
        if max_abs < cfg.tol {
            ok = true;
            break;
        }
        // Hessian H_jk = sum_i w_i φ_ij φ_ik  (+ ridge)
        let mut h = vec![0.0; m * m];
        for i in 0..n {
            let base = i * m;
            let wi = w[i];
            for j in 0..m {
                let pj = phi[base + j];
                if pj == 0.0 {
                    continue;
                }
                for k in j..m {
                    let pk = phi[base + k];
                    h[j * m + k] += wi * pj * pk;
                }
            }
        }
        for j in 0..m {
            for k in 0..j {
                h[j * m + k] = h[k * m + j];
            }
            h[j * m + j] += cfg.ridge;
        }
        match solve_spd(&h, m, &r) {
            Ok(step) => {
                // Damped Newton
                let mut alpha = 1.0;
                let mut accepted = false;
                for _ in 0..8 {
                    let mut trial = lambda.clone();
                    for j in 0..m {
                        trial[j] -= alpha * step[j];
                    }
                    let mut wt = vec![0.0; n];
                    for i in 0..n {
                        let mut dot = 0.0;
                        let base = i * m;
                        for j in 0..m {
                            dot += trial[j] * phi[base + j];
                        }
                        wt[i] = dot.clamp(-40.0, 40.0).exp();
                    }
                    normalize_to_n(&mut wt, n);
                    let (_, trial_res) = moment_residual(&phi, n, m, &wt, &target);
                    if trial_res < last_res * (1.0 - 1e-4 * alpha) || trial_res < cfg.tol {
                        lambda = trial;
                        w = wt;
                        last_res = trial_res;
                        accepted = true;
                        break;
                    }
                    alpha *= 0.5;
                }
                if !accepted {
                    // Take full step anyway and continue
                    for j in 0..m {
                        lambda[j] -= step[j];
                    }
                }
            }
            Err(_) => {
                ok = false;
                break;
            }
        }
        if last_res < cfg.tol {
            ok = true;
            break;
        }
    }

    if !ok {
        let w_ipf = rake_bin_marginals(codes, n, p, n_bins, target_marginals, cfg);
        let (_, res) = {
            let (phi2, m2) = bin_design(codes, n, p, n_bins);
            let t2 = free_targets(target_marginals, n_bins);
            moment_residual(&phi2, n, m2, &w_ipf, &t2)
        };
        return BalanceResult {
            weights: w_ipf.clone(),
            diag: weight_diagnostics(&w_ipf, iters, "ipf_fallback", res),
        };
    }

    trim_weights(&mut w, cfg.max_weight_ratio);
    normalize_to_n(&mut w, n);
    BalanceResult {
        weights: w.clone(),
        diag: weight_diagnostics(&w, iters, "entropy", last_res),
    }
}

/// Alias matching the public API name in the hand-off list.
pub fn ipf_bin_weights(
    codes: &[u16],
    n: usize,
    p: usize,
    n_bins: &[usize],
    target_marginals: &[f64],
    cfg: &BalanceConfig,
) -> Vec<f64> {
    rake_bin_marginals(codes, n, p, n_bins, target_marginals, cfg)
}

/// Apply already-computed bin weights (identity passthrough for API symmetry).
pub fn apply_bin_weights(weights: &[f64]) -> Vec<f64> {
    weights.to_vec()
}

/// Build a Riesz dictionary: bin indicators (all bins, all features) plus pairwise
/// products of the first `top_m` features' first-bin indicators (cheap proxy for
/// top-m feature pairs). Returns (n × d) row-major design.
pub fn riesz_dictionary(
    codes: &[u16],
    n: usize,
    p: usize,
    n_bins: &[usize],
    top_m: usize,
) -> (Vec<f64>, usize) {
    let mut d = 0usize;
    for &bj in n_bins {
        d += bj;
    }
    let m_pair = top_m.min(p);
    let n_pairs = m_pair * (m_pair.saturating_sub(1)) / 2;
    d += n_pairs;
    let mut phi = vec![0.0; n * d];
    // Bin indicators
    let mut col = 0usize;
    for j in 0..p {
        let bj = n_bins[j];
        for b in 0..bj {
            for i in 0..n {
                if codes[i * p + j] as usize == b {
                    phi[i * d + col] = 1.0;
                }
            }
            col += 1;
        }
    }
    // Pairwise products of "bin 0 present" for top_m features
    for a in 0..m_pair {
        for b in (a + 1)..m_pair {
            for i in 0..n {
                let ca = codes[i * p + a] as usize;
                let cb = codes[i * p + b] as usize;
                // product of one-hot for bin index 0 — use (code==0) flags
                let fa = if ca == 0 { 1.0 } else { 0.0 };
                let fb = if cb == 0 { 1.0 } else { 0.0 };
                phi[i * d + col] = fa * fb;
            }
            col += 1;
        }
    }
    assert_eq!(col, d);
    (phi, d)
}

/// Riesz ridge least squares for the density-ratio coefficients.
///
/// Solves min_β ‖Φβ − r‖² + λ‖β‖² where `r` is the target ratio hint
/// (typically 1 for balanced source, or n_t/n_s · 1_{period}). Returns
/// per-row weights w_i = max(Φβ, ε), renormalised to sum n.
pub fn riesz_ridge(
    codes: &[u16],
    n: usize,
    p: usize,
    n_bins: &[usize],
    target_ratio: &[f64],
    top_m: usize,
    cfg: &BalanceConfig,
) -> Result<BalanceResult> {
    assert_eq!(codes.len(), n * p);
    assert_eq!(target_ratio.len(), n);
    let (phi, d) = riesz_dictionary(codes, n, p, n_bins, top_m);
    if d == 0 {
        return Err(TraceError::InvalidArgument("empty Riesz dictionary".into()));
    }
    // Gram G = ΦᵀΦ + λI, rhs = Φᵀ r
    let mut g = vec![0.0; d * d];
    let mut rhs = vec![0.0; d];
    for i in 0..n {
        let base = i * d;
        let ri = target_ratio[i];
        for j in 0..d {
            let pj = phi[base + j];
            rhs[j] += pj * ri;
            for k in j..d {
                g[j * d + k] += pj * phi[base + k];
            }
        }
    }
    for j in 0..d {
        for k in 0..j {
            g[j * d + k] = g[k * d + j];
        }
        g[j * d + j] += cfg.ridge;
    }
    let beta = solve_spd(&g, d, &rhs)?;
    let mut w = vec![0.0; n];
    for i in 0..n {
        let mut pred = 0.0;
        let base = i * d;
        for j in 0..d {
            pred += phi[base + j] * beta[j];
        }
        w[i] = pred.max(1e-12);
    }
    normalize_to_n(&mut w, n);
    trim_weights(&mut w, cfg.max_weight_ratio);
    normalize_to_n(&mut w, n);
    Ok(BalanceResult {
        weights: w.clone(),
        diag: weight_diagnostics(&w, 1, "riesz_ridge", 0.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raking_matches_target_bin_masses() {
        let codes: Vec<u16> = vec![
            0, 0, //
            0, 1, //
            1, 0, //
            1, 2, //
            2, 1, //
            2, 2,
        ];
        let n = 6;
        let p = 2;
        let n_bins = [3usize, 3];
        let target = vec![2.0, 2.0, 2.0, 1.0, 2.0, 3.0];
        let w = rake_bin_marginals(&codes, n, p, &n_bins, &target, &BalanceConfig::new());
        let mut mass0 = [0.0; 3];
        let mut mass1 = [0.0; 3];
        for i in 0..n {
            mass0[codes[i * p] as usize] += w[i];
            mass1[codes[i * p + 1] as usize] += w[i];
        }
        for b in 0..3 {
            assert!((mass0[b] - target[b]).abs() < 1e-8, "f0 bin {b}");
            assert!((mass1[b] - target[3 + b]).abs() < 1e-8, "f1 bin {b}");
        }
    }

    #[test]
    fn entropy_matches_targets_d2() {
        // Same as D2: additive piecewise-constant should be matched exactly.
        let codes: Vec<u16> = vec![0, 0, 0, 1, 1, 0, 1, 1, 2, 0, 2, 1, 0, 2, 1, 2, 2, 2];
        // 9 rows, 2 features, 3 bins each
        let n = 9;
        let p = 2;
        let n_bins = [3usize, 3];
        let target = vec![3.0, 3.0, 3.0, 3.0, 3.0, 3.0];
        let cfg = BalanceConfig {
            max_iter: 100,
            tol: 1e-10,
            ridge: 1e-10,
            max_weight_ratio: 1e9,
        };
        let res = entropy_balance(&codes, n, p, &n_bins, &target, &cfg);
        assert!(
            res.diag.residual < 1e-8,
            "residual {} via {}",
            res.diag.residual,
            res.diag.method
        );
        // Weighted mean of any bin indicator matches target mass / n * n = target
        let mut mass0 = [0.0; 3];
        for i in 0..n {
            mass0[codes[i * p] as usize] += res.weights[i];
        }
        for b in 0..3 {
            assert!((mass0[b] - target[b]).abs() < 1e-8, "entropy f0 bin {b}");
        }
        assert!(res.diag.ess > 1.0);
        assert!(res.diag.max_weight.is_finite());
        assert!(res.diag.median_weight.is_finite());
    }

    #[test]
    fn riesz_ridge_recovers_constant_ratio() {
        let codes: Vec<u16> = vec![0, 0, 0, 1, 1, 0, 1, 1];
        let n = 4;
        let p = 2;
        let n_bins = [2usize, 2];
        let target = vec![1.0; n];
        let cfg = BalanceConfig::new();
        let res = riesz_ridge(&codes, n, p, &n_bins, &target, 2, &cfg).unwrap();
        let mean = res.weights.iter().sum::<f64>() / n as f64;
        assert!((mean - 1.0).abs() < 1e-6);
        for &wi in &res.weights {
            assert!((wi - 1.0).abs() < 0.5, "wi={wi}");
        }
    }
}
