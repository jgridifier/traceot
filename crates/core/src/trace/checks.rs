//! Leakage screens, conformal oddness and distributional scores.
//!
//! Everything here is deterministic given its inputs. Pearson correlations use a
//! two-pass sum in input order. One-dimensional Wasserstein distances are the L2
//! distance between empirical quantile functions on a shared midpoint grid.

use crate::error::{Result, TraceError};
use crate::estimators::SplitMix64;
use crate::trace::basis::empirical_quantile;
use crate::w2::{pava, quantile_grid};

/// A feature whose association with `y` exceeds the screen threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct ProxyFlag {
    pub feature: usize,
    /// `0`: same row. `+s`: `corr(x[:-s], y[s:])` (the column leads `y`, as a future
    /// copy would). `-s`: `corr(x[s:], y[:-s])`.
    pub shift: i32,
    pub corr: f64,
}

/// Result of the target-proxy screen.
#[derive(Debug, Clone)]
pub struct ProxyScreen {
    /// Contemporaneous Pearson correlation of each column with `y` (`NaN` if undefined).
    pub contemporaneous: Vec<f64>,
    pub flags: Vec<ProxyFlag>,
}

/// A pair of columns that match exactly or are nearly collinear.
#[derive(Debug, Clone, PartialEq)]
pub struct DupFlag {
    pub left: usize,
    pub right: usize,
    pub exact: bool,
    pub corr: f64,
}

/// Placebo-target summary. `placebo_max_abs_corr` is the largest feature correlation
/// seen on any shuffled copy of `y`.
#[derive(Debug, Clone)]
pub struct PlaceboScreen {
    pub real_abs_corr: Vec<f64>,
    pub placebo_max_abs_corr: f64,
    pub placebo_mean_max_abs_corr: f64,
}

/// Scores of a predictive quantile matrix against observations.
#[derive(Debug, Clone, Copy)]
pub struct DistScore {
    /// Mean pinball loss on the supplied grid.
    pub pinball: f64,
    /// `2 * pinball`, the midpoint-rule approximation to CRPS via
    /// `CRPS = ∫_0^1 2 ρ_u du`.
    pub crps: f64,
    /// Fraction of `y` inside `[Q(0.1), Q(0.9)]`.
    pub cover80: f64,
    /// Fraction of `y` inside `[Q(0.05), Q(0.95)]`.
    pub cover90: f64,
    /// Mean squared error of the interpolated median `Q(0.5)` against `y`.
    pub mse_median: f64,
    /// Mean squared error of the grid-mean of `Q` (a midpoint integral) against `y`.
    pub mse_mean: f64,
}

/// Two-pass Pearson correlation. Non-finite pairs are dropped. `NaN` if fewer than
/// two finite pairs or if either variable has zero variance.
pub fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    let mut count = 0usize;
    let mut sx = 0.0;
    let mut sy = 0.0;
    for i in 0..n {
        if x[i].is_finite() && y[i].is_finite() {
            sx += x[i];
            sy += y[i];
            count += 1;
        }
    }
    if count < 2 {
        return f64::NAN;
    }
    let mx = sx / count as f64;
    let my = sy / count as f64;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    for i in 0..n {
        if x[i].is_finite() && y[i].is_finite() {
            let dx = x[i] - mx;
            let dy = y[i] - my;
            sxx += dx * dx;
            syy += dy * dy;
            sxy += dx * dy;
        }
    }
    if sxx <= 0.0 || syy <= 0.0 {
        f64::NAN
    } else {
        sxy / (sxx.sqrt() * syy.sqrt())
    }
}

fn column(x: &[f64], n: usize, p: usize, j: usize) -> Vec<f64> {
    (0..n).map(|i| x[i * p + j]).collect()
}

/// Flag columns that track `y` itself or a lead/lag of `y` by at most `max_shift` rows.
pub fn target_proxy_screen(
    x: &[f64],
    n: usize,
    p: usize,
    y: &[f64],
    threshold: f64,
    max_shift: usize,
) -> Result<ProxyScreen> {
    if x.len() != n * p || y.len() != n {
        return Err(TraceError::Shape("proxy screen shape".into()));
    }
    if !(threshold > 0.0 && threshold <= 1.0) {
        return Err(TraceError::InvalidArgument(
            "threshold must be in (0, 1]".into(),
        ));
    }
    let mut contemporaneous = Vec::with_capacity(p);
    let mut flags = Vec::new();
    for j in 0..p {
        let col = column(x, n, p, j);
        let c0 = pearson(&col, y);
        contemporaneous.push(c0);
        if c0.is_finite() && c0.abs() >= threshold {
            flags.push(ProxyFlag {
                feature: j,
                shift: 0,
                corr: c0,
            });
        }
        for s in 1..=max_shift {
            if s >= n {
                break;
            }
            let lead = pearson(&col[..n - s], &y[s..]);
            if lead.is_finite() && lead.abs() >= threshold {
                flags.push(ProxyFlag {
                    feature: j,
                    shift: s as i32,
                    corr: lead,
                });
            }
            let lag = pearson(&col[s..], &y[..n - s]);
            if lag.is_finite() && lag.abs() >= threshold {
                flags.push(ProxyFlag {
                    feature: j,
                    shift: -(s as i32),
                    corr: lag,
                });
            }
        }
    }
    Ok(ProxyScreen {
        contemporaneous,
        flags,
    })
}

fn column_hash(col: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &v in col {
        h ^= v.to_bits();
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn columns_equal(x: &[f64], n: usize, p: usize, a: usize, b: usize) -> bool {
    (0..n).all(|i| x[i * p + a].to_bits() == x[i * p + b].to_bits())
}

/// Exact copies (identical bits) and pairs with `|corr| >= near`.
pub fn duplicate_screen(x: &[f64], n: usize, p: usize, near: f64) -> Result<Vec<DupFlag>> {
    if x.len() != n * p {
        return Err(TraceError::Shape("duplicate screen shape".into()));
    }
    if !(near > 0.0 && near <= 1.0) {
        return Err(TraceError::InvalidArgument(
            "near threshold must be in (0, 1]".into(),
        ));
    }
    let mut hashes = Vec::with_capacity(p);
    let mut cols = Vec::with_capacity(p);
    for j in 0..p {
        let col = column(x, n, p, j);
        hashes.push(column_hash(&col));
        cols.push(col);
    }
    let mut flags = Vec::new();
    for i in 0..p {
        for j in (i + 1)..p {
            let exact = hashes[i] == hashes[j] && columns_equal(x, n, p, i, j);
            let corr = pearson(&cols[i], &cols[j]);
            if exact || (corr.is_finite() && corr.abs() >= near) {
                flags.push(DupFlag {
                    left: i,
                    right: j,
                    exact,
                    corr,
                });
            }
        }
    }
    Ok(flags)
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

/// Shuffle `y` with SplitMix64(`seed`) and record how large a feature correlation gets.
/// Each permutation is a fresh Fisher–Yates shuffle of the previous one (still uniform).
pub fn placebo_screen(
    x: &[f64],
    n: usize,
    p: usize,
    y: &[f64],
    n_perm: usize,
    seed: u64,
) -> Result<PlaceboScreen> {
    if x.len() != n * p || y.len() != n || n_perm == 0 {
        return Err(TraceError::InvalidArgument(
            "placebo screen needs data and n_perm >= 1".into(),
        ));
    }
    let cols: Vec<Vec<f64>> = (0..p).map(|j| column(x, n, p, j)).collect();
    let real_abs_corr: Vec<f64> = cols.iter().map(|c| pearson(c, y).abs()).collect();
    let mut rng = SplitMix64(seed);
    let mut shuffled = y.to_vec();
    let mut placebo_max = 0.0;
    let mut placebo_sum = 0.0;
    for _ in 0..n_perm {
        fisher_yates(&mut shuffled, &mut rng);
        let mut m = 0.0;
        for c in &cols {
            let a = pearson(c, &shuffled).abs();
            if a.is_finite() && a > m {
                m = a;
            }
        }
        placebo_sum += m;
        if m > placebo_max {
            placebo_max = m;
        }
    }
    Ok(PlaceboScreen {
        real_abs_corr,
        placebo_max_abs_corr: placebo_max,
        placebo_mean_max_abs_corr: placebo_sum / n_perm as f64,
    })
}

/// Grid Wasserstein-2 distance between two empirical samples (square root of the mean
/// squared difference of their quantile functions). `NaN` if either sample has no
/// finite values.
pub fn empirical_w2(a: &[f64], b: &[f64], grid: usize) -> Result<f64> {
    if !(2..=4096).contains(&grid) {
        return Err(TraceError::InvalidArgument(
            "W2 grid must be in [2, 4096]".into(),
        ));
    }
    let aa: Vec<f64> = a.iter().copied().filter(|v| v.is_finite()).collect();
    let bb: Vec<f64> = b.iter().copied().filter(|v| v.is_finite()).collect();
    if aa.is_empty() || bb.is_empty() {
        return Ok(f64::NAN);
    }
    let u = quantile_grid(grid);
    let qa = empirical_quantile(&aa, &u)?;
    let qb = empirical_quantile(&bb, &u)?;
    let mut s = 0.0;
    for (x, y) in qa.iter().zip(&qb) {
        let d = x - y;
        s += d * d;
    }
    Ok((s / grid as f64).sqrt())
}

/// Per-column [`empirical_w2`] between two row-major matrices with the same `p`.
pub fn feature_w2(
    a: &[f64],
    n_a: usize,
    b: &[f64],
    n_b: usize,
    p: usize,
    grid: usize,
) -> Result<Vec<f64>> {
    if a.len() != n_a * p || b.len() != n_b * p {
        return Err(TraceError::Shape("feature_w2 shape".into()));
    }
    let mut out = Vec::with_capacity(p);
    for j in 0..p {
        let ca = column(a, n_a, p, j);
        let cb = column(b, n_b, p, j);
        out.push(empirical_w2(&ca, &cb, grid)?);
    }
    Ok(out)
}

/// Split-conformal p-values. `p = (1 + #{cal >= s}) / (n_cal + 1)`.
pub fn conformal_pvalues(cal: &[f64], test: &[f64]) -> Result<Vec<f64>> {
    if cal.is_empty() {
        return Err(TraceError::InvalidArgument(
            "calibration scores are empty".into(),
        ));
    }
    let mut sorted = cal.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len() as f64;
    Ok(test
        .iter()
        .map(|&v| {
            let below = sorted.partition_point(|&x| x < v);
            let ge = sorted.len() - below;
            (1.0 + ge as f64) / (n + 1.0)
        })
        .collect())
}

/// Per-row mean pinball loss of a row-major quantile matrix `q` with `k` columns.
pub fn pinball_scores(y: &[f64], q: &[f64], u: &[f64]) -> Result<Vec<f64>> {
    let k = u.len();
    let n = y.len();
    if k == 0 || q.len() != n * k {
        return Err(TraceError::Shape("pinball_scores shape".into()));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut s = 0.0;
        let row = &q[i * k..(i + 1) * k];
        for (kk, &ui) in u.iter().enumerate() {
            let res = y[i] - row[kk];
            let r = if res < 0.0 { ui - 1.0 } else { ui };
            s += res * r;
        }
        out.push(s / k as f64);
    }
    Ok(out)
}

fn quantile_at(row: &[f64], u: &[f64], level: f64) -> f64 {
    let k = u.len();
    if level <= u[0] {
        return row[0];
    }
    if level >= u[k - 1] {
        return row[k - 1];
    }
    let j = u.partition_point(|&v| v <= level) - 1;
    let t = (level - u[j]) / (u[j + 1] - u[j]);
    row[j] * (1.0 - t) + row[j + 1] * t
}

/// Pinball, CRPS approximation, interval coverage and squared errors.
pub fn score_quantiles(y: &[f64], q: &[f64], u: &[f64]) -> Result<DistScore> {
    let scores = pinball_scores(y, q, u)?;
    let n = y.len();
    let k = u.len();
    let pinball = scores.iter().sum::<f64>() / n.max(1) as f64;
    let mut c80 = 0usize;
    let mut c90 = 0usize;
    let mut sse_med = 0.0;
    let mut sse_mean = 0.0;
    for i in 0..n {
        let row = &q[i * k..(i + 1) * k];
        let lo80 = quantile_at(row, u, 0.1);
        let hi80 = quantile_at(row, u, 0.9);
        let lo90 = quantile_at(row, u, 0.05);
        let hi90 = quantile_at(row, u, 0.95);
        if (lo80..=hi80).contains(&y[i]) {
            c80 += 1;
        }
        if (lo90..=hi90).contains(&y[i]) {
            c90 += 1;
        }
        let med = quantile_at(row, u, 0.5);
        let mean = row.iter().sum::<f64>() / k as f64;
        let dmed = med - y[i];
        let dmean = mean - y[i];
        sse_med += dmed * dmed;
        sse_mean += dmean * dmean;
    }
    let denom = n.max(1) as f64;
    Ok(DistScore {
        pinball,
        crps: 2.0 * pinball,
        cover80: c80 as f64 / denom,
        cover90: c90 as f64 / denom,
        mse_median: sse_med / denom,
        mse_mean: sse_mean / denom,
    })
}

/// Gauss-quadrature CRPS: `2 · mean_i Σ_t w_t ρ_{u_t}(y_i − q_{it})`.
/// `q` is row-major `n × k`. Weights are the quadrature weights, not uniform.
pub fn crps_quadrature(y: &[f64], q: &[f64], u: &[f64], w: &[f64]) -> Result<f64> {
    let k = u.len();
    let n = y.len();
    if k == 0 || w.len() != k || q.len() != n * k {
        return Err(TraceError::Shape("crps_quadrature shape".into()));
    }
    let mut acc = 0.0;
    for i in 0..n {
        let row = &q[i * k..(i + 1) * k];
        for t in 0..k {
            let r = y[i] - row[t];
            let rho = if r >= 0.0 { u[t] * r } else { (u[t] - 1.0) * r };
            acc += w[t] * rho;
        }
    }
    Ok(2.0 * acc / n.max(1) as f64)
}

fn contrast_rms(q00: &[f64], q01: &[f64], q10: &[f64], q11: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..q00.len() {
        let c = q11[i] - q10[i] - q01[i] + q00[i];
        s += c * c;
    }
    (s / q00.len() as f64).sqrt()
}

/// Root-mean-square interaction contrast of four quantile curves, before and after
/// a row-wise isotonic projection. This is the blurring introduced by the projection.
pub fn projection_blur(q00: &[f64], q01: &[f64], q10: &[f64], q11: &[f64]) -> Result<(f64, f64)> {
    let n = q00.len();
    if q01.len() != n || q10.len() != n || q11.len() != n || n == 0 {
        return Err(TraceError::Shape("projection_blur shape".into()));
    }
    let before = contrast_rms(q00, q01, q10, q11);
    let a00 = pava(q00, None, None);
    let a01 = pava(q01, None, None);
    let a10 = pava(q10, None, None);
    let a11 = pava(q11, None, None);
    let after = contrast_rms(&a00, &a01, &a10, &a11);
    Ok((before, after))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conformal_tail_probabilities() {
        let p = conformal_pvalues(&[1.0, 2.0, 3.0], &[0.0, 2.5, 4.0]).unwrap();
        assert!((p[0] - 1.0).abs() < 1e-15);
        assert!((p[1] - 0.5).abs() < 1e-15);
        assert!((p[2] - 0.25).abs() < 1e-15);
    }

    #[test]
    fn crps_of_a_standard_normal_at_zero() {
        // CRPS(N(0,1), 0) = (sqrt(2) - 1) / sqrt(pi).
        let closed = (2.0_f64.sqrt() - 1.0) / std::f64::consts::PI.sqrt();
        let (u, basis) = crate::trace::basis::quantile_basis(49, 2).unwrap();
        // basis column 1 is z / rms(z). Rebuild the raw quantile z = Phi^{-1}(u).
        let z: Vec<f64> = u
            .iter()
            .map(|&ui| crate::trace::basis::inv_norm(ui).unwrap())
            .collect();
        let score = score_quantiles(&[0.0], &z, &u).unwrap();
        assert!(
            (score.crps - closed).abs() < 2e-3,
            "{} vs {}",
            score.crps,
            closed
        );
        assert!((score.crps - 2.0 * score.pinball).abs() < 1e-15);
        let _ = basis;
    }

    #[test]
    fn proxy_and_exact_duplicate() {
        let y = [0.0, 1.0, 2.0, 3.0, 4.0];
        let mut x = vec![0.0; 5 * 3];
        for i in 0..5 {
            x[i * 3] = i as f64; // equals y
            x[i * 3 + 1] = i as f64; // duplicate of column 0
            x[i * 3 + 2] = [1.0, -0.4, 0.2, -0.1, 0.05][i];
        }
        let screen = target_proxy_screen(&x, 5, 3, &y, 0.99, 1).unwrap();
        assert!(screen.flags.iter().any(|f| f.feature == 0 && f.shift == 0));
        assert!(!screen.flags.iter().any(|f| f.feature == 2));
        let dups = duplicate_screen(&x, 5, 3, 0.999).unwrap();
        assert!(dups.iter().any(|d| d.left == 0 && d.right == 1 && d.exact));
    }
}
