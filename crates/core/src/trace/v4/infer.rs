//! Split-product unbiased W2², bin-level IFs, multiplier bootstrap (M3).
//!
//! Spec: V4_IMPL_SPEC `infer.rs`. Half-sample cross product; Rademacher multiplier
//! bootstrap (B=2000); studentized permutation null.

#![allow(dead_code, clippy::too_many_arguments)]

use crate::trace::shift::SplitMix64;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Inference knobs.
#[derive(Clone, Debug)]
pub struct InferConfig {
    /// Multiplier bootstrap replicates (prereg: 2000).
    pub n_boot: usize,
    /// Permutation draws for the studentized null.
    pub n_perm: usize,
    pub seed: u64,
    /// Two-sided nominal level for percentile / studentized bands.
    pub alpha: f64,
}

impl Default for InferConfig {
    fn default() -> Self {
        Self {
            n_boot: 2000,
            n_perm: 999,
            seed: 0,
            alpha: 0.05,
        }
    }
}

/// One bin-level influence contribution (pre-aggregated player piece).
#[derive(Clone, Debug)]
pub struct InfluenceRow {
    pub bin: usize,
    pub contrib: f64,
}

/// Inference output for one scalar contrast / player sum.
#[derive(Clone, Debug)]
pub struct InferResult {
    pub estimate: f64,
    pub se: f64,
    pub ci_lo: f64,
    pub ci_hi: f64,
    pub p_value: f64,
    pub n_boot: usize,
    pub n_perm: usize,
}

/// Full v4 inference bundle (estimate + bands + perm).
#[derive(Clone, Debug)]
pub struct V4Inference {
    pub point: InferResult,
    pub boot_draws: Vec<f64>,
    pub perm_stats: Vec<f64>,
}

/// Split-product estimate of ‖D‖² from two independent half-sample vectors.
pub fn split_product(d1: &[f64], d2: &[f64]) -> f64 {
    assert_eq!(d1.len(), d2.len());
    d1.iter().zip(d2).map(|(a, b)| a * b).sum()
}

/// Symmetrised player pieces: ½ (d1_i d2_i + d2_i d1_i) = d1_i d2_i.
pub fn symmetrised_coord_players(d1: &[f64], d2: &[f64]) -> Vec<f64> {
    assert_eq!(d1.len(), d2.len());
    d1.iter()
        .zip(d2)
        .map(|(a, b)| 0.5 * (a * b + b * a))
        .collect()
}

/// One Rademacher multiplier replicate of a sum of bin-level contributions.
pub fn multiplier_replicate(contrib: &[f64], signs: &[f64]) -> f64 {
    assert_eq!(contrib.len(), signs.len());
    contrib.iter().zip(signs).map(|(c, s)| c * s).sum()
}

/// Aggregate row-level scores into bin-level influence contributions.
pub fn bin_level_ifs(scores: &[f64], bin_ids: &[usize], n_bins: usize) -> Vec<f64> {
    assert_eq!(scores.len(), bin_ids.len());
    let mut out = vec![0.0; n_bins];
    for (&s, &b) in scores.iter().zip(bin_ids) {
        if b < n_bins {
            out[b] += s;
        }
    }
    out
}

fn rademacher_signs(n: usize, rng: &mut SplitMix64) -> Vec<f64> {
    (0..n)
        .map(|_| if rng.next_f64() < 0.5 { -1.0 } else { 1.0 })
        .collect()
}

/// Multiplier bootstrap of a sum of contributions; returns centred draws.
pub fn bootstrap_multiplier(contrib: &[f64], cfg: &InferConfig) -> Vec<f64> {
    let n = contrib.len();
    let b = cfg.n_boot;
    let seed = cfg.seed ^ 0xB007_0001;

    #[cfg(feature = "parallel")]
    let draws: Vec<f64> = (0..b)
        .into_par_iter()
        .map(|r| {
            let mut rng =
                SplitMix64::new(seed.wrapping_add((r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
            let signs = rademacher_signs(n, &mut rng);
            multiplier_replicate(contrib, &signs)
        })
        .collect();

    #[cfg(not(feature = "parallel"))]
    let draws: Vec<f64> = (0..b)
        .map(|r| {
            let mut rng =
                SplitMix64::new(seed.wrapping_add((r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
            let signs = rademacher_signs(n, &mut rng);
            multiplier_replicate(contrib, &signs)
        })
        .collect();

    draws
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let n = sorted.len();
    let idx = ((p * (n - 1) as f64).round() as usize).min(n - 1);
    sorted[idx]
}

fn sample_sd(v: &[f64]) -> f64 {
    let n = v.len();
    if n < 2 {
        return 0.0;
    }
    let mean = v.iter().sum::<f64>() / n as f64;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1) as f64;
    var.sqrt()
}

/// Studentized permutation test: shuffle labels within the score stream.
///
/// `scores_a` / `scores_b` are period-tagged contributions; the test statistic
/// is (mean_b − mean_a) / se_pooled, and permutations reshuffle the pooled
/// labels. Returns two-sided p-value and the perm null of studentized stats.
pub fn studentized_perm(scores_a: &[f64], scores_b: &[f64], cfg: &InferConfig) -> (f64, Vec<f64>) {
    let na = scores_a.len();
    let nb = scores_b.len();
    assert!(na > 1 && nb > 1);
    let mean_a = scores_a.iter().sum::<f64>() / na as f64;
    let mean_b = scores_b.iter().sum::<f64>() / nb as f64;
    let se_a = sample_sd(scores_a) / (na as f64).sqrt();
    let se_b = sample_sd(scores_b) / (nb as f64).sqrt();
    let se = (se_a * se_a + se_b * se_b).sqrt().max(1e-18);
    let obs = (mean_b - mean_a) / se;

    let pooled: Vec<f64> = scores_a.iter().chain(scores_b.iter()).copied().collect();
    let n = pooled.len();
    let seed = cfg.seed ^ 0xFE12_0001;

    #[cfg(feature = "parallel")]
    let perm_stats: Vec<f64> = (0..cfg.n_perm)
        .into_par_iter()
        .map(|r| {
            let mut rng =
                SplitMix64::new(seed.wrapping_add((r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
            let mut v = pooled.clone();
            // Fisher–Yates
            for i in (1..n).rev() {
                let j = rng.index(i + 1);
                v.swap(i, j);
            }
            let ma: f64 = v[..na].iter().sum::<f64>() / na as f64;
            let mb: f64 = v[na..].iter().sum::<f64>() / nb as f64;
            let sa = sample_sd(&v[..na]) / (na as f64).sqrt();
            let sb = sample_sd(&v[na..]) / (nb as f64).sqrt();
            let sp = (sa * sa + sb * sb).sqrt().max(1e-18);
            (mb - ma) / sp
        })
        .collect();

    #[cfg(not(feature = "parallel"))]
    let perm_stats: Vec<f64> = {
        let _ = &pooled;
        (0..cfg.n_perm)
            .map(|r| {
                let mut rng = SplitMix64::new(
                    seed.wrapping_add((r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                );
                let mut v = pooled.clone();
                for i in (1..n).rev() {
                    let j = rng.index(i + 1);
                    v.swap(i, j);
                }
                let ma: f64 = v[..na].iter().sum::<f64>() / na as f64;
                let mb: f64 = v[na..].iter().sum::<f64>() / nb as f64;
                let sa = sample_sd(&v[..na]) / (na as f64).sqrt();
                let sb = sample_sd(&v[na..]) / (nb as f64).sqrt();
                let sp = (sa * sa + sb * sb).sqrt().max(1e-18);
                (mb - ma) / sp
            })
            .collect()
    };

    let mut ge = 1usize; // include observed
    for &t in &perm_stats {
        if t.abs() >= obs.abs() {
            ge += 1;
        }
    }
    let p = ge as f64 / (cfg.n_perm + 1) as f64;
    (p, perm_stats)
}

/// End-to-end: point = sum(contrib), multiplier CI, optional studentized perm
/// when period-split scores are provided.
pub fn infer_from_contrib(
    contrib: &[f64],
    scores_a: Option<&[f64]>,
    scores_b: Option<&[f64]>,
    cfg: &InferConfig,
) -> V4Inference {
    let estimate = contrib.iter().sum::<f64>();
    let draws = bootstrap_multiplier(contrib, cfg);
    let se = sample_sd(&draws);
    let mut sorted = draws.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Centred percentile CI around the point estimate
    let lo_q = percentile(&sorted, cfg.alpha / 2.0);
    let hi_q = percentile(&sorted, 1.0 - cfg.alpha / 2.0);
    // draws are already centred at 0 under the multiplier null of mean 0,
    // so CI for the sum is estimate + quantiles of (draw − 0) but the
    // replicate is sign-flipped sum ≈ N(0, var); use estimate ±
    let ci_lo = estimate + lo_q;
    let ci_hi = estimate + hi_q;

    let (p_value, perm_stats) = match (scores_a, scores_b) {
        (Some(a), Some(b)) => studentized_perm(a, b, cfg),
        _ => (f64::NAN, Vec::new()),
    };

    V4Inference {
        point: InferResult {
            estimate,
            se,
            ci_lo,
            ci_hi,
            p_value,
            n_boot: cfg.n_boot,
            n_perm: if p_value.is_nan() { 0 } else { cfg.n_perm },
        },
        boot_draws: draws,
        perm_stats,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetrised_sums_to_cross() {
        let d1 = vec![0.2, -0.1, 0.5, 0.0];
        let d2 = vec![0.3, 0.4, -0.2, 0.1];
        let cross = split_product(&d1, &d2);
        let players = symmetrised_coord_players(&d1, &d2);
        let s: f64 = players.iter().sum();
        assert!((s - cross).abs() < 1e-12);
    }

    #[test]
    fn bin_ifs_aggregate() {
        let scores = [1.0, 2.0, 3.0, 4.0];
        let bins = [0usize, 1, 0, 1];
        let ifs = bin_level_ifs(&scores, &bins, 2);
        assert!((ifs[0] - 4.0).abs() < 1e-15);
        assert!((ifs[1] - 6.0).abs() < 1e-15);
    }

    #[test]
    fn multiplier_boot_centred_near_zero() {
        let contrib = vec![0.1, -0.05, 0.2, -0.1, 0.05, 0.0, -0.02, 0.07];
        let cfg = InferConfig {
            n_boot: 500,
            n_perm: 0,
            seed: 7,
            alpha: 0.05,
        };
        let draws = bootstrap_multiplier(&contrib, &cfg);
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        assert!(mean.abs() < 0.05, "mean={mean}");
        let se = sample_sd(&draws);
        assert!(se > 0.0);
    }

    #[test]
    fn studentized_perm_nullish_pvalue() {
        // Identical distributions → p should not be tiny
        let a: Vec<f64> = (0..40).map(|i| (i as f64) * 0.01).collect();
        let b: Vec<f64> = (0..40).map(|i| (i as f64) * 0.01 + 0.001).collect();
        let cfg = InferConfig {
            n_boot: 0,
            n_perm: 199,
            seed: 3,
            alpha: 0.05,
        };
        let (p, stats) = studentized_perm(&a, &b, &cfg);
        assert_eq!(stats.len(), 199);
        assert!(p > 0.01, "p={p}");
    }

    #[test]
    fn infer_from_contrib_ci_covers_point() {
        let contrib = vec![1.0; 20];
        let cfg = InferConfig {
            n_boot: 400,
            n_perm: 0,
            seed: 1,
            alpha: 0.05,
        };
        let out = infer_from_contrib(&contrib, None, None, &cfg);
        assert!((out.point.estimate - 20.0).abs() < 1e-12);
        assert!(out.point.ci_lo < out.point.estimate);
        assert!(out.point.ci_hi > out.point.estimate);
        assert!(out.point.p_value.is_nan());
    }
}
