#![allow(dead_code)]
//! Bin-marginal raking / entropy balancing and Riesz ridge weights (M4).
//!
//! Spec: V4_IMPL_SPEC `balance.rs`. Tests: D2 exactness; overlap diagnostics.

/// Placeholder until the Newton dual / IPF path is ported from the Python verify.
#[derive(Clone, Debug, Default)]
pub struct BalanceConfig {
    pub max_iter: usize,
    pub tol: f64,
}

impl BalanceConfig {
    pub fn new() -> Self {
        Self {
            max_iter: 80,
            tol: 1e-10,
        }
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
        let s: f64 = w.iter().sum();
        let scale = n as f64 / s;
        for wi in &mut w {
            *wi *= scale;
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raking_matches_target_bin_masses() {
        // 2 features, 3 bins, 6 rows
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
        // Target: feature 0 uniform 2 each; feature 1 masses [1,2,3]
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
}
