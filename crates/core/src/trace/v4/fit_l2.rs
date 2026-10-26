#![allow(dead_code)]
//! L² channel-0 + PIT pseudo-outcome channels with Sobolev ridge (M2).
//!
//! Spec: V4_IMPL_SPEC `fit_l2.rs`. Channel 0 by squared-loss boosting; k≥1 by
//! least squares on (2k+1) Y P*_k(Û); ridge λ k^{2s} across k.

/// Configuration for the L² / PIT fit path.
#[derive(Clone, Debug)]
pub struct FitL2Config {
    pub k_channels: usize,
    pub ridge_lambda: f64,
    pub sobolev_s: f64,
    pub pit_iters: usize,
}

impl Default for FitL2Config {
    fn default() -> Self {
        Self {
            k_channels: 6,
            ridge_lambda: 1e-4,
            sobolev_s: 1.0,
            pit_iters: 2,
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
}
