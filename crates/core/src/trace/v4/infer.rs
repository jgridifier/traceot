#![allow(dead_code)]
//! Split-product unbiased W2², bin-level IFs, multiplier bootstrap (M3).
//!
//! Spec: V4_IMPL_SPEC `infer.rs`. Half-sample cross product; Rademacher multiplier
//! bootstrap (B=2000); studentized permutation null.

/// Split-product estimate of ‖D‖² from two independent half-sample vectors.
pub fn split_product(d1: &[f64], d2: &[f64]) -> f64 {
    assert_eq!(d1.len(), d2.len());
    d1.iter().zip(d2).map(|(a, b)| a * b).sum()
}

/// Symmetrised player pieces: ½ (d1_i d2_tot_form + swap); for coordinate games
/// this is ½ (d1_i d2_i + d2_i d1_i) = d1_i d2_i, summing to the cross product.
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
}
