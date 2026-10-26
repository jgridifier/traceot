//! Mixing-player reference-symmetric switch game (M1).
//!
//! Spec: V4_IMPL_SPEC `game.rs`. Budget sums ≤ 1e-9; swap antisymmetry; order
//! invariance; per-feature mix split by covariance weights. Path claim (M8/D4):
//! AS equals Φ iff all interactions vanish.

#![allow(dead_code)]
use crate::trace::shift_v2::shapley_two;

/// Two-player Shapley pieces for one (channel, feature) G-table, plus interaction.
#[derive(Clone, Debug)]
pub struct RolePieces {
    pub comp: f64,
    pub structure: f64,
    pub interaction: f64,
}

/// Order-free role split on a 2×2 G-table (rows s=A,B; cols t=A,B).
pub fn role_shapley(g_aa: f64, g_ba: f64, g_ab: f64, g_bb: f64) -> RolePieces {
    let tp = shapley_two(g_aa, g_ba, g_ab, g_bb);
    RolePieces {
        comp: tp.comp,
        structure: tp.structure,
        interaction: g_bb - g_ba - g_ab + g_aa,
    }
}

/// Φ_kjr = w_k * Δc_k * a_kjr for all channels/features; returns flat (K, p, 2).
pub fn phi_budget(
    w: &[f64],
    delta_c: &[f64],
    a_comp: &[f64],  // length K*p
    a_struct: &[f64],
) -> Vec<f64> {
    let k = w.len();
    let kp = a_comp.len();
    assert_eq!(kp, a_struct.len());
    assert_eq!(kp % k, 0);
    let p = kp / k;
    let mut out = vec![0.0; k * p * 2];
    for ki in 0..k {
        for j in 0..p {
            let idx = ki * p + j;
            out[idx * 2] = w[ki] * delta_c[ki] * a_comp[idx];
            out[idx * 2 + 1] = w[ki] * delta_c[ki] * a_struct[idx];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_sums_when_no_interaction() {
        let w = [1.0 / 3.0];
        let delta = [3.0];
        // I=0 table: G=[[0,1],[1,2]] → a_comp=1, a_struct=1, Δc from one feature = 2? 
        // Use two features sharing Δc=3 with a summing to 3.
        let a_c = [2.0];
        let a_s = [1.0];
        let phi = phi_budget(&w, &delta, &a_c, &a_s);
        let sum: f64 = phi.iter().sum();
        let target = w[0] * delta[0] * (a_c[0] + a_s[0]);
        assert!((sum - target).abs() < 1e-12);
        let rp = role_shapley(0.0, 1.0, 1.0, 2.0);
        assert!(rp.interaction.abs() < 1e-15);
        assert!((rp.comp + rp.structure - 2.0).abs() < 1e-12);
    }
}
