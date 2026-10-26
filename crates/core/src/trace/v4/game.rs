//! Mixing-player reference-symmetric switch game (M1).
//!
//! Spec: V4_IMPL_SPEC `game.rs`. Budget sums ≤ 1e-9; swap antisymmetry; order
//! invariance; per-feature mix split by covariance weights. Path claim (M8/D4):
//! AS equals Φ iff all interactions vanish.

#![allow(dead_code, clippy::too_many_arguments)]

use crate::trace::shift_v2::{shapley_two, shapley_two_swapped};

/// Two-player Shapley pieces for one (channel, feature) G-table, plus interaction.
#[derive(Clone, Debug)]
pub struct RolePieces {
    pub comp: f64,
    pub structure: f64,
    pub interaction: f64,
}

/// One player's share in the mixing game.
#[derive(Clone, Debug, Default)]
pub struct PlayerShare {
    pub channel: usize,
    pub feature: usize,
    /// 0 = composition, 1 = structure, 2 = composition-mix, 3 = structure-mix.
    pub role: u8,
    pub value: f64,
}

/// Configuration for the mixing-player game.
#[derive(Clone, Debug)]
pub struct MixingGameConfig {
    /// Channel Parseval weights w_k (length K); default 1/(2k+1).
    pub channel_weights: Vec<f64>,
}

impl MixingGameConfig {
    pub fn from_k(k: usize) -> Self {
        let channel_weights: Vec<f64> = (0..k).map(|j| 1.0 / (2 * j + 1) as f64).collect();
        Self { channel_weights }
    }
}

/// Full game output: per-player shares + diagnostics.
#[derive(Clone, Debug)]
pub struct MixingGameResult {
    pub players: Vec<PlayerShare>,
    /// Sum of all player values (should equal model-marginal W₂²).
    pub budget: f64,
    /// Sum of absolute interaction terms across (k,j).
    pub interaction_mass: f64,
    /// Residual mixing mass not attributed by the covariance split (beyond 2nd order).
    pub mixing_residual: f64,
}

/// Order-free role split on a 2×2 G-table (rows s=A,B; cols t=A,B).
pub fn role_shapley(g_aa: f64, g_ba: f64, g_ab: f64, g_bb: f64) -> RolePieces {
    // shapley_two(A, Cp, C, B): Cp = covariates B structure A = g_ba; C = cov A struct B = g_ab
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
    a_comp: &[f64], // length K*p
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

/// Covariance-based mixing weights per feature: Cov(f_j, μ) / sum_j Cov.
/// Negative covs are floored at 0 for the share (residual reported separately).
pub fn cov_mix_shares(cov_fj_mu: &[f64]) -> (Vec<f64>, f64) {
    let p = cov_fj_mu.len();
    let mut pos = vec![0.0; p];
    let mut neg_mass = 0.0;
    for (j, &c) in cov_fj_mu.iter().enumerate() {
        if c >= 0.0 {
            pos[j] = c;
        } else {
            neg_mass += -c;
        }
    }
    let total: f64 = pos.iter().sum();
    if total <= 1e-18 {
        let eq = 1.0 / p.max(1) as f64;
        return (vec![eq; p], neg_mass);
    }
    for s in &mut pos {
        *s /= total;
    }
    (pos, neg_mass)
}

/// Split a scalar mixing budget across features by covariance shares, then
/// across composition/structure by the two-player Shapley on the mixing G-table.
///
/// `mix_g` is length 4: [M_aa, M_ba, M_ab, M_bb] channel-level mixing values
/// (or their projection ⟨D_M, ΔM⟩ pieces). Returns (comp_mix[j], struct_mix[j]).
pub fn split_mixing_by_cov(
    mix_g: [f64; 4],
    cov_fj_mu: &[f64],
) -> (Vec<f64>, Vec<f64>, f64 /* residual */) {
    let roles = role_shapley(mix_g[0], mix_g[1], mix_g[2], mix_g[3]);
    let total_mix = roles.comp + roles.structure;
    let (shares, neg) = cov_mix_shares(cov_fj_mu);
    let p = shares.len();
    let mut comp_mix = vec![0.0; p];
    let mut struct_mix = vec![0.0; p];
    // If total_mix ≈ 0, leave zeros
    if total_mix.abs() < 1e-18 {
        return (comp_mix, struct_mix, neg);
    }
    let c_frac = roles.comp / total_mix;
    let s_frac = roles.structure / total_mix;
    for j in 0..p {
        let piece = total_mix * shares[j];
        comp_mix[j] = piece * c_frac;
        struct_mix[j] = piece * s_frac;
    }
    (comp_mix, struct_mix, neg)
}

/// Play the full mixing-player game for one set of channel G-tables.
///
/// `g_bary[k][4]` = [G_aa, G_ba, G_ab, G_bb] channel coefficients (or W₂ pieces).
/// `g_mix[k][4]`  = mixing-term G-table on the same scale.
/// `a_comp`/`a_struct`: length K*p feature shares of the barycenter Δ within channel
///   (should sum to 1 per channel when I=0; used via phi_budget).
/// `cov_fj_mu`: length p, for mixing split.
/// `delta_c`: length K, channel coefficient gaps (Q_B − Q_A projected).
pub fn play_mixing_game(
    g_bary: &[[f64; 4]],
    g_mix: &[[f64; 4]],
    delta_c: &[f64],
    a_comp: &[f64],
    a_struct: &[f64],
    cov_fj_mu: &[f64],
    cfg: &MixingGameConfig,
) -> MixingGameResult {
    let k = cfg.channel_weights.len();
    assert_eq!(g_bary.len(), k);
    assert_eq!(g_mix.len(), k);
    assert_eq!(delta_c.len(), k);
    let p = cov_fj_mu.len();
    assert_eq!(a_comp.len(), k * p);
    assert_eq!(a_struct.len(), k * p);

    let phi = phi_budget(&cfg.channel_weights, delta_c, a_comp, a_struct);
    let mut players = Vec::with_capacity(k * p * 4);
    let mut interaction_mass = 0.0;
    let mut mixing_residual = 0.0;
    let mut budget = 0.0;

    for ki in 0..k {
        let roles = role_shapley(g_bary[ki][0], g_bary[ki][1], g_bary[ki][2], g_bary[ki][3]);
        interaction_mass += roles.interaction.abs() * cfg.channel_weights[ki];

        // Barycenter players from phi_budget
        for j in 0..p {
            let idx = ki * p + j;
            let vc = phi[idx * 2];
            let vs = phi[idx * 2 + 1];
            players.push(PlayerShare {
                channel: ki,
                feature: j,
                role: 0,
                value: vc,
            });
            players.push(PlayerShare {
                channel: ki,
                feature: j,
                role: 1,
                value: vs,
            });
            budget += vc + vs;
        }

        // Mixing players via cov split of ⟨D_M, ΔM⟩ ≈ w_k * (mix gap)
        let mix_roles = role_shapley(g_mix[ki][0], g_mix[ki][1], g_mix[ki][2], g_mix[ki][3]);
        let mix_total = (mix_roles.comp + mix_roles.structure) * cfg.channel_weights[ki];
        let scaled = [
            g_mix[ki][0] * cfg.channel_weights[ki],
            g_mix[ki][1] * cfg.channel_weights[ki],
            g_mix[ki][2] * cfg.channel_weights[ki],
            g_mix[ki][3] * cfg.channel_weights[ki],
        ];
        let (cm, sm, neg) = split_mixing_by_cov(scaled, cov_fj_mu);
        mixing_residual += neg;
        // If cov shares don't cover, distribute residual uniformly into reported residual
        let attributed: f64 = cm.iter().chain(sm.iter()).sum();
        if (attributed - mix_total).abs() > 1e-9 && mix_total.abs() > 1e-12 {
            mixing_residual += (mix_total - attributed).abs();
        }
        for j in 0..p {
            players.push(PlayerShare {
                channel: ki,
                feature: j,
                role: 2,
                value: cm[j],
            });
            players.push(PlayerShare {
                channel: ki,
                feature: j,
                role: 3,
                value: sm[j],
            });
            budget += cm[j] + sm[j];
        }
    }

    MixingGameResult {
        players,
        budget,
        interaction_mass,
        mixing_residual,
    }
}

/// Swap antisymmetry check: flipping A↔B negates all player values.
pub fn swap_negate(players: &[PlayerShare]) -> Vec<PlayerShare> {
    players
        .iter()
        .map(|p| PlayerShare {
            channel: p.channel,
            feature: p.feature,
            role: p.role,
            value: -p.value,
        })
        .collect()
}

/// Period-swapped G-table: [aa,ba,ab,bb] → [bb,ab,ba,aa].
pub fn swap_g_table(g: [f64; 4]) -> [f64; 4] {
    [g[3], g[2], g[1], g[0]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_sums_when_no_interaction() {
        let w = [1.0 / 3.0];
        let delta = [3.0];
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

    #[test]
    fn swap_antisymmetry_on_roles() {
        let g = [0.1, 0.4, 0.2, 0.9];
        let fwd = role_shapley(g[0], g[1], g[2], g[3]);
        let sg = swap_g_table(g);
        let bwd = role_shapley(sg[0], sg[1], sg[2], sg[3]);
        assert!((fwd.comp + bwd.comp).abs() < 1e-12);
        assert!((fwd.structure + bwd.structure).abs() < 1e-12);
    }

    #[test]
    fn order_invariance_matches_swapped_shapley() {
        let a = 0.3;
        let ba = -0.2;
        let ab = 0.7;
        let b = 0.1;
        let s = shapley_two(a, ba, ab, b);
        let t = shapley_two_swapped(a, ba, ab, b);
        assert!((s.comp - t.comp).abs() < 1e-15);
        assert!((s.structure - t.structure).abs() < 1e-15);
        let rp = role_shapley(a, ba, ab, b);
        assert!((rp.comp - s.comp).abs() < 1e-15);
        assert!((rp.structure - s.structure).abs() < 1e-15);
    }

    #[test]
    fn cov_split_sums_to_total_mix() {
        let mix_g = [0.0, 0.1, 0.2, 0.5]; // Δmix roles
        let cov = [1.0, 3.0, 0.0];
        let (cm, sm, _) = split_mixing_by_cov(mix_g, &cov);
        let roles = role_shapley(mix_g[0], mix_g[1], mix_g[2], mix_g[3]);
        let total = roles.comp + roles.structure;
        let attributed: f64 = cm.iter().chain(sm.iter()).sum();
        assert!((attributed - total).abs() < 1e-12);
        // shares 1:3:0
        assert!((cm[0] + sm[0] - 0.25 * total).abs() < 1e-12);
        assert!((cm[1] + sm[1] - 0.75 * total).abs() < 1e-12);
        assert!((cm[2] + sm[2]).abs() < 1e-12);
    }

    #[test]
    fn full_game_budget_within_tol() {
        let k = 2usize;
        let p = 2usize;
        let cfg = MixingGameConfig::from_k(k);
        // I=0 bary tables: G = [[0,1],[1,2]] style per channel
        let g_bary = [[0.0, 1.0, 1.0, 2.0], [0.0, 0.5, 0.5, 1.0]];
        let g_mix = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.1, 0.1, 0.2]];
        let delta_c = [2.0, 1.0];
        // feature shares of Δ within channel: equal
        let a_comp = vec![0.5, 0.5, 0.5, 0.5];
        let a_struct = vec![0.5, 0.5, 0.5, 0.5];
        let cov = vec![1.0, 1.0];
        let res = play_mixing_game(&g_bary, &g_mix, &delta_c, &a_comp, &a_struct, &cov, &cfg);
        // Bary Φ sum = sum_k w_k Δc_k * (sum a_c + sum a_s) = sum_k w_k Δc_k * 1
        // wait a_c sums to 1 per channel (0.5+0.5), a_s too → factor 2
        let mut expected_bary = 0.0;
        for ki in 0..k {
            expected_bary += cfg.channel_weights[ki] * delta_c[ki] * 2.0;
        }
        let mix_roles = role_shapley(0.0, 0.1, 0.1, 0.2);
        let expected_mix = (mix_roles.comp + mix_roles.structure) * cfg.channel_weights[1];
        assert!(
            (res.budget - expected_bary - expected_mix).abs() < 1e-9,
            "budget={} expected={}",
            res.budget,
            expected_bary + expected_mix
        );
        assert_eq!(res.players.len(), k * p * 4);
        assert!(res.interaction_mass < 1e-12); // I=0 tables
    }
}
