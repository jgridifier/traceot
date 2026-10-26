//! TRACE v4: mixing law, balancing, game, L² fit, inference.
//!
//! Spec: `docs/theory/V4_IMPL_SPEC.md`. Theory docs and PREREG are frozen;
//! this module is the Rust core. Python bindings stay thin.

#![allow(dead_code)]
#![allow(unused_imports)]

pub mod balance;
pub mod fit_l2;
pub mod game;
pub mod infer;
pub mod mixture;

pub use balance::{
    apply_bin_weights, entropy_balance, ipf_bin_weights, rake_bin_marginals, riesz_dictionary,
    riesz_ridge, BalanceConfig, BalanceDiagnostics, BalanceResult,
};
pub use fit_l2::{
    fit_l2_channels, predict_channel, sobolev_penalty, ChannelFit, FitL2Config, FitL2Result,
};
pub use game::{
    cov_mix_shares, phi_budget, play_mixing_game, role_shapley, split_mixing_by_cov, swap_g_table,
    swap_negate, MixingGameConfig, MixingGameResult, PlayerShare, RolePieces,
};
pub use infer::{
    bin_level_ifs, bootstrap_multiplier, infer_from_contrib, multiplier_replicate, split_product,
    studentized_perm, symmetrised_coord_players, InferConfig, InferResult, InfluenceRow,
    V4Inference,
};
pub use mixture::{
    barycenter_from_conditionals, cdf_from_q, interp_q, mixing_term, mixture_from_conditionals,
    mixture_from_conditionals_weighted, mixture_hybrid, mixture_identical_rows, pooled_y_range,
    subsample_indices, MixtureConfig, MixtureLaw,
};
