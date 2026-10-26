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

pub use balance::{rake_bin_marginals, BalanceConfig};
pub use fit_l2::{sobolev_penalty, FitL2Config};
pub use game::{phi_budget, role_shapley, RolePieces};
pub use infer::{multiplier_replicate, split_product, symmetrised_coord_players};
pub use mixture::{
    barycenter_from_conditionals, cdf_from_q, interp_q, mixing_term, mixture_from_conditionals,
    mixture_from_conditionals_weighted, mixture_hybrid, mixture_identical_rows, pooled_y_range,
    subsample_indices, MixtureConfig, MixtureLaw,
};
