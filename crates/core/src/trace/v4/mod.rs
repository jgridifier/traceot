#![allow(unused_imports)]
//! TRACE v4: mixture-law shift decomposition (M1–M5 building blocks).
//!
//! Theory: `docs/theory/V4.md`. Spec: `docs/theory/V4_IMPL_SPEC.md`.
//! Modules land incrementally; Python exposes `explain_v4` once the game + infer
//! path is wired end-to-end.

#![allow(dead_code)]
pub mod balance;
pub mod fit_l2;
pub mod game;
pub mod infer;
pub mod mixture;

pub use balance::{rake_bin_marginals, BalanceConfig};
pub use fit_l2::{sobolev_penalty, FitL2Config};
pub use game::{phi_budget, role_shapley, RolePieces};
pub use infer::{multiplier_replicate, split_product, symmetrised_coord_players};
pub use mixture::{mixture_from_conditionals, mixture_from_conditionals_weighted, mixture_identical_rows, MixtureConfig, MixtureLaw};
