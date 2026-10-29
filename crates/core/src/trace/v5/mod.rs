//! TRACE v5: the Eulerian mixture counterfactual (Direction A, PREREG_V5 §2).
//!
//! For periods `s, t` the mixture counterfactual CDF is
//! `F_[s,t](y) = E_{x~F_s} F_t(y | x)` (covariates from `s`, conditional law from `t`),
//! and `Q_[s,t] = F_[s,t]^{-1}` is its left-inverse quantile function. This is the
//! counterfactual distribution of Chernozhukov, Fernández-Val and Melly (2013).
//!
//! The module is self-contained and usable on its own:
//!
//! * [`mixture`] — mixture CDF on a grid (fast exact pair counting for a
//!   location + residual conditional, or generic per-row conditional quantiles),
//!   inversion to quantiles, exact L-moments of the resulting law.
//! * [`explain`] — end-to-end two-period decomposition: fit conditionals, build the
//!   four hybrids `Q_[A,A]`, `Q_[A,B]`, `Q_[B,A]`, `Q_[B,B]`, and split each L-moment
//!   change into an order-free (two-player Shapley) composition / response piece.
//!   Also reports the Lagrangian barycenter and the mixing residual `M = Q - Q̄`.

pub mod explain;
pub mod mixture;

pub use explain::{explain_v5, ExplainV5Config, ExplainV5Report, Hybrid};
pub use mixture::{
    lmoments_from_cdf_grid, mixture_cdf_location, mixture_cdf_quantile_rows, quantile_from_cdf,
    support_grid, MixtureLawV5,
};
