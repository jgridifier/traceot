//! TRACE: a binned additive model of a conditional quantile function.
//!
//! Unpublished. The model is a sum of per-feature (and a few pairwise) functions of
//! `(x, u)`, so it does not depend on feature order. A final isotonic projection
//! makes each predicted curve monotone in `u`. See `docs/TRACE_DESIGN.md`.

mod basis;
mod checks;
mod fit;

pub use basis::{column_order, empirical_quantile, inv_norm, quantile_basis};
pub use checks::{
    conformal_pvalues, duplicate_screen, empirical_w2, feature_w2, pearson, pinball_scores,
    placebo_screen, projection_blur, score_quantiles, target_proxy_screen, DistScore, DupFlag,
    PlaceboScreen, ProxyFlag, ProxyScreen,
};
pub use fit::{fit_codes_u16, fit_codes_u8, fit_trace, PairTerm, TraceConfig, TraceModel};
