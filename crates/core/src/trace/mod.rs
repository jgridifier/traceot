//! TRACE: a binned additive model of a conditional quantile function.
//!
//! Unpublished. The model is a sum of per-feature (and a few pairwise) functions of
//! `(x, u)`, so it does not depend on feature order. A final isotonic projection
//! makes each predicted curve monotone in `u`. See `docs/TRACE_DESIGN.md`.

mod basis;
mod checks;
mod dro;
mod fit;
mod frechet;
mod hybrid;
mod legendre;
mod meanboost;
mod shift;
mod shift_v2;
mod v1fit;
mod v4;

pub use basis::{
    column_order, conformal_quantile, empirical_quantile, inv_norm, pinball_quantile_index,
    quantile_basis,
};
pub use checks::{
    conformal_pvalues, crps_quadrature, duplicate_screen, empirical_w2, feature_w2, pearson,
    pinball_scores, placebo_screen, projection_blur, score_quantiles, target_proxy_screen,
    DistScore, DupFlag, PlaceboScreen, ProxyFlag, ProxyScreen,
};
pub use dro::{
    codes_for_robust, coupling_w1, fit_robust, integrated_pinball_on, max_jump_bound,
    outcome_only_w1_gap, pinball, predict_robust, robust_risk_breakpoints, robust_risk_level,
    winf_risk, Ball, RobustConfig, RobustModel,
};
pub use fit::{fit_codes_u16, fit_codes_u8, fit_trace, PairTerm, TraceConfig, TraceModel};
pub use frechet::{fit_frechet, predict_frechet, weighted_slope, FrechetConfig, FrechetModel};
pub use hybrid::{
    fit_hybrid, hybrid_channel_energy, predict_hybrid, predict_mean_channel, HybridModel,
};
pub use legendre::{
    a_to_c, a_to_lambda, bernstein_nonnegative, derivative_bernstein, eval_a, eval_a_deriv,
    gauss_legendre_01, hermite_he, hosking_lmoments, k3_monotone, k3_non_representable,
    lambda_to_a, min_derivative_grid, pstar_all, pstar_coef, report_matrix, tau3_from_a,
    weighted_pav, wnorm, GaussLegendre, QuadBasis,
};
pub use meanboost::{fit_mean, predict_mean, MeanConfig, MeanModel};
pub use shift::{
    bootstrap_indices, cross_fit_energy, lmoment_channel_energy, permutation_pvalue,
    shift_conditional, shift_marginal, ConditionalShift, MarginalShift, SplitMix64,
};
pub use shift_v2::{
    barycenter_view, bootstrap_barycenter, bootstrap_entry_exit, bootstrap_marginal,
    entry_exit_from_models, entry_exit_split, explain_shift, hk_mass, marginal_from_laws,
    marginal_view, mean_curve, percentile_bands, permutation_marginal, permutation_w2_floor,
    quadratic_shapley, sample_lambda, shapley_two, shift_core_version, shift_decompose,
    w2_from_lambda, BaryDraw, BaryView, DecomposeConfig, EntryExit, MarginalDraw, MarginalView,
    PercentileBand, PermutationFloor, ShiftDecompose, ShiftModelHandle, ShiftReport, ShiftV2Config,
    TwoPlayer,
};
pub use v1fit::{
    attribute, channel_energy, channel_slope, codes_of, feature_pinball_gains,
    feature_report_table, fit_v1, leave_feature_gains, monotonicity_report, pinball_gh,
    player_coefficients, predict_quantiles, projection_pair, Attribution, FeatureGains, MonoMode,
    MonoReport, ProjectionPair, ReportKind, V1Config, V1Model,
};
pub use v4::{
    mixture_from_conditionals, mixture_from_conditionals_weighted, mixture_identical_rows,
    phi_budget, rake_bin_marginals, role_shapley, split_product, symmetrised_coord_players,
    BalanceConfig, FitL2Config, MixtureConfig, MixtureLaw, RolePieces,
};
