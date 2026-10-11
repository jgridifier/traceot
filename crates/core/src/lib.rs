//! # traceot-core
//!
//! Numerical core of `traceot`: 1-D optimal transport on quantile grids, local and
//! global Fréchet smoothers, and ADOPT transport backfitting (Song & Müller,
//! AISTATS 2026, PMLR v300, <https://proceedings.mlr.press/v300/song26c.html>).
//!
//! All numerical logic of traceot lives here; the Python package is a thin binding.
//! Every routine has a pure-numpy reference in `tests/python/reference/`; the
//! Python test-suite checks that both agree to a tight tolerance. The crate has no required
//! dependencies (rayon is an optional default feature) and no I/O, so that it can
//! later be compiled to WASM.
//!
//! Status: research, unvalidated.

// Index loops mirror the reference formulas; `!(h > 0.0)` deliberately rejects NaN.
#![allow(clippy::needless_range_loop, clippy::neg_cmp_op_on_partial_ord)]

pub mod additive;
pub mod adopt;
pub mod error;
pub mod estimators;
pub mod linalg;
pub mod logchol;
pub mod scalable;
pub mod smoother;
pub mod w2;

pub use error::{Result, TraceError};
pub use linalg::Mat;
pub use smoother::Kernel;
