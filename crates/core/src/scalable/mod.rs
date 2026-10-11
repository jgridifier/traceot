//! Scalable primitives for the TRACE core (kept separate from the ADOPT module).
//!
//! Designed for millions of rows and ~100 features:
//! * [`sketch::QuantileSketch`] - deterministic, mergeable streaming quantiles;
//! * [`binning`] - EBM-style histogram binning to `u8`/`u16` codes, columnar layout;
//! * [`histogram`] - per-bin gradient sums in one O(n) pass per feature;
//! * [`pinball`] - pinball (quantile) subgradients projected onto u-basis channels,
//!   O(n K) time, O(n M) memory.
//!
//! No routine here is worse than O(n log n); the main path is O(n) per pass.
//! Parallelism (rayon) is behind the default `parallel` feature so that the crate
//! can still be compiled single-threaded (e.g. for WASM).
//!
//! Reference implementations: `tests/python/reference/scalable.py`.

pub mod binning;
pub mod histogram;
pub mod pinball;
pub mod sketch;

/// Rows per work chunk for chunk-parallel passes.
pub const CHUNK: usize = 1 << 16;

/// Integer bin code (`u8` or `u16`).
pub trait BinCode: Copy + Send + Sync + Into<usize> + 'static {
    /// Largest representable code.
    const MAX: usize;
    /// Narrowing conversion (caller guarantees `v <= MAX`).
    fn from_usize(v: usize) -> Self;
}

impl BinCode for u8 {
    const MAX: usize = u8::MAX as usize;
    #[inline]
    fn from_usize(v: usize) -> Self {
        v as u8
    }
}

impl BinCode for u16 {
    const MAX: usize = u16::MAX as usize;
    #[inline]
    fn from_usize(v: usize) -> Self {
        v as u16
    }
}
