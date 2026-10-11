//! EBM-style histogram binning to small integer codes.
//!
//! Code 0 = missing (NaN); value codes are `1 + #{edges <= x}`, so a feature with
//! `e` edges uses codes `0..=e+1`. `u8` codes allow up to 254 edges (255 value bins).

use super::{BinCode, CHUNK};
use crate::error::{Result, TraceError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Levels `k / max_bins`, k = 1..max_bins-1.
pub fn edge_levels(max_bins: usize) -> Result<Vec<f64>> {
    if !(2..=65535).contains(&max_bins) {
        return Err(TraceError::InvalidArgument(
            "max_bins must be in [2, 65535]".into(),
        ));
    }
    Ok((1..max_bins).map(|k| k as f64 / max_bins as f64).collect())
}

/// Sorted, de-duplicated, finite cut points from candidate quantiles.
pub fn edges_from_quantiles(q: &[f64]) -> Vec<f64> {
    let mut e: Vec<f64> = q.iter().copied().filter(|v| v.is_finite()).collect();
    e.sort_by(f64::total_cmp);
    e.dedup();
    e
}

/// Exact edges from sorted non-NaN data (order statistics at `floor(k n / B)`).
pub fn edges_from_sorted(x_sorted: &[f64], max_bins: usize) -> Result<Vec<f64>> {
    let levels = edge_levels(max_bins)?;
    let n = x_sorted.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let q: Vec<f64> = levels
        .iter()
        .map(|lv| x_sorted[((lv * n as f64).floor() as usize).min(n - 1)])
        .collect();
    Ok(edges_from_quantiles(&q))
}

#[inline]
fn code_of(x: f64, edges: &[f64]) -> usize {
    if x.is_nan() {
        0
    } else {
        1 + edges.partition_point(|&e| e <= x)
    }
}

/// Bin one column into `out` (same length). Errors if codes would overflow `T`.
pub fn bin_column_into<T: BinCode>(x: &[f64], edges: &[f64], out: &mut [T]) -> Result<()> {
    if x.len() != out.len() {
        return Err(TraceError::Shape("x and out lengths differ".into()));
    }
    if edges.len() + 1 > T::MAX {
        return Err(TraceError::InvalidArgument(format!(
            "{} edges do not fit the code type (max code {})",
            edges.len(),
            T::MAX
        )));
    }
    let work = |(xs, os): (&[f64], &mut [T])| {
        for (o, &v) in os.iter_mut().zip(xs) {
            *o = T::from_usize(code_of(v, edges));
        }
    };
    #[cfg(feature = "parallel")]
    x.par_chunks(CHUNK)
        .zip(out.par_chunks_mut(CHUNK))
        .for_each(work);
    #[cfg(not(feature = "parallel"))]
    x.chunks(CHUNK).zip(out.chunks_mut(CHUNK)).for_each(work);
    Ok(())
}

/// Bin one column, allocating the output.
pub fn bin_column<T: BinCode + Default>(x: &[f64], edges: &[f64]) -> Result<Vec<T>> {
    let mut out = vec![T::default(); x.len()];
    bin_column_into(x, edges, &mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes() {
        let edges = [1.0, 2.0];
        let c: Vec<u8> = bin_column(&[0.5, 1.0, 1.5, 2.0, 9.0, f64::NAN], &edges).unwrap();
        assert_eq!(c, vec![1, 2, 2, 3, 3, 0]);
        let too_many: Vec<f64> = (0..300).map(|i| i as f64).collect();
        assert!(bin_column::<u8>(&[0.0], &too_many).is_err());
        assert!(bin_column::<u16>(&[0.0], &too_many).is_ok());
    }

    #[test]
    fn exact_edges_dedup() {
        let x = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(edges_from_sorted(&x, 4).unwrap(), vec![0.0, 1.0, 3.0]);
        assert!(edge_levels(1).is_err());
    }
}
