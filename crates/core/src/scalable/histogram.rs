//! Per-bin gradient histograms: one O(n) pass per feature.

use super::{BinCode, CHUNK};
use crate::error::{Result, TraceError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Histogram of one feature: `sums[b * M + m]` and `counts[b]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Histogram {
    pub n_bins: usize,
    pub n_channels: usize,
    pub sums: Vec<f64>,
    pub counts: Vec<u64>,
}

impl Histogram {
    fn zeros(n_bins: usize, n_channels: usize) -> Self {
        Histogram {
            n_bins,
            n_channels,
            sums: vec![0.0; n_bins * n_channels],
            counts: vec![0; n_bins],
        }
    }

    fn add(mut self, other: Histogram) -> Histogram {
        self.sums
            .iter_mut()
            .zip(&other.sums)
            .for_each(|(a, b)| *a += b);
        self.counts
            .iter_mut()
            .zip(&other.counts)
            .for_each(|(a, b)| *a += b);
        self
    }
}

fn accumulate<T: BinCode>(codes: &[T], grads: &[&[f64]], start: usize, h: &mut Histogram) {
    let m_ch = grads.len();
    for (r, &c) in codes.iter().enumerate() {
        let b: usize = c.into();
        h.counts[b] += 1;
        let base = b * m_ch;
        for (m, g) in grads.iter().enumerate() {
            h.sums[base + m] += g[start + r];
        }
    }
}

/// Histogram of one binned feature. `grads` are channel-major columns of length n.
/// Chunk-parallel over rows when the `parallel` feature is on.
pub fn histogram<T: BinCode>(codes: &[T], grads: &[&[f64]], n_bins: usize) -> Result<Histogram> {
    let n = codes.len();
    if grads.iter().any(|g| g.len() != n) {
        return Err(TraceError::Shape(
            "gradient columns must match the number of rows".into(),
        ));
    }
    if let Some(&mx) = codes.iter().max_by_key(|c| Into::<usize>::into(**c)) {
        if Into::<usize>::into(mx) >= n_bins {
            return Err(TraceError::InvalidArgument("bin code >= n_bins".into()));
        }
    }
    let m_ch = grads.len();
    let n_chunks = n.div_ceil(CHUNK).max(1);
    let one = |ci: usize| {
        let s = ci * CHUNK;
        let e = (s + CHUNK).min(n);
        let mut h = Histogram::zeros(n_bins, m_ch);
        accumulate(&codes[s..e], grads, s, &mut h);
        h
    };
    #[cfg(feature = "parallel")]
    let h = (0..n_chunks)
        .into_par_iter()
        .map(one)
        .reduce(|| Histogram::zeros(n_bins, m_ch), Histogram::add);
    #[cfg(not(feature = "parallel"))]
    let h = (0..n_chunks)
        .map(one)
        .fold(Histogram::zeros(n_bins, m_ch), Histogram::add);
    Ok(h)
}

/// Histograms of many binned features (columnar: one slice per feature), parallel over
/// features. Each feature is processed sequentially, so summation order is the
/// natural row order.
pub fn histograms<T: BinCode>(
    columns: &[&[T]],
    grads: &[&[f64]],
    n_bins: usize,
) -> Result<Vec<Histogram>> {
    let m_ch = grads.len();
    let one = |col: &&[T]| -> Result<Histogram> {
        if grads.iter().any(|g| g.len() != col.len()) {
            return Err(TraceError::Shape("gradient columns must match rows".into()));
        }
        let mut h = Histogram::zeros(n_bins, m_ch);
        if col.iter().any(|&c| Into::<usize>::into(c) >= n_bins) {
            return Err(TraceError::InvalidArgument("bin code >= n_bins".into()));
        }
        accumulate(col, grads, 0, &mut h);
        Ok(h)
    };
    #[cfg(feature = "parallel")]
    return columns.par_iter().map(one).collect();
    #[cfg(not(feature = "parallel"))]
    return columns.iter().map(one).collect();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_histogram() {
        let codes: Vec<u8> = vec![0, 1, 1, 2, 1];
        let g0 = [1.0, 2.0, 3.0, 4.0, 5.0];
        let g1 = [1.0, 1.0, 1.0, 1.0, 1.0];
        let h = histogram(&codes, &[&g0, &g1], 3).unwrap();
        assert_eq!(h.counts, vec![1, 3, 1]);
        assert_eq!(h.sums, vec![1.0, 1.0, 10.0, 3.0, 4.0, 1.0]);
        let hs = histograms(&[&codes[..], &codes[..]], &[&g0, &g1], 3).unwrap();
        assert_eq!(hs[1], h);
        assert!(histogram(&codes, &[&g0], 2).is_err());
    }

    #[test]
    fn chunked_equals_single_pass() {
        let n = 3 * CHUNK + 17;
        let codes: Vec<u16> = (0..n).map(|i| (i % 300) as u16).collect();
        let g: Vec<f64> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();
        let a = histogram(&codes, &[&g], 300).unwrap();
        let b = histograms(&[&codes[..]], &[&g], 300)
            .unwrap()
            .pop()
            .unwrap();
        assert_eq!(a.counts, b.counts);
        for (x, y) in a.sums.iter().zip(&b.sums) {
            assert!((x - y).abs() < 1e-9);
        }
    }
}
