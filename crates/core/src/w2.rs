//! 1-D Wasserstein-2 geometry on a common quantile grid.
//!
//! Mirrors `python/traceot/reference/w2.py`. A distribution is its quantile function
//! on a fixed grid; W2 is the L2 distance between quantile functions, the Fréchet
//! mean is the pointwise mean, and the optimal transport map `F2^{-1} o F1` is the
//! monotone map sending `Q1(u_k)` to `Q2(u_k)`.

/// Midpoint grid `u_k = (k - 1/2)/q`.
pub fn quantile_grid(q: usize) -> Vec<f64> {
    (0..q).map(|k| (k as f64 + 0.5) / q as f64).collect()
}

/// Squared W2 distance (mean over the grid) between two quantile vectors.
pub fn w2_sq(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let s: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    s / a.len() as f64
}

/// L2 projection onto non-decreasing vectors (pool-adjacent-violators, equal
/// weights), followed by optional clipping to `[lower, upper]`. Pooled blocks get an
/// identical value, so ties are exact.
pub fn pava(y: &[f64], lower: Option<f64>, upper: Option<f64>) -> Vec<f64> {
    let mut sums: Vec<f64> = Vec::with_capacity(y.len());
    let mut counts: Vec<usize> = Vec::with_capacity(y.len());
    for &v in y {
        let mut s = v;
        let mut c = 1usize;
        while let (Some(&ls), Some(&lc)) = (sums.last(), counts.last()) {
            if ls / lc as f64 > s / c as f64 {
                s += ls;
                c += lc;
                sums.pop();
                counts.pop();
            } else {
                break;
            }
        }
        sums.push(s);
        counts.push(c);
    }
    let mut out = Vec::with_capacity(y.len());
    for (s, c) in sums.iter().zip(&counts) {
        let v = s / *c as f64;
        out.extend(std::iter::repeat_n(v, *c));
    }
    if let Some(lo) = lower {
        out.iter_mut().for_each(|v| *v = v.max(lo));
    }
    if let Some(hi) = upper {
        out.iter_mut().for_each(|v| *v = v.min(hi));
    }
    out
}

/// Merge runs of exactly equal source knots `a`, averaging the targets `b`.
pub fn collapse_knots(a: &[f64], b: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = a.len();
    let mut au = Vec::with_capacity(n);
    let mut bu = Vec::with_capacity(n);
    let mut k = 0;
    while k < n {
        let mut m = k + 1;
        let mut s = b[k];
        while m < n && a[m] == a[k] {
            s += b[m];
            m += 1;
        }
        au.push(a[k]);
        bu.push(s / (m - k) as f64);
        k = m;
    }
    (au, bu)
}

/// A monotone transport map given by tie-collapsed knots `a -> b`.
///
/// Piecewise-linear between knots, extended outside `[a_1, a_q]` as a translation by
/// the boundary displacement.
#[derive(Debug, Clone)]
pub struct TransportMap {
    a: Vec<f64>,
    b: Vec<f64>,
}

impl TransportMap {
    /// Build from source and target quantile vectors (same grid).
    pub fn new(a: &[f64], b: &[f64]) -> Self {
        let (a, b) = collapse_knots(a, b);
        TransportMap { a, b }
    }

    /// Evaluate at one point (same arithmetic as `numpy.interp`).
    #[inline]
    pub fn eval(&self, y: f64) -> f64 {
        let a = &self.a;
        let b = &self.b;
        let n = a.len();
        if n == 1 {
            return y + (b[0] - a[0]);
        }
        if y < a[0] {
            return y + (b[0] - a[0]);
        }
        if y > a[n - 1] {
            return y + (b[n - 1] - a[n - 1]);
        }
        if y == a[n - 1] {
            return b[n - 1];
        }
        // largest j with a[j] <= y  (a strictly increasing, a[0] <= y < a[n-1])
        let j = a.partition_point(|&v| v <= y) - 1;
        let slope = (b[j + 1] - b[j]) / (a[j + 1] - a[j]);
        slope * (y - a[j]) + b[j]
    }

    /// Evaluate at many points, in place.
    pub fn apply_inplace(&self, ys: &mut [f64]) {
        for y in ys.iter_mut() {
            *y = self.eval(*y);
        }
    }
}

/// Convenience: evaluate the map with knots `a -> b` at `y`.
pub fn transport_apply(a: &[f64], b: &[f64], y: &[f64]) -> Vec<f64> {
    let t = TransportMap::new(a, b);
    let mut out = y.to_vec();
    t.apply_inplace(&mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid() {
        assert_eq!(quantile_grid(4), vec![0.125, 0.375, 0.625, 0.875]);
    }

    #[test]
    fn pava_basic() {
        assert_eq!(
            pava(&[1.0, 3.0, 2.0, 4.0], None, None),
            vec![1.0, 2.5, 2.5, 4.0]
        );
        assert_eq!(pava(&[3.0, 2.0, 1.0], None, None), vec![2.0, 2.0, 2.0]);
        assert_eq!(
            pava(&[-1.0, 0.5, 2.0], Some(0.0), Some(1.0)),
            vec![0.0, 0.5, 1.0]
        );
        let y = [0.3, -0.2, 0.9, 0.1, 0.5, 2.0, 1.0];
        let p = pava(&y, None, None);
        assert!(p.windows(2).all(|w| w[0] <= w[1]));
        // mean is preserved by the projection
        let (sy, sp): (f64, f64) = (y.iter().sum(), p.iter().sum());
        assert!((sy - sp).abs() < 1e-12);
    }

    #[test]
    fn collapse() {
        let (a, b) = collapse_knots(&[0.0, 1.0, 1.0, 2.0], &[0.0, 1.0, 3.0, 4.0]);
        assert_eq!(a, vec![0.0, 1.0, 2.0]);
        assert_eq!(b, vec![0.0, 2.0, 4.0]);
    }

    #[test]
    fn map_interp_and_extension() {
        let t = TransportMap::new(&[0.0, 1.0, 2.0], &[0.0, 2.0, 3.0]);
        assert_eq!(t.eval(0.5), 1.0);
        assert_eq!(t.eval(1.5), 2.5);
        assert_eq!(t.eval(2.0), 3.0);
        assert_eq!(t.eval(-1.0), -1.0);
        assert_eq!(t.eval(5.0), 6.0);
        let single = TransportMap::new(&[1.0, 1.0], &[2.0, 4.0]);
        assert_eq!(single.eval(0.0), 2.0);
    }

    #[test]
    fn map_inverse_roundtrip() {
        let a = [0.0, 0.1, 0.4, 0.9];
        let b = [1.0, 1.5, 1.6, 3.0];
        let ys = [-0.5, 0.0, 0.05, 0.3, 0.9, 2.0];
        let fwd = transport_apply(&a, &b, &ys);
        let back = transport_apply(&b, &a, &fwd);
        for (y, z) in ys.iter().zip(back) {
            assert!((y - z).abs() < 1e-12);
        }
    }
}
