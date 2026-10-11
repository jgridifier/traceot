//! Pinball (quantile) subgradients for a location-scale quantile model.
//!
//! `Q_ik = loc_i + scale_i z_k`, `r_ik = u_k - 1{y_i < Q_ik}`,
//! `G[m][i] = sum_k r_ik basis[k][m]`, loss = mean over i, k of the pinball loss.
//! O(n K) time, O(n M) memory: the n x K quantile matrix is never stored.

use super::CHUNK;
use crate::error::{Result, TraceError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Returns channel-major gradients (M columns of length n) and the mean pinball loss.
/// `basis` is row-major K x M.
pub fn pinball_grad_location_scale(
    y: &[f64],
    loc: &[f64],
    scale: &[f64],
    z: &[f64],
    u: &[f64],
    basis: &[f64],
    n_channels: usize,
) -> Result<(Vec<Vec<f64>>, f64)> {
    let n = y.len();
    let k = z.len();
    if loc.len() != n || scale.len() != n {
        return Err(TraceError::Shape(
            "y, loc, scale must have equal length".into(),
        ));
    }
    if u.len() != k || basis.len() != k * n_channels || n_channels == 0 {
        return Err(TraceError::Shape(
            "z, u, basis shapes are inconsistent".into(),
        ));
    }
    // row-major per-chunk buffer (rows x M), then scattered to channel-major columns
    let one = |ci: usize| -> (usize, Vec<f64>, f64) {
        let s = ci * CHUNK;
        let e = (s + CHUNK).min(n);
        let mut g = vec![0.0; (e - s) * n_channels];
        let mut loss = 0.0;
        for i in s..e {
            let row = &mut g[(i - s) * n_channels..(i - s + 1) * n_channels];
            for kk in 0..k {
                let res = y[i] - (loc[i] + scale[i] * z[kk]);
                let r = if res < 0.0 { u[kk] - 1.0 } else { u[kk] };
                loss += res * r;
                let b = &basis[kk * n_channels..(kk + 1) * n_channels];
                for (o, bv) in row.iter_mut().zip(b) {
                    *o += r * bv;
                }
            }
        }
        (s, g, loss)
    };
    let n_chunks = n.div_ceil(CHUNK);
    #[cfg(feature = "parallel")]
    let parts: Vec<(usize, Vec<f64>, f64)> = (0..n_chunks).into_par_iter().map(one).collect();
    #[cfg(not(feature = "parallel"))]
    let parts: Vec<(usize, Vec<f64>, f64)> = (0..n_chunks).map(one).collect();
    let mut cols = vec![vec![0.0; n]; n_channels];
    let mut loss = 0.0;
    for (s, g, l) in parts {
        loss += l;
        let rows = g.len() / n_channels;
        for r in 0..rows {
            for m in 0..n_channels {
                cols[m][s + r] = g[r * n_channels + m];
            }
        }
    }
    let denom = (n * k).max(1) as f64;
    Ok((cols, loss / denom))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_gradient_signs() {
        // single level u = 0.5, basis = [1]: gradient is +0.5 above Q, -0.5 below
        let (g, loss) = pinball_grad_location_scale(
            &[2.0, -2.0],
            &[0.0, 0.0],
            &[1.0, 1.0],
            &[0.0],
            &[0.5],
            &[1.0],
            1,
        )
        .unwrap();
        assert_eq!(g[0], vec![0.5, -0.5]);
        assert!((loss - 1.0).abs() < 1e-15);
        assert!(
            pinball_grad_location_scale(&[1.0], &[], &[1.0], &[0.0], &[0.5], &[1.0], 1).is_err()
        );
    }
}
