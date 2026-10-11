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

/// Pinball subgradients for `Q(u) = q0(u) + sum_m theta_m basis_m(u)`.
///
/// `theta` and the returned gradient are interleaved `(n, M)`: index `i * M + m`.
/// `basis` is row-major `(K, M)`. Also returns the per-row mean pinball loss.
/// The `n x K` quantile matrix is never stored.
pub fn pinball_grad_channels(
    y: &[f64],
    theta: &[f64],
    q0: &[f64],
    basis: &[f64],
    u: &[f64],
    n_channels: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let n = y.len();
    let k = u.len();
    let m_ch = n_channels;
    if m_ch == 0 || q0.len() != k || basis.len() != k * m_ch || theta.len() != n * m_ch {
        return Err(TraceError::Shape(
            "pinball channel shapes are inconsistent".into(),
        ));
    }
    let mut grad = vec![0.0; n * m_ch];
    let mut row_loss = vec![0.0; n];
    pinball_grad_into(y, theta, q0, basis, u, m_ch, &mut grad, &mut row_loss);
    Ok((grad, row_loss))
}

/// Fill `grad` `(n, M)` interleaved and `row_loss` `(n,)`. Buffers are overwritten.
#[allow(clippy::too_many_arguments)]
pub fn pinball_grad_into(
    y: &[f64],
    theta: &[f64],
    q0: &[f64],
    basis: &[f64],
    u: &[f64],
    m_ch: usize,
    grad: &mut [f64],
    row_loss: &mut [f64],
) {
    let n = y.len();
    let k = u.len();
    let one = |s: usize, g: &mut [f64], loss: &mut [f64]| {
        for r in 0..loss.len() {
            let i = s + r;
            let th = &theta[i * m_ch..(i + 1) * m_ch];
            let grow = &mut g[r * m_ch..(r + 1) * m_ch];
            grow.fill(0.0);
            let mut acc = 0.0;
            for kk in 0..k {
                let b = &basis[kk * m_ch..(kk + 1) * m_ch];
                let mut q = q0[kk];
                for m in 0..m_ch {
                    q += th[m] * b[m];
                }
                let res = y[i] - q;
                let rr = if res < 0.0 { u[kk] - 1.0 } else { u[kk] };
                acc += res * rr;
                for m in 0..m_ch {
                    grow[m] += rr * b[m];
                }
            }
            loss[r] = acc / k as f64;
        }
    };
    #[cfg(feature = "parallel")]
    {
        let threads = rayon::current_num_threads().max(1).min(n.max(1));
        let parts = disjoint_rows(grad, row_loss, m_ch, threads);
        parts
            .into_par_iter()
            .enumerate()
            .for_each(|(t, (g, loss))| {
                let s = t * n / threads;
                one(s, g, loss);
            });
    }
    #[cfg(not(feature = "parallel"))]
    one(0, grad, row_loss);
}

/// Split gradient and row-loss buffers into `threads` disjoint row ranges.
#[cfg(feature = "parallel")]
fn disjoint_rows<'a>(
    grad: &'a mut [f64],
    row_loss: &'a mut [f64],
    m_ch: usize,
    threads: usize,
) -> Vec<(&'a mut [f64], &'a mut [f64])> {
    let n = row_loss.len();
    let mut g = grad;
    let mut loss = row_loss;
    let mut out = Vec::with_capacity(threads);
    for t in 0..threads {
        let begin = t * n / threads;
        let end = (t + 1) * n / threads;
        let rows = end - begin;
        let (gl, gr) = g.split_at_mut(rows * m_ch);
        let (ll, lr) = loss.split_at_mut(rows);
        out.push((gl, ll));
        g = gr;
        loss = lr;
    }
    out
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

    #[test]
    fn channel_gradient_matches_location_scale() {
        // Q = theta0 * 1 + theta1 * z, basis = [[1, 0], [1, 1]], two rows.
        let (g, loss) = pinball_grad_channels(
            &[1.0, -1.0],
            &[0.0, 0.0, 0.0, 0.0],
            &[0.0, 0.0],
            &[1.0, 0.0, 1.0, 1.0],
            &[0.25, 0.75],
            2,
        )
        .unwrap();
        // y=1 > 0: r = u. y=-1 < 0: r = u-1.
        // G[0,0] = 0.25*1 + 0.75*1 = 1; G[0,1] = 0.25*0 + 0.75*1 = 0.75
        assert!((g[0] - 1.0).abs() < 1e-15);
        assert!((g[1] - 0.75).abs() < 1e-15);
        assert!((g[2] - ((0.25 - 1.0) + (0.75 - 1.0))).abs() < 1e-15);
        assert!((loss[0] - 0.5).abs() < 1e-15, "{loss:?}");
        assert!(pinball_grad_channels(&[1.0], &[], &[0.0], &[1.0], &[0.5], 1).is_err());
    }
}
