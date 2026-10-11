//! Log-Cholesky coordinates of SPD matrices (Lin 2019).
//!
//! Coordinates: strict lower part of the Cholesky factor in row-major order
//! (`numpy.tril_indices(m, -1)`), then the log of its diagonal. Mirrors
//! `python/traceot/reference/logchol.py`.

use crate::error::{Result, TraceError};
use crate::linalg::cholesky;

/// Number of coordinates for an `m x m` SPD matrix.
pub fn coord_dim(m: usize) -> usize {
    m * (m + 1) / 2
}

/// SPD matrix (row-major, `m x m`) -> coordinates.
pub fn spd_to_coords(s: &[f64], m: usize) -> Result<Vec<f64>> {
    let l = cholesky(s, m)?;
    let mut out = Vec::with_capacity(coord_dim(m));
    for r in 1..m {
        for c in 0..r {
            out.push(l[r * m + c]);
        }
    }
    for k in 0..m {
        out.push(l[k * m + k].ln());
    }
    Ok(out)
}

/// Coordinates -> SPD matrix `L L^T` (row-major).
pub fn coords_to_spd(z: &[f64], m: usize) -> Result<Vec<f64>> {
    if z.len() != coord_dim(m) {
        return Err(TraceError::Shape(format!(
            "expected {} coordinates",
            coord_dim(m)
        )));
    }
    let mut l = vec![0.0; m * m];
    let mut t = 0;
    for r in 1..m {
        for c in 0..r {
            l[r * m + c] = z[t];
            t += 1;
        }
    }
    for k in 0..m {
        l[k * m + k] = z[t + k].exp();
    }
    let mut s = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            let mut acc = 0.0;
            for k in 0..=i.min(j) {
                acc += l[i * m + k] * l[j * m + k];
            }
            s[i * m + j] = acc;
        }
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let z = vec![0.3, -0.2, 0.1, 0.05, -0.4, 0.7];
        let s = coords_to_spd(&z, 3).unwrap();
        let back = spd_to_coords(&s, 3).unwrap();
        for (a, b) in z.iter().zip(back) {
            assert!((a - b).abs() < 1e-12);
        }
        assert!(coords_to_spd(&z, 2).is_err());
    }
}
