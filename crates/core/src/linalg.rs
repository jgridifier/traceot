//! Minimal dense row-major matrix and the few linear-algebra helpers we need.

use crate::error::{Result, TraceError};

/// Dense row-major matrix of `f64`.
#[derive(Debug, Clone, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Mat {
    /// All-zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Mat {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Wrap a row-major buffer.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(TraceError::Shape(format!(
                "buffer of length {} cannot be {rows}x{cols}",
                data.len()
            )));
        }
        Ok(Mat { rows, cols, data })
    }

    /// Row `i` as a slice.
    #[inline]
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// Mutable row `i`.
    #[inline]
    pub fn row_mut(&mut self, i: usize) -> &mut [f64] {
        &mut self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// Element access.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    /// Column `j` copied into a vector.
    pub fn col(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self.get(i, j)).collect()
    }

    /// `self @ other`.
    pub fn matmul(&self, other: &Mat) -> Result<Mat> {
        if self.cols != other.rows {
            return Err(TraceError::Shape(format!(
                "matmul {}x{} @ {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )));
        }
        let mut out = Mat::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            let orow = &mut out.data[i * other.cols..(i + 1) * other.cols];
            for l in 0..self.cols {
                let w = self.data[i * self.cols + l];
                if w == 0.0 {
                    continue;
                }
                let brow = &other.data[l * other.cols..(l + 1) * other.cols];
                for (o, b) in orow.iter_mut().zip(brow) {
                    *o += w * b;
                }
            }
        }
        Ok(out)
    }

    /// Column means.
    pub fn col_means(&self) -> Vec<f64> {
        let mut m = vec![0.0; self.cols];
        for i in 0..self.rows {
            for (acc, v) in m.iter_mut().zip(self.row(i)) {
                *acc += v;
            }
        }
        let n = self.rows as f64;
        m.iter_mut().for_each(|v| *v /= n);
        m
    }
}

/// Cholesky factor `L` (lower, row-major) of a symmetric positive-definite `a` (k x k).
pub fn cholesky(a: &[f64], k: usize) -> Result<Vec<f64>> {
    if a.len() != k * k {
        return Err(TraceError::Shape("cholesky: matrix is not k x k".into()));
    }
    let mut l = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut s = a[i * k + j];
            for t in 0..j {
                s -= l[i * k + t] * l[j * k + t];
            }
            if i == j {
                if s <= 0.0 || !s.is_finite() {
                    return Err(TraceError::Numerical(
                        "cholesky: matrix is not positive definite".into(),
                    ));
                }
                l[i * k + i] = s.sqrt();
            } else {
                l[i * k + j] = s / l[j * k + j];
            }
        }
    }
    Ok(l)
}

/// Solve `A x = b` for SPD `A` via Cholesky.
pub fn solve_spd(a: &[f64], k: usize, b: &[f64]) -> Result<Vec<f64>> {
    let l = cholesky(a, k)?;
    let mut y = vec![0.0; k];
    for i in 0..k {
        let mut s = b[i];
        for t in 0..i {
            s -= l[i * k + t] * y[t];
        }
        y[i] = s / l[i * k + i];
    }
    let mut x = vec![0.0; k];
    for i in (0..k).rev() {
        let mut s = y[i];
        for t in i + 1..k {
            s -= l[t * k + i] * x[t];
        }
        x[i] = s / l[i * k + i];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_small() {
        let a = Mat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Mat::from_vec(2, 1, vec![1.0, 1.0]).unwrap();
        assert_eq!(a.matmul(&b).unwrap().data, vec![3.0, 7.0]);
        assert!(a.matmul(&Mat::zeros(3, 1)).is_err());
    }

    #[test]
    fn solve_spd_roundtrip() {
        let a = vec![4.0, 1.0, 1.0, 3.0];
        let x = solve_spd(&a, 2, &[1.0, 2.0]).unwrap();
        let r0 = 4.0 * x[0] + x[1] - 1.0;
        let r1 = x[0] + 3.0 * x[1] - 2.0;
        assert!(r0.abs() < 1e-14 && r1.abs() < 1e-14);
        assert!(cholesky(&[1.0, 2.0, 2.0, 1.0], 2).is_err());
    }
}
