//! Estimator-level logic: response spaces, model fitting and prediction, K-fold
//! bandwidth cross-validation, global Fréchet regression and the Fréchet R².
//!
//! This is everything the Python estimators need: the Python package only validates
//! inputs, converts arrays and offers a scikit-learn-style API. The test reference is
//! `tests/python/reference/estimators.py`.

use crate::additive::{additive_euclid_fit, additive_euclid_predict};
use crate::adopt::{
    adopt_w2_component, adopt_w2_fit, adopt_w2_predict, BackfitConfig, BackfitResult,
};
use crate::error::{Result, TraceError};
use crate::linalg::Mat;
use crate::logchol;
use crate::smoother::{global_frechet_weights, local_linear_weights};
use crate::w2::{pava, TransportMap};

/// Response space. Responses are stored in a flat encoding: quantile functions on the
/// midpoint grid (W2) or log-Cholesky coordinates (SPD).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Space {
    Wasserstein,
    LogCholesky,
}

impl Space {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "wasserstein" => Ok(Space::Wasserstein),
            "logcholesky" => Ok(Space::LogCholesky),
            _ => Err(TraceError::InvalidArgument(format!(
                "unknown space {name:?} (expected 'wasserstein' or 'logcholesky')"
            ))),
        }
    }

    /// Squared distance between two encoded responses.
    pub fn dist_sq(self, a: &[f64], b: &[f64]) -> f64 {
        let s: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        match self {
            Space::Wasserstein => s / a.len() as f64,
            Space::LogCholesky => s,
        }
    }
}

/// Encode `n` SPD matrices (`n x m x m`, row-major) as log-Cholesky coordinates.
pub fn encode_spd(s: &[f64], n: usize, m: usize) -> Result<Mat> {
    if s.len() != n * m * m {
        return Err(TraceError::Shape("expected n*m*m values".into()));
    }
    let d = logchol::coord_dim(m);
    let mut out = Mat::zeros(n, d);
    for i in 0..n {
        let z = logchol::spd_to_coords(&s[i * m * m..(i + 1) * m * m], m)?;
        out.row_mut(i).copy_from_slice(&z);
    }
    Ok(out)
}

/// Decode log-Cholesky coordinates to SPD matrices (`n x m x m`, row-major).
pub fn decode_spd(z: &Mat, m: usize) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(z.rows * m * m);
    for i in 0..z.rows {
        out.extend(logchol::coords_to_spd(z.row(i), m)?);
    }
    Ok(out)
}

/// Row-wise squared distances.
pub fn dist_sq_rows(space: Space, a: &Mat, b: &Mat) -> Result<Vec<f64>> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(TraceError::Shape("arrays differ in shape".into()));
    }
    Ok((0..a.rows)
        .map(|i| space.dist_sq(a.row(i), b.row(i)))
        .collect())
}

/// Fréchet R²: `1 - sum_i d²(pred_i, z_i) / sum_i d²(mean, z_i)` (NaN if the
/// denominator is zero).
pub fn frechet_r2(space: Space, pred: &Mat, z: &Mat, mean: &[f64]) -> Result<f64> {
    let num: f64 = dist_sq_rows(space, pred, z)?.iter().sum();
    if mean.len() != z.cols {
        return Err(TraceError::Shape("mean has wrong length".into()));
    }
    let den: f64 = (0..z.rows).map(|i| space.dist_sq(mean, z.row(i))).sum();
    Ok(if den > 0.0 { 1.0 - num / den } else { f64::NAN })
}

/// Additive models that share the backfitting machinery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// ADOPT. W2: transport backfitting. Log-Cholesky: the space is flat, so ADOPT is
    /// exactly additive backfitting in coordinates.
    Adopt,
    /// Ours (not in the paper): additive backfitting of quantile vectors, then PAVA.
    L2AdditiveQuantile,
}

impl Method {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "adopt" => Ok(Method::Adopt),
            "l2additive" => Ok(Method::L2AdditiveQuantile),
            _ => Err(TraceError::InvalidArgument(format!(
                "unknown method {name:?}"
            ))),
        }
    }
}

/// A fitted additive model with its training inputs and bandwidths.
#[derive(Debug, Clone)]
pub struct AdditiveModel {
    pub method: Method,
    pub space: Space,
    pub cfg: BackfitConfig,
    pub x_train: Mat,
    pub h: Vec<f64>,
    pub fit: BackfitResult,
}

fn pava_rows(a: Mat, cfg: &BackfitConfig) -> Mat {
    let mut out = a;
    for i in 0..out.rows {
        let r = pava(out.row(i), cfg.lower, cfg.upper);
        out.row_mut(i).copy_from_slice(&r);
    }
    out
}

impl AdditiveModel {
    pub fn fit(
        method: Method,
        space: Space,
        x: &Mat,
        z: &Mat,
        h: &[f64],
        cfg: BackfitConfig,
    ) -> Result<Self> {
        if method == Method::L2AdditiveQuantile && space != Space::Wasserstein {
            return Err(TraceError::InvalidArgument(
                "the L2-additive quantile model is defined for the wasserstein space only".into(),
            ));
        }
        let fit = match (method, space) {
            (Method::Adopt, Space::Wasserstein) => adopt_w2_fit(x, z, h, &cfg)?,
            _ => additive_euclid_fit(x, z, h, &cfg)?,
        };
        Ok(AdditiveModel {
            method,
            space,
            cfg,
            x_train: x.clone(),
            h: h.to_vec(),
            fit,
        })
    }

    /// Predictions in the encoded representation.
    pub fn predict(&self, x_new: &Mat) -> Result<Mat> {
        let f = &self.fit;
        match (self.method, self.space) {
            (Method::Adopt, Space::Wasserstein) => adopt_w2_predict(
                &self.x_train,
                &f.resid,
                &f.m,
                &f.mu,
                &self.h,
                x_new,
                &self.cfg,
            ),
            (Method::Adopt, Space::LogCholesky) => additive_euclid_predict(
                &self.x_train,
                &f.resid,
                &f.m,
                &f.mu,
                &self.h,
                x_new,
                &self.cfg,
            ),
            (Method::L2AdditiveQuantile, _) => {
                let raw = additive_euclid_predict(
                    &self.x_train,
                    &f.resid,
                    &f.m,
                    &f.mu,
                    &self.h,
                    x_new,
                    &self.cfg,
                )?;
                Ok(pava_rows(raw, &self.cfg))
            }
        }
    }

    /// Component curve `g_j(x)` at points `x` (encoded; one row per point).
    pub fn component(&self, j: usize, x: &[f64]) -> Result<Mat> {
        if j >= self.x_train.cols {
            return Err(TraceError::InvalidArgument(format!(
                "feature index {j} out of range"
            )));
        }
        let xj = self.x_train.col(j);
        match (self.method, self.space) {
            (Method::Adopt, Space::Wasserstein) => {
                adopt_w2_component(&xj, &self.fit.resid[j], self.h[j], x, &self.cfg)
            }
            _ => {
                local_linear_weights(&xj, x, self.h[j], self.cfg.kernel)?.matmul(&self.fit.resid[j])
            }
        }
    }

    /// Apply the j-th additive transport `T_j(x0) = T_{M_j, g_j(x0)}` to response `v`.
    pub fn transport(&self, j: usize, x0: f64, v: &[f64]) -> Result<Vec<f64>> {
        let g = self.component(j, &[x0])?;
        let center = &self.fit.m[j];
        match self.space {
            Space::Wasserstein if self.method == Method::Adopt => {
                let mut out = v.to_vec();
                TransportMap::new(center, g.row(0)).apply_inplace(&mut out);
                Ok(out)
            }
            _ => Ok(v
                .iter()
                .zip(g.row(0))
                .zip(center)
                .map(|((a, b), c)| a + b - c)
                .collect()),
        }
    }
}

/// SplitMix64 (Steele, Lea & Flood 2014): small, fast and fully specified, so that
/// the test reference can reproduce it bit for bit.
#[derive(Debug, Clone)]
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Shuffled K-fold split. Fisher–Yates with `SplitMix64(seed)` (index `i` from `n-1`
/// down to 1 swaps with `next_u64() % (i + 1)`), then consecutive blocks with the sizes
/// of `numpy.array_split`; each fold is returned sorted.
pub fn kfold_indices(n: usize, k: usize, seed: u64) -> Result<Vec<Vec<usize>>> {
    if k < 2 || k > n {
        return Err(TraceError::InvalidArgument(format!(
            "cv must be between 2 and n_samples={n}"
        )));
    }
    let mut perm: Vec<usize> = (0..n).collect();
    let mut rng = SplitMix64(seed);
    for i in (1..n).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        perm.swap(i, j);
    }
    let (base, extra) = (n / k, n % k);
    let mut folds = Vec::with_capacity(k);
    let mut start = 0;
    for f in 0..k {
        let len = base + usize::from(f < extra);
        let mut fold = perm[start..start + len].to_vec();
        fold.sort_unstable();
        folds.push(fold);
        start += len;
    }
    Ok(folds)
}

/// Observed range of each column; zero ranges are replaced by one.
pub fn column_ranges(x: &Mat) -> Vec<f64> {
    (0..x.cols)
        .map(|j| {
            let c = x.col(j);
            let lo = c.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = c.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let r = hi - lo;
            if r > 0.0 {
                r
            } else {
                1.0
            }
        })
        .collect()
}

fn take_rows(a: &Mat, idx: &[usize]) -> Mat {
    let mut out = Mat::zeros(idx.len(), a.cols);
    for (r, &i) in idx.iter().enumerate() {
        out.row_mut(r).copy_from_slice(a.row(i));
    }
    out
}

/// Result of bandwidth cross-validation.
#[derive(Debug, Clone)]
pub struct CvResult {
    pub grid: Vec<f64>,
    /// `(1/n) sum_i d²(Ŷ_i^(-fold(i)), Y_i)` for each candidate fraction.
    pub mean_dist_sq: Vec<f64>,
    pub best_fraction: f64,
}

/// K-fold CV over a common bandwidth fraction `c` (predictor j uses
/// `h_j = c * range(X_j)`, ranges of the full sample). Ties go to the first minimum.
/// The (candidate, fold) fits are independent and run in parallel with rayon.
#[allow(clippy::too_many_arguments)]
pub fn select_bandwidth_cv(
    method: Method,
    space: Space,
    x: &Mat,
    z: &Mat,
    grid: &[f64],
    k: usize,
    seed: u64,
    cfg: BackfitConfig,
) -> Result<CvResult> {
    if grid.is_empty() || grid.iter().any(|c| !(*c > 0.0)) {
        return Err(TraceError::InvalidArgument(
            "bandwidth grid must be non-empty and > 0".into(),
        ));
    }
    if x.rows != z.rows {
        return Err(TraceError::Shape(
            "X and Y have different numbers of samples".into(),
        ));
    }
    let n = x.rows;
    let folds = kfold_indices(n, k, seed)?;
    let ranges = column_ranges(x);
    let jobs: Vec<(usize, usize)> = (0..grid.len())
        .flat_map(|g| (0..k).map(move |f| (g, f)))
        .collect();
    let run = |&(g, f): &(usize, usize)| -> Result<f64> {
        let test = &folds[f];
        let mut is_test = vec![false; n];
        test.iter().for_each(|&i| is_test[i] = true);
        let train: Vec<usize> = (0..n).filter(|&i| !is_test[i]).collect();
        let h: Vec<f64> = ranges.iter().map(|r| grid[g] * r).collect();
        let model = AdditiveModel::fit(
            method,
            space,
            &take_rows(x, &train),
            &take_rows(z, &train),
            &h,
            cfg,
        )?;
        let pred = model.predict(&take_rows(x, test))?;
        let zt = take_rows(z, test);
        Ok(dist_sq_rows(space, &pred, &zt)?.iter().sum())
    };
    #[cfg(feature = "parallel")]
    let sums: Vec<Result<f64>> = {
        use rayon::prelude::*;
        jobs.par_iter().map(run).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let sums: Vec<Result<f64>> = jobs.iter().map(run).collect();
    let mut errs = vec![0.0; grid.len()];
    // Sum folds in a fixed order so that results do not depend on scheduling.
    for (job, s) in jobs.iter().zip(sums) {
        errs[job.0] += s?;
    }
    errs.iter_mut().for_each(|e| *e /= n as f64);
    let mut best = 0;
    for g in 1..errs.len() {
        if errs[g] < errs[best] {
            best = g;
        }
    }
    Ok(CvResult {
        grid: grid.to_vec(),
        mean_dist_sq: errs,
        best_fraction: grid[best],
    })
}

/// Global Fréchet regression (Petersen & Müller 2019) at the rows of `x_new`:
/// global weights applied to the encoded responses, then (W2) PAVA projection.
pub fn global_frechet_predict(
    space: Space,
    x_train: &Mat,
    z_train: &Mat,
    x_new: &Mat,
    cfg: &BackfitConfig,
) -> Result<Mat> {
    let w = global_frechet_weights(x_train, x_new)?;
    let raw = w.matmul(z_train)?;
    Ok(match space {
        Space::Wasserstein => pava_rows(raw, cfg),
        Space::LogCholesky => raw,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_reference_values() {
        // First outputs for seed 0 (published test vector of SplitMix64).
        let mut r = SplitMix64(0);
        assert_eq!(r.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(r.next_u64(), 0x6E78_9E6A_A1B9_65F4);
    }

    #[test]
    fn kfold_partitions() {
        let folds = kfold_indices(23, 5, 7).unwrap();
        let sizes: Vec<usize> = folds.iter().map(|f| f.len()).collect();
        assert_eq!(sizes, vec![5, 5, 5, 4, 4]);
        let mut all: Vec<usize> = folds.concat();
        all.sort_unstable();
        assert_eq!(all, (0..23).collect::<Vec<_>>());
        assert!(kfold_indices(3, 5, 0).is_err());
    }

    #[test]
    fn cv_picks_a_grid_value_and_r2_is_sane() {
        let n = 60;
        let q = 12;
        let mut x = Mat::zeros(n, 2);
        let mut z = Mat::zeros(n, q);
        for i in 0..n {
            let a = (i as f64 * 0.618_033_988_7).fract();
            let b = (i as f64 * 0.414_213_562_3).fract();
            x.row_mut(i).copy_from_slice(&[a, b]);
            for k in 0..q {
                let u = (k as f64 + 0.5) / q as f64;
                z.row_mut(i)[k] = (a - 0.5) + (1.0 + b) * (u - 0.5);
            }
        }
        let cfg = BackfitConfig::default();
        let grid = [0.05, 0.1, 0.3];
        let cv = select_bandwidth_cv(Method::Adopt, Space::Wasserstein, &x, &z, &grid, 5, 0, cfg)
            .unwrap();
        assert!(grid.contains(&cv.best_fraction));
        let h: Vec<f64> = column_ranges(&x)
            .iter()
            .map(|r| r * cv.best_fraction)
            .collect();
        let model = AdditiveModel::fit(Method::Adopt, Space::Wasserstein, &x, &z, &h, cfg).unwrap();
        let pred = model.predict(&x).unwrap();
        let r2 = frechet_r2(Space::Wasserstein, &pred, &z, &model.fit.mu).unwrap();
        assert!(r2 > 0.95, "r2 {r2}");
    }
}
