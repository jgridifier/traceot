//! Conditional and marginal shift decompositions, plus noise-floor helpers.
//!
//! Marginal composition and structure are exact as a telescoping split of the
//! mixture quantile functions. They are not an exact per-feature split.
//! Cross-fit energies are returned unclipped.

use super::legendre::{eval_a, hosking_lmoments};
use super::v1fit::{codes_of, curve_coefficients, player_coefficients, V1Model};
use crate::error::{Result, TraceError};

#[derive(Clone, Debug)]
pub struct ConditionalShift {
    /// Mean over rows of per-player, per-channel Shapley shares. Length `(p+1) * k_fit`.
    pub phi_mean: Vec<f64>,
    /// `E W2²`, equal to `phi_mean.sum()` before projection.
    pub total: f64,
    pub n: usize,
    pub p: usize,
    pub k_fit: usize,
}

/// Exact channel × player decomposition of the conditional shift on `X_eval`.
/// Both models are recentred under the evaluation rows before the split.
pub fn shift_conditional(
    model_a: &V1Model,
    model_b: &V1Model,
    x_eval: &[f64],
    n: usize,
) -> Result<ConditionalShift> {
    if model_a.k_fit != model_b.k_fit || model_a.edges.len() != model_b.edges.len() {
        return Err(TraceError::InvalidArgument(
            "the two periods must share K_fit and the feature count".into(),
        ));
    }
    let a = recenter(model_a, x_eval, n)?;
    let b = recenter(model_b, x_eval, n)?;
    let p = a.edges.len();
    let k = a.k_fit;
    let pa = player_coefficients(&a, x_eval, n)?;
    let pb = player_coefficients(&b, x_eval, n)?;
    let mut phi_mean = vec![0.0; (p + 1) * k];
    for i in 0..n {
        for c in 0..k {
            let mut total = 0.0;
            for j in 0..=p {
                total += pb[i * (p + 1) * k + j * k + c] - pa[i * (p + 1) * k + j * k + c];
            }
            for j in 0..=p {
                let d = pb[i * (p + 1) * k + j * k + c] - pa[i * (p + 1) * k + j * k + c];
                phi_mean[j * k + c] += d * total;
            }
        }
    }
    let scale = 1.0 / n as f64;
    for v in &mut phi_mean {
        *v *= scale;
    }
    let total = phi_mean.iter().sum();
    Ok(ConditionalShift {
        phi_mean,
        total,
        n,
        p,
        k_fit: k,
    })
}

fn recenter(model: &V1Model, x: &[f64], n: usize) -> Result<V1Model> {
    let codes = codes_of(model, x, n)?;
    let mut out = model.clone();
    let k = model.k_fit;
    let p = codes.len();
    for j in 0..p {
        let nb = out.n_bins[j];
        let mut counts = vec![0u64; nb];
        for i in 0..n {
            counts[codes[j][i] as usize] += 1;
        }
        let mut mean = vec![0.0; k];
        let mut total = 0.0;
        for b in 0..nb {
            total += counts[b] as f64;
            for c in 0..k {
                mean[c] += counts[b] as f64 * out.effects[j][b * k + c];
            }
        }
        if total == 0.0 {
            continue;
        }
        for c in 0..k {
            mean[c] /= total;
            out.intercept[c] += mean[c];
        }
        for b in 0..nb {
            for c in 0..k {
                out.effects[j][b * k + c] -= mean[c];
            }
        }
    }
    Ok(out)
}

#[derive(Clone, Debug)]
pub struct MarginalShift {
    pub total: f64,
    pub composition: f64,
    pub structure: f64,
    pub cross: f64,
    /// Composition term on the composition-first path (`‖Q_C − Q_A‖²`).
    pub composition_first: f64,
    /// Composition piece on the structure-first path (`total − ‖Q^{(B,P_A)} − Q_A‖²`).
    pub structure_first_composition: f64,
    pub shapley_composition: f64,
    pub shapley_structure: f64,
    /// Bins present in `X_B` and empty in A's training counts.
    pub overlap_failures: Vec<Vec<usize>>,
    pub qa: Vec<f64>,
    pub qb: Vec<f64>,
    pub qc: Vec<f64>,
}

/// Mixture-quantile decomposition of periods A and B.
/// `grid` is the number of uniform midpoint levels used both to build each
/// conditional curve and to read the mixture quantiles back.
pub fn shift_marginal(
    model_a: &V1Model,
    model_b: &V1Model,
    x_a: &[f64],
    n_a: usize,
    x_b: &[f64],
    n_b: usize,
    grid: usize,
) -> Result<MarginalShift> {
    if grid < 8 {
        return Err(TraceError::InvalidArgument(
            "mixture grid must have at least 8 levels".into(),
        ));
    }
    let qa = mixture_quantile(model_a, x_a, n_a, grid)?;
    let qb = mixture_quantile(model_b, x_b, n_b, grid)?;
    let qc = mixture_quantile(model_a, x_b, n_b, grid)?;
    let qd = mixture_quantile(model_b, x_a, n_a, grid)?;
    let total = mean_sq(&qb, &qa);
    let composition = mean_sq(&qc, &qa);
    let structure = mean_sq(&qb, &qc);
    let cross = 2.0
        * qb.iter()
            .zip(&qc)
            .zip(&qa)
            .map(|((b, c), a)| (b - c) * (c - a))
            .sum::<f64>()
        / grid as f64;
    let structure_only = mean_sq(&qd, &qa);
    let shapley_composition = 0.5 * composition + 0.5 * (total - structure_only);
    let shapley_structure = 0.5 * structure_only + 0.5 * (total - composition);
    let codes_b = codes_of(model_a, x_b, n_b)?;
    let overlap_failures = super::v1fit::overlap_failures(model_a, &codes_b);
    Ok(MarginalShift {
        total,
        composition,
        structure,
        cross,
        composition_first: composition,
        structure_first_composition: total - structure_only,
        shapley_composition,
        shapley_structure,
        overlap_failures,
        qa,
        qb,
        qc,
    })
}

fn mean_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>() / a.len() as f64
}

fn mixture_quantile(model: &V1Model, x: &[f64], n: usize, grid: usize) -> Result<Vec<f64>> {
    if n == 0 {
        return Err(TraceError::InvalidArgument(
            "mixture needs at least one row".into(),
        ));
    }
    let p = model.edges.len();
    let k = model.k_fit;
    let players = player_coefficients(model, x, n)?;
    let curves = curve_coefficients(&players, n, p, k);
    let mut pooled = Vec::with_capacity(n * grid);
    for i in 0..n {
        let a = &curves[i * k..(i + 1) * k];
        for g in 0..grid {
            let u = (g as f64 + 0.5) / grid as f64;
            pooled.push(eval_a(a, u));
        }
    }
    pooled.sort_by(f64::total_cmp);
    let mut q = vec![0.0; grid];
    let m = pooled.len();
    for g in 0..grid {
        let u = (g as f64 + 0.5) / grid as f64;
        let idx = ((u * m as f64).floor() as usize).min(m - 1);
        q[g] = pooled[idx];
    }
    Ok(q)
}

/// Unclipped cross-fit energy `⟨Δ_1, Δ_2⟩` of two coefficient-change vectors.
pub fn cross_fit_energy(delta1: &[f64], delta2: &[f64]) -> Result<f64> {
    if delta1.len() != delta2.len() {
        return Err(TraceError::Shape(
            "cross-fit vectors must have the same length".into(),
        ));
    }
    Ok(delta1.iter().zip(delta2).map(|(a, b)| a * b).sum())
}

/// SplitMix64. Used only for permutation and bootstrap index draws.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    pub state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn index(&mut self, n: usize) -> usize {
        let n64 = n as u64;
        let thresh = n64.wrapping_neg() % n64;
        loop {
            let r = self.next_u64();
            if r >= thresh {
                return (r % n64) as usize;
            }
        }
    }
}

/// In-place Fisher–Yates. When `strata` is set, labels are shuffled inside each cell.
pub fn permute_labels(labels: &mut [u8], strata: Option<&[usize]>, rng: &mut SplitMix64) {
    if let Some(strata) = strata {
        let mut cells: Vec<Vec<usize>> = Vec::new();
        for (i, &s) in strata.iter().enumerate() {
            if s >= cells.len() {
                cells.resize(s + 1, Vec::new());
            }
            cells[s].push(i);
        }
        for cell in cells {
            shuffle_subset(labels, &cell, rng);
        }
    } else {
        for i in (1..labels.len()).rev() {
            let j = rng.index(i + 1);
            labels.swap(i, j);
        }
    }
}

fn shuffle_subset(labels: &mut [u8], idx: &[usize], rng: &mut SplitMix64) {
    let mut vals: Vec<u8> = idx.iter().map(|&i| labels[i]).collect();
    for i in (1..vals.len()).rev() {
        let j = rng.index(i + 1);
        vals.swap(i, j);
    }
    for (slot, &i) in idx.iter().enumerate() {
        labels[i] = vals[slot];
    }
}

/// Randomisation p-value `(1 + #{T_π ≥ T}) / (1 + R)`.
pub fn permutation_pvalue<F>(
    mut stat: F,
    labels: &mut [u8],
    strata: Option<&[usize]>,
    r: usize,
    seed: u64,
) -> f64
where
    F: FnMut(&[u8]) -> f64,
{
    let observed = stat(labels);
    let mut rng = SplitMix64::new(seed);
    let mut cnt = 0usize;
    let original = labels.to_vec();
    for _ in 0..r {
        labels.copy_from_slice(&original);
        permute_labels(labels, strata, &mut rng);
        if stat(labels) >= observed {
            cnt += 1;
        }
    }
    labels.copy_from_slice(&original);
    (1.0 + cnt as f64) / (1.0 + r as f64)
}

/// Within-period row bootstrap indices. Returns `reps` concatenated draws of length `n_a + n_b`.
pub fn bootstrap_indices(n_a: usize, n_b: usize, reps: usize, seed: u64) -> Vec<usize> {
    let mut rng = SplitMix64::new(seed);
    let mut out = Vec::with_capacity(reps * (n_a + n_b));
    for _ in 0..reps {
        for _ in 0..n_a {
            out.push(rng.index(n_a));
        }
        for _ in 0..n_b {
            out.push(n_a + rng.index(n_b));
        }
    }
    out
}

/// Within-bin L-moment energy of channel `k` (0-based; `k = 3` is λ₄), matching the
/// exchangeable statistic used for the permutation level check.
pub fn lmoment_channel_energy(
    y: &[f64],
    bins: &[usize],
    labels: &[u8],
    k: usize,
    n_bins: usize,
) -> Result<f64> {
    if y.len() != bins.len() || y.len() != labels.len() {
        return Err(TraceError::Shape(
            "y, bins and labels must have the same length".into(),
        ));
    }
    let order = k + 1;
    let weight = (2 * k + 1) as f64;
    let n = y.len();
    let mut total = 0.0;
    for b in 0..n_bins {
        let mut ya = Vec::new();
        let mut yb = Vec::new();
        for i in 0..n {
            if bins[i] != b {
                continue;
            }
            if labels[i] == 0 {
                ya.push(y[i]);
            } else {
                yb.push(y[i]);
            }
        }
        if ya.len() <= k || yb.len() <= k {
            continue;
        }
        let la = hosking_lmoments(&ya, order)?;
        let lb = hosking_lmoments(&yb, order)?;
        let diff = la[k] - lb[k];
        total += weight * diff * diff * (ya.len() + yb.len()) as f64 / (2.0 * (n as f64 / 2.0));
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_and_identity_signs() {
        let e = cross_fit_energy(&[1.0, -2.0], &[1.0, 0.5]).unwrap();
        assert!((e - 0.0).abs() < 1e-15 || (e - (1.0 - 1.0)).abs() < 1e-12);
        assert!((cross_fit_energy(&[0.5], &[-0.25]).unwrap() + 0.125).abs() < 1e-15);
    }

    #[test]
    fn permutation_of_a_constant_is_one() {
        let labels = &mut [0u8, 1, 0, 1];
        let p = permutation_pvalue(|_| 1.0, labels, None, 20, 1);
        assert!((p - 1.0).abs() < 1e-12);
    }
}
