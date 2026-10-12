//! Shifted Legendre basis, Gauss–Legendre nodes, and monotone projections.
//!
//! Coefficients `a_k` are on the orthonormal basis `e_k = √(2k+1) P*_k`,
//! with `P*_k(u) = P_k(2u−1)` in the Hosking (1990) convention.

use crate::error::{Result, TraceError};

/// Gauss–Legendre nodes and weights on `(0, 1)`.
#[derive(Clone, Debug)]
pub struct GaussLegendre {
    pub nodes: Vec<f64>,
    pub weights: Vec<f64>,
}

/// Nodes and weights of the degree-`n` Gauss–Legendre rule on `(0, 1)`.
/// Exact for polynomials of degree at most `2n − 1`.
pub fn gauss_legendre_01(n: usize) -> Result<GaussLegendre> {
    if n == 0 {
        return Err(TraceError::InvalidArgument(
            "Gauss–Legendre order must be positive".into(),
        ));
    }
    let (x, w) = leggauss(n);
    let nodes = x.iter().map(|z| 0.5 * (z + 1.0)).collect();
    let weights = w.iter().map(|wi| 0.5 * wi).collect();
    Ok(GaussLegendre { nodes, weights })
}

/// Legendre nodes and weights on `[-1, 1]` (Newton on the roots).
fn leggauss(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    let m = n.div_ceil(2);
    for i in 0..m {
        let mut z = ((std::f64::consts::PI * (i as f64 + 0.75)) / (n as f64 + 0.5)).cos();
        let mut dp;
        for _ in 0..80 {
            let (p_n, p_nm1) = legendre_pair(n, z);
            let denom = (1.0 - z * z).max(1e-30);
            dp = n as f64 * (p_nm1 - z * p_n) / denom;
            let step = p_n / dp;
            z -= step;
            if step.abs() < 1e-15 {
                break;
            }
        }
        let (_p_n, _p_nm1) = legendre_pair(n, z);
        let denom = (1.0 - z * z).max(1e-30);
        dp = n as f64 * (_p_nm1 - z * _p_n) / denom;
        x[i] = -z;
        x[n - 1 - i] = z;
        let wi = 2.0 / (denom * dp * dp);
        w[i] = wi;
        w[n - 1 - i] = wi;
    }
    (x, w)
}

fn legendre_pair(n: usize, z: f64) -> (f64, f64) {
    let mut p0 = 1.0;
    let mut p1 = z;
    if n == 0 {
        return (1.0, 0.0);
    }
    if n == 1 {
        return (z, 1.0);
    }
    for k in 2..=n {
        let kf = k as f64;
        let p2 = ((2.0 * kf - 1.0) * z * p1 - (kf - 1.0) * p0) / kf;
        p0 = p1;
        p1 = p2;
    }
    (p1, p0)
}

/// `P*_0, …, P*_{k_max-1}` and their derivatives at `u`.
pub fn pstar_all(u: f64, k_max: usize) -> (Vec<f64>, Vec<f64>) {
    let mut p = vec![0.0; k_max];
    let mut d = vec![0.0; k_max];
    if k_max == 0 {
        return (p, d);
    }
    p[0] = 1.0;
    if k_max == 1 {
        return (p, d);
    }
    p[1] = 2.0 * u - 1.0;
    d[1] = 2.0;
    for k in 1..k_max - 1 {
        let kf = k as f64;
        let x = 2.0 * u - 1.0;
        p[k + 1] = ((2.0 * kf + 1.0) * x * p[k] - kf * p[k - 1]) / (kf + 1.0);
        d[k + 1] = ((2.0 * kf + 1.0) * 2.0 * p[k] + (2.0 * kf + 1.0) * x * d[k] - kf * d[k - 1])
            / (kf + 1.0);
    }
    (p, d)
}

/// Orthonormal shifted Legendre `e_k(u) = √(2k+1) P*_k(u)`.
pub fn orthonormal_at(u: f64, k_max: usize) -> (Vec<f64>, Vec<f64>) {
    let (p, d) = pstar_all(u, k_max);
    let e = p
        .iter()
        .enumerate()
        .map(|(k, v)| v * ((2 * k + 1) as f64).sqrt())
        .collect();
    let de = d
        .iter()
        .enumerate()
        .map(|(k, v)| v * ((2 * k + 1) as f64).sqrt())
        .collect();
    (e, de)
}

/// Evaluate `Q(u) = Σ a_k e_k(u)`.
pub fn eval_a(a: &[f64], u: f64) -> f64 {
    let (e, _) = orthonormal_at(u, a.len());
    a.iter().zip(e).map(|(c, ek)| c * ek).sum()
}

/// Derivative `Q'(u)`.
pub fn eval_a_deriv(a: &[f64], u: f64) -> f64 {
    let (_, de) = orthonormal_at(u, a.len());
    a.iter().zip(de).map(|(c, dk)| c * dk).sum()
}

/// Basis tabulated on a quadrature rule. Layout is node-major: `t * k + c`.
#[derive(Clone, Debug)]
pub struct QuadBasis {
    pub nodes: Vec<f64>,
    pub weights: Vec<f64>,
    pub k: usize,
    pub e: Vec<f64>,
    pub de: Vec<f64>,
}

impl QuadBasis {
    pub fn new(n_u: usize, k: usize) -> Result<Self> {
        if k == 0 {
            return Err(TraceError::InvalidArgument(
                "basis degree must be positive".into(),
            ));
        }
        if n_u < k {
            return Err(TraceError::InvalidArgument(
                "need at least K_fit Gauss–Legendre nodes".into(),
            ));
        }
        let gl = gauss_legendre_01(n_u)?;
        let mut e = vec![0.0; n_u * k];
        let mut de = vec![0.0; n_u * k];
        for (t, &u) in gl.nodes.iter().enumerate() {
            let (ev, dv) = orthonormal_at(u, k);
            e[t * k..(t + 1) * k].copy_from_slice(&ev);
            de[t * k..(t + 1) * k].copy_from_slice(&dv);
        }
        Ok(Self {
            nodes: gl.nodes,
            weights: gl.weights,
            k,
            e,
            de,
        })
    }

    pub fn n_u(&self) -> usize {
        self.nodes.len()
    }

    /// Discrete Gram matrix of `e_0..e_{k-1}`. Exact identity when `n_u >= k`.
    pub fn gram(&self) -> Vec<f64> {
        let k = self.k;
        let mut g = vec![0.0; k * k];
        for t in 0..self.n_u() {
            let w = self.weights[t];
            for i in 0..k {
                for j in 0..k {
                    g[i * k + j] += w * self.e[t * k + i] * self.e[t * k + j];
                }
            }
        }
        g
    }

    pub fn gram_error(&self) -> f64 {
        let g = self.gram();
        let k = self.k;
        let mut m = 0.0_f64;
        for i in 0..k {
            for j in 0..k {
                let target = if i == j { 1.0 } else { 0.0 };
                m = m.max((g[i * k + j] - target).abs());
            }
        }
        m
    }
}

/// `c_k = a_k √(2k+1)`, `λ_{k+1} = a_k / √(2k+1)`.
pub fn a_to_c(a: &[f64]) -> Vec<f64> {
    a.iter()
        .enumerate()
        .map(|(k, v)| v * ((2 * k + 1) as f64).sqrt())
        .collect()
}

pub fn a_to_lambda(a: &[f64]) -> Vec<f64> {
    a.iter()
        .enumerate()
        .map(|(k, v)| v / ((2 * k + 1) as f64).sqrt())
        .collect()
}

pub fn lambda_to_a(lambda: &[f64]) -> Vec<f64> {
    lambda
        .iter()
        .enumerate()
        .map(|(k, v)| v * ((2 * k + 1) as f64).sqrt())
        .collect()
}

/// `|τ_3| ≤ 1/5` with `τ_3 = λ_3 / λ_2`, the K = 3 monotone region when `λ_2 > 0`.
pub fn tau3_from_a(a: &[f64]) -> Option<f64> {
    if a.len() < 3 {
        return None;
    }
    let lam = a_to_lambda(a);
    if lam[1].abs() < 1e-15 {
        return None;
    }
    Some(lam[2] / lam[1])
}

/// Proposition 1.5: `|c_2| ≤ c_1 / 3` (and `c_1 ≥ 0`).
pub fn k3_monotone(a: &[f64]) -> bool {
    if a.len() < 3 {
        return false;
    }
    let c = a_to_c(a);
    c[1] >= -1e-12 && c[2].abs() <= c[1] / 3.0 + 1e-12
}

/// `|τ_3| > 1/5` cannot be represented by a monotone 3-term expansion.
pub fn k3_non_representable(a: &[f64]) -> bool {
    match tau3_from_a(a) {
        Some(t) => t.abs() > 0.2 + 1e-12,
        None => !k3_monotone(a),
    }
}

/// Weighted pool-adjacent-violators. Minimises `Σ w (y − z)^2` over nondecreasing `z`.
pub fn weighted_pav(y: &[f64], w: &[f64]) -> Result<Vec<f64>> {
    if y.len() != w.len() || y.is_empty() {
        return Err(TraceError::Shape(
            "weighted PAV needs equal non-empty lengths".into(),
        ));
    }
    if w.iter().any(|v| !(*v > 0.0) || !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return Err(TraceError::InvalidArgument(
            "PAV weights must be positive and finite".into(),
        ));
    }
    struct Block {
        sw: f64,
        w: f64,
        start: usize,
        end: usize,
    }
    let mut blocks: Vec<Block> = Vec::with_capacity(y.len());
    for i in 0..y.len() {
        blocks.push(Block {
            sw: w[i] * y[i],
            w: w[i],
            start: i,
            end: i + 1,
        });
        while blocks.len() >= 2 {
            let n = blocks.len();
            let left = blocks[n - 2].sw / blocks[n - 2].w;
            let right = blocks[n - 1].sw / blocks[n - 1].w;
            if left <= right {
                break;
            }
            let b2 = blocks.pop().unwrap();
            let b1 = blocks.pop().unwrap();
            blocks.push(Block {
                sw: b1.sw + b2.sw,
                w: b1.w + b2.w,
                start: b1.start,
                end: b2.end,
            });
        }
    }
    let mut out = vec![0.0; y.len()];
    for b in blocks {
        let m = b.sw / b.w;
        for i in b.start..b.end {
            out[i] = m;
        }
    }
    Ok(out)
}

/// `√(Σ w (a − b)^2)`.
pub fn wnorm(a: &[f64], b: &[f64], w: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .zip(w)
        .map(|((x, y), wi)| wi * (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Hosking unbiased sample L-moments `λ_1, …, λ_R` via probability-weighted moments.
/// `x` is copied and sorted. Requires `x.len() >= r`.
pub fn hosking_lmoments(x: &[f64], r: usize) -> Result<Vec<f64>> {
    if r == 0 {
        return Err(TraceError::InvalidArgument(
            "L-moment order must be positive".into(),
        ));
    }
    if x.len() < r {
        return Err(TraceError::InvalidArgument(
            "need at least r observations for L-moment r".into(),
        ));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(TraceError::InvalidArgument(
            "L-moments need finite samples".into(),
        ));
    }
    let mut xs = x.to_vec();
    xs.sort_by(f64::total_cmp);
    let m = xs.len();
    let mut b = vec![0.0; r];
    for (i, &xi) in xs.iter().enumerate() {
        let j = (i + 1) as f64;
        let mut wt = 1.0;
        b[0] += xi;
        for ord in 1..r {
            if i < ord {
                wt = 0.0;
            } else {
                wt *= (j - ord as f64) / (m as f64 - ord as f64);
            }
            b[ord] += xi * wt;
        }
    }
    for bi in &mut b {
        *bi /= m as f64;
    }
    let mut lam = vec![0.0; r];
    for ord in 0..r {
        let mut s = 0.0;
        for k in 0..=ord {
            s += pstar_coef(ord, k) * b[k];
        }
        lam[ord] = s;
    }
    Ok(lam)
}

/// Hosking coefficient `p*_{r,k} = (−1)^{r−k} C(r, k) C(r+k, k)`.
pub fn pstar_coef(r: usize, k: usize) -> f64 {
    if k > r {
        return 0.0;
    }
    let sign = if (r - k).is_multiple_of(2) { 1.0 } else { -1.0 };
    sign * binom(r, k) * binom(r + k, k)
}

fn binom(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut c = 1.0;
    for i in 0..k {
        c *= (n - i) as f64;
        c /= (i + 1) as f64;
    }
    c
}

/// Power-basis coefficients of `Σ c_k P*_k`, degree `< c.len()`, low order first.
pub fn expansion_to_power(c: &[f64]) -> Vec<f64> {
    let kmax = c.len();
    let mut power = vec![0.0; kmax];
    for (r, &cr) in c.iter().enumerate() {
        for k in 0..=r {
            power[k] += cr * pstar_coef(r, k);
        }
    }
    power
}

/// Bernstein coefficients of a polynomial given by power coefficients, degree `deg`
/// (the vector length is `deg + 1`; higher input powers are ignored).
pub fn power_to_bernstein(power: &[f64], deg: usize) -> Vec<f64> {
    let mut gamma = vec![0.0; deg + 1];
    for (j, &pj) in power.iter().enumerate() {
        if j > deg {
            break;
        }
        for k in j..=deg {
            gamma[k] += pj * binom(k, j) / binom(deg, j);
        }
    }
    gamma
}

/// Bernstein coefficients of `Q'` after degree elevation `elevation`.
/// `Q = Σ a_k e_k`. Returns coefficients of degree `K_fit − 2 + elevation`
/// (empty if `K_fit < 2`).
pub fn derivative_bernstein(a: &[f64], elevation: usize) -> Vec<f64> {
    if a.len() < 2 {
        return Vec::new();
    }
    let c = a_to_c(a);
    let power = expansion_to_power(&c);
    let mut dpower = vec![0.0; power.len().saturating_sub(1)];
    for (r, &pr) in power.iter().enumerate().skip(1) {
        dpower[r - 1] = pr * r as f64;
    }
    let deg0 = a.len() - 2;
    let deg = deg0 + elevation;
    if dpower.len() < deg + 1 {
        dpower.resize(deg + 1, 0.0);
    }
    power_to_bernstein(&dpower, deg)
}

/// True when every Bernstein coefficient of `Q'` is nonnegative.
pub fn bernstein_nonnegative(a: &[f64], elevation: usize) -> bool {
    if a.len() < 2 {
        return true;
    }
    derivative_bernstein(a, elevation)
        .iter()
        .all(|g| *g >= -1e-10)
}

/// Minimum of `Q'` on a uniform grid of `n` points in `[0, 1]`.
pub fn min_derivative_grid(a: &[f64], n: usize) -> f64 {
    let n = n.max(2);
    let mut m = f64::INFINITY;
    for i in 0..n {
        let u = i as f64 / (n - 1) as f64;
        m = m.min(eval_a_deriv(a, u));
    }
    m
}

/// Probabilists' Hermite `He_m(z)` for `m < out.len()`, via `He_{n+1} = z He_n − n He_{n−1}`.
pub fn hermite_he(z: f64, n_max: usize) -> Vec<f64> {
    let mut h = vec![0.0; n_max];
    if n_max == 0 {
        return h;
    }
    h[0] = 1.0;
    if n_max == 1 {
        return h;
    }
    h[1] = z;
    for m in 1..n_max - 1 {
        h[m + 1] = z * h[m] - m as f64 * h[m - 1];
    }
    h
}

fn factorial(m: usize) -> f64 {
    let mut f = 1.0;
    for i in 1..=m {
        f *= i as f64;
    }
    f
}

/// Inner-product matrix `G[k, m] = ⟨e_k, r_m⟩` from the fitting basis into an
/// orthonormal reporting basis, computed on a fine Gauss–Legendre grid.
/// `kind`: 0 Legendre (the first `k_report` fitting channels), 1 Hermite-in-z,
/// 2 reference-adapted (`reference` is a quantile curve sampled on the fine grid;
/// pass an empty slice to use `z` itself).
pub fn report_matrix(
    k_fit: usize,
    k_report: usize,
    kind: u8,
    reference: &[f64],
) -> Result<Vec<f64>> {
    if k_report == 0 || k_report > k_fit {
        return Err(TraceError::InvalidArgument(
            "K_report must lie in 1..=K_fit".into(),
        ));
    }
    if kind == 0 {
        let mut g = vec![0.0; k_fit * k_report];
        for m in 0..k_report {
            g[m * k_report + m] = 1.0;
        }
        return Ok(g);
    }
    let n_fine = 1024.max(k_fit * 4);
    let gl = gauss_legendre_01(n_fine)?;
    let mut columns = vec![vec![0.0; n_fine]; k_report];
    if kind == 1 {
        if k_report > 8 {
            return Err(TraceError::InvalidArgument(
                "Hermite-in-z reporting supports at most 8 channels".into(),
            ));
        }
        for (t, &u) in gl.nodes.iter().enumerate() {
            let z = super::basis::inv_norm(u.clamp(1e-12, 1.0 - 1e-12))?;
            let h = hermite_he(z, k_report);
            for m in 0..k_report {
                columns[m][t] = h[m] / factorial(m).sqrt();
            }
        }
    } else if kind == 2 {
        let mut raw = if reference.len() == n_fine {
            reference.to_vec()
        } else if reference.is_empty() {
            let mut z = vec![0.0; n_fine];
            for (t, &u) in gl.nodes.iter().enumerate() {
                z[t] = super::basis::inv_norm(u.clamp(1e-12, 1.0 - 1e-12))?;
            }
            z
        } else {
            return Err(TraceError::Shape(
                "reference curve must be empty or match the fine grid".into(),
            ));
        };
        gram_schmidt_columns(&mut columns, &gl.weights, &mut raw, k_fit)?;
    } else {
        return Err(TraceError::InvalidArgument(
            "reporting basis must be legendre, hermite, or reference".into(),
        ));
    }
    let mut g = vec![0.0; k_fit * k_report];
    for (t, &u) in gl.nodes.iter().enumerate() {
        let (e, _) = orthonormal_at(u, k_fit);
        let w = gl.weights[t];
        for k in 0..k_fit {
            for m in 0..k_report {
                g[k * k_report + m] += w * e[k] * columns[m][t];
            }
        }
    }
    Ok(g)
}

fn gram_schmidt_columns(
    columns: &mut [Vec<f64>],
    w: &[f64],
    reference: &mut [f64],
    k_fit: usize,
) -> Result<()> {
    let n = w.len();
    let k_report = columns.len();
    // e_0 = 1
    for t in 0..n {
        columns[0][t] = 1.0;
    }
    if k_report == 1 {
        return Ok(());
    }
    // Centre the reference against e_0 and normalise.
    let mean: f64 = reference.iter().zip(w).map(|(v, wi)| wi * v).sum();
    for v in reference.iter_mut() {
        *v -= mean;
    }
    let norm = wnorm(reference, &vec![0.0; n], w);
    if norm < 1e-14 {
        return Err(TraceError::Numerical(
            "reference curve has no variation after centring".into(),
        ));
    }
    for t in 0..n {
        columns[1][t] = reference[t] / norm;
    }
    let gl = gauss_legendre_01(n)?;
    for m in 2..k_report {
        let src_k = m.min(k_fit - 1);
        for (t, &u) in gl.nodes.iter().enumerate() {
            let (e, _) = orthonormal_at(u, src_k + 1);
            columns[m][t] = e[src_k];
        }
        for prev in 0..m {
            let proj: f64 = columns[m]
                .iter()
                .zip(columns[prev].iter())
                .zip(w)
                .map(|((a, b), wi)| wi * a * b)
                .sum();
            for t in 0..n {
                let b = columns[prev][t];
                columns[m][t] -= proj * b;
            }
        }
        let norm = wnorm(&columns[m], &vec![0.0; n], w);
        if norm < 1e-12 {
            return Err(TraceError::Numerical(
                "Gram–Schmidt produced a zero reporting direction".into(),
            ));
        }
        for v in &mut columns[m] {
            *v /= norm;
        }
    }
    Ok(())
}

/// Project coefficient vector `a` (length `k_fit`) through `G` (`k_fit × k_report`).
pub fn apply_report(a: &[f64], g: &[f64], k_report: usize) -> Vec<f64> {
    let mut out = vec![0.0; k_report];
    for (k, &ak) in a.iter().enumerate() {
        for m in 0..k_report {
            out[m] += ak * g[k * k_report + m];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gl_two_point_and_pstar_endpoint() {
        let gl = gauss_legendre_01(2).unwrap();
        let s: f64 = gl.weights.iter().sum();
        assert!((s - 1.0).abs() < 1e-14);
        let (p, _) = pstar_all(1.0, 6);
        for v in p {
            assert!((v - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn gram_is_identity() {
        let b = QuadBasis::new(8, 6).unwrap();
        assert!(b.gram_error() < 1e-12);
    }

    #[test]
    fn higher_channels_integrate_to_zero_on_the_nodes() {
        let k = 8;
        let b = QuadBasis::new(12, k).unwrap();
        for c in 0..k {
            let mut s = 0.0;
            for t in 0..b.nodes.len() {
                s += b.weights[t] * b.e[t * k + c];
            }
            if c == 0 {
                assert!((s - 1.0).abs() < 1e-12, "e0 integral {s}");
            } else {
                assert!(s.abs() < 1e-12, "e{c} integral {s}");
            }
        }
    }

    #[test]
    fn parseval_pair() {
        let b = QuadBasis::new(8, 4).unwrap();
        let a = [0.3, -0.2, 0.05, 0.01];
        let mut quad = 0.0;
        for t in 0..b.n_u() {
            let mut q = 0.0;
            for k in 0..4 {
                q += a[k] * b.e[t * 4 + k];
            }
            quad += b.weights[t] * q * q;
        }
        let exact: f64 = a.iter().map(|v| v * v).sum();
        assert!((quad - exact).abs() <= 1e-10 * exact.max(1.0));
    }

    #[test]
    fn pav_respects_weights_and_order() {
        let y = [0.0, 2.0, 1.0, 3.0];
        let w = [1.0, 1.0, 1.0, 1.0];
        let z = weighted_pav(&y, &w).unwrap();
        for i in 1..z.len() {
            assert!(z[i] + 1e-12 >= z[i - 1]);
        }
        let y2 = [0.0, 1.0, 3.0];
        let z2 = weighted_pav(&y2, &[1.0, 1.0, 1.0]).unwrap();
        assert!((z2[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn exponential_tau3_and_k4_margin() {
        // Population L-moments of a unit exponential: 1, 1/2, 1/6, 1/12.
        let lam = [1.0, 0.5, 1.0 / 6.0, 1.0 / 12.0];
        let a = lambda_to_a(&lam);
        assert!(k3_non_representable(&a[..3]));
        assert!(k3_monotone(&lambda_to_a(&[1.0, 0.5, 0.05])));
        let min_d = min_derivative_grid(&a, 2001);
        assert!(min_d > 0.4);
    }

    #[test]
    fn hosking_matches_population_uniform() {
        // A long uniform sample: λ2 → 1/6.
        let mut x = Vec::new();
        for i in 0..4000 {
            x.push((i as f64 + 0.5) / 4000.0);
        }
        let lam = hosking_lmoments(&x, 3).unwrap();
        assert!((lam[0] - 0.5).abs() < 1e-12);
        assert!((lam[1] - 1.0 / 6.0).abs() < 1e-4);
    }

    #[test]
    fn hermite_norm_and_legendre_report_is_identity() {
        let g = report_matrix(4, 4, 0, &[]).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                let target = if i == j { 1.0 } else { 0.0 };
                assert!((g[i * 4 + j] - target).abs() < 1e-15);
            }
        }
        let h = hermite_he(0.5, 4);
        // He0=1, He1=z, He2=z^2-1, He3=z^3-3z
        assert!((h[2] - (0.25 - 1.0)).abs() < 1e-12);
        assert!((h[3] - (0.125 - 1.5)).abs() < 1e-12);
    }
}
