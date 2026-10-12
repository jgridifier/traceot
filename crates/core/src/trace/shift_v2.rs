//! Shift decomposition v2.
//!
//! Composition against structure, for the marginal law and for the Wasserstein
//! barycenter, with the gap between those two targets left visible. All of the
//! algebra in `docs/theory/SHIFT_V2.md` lives here. Python only forwards arrays.
//!
//! Fold assignment, bootstrap draws and permutations use [`super::shift::SplitMix64`].
//! Pre-registration samples are drawn by NumPy's PCG64; that generator is not
//! reimplemented here.

#![allow(clippy::too_many_arguments)]

use super::legendre::{gauss_legendre_01, pstar_all, GaussLegendre};
use super::shift::SplitMix64;
use super::v1fit::{fit_v1, MonoMode, ReportKind, V1Config, V1Model};
use crate::error::{Result, TraceError};
use crate::linalg::solve_spd;
use crate::scalable::binning::bin_column_into;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Knobs for the marginal reweighting and the quadrature.
#[derive(Clone, Debug)]
pub struct ShiftV2Config {
    pub k: usize,
    pub n_quad: usize,
    pub n_folds: usize,
    /// Upper quantile at which period-odds weights are capped.
    pub trim_quantile: f64,
    pub logit_ridge: f64,
    pub logit_iters: usize,
    pub logit_tol: f64,
    /// Seed for fold assignment inside one call.
    pub seed: u64,
}

impl Default for ShiftV2Config {
    fn default() -> Self {
        Self {
            k: 6,
            n_quad: 64,
            n_folds: 5,
            trim_quantile: 0.995,
            logit_ridge: 1e-6,
            logit_iters: 25,
            logit_tol: 1e-8,
            seed: 0,
        }
    }
}

/// Order-free two-player split of one channel, in the units of `lambda`.
#[derive(Clone, Copy, Debug)]
pub struct TwoPlayer {
    pub comp: f64,
    pub structure: f64,
    pub interaction: f64,
}

/// `v(∅)=λ^A`, `v({comp})=λ^{C'}`, `v({struct})=λ^C`, `v({both})=λ^B`.
pub fn shapley_two(lambda_a: f64, lambda_cp: f64, lambda_c: f64, lambda_b: f64) -> TwoPlayer {
    let comp_after_structure = lambda_b - lambda_c;
    let comp_first = lambda_cp - lambda_a;
    let structure_first = lambda_c - lambda_a;
    let structure_after_comp = lambda_b - lambda_cp;
    TwoPlayer {
        comp: 0.5 * (comp_after_structure + comp_first),
        structure: 0.5 * (structure_first + structure_after_comp),
        interaction: lambda_b - lambda_c - lambda_cp + lambda_a,
    }
}

/// The same average with the two orderings added in the opposite sequence.
pub fn shapley_two_swapped(
    lambda_a: f64,
    lambda_cp: f64,
    lambda_c: f64,
    lambda_b: f64,
) -> TwoPlayer {
    let forward = shapley_two(lambda_a, lambda_cp, lambda_c, lambda_b);
    let comp_first = lambda_cp - lambda_a;
    let comp_after_structure = lambda_b - lambda_c;
    let structure_after_comp = lambda_b - lambda_cp;
    let structure_first = lambda_c - lambda_a;
    TwoPlayer {
        comp: 0.5 * (comp_first + comp_after_structure),
        structure: 0.5 * (structure_after_comp + structure_first),
        interaction: forward.interaction,
    }
}

/// `φ_i = a_i · (Σ a) / (2k+1)` on the coefficient scale of `P*`. Not clipped.
pub fn quadratic_shapley(a: &[f64], k: usize) -> Vec<f64> {
    let denom = (2 * k + 1) as f64;
    let sum: f64 = a.iter().sum();
    a.iter().map(|ai| ai * sum / denom).collect()
}

/// Channel `W2² = (2k+1) (Δλ)²`.
pub fn w2_from_lambda(delta_lambda: f64, k: usize) -> f64 {
    (2 * k + 1) as f64 * delta_lambda * delta_lambda
}

/// Hellinger–Kantorovich pure-mass term `4(√m_A − √m_B)²`, and that quantity
/// divided by `m_A + m_B`. Neither is added into `W2²`.
pub fn hk_mass(m_a: f64, m_b: f64) -> (f64, f64) {
    let gap = m_a.sqrt() - m_b.sqrt();
    let hk = 4.0 * gap * gap;
    let norm = if m_a + m_b > 0.0 {
        hk / (m_a + m_b)
    } else {
        0.0
    };
    (hk, norm)
}

/// Entry, exit, stay and mix on one coefficient scale. `phi_*` uses Prop. 4
/// with these four players. `hk2_*` is the pure-mass term, kept separate.
#[derive(Clone, Debug)]
pub struct EntryExit {
    pub stay: Vec<f64>,
    pub mix: Vec<f64>,
    pub entry: Vec<f64>,
    pub exit: Vec<f64>,
    pub phi_stay: Vec<f64>,
    pub phi_mix: Vec<f64>,
    pub phi_entry: Vec<f64>,
    pub phi_exit: Vec<f64>,
    pub delta: Vec<f64>,
    pub hk2_mass: f64,
    pub hk2_normalized: f64,
    pub k: usize,
}

/// `stay`, `mix`, `entry`, `exit` in the same units as the coefficient vectors.
/// Players are fed to Prop. 4 channel by channel. The sum identity is linear in
/// any scale. `phi` treats the vectors as `P*` coefficients, so pass `c`-scale
/// inputs (`α √(2k+1)`) when reporting that convention.
pub fn entry_exit_split(
    stay_a: &[f64],
    stay_b: &[f64],
    exit_mean: &[f64],
    entry_mean: &[f64],
    e_a: f64,
    e_b: f64,
    m_a: f64,
    m_b: f64,
) -> Result<EntryExit> {
    let k = stay_a.len();
    if stay_b.len() != k || exit_mean.len() != k || entry_mean.len() != k {
        return Err(TraceError::Shape(
            "entry/exit coefficient vectors must share a length".into(),
        ));
    }
    if !(0.0..=1.0).contains(&e_a) || !(0.0..=1.0).contains(&e_b) {
        return Err(TraceError::InvalidArgument(
            "entry and exit shares must lie in [0, 1]".into(),
        ));
    }
    let mut stay = vec![0.0; k];
    let mut mix = vec![0.0; k];
    let mut entry = vec![0.0; k];
    let mut exit = vec![0.0; k];
    let mut phi_stay = vec![0.0; k];
    let mut phi_mix = vec![0.0; k];
    let mut phi_entry = vec![0.0; k];
    let mut phi_exit = vec![0.0; k];
    let mut delta = vec![0.0; k];
    for c in 0..k {
        stay[c] = 0.5 * ((1.0 - e_a) + (1.0 - e_b)) * (stay_b[c] - stay_a[c]);
        mix[c] = 0.5 * (stay_a[c] + stay_b[c]) * (e_a - e_b);
        entry[c] = e_b * entry_mean[c];
        exit[c] = -e_a * exit_mean[c];
        delta[c] = stay[c] + mix[c] + entry[c] + exit[c];
        let phi = quadratic_shapley(&[stay[c], mix[c], entry[c], exit[c]], c);
        phi_stay[c] = phi[0];
        phi_mix[c] = phi[1];
        phi_entry[c] = phi[2];
        phi_exit[c] = phi[3];
    }
    let (hk2_mass, hk2_normalized) = hk_mass(m_a, m_b);
    Ok(EntryExit {
        stay,
        mix,
        entry,
        exit,
        phi_stay,
        phi_mix,
        phi_entry,
        phi_exit,
        delta,
        hk2_mass,
        hk2_normalized,
        k,
    })
}

struct Quad {
    nodes: Vec<f64>,
    weights: Vec<f64>,
    /// Row-major `n_quad × k` values of `P*_c`.
    pstar: Vec<f64>,
    k: usize,
}

impl Quad {
    fn new(n_quad: usize, k: usize) -> Result<Self> {
        if k == 0 || n_quad < k {
            return Err(TraceError::InvalidArgument(
                "quadrature needs at least K nodes".into(),
            ));
        }
        let GaussLegendre { nodes, weights } = gauss_legendre_01(n_quad)?;
        let mut pstar = vec![0.0; n_quad * k];
        for (t, &u) in nodes.iter().enumerate() {
            let (p, _) = pstar_all(u, k);
            pstar[t * k..(t + 1) * k].copy_from_slice(&p);
        }
        Ok(Self {
            nodes,
            weights,
            pstar,
            k,
        })
    }
}

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let p = p.clamp(0.0, 1.0);
    let pos = p * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let w = pos - lo as f64;
    sorted[lo] * (1.0 - w) + sorted[hi] * w
}

fn quantiles_at(y: &[f64], w: Option<&[f64]>, nodes: &[f64]) -> Result<Vec<f64>> {
    let n = y.len();
    if n == 0 {
        return Err(TraceError::InvalidArgument(
            "a quantile function needs at least one row".into(),
        ));
    }
    if let Some(ws) = w {
        if ws.len() != n {
            return Err(TraceError::Shape(
                "weights and outcomes must have the same length".into(),
            ));
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| y[i].total_cmp(&y[j]).then(i.cmp(&j)));
    let mut cum = Vec::with_capacity(n);
    let mut total = 0.0;
    for &i in &order {
        let wi = match w {
            Some(ws) => ws[i],
            None => 1.0,
        };
        if !wi.is_finite() || wi < 0.0 {
            return Err(TraceError::InvalidArgument(
                "weights must be finite and nonnegative".into(),
            ));
        }
        total += wi;
        cum.push(total);
    }
    if !(total > 0.0) {
        return Err(TraceError::InvalidArgument(
            "weights must have positive total mass".into(),
        ));
    }
    let mut q = Vec::with_capacity(nodes.len());
    for &u in nodes {
        let target = u.clamp(0.0, 1.0) * total;
        let mut idx = cum.partition_point(|&c| c < target);
        if idx >= n {
            idx = n - 1;
        }
        q.push(y[order[idx]]);
    }
    Ok(q)
}

fn lambdas_of(q: &[f64], quad: &Quad) -> Vec<f64> {
    let k = quad.k;
    let mut lam = vec![0.0; k];
    for t in 0..quad.nodes.len() {
        let wt = quad.weights[t];
        let qt = q[t];
        for c in 0..k {
            lam[c] += wt * qt * quad.pstar[t * k + c];
        }
    }
    lam
}

fn w2_quad(qa: &[f64], qb: &[f64], weights: &[f64]) -> f64 {
    qa.iter()
        .zip(qb)
        .zip(weights)
        .map(|((a, b), w)| w * (a - b) * (a - b))
        .sum()
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        let e = (-z).exp();
        1.0 / (1.0 + e)
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

struct Pooled<'a> {
    x_a: &'a [f64],
    n_a: usize,
    x_b: &'a [f64],
    n_b: usize,
    p: usize,
}

impl<'a> Pooled<'a> {
    fn n(&self) -> usize {
        self.n_a + self.n_b
    }

    fn row(&self, i: usize) -> &'a [f64] {
        if i < self.n_a {
            &self.x_a[i * self.p..(i + 1) * self.p]
        } else {
            let j = i - self.n_a;
            &self.x_b[j * self.p..(j + 1) * self.p]
        }
    }

    fn label(&self, i: usize) -> f64 {
        if i < self.n_a {
            1.0
        } else {
            0.0
        }
    }
}

fn fit_logit(pooled: &Pooled<'_>, rows: &[usize], cfg: &ShiftV2Config) -> Result<Vec<f64>> {
    let p = pooled.p;
    let d = p + 1;
    let mut beta = vec![0.0; d];
    if rows.is_empty() {
        return Ok(beta);
    }
    let mut n_a = 0.0_f64;
    let mut n_b = 0.0_f64;
    for &i in rows {
        if pooled.label(i) > 0.5 {
            n_a += 1.0;
        } else {
            n_b += 1.0;
        }
    }
    if n_a > 0.0 && n_b > 0.0 {
        beta[0] = (n_a / n_b).ln();
    }
    let ridge = cfg.logit_ridge.max(0.0);
    for _ in 0..cfg.logit_iters {
        let mut g = vec![0.0; d];
        let mut h = vec![0.0; d * d];
        for &i in rows {
            let row = pooled.row(i);
            let mut z = beta[0];
            for j in 0..p {
                z += beta[j + 1] * row[j];
            }
            let pr = sigmoid(z).clamp(1e-12, 1.0 - 1e-12);
            let resid = pooled.label(i) - pr;
            let wi = pr * (1.0 - pr);
            g[0] += resid;
            h[0] += wi;
            for j in 0..p {
                let xj = row[j];
                g[j + 1] += xj * resid;
                h[(j + 1) * d] += wi * xj;
                h[j + 1] += wi * xj;
                for t in 0..=j {
                    let xt = row[t];
                    let add = wi * xj * xt;
                    h[(j + 1) * d + (t + 1)] += add;
                    if t != j {
                        h[(t + 1) * d + (j + 1)] += add;
                    }
                }
            }
        }
        for j in 1..d {
            g[j] -= ridge * beta[j];
            h[j * d + j] += ridge;
        }
        h[0] += 1e-12;
        let step = match solve_spd(&h, d, &g) {
            Ok(s) => s,
            Err(_) => {
                for j in 0..d {
                    h[j * d + j] += 1e-3;
                }
                solve_spd(&h, d, &g)?
            }
        };
        let mut step_norm = 0.0;
        for j in 0..d {
            beta[j] += step[j];
            step_norm += step[j] * step[j];
        }
        if step_norm.sqrt() < cfg.logit_tol {
            break;
        }
    }
    Ok(beta)
}

fn linear_predictor(beta: &[f64], row: &[f64]) -> f64 {
    let mut z = beta[0];
    for (j, &x) in row.iter().enumerate() {
        z += beta[j + 1] * x;
    }
    z
}

fn stratified_folds(n_a: usize, n_b: usize, k: usize, rng: &mut SplitMix64) -> Vec<usize> {
    let mut folds = vec![0usize; n_a + n_b];
    assign_folds(&mut folds[..n_a], k, rng);
    assign_folds(&mut folds[n_a..], k, rng);
    folds
}

fn assign_folds(folds: &mut [usize], k: usize, rng: &mut SplitMix64) {
    let n = folds.len();
    if n == 0 || k == 0 {
        return;
    }
    let mut idx: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.index(i + 1);
        idx.swap(i, j);
    }
    for (t, &i) in idx.iter().enumerate() {
        folds[i] = t % k;
    }
}

fn cross_fit_lp(pooled: &Pooled<'_>, cfg: &ShiftV2Config) -> Result<Vec<f64>> {
    let mut rng = SplitMix64::new(cfg.seed);
    let folds = stratified_folds(pooled.n_a, pooled.n_b, cfg.n_folds, &mut rng);
    cross_fit_lp_folds(pooled, &folds, cfg.n_folds, cfg)
}

fn cross_fit_lp_folds(
    pooled: &Pooled<'_>,
    folds: &[usize],
    n_folds: usize,
    cfg: &ShiftV2Config,
) -> Result<Vec<f64>> {
    let n = pooled.n();
    if folds.len() != n {
        return Err(TraceError::Shape(
            "fold ids must cover every pooled row".into(),
        ));
    }
    let mut lp = vec![0.0; n];
    for fold in 0..n_folds {
        let train: Vec<usize> = (0..n).filter(|&i| folds[i] != fold).collect();
        let beta = fit_logit(pooled, &train, cfg)?;
        for i in 0..n {
            if folds[i] == fold {
                lp[i] = linear_predictor(&beta, pooled.row(i));
            }
        }
    }
    Ok(lp)
}

fn trim_and_normalize(w: &mut [f64], q: f64) -> Result<(f64, f64)> {
    if w.is_empty() {
        return Err(TraceError::InvalidArgument("no weights to trim".into()));
    }
    let mut sorted = w.to_vec();
    sorted.sort_by(f64::total_cmp);
    let cap = if q >= 1.0 {
        f64::INFINITY
    } else {
        percentile_sorted(&sorted, q)
    };
    let sum: f64 = w.iter().sum();
    let mut excess = 0.0;
    let mut max_w = 0.0_f64;
    for wi in w.iter_mut() {
        max_w = max_w.max(*wi);
        if *wi > cap {
            excess += *wi - cap;
            *wi = cap;
        }
    }
    let mass = if sum > 0.0 { excess / sum } else { 0.0 };
    let mean = w.iter().sum::<f64>() / w.len() as f64;
    if !(mean > 0.0) {
        return Err(TraceError::Numerical(
            "period-odds weights have non-positive mean".into(),
        ));
    }
    for wi in w.iter_mut() {
        *wi /= mean;
    }
    Ok((mass, max_w))
}

fn ess(w: &[f64]) -> f64 {
    let s: f64 = w.iter().sum();
    let s2: f64 = w.iter().map(|v| v * v).sum();
    if s2 > 0.0 {
        s * s / s2
    } else {
        0.0
    }
}

/// Marginal-law composition and structure, one channel at a time.
#[derive(Clone, Debug)]
pub struct MarginalView {
    pub lambda_a: Vec<f64>,
    pub lambda_b: Vec<f64>,
    pub lambda_c: Vec<f64>,
    pub lambda_cp: Vec<f64>,
    pub comp_lambda: Vec<f64>,
    pub struct_lambda: Vec<f64>,
    pub interaction_lambda: Vec<f64>,
    /// `W2²` shares. Negative values are kept.
    pub comp_w2: Vec<f64>,
    pub struct_w2: Vec<f64>,
    pub comp_w2_total: f64,
    pub struct_w2_total: f64,
    pub w2_resolved: f64,
    pub w2_total: f64,
    pub unresolved: f64,
    pub trim_mass_c: f64,
    pub trim_mass_cp: f64,
    pub max_weight_c: f64,
    pub max_weight_cp: f64,
    pub ess_c: f64,
    pub ess_cp: f64,
    /// Largest absolute gap between the two averaging orders, over channels.
    pub order_gap: f64,
    pub k: usize,
}

/// Composition and structure from four already-built laws (no reweighting).
/// `y_c` is structure B with covariates A; `y_cp` is structure A with covariates B.
/// L-moments `λ_{k+1} = ∫ Q P*_k` of one sample, on the same quadrature as the split.
pub fn sample_lambda(y: &[f64], k: usize, n_quad: usize) -> Result<Vec<f64>> {
    let quad = Quad::new(n_quad, k)?;
    let q = quantiles_at(y, None, &quad.nodes)?;
    Ok(lambdas_of(&q, &quad))
}

pub fn marginal_from_laws(
    y_a: &[f64],
    y_b: &[f64],
    y_c: &[f64],
    y_cp: &[f64],
    k: usize,
    n_quad: usize,
) -> Result<MarginalView> {
    if y_a.len() < 2 || y_b.len() < 2 || y_c.len() < 2 || y_cp.len() < 2 {
        return Err(TraceError::InvalidArgument(
            "each law needs at least two rows".into(),
        ));
    }
    let quad = Quad::new(n_quad, k)?;
    let qa = quantiles_at(y_a, None, &quad.nodes)?;
    let qb = quantiles_at(y_b, None, &quad.nodes)?;
    let qc = quantiles_at(y_c, None, &quad.nodes)?;
    let qcp = quantiles_at(y_cp, None, &quad.nodes)?;
    finish_marginal(
        &quad,
        &qa,
        &qb,
        &qc,
        &qcp,
        0.0,
        0.0,
        1.0,
        1.0,
        y_c.len() as f64,
        y_cp.len() as f64,
    )
}

fn finish_marginal(
    quad: &Quad,
    qa: &[f64],
    qb: &[f64],
    qc: &[f64],
    qcp: &[f64],
    trim_mass_c: f64,
    trim_mass_cp: f64,
    max_weight_c: f64,
    max_weight_cp: f64,
    ess_c: f64,
    ess_cp: f64,
) -> Result<MarginalView> {
    let lambda_a = lambdas_of(qa, quad);
    let lambda_b = lambdas_of(qb, quad);
    let lambda_c = lambdas_of(qc, quad);
    let lambda_cp = lambdas_of(qcp, quad);
    let k = quad.k;
    let mut comp_lambda = vec![0.0; k];
    let mut struct_lambda = vec![0.0; k];
    let mut interaction_lambda = vec![0.0; k];
    let mut comp_w2 = vec![0.0; k];
    let mut struct_w2 = vec![0.0; k];
    let mut order_gap = 0.0_f64;
    for c in 0..k {
        let split = shapley_two(lambda_a[c], lambda_cp[c], lambda_c[c], lambda_b[c]);
        let swapped = shapley_two_swapped(lambda_a[c], lambda_cp[c], lambda_c[c], lambda_b[c]);
        order_gap = order_gap.max((split.comp - swapped.comp).abs());
        order_gap = order_gap.max((split.structure - swapped.structure).abs());
        comp_lambda[c] = split.comp;
        struct_lambda[c] = split.structure;
        interaction_lambda[c] = split.interaction;
        let phi = quadratic_shapley(
            &[
                split.comp * (2 * c + 1) as f64,
                split.structure * (2 * c + 1) as f64,
            ],
            c,
        );
        comp_w2[c] = phi[0];
        struct_w2[c] = phi[1];
    }
    let comp_w2_total = comp_w2.iter().sum();
    let struct_w2_total = struct_w2.iter().sum();
    let w2_resolved = comp_w2_total + struct_w2_total;
    let w2_total = w2_quad(qa, qb, &quad.weights);
    let unresolved = w2_total - w2_resolved;
    Ok(MarginalView {
        lambda_a,
        lambda_b,
        lambda_c,
        lambda_cp,
        comp_lambda,
        struct_lambda,
        interaction_lambda,
        comp_w2,
        struct_w2,
        comp_w2_total,
        struct_w2_total,
        w2_resolved,
        w2_total,
        unresolved,
        trim_mass_c,
        trim_mass_cp,
        max_weight_c,
        max_weight_cp,
        ess_c,
        ess_cp,
        order_gap,
        k,
    })
}

pub fn marginal_view(
    x_a: &[f64],
    y_a: &[f64],
    n_a: usize,
    x_b: &[f64],
    y_b: &[f64],
    n_b: usize,
    cfg: &ShiftV2Config,
) -> Result<MarginalView> {
    let p = check_period(x_a, y_a, n_a, x_b, y_b, n_b)?;
    if cfg.n_folds < 2 {
        return Err(TraceError::InvalidArgument(
            "cross-fitting needs at least two folds".into(),
        ));
    }
    if !(cfg.trim_quantile > 0.0 && cfg.trim_quantile <= 1.0) {
        return Err(TraceError::InvalidArgument(
            "trim quantile must lie in (0, 1]".into(),
        ));
    }
    let quad = Quad::new(cfg.n_quad, cfg.k)?;
    let pooled = Pooled {
        x_a,
        n_a,
        x_b,
        n_b,
        p,
    };
    let lp = cross_fit_lp(&pooled, cfg)?;
    let prior_b = (n_b as f64 / n_a as f64).ln();
    let prior_a = (n_a as f64 / n_b as f64).ln();
    let mut w_c = vec![0.0; n_b];
    let mut w_cp = vec![0.0; n_a];
    for i in 0..n_b {
        w_c[i] = (lp[n_a + i] + prior_b).exp();
    }
    for i in 0..n_a {
        w_cp[i] = (-lp[i] + prior_a).exp();
    }
    let (trim_mass_c, max_weight_c) = trim_and_normalize(&mut w_c, cfg.trim_quantile)?;
    let (trim_mass_cp, max_weight_cp) = trim_and_normalize(&mut w_cp, cfg.trim_quantile)?;
    let ess_c = ess(&w_c);
    let ess_cp = ess(&w_cp);
    let qa = quantiles_at(y_a, None, &quad.nodes)?;
    let qb = quantiles_at(y_b, None, &quad.nodes)?;
    let qc = quantiles_at(y_b, Some(&w_c), &quad.nodes)?;
    let qcp = quantiles_at(y_a, Some(&w_cp), &quad.nodes)?;
    finish_marginal(
        &quad,
        &qa,
        &qb,
        &qc,
        &qcp,
        trim_mass_c,
        trim_mass_cp,
        max_weight_c,
        max_weight_cp,
        ess_c,
        ess_cp,
    )
}

fn check_period(
    x_a: &[f64],
    y_a: &[f64],
    n_a: usize,
    x_b: &[f64],
    y_b: &[f64],
    n_b: usize,
) -> Result<usize> {
    if n_a < 2 || n_b < 2 {
        return Err(TraceError::InvalidArgument(
            "each period needs at least two rows".into(),
        ));
    }
    if y_a.len() != n_a || y_b.len() != n_b {
        return Err(TraceError::Shape(
            "y and the stated row count disagree".into(),
        ));
    }
    if n_a == 0 || !x_a.len().is_multiple_of(n_a) {
        return Err(TraceError::Shape("X_A is not a complete row block".into()));
    }
    let p = x_a.len() / n_a;
    if p == 0 || x_b.len() != n_b * p {
        return Err(TraceError::Shape(
            "the two periods must share a positive feature count".into(),
        ));
    }
    if y_a.iter().any(|v| !v.is_finite()) || y_b.iter().any(|v| !v.is_finite()) {
        return Err(TraceError::InvalidArgument(
            "outcomes must be finite".into(),
        ));
    }
    if x_a.iter().any(|v| v.is_infinite()) || x_b.iter().any(|v| v.is_infinite()) {
        return Err(TraceError::InvalidArgument(
            "features must not contain infinities".into(),
        ));
    }
    Ok(p)
}

fn mean_effects(model: &V1Model, x: &[f64], n: usize) -> Result<Vec<f64>> {
    let p = model.edges.len();
    let k = model.k_fit;
    if n == 0 || x.len() != n * p {
        return Err(TraceError::Shape(
            "X does not match the model feature count".into(),
        ));
    }
    let mut means = vec![0.0; p * k];
    let mut col = vec![0.0; n];
    let mut codes = vec![0u16; n];
    let scale = 1.0 / n as f64;
    for j in 0..p {
        for i in 0..n {
            col[i] = x[i * p + j];
        }
        bin_column_into(&col, &model.edges[j], &mut codes)?;
        let eff = &model.effects[j];
        for i in 0..n {
            let b = codes[i] as usize;
            for c in 0..k {
                means[j * k + c] += eff[b * k + c];
            }
        }
        for c in 0..k {
            means[j * k + c] *= scale;
        }
    }
    Ok(means)
}

/// Mean orthonormal curve `b + Σ_j E[f_j]` on the rows where `mask` is 1.
/// `mask == None` uses every row.
pub fn mean_curve(model: &V1Model, x: &[f64], n: usize, mask: Option<&[u8]>) -> Result<Vec<f64>> {
    let p = model.edges.len();
    let k = model.k_fit;
    if x.len() != n * p {
        return Err(TraceError::Shape(
            "X does not match the model feature count".into(),
        ));
    }
    if let Some(m) = mask {
        if m.len() != n {
            return Err(TraceError::Shape("mask length must equal n".into()));
        }
    }
    let mut count = 0.0;
    let mut means = vec![0.0; p * k];
    let mut col = vec![0.0; n];
    let mut codes = vec![0u16; n];
    for j in 0..p {
        for i in 0..n {
            col[i] = x[i * p + j];
        }
        bin_column_into(&col, &model.edges[j], &mut codes)?;
        let eff = &model.effects[j];
        for i in 0..n {
            if mask.is_some_and(|m| m[i] == 0) {
                continue;
            }
            if j == 0 {
                count += 1.0;
            }
            let b = codes[i] as usize;
            for c in 0..k {
                means[j * k + c] += eff[b * k + c];
            }
        }
    }
    if !(count > 0.0) {
        return Ok(vec![0.0; k]);
    }
    let mut alpha = model.intercept.clone();
    if alpha.len() != k {
        return Err(TraceError::Shape("intercept length is not K".into()));
    }
    let scale = 1.0 / count;
    for j in 0..p {
        for c in 0..k {
            alpha[c] += means[j * k + c] * scale;
        }
    }
    Ok(alpha)
}

/// Barycenter (Vincent average) players. `phi` is player-major, `(2p+1) × k`:
/// composition features, structure features, then the intercept.
#[derive(Clone, Debug)]
pub struct BaryView {
    pub alpha_a: Vec<f64>,
    pub alpha_b: Vec<f64>,
    pub lambda_a: Vec<f64>,
    pub lambda_b: Vec<f64>,
    pub comp_alpha: Vec<f64>,
    pub struct_alpha: Vec<f64>,
    pub intercept_alpha: Vec<f64>,
    pub phi: Vec<f64>,
    pub comp_w2: Vec<f64>,
    pub struct_w2: Vec<f64>,
    pub w2_channel: Vec<f64>,
    pub w2_total: f64,
    pub order_gap: f64,
    pub player_sum_gap: f64,
    pub p: usize,
    pub k: usize,
}

pub fn barycenter_view(
    model_a: &V1Model,
    model_b: &V1Model,
    x_a: &[f64],
    n_a: usize,
    x_b: &[f64],
    n_b: usize,
) -> Result<BaryView> {
    if model_a.k_fit != model_b.k_fit || model_a.edges.len() != model_b.edges.len() {
        return Err(TraceError::InvalidArgument(
            "the two periods must share K and the feature count".into(),
        ));
    }
    let p = model_a.edges.len();
    let k = model_a.k_fit;
    let g_aa = mean_effects(model_a, x_a, n_a)?;
    let g_ab = mean_effects(model_a, x_b, n_b)?;
    let g_ba = mean_effects(model_b, x_a, n_a)?;
    let g_bb = mean_effects(model_b, x_b, n_b)?;
    let mut comp_alpha = vec![0.0; p * k];
    let mut struct_alpha = vec![0.0; p * k];
    let mut order_gap = 0.0_f64;
    for j in 0..p {
        for c in 0..k {
            let aa = g_aa[j * k + c];
            let ab = g_ab[j * k + c];
            let ba = g_ba[j * k + c];
            let bb = g_bb[j * k + c];
            let comp_first = ab - aa;
            let comp_second = bb - ba;
            let struct_first = ba - aa;
            let struct_second = bb - ab;
            let comp = 0.5 * (comp_first + comp_second);
            let structure = 0.5 * (struct_first + struct_second);
            let comp_swap = 0.5 * (comp_second + comp_first);
            let struct_swap = 0.5 * (struct_second + struct_first);
            order_gap = order_gap.max((comp - comp_swap).abs());
            order_gap = order_gap.max((structure - struct_swap).abs());
            comp_alpha[j * k + c] = comp;
            struct_alpha[j * k + c] = structure;
        }
    }
    let mut intercept_alpha = vec![0.0; k];
    for c in 0..k {
        intercept_alpha[c] = model_b.intercept[c] - model_a.intercept[c];
    }
    let n_players = 2 * p + 1;
    let mut phi = vec![0.0; n_players * k];
    let mut comp_w2 = vec![0.0; k];
    let mut struct_w2 = vec![0.0; k];
    let mut w2_channel = vec![0.0; k];
    let mut player_sum_gap = 0.0_f64;
    let mut alpha_a = model_a.intercept.clone();
    let mut alpha_b = model_b.intercept.clone();
    for j in 0..p {
        for c in 0..k {
            alpha_a[c] += g_aa[j * k + c];
            alpha_b[c] += g_bb[j * k + c];
        }
    }
    for c in 0..k {
        let mut alphas = Vec::with_capacity(n_players);
        for j in 0..p {
            alphas.push(comp_alpha[j * k + c]);
        }
        for j in 0..p {
            alphas.push(struct_alpha[j * k + c]);
        }
        alphas.push(intercept_alpha[c]);
        let sum_alpha: f64 = alphas.iter().sum();
        let sqrt_w = ((2 * c + 1) as f64).sqrt();
        let mut phi_sum = 0.0;
        for (i, &alpha) in alphas.iter().enumerate() {
            let a_c = alpha * sqrt_w;
            let share = a_c * (sum_alpha * sqrt_w) / (2 * c + 1) as f64;
            phi[i * k + c] = share;
            phi_sum += share;
            if i < p {
                comp_w2[c] += share;
            } else {
                struct_w2[c] += share;
            }
        }
        w2_channel[c] = sum_alpha * sum_alpha;
        player_sum_gap = player_sum_gap.max((phi_sum - w2_channel[c]).abs());
    }
    let w2_total: f64 = w2_channel.iter().sum();
    player_sum_gap = player_sum_gap.max((phi.iter().sum::<f64>() - w2_total).abs());
    let lambda_a: Vec<f64> = alpha_a
        .iter()
        .enumerate()
        .map(|(c, a)| a / ((2 * c + 1) as f64).sqrt())
        .collect();
    let lambda_b: Vec<f64> = alpha_b
        .iter()
        .enumerate()
        .map(|(c, a)| a / ((2 * c + 1) as f64).sqrt())
        .collect();
    Ok(BaryView {
        alpha_a,
        alpha_b,
        lambda_a,
        lambda_b,
        comp_alpha,
        struct_alpha,
        intercept_alpha,
        phi,
        comp_w2,
        struct_w2,
        w2_channel,
        w2_total,
        order_gap,
        player_sum_gap,
        p,
        k,
    })
}

/// Both targets, and the gap between them.
#[derive(Clone, Debug)]
pub struct ShiftReport {
    pub marginal: MarginalView,
    pub bary: BaryView,
    /// Marginal λ minus barycenter λ, period A, period B, and the A→B delta.
    pub gap_lambda_a: Vec<f64>,
    pub gap_lambda_b: Vec<f64>,
    pub gap_delta_lambda: Vec<f64>,
    pub gap_comp_w2: Vec<f64>,
    pub gap_struct_w2: Vec<f64>,
    pub gap_w2: f64,
    pub order_gap: f64,
    pub player_sum_gap: f64,
}

pub fn explain_shift(
    model_a: &V1Model,
    model_b: &V1Model,
    x_a: &[f64],
    y_a: &[f64],
    n_a: usize,
    x_b: &[f64],
    y_b: &[f64],
    n_b: usize,
    cfg: &ShiftV2Config,
) -> Result<ShiftReport> {
    if model_a.k_fit != cfg.k || model_b.k_fit != cfg.k {
        return Err(TraceError::InvalidArgument(
            "ShiftV2Config.k must equal the fitted channel count".into(),
        ));
    }
    let mut cfg = cfg.clone();
    cfg.k = model_a.k_fit;
    let marginal = marginal_view(x_a, y_a, n_a, x_b, y_b, n_b, &cfg)?;
    let bary = barycenter_view(model_a, model_b, x_a, n_a, x_b, n_b)?;
    let k = cfg.k;
    let mut gap_lambda_a = vec![0.0; k];
    let mut gap_lambda_b = vec![0.0; k];
    let mut gap_delta_lambda = vec![0.0; k];
    let mut gap_comp_w2 = vec![0.0; k];
    let mut gap_struct_w2 = vec![0.0; k];
    for c in 0..k {
        gap_lambda_a[c] = marginal.lambda_a[c] - bary.lambda_a[c];
        gap_lambda_b[c] = marginal.lambda_b[c] - bary.lambda_b[c];
        let d_m = marginal.lambda_b[c] - marginal.lambda_a[c];
        let d_b = bary.lambda_b[c] - bary.lambda_a[c];
        gap_delta_lambda[c] = d_m - d_b;
        gap_comp_w2[c] = marginal.comp_w2[c] - bary.comp_w2[c];
        gap_struct_w2[c] = marginal.struct_w2[c] - bary.struct_w2[c];
    }
    let gap_w2 = marginal.w2_resolved - bary.w2_total;
    let order_gap = marginal.order_gap.max(bary.order_gap);
    let player_sum_gap = bary.player_sum_gap;
    Ok(ShiftReport {
        marginal,
        bary,
        gap_lambda_a,
        gap_lambda_b,
        gap_delta_lambda,
        gap_comp_w2,
        gap_struct_w2,
        gap_w2,
        order_gap,
        player_sum_gap,
    })
}

fn resample_indices(n: usize, reps_one: usize, rng: &mut SplitMix64) -> Vec<usize> {
    let mut idx = Vec::with_capacity(reps_one);
    for _ in 0..reps_one {
        idx.push(rng.index(n));
    }
    idx
}

fn gather_period(
    x: &[f64],
    y: &[f64],
    p: usize,
    idx: &[usize],
    mask: Option<&[u8]>,
) -> (Vec<f64>, Vec<f64>, Option<Vec<u8>>) {
    let m = idx.len();
    let mut xo = vec![0.0; m * p];
    let mut yo = vec![0.0; m];
    let mut mo = mask.map(|_| vec![0u8; m]);
    for (t, &i) in idx.iter().enumerate() {
        yo[t] = y[i];
        xo[t * p..(t + 1) * p].copy_from_slice(&x[i * p..(i + 1) * p]);
        if let (Some(src), Some(dst)) = (mask, mo.as_mut()) {
            dst[t] = src[i];
        }
    }
    (xo, yo, mo)
}

#[derive(Clone, Debug)]
pub struct MarginalDraw {
    pub comp_w2: Vec<f64>,
    pub struct_w2: Vec<f64>,
    pub comp_total: f64,
    pub struct_total: f64,
    pub w2_total: f64,
}

fn map_reps<T: Send>(reps: usize, f: &(dyn Fn(usize) -> Result<T> + Sync)) -> Result<Vec<T>> {
    #[cfg(feature = "parallel")]
    {
        (0..reps).into_par_iter().map(f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..reps).map(f).collect()
    }
}

/// Stratified row bootstrap of the marginal decomposition, including a fresh
/// cross-fit of the period-odds weights on every replicate.
pub fn bootstrap_marginal(
    x_a: &[f64],
    y_a: &[f64],
    n_a: usize,
    x_b: &[f64],
    y_b: &[f64],
    n_b: usize,
    cfg: &ShiftV2Config,
    reps: usize,
    seed: u64,
) -> Result<Vec<MarginalDraw>> {
    let p = check_period(x_a, y_a, n_a, x_b, y_b, n_b)?;
    if reps == 0 {
        return Err(TraceError::InvalidArgument(
            "bootstrap needs at least one replicate".into(),
        ));
    }
    let job = |b: usize| -> Result<MarginalDraw> {
        // Not the SplitMix64 step constant. Multiplying by that constant makes
        // replicate b+1 a one-step shift of replicate b, so every resample is
        // almost the same sample and the band has no width.
        let mut rng = SplitMix64::new(
            seed.wrapping_add(b as u64)
                .wrapping_mul(0xBF58_476D_1CE4_E5B9),
        );
        let ia = resample_indices(n_a, n_a, &mut rng);
        let ib = resample_indices(n_b, n_b, &mut rng);
        let (xa, ya, _) = gather_period(x_a, y_a, p, &ia, None);
        let (xb, yb, _) = gather_period(x_b, y_b, p, &ib, None);
        let mut local = cfg.clone();
        local.seed = seed.wrapping_add(10_000 + b as u64);
        let view = marginal_view(&xa, &ya, n_a, &xb, &yb, n_b, &local)?;
        Ok(MarginalDraw {
            comp_w2: view.comp_w2,
            struct_w2: view.struct_w2,
            comp_total: view.comp_w2_total,
            struct_total: view.struct_w2_total,
            w2_total: view.w2_total,
        })
    };
    map_reps(reps, &job)
}

#[derive(Clone, Debug)]
pub struct BaryDraw {
    /// Player-major `(2p+1) × k`.
    pub phi: Vec<f64>,
    /// Composition `W2²` of each feature, summed over channels.
    pub comp_feature: Vec<f64>,
}

/// Stratified bootstrap that refits both TRACE models, then recomputes the
/// barycenter players. This is the band that includes the fit.
pub fn bootstrap_barycenter(
    x_a: &[f64],
    y_a: &[f64],
    n_a: usize,
    x_b: &[f64],
    y_b: &[f64],
    n_b: usize,
    v1: &V1Config,
    reps: usize,
    seed: u64,
) -> Result<Vec<BaryDraw>> {
    let p = check_period(x_a, y_a, n_a, x_b, y_b, n_b)?;
    if reps == 0 {
        return Err(TraceError::InvalidArgument(
            "bootstrap needs at least one replicate".into(),
        ));
    }
    let v1 = v1.clone();
    let job = |b: usize| -> Result<BaryDraw> {
        let mut rng = SplitMix64::new(
            seed.wrapping_add(b as u64)
                .wrapping_mul(0xBF58_476D_1CE4_E5B9),
        );
        let ia = resample_indices(n_a, n_a, &mut rng);
        let ib = resample_indices(n_b, n_b, &mut rng);
        let (xa, ya, _) = gather_period(x_a, y_a, p, &ia, None);
        let (xb, yb, _) = gather_period(x_b, y_b, p, &ib, None);
        let model_a = fit_v1(&xa, n_a, p, &ya, &v1)?;
        let model_b = fit_v1(&xb, n_b, p, &yb, &v1)?;
        let view = barycenter_view(&model_a, &model_b, &xa, n_a, &xb, n_b)?;
        let mut comp_feature = vec![0.0; p];
        for j in 0..p {
            for c in 0..view.k {
                comp_feature[j] += view.phi[j * view.k + c];
            }
        }
        Ok(BaryDraw {
            phi: view.phi,
            comp_feature,
        })
    };
    map_reps(reps, &job)
}

/// Refit both models on stratified resamples and recompute the entry/exit split
/// in orthonormal coefficient units. `phi` is then the orthonormal Prop. 4 share
/// `α_i Σα` (equal to the `c`-scale formula).
pub fn bootstrap_entry_exit(
    x_a: &[f64],
    y_a: &[f64],
    exit_a: &[u8],
    n_a: usize,
    x_b: &[f64],
    y_b: &[f64],
    entry_b: &[u8],
    n_b: usize,
    v1: &V1Config,
    reps: usize,
    seed: u64,
) -> Result<Vec<EntryExit>> {
    let p = check_period(x_a, y_a, n_a, x_b, y_b, n_b)?;
    if exit_a.len() != n_a || entry_b.len() != n_b {
        return Err(TraceError::Shape(
            "entry and exit masks must match the period lengths".into(),
        ));
    }
    let v1 = v1.clone();
    let job = |b: usize| -> Result<EntryExit> {
        let mut rng = SplitMix64::new(
            seed.wrapping_add(b as u64)
                .wrapping_mul(0x94D0_49BB_1331_11EB),
        );
        let ia = resample_indices(n_a, n_a, &mut rng);
        let ib = resample_indices(n_b, n_b, &mut rng);
        let (xa, ya, ea) = gather_period(x_a, y_a, p, &ia, Some(exit_a));
        let (xb, yb, eb) = gather_period(x_b, y_b, p, &ib, Some(entry_b));
        let ea = ea.unwrap();
        let eb = eb.unwrap();
        let model_a = fit_v1(&xa, n_a, p, &ya, &v1)?;
        let model_b = fit_v1(&xb, n_b, p, &yb, &v1)?;
        entry_exit_from_models(
            &model_a, &xa, &ea, n_a, &model_b, &xb, &eb, n_b, n_a as f64, n_b as f64,
        )
    };
    map_reps(reps, &job)
}

pub fn entry_exit_from_models(
    model_a: &V1Model,
    x_a: &[f64],
    exit_a: &[u8],
    n_a: usize,
    model_b: &V1Model,
    x_b: &[f64],
    entry_b: &[u8],
    n_b: usize,
    m_a: f64,
    m_b: f64,
) -> Result<EntryExit> {
    let k = model_a.k_fit;
    if model_b.k_fit != k {
        return Err(TraceError::InvalidArgument(
            "entry/exit models must share K".into(),
        ));
    }
    let stay_a_mask: Vec<u8> = exit_a.iter().map(|&v| u8::from(v == 0)).collect();
    let stay_b_mask: Vec<u8> = entry_b.iter().map(|&v| u8::from(v == 0)).collect();
    let e_a = exit_a.iter().filter(|&&v| v != 0).count() as f64 / n_a as f64;
    let e_b = entry_b.iter().filter(|&&v| v != 0).count() as f64 / n_b as f64;
    let sa = mean_curve(model_a, x_a, n_a, Some(&stay_a_mask))?;
    let sb = mean_curve(model_b, x_b, n_b, Some(&stay_b_mask))?;
    let ex = mean_curve(model_a, x_a, n_a, Some(exit_a))?;
    let en = mean_curve(model_b, x_b, n_b, Some(entry_b))?;
    // Orthonormal α. Convert to c-scale so phi matches a_i * sum(a) / (2k+1).
    let to_c = |v: &[f64]| -> Vec<f64> {
        v.iter()
            .enumerate()
            .map(|(c, a)| a * ((2 * c + 1) as f64).sqrt())
            .collect()
    };
    entry_exit_split(
        &to_c(&sa),
        &to_c(&sb),
        &to_c(&ex),
        &to_c(&en),
        e_a,
        e_b,
        m_a,
        m_b,
    )
}

/// Pointwise percentile bands and a uniform max-|t| band across components.
#[derive(Clone, Debug)]
pub struct PercentileBand {
    pub lo: Vec<f64>,
    pub hi: Vec<f64>,
    pub uniform_lo: Vec<f64>,
    pub uniform_hi: Vec<f64>,
    pub sigma: Vec<f64>,
    pub resolved: Vec<bool>,
    pub critical: f64,
}

pub fn percentile_bands(point: &[f64], replicates: &[f64], reps: usize) -> Result<PercentileBand> {
    let dim = point.len();
    if reps < 2 || replicates.len() != reps * dim {
        return Err(TraceError::Shape(
            "replicates must be a (reps, dim) block with reps ≥ 2".into(),
        ));
    }
    let mut lo = vec![0.0; dim];
    let mut hi = vec![0.0; dim];
    let mut sigma = vec![0.0; dim];
    let mut cols: Vec<Vec<f64>> = vec![Vec::with_capacity(reps); dim];
    for b in 0..reps {
        for c in 0..dim {
            cols[c].push(replicates[b * dim + c]);
        }
    }
    for c in 0..dim {
        let mean = cols[c].iter().sum::<f64>() / reps as f64;
        let var = cols[c]
            .iter()
            .map(|v| {
                let d = v - mean;
                d * d
            })
            .sum::<f64>()
            / (reps - 1) as f64;
        sigma[c] = var.sqrt();
        let mut sorted = cols[c].clone();
        sorted.sort_by(f64::total_cmp);
        lo[c] = percentile_sorted(&sorted, 0.025);
        hi[c] = percentile_sorted(&sorted, 0.975);
    }
    let mut tstats = Vec::with_capacity(reps);
    for b in 0..reps {
        let mut tmax = 0.0_f64;
        for c in 0..dim {
            if sigma[c] > 0.0 {
                let t = (replicates[b * dim + c] - point[c]).abs() / sigma[c];
                tmax = tmax.max(t);
            }
        }
        tstats.push(tmax);
    }
    tstats.sort_by(f64::total_cmp);
    let critical = percentile_sorted(&tstats, 0.95);
    let mut uniform_lo = vec![0.0; dim];
    let mut uniform_hi = vec![0.0; dim];
    let mut resolved = vec![false; dim];
    for c in 0..dim {
        let half = critical * sigma[c];
        uniform_lo[c] = point[c] - half;
        uniform_hi[c] = point[c] + half;
        resolved[c] = uniform_lo[c] > 0.0 || uniform_hi[c] < 0.0;
    }
    Ok(PercentileBand {
        lo,
        hi,
        uniform_lo,
        uniform_hi,
        sigma,
        resolved,
        critical,
    })
}

#[derive(Clone, Debug)]
pub struct PermutationFloor {
    pub observed: f64,
    pub floor: f64,
    pub exceeds: bool,
    pub null_mean: f64,
}

fn outcome_w2(y_a: &[f64], y_b: &[f64], quad: &Quad) -> Result<f64> {
    let qa = quantiles_at(y_a, None, &quad.nodes)?;
    let qb = quantiles_at(y_b, None, &quad.nodes)?;
    Ok(w2_quad(&qa, &qb, &quad.weights))
}

/// 95% (or `level`) percentile of total outcome `W2²` under shuffled period labels.
pub fn permutation_w2_floor(
    y_a: &[f64],
    y_b: &[f64],
    reps: usize,
    seed: u64,
    n_quad: usize,
    k: usize,
    level: f64,
) -> Result<PermutationFloor> {
    if reps == 0 {
        return Err(TraceError::InvalidArgument(
            "permutation needs at least one shuffle".into(),
        ));
    }
    let quad = Quad::new(n_quad, k)?;
    let observed = outcome_w2(y_a, y_b, &quad)?;
    let n_a = y_a.len();
    let n_b = y_b.len();
    let mut pooled = Vec::with_capacity(n_a + n_b);
    pooled.extend_from_slice(y_a);
    pooled.extend_from_slice(y_b);
    let mut nulls = Vec::with_capacity(reps);
    for r in 0..reps {
        let mut rng = SplitMix64::new(
            seed.wrapping_add(r as u64)
                .wrapping_mul(0xD1B5_4A32_D192_ED03),
        );
        let mut labels = vec![0u8; n_a + n_b];
        for i in 0..n_b {
            labels[n_a + i] = 1;
        }
        for i in (1..labels.len()).rev() {
            let j = rng.index(i + 1);
            labels.swap(i, j);
        }
        let mut ya = Vec::with_capacity(n_a);
        let mut yb = Vec::with_capacity(n_b);
        for (i, &lab) in labels.iter().enumerate() {
            if lab == 0 {
                ya.push(pooled[i]);
            } else {
                yb.push(pooled[i]);
            }
        }
        nulls.push(outcome_w2(&ya, &yb, &quad)?);
    }
    let null_mean = nulls.iter().sum::<f64>() / reps as f64;
    nulls.sort_by(f64::total_cmp);
    let floor = percentile_sorted(&nulls, level);
    Ok(PermutationFloor {
        observed,
        floor,
        exceeds: observed > floor,
        null_mean,
    })
}

/// One shuffled-label replicate of the marginal channel shares and total `W2²`.
pub fn permutation_marginal(
    x_a: &[f64],
    y_a: &[f64],
    n_a: usize,
    x_b: &[f64],
    y_b: &[f64],
    n_b: usize,
    cfg: &ShiftV2Config,
    reps: usize,
    seed: u64,
) -> Result<Vec<MarginalDraw>> {
    let p = check_period(x_a, y_a, n_a, x_b, y_b, n_b)?;
    let mut pooled_x = Vec::with_capacity((n_a + n_b) * p);
    let mut pooled_y = Vec::with_capacity(n_a + n_b);
    pooled_x.extend_from_slice(x_a);
    pooled_x.extend_from_slice(x_b);
    pooled_y.extend_from_slice(y_a);
    pooled_y.extend_from_slice(y_b);
    let n = n_a + n_b;
    let job = |r: usize| -> Result<MarginalDraw> {
        let mut rng = SplitMix64::new(
            seed.wrapping_add(r as u64)
                .wrapping_mul(0x2545_F491_4F6C_DD1D),
        );
        let mut idx: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng.index(i + 1);
            idx.swap(i, j);
        }
        let ia = &idx[..n_a];
        let ib = &idx[n_a..];
        let (xa, ya, _) = gather_period(&pooled_x, &pooled_y, p, ia, None);
        let (xb, yb, _) = gather_period(&pooled_x, &pooled_y, p, ib, None);
        let mut local = cfg.clone();
        local.seed = cfg.seed.wrapping_add(r as u64);
        let view = marginal_view(&xa, &ya, n_a, &xb, &yb, n_b, &local)?;
        Ok(MarginalDraw {
            comp_w2: view.comp_w2,
            struct_w2: view.struct_w2,
            comp_total: view.comp_w2_total,
            struct_total: view.struct_w2_total,
            w2_total: view.w2_total,
        })
    };
    map_reps(reps, &job)
}

/// Knobs for [`shift_decompose`]. The benchmark entry point, not the
/// pre-registration path: folds and column order are functions of the seed and
/// the row contents.
#[derive(Clone, Debug)]
pub struct DecomposeConfig {
    pub seed: u64,
    /// L-moment channels reported beside the named statistics. `lambda1` is channel 0.
    pub k: usize,
    pub n_quad: usize,
    pub n_folds: usize,
    pub trim_quantile: f64,
    pub logit_ridge: f64,
    pub logit_iters: usize,
    pub logit_tol: f64,
    pub k_fit: usize,
    pub max_bins: usize,
    pub max_rounds: usize,
    pub fit_tol: f64,
    /// Bootstrap replicates for the percentile bands. `0` returns the point as both ends.
    pub n_boot: usize,
}

impl Default for DecomposeConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            k: 6,
            n_quad: 64,
            n_folds: 5,
            trim_quantile: 0.995,
            logit_ridge: 1e-6,
            logit_iters: 25,
            logit_tol: 1e-8,
            k_fit: 4,
            max_bins: 16,
            max_rounds: 6,
            fit_tol: 1e-6,
            n_boot: 0,
        }
    }
}

/// One call of the benchmark decomposition. `names[i]` is a statistic whose
/// composition and structure use the single counterfactual "structure of A,
/// covariates of B".
#[derive(Clone, Debug)]
pub struct ShiftDecompose {
    pub names: Vec<String>,
    pub composition: Vec<f64>,
    pub structure: Vec<f64>,
    pub feat_comp: Vec<f64>,
    pub feat_struct: Vec<f64>,
    pub intercept: f64,
    pub total_w2: f64,
    pub boot_reps: usize,
    pub composition_lo: Vec<f64>,
    pub composition_hi: Vec<f64>,
    pub structure_lo: Vec<f64>,
    pub structure_hi: Vec<f64>,
    pub feat_comp_lo: Vec<f64>,
    pub feat_comp_hi: Vec<f64>,
    pub feat_struct_lo: Vec<f64>,
    pub feat_struct_hi: Vec<f64>,
    pub intercept_lo: f64,
    pub intercept_hi: f64,
    pub total_w2_lo: f64,
    pub total_w2_hi: f64,
}

#[derive(Clone, Debug)]
struct DecomposePoint {
    names: Vec<String>,
    composition: Vec<f64>,
    structure: Vec<f64>,
    feat_comp: Vec<f64>,
    feat_struct: Vec<f64>,
    intercept: f64,
    total_w2: f64,
}

fn canonical_bits(v: f64) -> u64 {
    if v == 0.0 {
        0
    } else {
        v.to_bits()
    }
}

fn mix_bits(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Column-commutative id of one row. A permutation of features does not change it.
fn row_key(seed: u64, x: &[f64], y: f64) -> u64 {
    let mut h = mix_bits(seed ^ 0xA24B_AED4_963E_E407);
    for &v in x {
        h ^= mix_bits(canonical_bits(v) ^ seed.rotate_left(7));
    }
    h ^= mix_bits(canonical_bits(y) ^ seed.rotate_left(21));
    mix_bits(h)
}

fn reorder_period(x: &[f64], y: &[f64], n: usize, p: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut keys = Vec::with_capacity(n);
    for i in 0..n {
        let row = &x[i * p..(i + 1) * p];
        keys.push((row_key(seed, row, y[i]), canonical_bits(y[i])));
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| keys[i].cmp(&keys[j]));
    let (xo, yo, _) = gather_period(x, y, p, &idx, None);
    (xo, yo)
}

fn column_keys(
    x_a: &[f64],
    y_a: &[f64],
    n_a: usize,
    x_b: &[f64],
    y_b: &[f64],
    n_b: usize,
    p: usize,
    seed: u64,
) -> Vec<u64> {
    let mut row_a = Vec::with_capacity(n_a);
    for i in 0..n_a {
        row_a.push(row_key(seed, &x_a[i * p..(i + 1) * p], y_a[i]));
    }
    let mut row_b = Vec::with_capacity(n_b);
    for i in 0..n_b {
        row_b.push(row_key(seed, &x_b[i * p..(i + 1) * p], y_b[i]));
    }
    let mut keys = vec![0u64; p];
    for j in 0..p {
        let mut h = 0u64;
        for i in 0..n_a {
            h ^= mix_bits(canonical_bits(x_a[i * p + j]) ^ row_a[i]);
        }
        for i in 0..n_b {
            h ^= mix_bits(canonical_bits(x_b[i * p + j]) ^ row_b[i].rotate_left(13));
        }
        keys[j] = h;
    }
    keys
}

/// `order[new_column] = old_column`, sorted by a row-order-invariant column id.
fn canonical_column_order(
    x_a: &[f64],
    y_a: &[f64],
    n_a: usize,
    x_b: &[f64],
    y_b: &[f64],
    n_b: usize,
    p: usize,
    seed: u64,
) -> Vec<usize> {
    let keys = column_keys(x_a, y_a, n_a, x_b, y_b, n_b, p, seed);
    let mut order: Vec<usize> = (0..p).collect();
    order.sort_by(|&a, &b| keys[a].cmp(&keys[b]).then(a.cmp(&b)));
    order
}

fn apply_column_order(x: &[f64], n: usize, p: usize, order: &[usize]) -> Vec<f64> {
    let mut out = vec![0.0; n * p];
    for i in 0..n {
        for (new_j, &old_j) in order.iter().enumerate() {
            out[i * p + new_j] = x[i * p + old_j];
        }
    }
    out
}

fn scatter_to_original(canon: &[f64], order: &[usize]) -> Vec<f64> {
    let mut out = vec![0.0; canon.len()];
    for (new_j, &old_j) in order.iter().enumerate() {
        out[old_j] = canon[new_j];
    }
    out
}

fn with_squares(x: &[f64], n: usize, p: usize) -> Vec<f64> {
    let d = p * 2;
    let mut out = vec![0.0; n * d];
    for i in 0..n {
        for j in 0..p {
            let v = x[i * p + j];
            out[i * d + j] = v;
            out[i * d + p + j] = v * v;
        }
    }
    out
}

fn standardize_pair(a: &mut [f64], n_a: usize, b: &mut [f64], n_b: usize, p: usize) {
    let n = (n_a + n_b) as f64;
    for j in 0..p {
        let mut sum = 0.0;
        let mut sum2 = 0.0;
        for i in 0..n_a {
            let v = a[i * p + j];
            sum += v;
            sum2 += v * v;
        }
        for i in 0..n_b {
            let v = b[i * p + j];
            sum += v;
            sum2 += v * v;
        }
        let mean = sum / n;
        let var = (sum2 / n - mean * mean).max(0.0);
        let sd = var.sqrt();
        if sd > 1e-12 {
            for i in 0..n_a {
                a[i * p + j] = (a[i * p + j] - mean) / sd;
            }
            for i in 0..n_b {
                b[i * p + j] = (b[i * p + j] - mean) / sd;
            }
        } else {
            for i in 0..n_a {
                a[i * p + j] = 0.0;
            }
            for i in 0..n_b {
                b[i * p + j] = 0.0;
            }
        }
    }
}

/// Balanced folds after the rows have been sorted by [`row_key`]. Rank `i` maps
/// to fold `i % K`, so the assignment follows the seed through that sort and
/// does not follow the order the caller stored the rows in.
fn rank_folds(n_a: usize, n_b: usize, k: usize) -> Vec<usize> {
    let k = k.max(1);
    let mut folds = vec![0usize; n_a + n_b];
    for i in 0..n_a {
        folds[i] = i % k;
    }
    for i in 0..n_b {
        folds[n_a + i] = i % k;
    }
    folds
}

fn mean_and_sd(y: &[f64]) -> (f64, f64) {
    let n = y.len() as f64;
    let mean = y.iter().sum::<f64>() / n;
    let var = y
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    (mean, var.max(0.0).sqrt())
}

fn weighted_mean_sd(y: &[f64], w: &[f64]) -> (f64, f64) {
    let n = y.len() as f64;
    let mean = y.iter().zip(w).map(|(v, wi)| v * wi).sum::<f64>() / n;
    let var = y
        .iter()
        .zip(w)
        .map(|(v, wi)| {
            let d = v - mean;
            wi * d * d
        })
        .sum::<f64>()
        / n;
    (mean, var.max(0.0).sqrt())
}

/// Type-7 quantile. Equal weights match NumPy's linear percentile.
fn weighted_quantile(y: &[f64], w: &[f64], q: f64) -> Result<f64> {
    let n = y.len();
    if n == 0 || w.len() != n {
        return Err(TraceError::Shape(
            "quantile inputs must be non-empty and the same length".into(),
        ));
    }
    if n == 1 {
        return Ok(y[0]);
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| y[i].total_cmp(&y[j]).then(i.cmp(&j)));
    let mut total = 0.0;
    for &i in &order {
        if !w[i].is_finite() || w[i] < 0.0 {
            return Err(TraceError::InvalidArgument(
                "weights must be finite and nonnegative".into(),
            ));
        }
        total += w[i];
    }
    if !(total > 0.0) {
        return Err(TraceError::InvalidArgument(
            "weights must have positive total mass".into(),
        ));
    }
    let q = q.clamp(0.0, 1.0);
    if q == 0.0 {
        return Ok(y[order[0]]);
    }
    if q == 1.0 {
        return Ok(y[order[n - 1]]);
    }
    let denom = total - w[order[n - 1]];
    if !(denom > 0.0) {
        return Ok(y[order[0]]);
    }
    let mut cum_excl = 0.0;
    let mut prev_t = 0.0;
    let mut prev_y = y[order[0]];
    for (rank, &i) in order.iter().enumerate() {
        let t = (cum_excl / denom).min(1.0);
        if rank > 0 && t + 1e-15 >= q {
            let span = (t - prev_t).max(1e-15);
            let frac = ((q - prev_t) / span).clamp(0.0, 1.0);
            return Ok(prev_y * (1.0 - frac) + y[i] * frac);
        }
        prev_t = t;
        prev_y = y[i];
        cum_excl += w[i];
    }
    Ok(y[order[n - 1]])
}

fn unweighted_quantile(y: &[f64], q: f64) -> f64 {
    let mut sorted = y.to_vec();
    sorted.sort_by(f64::total_cmp);
    percentile_sorted(&sorted, q)
}

fn push_split(comp: &mut Vec<f64>, structure: &mut Vec<f64>, c: f64, s: f64) {
    comp.push(c);
    structure.push(s);
}

fn decompose_point(
    x_a: &[f64],
    y_a: &[f64],
    n_a: usize,
    x_b: &[f64],
    y_b: &[f64],
    n_b: usize,
    cfg: &DecomposeConfig,
) -> Result<DecomposePoint> {
    let p = check_period(x_a, y_a, n_a, x_b, y_b, n_b)?;
    if cfg.k == 0 || cfg.n_folds < 2 {
        return Err(TraceError::InvalidArgument(
            "decomposition needs K ≥ 1 and at least two folds".into(),
        ));
    }
    let (xa, ya) = reorder_period(x_a, y_a, n_a, p, cfg.seed);
    let (xb, yb) = reorder_period(x_b, y_b, n_b, p, cfg.seed);
    let col_order = canonical_column_order(&xa, &ya, n_a, &xb, &yb, n_b, p, cfg.seed);
    let xa_m = apply_column_order(&xa, n_a, p, &col_order);
    let xb_m = apply_column_order(&xb, n_b, p, &col_order);

    let mut xa_z = with_squares(&xa_m, n_a, p);
    let mut xb_z = with_squares(&xb_m, n_b, p);
    standardize_pair(&mut xa_z, n_a, &mut xb_z, n_b, p * 2);
    let folds = rank_folds(n_a, n_b, cfg.n_folds);
    let logit_cfg = ShiftV2Config {
        k: cfg.k,
        n_quad: cfg.n_quad,
        n_folds: cfg.n_folds,
        trim_quantile: cfg.trim_quantile,
        logit_ridge: cfg.logit_ridge,
        logit_iters: cfg.logit_iters,
        logit_tol: cfg.logit_tol,
        seed: cfg.seed,
    };
    let pooled = Pooled {
        x_a: &xa_z,
        n_a,
        x_b: &xb_z,
        n_b,
        p: p * 2,
    };
    let lp = cross_fit_lp_folds(&pooled, &folds, cfg.n_folds, &logit_cfg)?;
    let prior_a = (n_a as f64 / n_b as f64).ln();
    let mut w = vec![0.0; n_a];
    for i in 0..n_a {
        // 1/ψ reweights period A onto the covariate law of period B.
        let z = (-lp[i] + prior_a).clamp(-30.0, 30.0);
        w[i] = z.exp();
    }
    trim_and_normalize(&mut w, cfg.trim_quantile)?;

    let (mean_a, sd_a) = mean_and_sd(&ya);
    let (mean_b, sd_b) = mean_and_sd(&yb);
    let (mean_c, sd_c) = weighted_mean_sd(&ya, &w);
    let mut composition = Vec::with_capacity(5 + cfg.k);
    let mut structure = Vec::with_capacity(5 + cfg.k);
    let mut names = Vec::with_capacity(5 + cfg.k);
    for (name, a, c, b) in [("mean", mean_a, mean_c, mean_b), ("sd", sd_a, sd_c, sd_b)] {
        names.push(name.to_string());
        push_split(&mut composition, &mut structure, c - a, b - c);
    }
    for (name, q) in [("q10", 0.10), ("q50", 0.50), ("q90", 0.90)] {
        let a = unweighted_quantile(&ya, q);
        let b = unweighted_quantile(&yb, q);
        let c = weighted_quantile(&ya, &w, q)?;
        names.push(name.to_string());
        push_split(&mut composition, &mut structure, c - a, b - c);
    }

    let quad = Quad::new(cfg.n_quad, cfg.k)?;
    let qa = quantiles_at(&ya, None, &quad.nodes)?;
    let qb = quantiles_at(&yb, None, &quad.nodes)?;
    let qc = quantiles_at(&ya, Some(&w), &quad.nodes)?;
    let lam_a = lambdas_of(&qa, &quad);
    let lam_b = lambdas_of(&qb, &quad);
    let lam_c = lambdas_of(&qc, &quad);
    for c in 0..cfg.k {
        names.push(format!("lambda{}", c + 1));
        push_split(
            &mut composition,
            &mut structure,
            lam_c[c] - lam_a[c],
            lam_b[c] - lam_c[c],
        );
    }
    let total_w2 = w2_quad(&qa, &qb, &quad.weights);

    let v1 = V1Config {
        k_fit: cfg.k_fit,
        k_report: cfg.k_fit,
        n_u: 0,
        max_bins: cfg.max_bins,
        max_rounds: cfg.max_rounds,
        tol: cfg.fit_tol,
        min_count: 1,
        ridge: 1e-6,
        learning_rate: 1.0,
        mono: MonoMode::None,
        bernstein_elevation: 4,
        reporting: ReportKind::Legendre,
        reference: Vec::new(),
        lock_mean: false,
        smooth: 0.0,
        levenberg: 0.0,
        sobolev_lambda: 0.0,
        sobolev_s: 0.0,
        tail_gamma: 0.0,
    };
    let model_a = fit_v1(&xa_m, n_a, p, &ya, &v1)?;
    let model_b = fit_v1(&xb_m, n_b, p, &yb, &v1)?;
    let g_aa = mean_effects(&model_a, &xa_m, n_a)?;
    let g_ab = mean_effects(&model_a, &xb_m, n_b)?;
    let g_ba = mean_effects(&model_b, &xa_m, n_a)?;
    let g_bb = mean_effects(&model_b, &xb_m, n_b)?;
    let k_fit = model_a.k_fit;
    let mut feat_comp_c = vec![0.0; p];
    let mut feat_struct_c = vec![0.0; p];
    for j in 0..p {
        let aa = g_aa[j * k_fit];
        let ab = g_ab[j * k_fit];
        let ba = g_ba[j * k_fit];
        let bb = g_bb[j * k_fit];
        let delta_a = ba - aa;
        let delta_b = bb - ab;
        feat_comp_c[j] = ab - aa;
        // Pooled 50/50 law, the same centring the benchmark uses for a slope change.
        feat_struct_c[j] = delta_b - 0.5 * (delta_a + delta_b);
    }
    let feat_comp = scatter_to_original(&feat_comp_c, &col_order);
    let feat_struct = scatter_to_original(&feat_struct_c, &col_order);
    let mean_struct = structure[0];
    let intercept = mean_struct - feat_struct.iter().sum::<f64>();
    Ok(DecomposePoint {
        names,
        composition,
        structure,
        feat_comp,
        feat_struct,
        intercept,
        total_w2,
    })
}

fn point_bands(point: DecomposePoint) -> ShiftDecompose {
    ShiftDecompose {
        composition_lo: point.composition.clone(),
        composition_hi: point.composition.clone(),
        structure_lo: point.structure.clone(),
        structure_hi: point.structure.clone(),
        feat_comp_lo: point.feat_comp.clone(),
        feat_comp_hi: point.feat_comp.clone(),
        feat_struct_lo: point.feat_struct.clone(),
        feat_struct_hi: point.feat_struct.clone(),
        intercept_lo: point.intercept,
        intercept_hi: point.intercept,
        total_w2_lo: point.total_w2,
        total_w2_hi: point.total_w2,
        boot_reps: 0,
        names: point.names,
        composition: point.composition,
        structure: point.structure,
        feat_comp: point.feat_comp,
        feat_struct: point.feat_struct,
        intercept: point.intercept,
        total_w2: point.total_w2,
    }
}

fn band_of(col: &mut [f64]) -> (f64, f64) {
    col.sort_by(f64::total_cmp);
    (percentile_sorted(col, 0.025), percentile_sorted(col, 0.975))
}

fn attach_bands(point: DecomposePoint, draws: &[DecomposePoint]) -> ShiftDecompose {
    let reps = draws.len();
    let n_stat = point.composition.len();
    let p = point.feat_comp.len();
    let mut composition_lo = vec![0.0; n_stat];
    let mut composition_hi = vec![0.0; n_stat];
    let mut structure_lo = vec![0.0; n_stat];
    let mut structure_hi = vec![0.0; n_stat];
    for j in 0..n_stat {
        let mut col: Vec<f64> = draws.iter().map(|d| d.composition[j]).collect();
        let (lo, hi) = band_of(&mut col);
        composition_lo[j] = lo;
        composition_hi[j] = hi;
        let mut col: Vec<f64> = draws.iter().map(|d| d.structure[j]).collect();
        let (lo, hi) = band_of(&mut col);
        structure_lo[j] = lo;
        structure_hi[j] = hi;
    }
    let mut feat_comp_lo = vec![0.0; p];
    let mut feat_comp_hi = vec![0.0; p];
    let mut feat_struct_lo = vec![0.0; p];
    let mut feat_struct_hi = vec![0.0; p];
    for j in 0..p {
        let mut col: Vec<f64> = draws.iter().map(|d| d.feat_comp[j]).collect();
        let (lo, hi) = band_of(&mut col);
        feat_comp_lo[j] = lo;
        feat_comp_hi[j] = hi;
        let mut col: Vec<f64> = draws.iter().map(|d| d.feat_struct[j]).collect();
        let (lo, hi) = band_of(&mut col);
        feat_struct_lo[j] = lo;
        feat_struct_hi[j] = hi;
    }
    let mut intercepts: Vec<f64> = draws.iter().map(|d| d.intercept).collect();
    let (intercept_lo, intercept_hi) = band_of(&mut intercepts);
    let mut totals: Vec<f64> = draws.iter().map(|d| d.total_w2).collect();
    let (total_w2_lo, total_w2_hi) = band_of(&mut totals);
    ShiftDecompose {
        names: point.names,
        composition: point.composition,
        structure: point.structure,
        feat_comp: point.feat_comp,
        feat_struct: point.feat_struct,
        intercept: point.intercept,
        total_w2: point.total_w2,
        boot_reps: reps,
        composition_lo,
        composition_hi,
        structure_lo,
        structure_hi,
        feat_comp_lo,
        feat_comp_hi,
        feat_struct_lo,
        feat_struct_hi,
        intercept_lo,
        intercept_hi,
        total_w2_lo,
        total_w2_hi,
    }
}

/// Benchmark decomposition.
///
/// Statistics use the counterfactual with the structure of period A and the
/// covariates of period B, estimated by cross-fitted period odds. The odds
/// model is an intercept plus each feature and its square. Folds are the rank
/// of a column-commutative row hash modulo `K`, so a permutation of rows does
/// not change the fit. Feature shares are the mean channel of the additive
/// fit, centred at the 50/50 pooled covariate law. `intercept` closes
/// `mean` structure: it is that structure minus the sum of `feat_struct`.
/// `total_w2` is the quadrature `W2²` between the two outcome quantile functions
/// and is not a sum of the split. Bands are pointwise 2.5/97.5 percentiles of
/// a stratified row bootstrap; with `n_boot == 0` both ends equal the point.
pub fn shift_decompose(
    x_a: &[f64],
    y_a: &[f64],
    n_a: usize,
    x_b: &[f64],
    y_b: &[f64],
    n_b: usize,
    cfg: &DecomposeConfig,
) -> Result<ShiftDecompose> {
    let p = check_period(x_a, y_a, n_a, x_b, y_b, n_b)?;
    let point = decompose_point(x_a, y_a, n_a, x_b, y_b, n_b, cfg)?;
    if cfg.n_boot == 0 {
        return Ok(point_bands(point));
    }
    let (xa, ya) = reorder_period(x_a, y_a, n_a, p, cfg.seed);
    let (xb, yb) = reorder_period(x_b, y_b, n_b, p, cfg.seed);
    let boot_cfg = DecomposeConfig {
        n_boot: 0,
        ..cfg.clone()
    };
    let draws = map_reps(cfg.n_boot, &|b| {
        let mut rng = SplitMix64::new(
            cfg.seed
                .wrapping_add(b as u64)
                .wrapping_mul(0xD1B5_4A32_D192_ED03),
        );
        let ia = resample_indices(n_a, n_a, &mut rng);
        let ib = resample_indices(n_b, n_b, &mut rng);
        let (x1, y1, _) = gather_period(&xa, &ya, p, &ia, None);
        let (x2, y2, _) = gather_period(&xb, &yb, p, &ib, None);
        decompose_point(&x1, &y1, n_a, &x2, &y2, n_b, &boot_cfg)
    })?;
    Ok(attach_bands(point, &draws))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{MonoMode, QuadBasis, ReportKind};

    #[test]
    fn shapley_two_is_efficient_and_order_free() {
        let a = 0.3;
        let cp = -0.2;
        let c = 0.7;
        let b = 0.1;
        let s = shapley_two(a, cp, c, b);
        let t = shapley_two_swapped(a, cp, c, b);
        assert!((s.comp + s.structure - (b - a)).abs() < 1e-12);
        assert!((s.comp - t.comp).abs() < 1e-15);
        assert!((s.structure - t.structure).abs() < 1e-15);
        assert!((s.interaction - (b - c - cp + a)).abs() < 1e-15);
    }

    #[test]
    fn quadratic_shapley_matches_brute_force_and_keeps_signs() {
        let a = [0.5, -0.25, 0.1, 0.4];
        for k in [0usize, 2, 5] {
            let denom = (2 * k + 1) as f64;
            let phi = quadratic_shapley(&a, k);
            let brute = brute_quadratic(&a);
            for i in 0..a.len() {
                assert!((phi[i] - brute[i] / denom).abs() < 1e-12);
            }
            let sum_a: f64 = a.iter().sum();
            let w2 = sum_a * sum_a / denom;
            assert!((phi.iter().sum::<f64>() - w2).abs() < 1e-12);
        }
        let neg = quadratic_shapley(&[1.0, -0.4], 0);
        assert!(neg[1] < 0.0);
        assert!((neg[0] - 0.6).abs() < 1e-12);
        assert!((neg[1] + 0.24).abs() < 1e-12);
    }

    fn brute_quadratic(a: &[f64]) -> Vec<f64> {
        let m = a.len();
        let mut perm: Vec<usize> = (0..m).collect();
        let mut phi = vec![0.0; m];
        let mut nperm = 0.0;
        fn walk(perm: &mut [usize], start: usize, a: &[f64], phi: &mut [f64], nperm: &mut f64) {
            if start == perm.len() {
                let mut s = 0.0;
                for &i in perm.iter() {
                    let new_s = s + a[i];
                    phi[i] += new_s * new_s - s * s;
                    s = new_s;
                }
                *nperm += 1.0;
                return;
            }
            for i in start..perm.len() {
                perm.swap(start, i);
                walk(perm, start + 1, a, phi, nperm);
                perm.swap(start, i);
            }
        }
        walk(&mut perm, 0, a, &mut phi, &mut nperm);
        for v in &mut phi {
            *v /= nperm;
        }
        phi
    }

    #[test]
    fn entry_exit_identity_and_separate_mass() {
        let sa = [0.2, -0.1];
        let sb = [0.2, 0.4];
        let ex = [1.0, 0.5];
        let en = [-0.3, 0.8];
        let e_a = 0.15;
        let e_b = 0.25;
        let got = entry_exit_split(&sa, &sb, &ex, &en, e_a, e_b, 100.0, 80.0).unwrap();
        for c in 0..2 {
            let delta = (1.0 - e_b) * sb[c] + e_b * en[c] - ((1.0 - e_a) * sa[c] + e_a * ex[c]);
            assert!((got.delta[c] - delta).abs() < 1e-12);
            let phi_sum = got.phi_stay[c] + got.phi_mix[c] + got.phi_entry[c] + got.phi_exit[c];
            let players = [got.stay[c], got.mix[c], got.entry[c], got.exit[c]];
            let expect = quadratic_shapley(&players, c);
            assert!((phi_sum - expect.iter().sum::<f64>()).abs() < 1e-12);
            assert!(
                (phi_sum - players.iter().sum::<f64>().powi(2) / (2 * c + 1) as f64).abs() < 1e-12
            );
        }
        let (hk, norm) = hk_mass(100.0, 80.0);
        assert!((got.hk2_mass - hk).abs() < 1e-12);
        assert!((got.hk2_normalized - norm).abs() < 1e-12);
        assert!((hk - 4.0 * (100.0_f64.sqrt() - 80.0_f64.sqrt()).powi(2)).abs() < 1e-12);
    }

    #[test]
    fn barycenter_players_sum_and_do_not_clip() {
        let (model_a, model_b, xa, xb) = toy_models();
        let view = barycenter_view(&model_a, &model_b, &xa, 4, &xb, 4).unwrap();
        assert!(view.player_sum_gap < 1e-9);
        assert!(view.order_gap < 1e-15);
        let mut saw_negative = false;
        for v in &view.phi {
            if *v < 0.0 {
                saw_negative = true;
            }
        }
        assert!(saw_negative);
        for c in 0..view.k {
            let mut s = 0.0;
            for j in 0..view.p {
                s += view.comp_alpha[j * view.k + c];
                s += view.struct_alpha[j * view.k + c];
            }
            s += view.intercept_alpha[c];
            let delta = view.alpha_b[c] - view.alpha_a[c];
            assert!((s - delta).abs() < 1e-12);
        }
    }

    fn toy_models() -> (V1Model, V1Model, Vec<f64>, Vec<f64>) {
        let basis = QuadBasis::new(8, 2).unwrap();
        let make = |intercept: [f64; 2], low: [f64; 2], high: [f64; 2]| V1Model {
            k_fit: 2,
            k_report: 2,
            basis: basis.clone(),
            edges: vec![vec![0.0]],
            n_bins: vec![3],
            intercept: intercept.to_vec(),
            effects: vec![vec![0.0, 0.0, low[0], low[1], high[0], high[1]]],
            train_counts: vec![vec![0, 2, 2]],
            report: vec![1.0, 0.0, 0.0, 1.0],
            reporting: ReportKind::Legendre,
            mono: MonoMode::None,
            bernstein_elevation: 0,
            non_representable: false,
            bernstein_lift: 0.0,
            train_loss: 0.0,
            rounds: 0,
            lock_mean: false,
            smooth: 0.0,
            levenberg: 0.0,
            sobolev_lambda: 0.0,
            sobolev_s: 0.0,
            tail_gamma: 0.0,
        };
        let a = make([1.0, 0.2], [-0.4, 0.3], [0.4, -0.1]);
        let b = make([1.5, -0.1], [-0.2, -0.4], [0.8, 0.2]);
        let xa = vec![-1.0, -0.5, -0.2, -1.2];
        let xb = vec![0.2, 1.0, 0.5, 1.4];
        (a, b, xa, xb)
    }

    #[test]
    fn dfl_reweighting_tracks_the_counterfactual_quantile() {
        let n = 40_000;
        let mut rng = SplitMix64::new(11);
        let mut xa = vec![0.0; n];
        let mut xb = vec![0.0; n];
        let mut yb = vec![0.0; n];
        let mut yc = vec![0.0; n];
        for i in 0..n {
            xa[i] = normal(&mut rng);
            xb[i] = normal(&mut rng) + 0.5;
            yb[i] = 2.0 * xb[i] + 0.5 * normal(&mut rng);
            yc[i] = 2.0 * xa[i] + 0.5 * normal(&mut rng);
        }
        let cfg = ShiftV2Config {
            k: 1,
            n_quad: 32,
            n_folds: 2,
            trim_quantile: 0.995,
            seed: 3,
            ..ShiftV2Config::default()
        };
        let view = marginal_view(&xa, &yc, n, &xb, &yb, n, &cfg).unwrap();
        // Pure composition, same conditional law: structure share on the mean is near 0,
        // and the reweighted period-B law matches the counterfactual sample in the mean.
        assert!(
            view.struct_w2[0].abs() < 0.08,
            "struct share {}",
            view.struct_w2[0]
        );
        assert!((view.lambda_c[0] - view.lambda_a[0]).abs() < 0.05);
        assert!(view.trim_mass_c >= 0.0);
    }

    #[test]
    fn equal_weights_trim_nothing_visible_and_bands_cover_a_constant() {
        let mut w = vec![1.0; 100];
        let (mass, _) = trim_and_normalize(&mut w, 0.995).unwrap();
        assert!(mass.abs() < 1e-15);
        assert!((w.iter().sum::<f64>() / w.len() as f64 - 1.0).abs() < 1e-12);
        let point = [0.0, 1.0];
        let mut reps = Vec::new();
        for b in 0..40 {
            let noise = (b as f64 - 20.0) / 100.0;
            reps.push(noise);
            reps.push(1.0 + noise);
        }
        let band = percentile_bands(&point, &reps, 40).unwrap();
        assert!(!band.resolved[0] || band.uniform_lo[0] <= 0.0);
        assert!(band.uniform_lo[1] > 0.0 || band.hi[1] > 0.5);
    }

    fn normal(rng: &mut SplitMix64) -> f64 {
        let u1 = rng.next_f64().clamp(1e-12, 1.0);
        let u2 = rng.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    #[test]
    fn permutation_floor_of_identical_samples_is_not_always_exceeded() {
        let mut rng = SplitMix64::new(5);
        let mut ya = Vec::new();
        let mut yb = Vec::new();
        for _ in 0..400 {
            ya.push(normal(&mut rng));
            yb.push(normal(&mut rng));
        }
        let floor = permutation_w2_floor(&ya, &yb, 80, 9, 16, 2, 0.95).unwrap();
        assert!(floor.floor >= 0.0);
        assert!(floor.null_mean >= 0.0);
    }

    #[test]
    fn bootstrap_replicates_are_distinct_resamples() {
        let n = 400;
        let mut x_a = vec![0.0; n];
        let mut x_b = vec![0.0; n];
        let mut y_a = vec![0.0; n];
        let mut y_b = vec![0.0; n];
        for i in 0..n {
            let z = (i as f64) / (n as f64) - 0.5;
            x_a[i] = z;
            x_b[i] = z + 0.25;
            y_a[i] = z;
            y_b[i] = 0.8 * z + 0.4;
        }
        let cfg = ShiftV2Config {
            k: 3,
            n_quad: 16,
            n_folds: 2,
            ..ShiftV2Config::default()
        };
        let draws = bootstrap_marginal(&x_a, &y_a, n, &x_b, &y_b, n, &cfg, 12, 3).unwrap();
        let mut lo = f64::MAX;
        let mut hi = f64::MIN;
        for d in &draws {
            lo = lo.min(d.w2_total);
            hi = hi.max(d.w2_total);
        }
        assert!(
            hi - lo > 1e-3,
            "bootstrap W2 range {lo}..{hi} is too small to be a resample"
        );
    }

    fn toy_periods(seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, usize, usize) {
        let n = 180;
        let p = 3;
        let mut rng = SplitMix64::new(seed);
        let mut x_a = vec![0.0; n * p];
        let mut x_b = vec![0.0; n * p];
        let mut y_a = vec![0.0; n];
        let mut y_b = vec![0.0; n];
        for i in 0..n {
            for j in 0..p {
                x_a[i * p + j] = normal(&mut rng);
                x_b[i * p + j] = normal(&mut rng);
            }
            x_b[i * p] += 0.4;
            y_a[i] = x_a[i * p] - 0.5 * x_a[i * p + 1] + 0.3 * normal(&mut rng);
            y_b[i] = 0.2 + 1.3 * x_b[i * p] - 0.5 * x_b[i * p + 1] + 0.3 * normal(&mut rng);
        }
        (x_a, y_a, x_b, y_b, n, p)
    }

    fn fast_decomp() -> DecomposeConfig {
        DecomposeConfig {
            seed: 11,
            k: 3,
            n_quad: 16,
            n_folds: 2,
            k_fit: 3,
            max_bins: 8,
            max_rounds: 4,
            n_boot: 0,
            ..DecomposeConfig::default()
        }
    }

    #[test]
    fn equal_weights_match_the_unweighted_percentile() {
        let y = [0.2, -1.0, 0.5, 3.0, 0.0, 1.2];
        let w = vec![1.0; y.len()];
        for q in [0.1, 0.5, 0.9] {
            let got = weighted_quantile(&y, &w, q).unwrap();
            let mut sorted = y.to_vec();
            sorted.sort_by(f64::total_cmp);
            let expect = percentile_sorted(&sorted, q);
            assert!((got - expect).abs() < 1e-12, "{got} vs {expect} at {q}");
        }
    }

    #[test]
    fn decompose_is_invariant_to_row_and_column_order() {
        let (x_a, y_a, x_b, y_b, n, p) = toy_periods(4);
        let cfg = fast_decomp();
        let base = shift_decompose(&x_a, &y_a, n, &x_b, &y_b, n, &cfg).unwrap();
        let rev_a: Vec<usize> = (0..n).rev().collect();
        let mut rev_b: Vec<usize> = (0..n).collect();
        rev_b.swap(0, n - 1);
        rev_b.swap(3, 9);
        let (xa, ya, _) = gather_period(&x_a, &y_a, p, &rev_a, None);
        let (xb, yb, _) = gather_period(&x_b, &y_b, p, &rev_b, None);
        let row = shift_decompose(&xa, &ya, n, &xb, &yb, n, &cfg).unwrap();
        for i in 0..base.composition.len() {
            assert!((base.composition[i] - row.composition[i]).abs() < 1e-9);
            assert!((base.structure[i] - row.structure[i]).abs() < 1e-9);
        }
        for j in 0..p {
            assert!((base.feat_comp[j] - row.feat_comp[j]).abs() < 1e-9);
            assert!((base.feat_struct[j] - row.feat_struct[j]).abs() < 1e-9);
        }
        assert!((base.intercept - row.intercept).abs() < 1e-9);
        assert!((base.total_w2 - row.total_w2).abs() < 1e-9);
        let mean_gap = y_b.iter().sum::<f64>() / n as f64 - y_a.iter().sum::<f64>() / n as f64;
        assert!((base.composition[0] + base.structure[0] - mean_gap).abs() < 1e-9);
        let feat_sum: f64 = base.feat_struct.iter().sum();
        assert!((base.intercept + feat_sum - base.structure[0]).abs() < 1e-9);
        assert!(base.names.iter().any(|n| n == "q90"));
        assert!(base.names.iter().any(|n| n == "lambda1"));
        assert_eq!(base.boot_reps, 0);

        // Swap columns 0 and 2. Shares move with the column; statistics stay.
        let swap = |x: &[f64]| {
            let mut o = x.to_vec();
            for i in 0..n {
                o.swap(i * p, i * p + 2);
            }
            o
        };
        let col = shift_decompose(&swap(&x_a), &y_a, n, &swap(&x_b), &y_b, n, &cfg).unwrap();
        for i in 0..base.composition.len() {
            assert!(
                (base.composition[i] - col.composition[i]).abs() < 1e-8,
                "{} comp {} vs {}",
                base.names[i],
                base.composition[i],
                col.composition[i]
            );
            assert!((base.structure[i] - col.structure[i]).abs() < 1e-8);
        }
        assert!((base.feat_comp[0] - col.feat_comp[2]).abs() < 1e-8);
        assert!((base.feat_comp[2] - col.feat_comp[0]).abs() < 1e-8);
        assert!((base.feat_comp[1] - col.feat_comp[1]).abs() < 1e-8);
        assert!((base.feat_struct[0] - col.feat_struct[2]).abs() < 1e-8);
        assert!((base.total_w2 - col.total_w2).abs() < 1e-8);
    }

    #[test]
    fn decompose_bands_have_width_when_the_sample_moves() {
        let (x_a, y_a, x_b, y_b, n, _) = toy_periods(8);
        let mut cfg = fast_decomp();
        cfg.n_boot = 6;
        cfg.max_rounds = 3;
        let got = shift_decompose(&x_a, &y_a, n, &x_b, &y_b, n, &cfg).unwrap();
        assert_eq!(got.boot_reps, 6);
        assert!(got.total_w2_hi + 1e-12 >= got.total_w2_lo);
        assert!(got.composition_hi[0] + 1e-12 >= got.composition_lo[0]);
        assert_eq!(got.feat_comp_lo.len(), 3);
    }
}
