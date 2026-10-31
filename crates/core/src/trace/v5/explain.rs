//! End-to-end v5 decomposition on the Eulerian mixture counterfactual.
//!
//! Conditional model per period `t`: `Y | x = μ_t(x) + ε`, with `μ_t` the in-repo
//! binned mean booster (v4 `FitL2Config::default().mean`, unchanged) and `ε` drawn from
//! the empirical residual law of period `t`. The four hybrids
//! `F_[s,t](y) = E_{x~F_s} F_t(y | x)` are tabulated on one shared grid (exact pair
//! counts), inverted, and read out as exact L-moments. Each L-moment change
//! `λ(Q_[B,B]) - λ(Q_[A,A])` is split by the two-player Shapley value (order-free):
//!
//! ```text
//! comp   = ½[(λ_[B,A] - λ_[A,A]) + (λ_[B,B] - λ_[A,B])]   (covariates switched)
//! struct = ½[(λ_[A,B] - λ_[A,A]) + (λ_[B,B] - λ_[B,A])]   (conditional law switched)
//! ```
//!
//! `comp + struct` equals the total change exactly.

use std::time::Instant;

use crate::error::{Result, TraceError};
use crate::trace::legendre::gauss_legendre_01;
use crate::trace::meanboost::{fit_mean, predict_mean, MeanConfig};
use crate::trace::v4::FitL2Config;

use super::mixture::{
    lmoments_from_cdf_grid, mixture_cdf_location, quantile_from_cdf, support_grid,
};

/// Location model for `μ_t`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LocationModel {
    /// v5: binned mean booster (`ExplainV5Config::mean`).
    #[default]
    Booster,
    /// TRACE_V5_1 (PREREG_V5_1 §2): per-period OLS on an intercept plus all `p`
    /// columns. Closed form, no tunable parameters.
    Ols,
}

/// Per-period OLS location fit: coefficients `[β0, β1..βp]`.
///
/// Solved by Householder QR on `[1 | X]` (no penalty, no selection, no transforms).
pub fn fit_ols(x: &[f64], n: usize, p: usize, y: &[f64]) -> Result<Vec<f64>> {
    let m = p + 1;
    if n < m || x.len() != n * p || y.len() != n {
        return Err(TraceError::InvalidArgument(
            "OLS location fit needs n >= p + 1 and matching shapes".into(),
        ));
    }
    // column-major design A (n × m)
    let mut a = vec![0.0f64; n * m];
    for i in 0..n {
        a[i] = 1.0;
        for j in 0..p {
            a[(j + 1) * n + i] = x[i * p + j];
        }
    }
    let mut b = y.to_vec();
    let mut rdiag = vec![0.0f64; m];
    for k in 0..m {
        let col = &mut a[k * n..(k + 1) * n];
        let norm = col[k..].iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm == 0.0 {
            return Err(TraceError::InvalidArgument(
                "OLS location fit: design is rank deficient".into(),
            ));
        }
        let alpha = if col[k] > 0.0 { -norm } else { norm };
        col[k] -= alpha;
        let vnorm2 = col[k..].iter().map(|v| v * v).sum::<f64>();
        rdiag[k] = alpha;
        let v: Vec<f64> = col[k..].to_vec();
        for jj in (k + 1)..m {
            let cj = &mut a[jj * n + k..(jj + 1) * n];
            let dot: f64 = v.iter().zip(cj.iter()).map(|(p, q)| p * q).sum();
            let f = 2.0 * dot / vnorm2;
            for (c, vv) in cj.iter_mut().zip(&v) {
                *c -= f * vv;
            }
        }
        let dot: f64 = v.iter().zip(&b[k..]).map(|(p, q)| p * q).sum();
        let f = 2.0 * dot / vnorm2;
        for (c, vv) in b[k..].iter_mut().zip(&v) {
            *c -= f * vv;
        }
    }
    let scale = rdiag.iter().fold(0.0f64, |s, r| s.max(r.abs()));
    if rdiag.iter().any(|r| r.abs() <= 1e-12 * scale) {
        return Err(TraceError::InvalidArgument(
            "OLS location fit: design is rank deficient".into(),
        ));
    }
    let mut beta = vec![0.0f64; m];
    for k in (0..m).rev() {
        let mut s = b[k];
        for jj in (k + 1)..m {
            s -= a[jj * n + k] * beta[jj];
        }
        beta[k] = s / rdiag[k];
    }
    Ok(beta)
}

/// `β0 + xᵀβ` for each row of `x` (`n × p` row-major).
pub fn predict_ols(beta: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    let p = beta.len() - 1;
    (0..n)
        .map(|i| {
            beta[0]
                + x[i * p..(i + 1) * p]
                    .iter()
                    .zip(&beta[1..])
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
        })
        .collect()
}

/// Knobs for [`explain_v5`]. Defaults are the frozen in-repo values.
#[derive(Clone, Debug)]
pub struct ExplainV5Config {
    /// Location model (default [`LocationModel::Booster`] = v5; `Ols` = TRACE_V5_1).
    pub mean_model: LocationModel,
    /// Mean booster for `μ_t` (default: v4 `FitL2Config::default().mean`).
    pub mean: MeanConfig,
    /// Shared y-grid size (default 256, v4 `MixtureConfig::default().n_y`).
    pub n_y: usize,
    /// Number of L-moments to report (default 4).
    pub k_lmoments: usize,
    /// Gauss–Legendre nodes for quantile curves / mixing residual (default 64).
    pub n_quad: usize,
}

impl Default for ExplainV5Config {
    fn default() -> Self {
        Self {
            mean_model: LocationModel::Booster,
            mean: FitL2Config::default().mean,
            n_y: 256,
            k_lmoments: 4,
            n_quad: 64,
        }
    }
}

/// One hybrid law `[s, t]`: covariates from `s`, conditional from `t`.
#[derive(Clone, Debug)]
pub struct Hybrid {
    /// Eulerian mixture quantile `Q_[s,t]` on the GL nodes.
    pub q_mix: Vec<f64>,
    /// Lagrangian barycenter `Q̄_[s,t]` on the GL nodes.
    pub q_bary: Vec<f64>,
    /// L-moments of `Q_[s,t]`.
    pub lmom: Vec<f64>,
    /// L-moments of `Q̄_[s,t]`.
    pub lmom_bary: Vec<f64>,
    /// `‖Q_[s,t] - Q̄_[s,t]‖_{L²(0,1)}` (mixing residual size).
    pub mixing_residual: f64,
}

/// Report of [`explain_v5`].
#[derive(Clone, Debug)]
pub struct ExplainV5Report {
    pub u_nodes: Vec<f64>,
    pub y_grid: Vec<f64>,
    /// Hybrids `[A,A]`, `[A,B]`, `[B,A]`, `[B,B]`.
    pub aa: Hybrid,
    pub ab: Hybrid,
    pub ba: Hybrid,
    pub bb: Hybrid,
    /// Per-L-moment composition piece (Shapley).
    pub lsplit_comp: Vec<f64>,
    /// Per-L-moment response (structure) piece (Shapley).
    pub lsplit_struct: Vec<f64>,
    pub seconds_fit: f64,
    pub seconds_mixture: f64,
    pub seconds_total: f64,
}

fn check(x: &[f64], n: usize, p: usize, y: &[f64], who: &str) -> Result<()> {
    if n < 2 || p == 0 || x.len() != n * p || y.len() != n {
        return Err(TraceError::InvalidArgument(format!(
            "{who}: x must be n×p row-major and y length n (n >= 2)"
        )));
    }
    Ok(())
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

/// Decompose the shift from period A to period B on the mixture counterfactual.
///
/// `x_a` is `n_a × p` row-major, `x_b` is `n_b × p`. No randomness beyond the
/// deterministic booster validation split.
#[allow(clippy::too_many_arguments)]
pub fn explain_v5(
    x_a: &[f64],
    n_a: usize,
    y_a: &[f64],
    x_b: &[f64],
    n_b: usize,
    y_b: &[f64],
    p: usize,
    cfg: &ExplainV5Config,
) -> Result<ExplainV5Report> {
    check(x_a, n_a, p, y_a, "period A")?;
    check(x_b, n_b, p, y_b, "period B")?;
    if cfg.k_lmoments == 0 || cfg.n_quad == 0 || cfg.n_y < 2 {
        return Err(TraceError::InvalidArgument("bad ExplainV5Config".into()));
    }
    let t0 = Instant::now();
    let (mu_aa, mu_ab, mu_ba, mu_bb) = match cfg.mean_model {
        LocationModel::Booster => {
            let m_a = fit_mean(x_a, n_a, p, y_a, &cfg.mean)?;
            let m_b = fit_mean(x_b, n_b, p, y_b, &cfg.mean)?;
            (
                predict_mean(&m_a, x_a, n_a)?, // cov A, cond A
                predict_mean(&m_b, x_a, n_a)?, // cov A, cond B
                predict_mean(&m_a, x_b, n_b)?, // cov B, cond A
                predict_mean(&m_b, x_b, n_b)?, // cov B, cond B
            )
        }
        LocationModel::Ols => {
            let b_a = fit_ols(x_a, n_a, p, y_a)?;
            let b_b = fit_ols(x_b, n_b, p, y_b)?;
            (
                predict_ols(&b_a, x_a, n_a),
                predict_ols(&b_b, x_a, n_a),
                predict_ols(&b_a, x_b, n_b),
                predict_ols(&b_b, x_b, n_b),
            )
        }
    };
    let r_a: Vec<f64> = y_a.iter().zip(&mu_aa).map(|(y, m)| y - m).collect();
    let r_b: Vec<f64> = y_b.iter().zip(&mu_bb).map(|(y, m)| y - m).collect();
    let seconds_fit = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let mm = |v: &[f64]| {
        v.iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &z| {
                (lo.min(z), hi.max(z))
            })
    };
    let (ra_lo, ra_hi) = mm(&r_a);
    let (rb_lo, rb_hi) = mm(&r_b);
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for (mu, (rl, rh)) in [
        (&mu_aa, (ra_lo, ra_hi)),
        (&mu_ab, (rb_lo, rb_hi)),
        (&mu_ba, (ra_lo, ra_hi)),
        (&mu_bb, (rb_lo, rb_hi)),
    ] {
        let (ml, mh) = mm(mu);
        lo = lo.min(ml + rl);
        hi = hi.max(mh + rh);
    }
    let y_grid = support_grid(lo, hi, cfg.n_y)?;
    let gl = gauss_legendre_01(cfg.n_quad)?;
    let u = gl.nodes.clone();
    let k = cfg.k_lmoments;

    let mut sorted_ra = r_a.clone();
    sorted_ra.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut sorted_rb = r_b.clone();
    sorted_rb.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let resid_q = |s: &[f64], uu: f64| {
        // left-inverse of the empirical CDF
        let n = s.len();
        let idx = ((uu * n as f64).ceil() as usize).clamp(1, n) - 1;
        s[idx]
    };
    let resid_lmom = |s: &[f64]| {
        // exact L-moments of the empirical residual law (step quantile)
        let n = s.len();
        let grid: Vec<f64> = s.to_vec();
        let f: Vec<f64> = (1..=n).map(|i| i as f64 / n as f64).collect();
        // represent the step law on a fine pseudo-grid: duplicate points make linear
        // pieces vertical-free; use (y_i, F_{i-1}) and (y_i, F_i) pairs.
        let mut yg = Vec::with_capacity(2 * n);
        let mut fg = Vec::with_capacity(2 * n);
        for i in 0..n {
            yg.push(grid[i]);
            fg.push(if i == 0 { 0.0 } else { f[i - 1] });
            yg.push(grid[i]);
            fg.push(f[i]);
        }
        lmoments_from_cdf_grid(&yg, &fg, k)
    };
    let lam_ra = resid_lmom(&sorted_ra);
    let lam_rb = resid_lmom(&sorted_rb);

    let build = |mu: &[f64], r: &[f64], rs: &[f64], lam_r: &[f64]| -> Result<Hybrid> {
        let f = mixture_cdf_location(mu, r, &y_grid)?;
        let q_mix = quantile_from_cdf(&y_grid, &f, &u);
        let lmom = lmoments_from_cdf_grid(&y_grid, &f, k);
        let mbar = mean(mu);
        let q_bary: Vec<f64> = u.iter().map(|&uu| mbar + resid_q(rs, uu)).collect();
        let mut lmom_bary = lam_r.to_vec();
        lmom_bary[0] += mbar;
        let mixing_residual = q_mix
            .iter()
            .zip(&q_bary)
            .zip(&gl.weights)
            .map(|((a, b), w)| w * (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        Ok(Hybrid {
            q_mix,
            q_bary,
            lmom,
            lmom_bary,
            mixing_residual,
        })
    };
    let aa = build(&mu_aa, &r_a, &sorted_ra, &lam_ra)?;
    let ab = build(&mu_ab, &r_b, &sorted_rb, &lam_rb)?;
    let ba = build(&mu_ba, &r_a, &sorted_ra, &lam_ra)?;
    let bb = build(&mu_bb, &r_b, &sorted_rb, &lam_rb)?;
    let seconds_mixture = t1.elapsed().as_secs_f64();

    let lsplit_comp: Vec<f64> = (0..k)
        .map(|j| 0.5 * ((ba.lmom[j] - aa.lmom[j]) + (bb.lmom[j] - ab.lmom[j])))
        .collect();
    let lsplit_struct: Vec<f64> = (0..k)
        .map(|j| 0.5 * ((ab.lmom[j] - aa.lmom[j]) + (bb.lmom[j] - ba.lmom[j])))
        .collect();
    Ok(ExplainV5Report {
        u_nodes: u,
        y_grid,
        aa,
        ab,
        ba,
        bb,
        lsplit_comp,
        lsplit_struct,
        seconds_fit,
        seconds_mixture,
        seconds_total: t0.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
mod tests {
    use super::super::mixture::{mixture_cdf_quantile_rows, support_grid};
    use super::*;
    use crate::trace::shift::SplitMix64;

    fn normal(rng: &mut SplitMix64) -> f64 {
        let u1 = rng.next_f64().max(1e-300);
        let u2 = rng.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    fn draw_group(rng: &mut SplitMix64, probs: &[f64]) -> usize {
        let u = rng.next_f64();
        let mut c = 0.0;
        for (g, p) in probs.iter().enumerate() {
            c += p;
            if u < c {
                return g;
            }
        }
        probs.len() - 1
    }

    /// Toy check: the mixture counterfactual F_[B,A] (covariates from B, conditional
    /// law from A) equals DiNardo–Fortin–Lemieux (1996) reweighting of the A sample by
    /// w(x) = P_B(x) / P_A(x). Discrete x with group-specific (non-location) laws.
    #[test]
    fn mixture_matches_dfl_reweighting() {
        let mut rng = SplitMix64::new(1996);
        let pa = [0.5, 0.3, 0.2];
        let pb = [0.2, 0.3, 0.5];
        let law = |g: usize, rng: &mut SplitMix64| -> f64 {
            let z = normal(rng);
            match g {
                0 => z,
                1 => 1.5 + 0.5 * z * z, // skewed
                _ => -1.0 + 2.0 * z,    // wide
            }
        };
        let (na, nb) = (60_000, 60_000);
        let ga: Vec<usize> = (0..na).map(|_| draw_group(&mut rng, &pa)).collect();
        let ya: Vec<f64> = ga.iter().map(|&g| law(g, &mut rng)).collect();
        let gb: Vec<usize> = (0..nb).map(|_| draw_group(&mut rng, &pb)).collect();

        // DFL: weighted empirical CDF of A with estimated frequency ratios.
        let freq = |gs: &[usize]| {
            let mut c = [0.0f64; 3];
            for &g in gs {
                c[g] += 1.0;
            }
            c.map(|v| v / gs.len() as f64)
        };
        let (fa, fb) = (freq(&ga), freq(&gb));
        let w: Vec<f64> = ga.iter().map(|&g| fb[g] / fa[g]).collect();
        let wsum: f64 = w.iter().sum();

        // Mixture: per-group empirical conditional quantiles of A, rows indexed by B.
        let nq = 2000;
        let u: Vec<f64> = (0..nq).map(|i| (i as f64 + 0.5) / nq as f64).collect();
        let mut per_group: Vec<Vec<f64>> = vec![vec![]; 3];
        for (g, y) in ga.iter().zip(&ya) {
            per_group[*g].push(*y);
        }
        let q_of: Vec<Vec<f64>> = per_group
            .iter_mut()
            .map(|v| {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                u.iter()
                    .map(|&uu| v[((uu * v.len() as f64) as usize).min(v.len() - 1)])
                    .collect()
            })
            .collect();
        let mut q_rows = Vec::with_capacity(nb * nq);
        for &g in &gb {
            q_rows.extend_from_slice(&q_of[g]);
        }
        let grid = support_grid(-6.0, 8.0, 141).unwrap();
        let f_mix = mixture_cdf_quantile_rows(&u, &q_rows, nb, None, &grid).unwrap();
        let mut sup = 0.0f64;
        for (g, y) in grid.iter().enumerate() {
            let f_dfl: f64 = ya
                .iter()
                .zip(&w)
                .filter(|(yi, _)| **yi <= *y)
                .map(|(_, wi)| wi)
                .sum::<f64>()
                / wsum;
            sup = sup.max((f_mix[g] - f_dfl).abs());
        }
        assert!(sup < 5e-3, "sup |F_mix - F_DFL| = {sup}");
    }

    /// Same check through the full `explain_v5` path on a location model: the
    /// composition piece of λ1..λ3 matches a DFL-reweighting composition split.
    #[test]
    fn explain_comp_matches_dfl_on_location_model() {
        let mut rng = SplitMix64::new(7);
        let n = 40_000;
        let p = 1;
        let mu = |g: f64| [0.0, 1.0, 3.0][g as usize];
        let draw = |rng: &mut SplitMix64, probs: &[f64], n: usize| {
            let x: Vec<f64> = (0..n).map(|_| draw_group(rng, probs) as f64).collect();
            let y: Vec<f64> = x.iter().map(|&g| mu(g) + normal(rng)).collect();
            (x, y)
        };
        let (xa, ya) = draw(&mut rng, &[0.6, 0.3, 0.1], n);
        let (xb, yb) = draw(&mut rng, &[0.1, 0.3, 0.6], n);
        let rep = explain_v5(&xa, n, &ya, &xb, n, &yb, p, &ExplainV5Config::default()).unwrap();
        // DFL counterfactual λ's: weighted sample L-moments of y_A with w = P_B/P_A.
        let wmap = [0.1 / 0.6, 1.0, 6.0];
        let w: Vec<f64> = xa.iter().map(|&g| wmap[g as usize]).collect();
        let lam_w = weighted_lmoments(&ya, &w, 3);
        let lam_a = weighted_lmoments(&ya, &vec![1.0; n], 3);
        let lam_b = weighted_lmoments(&yb, &vec![1.0; n], 3);
        // No structural change: comp ≈ total, DFL comp (C - A) ≈ total too.
        for j in 0..3 {
            let dfl_comp = lam_w[j] - lam_a[j];
            let tot = lam_b[j] - lam_a[j];
            assert!(
                (rep.lsplit_comp[j] - dfl_comp).abs() < 0.03,
                "λ{} comp v5={} dfl={} total={}",
                j + 1,
                rep.lsplit_comp[j],
                dfl_comp,
                tot
            );
            assert!(rep.lsplit_struct[j].abs() < 0.03);
        }
    }

    fn weighted_lmoments(y: &[f64], w: &[f64], k: usize) -> Vec<f64> {
        let mut idx: Vec<usize> = (0..y.len()).collect();
        idx.sort_by(|&a, &b| y[a].partial_cmp(&y[b]).unwrap());
        let ws: f64 = w.iter().sum();
        let mut yg = Vec::new();
        let mut fg = Vec::new();
        let mut c = 0.0;
        for &i in &idx {
            yg.push(y[i]);
            fg.push(c / ws);
            c += w[i];
            yg.push(y[i]);
            fg.push(c / ws);
        }
        lmoments_from_cdf_grid(&yg, &fg, k)
    }

    /// PREREG_V5 §6 kill criterion 3: on a two-component location mixture the mixing
    /// residual is non-zero while barycenter shape channels (λ2+) stay flat.
    #[test]
    fn mixing_residual_nonzero_while_barycenter_flat() {
        let mut rng = SplitMix64::new(11);
        let n = 20_000;
        let mk = |rng: &mut SplitMix64, pr: f64| {
            let x: Vec<f64> = (0..n)
                .map(|_| if rng.next_f64() < pr { 1.0 } else { 0.0 })
                .collect();
            let y: Vec<f64> = x.iter().map(|&g| 4.0 * g - 2.0 + normal(rng)).collect();
            (x, y)
        };
        let (xa, ya) = mk(&mut rng, 0.5);
        let (xb, yb) = mk(&mut rng, 0.1);
        let rep = explain_v5(&xa, n, &ya, &xb, n, &yb, 1, &ExplainV5Config::default()).unwrap();
        assert!(rep.aa.mixing_residual > 0.5, "{}", rep.aa.mixing_residual);
        // barycenter shape channels identical across covariate switches
        for j in 1..4 {
            assert!((rep.ba.lmom_bary[j] - rep.aa.lmom_bary[j]).abs() < 1e-12);
        }
        // mixture shape channels do move (λ3: 50/50 symmetric → 10/90 skewed)
        assert!((rep.ba.lmom[2] - rep.aa.lmom[2]).abs() > 0.1);
        // budget is exact
        for j in 0..4 {
            let tot = rep.bb.lmom[j] - rep.aa.lmom[j];
            assert!((rep.lsplit_comp[j] + rep.lsplit_struct[j] - tot).abs() < 1e-12);
        }
    }

    #[test]
    fn ols_recovers_linear_coefficients() {
        let mut rng = SplitMix64::new(51);
        let (n, p) = (4000, 3);
        let x: Vec<f64> = (0..n * p).map(|_| normal(&mut rng)).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 0.7 + 1.5 * x[i * p] - 2.0 * x[i * p + 1] + 0.25 * x[i * p + 2])
            .collect();
        let b = fit_ols(&x, n, p, &y).unwrap();
        for (got, want) in b.iter().zip([0.7, 1.5, -2.0, 0.25]) {
            assert!((got - want).abs() < 1e-10, "{b:?}");
        }
        let yh = predict_ols(&b, &x, n);
        assert!(yh.iter().zip(&y).all(|(a, c)| (a - c).abs() < 1e-9));
        // residuals orthogonal to the design under noise
        let yn: Vec<f64> = y.iter().map(|v| v + normal(&mut rng)).collect();
        let b = fit_ols(&x, n, p, &yn).unwrap();
        let r: Vec<f64> = yn
            .iter()
            .zip(predict_ols(&b, &x, n))
            .map(|(a, c)| a - c)
            .collect();
        assert!(r.iter().sum::<f64>().abs() < 1e-8);
        for j in 0..p {
            let d: f64 = (0..n).map(|i| r[i] * x[i * p + j]).sum();
            assert!(d.abs() < 1e-8, "col {j}: {d}");
        }
    }

    #[test]
    fn ols_rejects_rank_deficient_design() {
        let n = 50;
        let x: Vec<f64> = (0..n).flat_map(|i| [i as f64, 2.0 * i as f64]).collect();
        let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
        assert!(fit_ols(&x, n, 2, &y).is_err());
        assert!(fit_ols(&x[..4], 2, 2, &y[..2]).is_err());
    }

    /// v5.1 path: linear location model, comp/struct sum to the total and a pure
    /// covariate shift lands in composition.
    #[test]
    fn v5_1_pure_composition_shift() {
        let mut rng = SplitMix64::new(7);
        let (n, p) = (4000, 2);
        let mk = |rng: &mut SplitMix64, shift: f64| {
            let x: Vec<f64> = (0..n * p)
                .map(|k| {
                    let z = normal(rng);
                    if k % p == 0 {
                        shift + z * (1.0 + shift)
                    } else {
                        z
                    }
                })
                .collect();
            let y: Vec<f64> = (0..n)
                .map(|i| 1.0 + 2.0 * x[i * p] - x[i * p + 1] + 0.5 * normal(rng))
                .collect();
            (x, y)
        };
        let (xa, ya) = mk(&mut rng, 0.0);
        let (xb, yb) = mk(&mut rng, 0.5);
        let cfg = ExplainV5Config {
            mean_model: LocationModel::Ols,
            ..Default::default()
        };
        let rep = explain_v5(&xa, n, &ya, &xb, n, &yb, p, &cfg).unwrap();
        for j in 0..2 {
            let tot = rep.bb.lmom[j] - rep.aa.lmom[j];
            assert!((rep.lsplit_comp[j] + rep.lsplit_struct[j] - tot).abs() < 1e-10);
            assert!(
                rep.lsplit_comp[j].abs() > 5.0 * rep.lsplit_struct[j].abs(),
                "{j}: {:?} {:?}",
                rep.lsplit_comp,
                rep.lsplit_struct
            );
        }
        // default stays v5 (booster) and differs from the OLS path
        let rep0 = explain_v5(&xa, n, &ya, &xb, n, &yb, p, &ExplainV5Config::default()).unwrap();
        assert!(rep0.lsplit_comp != rep.lsplit_comp);
    }
}
