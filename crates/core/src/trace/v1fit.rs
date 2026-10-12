//! Newton pinball fit in the orthonormal shifted-Legendre basis.
//!
//! The v0 booster is untouched. This module is the v1 parameterisation:
//! Gauss–Legendre nodes, block Newton steps, a user-chosen reporting basis,
//! and a separate monotonisation adjustment.

#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use super::legendre::hosking_lmoments;
use super::legendre::{
    apply_report, bernstein_nonnegative, eval_a, k3_non_representable, lambda_to_a,
    min_derivative_grid, report_matrix, weighted_pav, wnorm, QuadBasis,
};
use crate::error::{Result, TraceError};
use crate::linalg::solve_spd;
use crate::scalable::binning::{bin_column_into, edges_from_sorted};

/// How monotonicity is enforced after the Newton fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MonoMode {
    /// No projection.
    None,
    /// Weighted PAV on the fitting nodes. Attribution stays pre-projection.
    Pav,
    /// Sufficient Bernstein margin: lift the intercept scale so every bin
    /// combination has a nonnegative derivative. Conservative.
    Bernstein,
}

/// Reporting basis. The fit itself is always orthonormal Legendre.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReportKind {
    Legendre = 0,
    Hermite = 1,
    Reference = 2,
}

#[derive(Clone, Debug)]
pub struct V1Config {
    pub k_fit: usize,
    pub k_report: usize,
    /// `0` selects `max(k_fit, 2 * k_fit.min(8))` and at least `k_fit`.
    pub n_u: usize,
    pub max_bins: usize,
    pub max_rounds: usize,
    pub tol: f64,
    pub min_count: u64,
    pub ridge: f64,
    pub learning_rate: f64,
    pub mono: MonoMode,
    pub bernstein_elevation: usize,
    pub reporting: ReportKind,
    /// Optional reference quantile curve on the fine grid (length 1024). Empty uses `z`.
    pub reference: Vec<f64>,
}

impl Default for V1Config {
    fn default() -> Self {
        Self {
            k_fit: 6,
            k_report: 4,
            n_u: 0,
            max_bins: 16,
            max_rounds: 30,
            tol: 1e-9,
            min_count: 1,
            ridge: 1e-6,
            learning_rate: 1.0,
            mono: MonoMode::Pav,
            bernstein_elevation: 4,
            reporting: ReportKind::Legendre,
            reference: Vec::new(),
        }
    }
}

/// Fitted additive Legendre model. Effects are centred under the training law.
#[derive(Clone, Debug)]
pub struct V1Model {
    pub k_fit: usize,
    pub k_report: usize,
    pub basis: QuadBasis,
    pub edges: Vec<Vec<f64>>,
    pub n_bins: Vec<usize>,
    pub intercept: Vec<f64>,
    /// `effects[j][b * k_fit + c]`, including the missing-value bin at `b = 0`.
    pub effects: Vec<Vec<f64>>,
    pub train_counts: Vec<Vec<u64>>,
    /// `k_fit × k_report`, row-major.
    pub report: Vec<f64>,
    pub reporting: ReportKind,
    pub mono: MonoMode,
    pub bernstein_elevation: usize,
    /// K = 3 and `|τ_3| > 1/5` on the intercept, so no monotone 3-term representation.
    pub non_representable: bool,
    /// Scale added to the intercept's `e_1` coefficient by the Bernstein margin.
    pub bernstein_lift: f64,
    pub train_loss: f64,
    pub rounds: usize,
}

/// Per-row Shapley shares in orthonormal units, plus the truncation and PAV lines.
#[derive(Clone, Debug)]
pub struct Attribution {
    /// `n × (p+1) × k_fit`, player 0 is the intercept. Sums to the pre-projection W2².
    pub phi_fit: Vec<f64>,
    /// `n × (p+1) × k_report` in the reporting basis.
    pub phi_report: Vec<f64>,
    /// Energy outside the reporting subspace, per row. Unclipped.
    pub truncation: Vec<f64>,
    /// `‖ΠQ‖² − ‖Q‖²` against the zero curve, per row. In `[-2 D d, 0]`.
    pub mono_adjustment: Vec<f64>,
    /// `‖Q − ΠQ‖` on the fitting nodes, per row.
    pub d: Vec<f64>,
    pub n: usize,
    pub p: usize,
    pub k_fit: usize,
    pub k_report: usize,
}

pub fn fit_v1(x: &[f64], n: usize, p: usize, y: &[f64], cfg: &V1Config) -> Result<V1Model> {
    if n < 2 || p < 1 {
        return Err(TraceError::InvalidArgument(
            "need at least 2 rows and 1 feature".into(),
        ));
    }
    if x.len() != n * p || y.len() != n {
        return Err(TraceError::Shape("X and y shapes disagree".into()));
    }
    if y.iter().any(|v| !v.is_finite()) || x.iter().any(|v| v.is_infinite()) {
        return Err(TraceError::InvalidArgument(
            "y must be finite and X must not contain infinities".into(),
        ));
    }
    if cfg.k_fit < 1 || cfg.k_fit > 12 || cfg.k_report < 1 || cfg.k_report > cfg.k_fit {
        return Err(TraceError::InvalidArgument(
            "K_fit must be in 1..=12 and K_report in 1..=K_fit".into(),
        ));
    }
    let n_u = if cfg.n_u == 0 {
        cfg.k_fit.max(12).max(cfg.k_fit)
    } else {
        cfg.n_u
    };
    let basis = QuadBasis::new(n_u, cfg.k_fit)?;
    let (codes, edges, n_bins) = bin_features(x, n, p, cfg.max_bins)?;
    let mut model = V1Model {
        k_fit: cfg.k_fit,
        k_report: cfg.k_report,
        basis,
        edges,
        n_bins,
        intercept: vec![0.0; cfg.k_fit],
        effects: Vec::new(),
        train_counts: Vec::new(),
        report: report_matrix(cfg.k_fit, cfg.k_report, cfg.reporting as u8, &cfg.reference)?,
        reporting: cfg.reporting,
        mono: cfg.mono,
        bernstein_elevation: cfg.bernstein_elevation,
        non_representable: false,
        bernstein_lift: 0.0,
        train_loss: 0.0,
        rounds: 0,
    };
    for j in 0..p {
        model.effects.push(vec![0.0; model.n_bins[j] * cfg.k_fit]);
        model.train_counts.push(vec![0; model.n_bins[j]]);
    }
    for j in 0..p {
        for i in 0..n {
            let b = codes[j][i] as usize;
            model.train_counts[j][b] += 1;
        }
    }
    if y.len() >= cfg.k_fit {
        let lam = hosking_lmoments(y, cfg.k_fit)?;
        model.intercept = lambda_to_a(&lam);
    } else {
        model.intercept[0] = y.iter().sum::<f64>() / n as f64;
    }
    let mut a_rows = vec![0.0; n * cfg.k_fit];
    scatter_all(&model, &codes, &mut a_rows);
    let mut prev = pinball_loss(&model.basis, &a_rows, y);
    for round in 0..cfg.max_rounds {
        newton_round(&mut model, &codes, y, &mut a_rows, cfg)?;
        let loss = pinball_loss(&model.basis, &a_rows, y);
        model.rounds = round + 1;
        let drop = (prev - loss) / (prev.abs() + 1.0);
        prev = loss;
        if drop < cfg.tol && round + 1 >= 2 {
            break;
        }
    }
    model.train_loss = prev;
    if cfg.k_fit == 3 {
        model.non_representable = k3_non_representable(&model.intercept);
    }
    if cfg.mono == MonoMode::Bernstein && !model.non_representable {
        model.bernstein_lift = bernstein_margin(&mut model, cfg.bernstein_elevation);
        scatter_all(&model, &codes, &mut a_rows);
        model.train_loss = pinball_loss(&model.basis, &a_rows, y);
    }
    Ok(model)
}

fn newton_round(
    model: &mut V1Model,
    codes: &[Vec<u16>],
    y: &[f64],
    a_rows: &mut [f64],
    cfg: &V1Config,
) -> Result<()> {
    let n = y.len();
    let k = model.k_fit;
    let p = codes.len();
    let (g, h) = accumulate_block(model, a_rows, y, None, codes);
    let step = solve_block(&h, &g, k, cfg.ridge, n as f64)?;
    linesearch_intercept(model, a_rows, y, &step, cfg.learning_rate);
    for j in 0..p {
        let nb = model.n_bins[j];
        let mut g_bins = vec![0.0; nb * k];
        let mut h_bins = vec![0.0; nb * k * k];
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); nb];
        for i in 0..n {
            let b = codes[j][i] as usize;
            members[b].push(i);
            let (gi, hi) = row_gh(model, &a_rows[i * k..(i + 1) * k], y[i]);
            for c in 0..k {
                g_bins[b * k + c] += gi[c];
            }
            for t in 0..k * k {
                h_bins[b * k * k + t] += hi[t];
            }
        }
        for b in 0..nb {
            let count = members[b].len() as u64;
            if count < cfg.min_count {
                continue;
            }
            let step = solve_block(
                &h_bins[b * k * k..(b + 1) * k * k],
                &g_bins[b * k..(b + 1) * k],
                k,
                cfg.ridge,
                count as f64,
            )?;
            // Bins of one feature partition the rows, so each block is scored
            // on its own rows. A step that raises pinball is shortened, not
            // allowed to reject the other blocks.
            linesearch_bin(
                model,
                a_rows,
                y,
                j,
                b,
                &members[b],
                &step,
                cfg.learning_rate,
            );
        }
        center_feature(model, j, a_rows, codes);
    }
    Ok(())
}

fn linesearch_intercept(
    model: &mut V1Model,
    a_rows: &mut [f64],
    y: &[f64],
    step: &[f64],
    lr0: f64,
) {
    let base = pinball_loss(&model.basis, a_rows, y);
    let saved_a = a_rows.to_vec();
    let saved_i = model.intercept.clone();
    let mut lr = lr0;
    loop {
        let scaled = scale(step, lr);
        apply_intercept(model, a_rows, &scaled);
        let loss = pinball_loss(&model.basis, a_rows, y);
        if loss <= base * (1.0 + 1e-12) + 1e-15 {
            return;
        }
        a_rows.copy_from_slice(&saved_a);
        model.intercept.clone_from(&saved_i);
        lr *= 0.5;
        if lr < 1e-4 {
            return;
        }
    }
}

fn linesearch_bin(
    model: &mut V1Model,
    a_rows: &mut [f64],
    y: &[f64],
    feature: usize,
    bin: usize,
    rows: &[usize],
    step: &[f64],
    lr0: f64,
) {
    let k = model.k_fit;
    let base = partial_pinball(&model.basis, a_rows, y, rows);
    let saved_effect: Vec<f64> = model.effects[feature][bin * k..(bin + 1) * k].to_vec();
    let saved_rows: Vec<f64> = rows
        .iter()
        .flat_map(|&i| a_rows[i * k..(i + 1) * k].to_vec())
        .collect();
    let mut lr = lr0;
    loop {
        let scaled = scale(step, lr);
        for c in 0..k {
            model.effects[feature][bin * k + c] = saved_effect[c] + scaled[c];
        }
        for (t, &i) in rows.iter().enumerate() {
            for c in 0..k {
                a_rows[i * k + c] = saved_rows[t * k + c] + scaled[c];
            }
        }
        let loss = partial_pinball(&model.basis, a_rows, y, rows);
        if loss <= base * (1.0 + 1e-12) + 1e-15 {
            return;
        }
        model.effects[feature][bin * k..(bin + 1) * k].copy_from_slice(&saved_effect);
        for (t, &i) in rows.iter().enumerate() {
            a_rows[i * k..(i + 1) * k].copy_from_slice(&saved_rows[t * k..(t + 1) * k]);
        }
        lr *= 0.5;
        if lr < 1e-4 {
            return;
        }
    }
}

fn partial_pinball(basis: &QuadBasis, a_rows: &[f64], y: &[f64], rows: &[usize]) -> f64 {
    let k = basis.k;
    let mut loss = 0.0;
    for &i in rows {
        for t in 0..basis.n_u() {
            let mut q = 0.0;
            for c in 0..k {
                q += a_rows[i * k + c] * basis.e[t * k + c];
            }
            let u = basis.nodes[t];
            let r = y[i] - q;
            let rho = if r >= 0.0 { u * r } else { (u - 1.0) * r };
            loss += basis.weights[t] * rho;
        }
    }
    loss
}

fn scale(step: &[f64], lr: f64) -> Vec<f64> {
    let mut s = step.iter().map(|v| v * lr).collect::<Vec<_>>();
    let max_abs = s.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    if max_abs > 8.0 {
        let f = 8.0 / max_abs;
        for v in &mut s {
            *v *= f;
        }
    }
    s
}

fn solve_block(h: &[f64], g: &[f64], k: usize, ridge: f64, count: f64) -> Result<Vec<f64>> {
    let mut a = h.to_vec();
    let diag = ridge * count.max(1.0);
    for i in 0..k {
        a[i * k + i] += diag;
    }
    let step = match solve_spd(&a, k, g) {
        Ok(sol) => sol.into_iter().map(|v| -v).collect::<Vec<_>>(),
        Err(_) => g.iter().map(|v| -v / (diag + 1.0)).collect(),
    };
    Ok(step)
}

fn row_gh(model: &V1Model, a: &[f64], y: f64) -> (Vec<f64>, Vec<f64>) {
    let k = model.k_fit;
    let nu = model.basis.n_u();
    let mut g = vec![0.0; k];
    let mut h = vec![0.0; k * k];
    for t in 0..nu {
        let mut q = 0.0;
        let mut qp = 0.0;
        for c in 0..k {
            q += a[c] * model.basis.e[t * k + c];
            qp += a[c] * model.basis.de[t * k + c];
        }
        let w = model.basis.weights[t];
        let ind = if y < q { 1.0 } else { 0.0 };
        let dens = if qp > 1e-3 {
            (1.0 / qp).clamp(1e-2, 40.0)
        } else {
            1.0
        };
        for c in 0..k {
            g[c] += w * (ind - model.basis.nodes[t]) * model.basis.e[t * k + c];
        }
        for c in 0..k {
            for d in 0..k {
                h[c * k + d] += w * dens * model.basis.e[t * k + c] * model.basis.e[t * k + d];
            }
        }
    }
    (g, h)
}

fn accumulate_block(
    model: &V1Model,
    a_rows: &[f64],
    y: &[f64],
    _feature: Option<usize>,
    _codes: &[Vec<u16>],
) -> (Vec<f64>, Vec<f64>) {
    let k = model.k_fit;
    let n = y.len();
    let mut g = vec![0.0; k];
    let mut h = vec![0.0; k * k];
    for i in 0..n {
        let (gi, hi) = row_gh(model, &a_rows[i * k..(i + 1) * k], y[i]);
        for c in 0..k {
            g[c] += gi[c];
        }
        for t in 0..k * k {
            h[t] += hi[t];
        }
    }
    (g, h)
}

fn apply_intercept(model: &mut V1Model, a_rows: &mut [f64], step: &[f64]) {
    let k = model.k_fit;
    let n = a_rows.len() / k;
    for c in 0..k {
        model.intercept[c] += step[c];
    }
    for i in 0..n {
        for c in 0..k {
            a_rows[i * k + c] += step[c];
        }
    }
}

fn center_feature(model: &mut V1Model, j: usize, _a_rows: &mut [f64], _codes: &[Vec<u16>]) {
    let k = model.k_fit;
    let mut total = 0.0_f64;
    let mut mean = vec![0.0; k];
    for (b, &cnt) in model.train_counts[j].iter().enumerate() {
        total += cnt as f64;
        for c in 0..k {
            mean[c] += cnt as f64 * model.effects[j][b * k + c];
        }
    }
    if total == 0.0 {
        return;
    }
    for c in 0..k {
        mean[c] /= total;
        model.intercept[c] += mean[c];
    }
    let nb = model.n_bins[j];
    for b in 0..nb {
        for c in 0..k {
            model.effects[j][b * k + c] -= mean[c];
        }
    }
}

fn scatter_all(model: &V1Model, codes: &[Vec<u16>], a_rows: &mut [f64]) {
    let n = a_rows.len() / model.k_fit;
    let k = model.k_fit;
    for i in 0..n {
        for c in 0..k {
            a_rows[i * k + c] = model.intercept[c];
        }
        for (j, col) in codes.iter().enumerate() {
            let b = col[i] as usize;
            for c in 0..k {
                a_rows[i * k + c] += model.effects[j][b * k + c];
            }
        }
    }
}

fn pinball_loss(basis: &QuadBasis, a_rows: &[f64], y: &[f64]) -> f64 {
    let n = y.len();
    let k = basis.k;
    let mut loss = 0.0;
    for i in 0..n {
        for t in 0..basis.n_u() {
            let mut q = 0.0;
            for c in 0..k {
                q += a_rows[i * k + c] * basis.e[t * k + c];
            }
            let u = basis.nodes[t];
            let r = y[i] - q;
            let rho = if r >= 0.0 { u * r } else { (u - 1.0) * r };
            loss += basis.weights[t] * rho;
        }
    }
    loss / n as f64
}

/// Lift the intercept's scale channel so the worst bin combination stays nonnegative.
fn bernstein_margin(model: &mut V1Model, elevation: usize) -> f64 {
    let k = model.k_fit;
    let p = model.effects.len();
    let mut worst_features = 0.0;
    for j in 0..p {
        let mut worst_j = 0.0_f64;
        let nb = model.n_bins[j];
        for b in 0..nb {
            if model.train_counts[j][b] == 0 && b == 0 {
                continue;
            }
            let coef = &model.effects[j][b * k..(b + 1) * k];
            let m = min_derivative_grid(coef, 401);
            worst_j = worst_j.min(m);
        }
        worst_features += worst_j.min(0.0);
    }
    let need = -worst_features;
    let current = min_derivative_grid(&model.intercept, 401);
    if current >= need - 1e-10 {
        return 0.0;
    }
    // e_1' = √3 * 2.
    let slope = 2.0 * 3.0_f64.sqrt();
    let delta = (need - current) / slope + 1e-8;
    if k > 1 {
        model.intercept[1] += delta;
    }
    let _ = elevation;
    delta
}

fn bin_features(
    x: &[f64],
    n: usize,
    p: usize,
    max_bins: usize,
) -> Result<(Vec<Vec<u16>>, Vec<Vec<f64>>, Vec<usize>)> {
    let mut codes = Vec::with_capacity(p);
    let mut edges = Vec::with_capacity(p);
    let mut n_bins = Vec::with_capacity(p);
    let mut col = vec![0.0; n];
    for j in 0..p {
        for i in 0..n {
            col[i] = x[i * p + j];
        }
        let mut finite: Vec<f64> = col.iter().copied().filter(|v| v.is_finite()).collect();
        finite.sort_by(f64::total_cmp);
        let e = edges_from_sorted(&finite, max_bins)?;
        let nb = e.len() + 2;
        let mut c = vec![0u16; n];
        bin_column_into(&col, &e, &mut c)?;
        codes.push(c);
        n_bins.push(nb);
        edges.push(e);
    }
    Ok((codes, edges, n_bins))
}

/// Bin new rows with the training edges. `X` is row-major `n × p`.
pub fn codes_of(model: &V1Model, x: &[f64], n: usize) -> Result<Vec<Vec<u16>>> {
    let p = model.edges.len();
    if x.len() != n * p {
        return Err(TraceError::Shape("X has the wrong shape".into()));
    }
    let mut codes = Vec::with_capacity(p);
    let mut col = vec![0.0; n];
    for j in 0..p {
        for i in 0..n {
            col[i] = x[i * p + j];
        }
        let mut c = vec![0u16; n];
        bin_column_into(&col, &model.edges[j], &mut c)?;
        codes.push(c);
    }
    Ok(codes)
}

/// Player coefficients in the fitting basis: `n × (p+1) × k_fit`, player 0 = intercept.
pub fn player_coefficients(model: &V1Model, x: &[f64], n: usize) -> Result<Vec<f64>> {
    let codes = codes_of(model, x, n)?;
    let p = codes.len();
    let k = model.k_fit;
    let mut out = vec![0.0; n * (p + 1) * k];
    for i in 0..n {
        for c in 0..k {
            out[i * (p + 1) * k + c] = model.intercept[c];
        }
        for j in 0..p {
            let b = codes[j][i] as usize;
            for c in 0..k {
                out[i * (p + 1) * k + (j + 1) * k + c] = model.effects[j][b * k + c];
            }
        }
    }
    Ok(out)
}

/// Summed curve coefficients in the fitting basis, `n × k_fit`.
pub fn curve_coefficients(players: &[f64], n: usize, p: usize, k: usize) -> Vec<f64> {
    let mut a = vec![0.0; n * k];
    for i in 0..n {
        for j in 0..=p {
            for c in 0..k {
                a[i * k + c] += players[i * (p + 1) * k + j * k + c];
            }
        }
    }
    a
}

pub fn attribute(model: &V1Model, x: &[f64], n: usize) -> Result<Attribution> {
    let p = model.edges.len();
    let k = model.k_fit;
    let kr = model.k_report;
    let players = player_coefficients(model, x, n)?;
    let curves = curve_coefficients(&players, n, p, k);
    let mut phi_fit = vec![0.0; n * (p + 1) * k];
    let mut phi_report = vec![0.0; n * (p + 1) * kr];
    let mut truncation = vec![0.0; n];
    let mut mono_adjustment = vec![0.0; n];
    let mut d = vec![0.0; n];
    for i in 0..n {
        let a = &curves[i * k..(i + 1) * k];
        for c in 0..k {
            let total = a[c];
            for j in 0..=p {
                let aj = players[i * (p + 1) * k + j * k + c];
                phi_fit[i * (p + 1) * k + j * k + c] = aj * total;
            }
        }
        let reported = apply_report(a, &model.report, kr);
        let energy: f64 = a.iter().map(|v| v * v).sum();
        let reported_energy: f64 = reported.iter().map(|v| v * v).sum();
        truncation[i] = energy - reported_energy;
        for j in 0..=p {
            let aj = &players[i * (p + 1) * k + j * k..(i * (p + 1) * k + (j + 1) * k)];
            let rj = apply_report(aj, &model.report, kr);
            for m in 0..kr {
                phi_report[i * (p + 1) * kr + j * kr + m] = rj[m] * reported[m];
            }
        }
        let (dist, adj) = mono_stats(model, a);
        d[i] = dist;
        mono_adjustment[i] = adj;
    }
    Ok(Attribution {
        phi_fit,
        phi_report,
        truncation,
        mono_adjustment,
        d,
        n,
        p,
        k_fit: k,
        k_report: kr,
    })
}

fn mono_stats(model: &V1Model, a: &[f64]) -> (f64, f64) {
    let nu = model.basis.n_u();
    let k = model.k_fit;
    let mut q = vec![0.0; nu];
    for t in 0..nu {
        for c in 0..k {
            q[t] += a[c] * model.basis.e[t * k + c];
        }
    }
    let zeros = vec![0.0; nu];
    let pav = weighted_pav(&q, &model.basis.weights).unwrap_or(q.clone());
    let d = wnorm(&q, &pav, &model.basis.weights);
    let big_d = wnorm(&q, &zeros, &model.basis.weights);
    let d_pi = wnorm(&pav, &zeros, &model.basis.weights);
    (d, d_pi * d_pi - big_d * big_d)
}

/// Quantile curves on an arbitrary `u` grid. Returns `n × n_u` row-major.
/// `project` applies weighted PAV when the model asks for it and `u` is the fitting grid.
pub fn predict_quantiles(
    model: &V1Model,
    x: &[f64],
    n: usize,
    u: &[f64],
    project: bool,
) -> Result<Vec<f64>> {
    let p = model.edges.len();
    let k = model.k_fit;
    let players = player_coefficients(model, x, n)?;
    let curves = curve_coefficients(&players, n, p, k);
    let nu = u.len();
    let mut out = vec![0.0; n * nu];
    let on_fit = u.len() == model.basis.nodes.len()
        && u.iter()
            .zip(&model.basis.nodes)
            .all(|(a, b)| (a - b).abs() < 1e-14);
    for i in 0..n {
        let a = &curves[i * k..(i + 1) * k];
        for (t, &ui) in u.iter().enumerate() {
            out[i * nu + t] = eval_a(a, ui);
        }
        if project && model.mono == MonoMode::Pav && on_fit {
            let slice = &out[i * nu..(i + 1) * nu];
            if let Ok(z) = weighted_pav(slice, &model.basis.weights) {
                out[i * nu..(i + 1) * nu].copy_from_slice(&z);
            }
        }
    }
    Ok(out)
}

/// Reporting-basis coefficients of each feature's centred effect, shape `p × n_bins × k_report`
/// flattened feature-major. Bin 0 is the missing bin.
pub fn feature_report_table(model: &V1Model) -> Vec<Vec<f64>> {
    let k = model.k_fit;
    let kr = model.k_report;
    model
        .effects
        .iter()
        .map(|eff| {
            let nb = eff.len() / k;
            let mut out = vec![0.0; nb * kr];
            for b in 0..nb {
                let r = apply_report(&eff[b * k..(b + 1) * k], &model.report, kr);
                out[b * kr..(b + 1) * kr].copy_from_slice(&r);
            }
            out
        })
        .collect()
}

/// Squared mean of value-bin coefficients in one reporting channel (bin 0 excluded).
pub fn channel_energy(table: &[f64], k_report: usize, channel: usize) -> f64 {
    let nb = table.len() / k_report;
    if nb <= 1 {
        return 0.0;
    }
    let mut s = 0.0;
    for b in 1..nb {
        let v = table[b * k_report + channel];
        s += v * v;
    }
    s / (nb - 1) as f64
}

/// Ordinary-least-squares slope of value-bin reporting coefficients against the
/// mean covariate in each occupied bin.
pub fn channel_slope(model: &V1Model, feature: usize, channel: usize, x: &[f64]) -> Result<f64> {
    let n = x.len();
    let k = model.k_fit;
    let kr = model.k_report;
    if feature >= model.edges.len() {
        return Err(TraceError::InvalidArgument(
            "feature index out of range".into(),
        ));
    }
    let codes = crate::scalable::binning::bin_column::<u16>(x, &model.edges[feature])?;
    if codes.len() != n {
        return Err(TraceError::Shape("x length does not match".into()));
    }
    let nb = model.n_bins[feature];
    let mut sw = vec![0.0; nb];
    let mut sx = vec![0.0; nb];
    for (i, &c) in codes.iter().enumerate() {
        let b = c as usize;
        if b == 0 || !x[i].is_finite() {
            continue;
        }
        sw[b] += 1.0;
        sx[b] += x[i];
    }
    let eff = &model.effects[feature];
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for b in 1..nb {
        if sw[b] == 0.0 {
            continue;
        }
        let coef = &eff[b * k..(b + 1) * k];
        let r = apply_report(coef, &model.report, kr);
        xs.push(sx[b] / sw[b]);
        ys.push(r[channel]);
    }
    Ok(ols_slope(&xs, &ys))
}

fn ols_slope(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for (xi, yi) in x.iter().zip(y) {
        num += (xi - mx) * (yi - my);
        den += (xi - mx) * (xi - mx);
    }
    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}

/// Pair of pre-projection curves: distances and the monotonisation adjustment.
pub fn projection_pair(q1: &[f64], q2: &[f64], w: &[f64]) -> Result<ProjectionPair> {
    let p1 = weighted_pav(q1, w)?;
    let p2 = weighted_pav(q2, w)?;
    let d = wnorm(q1, q2, w);
    let d_pi = wnorm(&p1, &p2, w);
    let d_a = wnorm(q1, &p1, w);
    let d_b = wnorm(q2, &p2, w);
    Ok(ProjectionPair {
        d,
        d_pi,
        d_a,
        d_b,
        adjustment: d_pi * d_pi - d * d,
    })
}

#[derive(Clone, Copy, Debug)]
pub struct ProjectionPair {
    pub d: f64,
    pub d_pi: f64,
    pub d_a: f64,
    pub d_b: f64,
    pub adjustment: f64,
}

/// Bins populated in `codes_b` that were empty in training, per feature.
pub fn overlap_failures(model: &V1Model, codes_b: &[Vec<u16>]) -> Vec<Vec<usize>> {
    codes_b
        .iter()
        .enumerate()
        .map(|(j, col)| {
            let mut seen = vec![false; model.n_bins[j]];
            for &c in col {
                let b = c as usize;
                if b < seen.len() && model.train_counts[j][b] == 0 {
                    seen[b] = true;
                }
            }
            seen.iter()
                .enumerate()
                .filter_map(|(b, on)| if *on { Some(b) } else { None })
                .collect()
        })
        .collect()
}

/// Derivative sign check of every training-support bin combination is not feasible
/// in `p`. This checks the stored intercept and each training bin, plus the
/// sufficient-margin certificate when Bernstein mode was used.
pub fn monotonicity_report(model: &V1Model, grid: usize) -> MonoReport {
    let k = model.k_fit;
    let mut worst = f64::INFINITY;
    let mut n_checked = 0usize;
    let mut n_mono = 0usize;
    worst = worst.min(min_derivative_grid(&model.intercept, grid));
    n_checked += 1;
    if min_derivative_grid(&model.intercept, grid) >= -1e-8 {
        n_mono += 1;
    }
    for (j, eff) in model.effects.iter().enumerate() {
        for b in 0..model.n_bins[j] {
            if model.train_counts[j][b] == 0 {
                continue;
            }
            let mut a = model.intercept.clone();
            for c in 0..k {
                a[c] += eff[b * k + c];
            }
            // Single-feature slice, not a full combination. Full combinations are
            // certified by the Bernstein margin when that mode is on.
            let m = min_derivative_grid(&a, grid.min(501));
            worst = worst.min(m);
            n_checked += 1;
            if m >= -1e-8 {
                n_mono += 1;
            }
        }
    }
    let certified = model.mono == MonoMode::Bernstein
        && !model.non_representable
        && bernstein_nonnegative(&model.intercept, model.bernstein_elevation);
    MonoReport {
        worst_derivative: worst,
        non_representable: model.non_representable,
        bernstein_certified_intercept: certified,
        rows_checked: n_checked,
        rows_monotone: n_mono,
    }
}

#[derive(Clone, Debug)]
pub struct MonoReport {
    pub worst_derivative: f64,
    pub non_representable: bool,
    pub bernstein_certified_intercept: bool,
    pub rows_checked: usize,
    pub rows_monotone: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_small() -> V1Config {
        V1Config {
            k_fit: 4,
            k_report: 4,
            n_u: 8,
            max_bins: 4,
            max_rounds: 12,
            tol: 1e-10,
            mono: MonoMode::None,
            ..V1Config::default()
        }
    }

    #[test]
    fn attribution_matches_w2() {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut state = 7u64;
        for i in 0..80 {
            let x0 = ((i * 17) % 100) as f64 / 100.0;
            let x1 = ((i * 29) % 100) as f64 / 100.0;
            x.push(x0);
            x.push(x1);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (state >> 11) as f64 / (u64::MAX as f64);
            let z = super::super::basis::inv_norm(u.clamp(1e-6, 1.0 - 1e-6)).unwrap();
            y.push(0.4 * x0 + (0.8 + 0.5 * x1) * z);
        }
        let model = fit_v1(&x, 80, 2, &y, &cfg_small()).unwrap();
        let attr = attribute(&model, &x, 80).unwrap();
        for i in 0..80 {
            let mut sum = 0.0;
            for t in 0..(2 + 1) * 4 {
                sum += attr.phi_fit[i * 12 + t];
            }
            let players = player_coefficients(&model, &x, 80).unwrap();
            let a = curve_coefficients(&players, 80, 2, 4);
            let w2: f64 = a[i * 4..(i + 1) * 4].iter().map(|v| v * v).sum();
            assert!((sum - w2).abs() <= 1e-10 * w2.max(1.0));
        }
    }

    #[test]
    fn location_feature_is_recovered() {
        let n = 200;
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..n {
            let x0 = i as f64 / (n - 1) as f64;
            x.push(x0);
            y.push(1.5 * x0 + 0.25);
        }
        let cfg = V1Config {
            k_fit: 2,
            k_report: 2,
            n_u: 4,
            max_bins: 5,
            max_rounds: 20,
            tol: 1e-12,
            mono: MonoMode::None,
            reporting: ReportKind::Legendre,
            ..V1Config::default()
        };
        let model = fit_v1(&x, n, 1, &y, &cfg).unwrap();
        let pred = predict_quantiles(&model, &x, n, &[0.5], false).unwrap();
        let mut err = 0.0;
        for i in 0..n {
            err += (pred[i] - y[i]).abs();
        }
        let mae = err / (n as f64);
        let slope = channel_slope(&model, 0, 0, &x).unwrap();
        assert!(
            mae < 0.09 && (slope - 1.5).abs() < 0.15,
            "mae {mae} slope {slope} loss {} rounds {}",
            model.train_loss,
            model.rounds
        );
    }
}
