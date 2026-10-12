//! Thin PyO3 surface for the shift decomposition. Numerics stay in the core.

#![allow(clippy::too_many_arguments)]

use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use traceot_core::trace::{
    barycenter_view, bootstrap_barycenter, bootstrap_entry_exit, bootstrap_marginal,
    entry_exit_from_models, entry_exit_split, explain_shift, hk_mass, marginal_from_laws,
    marginal_view, percentile_bands, permutation_marginal, permutation_w2_floor, quadratic_shapley,
    sample_lambda, BaryView, EntryExit, MarginalView, PercentileBand, ShiftReport, ShiftV2Config,
    V1Config,
};
use traceot_core::TraceError;

use super::v1_api::{mono_from, report_from, PyV1Model};

fn err(e: TraceError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn slice1<'a>(a: &'a PyReadonlyArray1<'a, f64>) -> PyResult<&'a [f64]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err("array must be C-contiguous"))
}

fn slice2<'a>(a: &'a PyReadonlyArray2<'a, f64>) -> PyResult<&'a [f64]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err("array must be C-contiguous"))
}

fn slice_u8<'a>(a: &'a PyReadonlyArray1<'a, u8>) -> PyResult<&'a [u8]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err("array must be C-contiguous"))
}

fn arr1<'py>(py: Python<'py>, v: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
    Array1::from_vec(v).into_pyarray(py)
}

fn arr2<'py>(py: Python<'py>, v: Vec<f64>, rows: usize, cols: usize) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_vec((rows, cols), v)
        .expect("shape")
        .into_pyarray(py)
}

fn shift_cfg(
    k: usize,
    n_quad: usize,
    n_folds: usize,
    trim_quantile: f64,
    logit_ridge: f64,
    logit_iters: usize,
    logit_tol: f64,
    seed: u64,
) -> ShiftV2Config {
    ShiftV2Config {
        k,
        n_quad,
        n_folds,
        trim_quantile,
        logit_ridge,
        logit_iters,
        logit_tol,
        seed,
    }
}

fn v1_cfg(
    k_fit: usize,
    k_report: usize,
    n_u: usize,
    max_bins: usize,
    max_rounds: usize,
    tol: f64,
    min_count: u64,
    ridge: f64,
    learning_rate: f64,
    mono: &str,
    bernstein_elevation: usize,
    reporting: &str,
) -> PyResult<V1Config> {
    Ok(V1Config {
        k_fit,
        k_report,
        n_u,
        max_bins,
        max_rounds,
        tol,
        min_count,
        ridge,
        learning_rate,
        mono: mono_from(mono)?,
        bernstein_elevation,
        reporting: report_from(reporting)?,
        reference: Vec::new(),
        lock_mean: false,
        smooth: 0.0,
        levenberg: 0.0,
        sobolev_lambda: 0.0,
        sobolev_s: 0.0,
        tail_gamma: 0.0,
    })
}

fn put_marginal(py: Python<'_>, d: &Bound<'_, PyDict>, v: &MarginalView) -> PyResult<()> {
    let k = v.k;
    d.set_item("lambda_a", arr1(py, v.lambda_a.clone()))?;
    d.set_item("lambda_b", arr1(py, v.lambda_b.clone()))?;
    d.set_item("lambda_c", arr1(py, v.lambda_c.clone()))?;
    d.set_item("lambda_cp", arr1(py, v.lambda_cp.clone()))?;
    d.set_item("comp_lambda", arr1(py, v.comp_lambda.clone()))?;
    d.set_item("struct_lambda", arr1(py, v.struct_lambda.clone()))?;
    d.set_item("interaction_lambda", arr1(py, v.interaction_lambda.clone()))?;
    d.set_item("comp_w2", arr1(py, v.comp_w2.clone()))?;
    d.set_item("struct_w2", arr1(py, v.struct_w2.clone()))?;
    d.set_item("comp_w2_total", v.comp_w2_total)?;
    d.set_item("struct_w2_total", v.struct_w2_total)?;
    d.set_item("w2_resolved", v.w2_resolved)?;
    d.set_item("w2_total", v.w2_total)?;
    d.set_item("unresolved", v.unresolved)?;
    d.set_item("trim_mass_c", v.trim_mass_c)?;
    d.set_item("trim_mass_cp", v.trim_mass_cp)?;
    d.set_item("max_weight_c", v.max_weight_c)?;
    d.set_item("max_weight_cp", v.max_weight_cp)?;
    d.set_item("ess_c", v.ess_c)?;
    d.set_item("ess_cp", v.ess_cp)?;
    d.set_item("order_gap", v.order_gap)?;
    d.set_item("k", k)?;
    Ok(())
}

fn put_bary(py: Python<'_>, d: &Bound<'_, PyDict>, v: &BaryView) -> PyResult<()> {
    d.set_item("alpha_a", arr1(py, v.alpha_a.clone()))?;
    d.set_item("alpha_b", arr1(py, v.alpha_b.clone()))?;
    d.set_item("lambda_a", arr1(py, v.lambda_a.clone()))?;
    d.set_item("lambda_b", arr1(py, v.lambda_b.clone()))?;
    d.set_item("comp_alpha", arr2(py, v.comp_alpha.clone(), v.p, v.k))?;
    d.set_item("struct_alpha", arr2(py, v.struct_alpha.clone(), v.p, v.k))?;
    d.set_item("intercept_alpha", arr1(py, v.intercept_alpha.clone()))?;
    let players = 2 * v.p + 1;
    d.set_item("phi", arr2(py, v.phi.clone(), players, v.k))?;
    d.set_item("comp_w2", arr1(py, v.comp_w2.clone()))?;
    d.set_item("struct_w2", arr1(py, v.struct_w2.clone()))?;
    d.set_item("w2_channel", arr1(py, v.w2_channel.clone()))?;
    d.set_item("w2_total", v.w2_total)?;
    d.set_item("order_gap", v.order_gap)?;
    d.set_item("player_sum_gap", v.player_sum_gap)?;
    d.set_item("p", v.p)?;
    d.set_item("k", v.k)?;
    Ok(())
}

fn put_entry(py: Python<'_>, d: &Bound<'_, PyDict>, v: &EntryExit) -> PyResult<()> {
    d.set_item("stay", arr1(py, v.stay.clone()))?;
    d.set_item("mix", arr1(py, v.mix.clone()))?;
    d.set_item("entry", arr1(py, v.entry.clone()))?;
    d.set_item("exit", arr1(py, v.exit.clone()))?;
    d.set_item("phi_stay", arr1(py, v.phi_stay.clone()))?;
    d.set_item("phi_mix", arr1(py, v.phi_mix.clone()))?;
    d.set_item("phi_entry", arr1(py, v.phi_entry.clone()))?;
    d.set_item("phi_exit", arr1(py, v.phi_exit.clone()))?;
    d.set_item("delta", arr1(py, v.delta.clone()))?;
    d.set_item("hk2_mass", v.hk2_mass)?;
    d.set_item("hk2_normalized", v.hk2_normalized)?;
    d.set_item("k", v.k)?;
    Ok(())
}

fn put_band(py: Python<'_>, d: &Bound<'_, PyDict>, b: &PercentileBand) -> PyResult<()> {
    d.set_item("lo", arr1(py, b.lo.clone()))?;
    d.set_item("hi", arr1(py, b.hi.clone()))?;
    d.set_item("uniform_lo", arr1(py, b.uniform_lo.clone()))?;
    d.set_item("uniform_hi", arr1(py, b.uniform_hi.clone()))?;
    d.set_item("sigma", arr1(py, b.sigma.clone()))?;
    let flags: Vec<bool> = b.resolved.clone();
    d.set_item("resolved", flags)?;
    d.set_item("critical", b.critical)?;
    Ok(())
}

fn put_report<'py>(py: Python<'py>, r: &ShiftReport) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let marginal = PyDict::new(py);
    put_marginal(py, &marginal, &r.marginal)?;
    let bary = PyDict::new(py);
    put_bary(py, &bary, &r.bary)?;
    d.set_item("marginal", marginal)?;
    d.set_item("bary", bary)?;
    d.set_item("gap_lambda_a", arr1(py, r.gap_lambda_a.clone()))?;
    d.set_item("gap_lambda_b", arr1(py, r.gap_lambda_b.clone()))?;
    d.set_item("gap_delta_lambda", arr1(py, r.gap_delta_lambda.clone()))?;
    d.set_item("gap_comp_w2", arr1(py, r.gap_comp_w2.clone()))?;
    d.set_item("gap_struct_w2", arr1(py, r.gap_struct_w2.clone()))?;
    d.set_item("gap_w2", r.gap_w2)?;
    d.set_item("order_gap", r.order_gap)?;
    d.set_item("player_sum_gap", r.player_sum_gap)?;
    Ok(d)
}

/// L-moments of one sample on the shift quadrature.
#[pyfunction]
fn shift_v2_sample_lambda<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    k: usize,
    n_quad: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let y = slice1(&y)?;
    let lam = py.detach(|| sample_lambda(y, k, n_quad)).map_err(err)?;
    Ok(arr1(py, lam))
}

/// Four observed laws, no classifier. `y_c` is structure B on covariates A.
#[pyfunction]
fn shift_v2_from_laws<'py>(
    py: Python<'py>,
    y_a: PyReadonlyArray1<'py, f64>,
    y_b: PyReadonlyArray1<'py, f64>,
    y_c: PyReadonlyArray1<'py, f64>,
    y_cp: PyReadonlyArray1<'py, f64>,
    k: usize,
    n_quad: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let (ya, yb, yc, ycp) = (slice1(&y_a)?, slice1(&y_b)?, slice1(&y_c)?, slice1(&y_cp)?);
    let view = py
        .detach(|| marginal_from_laws(ya, yb, yc, ycp, k, n_quad))
        .map_err(err)?;
    let d = PyDict::new(py);
    put_marginal(py, &d, &view)?;
    Ok(d)
}

/// `φ_i = a_i · sum(a) / (2k+1)`, with negative shares left as they are.
#[pyfunction]
fn shift_v2_quadratic_phi<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    k: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = slice1(&a)?;
    Ok(arr1(py, quadratic_shapley(a, k)))
}

/// Pure-mass Hellinger–Kantorovich term and the same term divided by `m_a + m_b`.
#[pyfunction]
fn shift_v2_hk_mass(m_a: f64, m_b: f64) -> (f64, f64) {
    hk_mass(m_a, m_b)
}

/// Four-player entry/exit split. Inputs are one coefficient per channel.
#[pyfunction]
fn shift_v2_entry_exit<'py>(
    py: Python<'py>,
    stay_a: PyReadonlyArray1<'py, f64>,
    stay_b: PyReadonlyArray1<'py, f64>,
    exit_mean: PyReadonlyArray1<'py, f64>,
    entry_mean: PyReadonlyArray1<'py, f64>,
    e_a: f64,
    e_b: f64,
    m_a: f64,
    m_b: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let got = entry_exit_split(
        slice1(&stay_a)?,
        slice1(&stay_b)?,
        slice1(&exit_mean)?,
        slice1(&entry_mean)?,
        e_a,
        e_b,
        m_a,
        m_b,
    )
    .map_err(err)?;
    let d = PyDict::new(py);
    put_entry(py, &d, &got)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (
    x_a, y_a, x_b, y_b, k=6, n_quad=64, n_folds=5, trim_quantile=0.995,
    logit_ridge=1e-6, logit_iters=25, logit_tol=1e-8, seed=0
))]
fn shift_v2_marginal<'py>(
    py: Python<'py>,
    x_a: PyReadonlyArray2<'py, f64>,
    y_a: PyReadonlyArray1<'py, f64>,
    x_b: PyReadonlyArray2<'py, f64>,
    y_b: PyReadonlyArray1<'py, f64>,
    k: usize,
    n_quad: usize,
    n_folds: usize,
    trim_quantile: f64,
    logit_ridge: f64,
    logit_iters: usize,
    logit_tol: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let (na, _p) = x_a.as_array().dim();
    let (nb, _) = x_b.as_array().dim();
    let cfg = shift_cfg(
        k,
        n_quad,
        n_folds,
        trim_quantile,
        logit_ridge,
        logit_iters,
        logit_tol,
        seed,
    );
    let (xa, ya, xb, yb) = (slice2(&x_a)?, slice1(&y_a)?, slice2(&x_b)?, slice1(&y_b)?);
    let view = py
        .detach(|| marginal_view(xa, ya, na, xb, yb, nb, &cfg))
        .map_err(err)?;
    let d = PyDict::new(py);
    put_marginal(py, &d, &view)?;
    Ok(d)
}

#[pyfunction]
fn shift_v2_barycenter<'py>(
    py: Python<'py>,
    model_a: &PyV1Model,
    model_b: &PyV1Model,
    x_a: PyReadonlyArray2<'py, f64>,
    x_b: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let (na, _) = x_a.as_array().dim();
    let (nb, _) = x_b.as_array().dim();
    let (xa, xb) = (slice2(&x_a)?, slice2(&x_b)?);
    let view = py
        .detach(|| barycenter_view(&model_a.inner, &model_b.inner, xa, na, xb, nb))
        .map_err(err)?;
    let d = PyDict::new(py);
    put_bary(py, &d, &view)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (
    model_a, model_b, x_a, y_a, x_b, y_b, n_quad=64, n_folds=5, trim_quantile=0.995,
    logit_ridge=1e-6, logit_iters=25, logit_tol=1e-8, seed=0
))]
fn shift_v2_explain<'py>(
    py: Python<'py>,
    model_a: &PyV1Model,
    model_b: &PyV1Model,
    x_a: PyReadonlyArray2<'py, f64>,
    y_a: PyReadonlyArray1<'py, f64>,
    x_b: PyReadonlyArray2<'py, f64>,
    y_b: PyReadonlyArray1<'py, f64>,
    n_quad: usize,
    n_folds: usize,
    trim_quantile: f64,
    logit_ridge: f64,
    logit_iters: usize,
    logit_tol: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let (na, _) = x_a.as_array().dim();
    let (nb, _) = x_b.as_array().dim();
    let k = model_a.inner.k_fit;
    let cfg = shift_cfg(
        k,
        n_quad,
        n_folds,
        trim_quantile,
        logit_ridge,
        logit_iters,
        logit_tol,
        seed,
    );
    let (xa, ya, xb, yb) = (slice2(&x_a)?, slice1(&y_a)?, slice2(&x_b)?, slice1(&y_b)?);
    let ma = &model_a.inner;
    let mb = &model_b.inner;
    let report = py
        .detach(|| explain_shift(ma, mb, xa, ya, na, xb, yb, nb, &cfg))
        .map_err(err)?;
    put_report(py, &report)
}

#[pyfunction]
#[pyo3(signature = (
    x_a, y_a, x_b, y_b, reps, seed, k=6, n_quad=64, n_folds=5, trim_quantile=0.995,
    logit_ridge=1e-6, logit_iters=25, logit_tol=1e-8
))]
fn shift_v2_bootstrap_marginal<'py>(
    py: Python<'py>,
    x_a: PyReadonlyArray2<'py, f64>,
    y_a: PyReadonlyArray1<'py, f64>,
    x_b: PyReadonlyArray2<'py, f64>,
    y_b: PyReadonlyArray1<'py, f64>,
    reps: usize,
    seed: u64,
    k: usize,
    n_quad: usize,
    n_folds: usize,
    trim_quantile: f64,
    logit_ridge: f64,
    logit_iters: usize,
    logit_tol: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let (na, _) = x_a.as_array().dim();
    let (nb, _) = x_b.as_array().dim();
    let cfg = shift_cfg(
        k,
        n_quad,
        n_folds,
        trim_quantile,
        logit_ridge,
        logit_iters,
        logit_tol,
        seed,
    );
    let (xa, ya, xb, yb) = (slice2(&x_a)?, slice1(&y_a)?, slice2(&x_b)?, slice1(&y_b)?);
    let draws = py
        .detach(|| bootstrap_marginal(xa, ya, na, xb, yb, nb, &cfg, reps, seed))
        .map_err(err)?;
    let reps = draws.len();
    let kk = draws.first().map(|d| d.comp_w2.len()).unwrap_or(k);
    let mut comp = Vec::with_capacity(reps * kk);
    let mut structure = Vec::with_capacity(reps * kk);
    let mut comp_total = Vec::with_capacity(reps);
    let mut struct_total = Vec::with_capacity(reps);
    let mut w2_total = Vec::with_capacity(reps);
    for d in &draws {
        comp.extend_from_slice(&d.comp_w2);
        structure.extend_from_slice(&d.struct_w2);
        comp_total.push(d.comp_total);
        struct_total.push(d.struct_total);
        w2_total.push(d.w2_total);
    }
    let d = PyDict::new(py);
    d.set_item("comp_w2", arr2(py, comp, reps, kk))?;
    d.set_item("struct_w2", arr2(py, structure, reps, kk))?;
    d.set_item("comp_total", arr1(py, comp_total))?;
    d.set_item("struct_total", arr1(py, struct_total))?;
    d.set_item("w2_total", arr1(py, w2_total))?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (
    x_a, y_a, x_b, y_b, reps, seed, k_fit=6, k_report=6, n_u=0, max_bins=16,
    max_rounds=12, tol=1e-6, min_count=1, ridge=1e-6, learning_rate=1.0, mono="none",
    bernstein_elevation=4, reporting="legendre"
))]
fn shift_v2_bootstrap_barycenter<'py>(
    py: Python<'py>,
    x_a: PyReadonlyArray2<'py, f64>,
    y_a: PyReadonlyArray1<'py, f64>,
    x_b: PyReadonlyArray2<'py, f64>,
    y_b: PyReadonlyArray1<'py, f64>,
    reps: usize,
    seed: u64,
    k_fit: usize,
    k_report: usize,
    n_u: usize,
    max_bins: usize,
    max_rounds: usize,
    tol: f64,
    min_count: u64,
    ridge: f64,
    learning_rate: f64,
    mono: &str,
    bernstein_elevation: usize,
    reporting: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let (na, p) = x_a.as_array().dim();
    let (nb, _) = x_b.as_array().dim();
    let v1 = v1_cfg(
        k_fit,
        k_report,
        n_u,
        max_bins,
        max_rounds,
        tol,
        min_count,
        ridge,
        learning_rate,
        mono,
        bernstein_elevation,
        reporting,
    )?;
    let (xa, ya, xb, yb) = (slice2(&x_a)?, slice1(&y_a)?, slice2(&x_b)?, slice1(&y_b)?);
    let draws = py
        .detach(|| bootstrap_barycenter(xa, ya, na, xb, yb, nb, &v1, reps, seed))
        .map_err(err)?;
    let reps = draws.len();
    let players = 2 * p + 1;
    let mut phi = Vec::new();
    let mut comp_feature = Vec::new();
    if let Some(first) = draws.first() {
        let flat = first.phi.len();
        phi.reserve(reps * flat);
        comp_feature.reserve(reps * p);
        for d in &draws {
            phi.extend_from_slice(&d.phi);
            comp_feature.extend_from_slice(&d.comp_feature);
        }
        let d = PyDict::new(py);
        d.set_item("phi", arr2(py, phi, reps, flat))?;
        d.set_item("comp_feature", arr2(py, comp_feature, reps, p))?;
        d.set_item("players", players)?;
        d.set_item("k", k_fit)?;
        return Ok(d);
    }
    let d = PyDict::new(py);
    d.set_item("phi", arr2(py, Vec::new(), 0, players * k_fit))?;
    d.set_item("comp_feature", arr2(py, Vec::new(), 0, p))?;
    d.set_item("players", players)?;
    d.set_item("k", k_fit)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (
    model_a, model_b, x_a, exit_a, x_b, entry_b, m_a, m_b
))]
fn shift_v2_entry_from_models<'py>(
    py: Python<'py>,
    model_a: &PyV1Model,
    model_b: &PyV1Model,
    x_a: PyReadonlyArray2<'py, f64>,
    exit_a: PyReadonlyArray1<'py, u8>,
    x_b: PyReadonlyArray2<'py, f64>,
    entry_b: PyReadonlyArray1<'py, u8>,
    m_a: f64,
    m_b: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let (na, _) = x_a.as_array().dim();
    let (nb, _) = x_b.as_array().dim();
    let got = entry_exit_from_models(
        &model_a.inner,
        slice2(&x_a)?,
        slice_u8(&exit_a)?,
        na,
        &model_b.inner,
        slice2(&x_b)?,
        slice_u8(&entry_b)?,
        nb,
        m_a,
        m_b,
    )
    .map_err(err)?;
    let d = PyDict::new(py);
    put_entry(py, &d, &got)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (
    x_a, y_a, exit_a, x_b, y_b, entry_b, reps, seed,
    k_fit=6, k_report=6, n_u=0, max_bins=16, max_rounds=12, tol=1e-6, min_count=1,
    ridge=1e-6, learning_rate=1.0, mono="none", bernstein_elevation=4, reporting="legendre"
))]
fn shift_v2_bootstrap_entry<'py>(
    py: Python<'py>,
    x_a: PyReadonlyArray2<'py, f64>,
    y_a: PyReadonlyArray1<'py, f64>,
    exit_a: PyReadonlyArray1<'py, u8>,
    x_b: PyReadonlyArray2<'py, f64>,
    y_b: PyReadonlyArray1<'py, f64>,
    entry_b: PyReadonlyArray1<'py, u8>,
    reps: usize,
    seed: u64,
    k_fit: usize,
    k_report: usize,
    n_u: usize,
    max_bins: usize,
    max_rounds: usize,
    tol: f64,
    min_count: u64,
    ridge: f64,
    learning_rate: f64,
    mono: &str,
    bernstein_elevation: usize,
    reporting: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let (na, _) = x_a.as_array().dim();
    let (nb, _) = x_b.as_array().dim();
    let (xa, ya, ea, xb, yb, eb) = (
        slice2(&x_a)?,
        slice1(&y_a)?,
        slice_u8(&exit_a)?,
        slice2(&x_b)?,
        slice1(&y_b)?,
        slice_u8(&entry_b)?,
    );
    let v1 = v1_cfg(
        k_fit,
        k_report,
        n_u,
        max_bins,
        max_rounds,
        tol,
        min_count,
        ridge,
        learning_rate,
        mono,
        bernstein_elevation,
        reporting,
    )?;
    let draws = py
        .detach(|| bootstrap_entry_exit(xa, ya, ea, na, xb, yb, eb, nb, &v1, reps, seed))
        .map_err(err)?;
    let reps = draws.len();
    let k = draws.first().map(|d| d.k).unwrap_or(k_fit);
    let pack = |pick: fn(&EntryExit) -> &Vec<f64>| {
        let mut out = Vec::with_capacity(reps * k);
        for d in &draws {
            out.extend_from_slice(pick(d));
        }
        out
    };
    let d = PyDict::new(py);
    d.set_item("stay", arr2(py, pack(|e| &e.stay), reps, k))?;
    d.set_item("mix", arr2(py, pack(|e| &e.mix), reps, k))?;
    d.set_item("entry", arr2(py, pack(|e| &e.entry), reps, k))?;
    d.set_item("exit", arr2(py, pack(|e| &e.exit), reps, k))?;
    Ok(d)
}

/// Pointwise 2.5/97.5 bands and a uniform max-|t| band. `replicates` is `(reps, dim)`.
#[pyfunction]
fn shift_v2_bands<'py>(
    py: Python<'py>,
    point: PyReadonlyArray1<'py, f64>,
    replicates: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let (reps, _) = replicates.as_array().dim();
    let band = percentile_bands(slice1(&point)?, slice2(&replicates)?, reps).map_err(err)?;
    let d = PyDict::new(py);
    put_band(py, &d, &band)?;
    Ok(d)
}

/// Permutation 95% floor of the total outcome W2².
#[pyfunction]
#[pyo3(signature = (y_a, y_b, reps, seed, n_quad=64, k=4, level=0.95))]
fn shift_v2_permutation_w2<'py>(
    py: Python<'py>,
    y_a: PyReadonlyArray1<'py, f64>,
    y_b: PyReadonlyArray1<'py, f64>,
    reps: usize,
    seed: u64,
    n_quad: usize,
    k: usize,
    level: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let (ya, yb) = (slice1(&y_a)?, slice1(&y_b)?);
    let got = py
        .detach(|| permutation_w2_floor(ya, yb, reps, seed, n_quad, k, level))
        .map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("observed", got.observed)?;
    d.set_item("floor", got.floor)?;
    d.set_item("exceeds", got.exceeds)?;
    d.set_item("null_mean", got.null_mean)?;
    Ok(d)
}

/// Shuffled-label replicates of the marginal channel shares.
#[pyfunction]
#[pyo3(signature = (
    x_a, y_a, x_b, y_b, reps, seed, k=6, n_quad=64, n_folds=5, trim_quantile=0.995,
    logit_ridge=1e-6, logit_iters=25, logit_tol=1e-8
))]
fn shift_v2_permutation_marginal<'py>(
    py: Python<'py>,
    x_a: PyReadonlyArray2<'py, f64>,
    y_a: PyReadonlyArray1<'py, f64>,
    x_b: PyReadonlyArray2<'py, f64>,
    y_b: PyReadonlyArray1<'py, f64>,
    reps: usize,
    seed: u64,
    k: usize,
    n_quad: usize,
    n_folds: usize,
    trim_quantile: f64,
    logit_ridge: f64,
    logit_iters: usize,
    logit_tol: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let (na, _) = x_a.as_array().dim();
    let (nb, _) = x_b.as_array().dim();
    let cfg = shift_cfg(
        k,
        n_quad,
        n_folds,
        trim_quantile,
        logit_ridge,
        logit_iters,
        logit_tol,
        seed,
    );
    let (xa, ya, xb, yb) = (slice2(&x_a)?, slice1(&y_a)?, slice2(&x_b)?, slice1(&y_b)?);
    let draws = py
        .detach(|| permutation_marginal(xa, ya, na, xb, yb, nb, &cfg, reps, seed))
        .map_err(err)?;
    let reps = draws.len();
    let kk = draws.first().map(|d| d.comp_w2.len()).unwrap_or(k);
    let mut comp = Vec::with_capacity(reps * kk);
    let mut structure = Vec::with_capacity(reps * kk);
    let mut w2_total = Vec::with_capacity(reps);
    for d in &draws {
        comp.extend_from_slice(&d.comp_w2);
        structure.extend_from_slice(&d.struct_w2);
        w2_total.push(d.w2_total);
    }
    let d = PyDict::new(py);
    d.set_item("comp_w2", arr2(py, comp, reps, kk))?;
    d.set_item("struct_w2", arr2(py, structure, reps, kk))?;
    d.set_item("w2_total", arr1(py, w2_total))?;
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(shift_v2_sample_lambda, m)?)?;
    m.add_function(wrap_pyfunction!(shift_v2_from_laws, m)?)?;
    m.add_function(wrap_pyfunction!(shift_v2_quadratic_phi, m)?)?;
    m.add_function(wrap_pyfunction!(shift_v2_hk_mass, m)?)?;
    m.add_function(wrap_pyfunction!(shift_v2_entry_exit, m)?)?;
    m.add_function(wrap_pyfunction!(shift_v2_marginal, m)?)?;
    m.add_function(wrap_pyfunction!(shift_v2_barycenter, m)?)?;
    m.add_function(wrap_pyfunction!(shift_v2_explain, m)?)?;
    m.add_function(wrap_pyfunction!(shift_v2_bootstrap_marginal, m)?)?;
    m.add_function(wrap_pyfunction!(shift_v2_bootstrap_barycenter, m)?)?;
    m.add_function(wrap_pyfunction!(shift_v2_entry_from_models, m)?)?;
    m.add_function(wrap_pyfunction!(shift_v2_bootstrap_entry, m)?)?;
    m.add_function(wrap_pyfunction!(shift_v2_bands, m)?)?;
    m.add_function(wrap_pyfunction!(shift_v2_permutation_w2, m)?)?;
    m.add_function(wrap_pyfunction!(shift_v2_permutation_marginal, m)?)?;
    Ok(())
}
