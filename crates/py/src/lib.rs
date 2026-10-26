//! PyO3 bindings for `traceot-core`, exposed to Python as `traceot._core`.
//!
//! All numerical logic lives in `traceot-core`; this crate only converts arrays and
//! releases the GIL. The numpy reference used to verify results lives in
//! `tests/python/reference/` and mirrors the low-level function names below.

use numpy::ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
mod shift_v2_api;
mod trace_api;
mod v1_api;
mod v4_api;

use traceot_core::adopt::{BackfitConfig, BackfitResult};
use traceot_core::scalable::{binning, histogram as hist, pinball, sketch};
use traceot_core::{additive, adopt, estimators, logchol, smoother, w2, Kernel, Mat, TraceError};

type ArrayPair<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

fn err(e: TraceError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn to_mat(a: ArrayView2<'_, f64>) -> Mat {
    let (r, c) = a.dim();
    Mat {
        rows: r,
        cols: c,
        data: a.iter().copied().collect(),
    }
}

fn to_mats(a: ArrayView3<'_, f64>) -> Vec<Mat> {
    a.outer_iter().map(to_mat).collect()
}

fn mat_to_py<'py>(py: Python<'py>, m: Mat) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_vec((m.rows, m.cols), m.data)
        .expect("shape")
        .into_pyarray(py)
}

fn stack_to_py<'py>(py: Python<'py>, ms: Vec<Mat>) -> Bound<'py, PyAny> {
    let p = ms.len();
    let (n, q) = if p > 0 {
        (ms[0].rows, ms[0].cols)
    } else {
        (0, 0)
    };
    let mut data = Vec::with_capacity(p * n * q);
    for m in ms {
        data.extend(m.data);
    }
    Array3::from_shape_vec((p, n, q), data)
        .expect("shape")
        .into_pyarray(py)
        .into_any()
}

fn rows_to_py<'py>(py: Python<'py>, rows: Vec<Vec<f64>>) -> Bound<'py, PyArray2<f64>> {
    let r = rows.len();
    let c = if r > 0 { rows[0].len() } else { 0 };
    let data: Vec<f64> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((r, c), data)
        .expect("shape")
        .into_pyarray(py)
}

#[allow(clippy::too_many_arguments)]
fn config(
    kernel: &str,
    tol: f64,
    max_iter: usize,
    lower: Option<f64>,
    upper: Option<f64>,
    damping: f64,
    certificate: bool,
    composition: &str,
    centering: &str,
    project_prediction: bool,
    bins: Option<usize>,
    boundary: &str,
) -> PyResult<BackfitConfig> {
    Ok(BackfitConfig {
        kernel: Kernel::parse(kernel).map_err(err)?,
        tol,
        max_iter,
        lower,
        upper,
        damping,
        certificate,
        composition: traceot_core::adopt::Composition::parse(composition).map_err(err)?,
        centering: traceot_core::adopt::Centering::parse(centering).map_err(err)?,
        project_prediction,
        bins,
        boundary: traceot_core::smoother::Boundary::parse(boundary).map_err(err)?,
    })
}

fn config_basic(
    kernel: &str,
    tol: f64,
    max_iter: usize,
    lower: Option<f64>,
    upper: Option<f64>,
) -> PyResult<BackfitConfig> {
    config(
        kernel, tol, max_iter, lower, upper, 1.0, false, "given", "frechet", false, None, "none",
    )
}

fn orders_to_py<'py>(py: Python<'py>, orders: &[adopt::OrderFit]) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for o in orders {
        let d = PyDict::new(py);
        let perm: Vec<i64> = o.perm.iter().map(|&i| i as i64).collect();
        d.set_item("perm", perm.into_pyarray(py))?;
        d.set_item("G", stack_to_py(py, o.g.clone()))?;
        d.set_item("M", rows_to_py(py, o.m.clone()))?;
        d.set_item("P", stack_to_py(py, o.resid.clone()))?;
        out.append(d)?;
    }
    Ok(out)
}

fn result_to_dict<'py>(py: Python<'py>, r: BackfitResult) -> PyResult<Bound<'py, PyDict>> {
    let orders = orders_to_py(py, &r.orders)?;
    let d = PyDict::new(py);
    d.set_item("mu", Array1::from_vec(r.mu).into_pyarray(py))?;
    d.set_item("G", stack_to_py(py, r.g))?;
    d.set_item("M", rows_to_py(py, r.m))?;
    d.set_item("P", stack_to_py(py, r.resid))?;
    d.set_item("fitted", mat_to_py(py, r.fitted))?;
    d.set_item("n_iter", r.n_iter)?;
    d.set_item("converged", r.converged)?;
    d.set_item("history", Array1::from_vec(r.history).into_pyarray(py))?;
    d.set_item("n_rejected", r.n_rejected)?;
    d.set_item("orders", orders)?;
    Ok(d)
}

fn require_item<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("order missing {key}")))
}

fn order_from_dict(d: &Bound<'_, PyDict>) -> PyResult<adopt::OrderFit> {
    let perm_obj = require_item(d, "perm")?;
    let perm: Vec<usize> = if let Ok(arr) = perm_obj.extract::<PyReadonlyArray1<i64>>() {
        arr.as_array().iter().map(|&v| v as usize).collect()
    } else {
        perm_obj
            .extract::<Vec<i64>>()?
            .into_iter()
            .map(|v| v as usize)
            .collect()
    };
    let g = require_item(d, "G")?.extract::<PyReadonlyArray3<f64>>()?;
    let m = require_item(d, "M")?.extract::<PyReadonlyArray2<f64>>()?;
    let p = require_item(d, "P")?.extract::<PyReadonlyArray3<f64>>()?;
    let mm = to_mat(m.as_array());
    Ok(adopt::OrderFit {
        perm,
        g: to_mats(g.as_array()),
        m: (0..mm.rows).map(|i| mm.row(i).to_vec()).collect(),
        resid: to_mats(p.as_array()),
    })
}

/// Midpoint quantile grid of size q.
#[pyfunction]
fn quantile_grid<'py>(py: Python<'py>, q: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if q < 1 {
        return Err(PyValueError::new_err("q must be >= 1"));
    }
    Ok(w2::quantile_grid(q).into_pyarray(py))
}

/// Squared W2 distances between matching rows of two (n, q) arrays.
#[pyfunction]
fn w2_sq_rows<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (a, b) = (to_mat(a.as_array()), to_mat(b.as_array()));
    if a.rows != b.rows || a.cols != b.cols {
        return Err(PyValueError::new_err("shape mismatch"));
    }
    let out: Vec<f64> = (0..a.rows).map(|i| w2::w2_sq(a.row(i), b.row(i))).collect();
    Ok(out.into_pyarray(py))
}

/// Isotonic (PAVA) projection with optional clipping.
#[pyfunction]
#[pyo3(signature = (y, lower=None, upper=None))]
fn pava<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    lower: Option<f64>,
    upper: Option<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let v: Vec<f64> = y.as_array().iter().copied().collect();
    w2::pava(&v, lower, upper).into_pyarray(py)
}

/// Row-wise PAVA.
#[pyfunction]
#[pyo3(signature = (y, lower=None, upper=None))]
fn pava_rows<'py>(
    py: Python<'py>,
    y: PyReadonlyArray2<'py, f64>,
    lower: Option<f64>,
    upper: Option<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let m = to_mat(y.as_array());
    let rows: Vec<Vec<f64>> = (0..m.rows)
        .map(|i| w2::pava(m.row(i), lower, upper))
        .collect();
    rows_to_py(py, rows)
}

/// Merge tied source knots, averaging targets.
#[pyfunction]
fn collapse_knots<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
) -> PyResult<ArrayPair<'py>> {
    let a: Vec<f64> = a.as_array().iter().copied().collect();
    let b: Vec<f64> = b.as_array().iter().copied().collect();
    if a.len() != b.len() || a.is_empty() {
        return Err(PyValueError::new_err(
            "a and b must be non-empty and of equal length",
        ));
    }
    let (au, bu) = w2::collapse_knots(&a, &b);
    Ok((au.into_pyarray(py), bu.into_pyarray(py)))
}

/// Evaluate the monotone transport map with knots a -> b at y.
#[pyfunction]
fn transport_apply<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a: Vec<f64> = a.as_array().iter().copied().collect();
    let b: Vec<f64> = b.as_array().iter().copied().collect();
    let y: Vec<f64> = y.as_array().iter().copied().collect();
    if a.len() != b.len() || a.is_empty() {
        return Err(PyValueError::new_err(
            "a and b must be non-empty and of equal length",
        ));
    }
    Ok(w2::transport_apply(&a, &b, &y).into_pyarray(py))
}

/// Local linear smoother weights (rows sum to one).
#[pyfunction]
#[pyo3(signature = (x_train, x_eval, h, kernel="gaussian"))]
fn local_linear_weights<'py>(
    py: Python<'py>,
    x_train: PyReadonlyArray1<'py, f64>,
    x_eval: PyReadonlyArray1<'py, f64>,
    h: f64,
    kernel: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let k = Kernel::parse(kernel).map_err(err)?;
    let xt: Vec<f64> = x_train.as_array().iter().copied().collect();
    let xe: Vec<f64> = x_eval.as_array().iter().copied().collect();
    let w = smoother::local_linear_weights(&xt, &xe, h, k).map_err(err)?;
    Ok(mat_to_py(py, w))
}

/// Local linear weights with reflections through both ends of the sample.
#[pyfunction]
#[pyo3(signature = (x_train, x_eval, h, kernel="gaussian"))]
fn reflected_local_linear_weights<'py>(
    py: Python<'py>,
    x_train: PyReadonlyArray1<'py, f64>,
    x_eval: PyReadonlyArray1<'py, f64>,
    h: f64,
    kernel: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let k = Kernel::parse(kernel).map_err(err)?;
    let xt: Vec<f64> = x_train.as_array().iter().copied().collect();
    let xe: Vec<f64> = x_eval.as_array().iter().copied().collect();
    let w = smoother::reflected_local_linear_weights(&xt, &xe, h, k).map_err(err)?;
    Ok(mat_to_py(py, w))
}

/// Multiplicity-weighted local linear weights. Unit weights match the unweighted call.
#[pyfunction]
#[pyo3(signature = (x_train, multiplicity, x_eval, h, kernel="gaussian"))]
fn local_linear_weights_weighted<'py>(
    py: Python<'py>,
    x_train: PyReadonlyArray1<'py, f64>,
    multiplicity: PyReadonlyArray1<'py, f64>,
    x_eval: PyReadonlyArray1<'py, f64>,
    h: f64,
    kernel: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let k = Kernel::parse(kernel).map_err(err)?;
    let xt: Vec<f64> = x_train.as_array().iter().copied().collect();
    let mult: Vec<f64> = multiplicity.as_array().iter().copied().collect();
    let xe: Vec<f64> = x_eval.as_array().iter().copied().collect();
    let w = smoother::local_linear_weights_weighted(&xt, &mult, &xe, h, k).map_err(err)?;
    Ok(mat_to_py(py, w))
}

/// Equal-width histogram smoother: bin means, then local-linear on the bin centres.
#[pyfunction]
#[pyo3(signature = (x_train, y, h, x_eval, n_bins, kernel="gaussian"))]
fn binned_smooth<'py>(
    py: Python<'py>,
    x_train: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    h: f64,
    x_eval: PyReadonlyArray1<'py, f64>,
    n_bins: usize,
    kernel: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let k = Kernel::parse(kernel).map_err(err)?;
    let xt: Vec<f64> = x_train.as_array().iter().copied().collect();
    let xe: Vec<f64> = x_eval.as_array().iter().copied().collect();
    let y = to_mat(y.as_array());
    let out = py
        .detach(|| smoother::binned_smooth(&xt, &y, h, &xe, n_bins, k))
        .map_err(err)?;
    Ok(mat_to_py(py, out))
}

/// Global Fréchet regression weights (rows sum to one).
#[pyfunction]
fn global_frechet_weights<'py>(
    py: Python<'py>,
    x_train: PyReadonlyArray2<'py, f64>,
    x_eval: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let w =
        smoother::global_frechet_weights(&to_mat(x_train.as_array()), &to_mat(x_eval.as_array()))
            .map_err(err)?;
    Ok(mat_to_py(py, w))
}

/// ADOPT transport backfitting in W2 (Algorithm 1).
#[pyfunction]
#[pyo3(signature = (x, y, h, kernel="gaussian", tol=1e-10, max_iter=200, lower=None, upper=None, damping=1.0, certificate=false, composition="given", centering="frechet", project_prediction=false, bins=None, boundary="none"))]
#[allow(clippy::too_many_arguments)]
fn adopt_w2_fit<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    h: PyReadonlyArray1<'py, f64>,
    kernel: &str,
    tol: f64,
    max_iter: usize,
    lower: Option<f64>,
    upper: Option<f64>,
    damping: f64,
    certificate: bool,
    composition: &str,
    centering: &str,
    project_prediction: bool,
    bins: Option<usize>,
    boundary: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let cfg = config(
        kernel,
        tol,
        max_iter,
        lower,
        upper,
        damping,
        certificate,
        composition,
        centering,
        project_prediction,
        bins,
        boundary,
    )?;
    let (x, y) = (to_mat(x.as_array()), to_mat(y.as_array()));
    let h: Vec<f64> = h.as_array().iter().copied().collect();
    let r = py
        .detach(|| adopt::adopt_w2_fit(&x, &y, &h, &cfg))
        .map_err(err)?;
    result_to_dict(py, r)
}

/// ADOPT prediction at new predictor values.
#[pyfunction]
#[pyo3(signature = (x_train, p, m, mu, h, x_new, kernel="gaussian", lower=None, upper=None, project_prediction=false, bins=None, boundary="none"))]
#[allow(clippy::too_many_arguments)]
fn adopt_w2_predict<'py>(
    py: Python<'py>,
    x_train: PyReadonlyArray2<'py, f64>,
    p: PyReadonlyArray3<'py, f64>,
    m: PyReadonlyArray2<'py, f64>,
    mu: PyReadonlyArray1<'py, f64>,
    h: PyReadonlyArray1<'py, f64>,
    x_new: PyReadonlyArray2<'py, f64>,
    kernel: &str,
    lower: Option<f64>,
    upper: Option<f64>,
    project_prediction: bool,
    bins: Option<usize>,
    boundary: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let cfg = config(
        kernel,
        0.0,
        1,
        lower,
        upper,
        1.0,
        false,
        "given",
        "frechet",
        project_prediction,
        bins,
        boundary,
    )?;
    let xt = to_mat(x_train.as_array());
    let resid = to_mats(p.as_array());
    let mm = to_mat(m.as_array());
    let mrows: Vec<Vec<f64>> = (0..mm.rows).map(|i| mm.row(i).to_vec()).collect();
    let mu: Vec<f64> = mu.as_array().iter().copied().collect();
    let h: Vec<f64> = h.as_array().iter().copied().collect();
    let xn = to_mat(x_new.as_array());
    let out = py
        .detach(|| adopt::adopt_w2_predict(&xt, &resid, &mrows, &mu, &h, &xn, &cfg))
        .map_err(err)?;
    Ok(mat_to_py(py, out))
}

/// Average the composed map over every stored feature order.
#[pyfunction]
#[pyo3(signature = (x_train, orders, mu, h, x_new, kernel="gaussian", lower=None, upper=None, project_prediction=false, bins=None, boundary="none"))]
#[allow(clippy::too_many_arguments)]
fn adopt_w2_predict_averaged<'py>(
    py: Python<'py>,
    x_train: PyReadonlyArray2<'py, f64>,
    orders: &Bound<'py, PyAny>,
    mu: PyReadonlyArray1<'py, f64>,
    h: PyReadonlyArray1<'py, f64>,
    x_new: PyReadonlyArray2<'py, f64>,
    kernel: &str,
    lower: Option<f64>,
    upper: Option<f64>,
    project_prediction: bool,
    bins: Option<usize>,
    boundary: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let cfg = config(
        kernel,
        0.0,
        1,
        lower,
        upper,
        1.0,
        false,
        "given",
        "frechet",
        project_prediction,
        bins,
        boundary,
    )?;
    let mut rust_orders = Vec::new();
    for item in orders.try_iter()? {
        let item = item?;
        let d = item.cast::<PyDict>()?;
        rust_orders.push(order_from_dict(d)?);
    }
    let xt = to_mat(x_train.as_array());
    let mu: Vec<f64> = mu.as_array().iter().copied().collect();
    let h: Vec<f64> = h.as_array().iter().copied().collect();
    let xn = to_mat(x_new.as_array());
    let out = py
        .detach(|| adopt::adopt_w2_predict_averaged(&xt, &rust_orders, &mu, &h, &xn, &cfg))
        .map_err(err)?;
    Ok(mat_to_py(py, out))
}

/// `g_j(x)` at new points from stored residuals.
#[pyfunction]
#[pyo3(signature = (x_train_j, p_j, h_j, x_eval, kernel="gaussian", lower=None, upper=None, bins=None, boundary="none"))]
#[allow(clippy::too_many_arguments)]
fn adopt_w2_component<'py>(
    py: Python<'py>,
    x_train_j: PyReadonlyArray1<'py, f64>,
    p_j: PyReadonlyArray2<'py, f64>,
    h_j: f64,
    x_eval: PyReadonlyArray1<'py, f64>,
    kernel: &str,
    lower: Option<f64>,
    upper: Option<f64>,
    bins: Option<usize>,
    boundary: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let cfg = config(
        kernel, 0.0, 1, lower, upper, 1.0, false, "given", "frechet", false, bins, boundary,
    )?;
    let xt: Vec<f64> = x_train_j.as_array().iter().copied().collect();
    let xe: Vec<f64> = x_eval.as_array().iter().copied().collect();
    let pj = to_mat(p_j.as_array());
    let out = adopt::adopt_w2_component(&xt, &pj, h_j, &xe, &cfg).map_err(err)?;
    Ok(mat_to_py(py, out))
}

/// Additive backfitting in flat coordinates.
#[pyfunction]
#[pyo3(signature = (x, z, h, kernel="gaussian", tol=1e-10, max_iter=200, damping=1.0, certificate=false, bins=None, boundary="none"))]
#[allow(clippy::too_many_arguments)]
fn additive_euclid_fit<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    z: PyReadonlyArray2<'py, f64>,
    h: PyReadonlyArray1<'py, f64>,
    kernel: &str,
    tol: f64,
    max_iter: usize,
    damping: f64,
    certificate: bool,
    bins: Option<usize>,
    boundary: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let cfg = config(
        kernel,
        tol,
        max_iter,
        None,
        None,
        damping,
        certificate,
        "given",
        "frechet",
        false,
        bins,
        boundary,
    )?;
    let (x, z) = (to_mat(x.as_array()), to_mat(z.as_array()));
    let h: Vec<f64> = h.as_array().iter().copied().collect();
    let r = py
        .detach(|| additive::additive_euclid_fit(&x, &z, &h, &cfg))
        .map_err(err)?;
    result_to_dict(py, r)
}

/// Additive prediction in flat coordinates.
#[pyfunction]
#[pyo3(signature = (x_train, p, m, mu, h, x_new, kernel="gaussian", bins=None, boundary="none"))]
#[allow(clippy::too_many_arguments)]
fn additive_euclid_predict<'py>(
    py: Python<'py>,
    x_train: PyReadonlyArray2<'py, f64>,
    p: PyReadonlyArray3<'py, f64>,
    m: PyReadonlyArray2<'py, f64>,
    mu: PyReadonlyArray1<'py, f64>,
    h: PyReadonlyArray1<'py, f64>,
    x_new: PyReadonlyArray2<'py, f64>,
    kernel: &str,
    bins: Option<usize>,
    boundary: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let cfg = config(
        kernel, 0.0, 1, None, None, 1.0, false, "given", "frechet", false, bins, boundary,
    )?;
    let xt = to_mat(x_train.as_array());
    let resid = to_mats(p.as_array());
    let mm = to_mat(m.as_array());
    let mrows: Vec<Vec<f64>> = (0..mm.rows).map(|i| mm.row(i).to_vec()).collect();
    let mu: Vec<f64> = mu.as_array().iter().copied().collect();
    let h: Vec<f64> = h.as_array().iter().copied().collect();
    let xn = to_mat(x_new.as_array());
    let out = py
        .detach(|| additive::additive_euclid_predict(&xt, &resid, &mrows, &mu, &h, &xn, &cfg))
        .map_err(err)?;
    Ok(mat_to_py(py, out))
}

/// Stack of SPD matrices (k, m, m) -> log-Cholesky coordinates (k, m(m+1)/2).
#[pyfunction]
fn spd_to_coords<'py>(
    py: Python<'py>,
    s: PyReadonlyArray3<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mats = to_mats(s.as_array());
    let mut rows = Vec::with_capacity(mats.len());
    for mt in mats {
        if mt.rows != mt.cols {
            return Err(PyValueError::new_err("matrices must be square"));
        }
        rows.push(logchol::spd_to_coords(&mt.data, mt.rows).map_err(err)?);
    }
    Ok(rows_to_py(py, rows))
}

/// Log-Cholesky coordinates (k, m(m+1)/2) -> SPD matrices (k, m, m).
#[pyfunction]
fn coords_to_spd<'py>(
    py: Python<'py>,
    z: PyReadonlyArray2<'py, f64>,
    m: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let zz = to_mat(z.as_array());
    let mut out = Vec::with_capacity(zz.rows);
    for i in 0..zz.rows {
        let s = logchol::coords_to_spd(zz.row(i), m).map_err(err)?;
        out.push(Mat {
            rows: m,
            cols: m,
            data: s,
        });
    }
    if out.is_empty() {
        return Ok(Array3::<f64>::zeros((0, m, m)).into_pyarray(py).into_any());
    }
    Ok(stack_to_py(py, out))
}

// ---------------------------------------------------------------------------
// Scalable (TRACE) primitives: zero-copy inputs via contiguous slices, GIL released.
// ---------------------------------------------------------------------------

fn slice1<'a>(a: &'a PyReadonlyArray1<'_, f64>) -> PyResult<&'a [f64]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err("array must be C-contiguous"))
}

/// Deterministic mergeable streaming quantile sketch.
#[pyclass(name = "QuantileSketch", module = "traceot._core")]
struct PyQuantileSketch {
    inner: sketch::QuantileSketch,
}

#[pymethods]
impl PyQuantileSketch {
    #[new]
    #[pyo3(signature = (k=2048))]
    fn new(k: usize) -> PyResult<Self> {
        Ok(PyQuantileSketch {
            inner: sketch::QuantileSketch::new(k).map_err(err)?,
        })
    }

    /// Absorb a 1-D float64 array (NaN ignored). Zero-copy for contiguous input.
    fn update(&mut self, py: Python<'_>, x: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let xs = slice1(&x)?;
        let inner = &mut self.inner;
        py.detach(|| inner.update(xs));
        Ok(())
    }

    /// Chunk-parallel update: sketch `chunk`-sized blocks in parallel, merge in order.
    #[pyo3(signature = (x, chunk=1_048_576))]
    fn update_chunked(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray1<'_, f64>,
        chunk: usize,
    ) -> PyResult<()> {
        let xs = slice1(&x)?;
        let inner = &mut self.inner;
        py.detach(|| inner.update_chunked(xs, chunk)).map_err(err)
    }

    fn merge(&mut self, other: &PyQuantileSketch) -> PyResult<()> {
        self.inner.merge(&other.inner).map_err(err)
    }

    fn quantiles<'py>(
        &self,
        py: Python<'py>,
        levels: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.inner.quantiles(slice1(&levels)?).into_pyarray(py))
    }

    fn bin_edges<'py>(
        &self,
        py: Python<'py>,
        max_bins: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .bin_edges(max_bins)
            .map_err(err)?
            .into_pyarray(py))
    }

    #[getter]
    fn count(&self) -> u64 {
        self.inner.count()
    }

    #[getter]
    fn retained(&self) -> usize {
        self.inner.retained()
    }
}

/// Exact bin edges from sorted, NaN-free data.
#[pyfunction]
fn bin_edges_from_sorted<'py>(
    py: Python<'py>,
    x_sorted: PyReadonlyArray1<'py, f64>,
    max_bins: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(binning::edges_from_sorted(slice1(&x_sorted)?, max_bins)
        .map_err(err)?
        .into_pyarray(py))
}

/// Bin a column: uint8 codes if `len(edges) + 2 <= 256`, else uint16.
#[pyfunction]
fn bin_column<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    edges: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let xs = slice1(&x)?;
    let es = slice1(&edges)?;
    if es.len() + 2 <= 256 {
        let out: Vec<u8> = py.detach(|| binning::bin_column(xs, es)).map_err(err)?;
        Ok(out.into_pyarray(py).into_any())
    } else {
        let out: Vec<u16> = py.detach(|| binning::bin_column(xs, es)).map_err(err)?;
        Ok(out.into_pyarray(py).into_any())
    }
}

fn hist_to_py<'py>(
    py: Python<'py>,
    hs: Vec<hist::Histogram>,
    n_bins: usize,
    m_ch: usize,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyArray2<u64>>)> {
    let p = hs.len();
    let mut sums = Vec::with_capacity(p * n_bins * m_ch);
    let mut counts = Vec::with_capacity(p * n_bins);
    for h in hs {
        sums.extend(h.sums);
        counts.extend(h.counts);
    }
    let s = Array3::from_shape_vec((p, n_bins, m_ch), sums)
        .expect("shape")
        .into_pyarray(py);
    let c = Array2::from_shape_vec((p, n_bins), counts)
        .expect("shape")
        .into_pyarray(py);
    Ok((s.into_any(), c))
}

fn grads_cols<'a>(g: &'a PyReadonlyArray2<'_, f64>) -> PyResult<(Vec<&'a [f64]>, usize)> {
    let (m, n) = g.as_array().dim();
    let all = g
        .as_slice()
        .map_err(|_| PyValueError::new_err("grads must be C-contiguous"))?;
    Ok(((0..m).map(|i| &all[i * n..(i + 1) * n]).collect(), n))
}

macro_rules! histograms_impl {
    ($name:ident, $t:ty) => {
        /// Columnar histograms: `columns` (p, n) codes, `grads` (M, n). Zero-copy inputs.
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>,
            columns: numpy::PyReadonlyArray2<'py, $t>,
            grads: PyReadonlyArray2<'py, f64>,
            n_bins: usize,
        ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyArray2<u64>>)> {
            let (p, n) = columns.as_array().dim();
            let all = columns
                .as_slice()
                .map_err(|_| PyValueError::new_err("columns must be C-contiguous"))?;
            let cols: Vec<&[$t]> = (0..p).map(|j| &all[j * n..(j + 1) * n]).collect();
            let (g, gn) = grads_cols(&grads)?;
            if gn != n {
                return Err(PyValueError::new_err("grads and columns disagree on n"));
            }
            let m_ch = g.len();
            let hs = py
                .detach(|| hist::histograms(&cols, &g, n_bins))
                .map_err(err)?;
            hist_to_py(py, hs, n_bins, m_ch)
        }
    };
}

histograms_impl!(histograms_u8, u8);
histograms_impl!(histograms_u16, u16);

/// Pinball subgradient channels (M, n) and mean pinball loss for Q = loc + scale z.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn pinball_grad_location_scale<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    loc: PyReadonlyArray1<'py, f64>,
    scale: PyReadonlyArray1<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
    u: PyReadonlyArray1<'py, f64>,
    basis: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, f64)> {
    let m_ch = basis.as_array().dim().1;
    let b = basis
        .as_slice()
        .map_err(|_| PyValueError::new_err("basis must be C-contiguous"))?;
    let (ys, ls, ss, zs, us) = (
        slice1(&y)?,
        slice1(&loc)?,
        slice1(&scale)?,
        slice1(&z)?,
        slice1(&u)?,
    );
    let (cols, loss) = py
        .detach(|| pinball::pinball_grad_location_scale(ys, ls, ss, zs, us, b, m_ch))
        .map_err(err)?;
    Ok((rows_to_py(py, cols), loss))
}

// ---------------------------------------------------------------------------
// Estimator layer: everything the Python estimators need, so that Python holds no
// numerical logic.
// ---------------------------------------------------------------------------

fn space_of(name: &str) -> PyResult<estimators::Space> {
    estimators::Space::parse(name).map_err(err)
}

fn vec_of(a: PyReadonlyArray1<'_, f64>) -> Vec<f64> {
    a.as_array().iter().copied().collect()
}

/// A fitted additive transport model (ADOPT or the L2-additive quantile model).
#[pyclass(name = "AdditiveModel", module = "traceot._core")]
struct PyAdditiveModel {
    inner: estimators::AdditiveModel,
}

#[pymethods]
impl PyAdditiveModel {
    /// Fit with fixed bandwidths ``h`` (one per predictor).
    #[staticmethod]
    #[pyo3(signature = (method, space, x, z, h, kernel="gaussian", tol=1e-10, max_iter=200, lower=None, upper=None, damping=1.0, certificate=false, composition="given", centering="frechet", project_prediction=false, bins=None, boundary="none"))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        py: Python<'_>,
        method: &str,
        space: &str,
        x: PyReadonlyArray2<'_, f64>,
        z: PyReadonlyArray2<'_, f64>,
        h: PyReadonlyArray1<'_, f64>,
        kernel: &str,
        tol: f64,
        max_iter: usize,
        lower: Option<f64>,
        upper: Option<f64>,
        damping: f64,
        certificate: bool,
        composition: &str,
        centering: &str,
        project_prediction: bool,
        bins: Option<usize>,
        boundary: &str,
    ) -> PyResult<Self> {
        let method = estimators::Method::parse(method).map_err(err)?;
        let space = space_of(space)?;
        let cfg = config(
            kernel,
            tol,
            max_iter,
            lower,
            upper,
            damping,
            certificate,
            composition,
            centering,
            project_prediction,
            bins,
            boundary,
        )?;
        let (x, z, h) = (to_mat(x.as_array()), to_mat(z.as_array()), vec_of(h));
        let inner = py
            .detach(|| estimators::AdditiveModel::fit(method, space, &x, &z, &h, cfg))
            .map_err(err)?;
        Ok(PyAdditiveModel { inner })
    }

    /// Predictions in the encoded representation, shape (m, d).
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x_new: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x = to_mat(x_new.as_array());
        let out = py.detach(|| self.inner.predict(&x)).map_err(err)?;
        Ok(mat_to_py(py, out))
    }

    /// Component curve g_j(x), shape (len(x), d).
    fn component<'py>(
        &self,
        py: Python<'py>,
        j: usize,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x = vec_of(x);
        Ok(mat_to_py(py, self.inner.component(j, &x).map_err(err)?))
    }

    /// Apply the j-th additive transport T_j(x0) to an encoded response v.
    fn transport<'py>(
        &self,
        py: Python<'py>,
        j: usize,
        x0: f64,
        v: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let v = vec_of(v);
        Ok(self
            .inner
            .transport(j, x0, &v)
            .map_err(err)?
            .into_pyarray(py))
    }

    #[getter]
    fn mu<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.fit.mu.clone().into_pyarray(py)
    }
    #[getter]
    fn centers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows_to_py(py, self.inner.fit.m.clone())
    }
    #[getter]
    fn bandwidths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.h.clone().into_pyarray(py)
    }
    #[getter]
    fn fitted<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat_to_py(py, self.inner.fit.fitted.clone())
    }
    #[getter]
    fn residuals<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        stack_to_py(py, self.inner.fit.resid.clone())
    }
    #[getter]
    fn history<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.fit.history.clone().into_pyarray(py)
    }
    #[getter]
    fn n_iter(&self) -> usize {
        self.inner.fit.n_iter
    }
    #[getter]
    fn n_rejected(&self) -> usize {
        self.inner.fit.n_rejected
    }
    #[getter]
    fn converged(&self) -> bool {
        self.inner.fit.converged
    }
    #[getter]
    fn n_features(&self) -> usize {
        self.inner.x_train.cols
    }
}

/// K-fold CV of a common bandwidth fraction; returns a dict with grid,
/// mean_dist_sq and best_fraction.
#[pyfunction]
#[pyo3(signature = (method, space, x, z, grid, k, seed, kernel="gaussian", tol=1e-10, max_iter=200, lower=None, upper=None, damping=1.0, certificate=false, composition="given", centering="frechet", project_prediction=false, bins=None, boundary="none"))]
#[allow(clippy::too_many_arguments)]
fn select_bandwidth_cv<'py>(
    py: Python<'py>,
    method: &str,
    space: &str,
    x: PyReadonlyArray2<'py, f64>,
    z: PyReadonlyArray2<'py, f64>,
    grid: PyReadonlyArray1<'py, f64>,
    k: usize,
    seed: u64,
    kernel: &str,
    tol: f64,
    max_iter: usize,
    lower: Option<f64>,
    upper: Option<f64>,
    damping: f64,
    certificate: bool,
    composition: &str,
    centering: &str,
    project_prediction: bool,
    bins: Option<usize>,
    boundary: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let method = estimators::Method::parse(method).map_err(err)?;
    let space = space_of(space)?;
    let cfg = config(
        kernel,
        tol,
        max_iter,
        lower,
        upper,
        damping,
        certificate,
        composition,
        centering,
        project_prediction,
        bins,
        boundary,
    )?;
    let (x, z, grid) = (to_mat(x.as_array()), to_mat(z.as_array()), vec_of(grid));
    let r = py
        .detach(|| estimators::select_bandwidth_cv(method, space, &x, &z, &grid, k, seed, cfg))
        .map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("grid", r.grid.into_pyarray(py))?;
    d.set_item("mean_dist_sq", r.mean_dist_sq.into_pyarray(py))?;
    d.set_item("best_fraction", r.best_fraction)?;
    Ok(d)
}

/// Shuffled K-fold split (SplitMix64 Fisher-Yates); list of sorted index arrays.
#[pyfunction]
fn kfold_indices<'py>(
    py: Python<'py>,
    n: usize,
    k: usize,
    seed: u64,
) -> PyResult<Vec<Bound<'py, PyArray1<usize>>>> {
    let folds = estimators::kfold_indices(n, k, seed).map_err(err)?;
    Ok(folds.into_iter().map(|f| f.into_pyarray(py)).collect())
}

/// Observed range of each column (zero ranges replaced by one).
#[pyfunction]
fn column_ranges<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, f64>) -> Bound<'py, PyArray1<f64>> {
    estimators::column_ranges(&to_mat(x.as_array())).into_pyarray(py)
}

/// Global Fréchet regression predictions (encoded).
#[pyfunction]
#[pyo3(signature = (space, x_train, z_train, x_new, lower=None, upper=None))]
fn global_frechet_predict<'py>(
    py: Python<'py>,
    space: &str,
    x_train: PyReadonlyArray2<'py, f64>,
    z_train: PyReadonlyArray2<'py, f64>,
    x_new: PyReadonlyArray2<'py, f64>,
    lower: Option<f64>,
    upper: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let space = space_of(space)?;
    let cfg = config_basic("gaussian", 0.0, 1, lower, upper)?;
    let (xt, zt, xn) = (
        to_mat(x_train.as_array()),
        to_mat(z_train.as_array()),
        to_mat(x_new.as_array()),
    );
    let out = py
        .detach(|| estimators::global_frechet_predict(space, &xt, &zt, &xn, &cfg))
        .map_err(err)?;
    Ok(mat_to_py(py, out))
}

/// Column means of encoded responses (the Fréchet mean in both spaces).
#[pyfunction]
fn frechet_mean<'py>(py: Python<'py>, z: PyReadonlyArray2<'py, f64>) -> Bound<'py, PyArray1<f64>> {
    to_mat(z.as_array()).col_means().into_pyarray(py)
}

/// Row-wise squared distances in the given space.
#[pyfunction]
fn dist_sq_rows<'py>(
    py: Python<'py>,
    space: &str,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let space = space_of(space)?;
    let out = estimators::dist_sq_rows(space, &to_mat(a.as_array()), &to_mat(b.as_array()))
        .map_err(err)?;
    Ok(out.into_pyarray(py))
}

/// Fréchet R^2 = 1 - sum d^2(pred, z) / sum d^2(mean, z).
#[pyfunction]
fn frechet_r2(
    space: &str,
    pred: PyReadonlyArray2<'_, f64>,
    z: PyReadonlyArray2<'_, f64>,
    mean: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    let space = space_of(space)?;
    estimators::frechet_r2(
        space,
        &to_mat(pred.as_array()),
        &to_mat(z.as_array()),
        &vec_of(mean),
    )
    .map_err(err)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(quantile_grid, m)?)?;
    m.add_function(wrap_pyfunction!(w2_sq_rows, m)?)?;
    m.add_function(wrap_pyfunction!(pava, m)?)?;
    m.add_function(wrap_pyfunction!(pava_rows, m)?)?;
    m.add_function(wrap_pyfunction!(collapse_knots, m)?)?;
    m.add_function(wrap_pyfunction!(transport_apply, m)?)?;
    m.add_function(wrap_pyfunction!(local_linear_weights, m)?)?;
    m.add_function(wrap_pyfunction!(reflected_local_linear_weights, m)?)?;
    m.add_function(wrap_pyfunction!(local_linear_weights_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(binned_smooth, m)?)?;
    m.add_function(wrap_pyfunction!(global_frechet_weights, m)?)?;
    m.add_function(wrap_pyfunction!(adopt_w2_fit, m)?)?;
    m.add_function(wrap_pyfunction!(adopt_w2_predict, m)?)?;
    m.add_function(wrap_pyfunction!(adopt_w2_predict_averaged, m)?)?;
    m.add_function(wrap_pyfunction!(adopt_w2_component, m)?)?;
    m.add_function(wrap_pyfunction!(additive_euclid_fit, m)?)?;
    m.add_function(wrap_pyfunction!(additive_euclid_predict, m)?)?;
    m.add_function(wrap_pyfunction!(spd_to_coords, m)?)?;
    m.add_function(wrap_pyfunction!(coords_to_spd, m)?)?;
    m.add_class::<PyQuantileSketch>()?;
    m.add_class::<PyAdditiveModel>()?;
    m.add_function(wrap_pyfunction!(select_bandwidth_cv, m)?)?;
    m.add_function(wrap_pyfunction!(kfold_indices, m)?)?;
    m.add_function(wrap_pyfunction!(column_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(global_frechet_predict, m)?)?;
    m.add_function(wrap_pyfunction!(frechet_mean, m)?)?;
    m.add_function(wrap_pyfunction!(dist_sq_rows, m)?)?;
    m.add_function(wrap_pyfunction!(frechet_r2, m)?)?;
    m.add_function(wrap_pyfunction!(bin_edges_from_sorted, m)?)?;
    m.add_function(wrap_pyfunction!(bin_column, m)?)?;
    m.add_function(wrap_pyfunction!(histograms_u8, m)?)?;
    m.add_function(wrap_pyfunction!(histograms_u16, m)?)?;
    m.add_function(wrap_pyfunction!(pinball_grad_location_scale, m)?)?;
    trace_api::register(m)?;
    v1_api::register(m)?;
    shift_v2_api::register(m)?;
    v4_api::register(m)?;
    Ok(())
}
