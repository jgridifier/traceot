//! PyO3 bindings for `traceot-core`, exposed to Python as `traceot._core`.
//!
//! Function names and signatures follow `python/traceot/reference/` so that the two
//! backends are interchangeable (see `python/traceot/_backend.py`).

use numpy::ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use traceot_core::adopt::{BackfitConfig, BackfitResult};
use traceot_core::{additive, adopt, logchol, smoother, w2, Kernel, Mat, TraceError};

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

fn config(
    kernel: &str,
    tol: f64,
    max_iter: usize,
    lower: Option<f64>,
    upper: Option<f64>,
) -> PyResult<BackfitConfig> {
    Ok(BackfitConfig {
        kernel: Kernel::parse(kernel).map_err(err)?,
        tol,
        max_iter,
        lower,
        upper,
    })
}

fn result_to_dict<'py>(py: Python<'py>, r: BackfitResult) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("mu", Array1::from_vec(r.mu).into_pyarray(py))?;
    d.set_item("G", stack_to_py(py, r.g))?;
    d.set_item("M", rows_to_py(py, r.m))?;
    d.set_item("P", stack_to_py(py, r.resid))?;
    d.set_item("fitted", mat_to_py(py, r.fitted))?;
    d.set_item("n_iter", r.n_iter)?;
    d.set_item("converged", r.converged)?;
    d.set_item("history", Array1::from_vec(r.history).into_pyarray(py))?;
    Ok(d)
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
#[pyo3(signature = (x, y, h, kernel="gaussian", tol=1e-10, max_iter=200, lower=None, upper=None))]
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
) -> PyResult<Bound<'py, PyDict>> {
    let cfg = config(kernel, tol, max_iter, lower, upper)?;
    let (x, y) = (to_mat(x.as_array()), to_mat(y.as_array()));
    let h: Vec<f64> = h.as_array().iter().copied().collect();
    let r = py
        .detach(|| adopt::adopt_w2_fit(&x, &y, &h, &cfg))
        .map_err(err)?;
    result_to_dict(py, r)
}

/// ADOPT prediction at new predictor values.
#[pyfunction]
#[pyo3(signature = (x_train, p, m, mu, h, x_new, kernel="gaussian", lower=None, upper=None))]
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
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let cfg = config(kernel, 0.0, 0, lower, upper)?;
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

/// `g_j(x)` at new points from stored residuals.
#[pyfunction]
#[pyo3(signature = (x_train_j, p_j, h_j, x_eval, kernel="gaussian", lower=None, upper=None))]
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
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let cfg = config(kernel, 0.0, 0, lower, upper)?;
    let xt: Vec<f64> = x_train_j.as_array().iter().copied().collect();
    let xe: Vec<f64> = x_eval.as_array().iter().copied().collect();
    let pj = to_mat(p_j.as_array());
    let out = adopt::adopt_w2_component(&xt, &pj, h_j, &xe, &cfg).map_err(err)?;
    Ok(mat_to_py(py, out))
}

/// Additive backfitting in flat coordinates.
#[pyfunction]
#[pyo3(signature = (x, z, h, kernel="gaussian", tol=1e-10, max_iter=200))]
fn additive_euclid_fit<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    z: PyReadonlyArray2<'py, f64>,
    h: PyReadonlyArray1<'py, f64>,
    kernel: &str,
    tol: f64,
    max_iter: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let cfg = config(kernel, tol, max_iter, None, None)?;
    let (x, z) = (to_mat(x.as_array()), to_mat(z.as_array()));
    let h: Vec<f64> = h.as_array().iter().copied().collect();
    let r = py
        .detach(|| additive::additive_euclid_fit(&x, &z, &h, &cfg))
        .map_err(err)?;
    result_to_dict(py, r)
}

/// Additive prediction in flat coordinates.
#[pyfunction]
#[pyo3(signature = (x_train, p, m, mu, h, x_new, kernel="gaussian"))]
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
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let cfg = config(kernel, 0.0, 0, None, None)?;
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
    m.add_function(wrap_pyfunction!(global_frechet_weights, m)?)?;
    m.add_function(wrap_pyfunction!(adopt_w2_fit, m)?)?;
    m.add_function(wrap_pyfunction!(adopt_w2_predict, m)?)?;
    m.add_function(wrap_pyfunction!(adopt_w2_component, m)?)?;
    m.add_function(wrap_pyfunction!(additive_euclid_fit, m)?)?;
    m.add_function(wrap_pyfunction!(additive_euclid_predict, m)?)?;
    m.add_function(wrap_pyfunction!(spd_to_coords, m)?)?;
    m.add_function(wrap_pyfunction!(coords_to_spd, m)?)?;
    Ok(())
}
