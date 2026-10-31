//! Thin PyO3 surface for TRACE v5. Numerics stay in `traceot_core::trace::v5`.

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use traceot_core::trace::{
    explain_v5, mixture_cdf_location, mixture_cdf_quantile_rows, quantile_from_cdf,
    ExplainV5Config, Hybrid, LocationModel,
};
use traceot_core::TraceError;

fn err(e: TraceError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn contig<'a>(a: &'a PyReadonlyArray1<'a, f64>) -> PyResult<&'a [f64]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err("array must be C-contiguous"))
}

fn contig2<'a>(a: &'a PyReadonlyArray2<'a, f64>) -> PyResult<&'a [f64]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err("array must be C-contiguous"))
}

fn hybrid_dict<'py>(py: Python<'py>, h: &Hybrid) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("q_mix", h.q_mix.clone().into_pyarray(py))?;
    d.set_item("q_bary", h.q_bary.clone().into_pyarray(py))?;
    d.set_item("lmom", h.lmom.clone())?;
    d.set_item("lmom_bary", h.lmom_bary.clone())?;
    d.set_item("mixing_residual", h.mixing_residual)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (x_a, y_a, x_b, y_b, *, n_y=256usize, k_lmoments=4usize, n_quad=64usize, mean_model="booster"))]
#[allow(clippy::too_many_arguments)]
fn explain_v5_py<'py>(
    py: Python<'py>,
    x_a: PyReadonlyArray2<'py, f64>,
    y_a: PyReadonlyArray1<'py, f64>,
    x_b: PyReadonlyArray2<'py, f64>,
    y_b: PyReadonlyArray1<'py, f64>,
    n_y: usize,
    k_lmoments: usize,
    n_quad: usize,
    mean_model: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let mean_model = match mean_model {
        "booster" => LocationModel::Booster,
        "ols" => LocationModel::Ols,
        _ => {
            return Err(PyValueError::new_err(
                "mean_model must be 'booster' or 'ols'",
            ))
        }
    };
    let (na, pa) = x_a.as_array().dim();
    let (nb, pb) = x_b.as_array().dim();
    if pa != pb {
        return Err(PyValueError::new_err("x_a and x_b must share p"));
    }
    let cfg = ExplainV5Config {
        n_y,
        k_lmoments,
        n_quad,
        mean_model,
        ..ExplainV5Config::default()
    };
    let (xa, ya, xb, yb) = (contig2(&x_a)?, contig(&y_a)?, contig2(&x_b)?, contig(&y_b)?);
    let rep = py
        .detach(|| explain_v5(xa, na, ya, xb, nb, yb, pa, &cfg))
        .map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("u_nodes", rep.u_nodes.clone().into_pyarray(py))?;
    d.set_item("y_grid", rep.y_grid.clone().into_pyarray(py))?;
    d.set_item("aa", hybrid_dict(py, &rep.aa)?)?;
    d.set_item("ab", hybrid_dict(py, &rep.ab)?)?;
    d.set_item("ba", hybrid_dict(py, &rep.ba)?)?;
    d.set_item("bb", hybrid_dict(py, &rep.bb)?)?;
    d.set_item("lsplit_comp", rep.lsplit_comp.clone())?;
    d.set_item("lsplit_struct", rep.lsplit_struct.clone())?;
    d.set_item("seconds_fit", rep.seconds_fit)?;
    d.set_item("seconds_mixture", rep.seconds_mixture)?;
    d.set_item("seconds_total", rep.seconds_total)?;
    Ok(d)
}

#[pyfunction]
fn mixture_cdf_location_py<'py>(
    py: Python<'py>,
    mu: PyReadonlyArray1<'py, f64>,
    resid: PyReadonlyArray1<'py, f64>,
    y_grid: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let (m, r, g) = (contig(&mu)?, contig(&resid)?, contig(&y_grid)?);
    let out = py.detach(|| mixture_cdf_location(m, r, g)).map_err(err)?;
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (u_nodes, q_rows, y_grid, weights=None))]
fn mixture_cdf_rows_py<'py>(
    py: Python<'py>,
    u_nodes: PyReadonlyArray1<'py, f64>,
    q_rows: PyReadonlyArray2<'py, f64>,
    y_grid: PyReadonlyArray1<'py, f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let (n, _) = q_rows.as_array().dim();
    let (u, q, g) = (contig(&u_nodes)?, contig2(&q_rows)?, contig(&y_grid)?);
    let w = match &weights {
        Some(w) => Some(contig(w)?),
        None => None,
    };
    let out = py
        .detach(|| mixture_cdf_quantile_rows(u, q, n, w, g))
        .map_err(err)?;
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn quantile_from_cdf_py<'py>(
    py: Python<'py>,
    y_grid: PyReadonlyArray1<'py, f64>,
    f_grid: PyReadonlyArray1<'py, f64>,
    u: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let (y, f, uu) = (contig(&y_grid)?, contig(&f_grid)?, contig(&u)?);
    if y.len() != f.len() || y.len() < 2 {
        return Err(PyValueError::new_err(
            "y_grid and f_grid must match, length >= 2",
        ));
    }
    Ok(quantile_from_cdf(y, f, uu).into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(explain_v5_py, m)?)?;
    m.add_function(wrap_pyfunction!(mixture_cdf_location_py, m)?)?;
    m.add_function(wrap_pyfunction!(mixture_cdf_rows_py, m)?)?;
    m.add_function(wrap_pyfunction!(quantile_from_cdf_py, m)?)?;
    Ok(())
}
