//! Thin PyO3 surface for TRACE v4. Numerics stay in `traceot_core::trace::v4`.

#![allow(clippy::too_many_arguments)]

use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use traceot_core::trace::{
    explain_v4, ExplainV4Config, FitL2Config, InferConfig, MeanConfig, MixtureConfig,
};
use traceot_core::TraceError;

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

#[pyfunction]
#[pyo3(signature = (
    x_a, y_a, x_b, y_b, *,
    k_channels=6, seed=0u64, n_boot=2000usize, alpha=0.05,
    max_bins=64usize, pit_iters=2usize, n_u=64usize
))]
fn explain_v4_py<'py>(
    py: Python<'py>,
    x_a: PyReadonlyArray2<'py, f64>,
    y_a: PyReadonlyArray1<'py, f64>,
    x_b: PyReadonlyArray2<'py, f64>,
    y_b: PyReadonlyArray1<'py, f64>,
    k_channels: usize,
    seed: u64,
    n_boot: usize,
    alpha: f64,
    max_bins: usize,
    pit_iters: usize,
    n_u: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let xa = x_a.as_array();
    let xb = x_b.as_array();
    let (na, pa) = xa.dim();
    let (nb, pb) = xb.dim();
    if pa != pb {
        return Err(PyValueError::new_err("x_a and x_b must share p"));
    }
    let p = pa;
    let cfg = ExplainV4Config {
        fit: FitL2Config {
            k_channels,
            pit_iters,
            max_bins,
            mean: MeanConfig {
                max_bins,
                learning_rate: 0.1,
                max_rounds: 200,
                early_stopping_rounds: 20,
                early_stopping_tolerance: 1e-5,
                validation_size: 0.15,
                min_samples_leaf: 4,
            },
            seed,
            ..FitL2Config::default()
        },
        mixture: MixtureConfig {
            n_y: 256,
            max_rows: Some(200_000),
            k_channels,
            n_quad: n_u,
            subsample_seed: seed,
        },
        infer: InferConfig {
            n_boot,
            n_perm: 0,
            seed,
            alpha,
        },
        n_u,
    };
    let (xa_s, ya_s, xb_s, yb_s) = (slice2(&x_a)?, slice1(&y_a)?, slice2(&x_b)?, slice1(&y_b)?);
    let report = py
        .detach(|| explain_v4(xa_s, ya_s, na, xb_s, yb_s, nb, p, &cfg))
        .map_err(err)?;

    let d = PyDict::new(py);
    d.set_item("k_channels", report.k_channels)?;
    d.set_item("p", report.p)?;
    d.set_item("budget", report.budget)?;
    d.set_item("budget_se", report.budget_se)?;
    d.set_item("budget_ci_lo", report.budget_ci_lo)?;
    d.set_item("budget_ci_hi", report.budget_ci_hi)?;
    d.set_item("interaction_mass", report.interaction_mass)?;
    d.set_item("mixing_residual", report.mixing_residual)?;
    d.set_item("seconds_fit", report.seconds_fit)?;
    d.set_item("seconds_game", report.seconds_game)?;
    d.set_item("seconds_infer", report.seconds_infer)?;
    d.set_item("seconds_total", report.seconds_total)?;

    let mean = PyDict::new(py);
    mean.set_item("comp", report.mean.comp)?;
    mean.set_item("struct", report.mean.structure)?;
    mean.set_item("mix_comp", report.mean.mix_comp)?;
    mean.set_item("mix_struct", report.mean.mix_structure)?;
    d.set_item("mean", mean)?;

    d.set_item(
        "feat_comp",
        Array1::from_vec(report.feat_comp).into_pyarray(py),
    )?;
    d.set_item(
        "feat_struct",
        Array1::from_vec(report.feat_struct).into_pyarray(py),
    )?;
    d.set_item(
        "feat_mix_comp",
        Array1::from_vec(report.feat_mix_comp).into_pyarray(py),
    )?;
    d.set_item(
        "feat_mix_struct",
        Array1::from_vec(report.feat_mix_struct).into_pyarray(py),
    )?;

    // ShiftBench-facing split for the mean statistic
    let split = PyDict::new(py);
    split.set_item(
        "mean",
        (
            report.mean.comp + report.mean.mix_comp,
            report.mean.structure + report.mean.mix_structure,
        ),
    )?;
    d.set_item("split", split)?;
    d.set_item(
        "intercept",
        report.mean.structure, // structure intercept proxy
    )?;

    let players = PyList::empty(py);
    for pl in &report.players {
        let row = PyDict::new(py);
        row.set_item("channel", pl.channel)?;
        row.set_item("feature", pl.feature)?;
        row.set_item("role", pl.role)?;
        row.set_item("value", pl.value)?;
        players.append(row)?;
    }
    d.set_item("players", players)?;

    let config = PyDict::new(py);
    config.set_item("k_channels", k_channels)?;
    config.set_item("seed", seed)?;
    config.set_item("n_boot", n_boot)?;
    config.set_item("version", "v4")?;
    d.set_item("config", config)?;
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(explain_v4_py, m)?)?;
    // Alias matching the Python name
    m.add("explain_v4", m.getattr("explain_v4_py")?)?;
    Ok(())
}
