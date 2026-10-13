//! PyO3 surface for the v1 Legendre fitter, Fréchet rows, shifts and robust risk.
//! Numerics stay in `traceot_core::trace`.

#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use traceot_core::trace::{
    attribute, bootstrap_indices, channel_energy, channel_slope, coupling_w1, cross_fit_energy,
    crps_quadrature, eval_a, feature_pinball_gains, feature_report_table, fit_frechet, fit_hybrid,
    fit_mean, fit_robust, fit_v1, gauss_legendre_01, hosking_lmoments, hybrid_channel_energy,
    integrated_pinball_on, leave_feature_gains, lmoment_channel_energy, max_jump_bound,
    monotonicity_report, outcome_only_w1_gap, permutation_pvalue, pinball, pinball_gh,
    predict_frechet, predict_hybrid, predict_mean, predict_mean_channel, predict_quantiles,
    predict_robust, projection_pair, report_matrix, robust_risk_breakpoints, robust_risk_level,
    shift_conditional, shift_marginal, weighted_pav, winf_risk, Ball, FrechetConfig, FrechetModel,
    HybridModel, MeanConfig, MonoMode, ReportKind, RobustConfig, RobustModel, V1Config, V1Model,
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

pub(crate) fn mono_from(name: &str) -> PyResult<MonoMode> {
    match name {
        "none" => Ok(MonoMode::None),
        "pav" => Ok(MonoMode::Pav),
        "bernstein" => Ok(MonoMode::Bernstein),
        _ => Err(PyValueError::new_err(
            "monotonicity must be none, pav, or bernstein",
        )),
    }
}

pub(crate) fn report_from(name: &str) -> PyResult<ReportKind> {
    match name {
        "legendre" => Ok(ReportKind::Legendre),
        "hermite" => Ok(ReportKind::Hermite),
        "reference" => Ok(ReportKind::Reference),
        _ => Err(PyValueError::new_err(
            "reporting must be legendre, hermite, or reference",
        )),
    }
}

fn ball_from(name: &str) -> PyResult<Ball> {
    match name {
        "w1" => Ok(Ball::W1),
        "w2" => Ok(Ball::W2),
        "winf" => Ok(Ball::WInf),
        _ => Err(PyValueError::new_err("ball must be w1, w2, or winf")),
    }
}

fn config_from(
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
        pit_rounds: 0,
    })
}

#[pyclass(name = "TraceV1Model", module = "traceot._core")]
pub struct PyV1Model {
    pub(crate) inner: V1Model,
}

#[pymethods]
impl PyV1Model {
    #[staticmethod]
    #[pyo3(signature = (
        x, y, k_fit=6, k_report=4, n_u=0, max_bins=16, max_rounds=30, tol=1e-9, min_count=1,
        ridge=1e-6, learning_rate=1.0, mono="pav", bernstein_elevation=4, reporting="legendre",
        smooth=0.0, levenberg=0.0, sobolev_lambda=0.0, sobolev_s=1.0, tail_gamma=0.0,
        pit_rounds=0
    ))]
    fn fit(
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
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
        smooth: f64,
        levenberg: f64,
        sobolev_lambda: f64,
        sobolev_s: f64,
        tail_gamma: f64,
        pit_rounds: usize,
    ) -> PyResult<Self> {
        let (n, p) = x.as_array().dim();
        let mut cfg = config_from(
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
        cfg.smooth = smooth;
        cfg.levenberg = levenberg;
        cfg.sobolev_lambda = sobolev_lambda;
        cfg.sobolev_s = sobolev_s;
        cfg.tail_gamma = tail_gamma;
        cfg.pit_rounds = pit_rounds;
        let (xs, ys) = (slice2(&x)?, slice1(&y)?);
        let inner = py.detach(|| fit_v1(xs, n, p, ys, &cfg)).map_err(err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn k_fit(&self) -> usize {
        self.inner.k_fit
    }
    #[getter]
    fn k_report(&self) -> usize {
        self.inner.k_report
    }
    #[getter]
    fn rounds(&self) -> usize {
        self.inner.rounds
    }
    #[getter]
    fn train_loss(&self) -> f64 {
        self.inner.train_loss
    }
    #[getter]
    fn non_representable(&self) -> bool {
        self.inner.non_representable
    }
    #[getter]
    fn bernstein_lift(&self) -> f64 {
        self.inner.bernstein_lift
    }
    #[getter]
    fn n_features(&self) -> usize {
        self.inner.edges.len()
    }

    fn u_nodes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from_vec(self.inner.basis.nodes.clone()).into_pyarray(py)
    }

    fn intercept<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from_vec(self.inner.intercept.clone()).into_pyarray(py)
    }

    fn edges<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for e in &self.inner.edges {
            list.append(Array1::from_vec(e.clone()).into_pyarray(py))?;
        }
        Ok(list)
    }

    /// Reporting-basis coefficients of feature `j`, shape `(n_bins, k_report)`.
    fn feature_table<'py>(&self, py: Python<'py>, j: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let tables = feature_report_table(&self.inner);
        if j >= tables.len() {
            return Err(PyValueError::new_err("feature index out of range"));
        }
        let kr = self.inner.k_report;
        let nb = tables[j].len() / kr;
        Ok(Array2::from_shape_vec((nb, kr), tables[j].clone())
            .expect("shape")
            .into_pyarray(py))
    }

    fn channel_energy(&self, feature: usize, channel: usize) -> PyResult<f64> {
        let tables = feature_report_table(&self.inner);
        if feature >= tables.len() || channel >= self.inner.k_report {
            return Err(PyValueError::new_err("channel or feature out of range"));
        }
        Ok(channel_energy(
            &tables[feature],
            self.inner.k_report,
            channel,
        ))
    }

    /// Held-out pinball gains. `rows` indexes the validation rows.
    fn feature_gains<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        rows: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (n, _) = x.as_array().dim();
        let idx: Vec<usize> = rows
            .as_slice()
            .map_err(|_| PyValueError::new_err("rows must be C-contiguous"))?
            .iter()
            .map(|&v| {
                if v < 0 {
                    Err(PyValueError::new_err("row index is negative"))
                } else {
                    Ok(v as usize)
                }
            })
            .collect::<PyResult<_>>()?;
        let (xs, ys) = (slice2(&x)?, slice1(&y)?);
        let screen = py
            .detach(|| feature_pinball_gains(&self.inner, xs, n, ys, &idx))
            .map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("gains", screen.gains)?;
        d.set_item("shape_gains", screen.shape_gains)?;
        d.set_item("selected", screen.selected)?;
        Ok(d)
    }

    fn channel_slope(
        &self,
        feature: usize,
        channel: usize,
        x: PyReadonlyArray1<f64>,
    ) -> PyResult<f64> {
        channel_slope(&self.inner, feature, channel, slice1(&x)?).map_err(err)
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        u: PyReadonlyArray1<'_, f64>,
        project: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (n, _) = x.as_array().dim();
        let (xs, uu) = (slice2(&x)?, slice1(&u)?);
        let flat = py
            .detach(|| predict_quantiles(&self.inner, xs, n, uu, project))
            .map_err(err)?;
        Ok(Array2::from_shape_vec((n, uu.len()), flat)
            .expect("shape")
            .into_pyarray(py))
    }

    fn attribute<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (n, _) = x.as_array().dim();
        let xs = slice2(&x)?;
        let attr = py.detach(|| attribute(&self.inner, xs, n)).map_err(err)?;
        let d = PyDict::new(py);
        let players = attr.p + 1;
        d.set_item(
            "phi_fit",
            Array3::from_shape_vec((n, players, attr.k_fit), attr.phi_fit)
                .expect("shape")
                .into_pyarray(py),
        )?;
        d.set_item(
            "phi_report",
            Array3::from_shape_vec((n, players, attr.k_report), attr.phi_report)
                .expect("shape")
                .into_pyarray(py),
        )?;
        d.set_item(
            "truncation",
            Array1::from_vec(attr.truncation).into_pyarray(py),
        )?;
        d.set_item(
            "mono_adjustment",
            Array1::from_vec(attr.mono_adjustment).into_pyarray(py),
        )?;
        d.set_item("d", Array1::from_vec(attr.d).into_pyarray(py))?;
        Ok(d)
    }

    fn mono_report<'py>(&self, py: Python<'py>, grid: usize) -> PyResult<Bound<'py, PyDict>> {
        let r = monotonicity_report(&self.inner, grid);
        let d = PyDict::new(py);
        d.set_item("worst_derivative", r.worst_derivative)?;
        d.set_item("non_representable", r.non_representable)?;
        d.set_item(
            "bernstein_certified_intercept",
            r.bernstein_certified_intercept,
        )?;
        d.set_item("rows_checked", r.rows_checked)?;
        d.set_item("rows_monotone", r.rows_monotone)?;
        Ok(d)
    }

    fn shift_conditional<'py>(
        &self,
        py: Python<'py>,
        other: &PyV1Model,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (n, _) = x.as_array().dim();
        let xs = slice2(&x)?;
        let s = py
            .detach(|| shift_conditional(&self.inner, &other.inner, xs, n))
            .map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("total", s.total)?;
        d.set_item(
            "phi_mean",
            Array2::from_shape_vec((s.p + 1, s.k_fit), s.phi_mean)
                .expect("shape")
                .into_pyarray(py),
        )?;
        Ok(d)
    }

    fn shift_marginal<'py>(
        &self,
        py: Python<'py>,
        other: &PyV1Model,
        x_a: PyReadonlyArray2<'_, f64>,
        x_b: PyReadonlyArray2<'_, f64>,
        grid: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (na, _) = x_a.as_array().dim();
        let (nb, _) = x_b.as_array().dim();
        let (xa, xb) = (slice2(&x_a)?, slice2(&x_b)?);
        let s = py
            .detach(|| shift_marginal(&self.inner, &other.inner, xa, na, xb, nb, grid))
            .map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("total", s.total)?;
        d.set_item("composition", s.composition)?;
        d.set_item("structure", s.structure)?;
        d.set_item("cross", s.cross)?;
        d.set_item("composition_first", s.composition_first)?;
        d.set_item("structure_first_composition", s.structure_first_composition)?;
        d.set_item("shapley_composition", s.shapley_composition)?;
        d.set_item("shapley_structure", s.shapley_structure)?;
        d.set_item("overlap_failures", s.overlap_failures)?;
        Ok(d)
    }
}

#[pyclass(name = "FrechetModel", module = "traceot._core")]
pub struct PyFrechet {
    inner: FrechetModel,
}

#[pymethods]
impl PyFrechet {
    #[staticmethod]
    #[pyo3(signature = (
        x, samples, lengths, k_fit=4, max_bins=16, max_rounds=40, tol=1e-12,
        estimate_weights=true, weights=None
    ))]
    fn fit(
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        samples: PyReadonlyArray2<'_, f64>,
        lengths: PyReadonlyArray1<'_, i64>,
        k_fit: usize,
        max_bins: usize,
        max_rounds: usize,
        tol: f64,
        estimate_weights: bool,
        weights: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Self> {
        let (n, p) = x.as_array().dim();
        let (_, m_max) = samples.as_array().dim();
        let lens: Vec<usize> = slice_i64(&lengths)?.iter().map(|&v| v as usize).collect();
        let cfg = FrechetConfig {
            k_fit,
            max_bins,
            max_rounds,
            tol,
            estimate_weights,
        };
        let w = match &weights {
            Some(v) => Some(slice1(v)?.to_vec()),
            None => None,
        };
        let (xs, samples) = (slice2(&x)?, slice2(&samples)?);
        let inner = py
            .detach(|| fit_frechet(xs, n, p, samples, m_max, &lens, &cfg, w.as_deref()))
            .map_err(err)?;
        Ok(Self { inner })
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (n, _) = x.as_array().dim();
        let xs = slice2(&x)?;
        let flat = py
            .detach(|| predict_frechet(&self.inner, xs, n))
            .map_err(err)?;
        Ok(Array2::from_shape_vec((n, self.inner.k_fit), flat)
            .expect("shape")
            .into_pyarray(py))
    }

    fn variance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let rows: Vec<f64> = self
            .inner
            .variance
            .iter()
            .flat_map(|(t, v)| [*t, *v])
            .collect();
        Array2::from_shape_vec((self.inner.k_fit, 2), rows)
            .expect("shape")
            .into_pyarray(py)
    }

    fn channel_weights<'py>(
        &self,
        py: Python<'py>,
        channel: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if channel >= self.inner.channel_weights.len() {
            return Err(PyValueError::new_err("channel out of range"));
        }
        Ok(Array1::from_vec(self.inner.channel_weights[channel].clone()).into_pyarray(py))
    }

    #[getter]
    fn intercept<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from_vec(self.inner.intercept.clone()).into_pyarray(py)
    }
}

#[pyclass(name = "RobustQuantileModel", module = "traceot._core")]
pub struct PyRobust {
    inner: RobustModel,
}

#[pymethods]
impl PyRobust {
    #[staticmethod]
    #[pyo3(signature = (
        x, y, eps=0.0, max_bins=8, rounds=40, step=0.6, ball="w1", kappa=None, kappa_y=None,
        levels=None
    ))]
    fn fit(
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        eps: f64,
        max_bins: usize,
        rounds: usize,
        step: f64,
        ball: &str,
        kappa: Option<PyReadonlyArray1<f64>>,
        kappa_y: Option<f64>,
        levels: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Self> {
        let (n, p) = x.as_array().dim();
        let cfg = RobustConfig {
            levels: match levels {
                Some(v) => slice1(&v)?.to_vec(),
                None => RobustConfig::default().levels,
            },
            max_bins,
            rounds,
            step,
            eps,
            kappa: match kappa {
                Some(v) => slice1(&v)?.to_vec(),
                None => Vec::new(),
            },
            ball: ball_from(ball)?,
            kappa_y,
        };
        let (xs, ys) = (slice2(&x)?, slice1(&y)?);
        let inner = py.detach(|| fit_robust(xs, n, p, ys, &cfg)).map_err(err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn train_loss(&self) -> f64 {
        self.inner.train_loss
    }
    #[getter]
    fn robust_value(&self) -> f64 {
        self.inner.robust_value
    }
    #[getter]
    fn eps(&self) -> f64 {
        self.inner.eps
    }

    fn kappa<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from_vec(self.inner.kappa.clone()).into_pyarray(py)
    }

    fn edges<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for e in &self.inner.edges {
            list.append(Array1::from_vec(e.clone()).into_pyarray(py))?;
        }
        Ok(list)
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (n, _) = x.as_array().dim();
        let xs = slice2(&x)?;
        let flat = py
            .detach(|| predict_robust(&self.inner, xs, n))
            .map_err(err)?;
        Ok(Array2::from_shape_vec((n, self.inner.levels.len()), flat)
            .expect("shape")
            .into_pyarray(py))
    }

    fn pinball(&self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
        integrated_pinball_on(&self.inner, slice2(&x)?, slice1(&y)?).map_err(err)
    }
}

fn slice_i64<'a>(a: &'a PyReadonlyArray1<'a, i64>) -> PyResult<&'a [i64]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err("array must be C-contiguous"))
}

fn read_shapes(list: &Bound<'_, PyList>) -> PyResult<Vec<Vec<f64>>> {
    let mut out = Vec::with_capacity(list.len());
    for item in list.iter() {
        let arr = item.extract::<PyReadonlyArray1<f64>>()?;
        out.push(slice1(&arr)?.to_vec());
    }
    Ok(out)
}

fn read_bins(a: &PyReadonlyArray2<i64>) -> PyResult<Vec<Vec<usize>>> {
    let arr = a.as_array();
    let (n, p) = arr.dim();
    let mut bins = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(p);
        for j in 0..p {
            let v = arr[[i, j]];
            if v < 0 {
                return Err(PyValueError::new_err("bin codes must be nonnegative"));
            }
            row.push(v as usize);
        }
        bins.push(row);
    }
    Ok(bins)
}

#[pyfunction]
fn v1_gauss_legendre<'py>(
    py: Python<'py>,
    n: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let gl = gauss_legendre_01(n).map_err(err)?;
    Ok((
        Array1::from_vec(gl.nodes).into_pyarray(py),
        Array1::from_vec(gl.weights).into_pyarray(py),
    ))
}

#[pyfunction]
fn v1_hosking<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    order: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let lam = hosking_lmoments(slice1(&x)?, order).map_err(err)?;
    Ok(Array1::from_vec(lam).into_pyarray(py))
}

#[pyfunction]
fn v1_weighted_pav<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    w: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let z = weighted_pav(slice1(&y)?, slice1(&w)?).map_err(err)?;
    Ok(Array1::from_vec(z).into_pyarray(py))
}

#[pyfunction]
fn v1_eval_a(a: PyReadonlyArray1<f64>, u: f64) -> PyResult<f64> {
    Ok(eval_a(slice1(&a)?, u))
}

#[pyfunction]
fn v1_report_matrix<'py>(
    py: Python<'py>,
    k_fit: usize,
    k_report: usize,
    kind: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let code = match kind {
        "legendre" => 0,
        "hermite" => 1,
        "reference" => 2,
        _ => return Err(PyValueError::new_err("unknown reporting basis")),
    };
    let g = report_matrix(k_fit, k_report, code, &[]).map_err(err)?;
    Ok(Array2::from_shape_vec((k_fit, k_report), g)
        .expect("shape")
        .into_pyarray(py))
}

#[pyfunction]
fn v1_robust_risk(
    intercept: f64,
    shapes: &Bound<'_, PyList>,
    bins: PyReadonlyArray2<i64>,
    y: PyReadonlyArray1<f64>,
    u: f64,
    kappa: PyReadonlyArray1<f64>,
    eps: f64,
    ball: &str,
    kappa_y: Option<f64>,
    breakpoints: bool,
) -> PyResult<f64> {
    let f = read_shapes(shapes)?;
    let b = read_bins(&bins)?;
    let ball = ball_from(ball)?;
    let y = slice1(&y)?;
    let kappa = slice1(&kappa)?;
    if breakpoints {
        robust_risk_breakpoints(intercept, &f, &b, y, u, kappa, eps, ball, kappa_y).map_err(err)
    } else {
        robust_risk_level(intercept, &f, &b, y, u, kappa, eps, ball, kappa_y).map_err(err)
    }
}

#[pyfunction]
fn v1_max_jump_bound(
    shapes: &Bound<'_, PyList>,
    kappa: PyReadonlyArray1<f64>,
    emp: f64,
    eps: f64,
    u: f64,
) -> PyResult<f64> {
    Ok(max_jump_bound(
        &read_shapes(shapes)?,
        slice1(&kappa)?,
        emp,
        eps,
        u,
    ))
}

#[pyfunction]
fn v1_outcome_w1_gap(eps: f64, u: f64, kappa_y: f64) -> f64 {
    outcome_only_w1_gap(eps, u, kappa_y)
}

#[pyfunction]
fn v1_winf(
    intercept: f64,
    shapes: &Bound<'_, PyList>,
    bins: PyReadonlyArray2<i64>,
    y: PyReadonlyArray1<f64>,
    u: f64,
    radius: usize,
    delta: f64,
) -> PyResult<(f64, f64)> {
    Ok(winf_risk(
        intercept,
        &read_shapes(shapes)?,
        &read_bins(&bins)?,
        slice1(&y)?,
        u,
        radius,
        delta,
    ))
}

#[pyfunction]
fn v1_pinball(r: f64, u: f64) -> f64 {
    pinball(r, u)
}

#[pyfunction]
fn v1_cross_fit_energy(a: PyReadonlyArray1<f64>, b: PyReadonlyArray1<f64>) -> PyResult<f64> {
    cross_fit_energy(slice1(&a)?, slice1(&b)?).map_err(err)
}

#[pyfunction]
fn v1_projection_pair<'py>(
    py: Python<'py>,
    q1: PyReadonlyArray1<f64>,
    q2: PyReadonlyArray1<f64>,
    w: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let p = projection_pair(slice1(&q1)?, slice1(&q2)?, slice1(&w)?).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("d", p.d)?;
    d.set_item("d_pi", p.d_pi)?;
    d.set_item("d_a", p.d_a)?;
    d.set_item("d_b", p.d_b)?;
    d.set_item("adjustment", p.adjustment)?;
    Ok(d)
}

#[pyfunction]
fn v1_lmoment_energy(
    y: PyReadonlyArray1<f64>,
    bins: PyReadonlyArray1<i64>,
    labels: PyReadonlyArray1<u8>,
    channel: usize,
    n_bins: usize,
) -> PyResult<f64> {
    let y = slice1(&y)?;
    let bins_a = bins
        .as_slice()
        .map_err(|_| PyValueError::new_err("bins must be C-contiguous"))?;
    let labels = labels
        .as_slice()
        .map_err(|_| PyValueError::new_err("labels must be C-contiguous"))?;
    let bins: Vec<usize> = bins_a.iter().map(|&v| v as usize).collect();
    lmoment_channel_energy(y, &bins, labels, channel, n_bins).map_err(err)
}

#[pyfunction]
fn v1_permutation_pvalue_lmoment(
    y: PyReadonlyArray1<f64>,
    bins: PyReadonlyArray1<i64>,
    labels: PyReadonlyArray1<u8>,
    channel: usize,
    n_bins: usize,
    reps: usize,
    seed: u64,
    stratified: bool,
) -> PyResult<f64> {
    let y = slice1(&y)?.to_vec();
    let bins_raw = bins
        .as_slice()
        .map_err(|_| PyValueError::new_err("bins must be C-contiguous"))?
        .to_vec();
    let mut labels = labels
        .as_slice()
        .map_err(|_| PyValueError::new_err("labels must be C-contiguous"))?
        .to_vec();
    let bins: Vec<usize> = bins_raw.iter().map(|&v| v as usize).collect();
    let strata = if stratified {
        Some(bins.as_slice())
    } else {
        None
    };
    let p = permutation_pvalue(
        |lab| lmoment_channel_energy(&y, &bins, lab, channel, n_bins).unwrap_or(0.0),
        &mut labels,
        strata,
        reps,
        seed,
    );
    Ok(p)
}

#[pyfunction]
fn v1_bootstrap_indices<'py>(
    py: Python<'py>,
    n_a: usize,
    n_b: usize,
    reps: usize,
    seed: u64,
) -> Bound<'py, PyArray1<i64>> {
    let idx = bootstrap_indices(n_a, n_b, reps, seed);
    Array1::from_vec(idx.into_iter().map(|v| v as i64).collect()).into_pyarray(py)
}

#[pyfunction]
fn v1_coupling_w1(
    bins_a: PyReadonlyArray2<i64>,
    bins_b: PyReadonlyArray2<i64>,
    kappa: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    Ok(coupling_w1(
        &read_bins(&bins_a)?,
        &read_bins(&bins_b)?,
        slice1(&kappa)?,
    ))
}

#[pyclass(name = "TraceHybridModel", module = "traceot._core")]
pub struct PyHybrid {
    inner: HybridModel,
}

#[pymethods]
impl PyHybrid {
    #[staticmethod]
    #[pyo3(signature = (
        x, y,
        k_fit=6, k_report=4, n_u=0, max_bins=16, max_rounds=40, tol=1e-8, min_count=1,
        ridge=1e-6, learning_rate=1.0, mono="none", bernstein_elevation=4, reporting="legendre",
        smooth=0.0, levenberg=0.0, sobolev_lambda=0.0, sobolev_s=1.0, tail_gamma=0.0,
        mean_max_bins=1024, mean_learning_rate=0.04, mean_max_rounds=50000,
        mean_early_stopping_rounds=100, mean_early_stopping_tolerance=1e-5,
        mean_validation_size=0.15, mean_min_samples_leaf=4
    ))]
    fn fit(
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
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
        smooth: f64,
        levenberg: f64,
        sobolev_lambda: f64,
        sobolev_s: f64,
        tail_gamma: f64,
        mean_max_bins: usize,
        mean_learning_rate: f64,
        mean_max_rounds: usize,
        mean_early_stopping_rounds: usize,
        mean_early_stopping_tolerance: f64,
        mean_validation_size: f64,
        mean_min_samples_leaf: u64,
    ) -> PyResult<Self> {
        let (n, p) = x.as_array().dim();
        let mut residual = config_from(
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
        residual.smooth = smooth;
        residual.levenberg = levenberg;
        residual.sobolev_lambda = sobolev_lambda;
        residual.sobolev_s = sobolev_s;
        residual.tail_gamma = tail_gamma;
        let mean = MeanConfig {
            max_bins: mean_max_bins,
            learning_rate: mean_learning_rate,
            max_rounds: mean_max_rounds,
            early_stopping_rounds: mean_early_stopping_rounds,
            early_stopping_tolerance: mean_early_stopping_tolerance,
            validation_size: mean_validation_size,
            min_samples_leaf: mean_min_samples_leaf,
        };
        let (xs, ys) = (slice2(&x)?, slice1(&y)?);
        let inner = py
            .detach(|| fit_hybrid(xs, n, p, ys, &mean, &residual))
            .map_err(err)?;
        Ok(Self { inner })
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        u: PyReadonlyArray1<'_, f64>,
        project: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (n, p) = x.as_array().dim();
        if p != self.inner.mean.edges.len() {
            return Err(PyValueError::new_err(
                "feature count does not match the fit",
            ));
        }
        let (xs, us) = (slice2(&x)?, slice1(&u)?);
        let flat = py
            .detach(|| predict_hybrid(&self.inner, xs, n, us, project))
            .map_err(err)?;
        let nu = us.len();
        Ok(Array2::from_shape_vec((n, nu), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_pyarray(py))
    }

    fn mean<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (n, p) = x.as_array().dim();
        if p != self.inner.mean.edges.len() {
            return Err(PyValueError::new_err(
                "feature count does not match the fit",
            ));
        }
        let xs = slice2(&x)?;
        let out = py
            .detach(|| predict_mean_channel(&self.inner, xs, n))
            .map_err(err)?;
        Ok(Array1::from_vec(out).into_pyarray(py))
    }

    fn channel_energy(&self, feature: usize, channel: usize) -> PyResult<f64> {
        hybrid_channel_energy(&self.inner, feature, channel).map_err(err)
    }

    #[getter]
    fn rounds(&self) -> usize {
        self.inner.residual.rounds
    }

    #[getter]
    fn mean_rounds(&self) -> usize {
        self.inner.mean.rounds
    }

    #[getter]
    fn u_nodes(&self) -> Vec<f64> {
        self.inner.residual.basis.nodes.clone()
    }

    #[getter]
    fn u_weights(&self) -> Vec<f64> {
        self.inner.residual.basis.weights.clone()
    }
}

#[pyfunction]
#[pyo3(signature = (
    x, y, max_bins=8, learning_rate=0.5, max_rounds=20, early_stopping_rounds=5,
    early_stopping_tolerance=1e-8, validation_size=0.0, min_samples_leaf=1
))]
fn v1_fit_mean<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    max_bins: usize,
    learning_rate: f64,
    max_rounds: usize,
    early_stopping_rounds: usize,
    early_stopping_tolerance: f64,
    validation_size: f64,
    min_samples_leaf: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let (n, p) = x.as_array().dim();
    let cfg = MeanConfig {
        max_bins,
        learning_rate,
        max_rounds,
        early_stopping_rounds,
        early_stopping_tolerance,
        validation_size,
        min_samples_leaf,
    };
    let (xs, ys) = (slice2(&x)?, slice1(&y)?);
    let model = py.detach(|| fit_mean(xs, n, p, ys, &cfg)).map_err(err)?;
    let pred = py.detach(|| predict_mean(&model, xs, n)).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("intercept", model.intercept)?;
    d.set_item("rounds", model.rounds)?;
    d.set_item("best_round", model.best_round)?;
    d.set_item("train_mse", model.train_mse)?;
    d.set_item("prediction", Array1::from_vec(pred).into_pyarray(py))?;
    let effects: Vec<Vec<f64>> = model.effects;
    d.set_item("effects", effects)?;
    Ok(d)
}

#[pyfunction]
fn v1_pinball_gh<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray1<f64>,
    weights: PyReadonlyArray1<f64>,
    basis_e: PyReadonlyArray1<f64>,
    basis_de: PyReadonlyArray1<f64>,
    a: PyReadonlyArray1<f64>,
    y: f64,
    smooth: f64,
    tail_gamma: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (g, h) = pinball_gh(
        slice1(&nodes)?,
        slice1(&weights)?,
        slice1(&basis_e)?,
        slice1(&basis_de)?,
        slice1(&a)?,
        y,
        smooth,
        tail_gamma,
    );
    Ok((
        Array1::from_vec(g).into_pyarray(py),
        Array1::from_vec(h).into_pyarray(py),
    ))
}

#[pyfunction]
fn v1_leave_feature_gains<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray1<f64>,
    weights: PyReadonlyArray1<f64>,
    basis_e: PyReadonlyArray1<f64>,
    players: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    rows: PyReadonlyArray1<i64>,
    p: usize,
    k: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let idx: Vec<usize> = rows
        .as_slice()
        .map_err(|_| PyValueError::new_err("rows must be C-contiguous"))?
        .iter()
        .map(|&v| v as usize)
        .collect();
    let (gains, shape) = leave_feature_gains(
        slice1(&nodes)?,
        slice1(&weights)?,
        slice1(&basis_e)?,
        slice1(&players)?,
        slice1(&y)?,
        &idx,
        p,
        k,
    )
    .map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("gains", gains)?;
    d.set_item("shape_gains", shape)?;
    Ok(d)
}

#[pyfunction]
fn v1_crps_quadrature(
    y: PyReadonlyArray1<f64>,
    q: PyReadonlyArray2<f64>,
    u: PyReadonlyArray1<f64>,
    w: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let ys = slice1(&y)?;
    let qs = slice2(&q)?;
    let us = slice1(&u)?;
    let ws = slice1(&w)?;
    crps_quadrature(ys, qs, us, ws).map_err(err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyV1Model>()?;
    m.add_class::<PyFrechet>()?;
    m.add_class::<PyRobust>()?;
    m.add_class::<PyHybrid>()?;
    m.add_function(wrap_pyfunction!(v1_gauss_legendre, m)?)?;
    m.add_function(wrap_pyfunction!(v1_hosking, m)?)?;
    m.add_function(wrap_pyfunction!(v1_weighted_pav, m)?)?;
    m.add_function(wrap_pyfunction!(v1_eval_a, m)?)?;
    m.add_function(wrap_pyfunction!(v1_report_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(v1_robust_risk, m)?)?;
    m.add_function(wrap_pyfunction!(v1_max_jump_bound, m)?)?;
    m.add_function(wrap_pyfunction!(v1_outcome_w1_gap, m)?)?;
    m.add_function(wrap_pyfunction!(v1_winf, m)?)?;
    m.add_function(wrap_pyfunction!(v1_pinball, m)?)?;
    m.add_function(wrap_pyfunction!(v1_cross_fit_energy, m)?)?;
    m.add_function(wrap_pyfunction!(v1_projection_pair, m)?)?;
    m.add_function(wrap_pyfunction!(v1_lmoment_energy, m)?)?;
    m.add_function(wrap_pyfunction!(v1_permutation_pvalue_lmoment, m)?)?;
    m.add_function(wrap_pyfunction!(v1_bootstrap_indices, m)?)?;
    m.add_function(wrap_pyfunction!(v1_coupling_w1, m)?)?;
    m.add_function(wrap_pyfunction!(v1_fit_mean, m)?)?;
    m.add_function(wrap_pyfunction!(v1_crps_quadrature, m)?)?;
    m.add_function(wrap_pyfunction!(v1_pinball_gh, m)?)?;
    m.add_function(wrap_pyfunction!(v1_leave_feature_gains, m)?)?;
    Ok(())
}
