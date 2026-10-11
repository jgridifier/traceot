//! PyO3 surface for TRACE. Numerics stay in `traceot_core::trace`.

use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use traceot_core::trace::{self, fit_codes_u16, fit_codes_u8, fit_trace, TraceConfig, TraceModel};
use traceot_core::TraceError;

fn err(e: TraceError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

type Arr1<'py> = Bound<'py, PyArray1<f64>>;
type Arr2<'py> = Bound<'py, PyArray2<f64>>;

fn slice1<'a>(a: &'a PyReadonlyArray1<'a, f64>) -> PyResult<&'a [f64]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err("array must be C-contiguous"))
}

fn slice2<'a>(a: &'a PyReadonlyArray2<'a, f64>) -> PyResult<&'a [f64]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err("array must be C-contiguous"))
}

#[allow(clippy::too_many_arguments)]
fn config_from(
    n_channels: usize,
    max_bins: usize,
    learning_rate: f64,
    max_rounds: usize,
    patience: usize,
    tol: f64,
    val_frac: f64,
    n_bags: usize,
    bag_frac: f64,
    gain_threshold: f64,
    select_frac: f64,
    max_interactions: usize,
    interaction_rounds: usize,
    screen_rows: usize,
    min_count: u64,
    u_levels: usize,
    seed: u64,
    min_rounds: usize,
) -> TraceConfig {
    TraceConfig {
        n_channels,
        max_bins,
        learning_rate,
        max_rounds,
        patience,
        tol,
        val_frac,
        n_bags,
        bag_frac,
        gain_threshold,
        select_frac,
        max_interactions,
        interaction_rounds,
        screen_rows,
        min_count,
        u_levels,
        seed,
        min_rounds,
    }
}

fn read_edges(list: &Bound<'_, PyList>) -> PyResult<Vec<Vec<f64>>> {
    let mut edges = Vec::with_capacity(list.len());
    for item in list.iter() {
        let arr = item.extract::<PyReadonlyArray1<f64>>()?;
        let xs = slice1(&arr)?;
        edges.push(xs.to_vec());
    }
    Ok(edges)
}

fn flat_to_2d<'py>(
    py: Python<'py>,
    flat: Vec<f64>,
    rows: usize,
    cols: usize,
) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_vec((rows, cols), flat)
        .expect("shape")
        .into_pyarray(py)
}

#[pyclass(name = "TraceModel", module = "traceot._core")]
pub struct PyTraceModel {
    inner: TraceModel,
}

#[pymethods]
impl PyTraceModel {
    /// Fit from a dense row-major ``X`` of shape ``(n, p)`` and a finite ``y``.
    #[staticmethod]
    #[pyo3(signature = (
        x, y, n_channels=3, max_bins=32, learning_rate=1.0, max_rounds=40, patience=8, tol=0.0,
        val_frac=0.2, n_bags=1, bag_frac=1.0, gain_threshold=0.0, select_frac=0.7,
        max_interactions=0, interaction_rounds=15, screen_rows=4096, min_count=1, u_levels=19, seed=0,
        min_rounds=20
    ))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        n_channels: usize,
        max_bins: usize,
        learning_rate: f64,
        max_rounds: usize,
        patience: usize,
        tol: f64,
        val_frac: f64,
        n_bags: usize,
        bag_frac: f64,
        gain_threshold: f64,
        select_frac: f64,
        max_interactions: usize,
        interaction_rounds: usize,
        screen_rows: usize,
        min_count: u64,
        u_levels: usize,
        seed: u64,
        min_rounds: usize,
    ) -> PyResult<Self> {
        let (n, p) = x.as_array().dim();
        let (xs, ys) = (slice2(&x)?, slice1(&y)?);
        let cfg = config_from(
            n_channels,
            max_bins,
            learning_rate,
            max_rounds,
            patience,
            tol,
            val_frac,
            n_bags,
            bag_frac,
            gain_threshold,
            select_frac,
            max_interactions,
            interaction_rounds,
            screen_rows,
            min_count,
            u_levels,
            seed,
            min_rounds,
        );
        let inner = py.detach(|| fit_trace(xs, n, p, ys, &cfg)).map_err(err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (
        codes, edges, y, n_channels=3, max_bins=32, learning_rate=1.0, max_rounds=40, patience=8,
        tol=0.0, val_frac=0.2, n_bags=1, bag_frac=1.0, gain_threshold=0.0, select_frac=0.7,
        max_interactions=0, interaction_rounds=15, screen_rows=4096, min_count=1, u_levels=19, seed=0,
        min_rounds=20
    ))]
    #[allow(clippy::too_many_arguments)]
    fn fit_codes_u8(
        py: Python<'_>,
        codes: numpy::PyReadonlyArray2<'_, u8>,
        edges: &Bound<'_, PyList>,
        y: PyReadonlyArray1<'_, f64>,
        n_channels: usize,
        max_bins: usize,
        learning_rate: f64,
        max_rounds: usize,
        patience: usize,
        tol: f64,
        val_frac: f64,
        n_bags: usize,
        bag_frac: f64,
        gain_threshold: f64,
        select_frac: f64,
        max_interactions: usize,
        interaction_rounds: usize,
        screen_rows: usize,
        min_count: u64,
        u_levels: usize,
        seed: u64,
        min_rounds: usize,
    ) -> PyResult<Self> {
        let (p, n) = codes.as_array().dim();
        let cs = codes
            .as_slice()
            .map_err(|_| PyValueError::new_err("codes must be C-contiguous"))?;
        let ys = slice1(&y)?;
        let edges = read_edges(edges)?;
        let cfg = config_from(
            n_channels,
            max_bins,
            learning_rate,
            max_rounds,
            patience,
            tol,
            val_frac,
            n_bags,
            bag_frac,
            gain_threshold,
            select_frac,
            max_interactions,
            interaction_rounds,
            screen_rows,
            min_count,
            u_levels,
            seed,
            min_rounds,
        );
        let inner = py
            .detach(|| fit_codes_u8(cs, n, p, edges, ys, &cfg))
            .map_err(err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (
        codes, edges, y, n_channels=3, max_bins=32, learning_rate=1.0, max_rounds=40, patience=8,
        tol=0.0, val_frac=0.2, n_bags=1, bag_frac=1.0, gain_threshold=0.0, select_frac=0.7,
        max_interactions=0, interaction_rounds=15, screen_rows=4096, min_count=1, u_levels=19, seed=0,
        min_rounds=20
    ))]
    #[allow(clippy::too_many_arguments)]
    fn fit_codes_u16(
        py: Python<'_>,
        codes: numpy::PyReadonlyArray2<'_, u16>,
        edges: &Bound<'_, PyList>,
        y: PyReadonlyArray1<'_, f64>,
        n_channels: usize,
        max_bins: usize,
        learning_rate: f64,
        max_rounds: usize,
        patience: usize,
        tol: f64,
        val_frac: f64,
        n_bags: usize,
        bag_frac: f64,
        gain_threshold: f64,
        select_frac: f64,
        max_interactions: usize,
        interaction_rounds: usize,
        screen_rows: usize,
        min_count: u64,
        u_levels: usize,
        seed: u64,
        min_rounds: usize,
    ) -> PyResult<Self> {
        let (p, n) = codes.as_array().dim();
        let cs = codes
            .as_slice()
            .map_err(|_| PyValueError::new_err("codes must be C-contiguous"))?;
        let ys = slice1(&y)?;
        let edges = read_edges(edges)?;
        let cfg = config_from(
            n_channels,
            max_bins,
            learning_rate,
            max_rounds,
            patience,
            tol,
            val_frac,
            n_bags,
            bag_frac,
            gain_threshold,
            select_frac,
            max_interactions,
            interaction_rounds,
            screen_rows,
            min_count,
            u_levels,
            seed,
            min_rounds,
        );
        let inner = py
            .detach(|| fit_codes_u16(cs, n, p, edges, ys, &cfg))
            .map_err(err)?;
        Ok(Self { inner })
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.predict_flag(py, x, true, false)
    }

    fn predict_raw<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.predict_flag(py, x, false, false)
    }

    #[pyo3(signature = (x, project=true))]
    fn predict_calibrated<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        project: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.predict_flag(py, x, project, true)
    }

    fn calibrate(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        let (n, _) = x.as_array().dim();
        let (xs, ys) = (slice2(&x)?, slice1(&y)?);
        py.detach(|| self.inner.calibrate(xs, n, ys)).map_err(err)
    }

    fn permutation_importance<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        repeats: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (n, _) = x.as_array().dim();
        let (xs, ys) = (slice2(&x)?, slice1(&y)?);
        let out = py
            .detach(|| self.inner.permutation_importance(xs, n, ys, repeats, seed))
            .map_err(err)?;
        Ok(Array1::from_vec(out).into_pyarray(py))
    }

    fn score<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (n, _) = x.as_array().dim();
        let (xs, ys) = (slice2(&x)?, slice1(&y)?);
        let s = py.detach(|| self.inner.score(xs, n, ys)).map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("pinball", s.pinball)?;
        d.set_item("crps", s.crps)?;
        d.set_item("cover80", s.cover80)?;
        d.set_item("cover90", s.cover90)?;
        d.set_item("mse_median", s.mse_median)?;
        d.set_item("mse_mean", s.mse_mean)?;
        Ok(d)
    }

    fn score_calibrated<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (n, _) = x.as_array().dim();
        let (xs, ys) = (slice2(&x)?, slice1(&y)?);
        let s = py
            .detach(|| self.inner.score_calibrated(xs, n, ys))
            .map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("pinball", s.pinball)?;
        d.set_item("crps", s.crps)?;
        d.set_item("cover80", s.cover80)?;
        d.set_item("cover90", s.cover90)?;
        d.set_item("mse_median", s.mse_median)?;
        d.set_item("mse_mean", s.mse_mean)?;
        Ok(d)
    }

    fn effect<'py>(&self, py: Python<'py>, j: usize) -> PyResult<(Arr2<'py>, Arr2<'py>)> {
        if j >= self.inner.coef.len() {
            return Err(PyValueError::new_err("feature index out of range"));
        }
        let m = self.inner.n_channels;
        let nb = self.inner.n_bins[j];
        Ok((
            flat_to_2d(py, self.inner.coef[j].clone(), nb, m),
            flat_to_2d(py, self.inner.std[j].clone(), nb, m),
        ))
    }

    fn edges<'py>(&self, py: Python<'py>, j: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if j >= self.inner.edges.len() {
            return Err(PyValueError::new_err("feature index out of range"));
        }
        Ok(Array1::from_vec(self.inner.edges[j].clone()).into_pyarray(py))
    }

    fn pairs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        let m = self.inner.n_channels;
        for p in &self.inner.pairs {
            let arr = Array3::from_shape_vec((p.n_bins_left, p.n_bins_right, m), p.coef.clone())
                .expect("shape")
                .into_pyarray(py);
            let d = PyDict::new(py);
            d.set_item("left", p.left)?;
            d.set_item("right", p.right)?;
            d.set_item("coef", arr)?;
            d.set_item("score", p.score)?;
            list.append(d)?;
        }
        Ok(list)
    }

    #[getter]
    fn u<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from_vec(self.inner.u.clone()).into_pyarray(py)
    }

    #[getter]
    fn basis<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let k = self.inner.u.len();
        flat_to_2d(py, self.inner.basis.clone(), k, self.inner.n_channels)
    }

    #[getter]
    fn q0<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from_vec(self.inner.q0.clone()).into_pyarray(py)
    }

    #[getter]
    fn order<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let v: Vec<i64> = self.inner.order.iter().map(|&j| j as i64).collect();
        Array1::from_vec(v).into_pyarray(py)
    }

    #[getter]
    fn selected<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        Array1::from_vec(self.inner.selected.clone()).into_pyarray(py)
    }

    #[getter]
    fn gains<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from_vec(self.inner.gains.clone()).into_pyarray(py)
    }

    #[getter]
    fn shape_gains<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from_vec(self.inner.shape_gains.clone()).into_pyarray(py)
    }

    #[getter]
    fn calibration<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from_vec(self.inner.calibration.clone()).into_pyarray(py)
    }

    #[getter]
    fn select_counts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let v: Vec<i64> = self.inner.select_counts.iter().map(|&c| c as i64).collect();
        Array1::from_vec(v).into_pyarray(py)
    }

    #[getter]
    fn n_bins<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let v: Vec<i64> = self.inner.n_bins.iter().map(|&c| c as i64).collect();
        Array1::from_vec(v).into_pyarray(py)
    }

    #[getter]
    fn train_loss(&self) -> f64 {
        self.inner.train_loss
    }

    #[getter]
    fn val_loss(&self) -> f64 {
        self.inner.val_loss
    }

    #[getter]
    fn bag_rounds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let v: Vec<i64> = self.inner.bag_rounds.iter().map(|&c| c as i64).collect();
        Array1::from_vec(v).into_pyarray(py)
    }

    #[getter]
    fn interaction_rounds(&self) -> usize {
        self.inner.interaction_rounds
    }

    #[getter]
    fn round_train<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from_vec(self.inner.round_train.clone()).into_pyarray(py)
    }

    #[getter]
    fn round_val<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from_vec(self.inner.round_val.clone()).into_pyarray(py)
    }

    #[getter]
    fn round_step<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from_vec(self.inner.round_step.clone()).into_pyarray(py)
    }

    #[getter]
    fn round_center_kept<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from_vec(self.inner.round_center_kept.clone()).into_pyarray(py)
    }

    #[getter]
    fn n_channels(&self) -> usize {
        self.inner.n_channels
    }

    fn predict_flag<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        project: bool,
        calibrated: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (n, _) = x.as_array().dim();
        let xs = slice2(&x)?;
        let q = py
            .detach(|| {
                if calibrated {
                    self.inner.predict_calibrated(xs, n, project)
                } else {
                    self.inner.predict(xs, n, project)
                }
            })
            .map_err(err)?;
        Ok(flat_to_2d(py, q, n, self.inner.u.len()))
    }
}

#[pyfunction]
fn trace_basis<'py>(
    py: Python<'py>,
    n_levels: usize,
    n_channels: usize,
) -> PyResult<(Arr1<'py>, Arr2<'py>)> {
    let (u, basis) = trace::quantile_basis(n_levels, n_channels).map_err(err)?;
    let k = u.len();
    Ok((
        Array1::from_vec(u).into_pyarray(py),
        flat_to_2d(py, basis, k, n_channels),
    ))
}

#[pyfunction]
fn trace_column_order<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let (n, p) = x.as_array().dim();
    let xs = slice2(&x)?;
    let order = trace::column_order(xs, n, p);
    let v: Vec<i64> = order.into_iter().map(|j| j as i64).collect();
    Ok(Array1::from_vec(v).into_pyarray(py))
}

#[pyfunction]
fn trace_empirical_quantile<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'_, f64>,
    u: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let q = trace::empirical_quantile(slice1(&y)?, slice1(&u)?).map_err(err)?;
    Ok(Array1::from_vec(q).into_pyarray(py))
}

#[pyfunction]
fn trace_conformal_quantile(y: PyReadonlyArray1<'_, f64>, u: f64) -> PyResult<f64> {
    trace::conformal_quantile(slice1(&y)?, u).map_err(err)
}

#[pyfunction]
fn trace_pinball_grad<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'_, f64>,
    theta: PyReadonlyArray2<'_, f64>,
    q0: PyReadonlyArray1<'_, f64>,
    basis: PyReadonlyArray2<'_, f64>,
    u: PyReadonlyArray1<'_, f64>,
) -> PyResult<(Arr2<'py>, Arr1<'py>)> {
    let (n, m_ch) = theta.as_array().dim();
    let (k, m2) = basis.as_array().dim();
    if m_ch != m2 {
        return Err(PyValueError::new_err(
            "theta and basis disagree on channels",
        ));
    }
    let (ys, th, q, b, us) = (
        slice1(&y)?,
        slice2(&theta)?,
        slice1(&q0)?,
        slice2(&basis)?,
        slice1(&u)?,
    );
    let (g, loss) = py
        .detach(|| traceot_core::scalable::pinball::pinball_grad_channels(ys, th, q, b, us, m_ch))
        .map_err(err)?;
    let _ = k;
    Ok((
        flat_to_2d(py, g, n, m_ch),
        Array1::from_vec(loss).into_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (x, y, threshold=0.9, max_shift=1))]
fn target_proxy_screen<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    threshold: f64,
    max_shift: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyList>)> {
    let (n, p) = x.as_array().dim();
    let screen = trace::target_proxy_screen(slice2(&x)?, n, p, slice1(&y)?, threshold, max_shift)
        .map_err(err)?;
    let flags = PyList::empty(py);
    for f in screen.flags {
        flags.append((f.feature, f.shift, f.corr))?;
    }
    Ok((
        Array1::from_vec(screen.contemporaneous).into_pyarray(py),
        flags,
    ))
}

#[pyfunction]
#[pyo3(signature = (x, near=0.999))]
fn duplicate_screen<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'_, f64>,
    near: f64,
) -> PyResult<Bound<'py, PyList>> {
    let (n, p) = x.as_array().dim();
    let flags = trace::duplicate_screen(slice2(&x)?, n, p, near).map_err(err)?;
    let list = PyList::empty(py);
    for f in flags {
        list.append((f.left, f.right, f.exact, f.corr))?;
    }
    Ok(list)
}

#[pyfunction]
#[pyo3(signature = (x, y, n_perm=20, seed=0))]
fn placebo_screen<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    n_perm: usize,
    seed: u64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64, f64)> {
    let (n, p) = x.as_array().dim();
    let (xs, ys) = (slice2(&x)?, slice1(&y)?);
    let s = py
        .detach(|| trace::placebo_screen(xs, n, p, ys, n_perm, seed))
        .map_err(err)?;
    Ok((
        Array1::from_vec(s.real_abs_corr).into_pyarray(py),
        s.placebo_max_abs_corr,
        s.placebo_mean_max_abs_corr,
    ))
}

#[pyfunction]
#[pyo3(signature = (a, b, grid=128))]
fn feature_w2<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f64>,
    b: PyReadonlyArray2<'_, f64>,
    grid: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (na, p) = a.as_array().dim();
    let (nb, p2) = b.as_array().dim();
    if p != p2 {
        return Err(PyValueError::new_err("feature counts differ"));
    }
    let w = trace::feature_w2(slice2(&a)?, na, slice2(&b)?, nb, p, grid).map_err(err)?;
    Ok(Array1::from_vec(w).into_pyarray(py))
}

#[pyfunction]
fn conformal_pvalues<'py>(
    py: Python<'py>,
    cal: PyReadonlyArray1<'_, f64>,
    test: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = trace::conformal_pvalues(slice1(&cal)?, slice1(&test)?).map_err(err)?;
    Ok(Array1::from_vec(p).into_pyarray(py))
}

#[pyfunction]
fn pinball_scores<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'_, f64>,
    q: PyReadonlyArray2<'_, f64>,
    u: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let s = trace::pinball_scores(slice1(&y)?, slice2(&q)?, slice1(&u)?).map_err(err)?;
    Ok(Array1::from_vec(s).into_pyarray(py))
}

#[pyfunction]
fn score_quantiles<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'_, f64>,
    q: PyReadonlyArray2<'_, f64>,
    u: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let s = trace::score_quantiles(slice1(&y)?, slice2(&q)?, slice1(&u)?).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("pinball", s.pinball)?;
    d.set_item("crps", s.crps)?;
    d.set_item("cover80", s.cover80)?;
    d.set_item("cover90", s.cover90)?;
    d.set_item("mse_median", s.mse_median)?;
    d.set_item("mse_mean", s.mse_mean)?;
    Ok(d)
}

#[pyfunction]
fn projection_blur(
    q00: PyReadonlyArray1<'_, f64>,
    q01: PyReadonlyArray1<'_, f64>,
    q10: PyReadonlyArray1<'_, f64>,
    q11: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, f64)> {
    trace::projection_blur(slice1(&q00)?, slice1(&q01)?, slice1(&q10)?, slice1(&q11)?).map_err(err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTraceModel>()?;
    m.add_function(wrap_pyfunction!(trace_basis, m)?)?;
    m.add_function(wrap_pyfunction!(trace_column_order, m)?)?;
    m.add_function(wrap_pyfunction!(trace_empirical_quantile, m)?)?;
    m.add_function(wrap_pyfunction!(trace_conformal_quantile, m)?)?;
    m.add_function(wrap_pyfunction!(trace_pinball_grad, m)?)?;
    m.add_function(wrap_pyfunction!(target_proxy_screen, m)?)?;
    m.add_function(wrap_pyfunction!(duplicate_screen, m)?)?;
    m.add_function(wrap_pyfunction!(placebo_screen, m)?)?;
    m.add_function(wrap_pyfunction!(feature_w2, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_pvalues, m)?)?;
    m.add_function(wrap_pyfunction!(pinball_scores, m)?)?;
    m.add_function(wrap_pyfunction!(score_quantiles, m)?)?;
    m.add_function(wrap_pyfunction!(projection_blur, m)?)?;
    Ok(())
}
