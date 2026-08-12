use inla_core::MathError;
use inla_core::ar1_precision;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Map [`MathError`] into a Python exception.
///
/// Singular / not-PD / not-symmetric become `numpy.linalg.LinAlgError` when
/// NumPy is importable, otherwise `scipy.linalg.LinAlgError`, else `ValueError`.
fn math_error_to_py(py: Python<'_>, err: MathError) -> PyErr {
    if err.is_linalg() {
        let msg = err.to_string();
        if let Ok(np) = py.import("numpy")
            && let Ok(linalg) = np.getattr("linalg")
            && let Ok(exc) = linalg.getattr("LinAlgError")
        {
            return PyErr::from_value(exc.call1((msg,)).unwrap_or_else(|_| {
                PyValueError::new_err(err.to_string())
                    .into_bound_py_any(py)
                    .unwrap()
            }));
        }
        if let Ok(sp) = py.import("scipy")
            && let Ok(linalg) = sp.getattr("linalg")
            && let Ok(exc) = linalg.getattr("LinAlgError")
        {
            return PyErr::from_value(exc.call1((msg,)).unwrap_or_else(|_| {
                PyValueError::new_err(err.to_string())
                    .into_bound_py_any(py)
                    .unwrap()
            }));
        }
        return PyValueError::new_err(msg);
    }
    match err {
        MathError::OutOfMemory => PyRuntimeError::new_err(err.to_string()),
        other => PyValueError::new_err(other.to_string()),
    }
}

/// Map a string error that may originate from [`MathError::to_string`].
fn string_error_to_py(py: Python<'_>, err: String) -> PyErr {
    let lower = err.to_ascii_lowercase();
    if lower.contains("keyboard") || lower.contains("interrupt") || lower.contains("cancelled") {
        return pyo3::exceptions::PyKeyboardInterrupt::new_err(());
    }
    if lower.contains("not positive definite")
        || lower.contains("singular")
        || lower.contains("not symmetric")
        || lower.contains("numerically unstable in ldl")
    {
        return math_error_to_py(py, MathError::NotPositiveDefinite);
    }
    PyValueError::new_err(err)
}

/// Computes a 1D AR1 precision matrix and returns sparse triplets (i, j, x)
/// in 1-based format (compatible with R sparse matrices).
#[pyfunction]
#[pyo3(signature = (n, rho, tau=1.0))]
fn ar1_precision_matrix(
    n: usize,
    rho: f64,
    tau: f64,
) -> PyResult<(Vec<usize>, Vec<usize>, Vec<f64>)> {
    let q = ar1_precision(n, rho, tau).map_err(PyValueError::new_err)?;
    Ok((q.i, q.j, q.x))
}

/// A wrapper around `sprs::CsMat<f64>` (Compressed Sparse Column matrix)
/// that exposes raw pointers for zero-copy SciPy integration.
#[pyclass]
#[derive(Clone)]
pub struct PyCscMatrix {
    pub matrix: inla_core::sparse::CscMatrix,
}

#[pymethods]
impl PyCscMatrix {
    /// Construct a PyCscMatrix from 1-based COO triplets.
    #[new]
    #[pyo3(signature = (nrow, ncol, i, j, x))]
    fn new(nrow: usize, ncol: usize, i: Vec<usize>, j: Vec<usize>, x: Vec<f64>) -> PyResult<Self> {
        let matrix = inla_core::sparse::triplets_to_csc(nrow, ncol, &i, &j, &x)
            .map_err(PyValueError::new_err)?;
        Ok(PyCscMatrix { matrix })
    }

    /// Build from a `scipy.sparse` CSC/CSR/COO matrix (copied into Rust CSC).
    #[staticmethod]
    fn from_scipy(mat: &Bound<'_, PyAny>) -> PyResult<Self> {
        csc_from_python(mat).map(|matrix| PyCscMatrix { matrix })
    }

    /// The (rows, cols) dimensions of the matrix.
    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.matrix.rows(), self.matrix.cols())
    }

    /// Memory address of the indptr slice.
    #[getter]
    fn indptr_ptr(&self) -> usize {
        self.matrix.indptr().raw_storage().as_ptr() as usize
    }

    /// Number of elements in the indptr slice.
    #[getter]
    fn indptr_len(&self) -> usize {
        self.matrix.indptr().raw_storage().len()
    }

    /// Memory address of the indices slice.
    #[getter]
    fn indices_ptr(&self) -> usize {
        self.matrix.indices().as_ptr() as usize
    }

    /// Number of elements in the indices slice.
    #[getter]
    fn indices_len(&self) -> usize {
        self.matrix.indices().len()
    }

    /// Memory address of the data slice.
    #[getter]
    fn data_ptr(&self) -> usize {
        self.matrix.data().as_ptr() as usize
    }

    /// Number of elements in the data slice.
    #[getter]
    fn data_len(&self) -> usize {
        self.matrix.data().len()
    }

    /// Zero-copy conversion of the underlying Rust matrix to a scipy.sparse.csc_matrix.
    fn to_scipy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let scipy = py.import("scipy.sparse")?;
        let np = py.import("numpy")?;
        let ctypes = py.import("ctypes")?;

        // indptr: cast raw pointer and create a numpy array view
        let c_uint64 = ctypes.getattr("c_uint64")?;
        let c_uint64_ptr_t = ctypes.call_method1("POINTER", (c_uint64,))?;
        let cast_indptr = ctypes.call_method1("cast", (self.indptr_ptr(), &c_uint64_ptr_t))?;
        let indptr_arr = np
            .getattr("ctypeslib")?
            .call_method1("as_array", (cast_indptr, (self.indptr_len(),)))?;

        // indices: cast raw pointer and create a numpy array view
        let cast_indices = ctypes.call_method1("cast", (self.indices_ptr(), &c_uint64_ptr_t))?;
        let indices_arr = np
            .getattr("ctypeslib")?
            .call_method1("as_array", (cast_indices, (self.indices_len(),)))?;

        // data: cast raw pointer and create a numpy array view
        let c_double = ctypes.getattr("c_double")?;
        let c_double_ptr_t = ctypes.call_method1("POINTER", (c_double,))?;
        let cast_data = ctypes.call_method1("cast", (self.data_ptr(), &c_double_ptr_t))?;
        let data_arr = np
            .getattr("ctypeslib")?
            .call_method1("as_array", (cast_data, (self.data_len(),)))?;

        // Build the scipy.sparse.csc_matrix
        let shape = self.shape();
        let kwargs = PyDict::new(py);
        kwargs.set_item("shape", shape)?;
        let csc_matrix = scipy.call_method(
            "csc_matrix",
            ((data_arr, indices_arr, indptr_arr),),
            Some(&kwargs),
        )?;

        // Custom attribute to keep the Rust wrapper and its memory alive
        let py_self = Bound::new(py, self.clone())?;
        csc_matrix.setattr("_base_matrix", py_self)?;

        Ok(csc_matrix)
    }
}

/// Accept `PyCscMatrix` or any SciPy sparse matrix convertible to CSC.
fn csc_from_python(obj: &Bound<'_, PyAny>) -> PyResult<inla_core::sparse::CscMatrix> {
    if let Ok(py_mat) = obj.extract::<PyRef<'_, PyCscMatrix>>() {
        return Ok(py_mat.matrix.clone());
    }
    let scipy = obj.py().import("scipy.sparse")?;
    let csc = if obj.hasattr("tocsc")? {
        obj.call_method0("tocsc")?
    } else {
        scipy.call_method1("csc_matrix", (obj,))?
    };
    let coo = csc.call_method0("tocoo")?;
    let nrow: usize = csc.getattr("shape")?.get_item(0)?.extract()?;
    let ncol: usize = csc.getattr("shape")?.get_item(1)?.extract()?;
    let rows: Vec<usize> = coo.getattr("row")?.extract()?;
    let cols: Vec<usize> = coo.getattr("col")?.extract()?;
    let data: Vec<f64> = coo.getattr("data")?.extract()?;
    if rows.len() != cols.len() || rows.len() != data.len() {
        return Err(PyValueError::new_err(
            "scipy sparse row/col/data length mismatch",
        ));
    }
    let trips: Vec<(usize, usize, f64)> = rows
        .into_iter()
        .zip(cols)
        .zip(data)
        .map(|((r, c), v)| (r, c, v))
        .collect();
    Ok(inla_core::sparse_from_triplets(nrow, ncol, &trips))
}

/// A 1D density grid `(x, y)` (classic INLA marginal shape).
#[pyclass]
#[derive(Clone)]
pub struct PyMarginal1D {
    #[pyo3(get)]
    pub x: Vec<f64>,
    #[pyo3(get)]
    pub y: Vec<f64>,
}

#[pymethods]
impl PyMarginal1D {
    /// Quantiles for probabilities in (0, 1), e.g. `[0.025, 0.5, 0.975]`.
    fn quantiles(&self, probs: Vec<f64>) -> PyResult<Vec<f64>> {
        let m = inla_core::Marginal1D {
            x: self.x.clone(),
            y: self.y.clone(),
        };
        inla_core::marginal_quantiles(&m, &probs).map_err(PyValueError::new_err)
    }
}

fn to_py_marginal(m: &inla_core::Marginal1D) -> PyMarginal1D {
    PyMarginal1D {
        x: m.x.clone(),
        y: m.y.clone(),
    }
}

/// The result of an end-to-end INLA inference call.
#[pyclass]
pub struct PyInferenceResult {
    /// Mode of the hyperparameter posterior.
    #[pyo3(get)]
    pub mode: Vec<f64>,
    /// Hessian matrix of the negative log-posterior at the mode.
    #[pyo3(get)]
    pub hessian: Vec<f64>,
    /// Mean values of the latent field.
    #[pyo3(get)]
    pub latent_means: Vec<f64>,
    /// Variances of the latent field.
    #[pyo3(get)]
    pub latent_variances: Vec<f64>,
    /// Linear predictor means η = A x.
    #[pyo3(get)]
    pub predictor_means: Vec<f64>,
    /// Approximate linear predictor variances.
    #[pyo3(get)]
    pub predictor_variances: Vec<f64>,
    /// Log marginal likelihood computed via integration.
    #[pyo3(get)]
    pub marginal_log_lik: f64,
    /// Log marginal likelihood computed via Gaussian approximation.
    #[pyo3(get)]
    pub marginal_log_lik_gaussian: f64,
    /// Deviance Information Criterion (DIC).
    #[pyo3(get)]
    pub dic: f64,
    /// Posterior mean of the deviance.
    #[pyo3(get)]
    pub mean_deviance: f64,
    /// Effective number of parameters (pD).
    #[pyo3(get)]
    pub effective_params: f64,
    /// Watanabe-Akaike Information Criterion.
    #[pyo3(get)]
    pub waic: f64,
    /// Log pointwise predictive density used in WAIC.
    #[pyo3(get)]
    pub waic_lppd: f64,
    /// WAIC effective number of parameters (p_waic).
    #[pyo3(get)]
    pub waic_effective_params: f64,
    /// Conditional Predictive Ordinates (CPO) for outlier detection.
    /// Elements can be None if the CPO computation failed for that observation.
    #[pyo3(get)]
    pub cpo: Vec<Option<f64>>,
    /// Probability Integral Transforms (PIT) for calibration checking.
    /// Elements can be None if the PIT computation failed or is unsupported.
    #[pyo3(get)]
    pub pit: Vec<Option<f64>>,
    /// Number of observations for which CPO computation failed.
    #[pyo3(get)]
    pub cpo_n_failures: usize,
    /// Normalized integration weights.
    #[pyo3(get)]
    pub node_weights: Vec<f64>,
    /// Internal-scale hyperparameter 1D marginals.
    #[pyo3(get)]
    pub internal_marginals_hyperpar: Vec<PyMarginal1D>,
    /// Opt-in latent mixture marginals (may be empty).
    #[pyo3(get)]
    pub marginals_latent: Vec<PyMarginal1D>,
    /// Indices corresponding to `marginals_latent`.
    #[pyo3(get)]
    pub marginals_latent_indices: Vec<usize>,
    /// Opt-in predictor mixture marginals (may be empty).
    #[pyo3(get)]
    pub marginals_predictor: Vec<PyMarginal1D>,
    /// Indices corresponding to `marginals_predictor`.
    #[pyo3(get)]
    pub marginals_predictor_indices: Vec<usize>,
}

/// Build an AR1 precision matrix.
#[pyfunction]
#[pyo3(signature = (n, rho, tau=1.0))]
fn ar1_precision_matrix_csc(n: usize, rho: f64, tau: f64) -> PyResult<PyCscMatrix> {
    let csc = inla_core::sparse::ar1_precision_csc(n, rho, tau).map_err(PyValueError::new_err)?;
    Ok(PyCscMatrix { matrix: csc })
}

/// Build an RW1 precision matrix.
#[pyfunction]
#[pyo3(signature = (n, tau=1.0))]
fn rw1_precision_matrix(n: usize, tau: f64) -> PyResult<PyCscMatrix> {
    let csc = inla_core::latent_models::rw1_precision_csc(n, tau).map_err(PyValueError::new_err)?;
    Ok(PyCscMatrix { matrix: csc })
}

/// Build an RW2 precision matrix.
#[pyfunction]
#[pyo3(signature = (n, tau=1.0))]
fn rw2_precision_matrix(n: usize, tau: f64) -> PyResult<PyCscMatrix> {
    let csc = inla_core::latent_models::rw2_precision_csc(n, tau).map_err(PyValueError::new_err)?;
    Ok(PyCscMatrix { matrix: csc })
}

/// Build an IID precision matrix.
#[pyfunction]
#[pyo3(signature = (n, tau=1.0))]
fn iid_precision_matrix(n: usize, tau: f64) -> PyResult<PyCscMatrix> {
    let csc = inla_core::latent_models::iid_precision_csc(n, tau).map_err(PyValueError::new_err)?;
    Ok(PyCscMatrix { matrix: csc })
}

/// Run end-to-end INLA inference.
///
/// Parameters
/// ----------
/// initial_theta : list[float]
///     Starting points for hyperparameter optimization.
/// build_prior : callable
///     A Python function `theta -> PyCscMatrix` that constructs the prior precision matrix.
/// log_prior_density : callable
///     A Python function `theta -> float` evaluating the log-prior density.
/// obs : list[dict]
///     A list of dictionaries representing observations (e.g. `{"family": "gaussian", "y": 1.0, "precision": 2.0}`).
/// strategy : str, optional
///     Integration strategy: `"ccd"` or `"grid"` (default `"ccd"`).
/// step_or_f0 : float, optional
///     Integration step size or f0 design parameter (default 1.0).
#[pyfunction(name = "run_inla_inference")]
#[pyo3(signature = (initial_theta, build_prior, log_prior_density, obs, strategy="ccd", step_or_f0=1.0, n_points=201, latent_marginal_indices=None, predictor_marginal_indices=None, a=None, constraints_a=None, constraints_e=None, deterministic=false))]
fn run_inla_inference_py(
    py: Python<'_>,
    initial_theta: Vec<f64>,
    build_prior: PyObject,
    log_prior_density: PyObject,
    obs: Vec<Bound<'_, PyAny>>,
    strategy: &str,
    step_or_f0: f64,
    n_points: usize,
    latent_marginal_indices: Option<Vec<usize>>,
    predictor_marginal_indices: Option<Vec<usize>>,
    a: Option<Bound<'_, PyAny>>,
    constraints_a: Option<Vec<f64>>,
    constraints_e: Option<Vec<f64>>,
    deterministic: bool,
) -> PyResult<PyInferenceResult> {
    // 1. Parse Python observation list to Rust Obs structs
    let mut rust_obs = Vec::with_capacity(obs.len());
    for item in obs {
        rust_obs.push(parse_obs(&item)?);
    }

    let a_mat = match a {
        Some(obj) => Some(csc_from_python(&obj)?),
        None => None,
    };

    let constr_spec = match (constraints_a, constraints_e) {
        (None, None) => None,
        (Some(a_vec), Some(e_vec)) => {
            let k = e_vec.len();
            if k == 0 {
                return Err(PyValueError::new_err("constraints_e must be non-empty"));
            }
            if a_vec.len() % k != 0 {
                return Err(PyValueError::new_err(
                    "constraints_a length must be divisible by len(constraints_e)",
                ));
            }
            let n = a_vec.len() / k;
            let spec = inla_core::ConstraintSpec {
                n,
                k,
                a: a_vec,
                e: e_vec,
                method: Default::default(),
            };
            spec.validate().map_err(PyValueError::new_err)?;
            Some(spec)
        }
        _ => {
            return Err(PyValueError::new_err(
                "constraints_a and constraints_e must both be provided or both omitted",
            ));
        }
    };

    let py_err_store = std::sync::Arc::new(std::sync::Mutex::new(None::<PyErr>));
    let store1 = py_err_store.clone();
    let store2 = py_err_store.clone();
    let store3 = py_err_store.clone();

    // 2. Closure for build_prior calling back to Python
    let build_prior_closure = move |theta: &[f64]| -> Result<inla_core::sparse::CscMatrix, String> {
        Python::with_gil(|py| {
            let theta_py = theta.to_vec();
            let res = match build_prior.call1(py, (theta_py,)) {
                Ok(val) => val,
                Err(e) => {
                    let mut lock = store1.lock().unwrap();
                    if lock.is_none() {
                        let py_err = if e
                            .is_instance_of::<pyo3::exceptions::PyKeyboardInterrupt>(py)
                            || e.to_string().contains("KeyboardInterrupt")
                        {
                            pyo3::exceptions::PyKeyboardInterrupt::new_err(())
                        } else {
                            e
                        };
                        *lock = Some(py_err);
                    }
                    return Err("Python build_prior callback failed".to_string());
                }
            };
            let bound = res.bind(py);
            csc_from_python(bound).map_err(|e| e.to_string())
        })
    };

    // 3. Closure for log_prior_density calling back to Python
    let log_prior_density_closure = move |theta: &[f64]| -> f64 {
        Python::with_gil(|py| {
            let theta_py = theta.to_vec();
            let res = log_prior_density.call1(py, (theta_py,));
            match res {
                Ok(val) => val.extract::<f64>(py).unwrap_or(f64::NEG_INFINITY),
                Err(e) => {
                    let mut lock = store2.lock().unwrap();
                    if lock.is_none() {
                        let py_err = if e
                            .is_instance_of::<pyo3::exceptions::PyKeyboardInterrupt>(py)
                            || e.to_string().contains("KeyboardInterrupt")
                        {
                            pyo3::exceptions::PyKeyboardInterrupt::new_err(())
                        } else {
                            e
                        };
                        *lock = Some(py_err);
                    }
                    f64::NEG_INFINITY
                }
            }
        })
    };

    let opts = inla_core::MarginalOptions {
        n_points,
        latent_indices: latent_marginal_indices.unwrap_or_default(),
        predictor_indices: predictor_marginal_indices.unwrap_or_default(),
        ..Default::default()
    };
    let check_cancel = move || {
        Python::with_gil(|py| {
            if let Err(e) = py.check_signals() {
                let mut lock = store3.lock().unwrap();
                if lock.is_none() {
                    let py_err = if e.is_instance_of::<pyo3::exceptions::PyKeyboardInterrupt>(py)
                        || e.to_string().contains("KeyboardInterrupt")
                    {
                        pyo3::exceptions::PyKeyboardInterrupt::new_err(())
                    } else {
                        e
                    };
                    *lock = Some(py_err);
                }
                Err("interrupted".to_string())
            } else {
                Ok(())
            }
        })
    };

    // 4. Run the core solver (releasing GIL for Rayon parallel execution)
    let result = py.allow_threads(|| {
        inla_core::run_inla_inference_a_cancellable(
            &initial_theta,
            &build_prior_closure,
            &log_prior_density_closure,
            &rust_obs,
            a_mat.as_ref(),
            constr_spec.as_ref(),
            strategy,
            step_or_f0,
            &opts,
            deterministic,
            Some(&check_cancel),
        )
    });

    py.check_signals()?;
    if let Some(err) = py_err_store.lock().unwrap().take() {
        if err.to_string().contains("KeyboardInterrupt") {
            return Err(pyo3::exceptions::PyKeyboardInterrupt::new_err(()));
        }
        return Err(err);
    }

    let result = result.map_err(|msg| string_error_to_py(py, msg))?;

    // 5. Build and return the wrapped result
    Ok(inference_result_to_py(result))
}

fn inference_result_to_py(result: inla_core::InferenceResult) -> PyInferenceResult {
    PyInferenceResult {
        mode: result.mode,
        hessian: result.hessian,
        latent_means: result.latent_means,
        latent_variances: result.latent_variances,
        predictor_means: result.predictor_means,
        predictor_variances: result.predictor_variances,
        marginal_log_lik: result.marginal_log_lik,
        marginal_log_lik_gaussian: result.marginal_log_lik_gaussian,
        dic: result.dic,
        mean_deviance: result.mean_deviance,
        effective_params: result.effective_params,
        waic: result.waic,
        waic_lppd: result.waic_lppd,
        waic_effective_params: result.waic_effective_params,
        cpo: result.cpo,
        pit: result.pit,
        cpo_n_failures: result.cpo_n_failures,
        node_weights: result.node_weights,
        internal_marginals_hyperpar: result
            .internal_marginals_hyperpar
            .iter()
            .map(to_py_marginal)
            .collect(),
        marginals_latent: result.marginals_latent.iter().map(to_py_marginal).collect(),
        marginals_latent_indices: result.marginals_latent_indices,
        marginals_predictor: result
            .marginals_predictor
            .iter()
            .map(to_py_marginal)
            .collect(),
        marginals_predictor_indices: result.marginals_predictor_indices,
    }
}

/// Orthonormal (constant, row-trend, col-trend) constraint rows for an intrinsic lattice.
///
/// Returns `(a, e)` with `a` row-major of shape `(3, nrow * ncol)`.
#[pyfunction]
fn plane_constraint_2d(nrow: usize, ncol: usize) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let c = inla_core::plane_constraint_2d(nrow, ncol).map_err(PyValueError::new_err)?;
    Ok((c.a, c.e))
}

/// Orthonormal basis of the seasonal null space (`season - 1` rows).
#[pyfunction]
fn seasonal_constraint(n: usize, season: usize) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let c = inla_core::seasonal_constraint(n, season).map_err(PyValueError::new_err)?;
    Ok((c.a, c.e))
}

/// Per-model metadata: θ length, defaults, rank deficiency, hyper labels/transforms.
#[pyfunction]
#[pyo3(signature = (model, order=0, group_model=None, cyclic=false))]
fn model_metadata(
    py: Python<'_>,
    model: &str,
    order: usize,
    group_model: Option<&str>,
    cyclic: bool,
) -> PyResult<Py<PyDict>> {
    let meta = inla_core::model_metadata(model, order, group_model, cyclic)
        .map_err(PyValueError::new_err)?;
    let d = PyDict::new(py);
    d.set_item("model", meta.model)?;
    d.set_item("theta_len", meta.theta_len)?;
    d.set_item("default_theta", meta.default_theta)?;
    d.set_item("rank_deficiency", meta.rank_deficiency)?;
    d.set_item("default_scale_model", meta.default_scale_model)?;
    d.set_item(
        "hyper_internal",
        meta.hyper
            .iter()
            .map(|h| h.internal_label.clone())
            .collect::<Vec<String>>(),
    )?;
    d.set_item(
        "hyper_labels",
        meta.hyper
            .iter()
            .map(|h| h.label.clone())
            .collect::<Vec<String>>(),
    )?;
    d.set_item(
        "hyper_transforms",
        meta.hyper
            .iter()
            .map(|h| h.transform_tag().to_string())
            .collect::<Vec<String>>(),
    )?;
    d.set_item("default_priors", meta.default_priors)?;
    Ok(d.into())
}

/// Latent model names accepted by the structured path.
#[pyfunction]
fn supported_models() -> Vec<String> {
    inla_core::SUPPORTED_MODELS
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// Validate + fill defaults for a control dict. Unknown keys raise.
#[pyfunction]
fn resolve_compute_options(py: Python<'_>, controls: &Bound<'_, PyDict>) -> PyResult<Py<PyDict>> {
    let mut pairs: Vec<(String, inla_core::OptionValue)> = Vec::new();
    for (key, value) in controls.iter() {
        let name: String = key.extract()?;
        pairs.push((name.clone(), py_to_option_value(&name, &value)?));
    }
    let opts = inla_core::resolve_compute_options(&pairs).map_err(PyValueError::new_err)?;

    let d = PyDict::new(py);
    d.set_item("strategy", opts.strategy)?;
    d.set_item("step_or_f0", opts.step_or_f0)?;
    d.set_item("deterministic", opts.deterministic)?;
    d.set_item("fixed_prec", opts.fixed_prec)?;
    d.set_item("dic", opts.dic)?;
    d.set_item("waic", opts.waic)?;
    d.set_item("cpo", opts.cpo)?;
    d.set_item(
        "return_marginals_latent",
        selection_to_py(py, &opts.return_marginals_latent)?,
    )?;
    d.set_item(
        "return_marginals_predictor",
        selection_to_py(py, &opts.return_marginals_predictor)?,
    )?;
    Ok(d.into())
}

fn selection_to_py(py: Python<'_>, sel: &inla_core::IndexSelection) -> PyResult<PyObject> {
    match sel {
        inla_core::IndexSelection::None => false.into_py_any(py),
        inla_core::IndexSelection::All => true.into_py_any(py),
        inla_core::IndexSelection::Some(idx) => idx.clone().into_py_any(py),
    }
}

fn py_to_option_value(name: &str, value: &Bound<'_, PyAny>) -> PyResult<inla_core::OptionValue> {
    if value.is_instance_of::<pyo3::types::PyBool>() {
        return Ok(inla_core::OptionValue::Bool(value.extract()?));
    }
    if let Ok(s) = value.extract::<String>() {
        return Ok(inla_core::OptionValue::Text(s));
    }
    if let Ok(v) = value.extract::<f64>() {
        return Ok(inla_core::OptionValue::Num(v));
    }
    if let Ok(v) = value.extract::<Vec<f64>>() {
        return Ok(inla_core::OptionValue::Nums(v));
    }
    Err(PyValueError::new_err(format!(
        "control '{name}': unsupported type (use bool, number, string, or sequence of numbers)"
    )))
}

/// Build block-diagonal Q from structured effect metadata (shared with R).
#[pyfunction(name = "build_structured_precision")]
#[pyo3(signature = (effects, theta, fixed_prec=1e-4))]
fn build_structured_precision_py(
    effects: Vec<Bound<'_, PyDict>>,
    theta: Vec<f64>,
    fixed_prec: f64,
) -> PyResult<PyCscMatrix> {
    let parsed = parse_structured_effects(&effects)?;
    let csc = inla_core::build_structured_precision(&parsed, &theta, fixed_prec)
        .map_err(PyValueError::new_err)?;
    Ok(PyCscMatrix { matrix: csc })
}

/// Linear constraints for a structured effect list: `(a_rowmajor, e)` or `None`.
#[pyfunction(name = "structured_constraints")]
fn structured_constraints_py(
    effects: Vec<Bound<'_, PyDict>>,
) -> PyResult<Option<(Vec<f64>, Vec<f64>)>> {
    let parsed = parse_structured_effects(&effects)?;
    match inla_core::structured_constraints(&parsed).map_err(PyValueError::new_err)? {
        None => Ok(None),
        Some(c) => Ok(Some((c.a, c.e))),
    }
}

fn parse_structured_effects(
    effects: &[Bound<'_, PyDict>],
) -> PyResult<Vec<inla_core::StructuredEffect>> {
    let mut out = Vec::with_capacity(effects.len());
    for d in effects {
        let model: String = d
            .get_item("model")?
            .ok_or_else(|| PyValueError::new_err("effect missing model"))?
            .extract()?;
        let n: usize = d
            .get_item("n")?
            .ok_or_else(|| PyValueError::new_err("effect missing n"))?
            .extract()?;
        let theta_len: usize = d
            .get_item("theta_len")?
            .ok_or_else(|| PyValueError::new_err("effect missing theta_len"))?
            .extract()?;
        let scale_model: bool = d
            .get_item("scale_model")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(false);
        let order: i32 = d
            .get_item("order")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(0);
        let nrow: usize = d
            .get_item("nrow")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(0);
        let ncol: usize = d
            .get_item("ncol")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(0);
        let cyclic: bool = d
            .get_item("cyclic")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(false);
        let matern_nu: usize = d
            .get_item("matern_nu")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(1);
        let crw2_layout: String = d
            .get_item("crw2_layout")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or_else(|| "simple".to_string());
        let positions: Option<Vec<f64>> =
            d.get_item("positions")?.map(|v| v.extract()).transpose()?;
        let adj: Option<Vec<Vec<usize>>> = d.get_item("adj")?.map(|v| v.extract()).transpose()?;
        out.push(inla_core::StructuredEffect {
            model,
            n,
            scale_model,
            theta_len,
            order,
            adj,
            positions,
            crw2_layout,
            nrow,
            ncol,
            cyclic,
            matern_nu,
        });
    }
    Ok(out)
}

/// Gaussian + single AR(1), η = x, via [`inla_core::ModelSpec`] / [`resolve`].
#[pyfunction]
#[pyo3(signature = (
    y,
    name="time",
    obs_precision=100.0,
    strategy="ccd",
    step_or_f0=1.0,
    initial_theta=None,
))]
fn run_gaussian_ar1_plan(
    py: Python<'_>,
    y: Vec<f64>,
    name: &str,
    obs_precision: f64,
    strategy: &str,
    step_or_f0: f64,
    initial_theta: Option<Vec<f64>>,
) -> PyResult<PyInferenceResult> {
    let n = y.len();
    let spec = inla_core::ModelSpec {
        likelihood: inla_core::LikelihoodSpec::Gaussian {
            precision: Some(obs_precision),
        },
        effects: vec![inla_core::LatentEffectSpec::Ar1 {
            name: name.to_string(),
            n,
            priors: None,
        }],
        computation: inla_core::ComputationSpec {
            strategy: Some(strategy.to_string()),
            step_or_f0: Some(step_or_f0),
        },
        initial_theta,
    };
    let result = py.allow_threads(|| {
        let plan = inla_core::resolve(spec).map_err(|e| e.0)?;
        inla_core::run_gaussian_ar1_plan(&plan, &y).map_err(|e| e.0)
    });
    Ok(inference_result_to_py(
        result.map_err(|msg| string_error_to_py(py, msg))?,
    ))
}

/// Helper function to parse python dicts representing observations.
fn dict_get_f64(dict: &Bound<'_, PyAny>, key: &str) -> PyResult<f64> {
    dict.get_item(key)?
        .extract()
        .map_err(|_| PyValueError::new_err(format!("missing or invalid observation field '{key}'")))
}

fn parse_obs(dict: &Bound<'_, PyAny>) -> PyResult<inla_core::Obs> {
    if dict.is_none() {
        return Ok(inla_core::Obs::None);
    }

    let family_item = dict.get_item("family")?;
    if family_item.is_none() {
        return Ok(inla_core::Obs::None);
    }
    let family: String = family_item.extract()?;

    let link_str: Option<String> = match dict.get_item("link") {
        Ok(item) => {
            if item.is_none() {
                None
            } else {
                Some(item.extract()?)
            }
        }
        Err(_) => None,
    };

    let link = match link_str.as_deref() {
        Some("identity") => inla_core::Link::Identity,
        Some("log") => inla_core::Link::Log,
        Some("logit") => inla_core::Link::Logit,
        None | Some("default") | Some("") => match family.as_str() {
            "gaussian" | "laplace" => inla_core::Link::Identity,
            "poisson"
            | "nbinomial"
            | "negative_binomial"
            | "zero_inflated_poisson"
            | "zeroinflatedpoisson0"
            | "zeroinflatedpoisson1"
            | "exponential"
            | "exponential_survival"
            | "weibull"
            | "weibull_survival" => inla_core::Link::Log,
            "binomial"
            | "zero_inflated_binomial"
            | "zeroinflatedbinomial0"
            | "zeroinflatedbinomial1" => inla_core::Link::Logit,
            _ => inla_core::Link::Identity,
        },
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "unknown link function: {}",
                other
            )));
        }
    };

    match family.as_str() {
        "none" => Ok(inla_core::Obs::None),
        "gaussian" => {
            let y: f64 = dict.get_item("y")?.extract()?;
            let precision: f64 = dict.get_item("precision")?.extract()?;
            Ok(inla_core::Obs::Gaussian(inla_core::GaussianObs {
                y,
                precision,
                link,
            }))
        }
        "poisson" => {
            let y: f64 = dict.get_item("y")?.extract()?;
            let exposure: f64 = if let Ok(e) = dict.get_item("exposure") {
                if e.is_none() {
                    dict_get_f64(dict, "E")?
                } else {
                    e.extract()?
                }
            } else {
                dict_get_f64(dict, "E")?
            };
            Ok(inla_core::Obs::Poisson(inla_core::PoissonObs {
                y,
                exposure,
                link,
            }))
        }
        "binomial" => {
            let y: f64 = dict.get_item("y")?.extract()?;
            let n: f64 = dict.get_item("n")?.extract()?;
            Ok(inla_core::Obs::Binomial(inla_core::BinomialObs {
                y,
                n,
                link,
            }))
        }
        "negative_binomial" | "nbinomial" => {
            let y: f64 = dict.get_item("y")?.extract()?;
            let exposure: f64 = dict.get_item("exposure")?.extract()?;
            let size: f64 = dict.get_item("size")?.extract()?;
            Ok(inla_core::Obs::NegativeBinomial(
                inla_core::NegativeBinomialObs {
                    y,
                    exposure,
                    size,
                    link,
                },
            ))
        }
        "zero_inflated_poisson" | "zeroinflatedpoisson0" | "zeroinflatedpoisson1" => {
            let y: f64 = dict.get_item("y")?.extract()?;
            let exposure: f64 = dict.get_item("exposure")?.extract()?;
            let zero_prob: f64 = dict.get_item("zero_prob")?.extract()?;
            let inflation_str: Option<String> = match dict.get_item("inflation") {
                Ok(item) => {
                    if item.is_none() {
                        None
                    } else {
                        Some(item.extract()?)
                    }
                }
                Err(_) => None,
            };
            let inflation = if family == "zeroinflatedpoisson1" {
                inla_core::ZeroInflationType::Type1
            } else {
                match inflation_str.as_deref() {
                    Some("type1") => inla_core::ZeroInflationType::Type1,
                    _ => inla_core::ZeroInflationType::Type0,
                }
            };
            Ok(inla_core::Obs::ZeroInflatedPoisson(
                inla_core::ZeroInflatedPoissonObs {
                    y,
                    exposure,
                    zero_prob,
                    link,
                    inflation,
                },
            ))
        }
        "zero_inflated_binomial" | "zeroinflatedbinomial0" | "zeroinflatedbinomial1" => {
            let y: f64 = dict.get_item("y")?.extract()?;
            let n: f64 = dict.get_item("n")?.extract()?;
            let zero_prob: f64 = dict.get_item("zero_prob")?.extract()?;
            let inflation_str: Option<String> = match dict.get_item("inflation") {
                Ok(item) => {
                    if item.is_none() {
                        None
                    } else {
                        Some(item.extract()?)
                    }
                }
                Err(_) => None,
            };
            let inflation = if family == "zeroinflatedbinomial1" {
                inla_core::ZeroInflationType::Type1
            } else {
                match inflation_str.as_deref() {
                    Some("type1") => inla_core::ZeroInflationType::Type1,
                    _ => inla_core::ZeroInflationType::Type0,
                }
            };
            Ok(inla_core::Obs::ZeroInflatedBinomial(
                inla_core::ZeroInflatedBinomialObs {
                    y,
                    n,
                    zero_prob,
                    link,
                    inflation,
                },
            ))
        }
        "laplace" => {
            let y: f64 = dict.get_item("y")?.extract()?;
            let alpha: f64 = dict.get_item("alpha")?.extract()?;
            let gamma: f64 = dict.get_item("gamma")?.extract()?;
            Ok(inla_core::Obs::Laplace(inla_core::LaplaceObs {
                y,
                alpha,
                gamma,
                link,
            }))
        }
        "exponential_survival" | "exponential" => {
            let y: f64 = dict.get_item("y")?.extract()?;
            let event: f64 = dict.get_item("event")?.extract()?;
            Ok(inla_core::Obs::ExponentialSurvival(
                inla_core::ExponentialSurvivalObs { y, event, link },
            ))
        }
        "weibull_survival" | "weibull" => {
            let y: f64 = dict.get_item("y")?.extract()?;
            let event: f64 = dict.get_item("event")?.extract()?;
            let shape: f64 = dict.get_item("shape")?.extract()?;
            Ok(inla_core::Obs::WeibullSurvival(
                inla_core::WeibullSurvivalObs {
                    y,
                    event,
                    shape,
                    link,
                },
            ))
        }
        _ => Err(PyValueError::new_err(format!(
            "unknown observation family: {}",
            family
        ))),
    }
}

/// Build a Besag/ICAR precision matrix from an adjacency list (0-based neighbors).
///
/// `adj` is a list of length `n`; `adj[i]` lists neighbors of node `i`.
#[pyfunction]
#[pyo3(signature = (adj, tau=1.0))]
fn besag_precision_matrix(adj: Vec<Vec<usize>>, tau: f64) -> PyResult<PyCscMatrix> {
    let csc = inla_core::besag_precision_csc(&adj, tau).map_err(PyValueError::new_err)?;
    Ok(PyCscMatrix { matrix: csc })
}

/// Classic BYM precision (2n): spatial ICAR ⊕ IID.
#[pyfunction]
#[pyo3(signature = (adj, tau_spatial=1.0, tau_iid=1.0))]
fn bym_precision_matrix(
    adj: Vec<Vec<usize>>,
    tau_spatial: f64,
    tau_iid: f64,
) -> PyResult<PyCscMatrix> {
    let csc =
        inla_core::bym_precision_csc(&adj, tau_spatial, tau_iid).map_err(PyValueError::new_err)?;
    Ok(PyCscMatrix { matrix: csc })
}

/// BYM2 precision (length n): `τ[(1-φ)I + φ Q★]`.
#[pyfunction]
#[pyo3(signature = (adj, tau=1.0, phi=0.5))]
fn bym2_precision_matrix(adj: Vec<Vec<usize>>, tau: f64, phi: f64) -> PyResult<PyCscMatrix> {
    let csc = inla_core::bym2_precision_csc(&adj, tau, phi).map_err(PyValueError::new_err)?;
    Ok(PyCscMatrix { matrix: csc })
}

/// Build an FGN (exact dense) precision matrix.
#[pyfunction]
#[pyo3(signature = (n, hurst, tau=1.0))]
fn fgn_precision_matrix(n: usize, hurst: f64, tau: f64) -> PyResult<PyCscMatrix> {
    let csc = inla_core::fgn_precision_csc(n, hurst, tau).map_err(PyValueError::new_err)?;
    Ok(PyCscMatrix { matrix: csc })
}

/// Build an R-INLA AR-mixture FGN approximation precision matrix (sparse).
///
/// `order` must be 3 or 4. Coefficients are interpolated from the legacy
/// `FGN_K3_PARAM` / `FGN_K4_PARAM` tables by Hurst parameter `H`.
///
/// `prec_eps` (default `1e8`) is the soft-constraint precision on
/// `z ≈ Σ x_i` in the AR mixture latent field, matching R-INLA's FGN
/// conditioning strength. Leave it large unless you intentionally weaken
/// that constraint; values ≪ `1e6` change the approximation materially.
#[pyfunction]
#[pyo3(signature = (n, hurst, tau=1.0, order=4, prec_eps=1e8))]
fn fgn_approx_precision_matrix(
    n: usize,
    hurst: f64,
    tau: f64,
    order: usize,
    prec_eps: f64,
) -> PyResult<PyCscMatrix> {
    let csc = inla_core::fgn_approx_precision_csc(n, hurst, tau, order, prec_eps)
        .map_err(PyValueError::new_err)?;
    Ok(PyCscMatrix { matrix: csc })
}

/// Map FGN internal hyperparameter → Hurst H ∈ (1/2, 1) (R-INLA `from.theta`).
#[pyfunction]
fn fgn_hurst_from_intern(h_intern: f64) -> f64 {
    inla_core::fgn_hurst_from_intern(h_intern)
}

/// Map Hurst H ∈ (1/2, 1) → FGN internal hyperparameter.
#[pyfunction]
fn fgn_intern_from_hurst(h: f64) -> PyResult<f64> {
    inla_core::fgn_intern_from_hurst(h).map_err(PyValueError::new_err)
}

/// Latent dimension for the AR-mixture FGN approx: `(order + 1) * n_obs`.
#[pyfunction]
fn fgn_approx_latent_len(n_obs: usize, order: usize) -> usize {
    inla_core::fgn_approx_latent_len(n_obs, order)
}

/// Build a seasonal precision matrix.
#[pyfunction]
#[pyo3(signature = (n, season=4, tau=1.0, cyclic=true))]
fn seasonal_precision_matrix(
    n: usize,
    season: usize,
    tau: f64,
    cyclic: bool,
) -> PyResult<PyCscMatrix> {
    let csc =
        inla_core::seasonal_precision_csc(n, season, tau, cyclic).map_err(PyValueError::new_err)?;
    Ok(PyCscMatrix { matrix: csc })
}

/// Build an AR(p) precision matrix from PACF values.
#[pyfunction]
#[pyo3(signature = (n, pacf, tau=1.0))]
fn arp_precision_matrix(n: usize, pacf: Vec<f64>, tau: f64) -> PyResult<PyCscMatrix> {
    let csc = inla_core::arp_precision_csc(n, &pacf, tau).map_err(PyValueError::new_err)?;
    Ok(PyCscMatrix { matrix: csc })
}

/// Build a CRW1 precision matrix from positions.
#[pyfunction]
#[pyo3(signature = (positions, tau=1.0))]
fn crw1_precision_matrix(positions: Vec<f64>, tau: f64) -> PyResult<PyCscMatrix> {
    let csc = inla_core::crw1_precision_csc(&positions, tau).map_err(PyValueError::new_err)?;
    Ok(PyCscMatrix { matrix: csc })
}

/// Matérn-on-lattice precision (`nrow * ncol` nodes).
#[pyfunction]
#[pyo3(signature = (nrow, ncol, nu=1, range=1.0, prec=1.0, cyclic=false))]
fn matern2d_precision_matrix(
    nrow: usize,
    ncol: usize,
    nu: usize,
    range: f64,
    prec: f64,
    cyclic: bool,
) -> PyResult<PyCscMatrix> {
    let csc = inla_core::matern2d_precision_csc(nrow, ncol, nu, range, prec, cyclic)
        .map_err(PyValueError::new_err)?;
    Ok(PyCscMatrix { matrix: csc })
}

/// 2D intrinsic random walk precision on an `nrow × ncol` lattice.
#[pyfunction]
#[pyo3(signature = (nrow, ncol, tau=1.0, cyclic=false, bvalue_zero=false))]
fn rw2d_precision_matrix(
    nrow: usize,
    ncol: usize,
    tau: f64,
    cyclic: bool,
    bvalue_zero: bool,
) -> PyResult<PyCscMatrix> {
    let csc = inla_core::rw2d_precision_csc(nrow, ncol, tau, cyclic, bvalue_zero)
        .map_err(PyValueError::new_err)?;
    Ok(PyCscMatrix { matrix: csc })
}

/// Sparse Kronecker product `A ⊗ B`.
#[pyfunction]
fn kronecker_csc(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyCscMatrix> {
    let a_m = csc_from_python(a)?;
    let b_m = csc_from_python(b)?;
    Ok(PyCscMatrix {
        matrix: inla_core::kronecker_csc(&a_m, &b_m),
    })
}

/// SPDE precision from a triangular mesh (`vertices` Nx2, `triangles` Mx3, 0-based).
#[pyfunction]
#[pyo3(signature = (vertices, triangles, kappa, tau=1.0))]
fn spde_precision_matrix(
    vertices: Vec<(f64, f64)>,
    triangles: Vec<(usize, usize, usize)>,
    kappa: f64,
    tau: f64,
) -> PyResult<PyCscMatrix> {
    let verts: Vec<inla_core::fmesher::Vertex2> = vertices
        .into_iter()
        .map(|(x, y)| inla_core::fmesher::Vertex2 { x, y })
        .collect();
    let tris: Vec<inla_core::fmesher::Triangle> = triangles
        .into_iter()
        .map(|(a, b, c)| inla_core::fmesher::Triangle([a, b, c]))
        .collect();
    let mesh = inla_core::fmesher::build_mesh2d(verts, tris).map_err(PyValueError::new_err)?;
    let fem = mesh.assemble_fem_blocks();
    let csc = inla_core::spde_precision_csc(&fem, kappa, tau).map_err(PyValueError::new_err)?;
    Ok(PyCscMatrix { matrix: csc })
}

/// Piecewise-linear SPDE projector A (`n_obs × n_vertices`).
#[pyfunction]
#[pyo3(signature = (vertices, triangles, loc_x, loc_y))]
fn spde_projector_matrix(
    vertices: Vec<(f64, f64)>,
    triangles: Vec<(usize, usize, usize)>,
    loc_x: Vec<f64>,
    loc_y: Vec<f64>,
) -> PyResult<PyCscMatrix> {
    let verts: Vec<inla_core::fmesher::Vertex2> = vertices
        .into_iter()
        .map(|(x, y)| inla_core::fmesher::Vertex2 { x, y })
        .collect();
    let tris: Vec<inla_core::fmesher::Triangle> = triangles
        .into_iter()
        .map(|(a, b, c)| inla_core::fmesher::Triangle([a, b, c]))
        .collect();
    let mesh = inla_core::fmesher::build_mesh2d(verts, tris).map_err(PyValueError::new_err)?;
    let csc =
        inla_core::spde_projector_from_xy(&mesh, &loc_x, &loc_y).map_err(PyValueError::new_err)?;
    Ok(PyCscMatrix { matrix: csc })
}

/// FEM mass (`c0` / C) and stiffness (`g1` / G) for a triangular mesh.
///
/// Analogous to classic INLA `spde$param.inla$M0` / `M1`.
#[pyfunction]
fn fem_blocks_mesh(
    py: Python<'_>,
    vertices: Vec<(f64, f64)>,
    triangles: Vec<(usize, usize, usize)>,
) -> PyResult<Bound<'_, PyDict>> {
    let verts: Vec<inla_core::fmesher::Vertex2> = vertices
        .into_iter()
        .map(|(x, y)| inla_core::fmesher::Vertex2 { x, y })
        .collect();
    let tris: Vec<inla_core::fmesher::Triangle> = triangles
        .into_iter()
        .map(|(a, b, c)| inla_core::fmesher::Triangle([a, b, c]))
        .collect();
    let mesh = inla_core::fmesher::build_mesh2d(verts, tris).map_err(PyValueError::new_err)?;
    let fem = mesh.assemble_fem_blocks();
    let c0 = inla_core::sparse_from_triplets(fem.c0.rows, fem.c0.cols, &fem.c0.entries);
    let g1 = inla_core::sparse_from_triplets(fem.g1.rows, fem.g1.cols, &fem.g1.entries);
    let out = PyDict::new(py);
    out.set_item("c0", PyCscMatrix { matrix: c0 })?;
    out.set_item("g1", PyCscMatrix { matrix: g1 })?;
    out.set_item("n_vertices", mesh.vertices.len())?;
    out.set_item("n_triangles", mesh.triangles.len())?;
    Ok(out)
}

/// Build a CRW2 precision matrix from positions.
#[pyfunction]
#[pyo3(signature = (positions, tau=1.0, layout="simple"))]
fn crw2_precision_matrix(positions: Vec<f64>, tau: f64, layout: &str) -> PyResult<PyCscMatrix> {
    let csc =
        inla_core::crw2_precision_csc(&positions, tau, layout).map_err(PyValueError::new_err)?;
    Ok(PyCscMatrix { matrix: csc })
}

/// Evaluate a named prior on internal θ: `log π(θ | prior, param)`.
#[pyfunction]
fn prior_log_density(name: &str, param: Vec<f64>, theta: Vec<f64>) -> PyResult<f64> {
    let spec =
        inla_core::PriorSpec::from_name_params(name, &param).map_err(PyValueError::new_err)?;
    spec.log_density(&theta).map_err(PyValueError::new_err)
}

/// Sum log-densities for a stack of named priors (concatenated θ).
#[pyfunction]
fn hyper_prior_stack_log_density(
    names: Vec<String>,
    params: Vec<Vec<f64>>,
    theta: Vec<f64>,
) -> PyResult<f64> {
    let stack = inla_core::HyperPriorStack::from_names_params(&names, &params)
        .map_err(PyValueError::new_err)?;
    stack.log_density(&theta).map_err(PyValueError::new_err)
}

/// Default `(prior_name, param)` list for an effect model (`besag`, `ar1`, …).
#[pyfunction]
fn default_hyper_priors(model: &str) -> PyResult<Vec<(String, Vec<f64>)>> {
    let stack =
        inla_core::HyperPriorStack::default_for_effect(model).map_err(PyValueError::new_err)?;
    Ok(stack.to_names_params())
}

/// The initialization function for the Python extension module `inla._native`.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCscMatrix>()?;
    m.add_class::<PyMarginal1D>()?;
    m.add_class::<PyInferenceResult>()?;
    m.add_function(wrap_pyfunction!(ar1_precision_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(ar1_precision_matrix_csc, m)?)?;
    m.add_function(wrap_pyfunction!(rw1_precision_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(rw2_precision_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(iid_precision_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(besag_precision_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(bym_precision_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(bym2_precision_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(fgn_precision_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(fgn_approx_precision_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(seasonal_precision_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(arp_precision_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(crw1_precision_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(crw2_precision_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(matern2d_precision_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(rw2d_precision_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(kronecker_csc, m)?)?;
    m.add_function(wrap_pyfunction!(spde_precision_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(spde_projector_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(fem_blocks_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(fgn_hurst_from_intern, m)?)?;
    m.add_function(wrap_pyfunction!(fgn_intern_from_hurst, m)?)?;
    m.add_function(wrap_pyfunction!(fgn_approx_latent_len, m)?)?;
    m.add_function(wrap_pyfunction!(prior_log_density, m)?)?;
    m.add_function(wrap_pyfunction!(hyper_prior_stack_log_density, m)?)?;
    m.add_function(wrap_pyfunction!(default_hyper_priors, m)?)?;
    m.add_function(wrap_pyfunction!(run_inla_inference_py, m)?)?;
    m.add_function(wrap_pyfunction!(run_gaussian_ar1_plan, m)?)?;
    m.add_function(wrap_pyfunction!(build_structured_precision_py, m)?)?;
    m.add_function(wrap_pyfunction!(structured_constraints_py, m)?)?;
    m.add_function(wrap_pyfunction!(model_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(plane_constraint_2d, m)?)?;
    m.add_function(wrap_pyfunction!(seasonal_constraint, m)?)?;
    m.add_function(wrap_pyfunction!(supported_models, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_compute_options, m)?)?;
    Ok(())
}
