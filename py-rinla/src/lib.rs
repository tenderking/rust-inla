use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::exceptions::PyValueError;
use inla_core::ar1_precision;

/// Computes a 1D AR1 precision matrix and returns sparse triplets (i, j, x)
/// in 1-based format (compatible with R sparse matrices).
#[pyfunction]
#[pyo3(signature = (n, rho, tau=1.0))]
fn ar1_precision_matrix(n: usize, rho: f64, tau: f64) -> PyResult<(Vec<usize>, Vec<usize>, Vec<f64>)> {
    let q = ar1_precision(n, rho, tau)
        .map_err(|e| PyValueError::new_err(e))?;
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
            .map_err(|e| PyValueError::new_err(e))?;
        Ok(PyCscMatrix { matrix })
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
        let indptr_arr = np.getattr("ctypeslib")?.call_method1(
            "as_array",
            (cast_indptr, (self.indptr_len(),)),
        )?;

        // indices: cast raw pointer and create a numpy array view
        let cast_indices = ctypes.call_method1("cast", (self.indices_ptr(), &c_uint64_ptr_t))?;
        let indices_arr = np.getattr("ctypeslib")?.call_method1(
            "as_array",
            (cast_indices, (self.indices_len(),)),
        )?;

        // data: cast raw pointer and create a numpy array view
        let c_double = ctypes.getattr("c_double")?;
        let c_double_ptr_t = ctypes.call_method1("POINTER", (c_double,))?;
        let cast_data = ctypes.call_method1("cast", (self.data_ptr(), &c_double_ptr_t))?;
        let data_arr = np.getattr("ctypeslib")?.call_method1(
            "as_array",
            (cast_data, (self.data_len(),)),
        )?;

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
}

/// Build an AR1 precision matrix.
#[pyfunction]
#[pyo3(signature = (n, rho, tau=1.0))]
fn ar1_precision_matrix_csc(n: usize, rho: f64, tau: f64) -> PyResult<PyCscMatrix> {
    let csc = inla_core::sparse::ar1_precision_csc(n, rho, tau)
        .map_err(|e| PyValueError::new_err(e))?;
    Ok(PyCscMatrix { matrix: csc })
}

/// Build an RW1 precision matrix.
#[pyfunction]
#[pyo3(signature = (n, tau=1.0))]
fn rw1_precision_matrix(n: usize, tau: f64) -> PyResult<PyCscMatrix> {
    let csc = inla_core::latent_models::rw1_precision_csc(n, tau)
        .map_err(|e| PyValueError::new_err(e))?;
    Ok(PyCscMatrix { matrix: csc })
}

/// Build an RW2 precision matrix.
#[pyfunction]
#[pyo3(signature = (n, tau=1.0))]
fn rw2_precision_matrix(n: usize, tau: f64) -> PyResult<PyCscMatrix> {
    let csc = inla_core::latent_models::rw2_precision_csc(n, tau)
        .map_err(|e| PyValueError::new_err(e))?;
    Ok(PyCscMatrix { matrix: csc })
}

/// Build an IID precision matrix.
#[pyfunction]
#[pyo3(signature = (n, tau=1.0))]
fn iid_precision_matrix(n: usize, tau: f64) -> PyResult<PyCscMatrix> {
    let csc = inla_core::latent_models::iid_precision_csc(n, tau)
        .map_err(|e| PyValueError::new_err(e))?;
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
#[pyo3(signature = (initial_theta, build_prior, log_prior_density, obs, strategy="ccd", step_or_f0=1.0))]
fn run_inla_inference_py(
    py: Python<'_>,
    initial_theta: Vec<f64>,
    build_prior: PyObject,
    log_prior_density: PyObject,
    obs: Vec<Bound<'_, PyAny>>,
    strategy: &str,
    step_or_f0: f64,
) -> PyResult<PyInferenceResult> {
    // 1. Parse Python observation list to Rust Obs structs
    let mut rust_obs = Vec::with_capacity(obs.len());
    for item in obs {
        rust_obs.push(parse_obs(&item)?);
    }

    // 2. Closure for build_prior calling back to Python
    let build_prior_closure = move |theta: &[f64]| -> Result<inla_core::sparse::CscMatrix, String> {
        Python::with_gil(|py| {
            let theta_py = theta.to_vec();
            let res = build_prior.call1(py, (theta_py,)).map_err(|e| e.to_string())?;
            let py_matrix: PyCscMatrix = res.extract(py).map_err(|e| e.to_string())?;
            Ok(py_matrix.matrix)
        })
    };

    // 3. Closure for log_prior_density calling back to Python
    let log_prior_density_closure = move |theta: &[f64]| -> f64 {
        Python::with_gil(|py| {
            let theta_py = theta.to_vec();
            let res = log_prior_density.call1(py, (theta_py,));
            match res {
                Ok(val) => val.extract::<f64>(py).unwrap_or(f64::NEG_INFINITY),
                Err(_) => f64::NEG_INFINITY,
            }
        })
    };

    // 4. Run the core solver (releasing GIL for Rayon parallel execution)
    let result = py.allow_threads(|| {
        inla_core::run_inla_inference(
            &initial_theta,
            &build_prior_closure,
            &log_prior_density_closure,
            &rust_obs,
            strategy,
            step_or_f0,
        )
    }).map_err(|e| PyValueError::new_err(e))?;

    // 5. Build and return the wrapped result
    Ok(PyInferenceResult {
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
        cpo: result.cpo,
        pit: result.pit,
        cpo_n_failures: result.cpo_n_failures,
    })
}

/// Helper function to parse python dicts representing observations.
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
        Ok(item) => if item.is_none() { None } else { Some(item.extract()?) },
        Err(_) => None,
    };
    
    let link = match link_str.as_deref() {
        Some("identity") | None => inla_core::Link::Identity,
        Some("log") => inla_core::Link::Log,
        Some("logit") => inla_core::Link::Logit,
        Some(other) => return Err(PyValueError::new_err(format!("unknown link function: {}", other))),
    };

    match family.as_str() {
        "none" => Ok(inla_core::Obs::None),
        "gaussian" => {
            let y: f64 = dict.get_item("y")?.extract()?;
            let precision: f64 = dict.get_item("precision")?.extract()?;
            Ok(inla_core::Obs::Gaussian(inla_core::GaussianObs { y, precision, link }))
        }
        "poisson" => {
            let y: f64 = dict.get_item("y")?.extract()?;
            let exposure: f64 = dict.get_item("exposure")?.extract()?;
            Ok(inla_core::Obs::Poisson(inla_core::PoissonObs { y, exposure, link }))
        }
        "binomial" => {
            let y: f64 = dict.get_item("y")?.extract()?;
            let n: f64 = dict.get_item("n")?.extract()?;
            Ok(inla_core::Obs::Binomial(inla_core::BinomialObs { y, n, link }))
        }
        "negative_binomial" | "nbinomial" => {
            let y: f64 = dict.get_item("y")?.extract()?;
            let exposure: f64 = dict.get_item("exposure")?.extract()?;
            let size: f64 = dict.get_item("size")?.extract()?;
            Ok(inla_core::Obs::NegativeBinomial(inla_core::NegativeBinomialObs { y, exposure, size, link }))
        }
        "zero_inflated_poisson" | "zeroinflatedpoisson0" | "zeroinflatedpoisson1" => {
            let y: f64 = dict.get_item("y")?.extract()?;
            let exposure: f64 = dict.get_item("exposure")?.extract()?;
            let zero_prob: f64 = dict.get_item("zero_prob")?.extract()?;
            let inflation_str: Option<String> = match dict.get_item("inflation") {
                Ok(item) => if item.is_none() { None } else { Some(item.extract()?) },
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
            Ok(inla_core::Obs::ZeroInflatedPoisson(inla_core::ZeroInflatedPoissonObs { y, exposure, zero_prob, link, inflation }))
        }
        "zero_inflated_binomial" | "zeroinflatedbinomial0" | "zeroinflatedbinomial1" => {
            let y: f64 = dict.get_item("y")?.extract()?;
            let n: f64 = dict.get_item("n")?.extract()?;
            let zero_prob: f64 = dict.get_item("zero_prob")?.extract()?;
            let inflation_str: Option<String> = match dict.get_item("inflation") {
                Ok(item) => if item.is_none() { None } else { Some(item.extract()?) },
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
            Ok(inla_core::Obs::ZeroInflatedBinomial(inla_core::ZeroInflatedBinomialObs { y, n, zero_prob, link, inflation }))
        }
        "laplace" => {
            let y: f64 = dict.get_item("y")?.extract()?;
            let alpha: f64 = dict.get_item("alpha")?.extract()?;
            let gamma: f64 = dict.get_item("gamma")?.extract()?;
            Ok(inla_core::Obs::Laplace(inla_core::LaplaceObs { y, alpha, gamma, link }))
        }
        "exponential_survival" | "exponential" => {
            let y: f64 = dict.get_item("y")?.extract()?;
            let event: f64 = dict.get_item("event")?.extract()?;
            Ok(inla_core::Obs::ExponentialSurvival(inla_core::ExponentialSurvivalObs { y, event, link }))
        }
        "weibull_survival" | "weibull" => {
            let y: f64 = dict.get_item("y")?.extract()?;
            let event: f64 = dict.get_item("event")?.extract()?;
            let shape: f64 = dict.get_item("shape")?.extract()?;
            Ok(inla_core::Obs::WeibullSurvival(inla_core::WeibullSurvivalObs { y, event, shape, link }))
        }
        _ => Err(PyValueError::new_err(format!("unknown observation family: {}", family))),
    }
}

/// Build an FGN (exact dense) precision matrix.
#[pyfunction]
#[pyo3(signature = (n, hurst, tau=1.0))]
fn fgn_precision_matrix(n: usize, hurst: f64, tau: f64) -> PyResult<PyCscMatrix> {
    let csc = inla_core::fgn_precision_csc(n, hurst, tau)
        .map_err(PyValueError::new_err)?;
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

/// The initialization function for the Python module.
#[pymodule]
fn rinla(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCscMatrix>()?;
    m.add_class::<PyInferenceResult>()?;
    m.add_function(wrap_pyfunction!(ar1_precision_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(ar1_precision_matrix_csc, m)?)?;
    m.add_function(wrap_pyfunction!(rw1_precision_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(rw2_precision_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(iid_precision_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(fgn_precision_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(fgn_approx_precision_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(run_inla_inference_py, m)?)?;
    Ok(())
}
