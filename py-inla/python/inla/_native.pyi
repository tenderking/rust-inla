"""Type stub for the PyO3 extension ``inla._native``."""

from typing import Any

class PyCscMatrix:
    def __init__(
        self, nrow: int, ncol: int, i: list[int], j: list[int], x: list[float]
    ) -> None: ...
    @staticmethod
    def from_scipy(mat: Any) -> PyCscMatrix: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    def to_scipy(self) -> Any: ...

class PyMarginal1D:
    x: list[float]
    y: list[float]
    def quantiles(self, probs: list[float]) -> list[float]: ...
    def emarginal(self, g_of_x: list[float]) -> float: ...

class PyInferenceResult:
    mode: list[float]
    hessian: list[float]
    latent_means: list[float]
    latent_variances: list[float]
    predictor_means: list[float]
    predictor_variances: list[float]
    marginal_log_lik: float
    marginal_log_lik_gaussian: float
    dic: float
    mean_deviance: float
    effective_params: float
    waic: float
    waic_lppd: float
    waic_effective_params: float
    cpo: list[float | None]
    pit: list[float | None]
    cpo_n_failures: int
    node_weights: list[float]
    internal_marginals_hyperpar: list[PyMarginal1D]
    marginals_latent: list[PyMarginal1D]
    marginals_latent_indices: list[int]
    marginals_predictor: list[PyMarginal1D]
    marginals_predictor_indices: list[int]
    def lincomb(self, combs: list[tuple[str, list[tuple[int, float]]]]) -> list[dict[str, Any]]: ...
    def posterior_sample(self, n_samples: int, seed: int = 1) -> list[float]: ...

def ar1_precision_matrix(n: int, rho: float, tau: float = 1.0) -> Any: ...
def ar1_precision_matrix_csc(n: int, rho: float, tau: float = 1.0) -> PyCscMatrix: ...
def rw1_precision_matrix(n: int, tau: float = 1.0) -> PyCscMatrix: ...
def rw2_precision_matrix(n: int, tau: float = 1.0) -> PyCscMatrix: ...
def iid_precision_matrix(n: int, tau: float = 1.0) -> PyCscMatrix: ...
def besag_precision_matrix(adj: list[list[int]], tau: float = 1.0) -> PyCscMatrix: ...
def bym_precision_matrix(
    adj: list[list[int]], tau_spatial: float = 1.0, tau_iid: float = 1.0
) -> PyCscMatrix: ...
def bym2_precision_matrix(
    adj: list[list[int]], tau: float = 1.0, phi: float = 0.5
) -> PyCscMatrix: ...
def fgn_precision_matrix(n: int, hurst: float, tau: float = 1.0) -> PyCscMatrix: ...
def fgn_approx_precision_matrix(
    n: int, hurst: float, tau: float = 1.0, order: int = 4, prec_eps: float = 1e8
) -> PyCscMatrix: ...
def fgn_hurst_from_intern(h_intern: float) -> float: ...
def fgn_intern_from_hurst(h: float) -> float: ...
def fgn_approx_latent_len(n_obs: int, order: int) -> int: ...
def seasonal_precision_matrix(
    n: int, season: int = 4, tau: float = 1.0, cyclic: bool = True
) -> PyCscMatrix: ...
def arp_precision_matrix(n: int, pacf: list[float], tau: float = 1.0) -> PyCscMatrix: ...
def crw1_precision_matrix(positions: list[float], tau: float = 1.0) -> PyCscMatrix: ...
def crw2_precision_matrix(
    positions: list[float], tau: float = 1.0, layout: str = "simple"
) -> PyCscMatrix: ...
def matern2d_precision_matrix(
    nrow: int,
    ncol: int,
    nu: int = 1,
    range: float = 1.0,
    prec: float = 1.0,
    cyclic: bool = False,
) -> PyCscMatrix: ...
def rw2d_precision_matrix(
    nrow: int,
    ncol: int,
    tau: float = 1.0,
    cyclic: bool = False,
    bvalue_zero: bool = False,
) -> PyCscMatrix: ...
def kronecker_csc(a: Any, b: Any) -> PyCscMatrix: ...
def spde_precision_matrix(
    vertices: list[tuple[float, float]],
    triangles: list[tuple[int, int, int]],
    kappa: float,
    tau: float = 1.0,
) -> PyCscMatrix: ...
def spde_projector_matrix(
    vertices: list[tuple[float, float]],
    triangles: list[tuple[int, int, int]],
    loc_x: list[float],
    loc_y: list[float],
) -> PyCscMatrix: ...
def fem_blocks_mesh(
    vertices: list[tuple[float, float]], triangles: list[tuple[int, int, int]]
) -> dict[str, Any]: ...
def prior_log_density(name: str, param: list[float], theta: list[float]) -> float: ...
def hyper_prior_stack_log_density(
    names: list[str], params: list[list[float]], theta: list[float]
) -> float: ...
def default_hyper_priors(model: str) -> list[tuple[str, list[float]]]: ...
def run_inla_inference(
    initial_theta: list[float],
    build_prior: Any,
    log_prior_density: Any,
    obs: list[Any],
    strategy: str = "ccd",
    step_or_f0: float = 1.0,
    n_points: int = 201,
    latent_marginal_indices: list[int] | None = None,
    predictor_marginal_indices: list[int] | None = None,
    a: Any | None = None,
    constraints_a: list[float] | None = None,
    constraints_e: list[float] | None = None,
    deterministic: bool = False,
    gaussian_free_prec: bool = False,
) -> PyInferenceResult: ...
def run_gaussian_ar1_plan(
    y: list[float],
    name: str = "time",
    obs_precision: float = 100.0,
    strategy: str = "ccd",
    step_or_f0: float = 1.0,
    initial_theta: list[float] | None = None,
) -> PyInferenceResult: ...
def build_structured_precision(
    effects: list[dict[str, Any]], theta: list[float], fixed_prec: float = 1e-4
) -> PyCscMatrix: ...
def structured_constraints(
    effects: list[dict[str, Any]],
) -> tuple[list[float], list[float]] | None: ...
def scale_model_csc(q: Any) -> PyCscMatrix: ...
def model_metadata(
    model: str,
    order: int = 0,
    group_model: str | None = None,
    cyclic: bool = False,
) -> dict[str, Any]: ...
def plane_constraint_2d(nrow: int, ncol: int) -> tuple[list[float], list[float]]: ...
def seasonal_constraint(n: int, season: int) -> tuple[list[float], list[float]]: ...
def supported_models() -> list[str]: ...
def resolve_compute_options(controls: dict[str, Any]) -> dict[str, Any]: ...
