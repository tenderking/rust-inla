//! Compatibility facade for the split workspace.
//!
//! | Crate | Role |
//! |-------|------|
//! | [`inla_fmesher`] | Mesh / FEM geometry |
//! | [`inla_math`] | Sparse LDLT, design matrices, CCD/grid, Nelder–Mead |
//! | [`inla_stats`] | Likelihoods, latent models, INLA inference, DIC/CPO |
//!
//! Downstream code (`r-inla`, `py-inla`) can keep `use inla_core::...`.
//! Prefer depending on the leaf crates directly when working inside one layer;
//! this facade exists so existing `inla_core::module::…` paths keep compiling.
//!
//! Module re-exports below mirror the pre-split layout. Each submodule doc
//! notes the owning crate (and any cross-crate exceptions).

pub use inla_fmesher as fmesher;
pub use inla_math as math;

// --- Geometry ([`inla_fmesher`]) ---
pub use inla_fmesher::{
    BoundaryInput, EdgeRef, FemBlocks, Mesh2D, MeshSummary, PathStep, PathTrace, PointLocation,
    SparseTriplet, Triangle, Vertex2, build_boundary_segments, build_mesh2d,
    load_fmesher_boundary_input, load_fmesher_raw_boundary_input, read_boundary_indices,
    read_mesh_summary, read_positions_xy,
};

// --- Math / sparse engine ([`inla_math`]) ---
// `math_compute_hessian` / `math_nelder_mead` are the generic math optimizers;
// flat `compute_hessian` / `nelder_mead` below are the INLA wrappers from stats.
pub use inla_math::{
    CscForR, CscMatrix, DenseLdltFactor, Eval1D, FaerCpuSolver, InlaSolver, LdltFactor,
    LdltScratch, MathError, SolverError, add_csc, at_diag_a, block_diag_csc, ccd_design,
    compute_hessian as math_compute_hessian, csc_for_r_dgcmatrix, csc_from_triplets_0based,
    csc_to_dense, grid_design, identity_csc, invert_symmetric_matrix, jacobi_eigen, kronecker_csc,
    laplace_newton_step, laplace_newton_step_a, laplace_newton_step_a_solver,
    laplace_newton_system_a, ldlt_diagonal_inverse, ldlt_factorize, ldlt_factorize_dense,
    ldlt_solve, ldlt_solve_in_place, matvec_csc, matvec_transpose_csc,
    nelder_mead as math_nelder_mead, predictor_variances_diag, scale_csc, scale_model_csc,
    sparse_from_triplets, triplets_to_csc, with_thread_scratch,
};

/// CSC helpers from [`inla_math::sparse`], plus [`ar1_precision_csc`] from [`inla_stats`]
/// (kept here for the historical `inla_core::sparse::ar1_precision_csc` path).
pub mod sparse {
    pub use inla_math::sparse::*;
    pub use inla_stats::ar1_precision_csc;
}
/// LDLᵀ factorization — [`inla_math::ldlt`].
pub mod ldlt {
    pub use inla_math::ldlt::*;
}
/// Design / A-matrix helpers — [`inla_math::design`].
pub mod design {
    pub use inla_math::design::*;
}
/// CCD / grid integration designs — [`inla_math::integration`].
pub mod integration {
    pub use inla_math::integration::*;
}
/// INLA hyperparameter optimization — [`inla_stats::hyper_opt`].
pub mod hyper_opt {
    pub use inla_stats::hyper_opt::*;
}
/// Likelihoods and INLA inference — [`inla_stats::inference`].
pub mod inference {
    pub use inla_stats::inference::*;
}
/// Generic latent GMRF builders — [`inla_stats::latent_models`].
pub mod latent_models {
    pub use inla_stats::latent_models::*;
}
/// DIC / CPO / PIT / marginal likelihood — [`inla_stats::model_selection`].
pub mod model_selection {
    pub use inla_stats::model_selection::*;
}
/// AR(1) precision — [`inla_stats::ar1`].
pub mod ar1 {
    pub use inla_stats::ar1::*;
}
/// AR(p) precision — [`inla_stats::arp`].
pub mod arp {
    pub use inla_stats::arp::*;
}
/// Besag / BYM — [`inla_stats::besag`].
pub mod besag {
    pub use inla_stats::besag::*;
}
/// Continuous RW — [`inla_stats::crw`].
pub mod crw {
    pub use inla_stats::crw::*;
}
/// Fractional Gaussian noise — [`inla_stats::fgn`].
pub mod fgn {
    pub use inla_stats::fgn::*;
}
/// Matérn 2D lattice — [`inla_stats::matern2d`].
pub mod matern2d {
    pub use inla_stats::matern2d::*;
}
/// RW2D — [`inla_stats::rw2d`].
pub mod rw2d {
    pub use inla_stats::rw2d::*;
}
/// SPDE precision from FEM blocks — [`inla_stats::spde`].
pub mod spde {
    pub use inla_stats::spde::*;
}
/// Mesh summary I/O — [`inla_fmesher`] (`mesh` module).
pub mod mesh {
    pub use inla_fmesher::{MeshSummary, read_mesh_summary};
}

/// Structured multi-effect θ→Q — [`inla_stats::structured`].
pub mod structured {
    pub use inla_stats::structured::*;
}

/// Per-model metadata registry — [`inla_stats::registry`].
pub mod registry {
    pub use inla_stats::registry::*;
}

/// Named control/option bag — [`inla_stats::options`].
pub mod options {
    pub use inla_stats::options::*;
}

/// Language-neutral ModelSpec / ModelPlan IR — [`inla_stats::plan`].
pub mod plan {
    pub use inla_stats::plan::*;
}

// --- Stats flat re-exports (previous inla_core public API) ---
pub use inla_stats::{
    Ar1Precision, BinomialObs, ClosureLatentModel, ComputationPlan, ComputationSpec,
    ConstraintMethod, ConstraintSpec, CpoResult, DicResult, DynClosureLatentModel,
    ExponentialSurvivalObs, GammaPrior, GaussianObs, GaussianPrior, HARD_CONSTRAINT_KAPPA,
    HyperPriorStack, HyperSlotPlan, HyperTransformKind, IdentityProjection, InferenceResult,
    LaplaceObs, LatentBlockLayout, LatentEffectPlan, LatentEffectSpec, LatentLayout, LatentModel,
    LikelihoodPlan, LikelihoodSpec, Link, Marginal1D, MarginalOptions, ModelConfig, ModelPlan,
    ModelSpec, NegativeBinomialObs, Obs, PlanError, PoissonObs, PriorFamily, PriorSpec,
    ProjectionMapper, SparseProjectionMapper, WaicResult, WeibullSurvivalObs,
    ZeroInflatedBinomialObs, ZeroInflatedPoissonObs, ZeroInflationType, ar1_precision,
    ar1_precision_csc, arp_precision_csc, augment_precision_csc, besag_precision_csc,
    bym_precision_csc, bym2_precision_csc, compute_cpo_pit, compute_dic, compute_hessian,
    compute_marginal_log_lik_gaussian, compute_waic, crw1_precision_csc, crw2_precision_csc,
    eval_likelihood_binomial, eval_likelihood_exponential_survival, eval_likelihood_gaussian,
    eval_likelihood_laplace, eval_likelihood_negative_binomial, eval_likelihood_poisson,
    eval_likelihood_weibull_survival, eval_likelihood_zero_inflated_binomial,
    eval_likelihood_zero_inflated_poisson, eval_prior_gamma, eval_prior_gaussian,
    eval_prior_loggamma, evaluate_neg_log_posterior, fgn_approx_latent_len,
    fgn_approx_precision_csc, fgn_ar_coeffs, fgn_hurst_from_intern, fgn_intern_from_hurst,
    fgn_precision_csc, find_latent_mode, find_latent_mode_a, find_latent_mode_a_with_solver,
    gaussian_mixture_marginal, hyperpar_marginals, iid_precision_csc, marginal_cdf,
    marginal_quantiles, marginal_summary_quantiles, matern2d_precision_csc, model_rank_deficiency,
    nelder_mead, plane_constraint_2d, project_constraints, read_graph_file, resolve,
    run_gaussian_ar1_plan, run_inla_inference, run_inla_inference_a,
    run_inla_inference_a_cancellable, run_inla_inference_model, rw1_cyclic_precision_csc,
    rw1_precision_csc, rw2_cyclic_precision_csc, rw2_precision_csc, rw2d_precision_csc,
    seasonal_constraint, seasonal_precision_csc, spde_params_from_theta, spde_precision_csc, spde_projector_csc,
    spde_projector_from_xy, structured_constraints, structured_prior_stack,
    sum_to_zero_constraint, two_diid_precision_csc, ComputeOptions, HyperSlotMeta, IndexSelection,
    ModelMeta, OptionValue, SUPPORTED_GROUP_MODELS, SUPPORTED_MODELS, StructuredEffect,
    build_structured_precision, model_metadata, resolve_compute_options,
};

pub mod marginals {
    pub use inla_stats::marginals::*;
}
