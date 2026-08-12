//! Statistical layer: likelihoods, latent GMRFs, INLA inference, model selection.

pub mod ar1;
pub mod arp;
pub mod besag;
pub mod crw;
pub mod fgn;
pub mod hyper_opt;
pub mod inference;
pub mod latent;
pub mod latent_models;
pub mod marginals;
pub mod matern2d;
pub mod model_selection;
pub mod options;
pub mod plan;
pub mod priors;
pub mod projection;
pub mod registry;
pub mod rw2d;
pub mod spde;
pub mod structured;

pub use ar1::{Ar1Precision, ar1_precision, ar1_precision_csc};
pub use arp::arp_precision_csc;
pub use besag::{besag_precision_csc, bym_precision_csc, bym2_precision_csc, read_graph_file};
pub use crw::{crw1_precision_csc, crw2_precision_csc};
pub use fgn::{
    fgn_approx_latent_len, fgn_approx_precision_csc, fgn_ar_coeffs, fgn_hurst_from_intern,
    fgn_intern_from_hurst,
};
pub use hyper_opt::{ModelConfig, compute_hessian, evaluate_neg_log_posterior, nelder_mead};
pub use inference::{
    BinomialObs, ExponentialSurvivalObs, GammaPrior, GaussianObs, GaussianPrior, InferenceResult,
    LaplaceObs, Link, NegativeBinomialObs, Obs, PoissonObs, WeibullSurvivalObs,
    ZeroInflatedBinomialObs, ZeroInflatedPoissonObs, ZeroInflationType, eval_likelihood,
    eval_likelihood_binomial, eval_likelihood_exponential_survival, eval_likelihood_gaussian,
    eval_likelihood_laplace, eval_likelihood_negative_binomial, eval_likelihood_poisson,
    eval_likelihood_weibull_survival, eval_likelihood_zero_inflated_binomial,
    eval_likelihood_zero_inflated_poisson, eval_prior_gamma, eval_prior_gaussian,
    eval_prior_loggamma, find_latent_mode, find_latent_mode_a, find_latent_mode_a_with_solver,
    run_inla_inference, run_inla_inference_a, run_inla_inference_a_cancellable,
    run_inla_inference_model,
};
pub use latent_models::{
    fgn_precision_csc, iid_precision_csc, rw1_cyclic_precision_csc, rw1_precision_csc,
    rw2_cyclic_precision_csc, rw2_precision_csc, seasonal_precision_csc, two_diid_precision_csc,
};
pub use marginals::{
    Marginal1D, MarginalOptions, gaussian_mixture_marginal, hyperpar_marginals, marginal_cdf,
    marginal_quantiles, marginal_summary_quantiles,
};
pub use matern2d::matern2d_precision_csc;
pub use model_selection::{
    CpoResult, DicResult, WaicResult, compute_cpo_pit, compute_dic, compute_marginal_log_lik_gaussian,
    compute_waic,
};
pub use options::{
    ComputeOptions, IndexSelection, OptionValue, resolve_compute_options,
};
pub use plan::{
    ComputationPlan, ComputationSpec, HyperSlotPlan, HyperTransformKind, LatentBlockLayout,
    LatentEffectPlan, LatentEffectSpec, LatentLayout, LikelihoodPlan, LikelihoodSpec, ModelPlan,
    ModelSpec, PlanError, resolve, run_gaussian_ar1_plan,
};
pub use registry::{
    HyperSlotMeta, ModelMeta, SUPPORTED_GROUP_MODELS, SUPPORTED_MODELS, model_metadata,
    rank_deficiency,
};
pub use priors::{HyperPriorStack, PriorFamily, PriorSpec};
pub use rw2d::rw2d_precision_csc;
pub use spde::{
    spde_params_from_theta, spde_precision_csc, spde_projector_csc, spde_projector_from_xy,
};
pub use structured::{
    StructuredEffect, build_structured_precision, structured_constraints, structured_prior_stack,
};

pub use latent::{ClosureLatentModel, DynClosureLatentModel, LatentModel};
pub use projection::{IdentityProjection, ProjectionMapper, SparseProjectionMapper};

// Re-export math primitives commonly used with stats APIs.
pub use inla_math::{
    ConstraintMethod, ConstraintSpec, CscMatrix, Eval1D, HARD_CONSTRAINT_KAPPA,
    augment_precision_csc, model_rank_deficiency, plane_constraint_2d, project_constraints,
    seasonal_constraint, sum_to_zero_constraint,
};
