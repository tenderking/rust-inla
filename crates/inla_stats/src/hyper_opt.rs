//! INLA-specific hyperparameter optimization (wraps [`inla_math::hyper_opt`]).

use inla_math::{ConstraintSpec, CscMatrix};

use crate::inference::{Obs, find_latent_mode_a};

pub struct ModelConfig<'a> {
    pub build_prior: &'a dyn Fn(&[f64]) -> Result<CscMatrix, String>,
    pub log_prior_density: &'a dyn Fn(&[f64]) -> f64,
    pub obs: &'a [Obs],
    /// Optional observation projector η = A x. `None` ⇒ identity.
    pub a: Option<&'a CscMatrix>,
    /// Optional linear constraints `A_c x = e` (hard extraconstr).
    pub constraints: Option<&'a ConstraintSpec>,
}

pub fn evaluate_neg_log_posterior(theta: &[f64], config: &ModelConfig) -> Result<f64, String> {
    let q_prior = (config.build_prior)(theta)?;
    match find_latent_mode_a(&q_prior, config.obs, config.a, config.constraints, 200, 1e-5) {
        Ok((_x_star, _factor, marginal_log_lik)) => {
            let log_prior = (config.log_prior_density)(theta);
            Ok(-(marginal_log_lik + log_prior))
        }
        // Keep hyperparameter search alive on rare Newton failures at extreme θ.
        Err(_) => Ok(1e12),
    }
}

pub fn nelder_mead(
    initial: &[f64],
    step_size: f64,
    max_iter: usize,
    tol: f64,
    config: &ModelConfig,
) -> Result<Vec<f64>, String> {
    inla_math::nelder_mead(initial, step_size, max_iter, tol, &|theta| {
        evaluate_neg_log_posterior(theta, config)
    })
}

pub fn compute_hessian(mode: &[f64], config: &ModelConfig, h: f64) -> Result<Vec<f64>, String> {
    inla_math::compute_hessian(mode, &|theta| evaluate_neg_log_posterior(theta, config), h)
}
