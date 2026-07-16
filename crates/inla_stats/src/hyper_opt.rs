//! INLA-specific hyperparameter optimization (wraps [`inla_math::hyper_opt`]).

use inla_math::CscMatrix;

use crate::inference::{find_latent_mode_a, Obs};

pub struct ModelConfig<'a> {
    pub build_prior: &'a dyn Fn(&[f64]) -> Result<CscMatrix, String>,
    pub log_prior_density: &'a dyn Fn(&[f64]) -> f64,
    pub obs: &'a [Obs],
    /// Optional observation projector η = A x. `None` ⇒ identity.
    pub a: Option<&'a CscMatrix>,
}

pub fn evaluate_neg_log_posterior(theta: &[f64], config: &ModelConfig) -> Result<f64, String> {
    let q_prior = (config.build_prior)(theta)?;
    let (_x_star, _factor, marginal_log_lik) =
        find_latent_mode_a(&q_prior, config.obs, config.a, 50, 1e-5)?;
    let log_prior = (config.log_prior_density)(theta);
    Ok(-(marginal_log_lik + log_prior))
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

pub fn compute_hessian(
    mode: &[f64],
    config: &ModelConfig,
    h: f64,
) -> Result<Vec<f64>, String> {
    inla_math::compute_hessian(mode, &|theta| evaluate_neg_log_posterior(theta, config), h)
}
