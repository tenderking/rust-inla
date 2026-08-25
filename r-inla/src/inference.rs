//! Observation builders and INLA inference entry points for R.

use crate::convert::{
    csc_from_r_slots, marginals_to_r_list, parse_adj_list_1based, parse_effect_positions,
    posterior_q_slots,
};
use extendr_api::prelude::*;

/// Canonicalize likelihood family strings (R-INLA aliases → internal names).
fn canonicalize_family(family: &str) -> String {
    match family.trim().to_lowercase().as_str() {
        "exponential.surv" | "exponential_surv" => "exponential_survival".into(),
        "weibull.surv" | "weibull_surv" => "weibull_survival".into(),
        "negbin" | "nbinomial" => "negative_binomial".into(),
        "zip" | "zeroinflatedpoisson0" => "zero_inflated_poisson".into(),
        "zib" | "zeroinflatedbinomial0" => "zero_inflated_binomial".into(),
        "cbinomial" => "binomial".into(),
        other => other.to_string(),
    }
}

fn parse_link(link: &str, family: &str) -> std::result::Result<inla_core::Link, Error> {
    let link = link.trim().to_lowercase();
    let family = canonicalize_family(family);
    if link.is_empty() || link == "default" {
        return Ok(match family.as_str() {
            "gaussian" | "laplace" => inla_core::Link::Identity,
            "poisson"
            | "nbinomial"
            | "negative_binomial"
            | "zeroinflatedpoisson0"
            | "zeroinflatedpoisson1"
            | "zero_inflated_poisson"
            | "exponential"
            | "exponential_survival"
            | "weibull"
            | "weibull_survival" => inla_core::Link::Log,
            "binomial"
            | "zeroinflatedbinomial0"
            | "zeroinflatedbinomial1"
            | "zero_inflated_binomial" => inla_core::Link::Logit,
            _ => inla_core::Link::Identity,
        });
    }
    match link.as_str() {
        "identity" => Ok(inla_core::Link::Identity),
        "log" => Ok(inla_core::Link::Log),
        "logit" => Ok(inla_core::Link::Logit),
        other => Err(Error::Other(format!("unknown link function: {other}"))),
    }
}

fn pad_or_default(values: &[f64], n: usize, default: f64) -> Vec<f64> {
    if values.is_empty() {
        return vec![default; n];
    }
    if values.len() == 1 {
        return vec![values[0]; n];
    }
    if values.len() == n {
        return values.to_vec();
    }
    // Truncate or pad with last/default so callers can pass partial vectors safely.
    let mut out = vec![default; n];
    let m = values.len().min(n);
    out[..m].copy_from_slice(&values[..m]);
    out
}

fn build_observations(
    family: &str,
    y_obs: &[f64],
    n_latent: usize,
    link: inla_core::Link,
    obs_precision: f64,
    exposure: &[f64],
    ntrials: &[f64],
    event: &[f64],
    size: f64,
    zero_prob: f64,
    inflation: &str,
    alpha: f64,
    gamma: f64,
    shape: f64,
) -> std::result::Result<Vec<inla_core::Obs>, Error> {
    let n = y_obs.len();
    let fam = canonicalize_family(family);
    let inflation_ty = match inflation.trim().to_lowercase().as_str() {
        "type1" | "1" => inla_core::ZeroInflationType::Type1,
        _ => inla_core::ZeroInflationType::Type0,
    };
    let e = pad_or_default(exposure, n, 1.0);
    let nt = pad_or_default(ntrials, n, 1.0);
    let ev = pad_or_default(event, n, 1.0);

    let mut obs = Vec::with_capacity(n_latent);
    for i in 0..n {
        let y = y_obs[i];
        let one = match fam.as_str() {
            "gaussian" => inla_core::Obs::Gaussian(inla_core::GaussianObs {
                y,
                precision: obs_precision,
                link,
            }),
            "poisson" => inla_core::Obs::Poisson(inla_core::PoissonObs {
                y,
                exposure: e[i],
                link,
            }),
            "binomial" => inla_core::Obs::Binomial(inla_core::BinomialObs { y, n: nt[i], link }),
            "negative_binomial" => {
                inla_core::Obs::NegativeBinomial(inla_core::NegativeBinomialObs {
                    y,
                    exposure: e[i],
                    size,
                    link,
                })
            }
            "zero_inflated_poisson" => {
                inla_core::Obs::ZeroInflatedPoisson(inla_core::ZeroInflatedPoissonObs {
                    y,
                    exposure: e[i],
                    zero_prob,
                    link,
                    inflation: inflation_ty,
                })
            }
            "zeroinflatedpoisson1" => {
                inla_core::Obs::ZeroInflatedPoisson(inla_core::ZeroInflatedPoissonObs {
                    y,
                    exposure: e[i],
                    zero_prob,
                    link,
                    inflation: inla_core::ZeroInflationType::Type1,
                })
            }
            "zero_inflated_binomial" => {
                inla_core::Obs::ZeroInflatedBinomial(inla_core::ZeroInflatedBinomialObs {
                    y,
                    n: nt[i],
                    zero_prob,
                    link,
                    inflation: inflation_ty,
                })
            }
            "zeroinflatedbinomial1" => {
                inla_core::Obs::ZeroInflatedBinomial(inla_core::ZeroInflatedBinomialObs {
                    y,
                    n: nt[i],
                    zero_prob,
                    link,
                    inflation: inla_core::ZeroInflationType::Type1,
                })
            }
            "laplace" => inla_core::Obs::Laplace(inla_core::LaplaceObs {
                y,
                alpha,
                gamma,
                link,
            }),
            "exponential" | "exponential_survival" => {
                inla_core::Obs::ExponentialSurvival(inla_core::ExponentialSurvivalObs {
                    y,
                    event: ev[i],
                    link,
                })
            }
            "weibull" | "weibull_survival" => {
                inla_core::Obs::WeibullSurvival(inla_core::WeibullSurvivalObs {
                    y,
                    event: ev[i],
                    shape,
                    link,
                })
            }
            other => {
                return Err(Error::Other(format!(
                    "unsupported observation family: {other}"
                )));
            }
        };
        obs.push(one);
    }
    for _ in n..n_latent {
        obs.push(inla_core::Obs::None);
    }
    Ok(obs)
}

#[extendr]
fn inla_rs_run_inla_inference(
    initial_theta: Vec<f64>,
    model_type: &str,
    y_obs: Vec<f64>,
    obs_precision: f64,
    strategy: &str,
    step_or_f0: f64,
    order: i32,
    family: &str,
    link: &str,
    exposure: Vec<f64>,
    ntrials: Vec<f64>,
    event: Vec<f64>,
    size: f64,
    zero_prob: f64,
    inflation: &str,
    alpha: f64,
    gamma: f64,
    shape: f64,
    adj_list: List,
    deterministic: bool,
) -> std::result::Result<List, Error> {
    let n = y_obs.len();
    let model_type_str = model_type.to_lowercase();
    let family_str = canonicalize_family(family);
    let use_fgn_approx = model_type_str == "fgn" && order > 0;
    let order_usize = if use_fgn_approx {
        usize::try_from(order)
            .map_err(|_| Error::Other("order must be non-negative".to_string()))?
    } else {
        0
    };
    if use_fgn_approx && order_usize != 3 && order_usize != 4 {
        return Err(Error::Other(
            "FGN order must be 3 or 4 (R-INLA AR mixture), or 0 for exact dense FGN".to_string(),
        ));
    }

    let adj: Option<Vec<Vec<usize>>> = if model_type_str == "besag" {
        let out = parse_adj_list_1based(&adj_list)?;
        if out.len() != n {
            return Err(Error::Other(format!(
                "besag adj_list length ({}) must equal length(y) ({n})",
                out.len()
            )));
        }
        Some(out)
    } else {
        None
    };

    // Observations: for approx FGN, only the first n (z-block) are observed.
    let n_latent = if use_fgn_approx {
        inla_core::fgn_approx_latent_len(n, order_usize)
    } else {
        n
    };
    let link_ty = parse_link(link, &family_str)?;
    let obs = build_observations(
        &family_str,
        &y_obs,
        n_latent,
        link_ty,
        obs_precision,
        &exposure,
        &ntrials,
        &event,
        size,
        zero_prob,
        inflation,
        alpha,
        gamma,
        shape,
    )?;

    let model_type_owned = model_type_str.clone();
    let adj_owned = adj.clone();
    let build_prior = move |theta: &[f64]| -> std::result::Result<inla_core::CscMatrix, String> {
        match model_type_owned.as_str() {
            "fgn" if use_fgn_approx => {
                if theta.len() < 2 {
                    return Err(
                        "FGN requires 2 hyperparameters: theta = [log(tau), H_intern]".to_string(),
                    );
                }
                let tau = theta[0].exp();
                let hurst = inla_core::fgn_hurst_from_intern(theta[1]);
                // R-INLA default f(..., precision = 1e8)
                inla_core::fgn_approx_precision_csc(n, hurst, tau, order_usize, 1e8)
            }
            "fgn" => {
                if theta.len() < 2 {
                    return Err(
                        "FGN requires 2 hyperparameters: theta = [log(tau), logit(H)]".to_string(),
                    );
                }
                let tau = theta[0].exp();
                let hurst = 1.0 / (1.0 + (-theta[1]).exp());
                inla_core::fgn_precision_csc(n, hurst, tau)
            }
            "ar1" => {
                if theta.len() < 2 {
                    return Err(
                        "AR1 requires 2 hyperparameters: theta = [log(tau), logit((rho+1)/2)]"
                            .to_string(),
                    );
                }
                let tau = theta[0].exp();
                let rho = 2.0 / (1.0 + (-theta[1]).exp()) - 1.0;
                inla_core::ar1_precision_csc(n, rho, tau)
            }
            "rw2" => {
                if theta.is_empty() {
                    return Err("RW2 requires 1 hyperparameter: theta = [log(tau)]".to_string());
                }
                let tau = theta[0].exp();
                inla_core::rw2_precision_csc(n, tau)
            }
            "iid" => {
                if theta.is_empty() {
                    return Err("IID requires 1 hyperparameter: theta = [log(tau)]".to_string());
                }
                let tau = theta[0].exp();
                inla_core::iid_precision_csc(n, tau)
            }
            "besag" => {
                if theta.is_empty() {
                    return Err("Besag requires 1 hyperparameter: theta = [log(tau)]".to_string());
                }
                let tau = theta[0].exp();
                let adj = adj_owned
                    .as_ref()
                    .ok_or_else(|| "besag requires adj_list".to_string())?;
                inla_core::besag_precision_csc(adj, tau)
            }
            _ => Err(format!(
                "Unsupported model type for formula solving: {}",
                model_type_owned
            )),
        }
    };

    let constr = match inla_core::model_rank_deficiency(&model_type_str) {
        0 => None,
        k => Some(inla_core::sum_to_zero_constraint(n_latent, k).map_err(Error::Other)?),
    };

    let prior_stack =
        inla_core::HyperPriorStack::default_for_effect(&model_type_str).map_err(Error::Other)?;
    let log_prior_density =
        move |theta: &[f64]| -> f64 { prior_stack.log_density(theta).unwrap_or(f64::NEG_INFINITY) };

    let result = inla_core::run_inla_inference_a(
        &initial_theta,
        &build_prior,
        &log_prior_density,
        &obs,
        None,
        constr.as_ref(),
        strategy,
        step_or_f0,
        &inla_core::MarginalOptions::default(),
        deterministic,
    )
    .map_err(Error::Other)?;

    // Report FGN of interest (z-block) and a Hurst summary matching R-INLA scale when approx.
    let latent_means: Vec<f64> = result.latent_means.iter().take(n).copied().collect();
    let latent_variances: Vec<f64> = result.latent_variances.iter().take(n).copied().collect();
    let hurst_est = if model_type_str == "fgn" && result.mode.len() >= 2 {
        if use_fgn_approx {
            inla_core::fgn_hurst_from_intern(result.mode[1])
        } else {
            1.0 / (1.0 + (-result.mode[1]).exp())
        }
    } else {
        f64::NAN
    };

    let (posterior_q_i, posterior_q_p, posterior_q_x, posterior_q_n) =
        posterior_q_slots(&result.posterior_precision)?;

    Ok(list!(
        mode = result.mode,
        hessian = result.hessian,
        latent_means = latent_means,
        latent_variances = latent_variances,
        predictor_means = result.predictor_means,
        predictor_variances = result.predictor_variances,
        marginal_log_lik = result.marginal_log_lik,
        marginal_log_lik_gaussian = result.marginal_log_lik_gaussian,
        dic = result.dic,
        mean_deviance = result.mean_deviance,
        effective_params = result.effective_params,
        waic = result.waic,
        waic_lppd = result.waic_lppd,
        waic_effective_params = result.waic_effective_params,
        hurst = hurst_est,
        order = order,
        posterior_q_i = posterior_q_i,
        posterior_q_p = posterior_q_p,
        posterior_q_x = posterior_q_x,
        posterior_q_n = posterior_q_n
    ))
}

/// Structured INLA fit: A-matrix projector + block-diagonal multi-effect prior.
///
/// `effect_types`: "iid"|"ar1"|"rw2"|"besag"|"fixed"|"fgn"
/// `adj_lists`: list of the same length as effects; non-besag entries may be `list()`.
#[extendr]
fn inla_rs_run_inla_structured(
    initial_theta: Vec<f64>,
    y_obs: Vec<f64>,
    obs_precision: f64,
    strategy: &str,
    step_or_f0: f64,
    family: &str,
    link: &str,
    a_i: Vec<i32>,
    a_j: Vec<i32>,
    a_x: Vec<f64>,
    a_nrow: i32,
    a_ncol: i32,
    effect_types: Vec<String>,
    effect_ns: Vec<i32>,
    effect_scales: Vec<i32>,
    effect_theta_lens: Vec<i32>,
    effect_orders: Vec<i32>,
    effect_copy_of: Vec<i32>,
    adj_lists: List,
    effect_positions: List,
    prior_names: Vec<String>,
    prior_params: List,
    fixed_prec: f64,
    exposure: Vec<f64>,
    ntrials: Vec<f64>,
    event: Vec<f64>,
    size: f64,
    zero_prob: f64,
    inflation: &str,
    alpha: f64,
    gamma: f64,
    shape: f64,
    deterministic: bool,
    gaussian_free_prec: bool,
    family_prior_name: &str,
    family_prior_param: Vec<f64>,
) -> std::result::Result<List, Error> {
    let n_obs = y_obs.len();
    let a_nrow_u = usize::try_from(a_nrow).map_err(|_| Error::Other("a_nrow".into()))?;
    let a_ncol_u = usize::try_from(a_ncol).map_err(|_| Error::Other("a_ncol".into()))?;
    if a_nrow_u != n_obs {
        return Err(Error::Other(format!(
            "A.nrows ({a_nrow_u}) must equal length(y) ({n_obs})"
        )));
    }
    if effect_types.len() != effect_ns.len()
        || effect_types.len() != effect_scales.len()
        || effect_types.len() != effect_theta_lens.len()
        || effect_types.len() != effect_orders.len()
        || (!effect_copy_of.is_empty() && effect_copy_of.len() != effect_types.len())
    {
        return Err(Error::Other(
            "effect_* vectors must have equal length".to_string(),
        ));
    }
    if adj_lists.len() != effect_types.len() {
        return Err(Error::Other(
            "adj_lists length must match number of effects".to_string(),
        ));
    }
    if effect_positions.len() != effect_types.len() {
        return Err(Error::Other(
            "effect_positions length must match number of effects".to_string(),
        ));
    }
    if prior_params.len() != prior_names.len() {
        return Err(Error::Other(
            "prior_params length must match prior_names".to_string(),
        ));
    }

    let rows: Vec<usize> = a_i.iter().map(|&v| v as usize).collect();
    let cols: Vec<usize> = a_j.iter().map(|&v| v as usize).collect();
    let a = inla_core::csc_from_triplets_0based(a_nrow_u, a_ncol_u, &rows, &cols, &a_x)
        .map_err(Error::Other)?;

    let family_str = canonicalize_family(family);
    let link_ty = parse_link(link, &family_str)?;
    let obs = build_observations(
        &family_str,
        &y_obs,
        n_obs,
        link_ty,
        obs_precision,
        &exposure,
        &ntrials,
        &event,
        size,
        zero_prob,
        inflation,
        alpha,
        gamma,
        shape,
    )?;

    // Parse adjacency once per besag effect
    let mut adjs: Vec<Option<Vec<Vec<usize>>>> = Vec::with_capacity(effect_types.len());
    for (ei, item) in adj_lists.values().enumerate() {
        let typ = effect_types[ei].to_lowercase();
        if typ == "besag" || typ == "bym" || typ == "bym2" {
            let sub: List = item.try_into().map_err(|e| {
                Error::Other(format!(
                    "adj_lists[{ei}] must be a list of integer vectors: {e}"
                ))
            })?;
            adjs.push(Some(parse_adj_list_1based(&sub)?));
        } else {
            adjs.push(None);
        }
    }

    let effect_types_owned = effect_types.clone();
    let effect_ns_u: Vec<usize> = effect_ns
        .iter()
        .map(|&v| usize::try_from(v).unwrap_or(0))
        .collect();
    let effect_scales_b: Vec<bool> = effect_scales.iter().map(|&v| v != 0).collect();
    let effect_theta_lens_u: Vec<usize> = effect_theta_lens
        .iter()
        .map(|&v| usize::try_from(v).unwrap_or(0))
        .collect();
    let effect_orders_i: Vec<i32> = effect_orders.clone();
    let expected_theta: usize =
        effect_theta_lens_u.iter().sum::<usize>() + if gaussian_free_prec { 1 } else { 0 };
    if initial_theta.len() != expected_theta {
        return Err(Error::Other(format!(
            "initial_theta length {} != expected {}",
            initial_theta.len(),
            expected_theta
        )));
    }
    let n_lat_expected: usize = effect_ns_u.iter().sum();
    if n_lat_expected != a_ncol_u {
        return Err(Error::Other(format!(
            "sum(effect_ns)={n_lat_expected} != A.ncols={a_ncol_u}"
        )));
    }

    let positions = parse_effect_positions(&effect_positions, &effect_ns_u)?;

    let effects: Vec<inla_core::StructuredEffect> = (0..effect_types_owned.len())
        .map(|ei| {
            let typ = effect_types_owned[ei].to_lowercase();
            let raw_order = effect_orders_i[ei];
            let (nrow, ncol, cyclic) = if typ == "rw2d" || typ == "matern2d" {
                let cyclic = raw_order < 0;
                let nrow = raw_order.unsigned_abs() as usize;
                let ncol = if nrow > 0 && effect_ns_u[ei].is_multiple_of(nrow) {
                    effect_ns_u[ei] / nrow
                } else {
                    0
                };
                (nrow, ncol, cyclic)
            } else {
                (0, 0, false)
            };
            let copy_of = if effect_copy_of.is_empty() || effect_copy_of[ei] < 0 {
                None
            } else {
                Some(effect_copy_of[ei] as usize)
            };
            inla_core::StructuredEffect {
                model: typ,
                n: effect_ns_u[ei],
                scale_model: effect_scales_b[ei],
                theta_len: effect_theta_lens_u[ei],
                order: raw_order,
                adj: adjs[ei].clone(),
                positions: positions[ei].clone(),
                crw2_layout: "simple".into(),
                nrow,
                ncol,
                cyclic,
                matern_nu: 1,
                copy_of,
            }
        })
        .collect();

    let constr_opt = inla_core::structured_constraints(&effects).map_err(Error::Other)?;

    let effects_for_q = effects.clone();
    let build_prior = move |theta: &[f64]| -> std::result::Result<inla_core::CscMatrix, String> {
        let latent_th = if gaussian_free_prec {
            if theta.is_empty() { &[] } else { &theta[1..] }
        } else {
            theta
        };
        inla_core::build_structured_precision(&effects_for_q, latent_th, fixed_prec)
    };

    let log_prior_density = {
        let stack = if prior_names.is_empty() {
            inla_core::structured_prior_stack(&effects).map_err(Error::Other)?
        } else {
            let mut params = Vec::with_capacity(prior_names.len());
            for (i, item) in prior_params.values().enumerate() {
                let values = item.as_real_vector().ok_or_else(|| {
                    Error::Other(format!("prior_params[[{}]] must be numeric", i + 1))
                })?;
                params.push(values.to_vec());
            }
            let stack = inla_core::HyperPriorStack::from_names_params(&prior_names, &params)
                .map_err(Error::Other)?;
            let latent_theta_len: usize = effect_theta_lens_u.iter().sum();
            if stack.theta_dim() != latent_theta_len {
                return Err(Error::Other(format!(
                    "prior theta dimension {} != latent theta dimension {latent_theta_len}",
                    stack.theta_dim()
                )));
            }
            stack
        };
        let fam_prior =
            inla_core::PriorSpec::from_name_params(family_prior_name, &family_prior_param)
                .map_err(Error::Other)?;
        if fam_prior.theta_dim() != 1 {
            return Err(Error::Other(format!(
                "Gaussian family precision prior '{}' must consume one theta coordinate",
                family_prior_name
            )));
        }
        move |theta: &[f64]| -> f64 {
            if gaussian_free_prec {
                if theta.is_empty() {
                    return f64::NEG_INFINITY;
                }
                let lp_fam = fam_prior
                    .log_density(&theta[..1])
                    .unwrap_or(f64::NEG_INFINITY);
                let lp_latent = stack.log_density(&theta[1..]).unwrap_or(f64::NEG_INFINITY);
                lp_fam + lp_latent
            } else {
                stack.log_density(theta).unwrap_or(f64::NEG_INFINITY)
            }
        }
    };

    let y_obs_copy = y_obs.clone();
    let build_obs_closure = move |th: &[f64]| -> Vec<inla_core::Obs> {
        let prec = if !th.is_empty() { th[0].exp() } else { 1.0 };
        y_obs_copy
            .iter()
            .map(|&y| {
                inla_core::Obs::Gaussian(inla_core::GaussianObs {
                    y,
                    precision: prec,
                    link: link_ty,
                })
            })
            .collect()
    };
    let build_obs_opt: Option<&(dyn Fn(&[f64]) -> Vec<inla_core::Obs> + Sync)> =
        if gaussian_free_prec {
            Some(&build_obs_closure)
        } else {
            None
        };

    let result = inla_core::run_inla_inference_a_cancellable(
        &initial_theta,
        &build_prior,
        &log_prior_density,
        &obs,
        Some(&a),
        constr_opt.as_ref(),
        strategy,
        step_or_f0,
        &inla_core::MarginalOptions::default(),
        deterministic,
        None,
        build_obs_opt,
    )
    .map_err(Error::Other)?;

    let internal_marginals_hyperpar = marginals_to_r_list(&result.internal_marginals_hyperpar)?;
    let (posterior_q_i, posterior_q_p, posterior_q_x, posterior_q_n) =
        posterior_q_slots(&result.posterior_precision)?;

    Ok(list!(
        mode = result.mode,
        hessian = result.hessian,
        latent_means = result.latent_means,
        latent_variances = result.latent_variances,
        predictor_means = result.predictor_means,
        predictor_variances = result.predictor_variances,
        marginal_log_lik = result.marginal_log_lik,
        marginal_log_lik_gaussian = result.marginal_log_lik_gaussian,
        dic = result.dic,
        mean_deviance = result.mean_deviance,
        effective_params = result.effective_params,
        waic = result.waic,
        waic_lppd = result.waic_lppd,
        waic_effective_params = result.waic_effective_params,
        cpo_n_failures = result.cpo_n_failures as i32,
        node_weights = result.node_weights,
        internal_marginals_hyperpar = internal_marginals_hyperpar,
        posterior_q_i = posterior_q_i,
        posterior_q_p = posterior_q_p,
        posterior_q_x = posterior_q_x,
        posterior_q_n = posterior_q_n
    ))
}

/// Gaussian + single AR(1) via [`inla_core::ModelSpec`] → [`resolve`] → plan runner.
#[extendr]
fn inla_rs_run_gaussian_ar1_plan(
    y_obs: Vec<f64>,
    name: &str,
    obs_precision: f64,
    strategy: &str,
    step_or_f0: f64,
    initial_theta: Vec<f64>,
) -> std::result::Result<List, Error> {
    let n = y_obs.len();
    let initial = if initial_theta.is_empty() {
        None
    } else {
        Some(initial_theta)
    };
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
        initial_theta: initial,
    };
    let plan = inla_core::resolve(spec).map_err(|e| Error::Other(e.0))?;
    let result = inla_core::run_gaussian_ar1_plan(&plan, &y_obs).map_err(|e| Error::Other(e.0))?;
    let internal_marginals_hyperpar = marginals_to_r_list(&result.internal_marginals_hyperpar)?;
    let (posterior_q_i, posterior_q_p, posterior_q_x, posterior_q_n) =
        posterior_q_slots(&result.posterior_precision)?;
    Ok(list!(
        mode = result.mode,
        hessian = result.hessian,
        latent_means = result.latent_means,
        latent_variances = result.latent_variances,
        predictor_means = result.predictor_means,
        predictor_variances = result.predictor_variances,
        marginal_log_lik = result.marginal_log_lik,
        marginal_log_lik_gaussian = result.marginal_log_lik_gaussian,
        dic = result.dic,
        mean_deviance = result.mean_deviance,
        effective_params = result.effective_params,
        waic = result.waic,
        waic_lppd = result.waic_lppd,
        waic_effective_params = result.waic_effective_params,
        cpo_n_failures = result.cpo_n_failures as i32,
        node_weights = result.node_weights,
        internal_marginals_hyperpar = internal_marginals_hyperpar,
        posterior_q_i = posterior_q_i,
        posterior_q_p = posterior_q_p,
        posterior_q_x = posterior_q_x,
        posterior_q_n = posterior_q_n
    ))
}

/// Linear combinations \(v = a^\top x\) from a stored posterior precision.
#[extendr]
fn inla_rs_lincomb(
    q_i: Vec<i32>,
    q_p: Vec<i32>,
    q_x: Vec<f64>,
    q_n: i32,
    means: Vec<f64>,
    comb_names: Vec<String>,
    comb_idx: List,
    comb_weights: List,
) -> std::result::Result<List, Error> {
    let n = usize::try_from(q_n).map_err(|_| Error::Other("q_n".into()))?;
    let q = csc_from_r_slots(n, &q_i, &q_p, &q_x)?;
    if comb_names.len() != comb_idx.len() || comb_names.len() != comb_weights.len() {
        return Err(Error::Other(
            "lincomb names/idx/weights length mismatch".into(),
        ));
    }
    let mut combs = Vec::with_capacity(comb_names.len());
    for ((idx_item, wt_item), name) in comb_idx
        .values()
        .zip(comb_weights.values())
        .zip(&comb_names)
    {
        let idx: Vec<i32> = idx_item
            .as_integer_vector()
            .ok_or_else(|| Error::Other("lincomb idx must be integer".into()))?;
        let wts: Vec<f64> = wt_item
            .as_real_vector()
            .ok_or_else(|| Error::Other("lincomb weights must be numeric".into()))?;
        if idx.len() != wts.len() {
            return Err(Error::Other(format!(
                "lincomb '{name}': idx/weights length mismatch"
            )));
        }
        let weights = idx
            .iter()
            .zip(&wts)
            .map(|(&i, &w)| (i as usize, w))
            .collect();
        combs.push(inla_core::LinComb {
            name: name.clone(),
            weights,
        });
    }
    let summaries = inla_core::lincomb_summaries(&means, &q, &combs).map_err(Error::Other)?;
    let names: Vec<String> = summaries.iter().map(|s| s.name.clone()).collect();
    let mean: Vec<f64> = summaries.iter().map(|s| s.mean).collect();
    let sd: Vec<f64> = summaries.iter().map(|s| s.sd).collect();
    let q025: Vec<f64> = summaries.iter().map(|s| s.q025).collect();
    let q50: Vec<f64> = summaries.iter().map(|s| s.q50).collect();
    let q975: Vec<f64> = summaries.iter().map(|s| s.q975).collect();
    Ok(list!(
        name = names,
        mean = mean,
        sd = sd,
        q025 = q025,
        q50 = q50,
        q975 = q975
    ))
}

/// Joint latent draws from \(\mathcal{N}(\mu, Q^{-1})\).
#[extendr]
fn inla_rs_posterior_sample(
    q_i: Vec<i32>,
    q_p: Vec<i32>,
    q_x: Vec<f64>,
    q_n: i32,
    means: Vec<f64>,
    n_samples: i32,
    seed: f64,
) -> std::result::Result<Vec<f64>, Error> {
    let n = usize::try_from(q_n).map_err(|_| Error::Other("q_n".into()))?;
    let ns = usize::try_from(n_samples).map_err(|_| Error::Other("n_samples".into()))?;
    let q = csc_from_r_slots(n, &q_i, &q_p, &q_x)?;
    inla_core::sample_latent_gaussian(&means, &q, ns, seed as u64).map_err(Error::Other)
}

/// \(\mathbb{E}[g(X)]\) on a 1D marginal grid.
#[extendr]
fn inla_rs_emarginal(
    x: Vec<f64>,
    y: Vec<f64>,
    g_of_x: Vec<f64>,
) -> std::result::Result<f64, Error> {
    let m = inla_core::Marginal1D { x, y };
    inla_core::emarginal(&m, &g_of_x).map_err(Error::Other)
}

extendr_module! {
    mod inference;
    fn inla_rs_run_inla_inference;
    fn inla_rs_run_inla_structured;
    fn inla_rs_run_gaussian_ar1_plan;
    fn inla_rs_lincomb;
    fn inla_rs_posterior_sample;
    fn inla_rs_emarginal;
}
