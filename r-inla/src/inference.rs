//! Observation builders and INLA inference entry points for R.

use crate::convert::{
    csc_from_r_precision, csc_from_r_slots, marginals_to_r_list, parse_adj_list_1based,
    parse_effect_positions, posterior_q_slots,
};
use crate::mesh::parse_effect_meshes;
use extendr_api::prelude::*;
use std::sync::Arc;

/// Canonicalize likelihood family strings (R-INLA aliases → internal names).
fn canonicalize_family(family: &str) -> String {
    match family.trim().to_lowercase().as_str() {
        "exponential.surv" | "exponential_surv" => "exponential_survival".into(),
        "weibull.surv" | "weibull_surv" => "weibull_survival".into(),
        "loglogistic.surv" | "loglogistic_surv" => "loglogistic_survival".into(),
        "lognormal.surv" | "lognormal_surv" => "lognormal_survival".into(),
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
            "poisson"
            | "nbinomial"
            | "negative_binomial"
            | "zeroinflatedpoisson0"
            | "zeroinflatedpoisson1"
            | "zero_inflated_poisson"
            | "exponential"
            | "exponential_survival"
            | "weibull"
            | "weibull_survival"
            | "loglogistic"
            | "loglogistic_survival" => inla_core::Link::Log,
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

/// Host `rgeneric` callbacks owned by R (`SEXP` closures). Valid for this `.Call`.
#[derive(Clone)]
struct HostGeneric {
    q: Robj,
    log_prior: Option<Robj>,
}

// R SEXPs stay protected by the `.Call` argument list; extendr `Function::call`
// serializes via `single_threaded`.
unsafe impl Send for HostGeneric {}
unsafe impl Sync for HostGeneric {}

fn list_named(list: &List, key: &str) -> Option<Robj> {
    for (name, value) in list.iter() {
        if name == key {
            return Some(value);
        }
    }
    None
}

fn parse_rgeneric_callbacks(
    list: &List,
    n_effects: usize,
) -> std::result::Result<Vec<Option<HostGeneric>>, Error> {
    if list.is_empty() {
        return Ok(vec![None; n_effects]);
    }
    if list.len() != n_effects {
        return Err(Error::Other(format!(
            "rgeneric_callbacks length ({}) must match number of effects ({n_effects})",
            list.len()
        )));
    }
    let mut out = Vec::with_capacity(n_effects);
    for (ei, item) in list.values().enumerate() {
        if item.is_null() {
            out.push(None);
            continue;
        }
        if item.as_function().is_some() {
            out.push(Some(HostGeneric {
                q: item.clone(),
                log_prior: None,
            }));
            continue;
        }
        let sub: List = item.try_into().map_err(|e| {
            Error::Other(format!(
                "rgeneric_callbacks[{ei}] must be NULL, a Q function, or list(Q=, log.prior=): {e}"
            ))
        })?;
        if sub.is_empty() {
            out.push(None);
            continue;
        }
        let q = list_named(&sub, "Q")
            .ok_or_else(|| Error::Other(format!("rgeneric_callbacks[{ei}] missing Q function")))?;
        if q.as_function().is_none() {
            return Err(Error::Other(format!(
                "rgeneric_callbacks[{ei}]$Q must be a function(theta)"
            )));
        }
        let log_prior = list_named(&sub, "log.prior")
            .or_else(|| list_named(&sub, "log_prior"))
            .filter(|v| !v.is_null() && v.as_function().is_some());
        out.push(Some(HostGeneric { q, log_prior }));
    }
    Ok(out)
}

fn eval_rgeneric_q(
    cb: &HostGeneric,
    theta: &[f64],
    n: usize,
) -> std::result::Result<inla_core::CscMatrix, String> {
    let f =
        cb.q.as_function()
            .ok_or_else(|| "rgeneric Q is not a function".to_string())?;
    let res = f
        .call(Pairlist::from_pairs([("", r!(theta.to_vec()))]))
        .map_err(|e| format!("rgeneric Q(theta) failed: {e}"))?;
    csc_from_r_precision(&res, n).map_err(|e| e.to_string())
}

fn eval_rgeneric_log_prior(cb: &HostGeneric, theta: &[f64]) -> f64 {
    match &cb.log_prior {
        None => -0.5 * 0.1 * theta.iter().map(|v| v * v).sum::<f64>(),
        Some(fun) => {
            let Some(f) = fun.as_function() else {
                return f64::NEG_INFINITY;
            };
            match f.call(Pairlist::from_pairs([("", r!(theta.to_vec()))])) {
                Ok(res) => res
                    .as_real_vector()
                    .and_then(|v| v.first().copied())
                    .filter(|x| x.is_finite())
                    .unwrap_or(f64::NEG_INFINITY),
                Err(_) => f64::NEG_INFINITY,
            }
        }
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
    y_upper: &[f64],
    size: f64,
    zero_prob: f64,
    inflation: &str,
    alpha: f64,
    gamma: f64,
    shape: f64,
    variant: i32,
    prec: f64,
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
    let yu = pad_or_default(y_upper, n, f64::NAN);

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
                    y_upper: yu[i],
                    link,
                })
            }
            "weibull" | "weibull_survival" => {
                inla_core::Obs::WeibullSurvival(inla_core::WeibullSurvivalObs {
                    y,
                    event: ev[i],
                    y_upper: yu[i],
                    shape,
                    variant,
                    link,
                })
            }
            "loglogistic" | "loglogistic_survival" => {
                inla_core::Obs::LoglogisticSurvival(inla_core::LoglogisticSurvivalObs {
                    y,
                    event: ev[i],
                    y_upper: yu[i],
                    shape,
                    link,
                })
            }
            "lognormal" | "lognormal_survival" => {
                inla_core::Obs::LognormalSurvival(inla_core::LognormalSurvivalObs {
                    y,
                    event: ev[i],
                    y_upper: yu[i],
                    prec,
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
    y_upper: Vec<f64>,
    size: f64,
    zero_prob: f64,
    inflation: &str,
    alpha: f64,
    gamma: f64,
    shape: f64,
    variant: i32,
    prec: f64,
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
        &y_upper,
        size,
        zero_prob,
        inflation,
        alpha,
        gamma,
        shape,
        variant,
        prec,
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
                let hurst = inla_core::fgn_hurst_from_intern(theta[1]);
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
        inla_core::fgn_hurst_from_intern(result.mode[1])
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
    effect_n_main: Vec<i32>,
    effect_group_models: Vec<String>,
    effect_group_ns: Vec<i32>,
    effect_group_scales: Vec<i32>,
    adj_lists: List,
    effect_positions: List,
    prior_names: Vec<String>,
    prior_params: List,
    fixed_prec: f64,
    exposure: Vec<f64>,
    ntrials: Vec<f64>,
    event: Vec<f64>,
    y_upper: Vec<f64>,
    size: f64,
    zero_prob: f64,
    inflation: &str,
    alpha: f64,
    gamma: f64,
    shape: f64,
    variant: i32,
    prec: f64,
    deterministic: bool,
    gaussian_free_prec: bool,
    family_prior_name: &str,
    family_prior_param: Vec<f64>,
    effect_nrow: Vec<i32>,
    effect_ncol: Vec<i32>,
    effect_cyclic: Vec<i32>,
    effect_season: Vec<i32>,
    effect_layouts: Vec<String>,
    effect_meshes: List,
    dic: bool,
    waic: bool,
    cpo: bool,
    rgeneric_callbacks: List,
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
        || effect_n_main.len() != effect_types.len()
        || effect_group_models.len() != effect_types.len()
        || effect_group_ns.len() != effect_types.len()
        || effect_group_scales.len() != effect_types.len()
        || (!effect_nrow.is_empty() && effect_nrow.len() != effect_types.len())
        || (!effect_ncol.is_empty() && effect_ncol.len() != effect_types.len())
        || (!effect_cyclic.is_empty() && effect_cyclic.len() != effect_types.len())
        || (!effect_season.is_empty() && effect_season.len() != effect_types.len())
        || (!effect_layouts.is_empty() && effect_layouts.len() != effect_types.len())
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
        &y_upper,
        size,
        zero_prob,
        inflation,
        alpha,
        gamma,
        shape,
        variant,
        prec,
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
    let layouts: Vec<String> = if effect_layouts.is_empty() {
        vec!["simple".into(); effect_types_owned.len()]
    } else {
        effect_layouts
            .iter()
            .map(|s| {
                let t = s.trim().to_lowercase();
                if t.is_empty() { "simple".into() } else { t }
            })
            .collect()
    };
    let meshes = parse_effect_meshes(&effect_meshes, effect_types_owned.len())?;

    let effects: Vec<inla_core::StructuredEffect> = (0..effect_types_owned.len())
        .map(|ei| {
            let typ = effect_types_owned[ei].to_lowercase();
            let raw_order = effect_orders_i[ei];
            let slot = |v: &[i32]| v.get(ei).copied().unwrap_or(0);
            let mut nrow = usize::try_from(slot(&effect_nrow).max(0)).unwrap_or(0);
            let mut ncol = usize::try_from(slot(&effect_ncol).max(0)).unwrap_or(0);
            let mut cyclic = slot(&effect_cyclic) != 0;
            if (typ == "rw2d" || typ == "matern2d") && nrow == 0 {
                cyclic = raw_order < 0;
                nrow = raw_order.unsigned_abs() as usize;
                ncol = if nrow > 0 && effect_ns_u[ei].is_multiple_of(nrow) {
                    effect_ns_u[ei] / nrow
                } else {
                    0
                };
            }
            let season = usize::try_from(slot(&effect_season).max(0)).unwrap_or(0);
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
                order: raw_order.max(0),
                season,
                adj: adjs[ei].clone(),
                positions: positions[ei].clone(),
                crw2_layout: layouts[ei].clone(),
                nrow,
                ncol,
                cyclic,
                matern_nu: 1,
                n_main: usize::try_from(effect_n_main[ei]).unwrap_or(0),
                group_model: if effect_group_models[ei].is_empty() {
                    None
                } else {
                    Some(effect_group_models[ei].to_lowercase())
                },
                group_n: usize::try_from(effect_group_ns[ei]).unwrap_or(0),
                group_scale_model: effect_group_scales[ei] != 0,
                copy_of,
                mesh: match (
                    meshes[ei].vertices.clone(),
                    meshes[ei].triangles.clone(),
                    meshes[ei].loc_1d.clone(),
                ) {
                    (_, _, Some(loc_1d)) => Some(Box::new(inla_core::SpdeMesh {
                        vertices: vec![],
                        triangles: vec![],
                        loc_1d: Some(loc_1d),
                        barrier_triangles: meshes[ei].barrier_triangles.clone(),
                        range_fraction: meshes[ei].range_fraction,
                        diffusion: meshes[ei].diffusion,
                    })),
                    (Some(vertices), Some(triangles), None) => {
                        Some(Box::new(inla_core::SpdeMesh {
                            vertices,
                            triangles,
                            loc_1d: None,
                            barrier_triangles: meshes[ei].barrier_triangles.clone(),
                            range_fraction: meshes[ei].range_fraction,
                            diffusion: meshes[ei].diffusion,
                        }))
                    }
                    _ => None,
                },
            }
        })
        .collect();

    let callbacks = parse_rgeneric_callbacks(&rgeneric_callbacks, effects.len())?;
    for (ei, effect) in effects.iter().enumerate() {
        let is_rgeneric = effect.model_key() == "rgeneric";
        match (is_rgeneric, callbacks[ei].is_some()) {
            (true, false) => {
                return Err(Error::Other(
                    "f(..., model='rgeneric') requires a host Q callback; Rust has no built-in rgeneric Q".into(),
                ));
            }
            (false, true) => {
                return Err(Error::Other(format!(
                    "host Q callback supplied for non-rgeneric effect '{}'",
                    effect.model_key()
                )));
            }
            _ => {}
        }
    }
    let has_host_q = callbacks.iter().any(|c| c.is_some());
    let callbacks = Arc::new(callbacks);

    let constr_opt = if has_host_q {
        let full_n: usize = effects.iter().map(|e| e.n).sum();
        let mut stacked: Option<inla_core::ConstraintSpec> = None;
        let mut offset = 0usize;
        for (ei, effect) in effects.iter().enumerate() {
            let n_e = effect.n;
            if callbacks[ei].is_some() {
                offset += n_e;
                continue;
            }
            if let Some(block) = inla_core::structured_constraints(std::slice::from_ref(effect))
                .map_err(Error::Other)?
            {
                let embedded = block.embed(full_n, offset).map_err(Error::Other)?;
                stacked = Some(match stacked {
                    None => embedded,
                    Some(prev) => prev.vstack(&embedded).map_err(Error::Other)?,
                });
            }
            offset += n_e;
        }
        stacked
    } else {
        inla_core::structured_constraints(&effects).map_err(Error::Other)?
    };

    let effects_for_q = effects.clone();
    let callbacks_q = Arc::clone(&callbacks);
    let build_prior = move |theta: &[f64]| -> std::result::Result<inla_core::CscMatrix, String> {
        let latent_th = if gaussian_free_prec {
            if theta.is_empty() { &[] } else { &theta[1..] }
        } else {
            theta
        };
        if !has_host_q {
            return inla_core::build_structured_precision(&effects_for_q, latent_th, fixed_prec);
        }
        let mut blocks = Vec::with_capacity(effects_for_q.len());
        let mut off = 0usize;
        for (ei, effect) in effects_for_q.iter().enumerate() {
            let tlen = effect.theta_len;
            if off + tlen > latent_th.len() {
                return Err(format!(
                    "theta length {} too short for rgeneric/structured mix (need at least {})",
                    latent_th.len(),
                    off + tlen
                ));
            }
            let ti = &latent_th[off..off + tlen];
            off += tlen;
            if let Some(cb) = &callbacks_q[ei] {
                blocks.push(eval_rgeneric_q(cb, ti, effect.n)?);
            } else {
                blocks.push(inla_core::build_structured_precision(
                    std::slice::from_ref(effect),
                    ti,
                    fixed_prec,
                )?);
            }
        }
        if off != latent_th.len() {
            return Err(format!(
                "theta length {} != consumed latent theta {off}",
                latent_th.len()
            ));
        }
        inla_core::block_diag_csc(&blocks)
    };

    let log_prior_density = {
        let structured_effects: Vec<inla_core::StructuredEffect> = effects
            .iter()
            .enumerate()
            .filter(|(i, _)| callbacks[*i].is_none())
            .map(|(_, e)| e.clone())
            .collect();
        let structured_theta_len: usize = structured_effects.iter().map(|e| e.theta_len).sum();
        let stack = if prior_names.is_empty() {
            if structured_effects.is_empty() {
                inla_core::HyperPriorStack::new(Vec::new())
            } else {
                inla_core::structured_prior_stack(&structured_effects).map_err(Error::Other)?
            }
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
            if stack.theta_dim() != structured_theta_len {
                return Err(Error::Other(format!(
                    "prior theta dimension {} != structured latent theta dimension {structured_theta_len}",
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
        let effects_lp = effects.clone();
        let callbacks_lp = Arc::clone(&callbacks);
        move |theta: &[f64]| -> f64 {
            let mut lp = 0.0;
            let latent_th = if gaussian_free_prec {
                if theta.is_empty() {
                    return f64::NEG_INFINITY;
                }
                lp += fam_prior
                    .log_density(&theta[..1])
                    .unwrap_or(f64::NEG_INFINITY);
                &theta[1..]
            } else {
                theta
            };
            if !has_host_q {
                return lp + stack.log_density(latent_th).unwrap_or(f64::NEG_INFINITY);
            }
            let mut off = 0usize;
            let mut structured_th = Vec::with_capacity(structured_theta_len);
            for (ei, effect) in effects_lp.iter().enumerate() {
                let tlen = effect.theta_len;
                if off + tlen > latent_th.len() {
                    return f64::NEG_INFINITY;
                }
                let ti = &latent_th[off..off + tlen];
                off += tlen;
                if let Some(cb) = &callbacks_lp[ei] {
                    lp += eval_rgeneric_log_prior(cb, ti);
                } else if tlen > 0 {
                    structured_th.extend_from_slice(ti);
                }
            }
            lp + stack
                .log_density(&structured_th)
                .unwrap_or(f64::NEG_INFINITY)
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

    let compute = inla_core::ComputeOptions {
        strategy: strategy.to_string(),
        step_or_f0,
        deterministic: deterministic || has_host_q,
        dic,
        waic,
        cpo,
        ..inla_core::ComputeOptions::default()
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
        deterministic || has_host_q,
        None,
        build_obs_opt,
        Some(&compute),
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
            ..Default::default()
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
