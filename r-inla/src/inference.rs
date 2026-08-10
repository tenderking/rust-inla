//! Observation builders and INLA inference entry points for R.

use crate::convert::{marginals_to_r_list, parse_adj_list_1based, scale_csc_entries};
use extendr_api::prelude::*;

fn parse_link(link: &str, family: &str) -> std::result::Result<inla_core::Link, Error> {
    let link = link.trim().to_lowercase();
    if link.is_empty() || link == "default" {
        return Ok(match family {
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
    let fam = family.trim().to_lowercase();
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
            "nbinomial" | "negative_binomial" => {
                inla_core::Obs::NegativeBinomial(inla_core::NegativeBinomialObs {
                    y,
                    exposure: e[i],
                    size,
                    link,
                })
            }
            "zeroinflatedpoisson0" | "zero_inflated_poisson" => {
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
            "zeroinflatedbinomial0" | "zero_inflated_binomial" => {
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
    let family_str = family.trim().to_lowercase();
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

    let prior_stack = match inla_core::HyperPriorStack::default_for_effect(&model_type_str) {
        Ok(s) => s,
        Err(_) => inla_core::HyperPriorStack::new(vec![inla_core::PriorSpec::gaussian(0.0, 0.1)]),
    };
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
        hurst = hurst_est,
        order = order
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
    adj_lists: List,
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

    let rows: Vec<usize> = a_i.iter().map(|&v| v as usize).collect();
    let cols: Vec<usize> = a_j.iter().map(|&v| v as usize).collect();
    let a = inla_core::csc_from_triplets_0based(a_nrow_u, a_ncol_u, &rows, &cols, &a_x)
        .map_err(Error::Other)?;

    let family_str = family.trim().to_lowercase();
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
    let effect_orders_u: Vec<usize> = effect_orders
        .iter()
        .map(|&v| usize::try_from(v).unwrap_or(0))
        .collect();
    let effect_orders_i: Vec<i32> = effect_orders.clone();
    let expected_theta: usize = effect_theta_lens_u.iter().sum();
    if initial_theta.len() != expected_theta {
        return Err(Error::Other(format!(
            "initial_theta length {} != sum(effect_theta_lens) {}",
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

    // Hard sum-to-zero on intrinsic random fields (before moving effect_* into build_prior).
    let constr_opt = {
        let mut stacked: Option<inla_core::ConstraintSpec> = None;
        let mut offset = 0usize;
        for (ei, typ) in effect_types.iter().enumerate() {
            let n_e = effect_ns_u[ei];
            let mut k = inla_core::model_rank_deficiency(typ);
            if typ.eq_ignore_ascii_case("rw2d") || typ.eq_ignore_ascii_case("matern2d") {
                // effect_orders: ±nrow (negative ⇒ cyclic). matern2d is proper → k=0.
                if typ.eq_ignore_ascii_case("rw2d") {
                    let raw = effect_orders[ei];
                    k = if raw < 0 { 1 } else { 2 };
                } else {
                    k = 0;
                }
            }
            if typ.eq_ignore_ascii_case("bym") {
                // Constrain spatial block only (first n = adj.len()).
                let n_sp = adjs
                    .get(ei)
                    .and_then(|a| a.as_ref())
                    .map(|a| a.len())
                    .unwrap_or(n_e / 2);
                let block = inla_core::sum_to_zero_constraint(n_sp, 1).map_err(Error::Other)?;
                let embedded = block.embed(a_ncol_u, offset).map_err(Error::Other)?;
                stacked = Some(match stacked {
                    None => embedded,
                    Some(prev) => prev.vstack(&embedded).map_err(Error::Other)?,
                });
                offset += n_e;
                continue;
            }
            if k > 0 {
                let block = inla_core::sum_to_zero_constraint(n_e, k).map_err(Error::Other)?;
                let embedded = block.embed(a_ncol_u, offset).map_err(Error::Other)?;
                stacked = Some(match stacked {
                    None => embedded,
                    Some(prev) => prev.vstack(&embedded).map_err(Error::Other)?,
                });
            }
            offset += n_e;
        }
        stacked
    };

    let build_prior = move |theta: &[f64]| -> std::result::Result<inla_core::CscMatrix, String> {
        let mut blocks = Vec::with_capacity(effect_types_owned.len());
        let mut off = 0usize;
        for ei in 0..effect_types_owned.len() {
            let typ = effect_types_owned[ei].to_lowercase();
            let n_e = effect_ns_u[ei];
            let tlen = effect_theta_lens_u[ei];
            let th = if tlen == 0 {
                &[][..]
            } else {
                &theta[off..off + tlen]
            };
            off += tlen;

            let q = match typ.as_str() {
                "fixed" => inla_core::identity_csc(n_e, fixed_prec)?,
                "iid" => {
                    let tau = th.first().copied().unwrap_or(0.0).exp();
                    let q0 = inla_core::iid_precision_csc(n_e, 1.0)?;
                    let q0 = if effect_scales_b[ei] {
                        inla_core::scale_model_csc(&q0)?
                    } else {
                        q0
                    };
                    scale_csc_entries(&q0, tau)?
                }
                "rw1" => {
                    let tau = th.first().copied().unwrap_or(0.0).exp();
                    let q0 = inla_core::rw1_precision_csc(n_e, 1.0)?;
                    let q0 = if effect_scales_b[ei] {
                        inla_core::scale_model_csc(&q0)?
                    } else {
                        q0
                    };
                    scale_csc_entries(&q0, tau)?
                }
                "rw2" => {
                    let tau = th.first().copied().unwrap_or(0.0).exp();
                    let q0 = inla_core::rw2_precision_csc(n_e, 1.0)?;
                    let q0 = if effect_scales_b[ei] {
                        inla_core::scale_model_csc(&q0)?
                    } else {
                        q0
                    };
                    scale_csc_entries(&q0, tau)?
                }
                "rw2d" => {
                    let tau = th.first().copied().unwrap_or(0.0).exp();
                    // effect_orders encodes ±nrow (negative ⇒ cyclic); ncol = n_e / nrow.
                    let raw = effect_orders_i[ei];
                    let cyclic = raw < 0;
                    let nrow = raw.unsigned_abs() as usize;
                    if nrow == 0 || !n_e.is_multiple_of(nrow) {
                        return Err(format!(
                            "rw2d: effect_orders (±nrow)={raw} incompatible with n={n_e}"
                        ));
                    }
                    let ncol = n_e / nrow;
                    let q0 = inla_core::rw2d_precision_csc(nrow, ncol, 1.0, cyclic, false)?;
                    scale_csc_entries(&q0, tau)?
                }
                "matern2d" => {
                    if th.len() < 2 {
                        return Err("matern2d needs theta=[log_prec, log_range]".into());
                    }
                    let prec = th[0].exp();
                    let range = th[1].exp();
                    let raw = effect_orders_i[ei];
                    let cyclic = raw < 0;
                    let nrow = raw.unsigned_abs() as usize;
                    if nrow == 0 || !n_e.is_multiple_of(nrow) {
                        return Err(format!(
                            "matern2d: effect_orders (±nrow)={raw} incompatible with n={n_e}"
                        ));
                    }
                    let ncol = n_e / nrow;
                    inla_core::matern2d_precision_csc(nrow, ncol, 1, range, prec, cyclic)?
                }
                "bym" => {
                    if th.len() < 2 {
                        return Err("bym needs theta=[log_tau_spatial, log_tau_iid]".into());
                    }
                    let adj = adjs[ei]
                        .as_ref()
                        .ok_or_else(|| "bym missing adj".to_string())?;
                    if adj.len() * 2 != n_e {
                        return Err(format!(
                            "bym adj length {} ⇒ latent {}, got n={}",
                            adj.len(),
                            adj.len() * 2,
                            n_e
                        ));
                    }
                    inla_core::bym_precision_csc(adj, th[0].exp(), th[1].exp())?
                }
                "bym2" => {
                    if th.len() < 2 {
                        return Err("bym2 needs theta=[log_tau, logit_phi]".into());
                    }
                    let adj = adjs[ei]
                        .as_ref()
                        .ok_or_else(|| "bym2 missing adj".to_string())?;
                    if adj.len() != n_e {
                        return Err(format!("bym2 adj length {} != effect n {}", adj.len(), n_e));
                    }
                    let tau = th[0].exp();
                    let phi = 1.0 / (1.0 + (-th[1]).exp());
                    let phi = phi.clamp(1e-6, 1.0 - 1e-6);
                    inla_core::bym2_precision_csc(adj, tau, phi)?
                }
                "seasonal" => {
                    let tau = th.first().copied().unwrap_or(0.0).exp();
                    let season = if effect_orders_u[ei] > 0 {
                        effect_orders_u[ei]
                    } else {
                        4
                    };
                    let q0 = inla_core::seasonal_precision_csc(n_e, season, 1.0, true)?;
                    let q0 = if effect_scales_b[ei] {
                        inla_core::scale_model_csc(&q0)?
                    } else {
                        q0
                    };
                    scale_csc_entries(&q0, tau)?
                }
                "ar1" => {
                    if th.len() < 2 {
                        return Err("ar1 needs theta=[log_tau, logit_rho]".into());
                    }
                    let tau = th[0].exp();
                    let rho = 2.0 / (1.0 + (-th[1]).exp()) - 1.0;
                    let q0 = inla_core::ar1_precision_csc(n_e, rho, 1.0)?;
                    let q0 = if effect_scales_b[ei] {
                        inla_core::scale_model_csc(&q0)?
                    } else {
                        q0
                    };
                    scale_csc_entries(&q0, tau)?
                }
                "ar" | "arp" => {
                    let tau = th.first().copied().unwrap_or(0.0).exp();
                    let pacf: Vec<f64> = if th.len() > 1 {
                        th[1..].iter().map(|&v| (v * 0.5).tanh()).collect()
                    } else {
                        vec![0.0]
                    };
                    let q0 = inla_core::arp_precision_csc(n_e, &pacf, 1.0)?;
                    let q0 = if effect_scales_b[ei] {
                        inla_core::scale_model_csc(&q0)?
                    } else {
                        q0
                    };
                    scale_csc_entries(&q0, tau)?
                }
                "crw1" => {
                    let tau = th.first().copied().unwrap_or(0.0).exp();
                    let positions: Vec<f64> = (0..n_e).map(|i| i as f64).collect();
                    let q0 = inla_core::crw1_precision_csc(&positions, 1.0)?;
                    let q0 = if effect_scales_b[ei] {
                        inla_core::scale_model_csc(&q0)?
                    } else {
                        q0
                    };
                    scale_csc_entries(&q0, tau)?
                }
                "crw2" => {
                    let tau = th.first().copied().unwrap_or(0.0).exp();
                    let positions: Vec<f64> = (0..n_e).map(|i| i as f64).collect();
                    let q0 = inla_core::crw2_precision_csc(&positions, 1.0, "simple")?;
                    let q0 = if effect_scales_b[ei] {
                        inla_core::scale_model_csc(&q0)?
                    } else {
                        q0
                    };
                    scale_csc_entries(&q0, tau)?
                }
                "besag" => {
                    let tau = th.first().copied().unwrap_or(0.0).exp();
                    let adj = adjs[ei]
                        .as_ref()
                        .ok_or_else(|| "besag missing adj".to_string())?;
                    if adj.len() != n_e {
                        return Err(format!(
                            "besag adj length {} != effect n {}",
                            adj.len(),
                            n_e
                        ));
                    }
                    // Scale the τ=1 structure first; then apply τ (otherwise
                    // scale.model cancels the precision hyperparameter).
                    let q0 = inla_core::besag_precision_csc(adj, 1.0)?;
                    let q0 = if effect_scales_b[ei] {
                        inla_core::scale_model_csc(&q0)?
                    } else {
                        q0
                    };
                    scale_csc_entries(&q0, tau)?
                }
                "fgn" => {
                    let order = effect_orders_u[ei];
                    if order == 3 || order == 4 {
                        if th.len() < 2 {
                            return Err("fgn approx needs [log_tau, H_intern]".into());
                        }
                        let tau = th[0].exp();
                        let hurst = inla_core::fgn_hurst_from_intern(th[1]);
                        let n_time = n_e / (order + 1);
                        let q0 =
                            inla_core::fgn_approx_precision_csc(n_time, hurst, 1.0, order, 1e8)?;
                        let q0 = if effect_scales_b[ei] {
                            inla_core::scale_model_csc(&q0)?
                        } else {
                            q0
                        };
                        scale_csc_entries(&q0, tau)?
                    } else {
                        if th.len() < 2 {
                            return Err("fgn needs [log_tau, logit_H]".into());
                        }
                        let tau = th[0].exp();
                        let hurst = 1.0 / (1.0 + (-th[1]).exp());
                        let q0 = inla_core::fgn_precision_csc(n_e, hurst, 1.0)?;
                        let q0 = if effect_scales_b[ei] {
                            inla_core::scale_model_csc(&q0)?
                        } else {
                            q0
                        };
                        scale_csc_entries(&q0, tau)?
                    }
                }
                other => return Err(format!("unsupported effect type: {other}")),
            };
            let _ = effect_scales_b[ei]; // scaling already applied above for random effects
            blocks.push(q);
        }
        inla_core::block_diag_csc(&blocks)
    };

    let log_prior_density = {
        let mut priors = Vec::new();
        for typ in &effect_types {
            let m = typ.to_lowercase();
            if m == "fixed" {
                continue;
            }
            match inla_core::HyperPriorStack::default_for_effect(&m) {
                Ok(s) => priors.extend(s.priors),
                Err(_) => priors.push(inla_core::PriorSpec::gaussian(0.0, 0.1)),
            }
        }
        let stack = inla_core::HyperPriorStack::new(priors);
        move |theta: &[f64]| -> f64 { stack.log_density(theta).unwrap_or(f64::NEG_INFINITY) }
    };

    let result = inla_core::run_inla_inference_a(
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
    )
    .map_err(Error::Other)?;

    let internal_marginals_hyperpar = marginals_to_r_list(&result.internal_marginals_hyperpar)?;

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
        cpo_n_failures = result.cpo_n_failures as i32,
        node_weights = result.node_weights,
        internal_marginals_hyperpar = internal_marginals_hyperpar
    ))
}

extendr_module! {
    mod inference;
    fn inla_rs_run_inla_inference;
    fn inla_rs_run_inla_structured;
}
