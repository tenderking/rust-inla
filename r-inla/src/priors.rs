//! Hyperprior helpers exported to R.

use extendr_api::prelude::*;

/// Evaluate named prior on internal θ. `param` may be empty (use defaults).
#[extendr]
fn inla_rs_prior_log_density(
    name: &str,
    param: Vec<f64>,
    theta: Vec<f64>,
) -> std::result::Result<f64, Error> {
    let spec = inla_core::PriorSpec::from_name_params(name, &param).map_err(Error::Other)?;
    spec.log_density(&theta).map_err(Error::Other)
}

/// Default hyperpriors for an effect model: list(names=..., params=list of numeric vectors).
#[extendr]
fn inla_rs_default_hyper_priors(model: &str) -> std::result::Result<List, Error> {
    let stack = inla_core::HyperPriorStack::default_for_effect(model).map_err(Error::Other)?;
    let pairs = stack.to_names_params();
    let names: Vec<String> = pairs.iter().map(|(n, _)| n.clone()).collect();
    let mut param_items: Vec<Robj> = Vec::with_capacity(pairs.len());
    for (_, p) in &pairs {
        param_items.push(Robj::from(p.clone()));
    }
    Ok(list!(
        names = names,
        params = List::from_values(param_items)
    ))
}

/// Sum log-density for a prior stack. `param_list` is a list of numeric vectors.
#[extendr]
fn inla_rs_hyper_prior_stack_log_density(
    names: Vec<String>,
    param_list: List,
    theta: Vec<f64>,
) -> std::result::Result<f64, Error> {
    if param_list.len() != names.len() {
        return Err(Error::Other(
            "param_list length must match names length".into(),
        ));
    }
    let mut params: Vec<Vec<f64>> = Vec::with_capacity(names.len());
    for item in param_list.values() {
        let v: Vec<f64> = item
            .as_real_vector()
            .ok_or_else(|| Error::Other("each params entry must be numeric".into()))?
            .to_vec();
        params.push(v);
    }
    let stack =
        inla_core::HyperPriorStack::from_names_params(&names, &params).map_err(Error::Other)?;
    stack.log_density(&theta).map_err(Error::Other)
}

extendr_module! {
    mod priors;
    fn inla_rs_prior_log_density;
    fn inla_rs_default_hyper_priors;
    fn inla_rs_hyper_prior_stack_log_density;
}
