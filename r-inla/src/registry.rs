//! Model metadata registry + control option resolution exposed to R.
//!
//! R must not keep its own θ-length / default-θ / label tables; it calls these.

use extendr_api::prelude::*;

/// Per-model metadata: θ length, defaults, rank deficiency, hyper labels/transforms.
///
/// `group_model = ""` means no `control.group`.
#[extendr]
fn inla_rs_model_metadata(
    model: &str,
    order: i32,
    group_model: &str,
    cyclic: bool,
) -> std::result::Result<List, Error> {
    let order_u = usize::try_from(order.max(0)).unwrap_or(0);
    let group = if group_model.trim().is_empty() {
        None
    } else {
        Some(group_model)
    };
    let meta = inla_core::model_metadata(model, order_u, group, cyclic).map_err(Error::Other)?;

    let internal: Vec<String> = meta
        .hyper
        .iter()
        .map(|h| h.internal_label.clone())
        .collect();
    let labels: Vec<String> = meta.hyper.iter().map(|h| h.label.clone()).collect();
    let transforms: Vec<String> = meta
        .hyper
        .iter()
        .map(|h| h.transform_tag().to_string())
        .collect();
    let prior_names: Vec<String> = meta.default_priors.iter().map(|(n, _)| n.clone()).collect();
    let prior_params = List::from_values(
        meta.default_priors
            .iter()
            .map(|(_, p)| r!(p.clone()))
            .collect::<Vec<Robj>>(),
    );

    Ok(list!(
        model = meta.model,
        theta_len = meta.theta_len as i32,
        default_theta = meta.default_theta,
        rank_deficiency = meta.rank_deficiency as i32,
        default_scale_model = meta.default_scale_model,
        hyper_internal = internal,
        hyper_labels = labels,
        hyper_transforms = transforms,
        prior_names = prior_names,
        prior_params = prior_params
    ))
}

/// Latent model names accepted by the structured path.
#[extendr]
fn inla_rs_supported_models() -> Vec<String> {
    inla_core::SUPPORTED_MODELS
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// Validate + fill defaults for a named control list (`list(waic = TRUE, ...)`).
///
/// Unknown names are an error, so R and Python cannot drift apart silently.
#[extendr]
fn inla_rs_resolve_compute_options(controls: List) -> std::result::Result<List, Error> {
    let mut pairs: Vec<(String, inla_core::OptionValue)> = Vec::new();
    for (name, value) in controls.iter() {
        if name.is_empty() {
            return Err(Error::Other(
                "control options must be a named list".to_string(),
            ));
        }
        pairs.push((name.to_string(), robj_to_option_value(name, &value)?));
    }
    let opts = inla_core::resolve_compute_options(&pairs).map_err(Error::Other)?;

    Ok(list!(
        strategy = opts.strategy,
        step_or_f0 = opts.step_or_f0,
        deterministic = opts.deterministic,
        fixed_prec = opts.fixed_prec,
        dic = opts.dic,
        waic = opts.waic,
        cpo = opts.cpo,
        return_marginals_latent = selection_to_r(&opts.return_marginals_latent),
        return_marginals_predictor = selection_to_r(&opts.return_marginals_predictor)
    ))
}

fn selection_to_r(sel: &inla_core::IndexSelection) -> Robj {
    match sel {
        inla_core::IndexSelection::None => r!(false),
        inla_core::IndexSelection::All => r!(true),
        inla_core::IndexSelection::Some(idx) => {
            r!(idx.iter().map(|&i| i as i32).collect::<Vec<i32>>())
        }
    }
}

fn robj_to_option_value(
    name: &str,
    value: &Robj,
) -> std::result::Result<inla_core::OptionValue, Error> {
    if value.is_logical() {
        let v: Vec<Rbool> = value
            .as_logical_vector()
            .ok_or_else(|| Error::Other(format!("control '{name}': bad logical")))?;
        let first = v
            .first()
            .ok_or_else(|| Error::Other(format!("control '{name}': empty logical")))?;
        return Ok(inla_core::OptionValue::Bool(first.is_true()));
    }
    if value.is_string() {
        let s: String = value
            .as_str()
            .ok_or_else(|| Error::Other(format!("control '{name}': bad string")))?
            .to_string();
        return Ok(inla_core::OptionValue::Text(s));
    }
    if value.is_number() {
        let nums: Vec<f64> = value
            .as_real_vector()
            .ok_or_else(|| Error::Other(format!("control '{name}': bad numeric")))?;
        if nums.len() == 1 {
            return Ok(inla_core::OptionValue::Num(nums[0]));
        }
        return Ok(inla_core::OptionValue::Nums(nums));
    }
    Err(Error::Other(format!(
        "control '{name}': unsupported type (use logical, numeric, or character)"
    )))
}

extendr_module! {
    mod registry;
    fn inla_rs_model_metadata;
    fn inla_rs_supported_models;
    fn inla_rs_resolve_compute_options;
}
