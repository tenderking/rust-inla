//! Language-neutral model IR (Option A: lives in `inla_stats`, re-exported by `inla_core`).
//!
//! Flow: host skins → [`ModelSpec`] → [`resolve`] → [`ModelPlan`] → engine.
//! Below [`ModelPlan`] there is no “what did the user mean?” — only executable semantics.
//!
//! R/Python concepts (formula, `NULL`/`None`, data frames) must not appear here.

use crate::ar1::ar1_precision_csc;
use crate::inference::{GaussianObs, InferenceResult, Link, Obs, run_inla_inference};
use crate::priors::{HyperPriorStack, PriorSpec};

/// Errors while validating or resolving a [`ModelSpec`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanError(pub String);

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for PlanError {}

impl From<String> for PlanError {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for PlanError {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Maps one internal θ coordinate to a natural-scale summary quantity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HyperTransformKind {
    /// τ = exp(θ) (precision).
    Exp,
    /// ρ = 2/(1+e^{-θ}) − 1 (AR1 / `pc.cor1` scale).
    RhoCor1,
    /// φ = 1/(1+e^{-θ}) (e.g. BYM2 mixing).
    Phi,
    /// Leave θ unchanged.
    Identity,
}

impl HyperTransformKind {
    pub fn to_natural(self, theta: f64) -> f64 {
        match self {
            Self::Exp => theta.exp(),
            Self::RhoCor1 => 2.0 / (1.0 + (-theta).exp()) - 1.0,
            Self::Phi => 1.0 / (1.0 + (-theta).exp()),
            Self::Identity => theta,
        }
    }

    /// Delta-method sd for a Gaussian approx on the internal scale.
    pub fn natural_sd(self, theta_mean: f64, theta_sd: f64) -> f64 {
        if !theta_sd.is_finite() {
            return f64::NAN;
        }
        match self {
            Self::Exp => theta_mean.exp() * theta_sd,
            Self::RhoCor1 => {
                let r = self.to_natural(theta_mean);
                0.5 * (1.0 - r * r) * theta_sd
            }
            Self::Phi => {
                let p = self.to_natural(theta_mean);
                p * (1.0 - p) * theta_sd
            }
            Self::Identity => theta_sd,
        }
    }
}

/// One hyperparameter slot after resolve (ordering matches concatenated θ).
#[derive(Debug, Clone, PartialEq)]
pub struct HyperSlotPlan {
    pub internal_name: String,
    pub natural_name: String,
    pub transform: HyperTransformKind,
    pub prior: PriorSpec,
}

/// Observation family as requested (no response vector — that is a buffer).
#[derive(Debug, Clone, PartialEq)]
pub enum LikelihoodSpec {
    /// Homoscedastic Gaussian with fixed observation precision.
    Gaussian {
        /// If `None`, resolve uses a default (large) precision.
        precision: Option<f64>,
    },
}

/// Fully resolved observation family.
#[derive(Debug, Clone, PartialEq)]
pub enum LikelihoodPlan {
    Gaussian { precision: f64 },
}

/// A single latent effect as requested (language-neutral).
#[derive(Debug, Clone, PartialEq)]
pub enum LatentEffectSpec {
    /// Stationary AR(1); θ = [log τ, logit((1+ρ)/2)].
    Ar1 {
        name: String,
        n: usize,
        /// Optional overrides as `(prior_name, param)` per θ slot; `None` ⇒ model defaults.
        priors: Option<Vec<(String, Vec<f64>)>>,
    },
}

/// Resolved latent effect.
#[derive(Debug, Clone, PartialEq)]
pub enum LatentEffectPlan {
    Ar1 {
        name: String,
        n: usize,
        hyper: Vec<HyperSlotPlan>,
        /// Internal-scale starting values for Nelder–Mead / CCD.
        initial_theta: Vec<f64>,
    },
}

impl LatentEffectPlan {
    pub fn name(&self) -> &str {
        match self {
            Self::Ar1 { name, .. } => name,
        }
    }

    pub fn latent_len(&self) -> usize {
        match self {
            Self::Ar1 { n, .. } => *n,
        }
    }

    pub fn theta_dim(&self) -> usize {
        match self {
            Self::Ar1 { hyper, .. } => hyper.len(),
        }
    }

    pub fn prior_stack(&self) -> HyperPriorStack {
        match self {
            Self::Ar1 { hyper, .. } => {
                HyperPriorStack::new(hyper.iter().map(|h| h.prior.clone()).collect())
            }
        }
    }
}

/// Engine / integration knobs as requested.
#[derive(Debug, Clone, PartialEq)]
pub struct ComputationSpec {
    /// `"ccd"` or `"grid"` (and future strategies).
    pub strategy: Option<String>,
    pub step_or_f0: Option<f64>,
}

impl Default for ComputationSpec {
    fn default() -> Self {
        Self {
            strategy: None,
            step_or_f0: None,
        }
    }
}

/// Resolved computation settings.
#[derive(Debug, Clone, PartialEq)]
pub struct ComputationPlan {
    pub strategy: String,
    pub step_or_f0: f64,
}

/// Index range of one named block in the stacked latent field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatentBlockLayout {
    pub name: String,
    pub start: usize,
    pub len: usize,
}

/// How `latent_means` / variances are stacked for this plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatentLayout {
    pub blocks: Vec<LatentBlockLayout>,
    pub total_len: usize,
}

/// Language-neutral **requested** model (still may contain `None` = “use default”).
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSpec {
    pub likelihood: LikelihoodSpec,
    pub effects: Vec<LatentEffectSpec>,
    pub computation: ComputationSpec,
    /// Optional override for concatenated internal θ₀ (must match resolved θ dim).
    pub initial_theta: Option<Vec<f64>>,
}

/// Fully resolved, executable IR. No host-language types.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelPlan {
    pub likelihood: LikelihoodPlan,
    pub effects: Vec<LatentEffectPlan>,
    pub computation: ComputationPlan,
    pub layout: LatentLayout,
    /// Concatenated initial θ across effects (engine order).
    pub initial_theta: Vec<f64>,
}

impl ModelPlan {
    pub fn prior_stack(&self) -> HyperPriorStack {
        let mut priors = Vec::new();
        for e in &self.effects {
            priors.extend(e.prior_stack().priors);
        }
        HyperPriorStack::new(priors)
    }

    pub fn hyper_slots(&self) -> Vec<&HyperSlotPlan> {
        let mut out = Vec::new();
        for e in &self.effects {
            match e {
                LatentEffectPlan::Ar1 { hyper, .. } => out.extend(hyper.iter()),
            }
        }
        out
    }
}

/// Validate + fill statistical/engine defaults → [`ModelPlan`].
pub fn resolve(spec: ModelSpec) -> Result<ModelPlan, PlanError> {
    validate_spec(&spec)?;

    let likelihood = match spec.likelihood {
        LikelihoodSpec::Gaussian { precision } => {
            let precision = precision.unwrap_or(100.0);
            if !(precision > 0.0 && precision.is_finite()) {
                return Err("Gaussian observation precision must be finite and > 0".into());
            }
            LikelihoodPlan::Gaussian { precision }
        }
    };

    let mut effects = Vec::with_capacity(spec.effects.len());
    let mut layout_blocks = Vec::with_capacity(spec.effects.len());
    let mut initial_theta = Vec::new();
    let mut offset = 0usize;

    for effect in spec.effects {
        let plan = resolve_effect(effect)?;
        layout_blocks.push(LatentBlockLayout {
            name: plan.name().to_string(),
            start: offset,
            len: plan.latent_len(),
        });
        offset += plan.latent_len();
        match &plan {
            LatentEffectPlan::Ar1 {
                initial_theta: th, ..
            } => initial_theta.extend_from_slice(th),
        }
        effects.push(plan);
    }

    let computation = ComputationPlan {
        strategy: spec
            .computation
            .strategy
            .unwrap_or_else(|| "ccd".to_string()),
        step_or_f0: spec.computation.step_or_f0.unwrap_or(1.0),
    };
    if computation.strategy != "ccd" && computation.strategy != "grid" {
        return Err(format!(
            "unsupported integration strategy '{}'",
            computation.strategy
        )
        .into());
    }

    if let Some(th) = spec.initial_theta {
        if th.len() != initial_theta.len() {
            return Err(format!(
                "initial_theta length {} != expected {}",
                th.len(),
                initial_theta.len()
            )
            .into());
        }
        if th.iter().any(|v| !v.is_finite()) {
            return Err("initial_theta must be finite".into());
        }
        initial_theta = th;
    }

    Ok(ModelPlan {
        likelihood,
        effects,
        computation,
        layout: LatentLayout {
            blocks: layout_blocks,
            total_len: offset,
        },
        initial_theta,
    })
}

fn validate_spec(spec: &ModelSpec) -> Result<(), PlanError> {
    if spec.effects.is_empty() {
        return Err("ModelSpec must contain at least one latent effect".into());
    }
    // v1: single AR(1) only (identity η = x).
    if spec.effects.len() != 1 {
        return Err(
            "ModelSpec v1 supports exactly one latent effect (AR1); multi-effect comes later"
                .into(),
        );
    }
    match &spec.effects[0] {
        LatentEffectSpec::Ar1 { n, name, .. } => {
            if name.is_empty() {
                return Err("latent effect name must be non-empty".into());
            }
            if *n < 2 {
                return Err("AR1 requires n >= 2".into());
            }
        }
    }
    Ok(())
}

fn resolve_effect(spec: LatentEffectSpec) -> Result<LatentEffectPlan, PlanError> {
    match spec {
        LatentEffectSpec::Ar1 { name, n, priors } => {
            let stack = match priors {
                None => HyperPriorStack::default_for_effect("ar1")?,
                Some(pairs) => {
                    if pairs.len() != 2 {
                        return Err(
                            "AR1 priors override must have 2 slots (prec, rho)".into(),
                        );
                    }
                    let mut specs = Vec::with_capacity(2);
                    for (nm, param) in &pairs {
                        specs.push(PriorSpec::from_name_params(nm, param)?);
                    }
                    HyperPriorStack::new(specs)
                }
            };
            if stack.theta_dim() != 2 {
                return Err("AR1 prior stack must have θ dimension 2".into());
            }
            let hyper = vec![
                HyperSlotPlan {
                    internal_name: format!("log_precision:{name}"),
                    natural_name: format!("Precision for {name}"),
                    transform: HyperTransformKind::Exp,
                    prior: stack.priors[0].clone(),
                },
                HyperSlotPlan {
                    internal_name: format!("logit_rho:{name}"),
                    natural_name: format!("Rho for {name}"),
                    transform: HyperTransformKind::RhoCor1,
                    prior: stack.priors[1].clone(),
                },
            ];
            Ok(LatentEffectPlan::Ar1 {
                name,
                n,
                hyper,
                // Mild start: τ≈1, ρ≈0
                initial_theta: vec![0.0, 0.0],
            })
        }
    }
}

/// Run inference for a v1 plan: one AR(1) + Gaussian observations, η = x.
///
/// `y.len()` must equal the AR(1) length. Observation buffers stay outside the plan.
pub fn run_gaussian_ar1_plan(
    plan: &ModelPlan,
    y: &[f64],
) -> Result<InferenceResult, PlanError> {
    let (n, _name) = match plan.effects.as_slice() {
        [LatentEffectPlan::Ar1 { n, name, .. }] => (*n, name.as_str()),
        _ => {
            return Err(
                "run_gaussian_ar1_plan: plan must contain exactly one AR1 effect".into(),
            );
        }
    };
    if y.len() != n {
        return Err(format!("y length {} != AR1 n {n}", y.len()).into());
    }
    let prec = match plan.likelihood {
        LikelihoodPlan::Gaussian { precision } => precision,
    };
    let obs: Vec<Obs> = y
        .iter()
        .map(|&yi| {
            Obs::Gaussian(GaussianObs {
                y: yi,
                precision: prec,
                link: Link::Identity,
            })
        })
        .collect();

    let stack = plan.prior_stack();
    let build_prior = move |theta: &[f64]| -> Result<inla_math::CscMatrix, String> {
        if theta.len() != 2 {
            return Err(format!("AR1 expects θ length 2, got {}", theta.len()));
        }
        let tau = theta[0].exp();
        let rho = (2.0 / (1.0 + (-theta[1]).exp()) - 1.0).clamp(-0.999, 0.999);
        ar1_precision_csc(n, rho, tau)
    };
    let log_prior = move |theta: &[f64]| stack.log_density(theta).unwrap_or(f64::NEG_INFINITY);

    run_inla_inference(
        &plan.initial_theta,
        &build_prior,
        &log_prior,
        &obs,
        &plan.computation.strategy,
        plan.computation.step_or_f0,
    )
    .map_err(PlanError)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ar1_spec(n: usize) -> ModelSpec {
        ModelSpec {
            likelihood: LikelihoodSpec::Gaussian {
                precision: Some(16.0),
            },
            effects: vec![LatentEffectSpec::Ar1 {
                name: "time".into(),
                n,
                priors: None,
            }],
            computation: ComputationSpec::default(),
            initial_theta: None,
        }
    }

    #[test]
    fn resolve_ar1_fills_defaults_and_layout() {
        let plan = resolve(ar1_spec(8)).unwrap();
        assert_eq!(plan.layout.total_len, 8);
        assert_eq!(plan.layout.blocks[0].name, "time");
        assert_eq!(plan.layout.blocks[0].start, 0);
        assert_eq!(plan.initial_theta.len(), 2);
        assert_eq!(plan.computation.strategy, "ccd");
        let slots = plan.hyper_slots();
        assert_eq!(slots.len(), 2);
        assert_eq!(slots[0].transform, HyperTransformKind::Exp);
        assert_eq!(slots[1].transform, HyperTransformKind::RhoCor1);
        assert!((slots[0].transform.to_natural(0.0) - 1.0).abs() < 1e-12);
        assert!(slots[1].transform.to_natural(0.0).abs() < 1e-12);
    }

    #[test]
    fn resolve_rejects_empty_effects() {
        let spec = ModelSpec {
            likelihood: LikelihoodSpec::Gaussian { precision: None },
            effects: vec![],
            computation: ComputationSpec::default(),
            initial_theta: None,
        };
        assert!(resolve(spec).is_err());
    }

    #[test]
    fn resolve_honours_initial_theta_override() {
        let mut spec = ar1_spec(6);
        spec.initial_theta = Some(vec![0.5, -0.2]);
        let plan = resolve(spec).unwrap();
        assert_eq!(plan.initial_theta, vec![0.5, -0.2]);
    }

    #[test]
    fn run_ar1_plan_smoke() {
        let n = 10;
        let plan = resolve(ar1_spec(n)).unwrap();
        let y: Vec<f64> = (0..n).map(|i| 0.2 * (i as f64 * 0.4).sin()).collect();
        let result = run_gaussian_ar1_plan(&plan, &y).expect("run");
        assert_eq!(result.latent_means.len(), n);
        assert!(result.marginal_log_lik.is_finite());
        assert_eq!(result.mode.len(), 2);
    }
}
