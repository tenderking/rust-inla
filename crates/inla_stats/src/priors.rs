//! R-INLA-compatible hyperparameter priors on the **internal θ scale**.
//!
//! Densities match `inlaprog/src/inla-priors.c` (`PRIOR_EVAL`): log π(θ) already
//! includes the natural→internal Jacobian. Callers must not add another Jacobian.

use inla_math::Eval1D;

use crate::inference::log_gamma;

const LOG_NORMC_GAUSSIAN: f64 = -0.918_938_533_204_672_8; // -½ log(2π)

/// Named prior family with R-INLA parameter conventions and cached calibrated λ.
#[derive(Debug, Clone, PartialEq)]
pub enum PriorFamily {
    /// PC prior on precision: `P(σ > u) = α`, θ = log τ, σ = τ^{-1/2}.
    PcPrec { u: f64, alpha: f64, lambda: f64 },
    /// PC prior on AR(1) correlation (base ρ=0): `P(|ρ| > u) = α`.
    /// θ = logit((1+ρ)/2). R name: `pc.cor0` / `pc.rho0`.
    PcCor0 { u: f64, alpha: f64, lambda: f64 },
    /// PC prior on AR(1) correlation (base ρ=1): `P(ρ > u) = α`.
    /// θ = logit((1+ρ)/2). R name: `pc.cor1` / `pc.rho1`.
    PcCor1 { u: f64, alpha: f64, lambda: f64 },
    /// PC prior on BYM2 spatial mixing parameter ϕ (base ϕ=0): `P(ϕ < u) = α`.
    /// θ = logit(ϕ). R name: `pc.bym2` / `pc.phi`.
    PcBym2 { u: f64, alpha: f64, lambda: f64 },
    /// PC prior on Matérn / SPDE range parameter: `P(ρ < r_0) = α_r`.
    /// θ = log ρ. R name: `pc.range`.
    PcRange {
        r0: f64,
        alpha_r: f64,
        d: f64,
        lambda: f64,
    },
    /// PC prior for 2D SPDE Matérn on internal θ = (log τ, log κ) mapped to (log ρ, log σ).
    /// R name: `pc.spde` / `pc.matern`.
    PcSpde { lambda1: f64, lambda2: f64, d: f64 },
    /// Gamma on τ = e^θ with **rate** `b` (R-INLA): mean = a/b.
    LogGamma { shape: f64, rate: f64 },
    /// Gaussian directly on θ: mean μ, precision τ (`τ=0` ⇒ flat).
    Gaussian { mean: f64, precision: f64 },
    /// Flat on θ: log π = 0.
    Flat,
    /// Beta(a,b) on p ∈ (0,1); θ = logit(p).
    LogitBeta { a: f64, b: f64 },
    /// Placeholder for remaining slots (`prior = "none"`): contributes 0 θ.
    NonePrior,
}

impl PriorFamily {
    /// Dimension of θ this prior expects.
    pub fn theta_dim(&self) -> usize {
        match self {
            PriorFamily::PcSpde { .. } => 2,
            PriorFamily::NonePrior => 0,
            _ => 1,
        }
    }
}

/// A concrete prior ready to evaluate.
#[derive(Debug, Clone, PartialEq)]
pub struct PriorSpec {
    pub family: PriorFamily,
    pub name: String,
}

impl PriorSpec {
    pub fn new(family: PriorFamily) -> Self {
        let name = family_canonical_name(&family).to_string();
        Self { family, name }
    }

    pub fn pc_prec(u: f64, alpha: f64) -> Result<Self, String> {
        if !(u > 0.0 && alpha > 0.0 && alpha < 1.0) {
            return Err(format!(
                "pc.prec: need u>0 and 0<alpha<1, got u={u}, alpha={alpha}"
            ));
        }
        let lambda = -alpha.ln() / u;
        Ok(Self::new(PriorFamily::PcPrec { u, alpha, lambda }))
    }

    pub fn pc_cor0(u: f64, alpha: f64) -> Result<Self, String> {
        let lambda = solve_pc_cor0_lambda(u, alpha)?;
        Ok(Self::new(PriorFamily::PcCor0 { u, alpha, lambda }))
    }

    pub fn pc_cor1(u: f64, alpha: f64) -> Result<Self, String> {
        let lambda = solve_pc_cor1_lambda(u, alpha)?;
        Ok(Self::new(PriorFamily::PcCor1 { u, alpha, lambda }))
    }

    pub fn pc_bym2(u: f64, alpha: f64) -> Result<Self, String> {
        let lambda = solve_pc_bym2_lambda(u, alpha)?;
        Ok(Self::new(PriorFamily::PcBym2 { u, alpha, lambda }))
    }

    pub fn pc_range(r0: f64, alpha_r: f64, d: f64) -> Result<Self, String> {
        if !(r0 > 0.0 && alpha_r > 0.0 && alpha_r < 1.0 && d > 0.0) {
            return Err(format!(
                "pc.range: need r0>0, 0<alpha_r<1, d>0, got r0={r0}, alpha_r={alpha_r}, d={d}"
            ));
        }
        let lambda = -alpha_r.ln() * r0.powf(d / 2.0);
        Ok(Self::new(PriorFamily::PcRange {
            r0,
            alpha_r,
            d,
            lambda,
        }))
    }

    pub fn pc_spde(lambda1: f64, lambda2: f64, d: f64) -> Result<Self, String> {
        check_pc_spde_params(lambda1, lambda2, d)?;
        Ok(Self::new(PriorFamily::PcSpde {
            lambda1,
            lambda2,
            d,
        }))
    }

    pub fn pc_spde_quantiles(
        r0: f64,
        alpha_r: f64,
        s0: f64,
        alpha_s: f64,
        d: f64,
    ) -> Result<Self, String> {
        if !(r0 > 0.0 && alpha_r > 0.0 && alpha_r < 1.0) {
            return Err(format!(
                "pc.spde quantiles: need r0>0 and 0<alpha_r<1, got r0={r0}, alpha_r={alpha_r}"
            ));
        }
        if !(s0 > 0.0 && alpha_s > 0.0 && alpha_s < 1.0) {
            return Err(format!(
                "pc.spde quantiles: need s0>0 and 0<alpha_s<1, got s0={s0}, alpha_s={alpha_s}"
            ));
        }
        check_pc_spde_dimension(d)?;
        let lambda1 = -alpha_r.ln() * r0.powf(d / 2.0);
        let lambda2 = -alpha_s.ln() / s0;
        Self::pc_spde(lambda1, lambda2, d)
    }

    pub fn loggamma(shape: f64, rate: f64) -> Self {
        Self::new(PriorFamily::LogGamma { shape, rate })
    }

    pub fn gaussian(mean: f64, precision: f64) -> Self {
        Self::new(PriorFamily::Gaussian { mean, precision })
    }

    pub fn flat() -> Self {
        Self::new(PriorFamily::Flat)
    }

    pub fn logitbeta(a: f64, b: f64) -> Self {
        Self::new(PriorFamily::LogitBeta { a, b })
    }

    /// Parse R-INLA `prior=` name + `param=` vector (defaults when `param` empty).
    pub fn from_name_params(name: &str, param: &[f64]) -> Result<Self, String> {
        let key = trim_family(name);
        let spec = match key.as_str() {
            "pcprec" => {
                let (u, alpha) = take2(param, 1.0, 0.01)?;
                Self::pc_prec(u, alpha)?
            }
            "pccor0" | "pcrho0" => {
                let (u, alpha) = take2(param, 0.5, 0.05)?;
                Self::pc_cor0(u, alpha)?
            }
            "pccor1" | "pcrho1" => {
                let (u, alpha) = take2(param, 0.5, 0.75)?;
                Self::pc_cor1(u, alpha)?
            }
            "pcbym2" | "pcphi" => {
                let (u, alpha) = take2(param, 0.5, 0.5)?;
                Self::pc_bym2(u, alpha)?
            }
            "pcrange" => {
                let r0 = param.first().copied().unwrap_or(1.0);
                let alpha_r = param.get(1).copied().unwrap_or(0.05);
                let d = param.get(2).copied().unwrap_or(2.0);
                Self::pc_range(r0, alpha_r, d)?
            }
            "pcspde" | "pcmatern" => {
                if param.len() == 5 {
                    Self::pc_spde_quantiles(param[0], param[1], param[2], param[3], param[4])?
                } else if param.len() >= 3 {
                    Self::pc_spde(param[0], param[1], param[2])?
                } else {
                    // Default for 2D SPDE Matérn (ν=1): P(ρ<1)=0.05, P(σ>1)=0.01
                    Self::pc_spde_quantiles(1.0, 0.05, 1.0, 0.01, 2.0)?
                }
            }
            "loggamma" => {
                let (shape, rate) = take2(param, 1.0, 5e-5)?;
                Self::loggamma(shape, rate)
            }
            "gaussian" | "normal" => {
                let (mean, precision) = take2(param, 0.0, 0.001)?;
                Self::gaussian(mean, precision)
            }
            "flat" | "uniform" => Self::flat(),
            "logitbeta" => {
                let (a, b) = take2(param, 1.0, 1.0)?;
                Self::logitbeta(a, b)
            }
            "none" => Self::new(PriorFamily::NonePrior),
            other => return Err(format!("unknown prior '{other}' (from '{name}')")),
        };
        Ok(spec)
    }

    pub fn theta_dim(&self) -> usize {
        self.family.theta_dim()
    }

    /// Convert prior to `(canonical_name, param_vec)` pair.
    pub fn to_pair(&self) -> (String, Vec<f64>) {
        (self.name.clone(), family_param_vec(&self.family))
    }

    /// Log-density on internal θ (length must match [`Self::theta_dim`]).
    pub fn log_density(&self, theta: &[f64]) -> Result<f64, String> {
        let n = self.theta_dim();
        if theta.len() != n {
            return Err(format!(
                "prior '{}': expected θ length {n}, got {}",
                self.name,
                theta.len()
            ));
        }
        match &self.family {
            PriorFamily::PcPrec { lambda, .. } => Ok(pc_prec_log_dens(theta[0], *lambda)?),
            PriorFamily::PcCor0 { lambda, .. } => Ok(pc_cor0_log_dens(theta[0], *lambda)?),
            PriorFamily::PcCor1 { lambda, .. } => Ok(pc_cor1_log_dens(theta[0], *lambda)?),
            PriorFamily::PcBym2 { lambda, .. } => Ok(pc_bym2_log_dens(theta[0], *lambda)?),
            PriorFamily::PcRange { d, lambda, .. } => Ok(pc_range_log_dens(theta[0], *d, *lambda)?),
            PriorFamily::PcSpde {
                lambda1,
                lambda2,
                d,
            } => Ok(pc_spde_log_dens(
                theta[0], theta[1], *lambda1, *lambda2, *d,
            )?),
            PriorFamily::LogGamma { shape, rate } => {
                Ok(loggamma_log_dens(theta[0], *shape, *rate)?)
            }
            PriorFamily::Gaussian { mean, precision } => {
                Ok(gaussian_log_dens(theta[0], *mean, *precision)?)
            }
            PriorFamily::Flat => Ok(0.0),
            PriorFamily::LogitBeta { a, b } => Ok(logitbeta_log_dens(theta[0], *a, *b)?),
            PriorFamily::NonePrior => Ok(0.0),
        }
    }

    /// 1D prior with analytic grad/hess (errors if θ-dim ≠ 1).
    pub fn eval1d(&self, theta: f64) -> Result<Eval1D, String> {
        if self.theta_dim() != 1 {
            return Err(format!(
                "prior '{}': eval1d requires 1D prior (got dim {})",
                self.name,
                self.theta_dim()
            ));
        }
        match &self.family {
            PriorFamily::PcPrec { lambda, .. } => pc_prec_eval(theta, *lambda),
            PriorFamily::PcCor0 { lambda, .. } => pc_cor0_eval(theta, *lambda),
            PriorFamily::PcCor1 { lambda, .. } => pc_cor1_eval(theta, *lambda),
            PriorFamily::PcBym2 { lambda, .. } => pc_bym2_eval(theta, *lambda),
            PriorFamily::PcRange { d, lambda, .. } => pc_range_eval(theta, *d, *lambda),
            PriorFamily::LogGamma { shape, rate } => loggamma_eval(theta, *shape, *rate),
            PriorFamily::Gaussian { mean, precision } => gaussian_eval(theta, *mean, *precision),
            PriorFamily::Flat => Ok(Eval1D {
                logp: 0.0,
                grad: 0.0,
                hess: 0.0,
            }),
            PriorFamily::LogitBeta { a, b } => logitbeta_eval(theta, *a, *b),
            PriorFamily::PcSpde { .. } => unreachable!(),
            PriorFamily::NonePrior => Ok(Eval1D {
                logp: 0.0,
                grad: 0.0,
                hess: 0.0,
            }),
        }
    }
}

/// Ordered priors matching concatenated internal θ blocks.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HyperPriorStack {
    pub priors: Vec<PriorSpec>,
}

impl HyperPriorStack {
    pub fn new(priors: Vec<PriorSpec>) -> Self {
        Self { priors }
    }

    pub fn theta_dim(&self) -> usize {
        self.priors.iter().map(|p| p.theta_dim()).sum()
    }

    pub fn log_density(&self, theta: &[f64]) -> Result<f64, String> {
        let need = self.theta_dim();
        if theta.len() != need {
            return Err(format!(
                "HyperPriorStack: θ length {} != expected {need}",
                theta.len()
            ));
        }
        let mut off = 0usize;
        let mut s = 0.0;
        for p in &self.priors {
            let n = p.theta_dim();
            s += p.log_density(&theta[off..off + n])?;
            off += n;
        }
        Ok(s)
    }

    /// Default prior for a latent effect hyperparameter block by model type.
    pub fn default_for_effect(model: &str) -> Result<Self, String> {
        let m = model.to_ascii_lowercase();
        match m.as_str() {
            "iid" | "rw1" | "rw2" | "rw2d" | "besag" | "besag2" | "seasonal" | "crw1" | "crw2" => {
                Ok(Self::new(vec![PriorSpec::pc_prec(1.0, 0.01)?]))
            }
            "bym" => Ok(Self::new(vec![
                PriorSpec::pc_prec(1.0, 0.01)?,
                PriorSpec::pc_prec(1.0, 0.01)?,
            ])),
            // θ = [log_tau, logit_phi]; PC prior on spatial proportion ϕ (base ϕ=0)
            "bym2" => Ok(Self::new(vec![
                PriorSpec::pc_prec(1.0, 0.01)?,
                PriorSpec::pc_bym2(0.5, 0.5)?,
            ])),
            // θ = [log_prec, log_range]; PC precision and PC range (d=2)
            "matern2d" => Ok(Self::new(vec![
                PriorSpec::pc_prec(1.0, 0.01)?,
                PriorSpec::pc_range(1.0, 0.05, 2.0)?,
            ])),
            // θ = [log_tau, logit_rho]; PC precision and PC correlation (base ρ=1)
            "ar1" => Ok(Self::new(vec![
                PriorSpec::pc_prec(1.0, 0.01)?,
                PriorSpec::pc_cor1(0.5, 0.75)?,
            ])),
            "ar" | "arp" => Ok(Self::new(vec![
                PriorSpec::pc_prec(1.0, 0.01)?,
                PriorSpec::gaussian(0.0, 0.1),
                PriorSpec::gaussian(0.0, 0.1),
            ])),
            "fgn" => Ok(Self::new(vec![
                PriorSpec::pc_prec(1.0, 0.01)?,
                PriorSpec::gaussian(0.0, 0.1),
            ])),
            // θ = [log_tau, log_kappa]; Joint PC Matérn on 2D SPDE
            "spde" => Ok(Self::new(vec![PriorSpec::pc_spde_quantiles(
                1.0, 0.05, 1.0, 0.01, 2.0,
            )?])),
            "fixed" => Ok(Self::new(vec![])),
            // Free scaling on a copied field: N(1, prec=0.1) on β (identity scale).
            "copy" => Ok(Self::new(vec![PriorSpec::gaussian(1.0, 0.1)])),
            other => Err(format!("no default hyperprior for effect type '{other}'")),
        }
    }

    /// Build from parallel `prior=` names and `param=` vectors (R/Python bridge).
    pub fn from_names_params(names: &[String], params: &[Vec<f64>]) -> Result<Self, String> {
        if names.len() != params.len() {
            return Err(format!(
                "prior names length {} != params length {}",
                names.len(),
                params.len()
            ));
        }
        let mut priors = Vec::with_capacity(names.len());
        for (n, p) in names.iter().zip(params.iter()) {
            priors.push(PriorSpec::from_name_params(n, p)?);
        }
        Ok(Self::new(priors))
    }

    /// Serialize each prior as `(canonical_name, param_vec)` for frontend round-trips.
    pub fn to_names_params(&self) -> Vec<(String, Vec<f64>)> {
        self.priors.iter().map(|p| p.to_pair()).collect()
    }
}

pub fn family_param_vec(f: &PriorFamily) -> Vec<f64> {
    match f {
        PriorFamily::PcPrec { u, alpha, .. } => vec![*u, *alpha],
        PriorFamily::PcCor0 { u, alpha, .. } => vec![*u, *alpha],
        PriorFamily::PcCor1 { u, alpha, .. } => vec![*u, *alpha],
        PriorFamily::PcBym2 { u, alpha, .. } => vec![*u, *alpha],
        PriorFamily::PcRange { r0, alpha_r, d, .. } => vec![*r0, *alpha_r, *d],
        PriorFamily::PcSpde {
            lambda1,
            lambda2,
            d,
        } => vec![*lambda1, *lambda2, *d],
        PriorFamily::LogGamma { shape, rate } => vec![*shape, *rate],
        PriorFamily::Gaussian { mean, precision } => vec![*mean, *precision],
        PriorFamily::Flat => vec![],
        PriorFamily::LogitBeta { a, b } => vec![*a, *b],
        PriorFamily::NonePrior => vec![],
    }
}

// --- helpers ----------------------------------------------------------------

fn trim_family(name: &str) -> String {
    name.trim()
        .to_ascii_lowercase()
        .chars()
        .filter(|c| *c != '.' && *c != '_' && *c != '-')
        .collect()
}

pub fn family_canonical_name(f: &PriorFamily) -> &'static str {
    match f {
        PriorFamily::PcPrec { .. } => "pc.prec",
        PriorFamily::PcCor0 { .. } => "pc.cor0",
        PriorFamily::PcCor1 { .. } => "pc.cor1",
        PriorFamily::PcBym2 { .. } => "pc.bym2",
        PriorFamily::PcRange { .. } => "pc.range",
        PriorFamily::PcSpde { .. } => "pc.spde",
        PriorFamily::LogGamma { .. } => "loggamma",
        PriorFamily::Gaussian { .. } => "gaussian",
        PriorFamily::Flat => "flat",
        PriorFamily::LogitBeta { .. } => "logitbeta",
        PriorFamily::NonePrior => "none",
    }
}

fn take2(param: &[f64], d0: f64, d1: f64) -> Result<(f64, f64), String> {
    match param.len() {
        0 => Ok((d0, d1)),
        1 => Ok((param[0], d1)),
        _ => Ok((param[0], param[1])),
    }
}

/// 2D Matérn SPDE (ν = 1): (log τ, log κ) → (log ρ, log σ) with |det J| = 1.
fn check_pc_spde_dimension(d: f64) -> Result<(), String> {
    if (d - 2.0).abs() > 1e-12 {
        return Err(format!(
            "pc.spde: only the 2D ν=1 map (d=2) is supported, got d={d}"
        ));
    }
    Ok(())
}

fn check_pc_spde_params(lambda1: f64, lambda2: f64, d: f64) -> Result<(), String> {
    check_pc_spde_dimension(d)?;
    if !(lambda1 > 0.0 && lambda1.is_finite() && lambda2 > 0.0 && lambda2.is_finite()) {
        return Err(format!(
            "pc.spde: need lambda1>0 and lambda2>0, got lambda1={lambda1}, lambda2={lambda2}"
        ));
    }
    Ok(())
}

// --- Density & Derivative Solvers -------------------------------------------

fn pc_prec_log_dens(theta: f64, lambda: f64) -> Result<f64, String> {
    if !theta.is_finite() {
        return Err("pc.prec: θ must be finite".into());
    }
    // log(λ/2) - λ e^{-θ/2} - θ/2
    Ok(lambda.ln() - std::f64::consts::LN_2 - lambda * (-0.5 * theta).exp() - 0.5 * theta)
}

fn pc_prec_eval(theta: f64, lambda: f64) -> Result<Eval1D, String> {
    let e = (-0.5 * theta).exp();
    let logp = lambda.ln() - std::f64::consts::LN_2 - lambda * e - 0.5 * theta;
    let grad = 0.5 * lambda * e - 0.5;
    let hess = -0.25 * lambda * e;
    Ok(Eval1D { logp, grad, hess })
}

/// Solve λ for `pc.cor0`: P(|ρ| > u) = α where d(ρ) = √(-ln(1-ρ²)).
/// Base model ρ=0. λ = -ln(α) / √(-ln(1-u²)).
fn solve_pc_cor0_lambda(u: f64, alpha: f64) -> Result<f64, String> {
    if !(u > 0.0 && u < 1.0) {
        return Err(format!("pc.cor0: u must be in (0,1), got {u}"));
    }
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(format!("pc.cor0: alpha must be in (0,1), got {alpha}"));
    }
    let d_u = (-(1.0 - u * u).ln()).sqrt();
    Ok(-alpha.ln() / d_u)
}

fn pc_cor0_log_dens(theta: f64, lambda: f64) -> Result<f64, String> {
    if !theta.is_finite() {
        return Err("pc.cor0: θ must be finite".into());
    }
    if theta.abs() < 1e-7 {
        // As θ → 0, ρ → 0, |ρ|/d(ρ) → 1, d(ρ) → 0, log π(0) = ln(λ) - 2 ln(2)
        return Ok(lambda.ln() - 2.0 * std::f64::consts::LN_2);
    }
    let e = theta.exp();
    let p = e / (1.0 + e);
    let rho = 2.0 * p - 1.0;
    let rho2 = rho * rho;
    if rho2 >= 1.0 {
        return Ok(f64::NEG_INFINITY);
    }
    let d_rho = (-(1.0 - rho2).ln()).sqrt();
    // log π(θ) = ln λ - 2 ln 2 + ln(|ρ|/d(ρ)) - λ d(ρ)
    Ok(lambda.ln() - 2.0 * std::f64::consts::LN_2 + (rho.abs() / d_rho).ln() - lambda * d_rho)
}

fn pc_cor0_eval(theta: f64, lambda: f64) -> Result<Eval1D, String> {
    let logp = pc_cor0_log_dens(theta, lambda)?;
    let e = theta.exp();
    let p = e / (1.0 + e);
    let rho = 2.0 * p - 1.0;
    if rho.abs() < 1e-6 {
        // Near zero: grad = 0, hessian evaluated smoothly
        let eps = 1e-5;
        let g0 = pc_cor0_log_dens(theta - eps, lambda)?;
        let g1 = pc_cor0_log_dens(theta + eps, lambda)?;
        let hess = (g1 - 2.0 * logp + g0) / (eps * eps);
        return Ok(Eval1D {
            logp,
            grad: 0.0,
            hess,
        });
    }
    let rho2 = rho * rho;
    let d_rho = (-(1.0 - rho2).ln()).sqrt();
    // d/dθ log π = (1-ρ²)/(2ρ) - ρ/(2 d(ρ)²) - λ ρ / (2 d(ρ))
    let grad =
        (1.0 - rho2) / (2.0 * rho) - rho / (2.0 * d_rho * d_rho) - lambda * rho / (2.0 * d_rho);
    let eps = 1e-5;
    let g0 = pc_cor0_log_dens(theta - eps, lambda)?;
    let g1 = pc_cor0_log_dens(theta + eps, lambda)?;
    let hess = (g1 - 2.0 * logp + g0) / (eps * eps);
    Ok(Eval1D { logp, grad, hess })
}

/// Solve λ from (1-e^{-λ√(1-u)})/(1-e^{-λ√2}) = α (R `inla.pc.cor1.lambda`).
fn solve_pc_cor1_lambda(u: f64, alpha: f64) -> Result<f64, String> {
    if !(-1.0..1.0).contains(&u) {
        return Err(format!("pc.cor1: u must be in (-1,1), got {u}"));
    }
    let alpha_min = ((1.0 - u) / 2.0).sqrt();
    if !(alpha > alpha_min && alpha < 1.0) {
        return Err(format!(
            "pc.cor1: need alpha_min < alpha < 1 with alpha_min={alpha_min}, got {alpha}"
        ));
    }
    let fun = |lam: f64| -> f64 {
        let ff = (1.0 - (-lam * (1.0 - u).sqrt()).exp())
            / (1.0 - (-lam * std::f64::consts::SQRT_2).exp());
        let d = ff - alpha;
        d * d
    };
    let mut best_lam = 1.0;
    let mut best_f = fun(best_lam);
    for i in 0..110 {
        let lam = if i < 100 {
            1e-7 + (10.0 - 1e-7) * (i as f64) / 99.0
        } else {
            10.0 + 90.0 * ((i - 100) as f64) / 9.0
        };
        let f = fun(lam);
        if f < best_f {
            best_f = f;
            best_lam = lam;
        }
    }
    let mut lo = (best_lam * 0.5).max(1e-8);
    let mut hi = best_lam * 2.0;
    for _ in 0..60 {
        let m1 = lo + 0.382 * (hi - lo);
        let m2 = lo + 0.618 * (hi - lo);
        if fun(m1) < fun(m2) {
            hi = m2;
        } else {
            lo = m1;
        }
    }
    let lambda = 0.5 * (lo + hi);
    if fun(lambda) > 1e-6 {
        return Err(format!(
            "pc.cor1: failed to solve λ (resid={})",
            fun(lambda).sqrt()
        ));
    }
    Ok(lambda)
}

fn pc_cor1_log_dens_rho(rho: f64, lambda: f64) -> Result<f64, String> {
    if !(-1.0..1.0).contains(&rho) {
        return Err(format!("pc.cor1: ρ must be in (-1,1), got {rho}"));
    }
    let s = (1.0 - rho).sqrt();
    // log λ - λ√(1-ρ) - log(1-e^{-√2 λ}) - log(2√(1-ρ))
    Ok(lambda.ln()
        - lambda * s
        - (1.0 - (-std::f64::consts::SQRT_2 * lambda).exp()).ln()
        - (2.0 * s).ln())
}

fn pc_cor1_log_dens(theta: f64, lambda: f64) -> Result<f64, String> {
    if !theta.is_finite() {
        return Err("pc.cor1: θ must be finite".into());
    }
    let e = theta.exp();
    let rho = 2.0 * e / (1.0 + e) - 1.0;
    let log_pi_rho = pc_cor1_log_dens_rho(rho, lambda)?;
    let log_j = std::f64::consts::LN_2 + theta - 2.0 * (1.0 + e).ln();
    Ok(log_pi_rho + log_j)
}

fn pc_cor1_eval(theta: f64, lambda: f64) -> Result<Eval1D, String> {
    let logp = pc_cor1_log_dens(theta, lambda)?;
    let e = theta.exp();
    let rho = 2.0 * e / (1.0 + e) - 1.0;
    let s = (1.0 - rho).max(0.0).sqrt();
    if s < 1e-8 {
        let eps = 1e-5;
        let g0 = pc_cor1_log_dens(theta - eps, lambda)?;
        let g1 = pc_cor1_log_dens(theta + eps, lambda)?;
        let grad = (g1 - g0) / (2.0 * eps);
        let hess = (g1 - 2.0 * logp + g0) / (eps * eps);
        return Ok(Eval1D { logp, grad, hess });
    }
    let drho = (1.0 - rho * rho) / 2.0;
    let ds = -drho / (2.0 * s);
    // d/dθ [log π(ρ) + log|dρ/dθ|] with ρ = tanh(θ/2)
    let grad = (lambda * s + 1.0) * (1.0 + rho) / 4.0 - rho;
    let hess = 0.25 * (lambda * ds * (1.0 + rho) + (lambda * s + 1.0) * drho) - drho;
    Ok(Eval1D { logp, grad, hess })
}

/// Solve λ for `pc.bym2`: P(ϕ < u) = α where d(ϕ) = √ϕ.
/// Equation: (1 - e^{-λ√u}) / (1 - e^{-λ}) = α.
fn solve_pc_bym2_lambda(u: f64, alpha: f64) -> Result<f64, String> {
    if !(u > 0.0 && u < 1.0) {
        return Err(format!("pc.bym2: u must be in (0,1), got {u}"));
    }
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(format!("pc.bym2: need 0 < alpha < 1, got {alpha}"));
    }
    let sqrt_u = u.sqrt();
    let fun = |lam: f64| -> f64 {
        let ff = if lam.abs() < 1e-6 {
            sqrt_u
        } else {
            (1.0 - (-lam * sqrt_u).exp()) / (1.0 - (-lam).exp())
        };
        let d = ff - alpha;
        d * d
    };
    let mut best_lam = 0.0;
    let mut best_f = fun(best_lam);
    for i in -100..=100 {
        let lam = (i as f64) * 0.2;
        let f = fun(lam);
        if f < best_f {
            best_f = f;
            best_lam = lam;
        }
    }
    let mut lo = best_lam - 0.5;
    let mut hi = best_lam + 0.5;
    for _ in 0..60 {
        let m1 = lo + 0.382 * (hi - lo);
        let m2 = lo + 0.618 * (hi - lo);
        if fun(m1) < fun(m2) {
            hi = m2;
        } else {
            lo = m1;
        }
    }
    let lambda = 0.5 * (lo + hi);
    if fun(lambda) > 1e-5 {
        return Err(format!(
            "pc.bym2: failed to solve λ (resid={})",
            fun(lambda).sqrt()
        ));
    }
    Ok(lambda)
}

fn pc_bym2_log_dens(theta: f64, lambda: f64) -> Result<f64, String> {
    if !theta.is_finite() {
        return Err("pc.bym2: θ must be finite".into());
    }
    let e = theta.exp();
    let phi = e / (1.0 + e);
    let norm = if lambda.abs() < 1e-6 {
        -std::f64::consts::LN_2
    } else {
        lambda.abs().ln() - (2.0 * (1.0 - (-lambda).exp()).abs()).ln()
    };
    Ok(norm + 0.5 * phi.ln() + (1.0 - phi).ln() - lambda * phi.sqrt())
}

fn pc_bym2_eval(theta: f64, lambda: f64) -> Result<Eval1D, String> {
    let logp = pc_bym2_log_dens(theta, lambda)?;
    let e = theta.exp();
    let phi = e / (1.0 + e);
    let sqrt_phi = phi.sqrt();
    // Analytic exact gradient: 1/2 - 3/2 ϕ - 1/2 λ √ϕ (1-ϕ)
    let grad = 0.5 - 1.5 * phi - 0.5 * lambda * sqrt_phi * (1.0 - phi);
    // Analytic exact hessian: (-3/2 - λ(1-3ϕ)/(4√ϕ)) ϕ(1-ϕ)
    let hess = if sqrt_phi < 1e-6 {
        -1.5 * phi * (1.0 - phi)
    } else {
        (-1.5 - lambda * (1.0 - 3.0 * phi) / (4.0 * sqrt_phi)) * phi * (1.0 - phi)
    };
    Ok(Eval1D { logp, grad, hess })
}

/// PC prior on spatial range parameter ρ (base ρ=∞): P(ρ < r_0) = α_r.
/// θ = ln ρ, λ = -ln(α_r) * r_0^{d/2}.
/// log π(θ) = ln(λ d / 2) - (d/2) θ - λ exp(-(d/2) θ).
fn pc_range_log_dens(theta: f64, d: f64, lambda: f64) -> Result<f64, String> {
    if !theta.is_finite() {
        return Err("pc.range: θ must be finite".into());
    }
    let d_half = d / 2.0;
    Ok((lambda * d_half).ln() - d_half * theta - lambda * (-d_half * theta).exp())
}

fn pc_range_eval(theta: f64, d: f64, lambda: f64) -> Result<Eval1D, String> {
    if !theta.is_finite() {
        return Err("pc.range: θ must be finite".into());
    }
    let d_half = d / 2.0;
    let e = (-d_half * theta).exp();
    let logp = (lambda * d_half).ln() - d_half * theta - lambda * e;
    let grad = -d_half + lambda * d_half * e;
    let hess = -lambda * d_half * d_half * e;
    Ok(Eval1D { logp, grad, hess })
}

/// PC prior on 2D SPDE Matérn: internal hyperparameters are (θ1, θ2) = (log τ, log κ).
/// Maps (log τ, log κ) → (log ρ, log σ) with |det J| = 1 (ν = 1, d = 2 only):
///   log ρ = 1/2 ln(8) - θ2
///   log σ = -1/2 ln(4π) - θ1 - θ2
fn pc_spde_log_dens(
    theta1: f64,
    theta2: f64,
    lambda1: f64,
    lambda2: f64,
    d: f64,
) -> Result<f64, String> {
    if !theta1.is_finite() || !theta2.is_finite() {
        return Err("pc.spde: θ parameters must be finite".into());
    }
    check_pc_spde_params(lambda1, lambda2, d)?;
    // Mapping from SPDE (log_tau, log_kappa) to (log_rho, log_sigma)
    let log_rho = 0.5 * (8.0_f64).ln() - theta2;
    let log_sigma = -0.5 * (4.0 * std::f64::consts::PI).ln() - theta1 - theta2;

    let d_half = d / 2.0;
    // Range part: log π(log ρ) = ln(λ1 d/2) - (d/2) log ρ - λ1 exp(-(d/2) log ρ)
    let s_rho = (lambda1 * d_half).ln() - d_half * log_rho - lambda1 * (-d_half * log_rho).exp();
    // Sigma part: log π(log σ) = ln λ2 + log σ - λ2 exp(log σ)
    let s_sigma = lambda2.ln() + log_sigma - lambda2 * log_sigma.exp();
    Ok(s_rho + s_sigma)
}

fn loggamma_log_dens(theta: f64, shape: f64, rate: f64) -> Result<f64, String> {
    if !theta.is_finite() {
        return Err("loggamma: θ must be finite".into());
    }
    if !(shape > 0.0 && rate > 0.0) {
        return Err(format!(
            "loggamma: need shape>0 rate>0, got shape={shape} rate={rate}"
        ));
    }
    // Gamma(shape, rate) on τ=e^θ plus Jacobian: a ln rate - lnΓ(a) + a θ - rate e^θ
    Ok(shape * rate.ln() - log_gamma(shape) + shape * theta - rate * theta.exp())
}

fn loggamma_eval(theta: f64, shape: f64, rate: f64) -> Result<Eval1D, String> {
    let logp = loggamma_log_dens(theta, shape, rate)?;
    let e = theta.exp();
    let grad = shape - rate * e;
    let hess = -rate * e;
    Ok(Eval1D { logp, grad, hess })
}

fn gaussian_log_dens(theta: f64, mean: f64, precision: f64) -> Result<f64, String> {
    if !theta.is_finite() {
        return Err("gaussian: θ must be finite".into());
    }
    if !(precision >= 0.0 && precision.is_finite()) {
        return Err(format!(
            "gaussian: precision must be finite and >= 0, got {precision}"
        ));
    }
    if precision == 0.0 {
        return Ok(0.0);
    }
    let d = theta - mean;
    Ok(LOG_NORMC_GAUSSIAN + 0.5 * precision.ln() - 0.5 * precision * d * d)
}

fn gaussian_eval(theta: f64, mean: f64, precision: f64) -> Result<Eval1D, String> {
    let logp = gaussian_log_dens(theta, mean, precision)?;
    if precision == 0.0 {
        return Ok(Eval1D {
            logp: 0.0,
            grad: 0.0,
            hess: 0.0,
        });
    }
    let d = theta - mean;
    Ok(Eval1D {
        logp,
        grad: -precision * d,
        hess: -precision,
    })
}

fn logitbeta_log_dens(theta: f64, a: f64, b: f64) -> Result<f64, String> {
    if !theta.is_finite() {
        return Err("logitbeta: θ must be finite".into());
    }
    if !(a > 0.0 && b > 0.0) {
        return Err(format!("logitbeta: need a>0 b>0, got a={a} b={b}"));
    }
    let e = theta.exp();
    let p = e / (1.0 + e);
    // log Beta(p;a,b) = (a-1)log p + (b-1)log(1-p) - log B(a,b)
    let log_b = log_gamma(a) + log_gamma(b) - log_gamma(a + b);
    let log_beta_dens = (a - 1.0) * p.ln() + (b - 1.0) * (1.0 - p).ln() - log_b;
    // + θ - 2 log(1+e^θ)
    Ok(log_beta_dens + theta - 2.0 * (1.0 + e).ln())
}

fn logitbeta_eval(theta: f64, a: f64, b: f64) -> Result<Eval1D, String> {
    let logp = logitbeta_log_dens(theta, a, b)?;
    let eps = 1e-5;
    let g0 = logitbeta_log_dens(theta - eps, a, b)?;
    let g1 = logitbeta_log_dens(theta + eps, a, b)?;
    let grad = (g1 - g0) / (2.0 * eps);
    let hess = (g1 - 2.0 * logp + g0) / (eps * eps);
    Ok(Eval1D { logp, grad, hess })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) {
        assert!((a - b).abs() < eps, "left={a}, right={b}, eps={eps}");
    }

    #[test]
    fn trim_aliases_and_canonical_names() {
        let p = PriorSpec::from_name_params("pc.prec", &[1.0, 0.01]).unwrap();
        assert_eq!(p.name, "pc.prec");
        assert_eq!(p.to_pair(), ("pc.prec".to_string(), vec![1.0, 0.01]));

        let g = PriorSpec::from_name_params("normal", &[]).unwrap();
        assert_eq!(g.name, "gaussian");

        let c0 = PriorSpec::from_name_params("pc.cor0", &[0.5, 0.05]).unwrap();
        assert_eq!(c0.name, "pc.cor0");

        let c1 = PriorSpec::from_name_params("pc.cor1", &[0.5, 0.75]).unwrap();
        assert_eq!(c1.name, "pc.cor1");

        let bym = PriorSpec::from_name_params("pc.bym2", &[0.5, 0.5]).unwrap();
        assert_eq!(bym.name, "pc.bym2");

        let r = PriorSpec::from_name_params("pc.range", &[10.0, 0.05, 2.0]).unwrap();
        assert_eq!(r.name, "pc.range");

        let spde = PriorSpec::from_name_params("pc.spde", &[10.0, 0.05, 1.0, 0.01, 2.0]).unwrap();
        assert_eq!(spde.name, "pc.spde");
    }

    #[test]
    fn pc_prec_matches_closed_form() {
        let u = 1.0_f64;
        let alpha = 0.01_f64;
        let lambda = -alpha.ln() / u;
        let theta = 0.5_f64;
        let expect =
            lambda.ln() - std::f64::consts::LN_2 - lambda * (-0.5 * theta).exp() - 0.5 * theta;
        let p = PriorSpec::pc_prec(u, alpha).unwrap();
        let got = p.log_density(&[theta]).unwrap();
        approx(got, expect, 1e-12);
        let e = p.eval1d(theta).unwrap();
        approx(e.logp, expect, 1e-12);
        let eps = 1e-6;
        let fd = (p.log_density(&[theta + eps]).unwrap() - p.log_density(&[theta - eps]).unwrap())
            / (2.0 * eps);
        approx(e.grad, fd, 1e-5);
    }

    #[test]
    fn pc_cor0_finite_and_symmetric() {
        let p = PriorSpec::pc_cor0(0.5, 0.05).unwrap();
        let lp0 = p.log_density(&[0.0]).unwrap();
        assert!(lp0.is_finite());
        // Symmetry about 0
        let lp_pos = p.log_density(&[1.5]).unwrap();
        let lp_neg = p.log_density(&[-1.5]).unwrap();
        approx(lp_pos, lp_neg, 1e-12);

        let e = p.eval1d(0.5).unwrap();
        assert!(e.grad.is_finite() && e.hess.is_finite());
    }

    #[test]
    fn pc_cor1_golden_and_analytic_derivatives() {
        let p = PriorSpec::pc_cor1(0.5, 0.75).unwrap();
        let lp0 = p.log_density(&[0.0]).unwrap();
        assert!(lp0.is_finite());
        let theta = 0.4_f64;
        let e = p.eval1d(theta).unwrap();
        assert!(e.grad.is_finite() && e.hess.is_finite());
        let eps = 1e-6;
        let fd_grad = (p.log_density(&[theta + eps]).unwrap()
            - p.log_density(&[theta - eps]).unwrap())
            / (2.0 * eps);
        approx(e.grad, fd_grad, 1e-5);
        let eps_h = 1e-4;
        let fd_hess = (p.log_density(&[theta + eps_h]).unwrap()
            - 2.0 * p.log_density(&[theta]).unwrap()
            + p.log_density(&[theta - eps_h]).unwrap())
            / (eps_h * eps_h);
        approx(e.hess, fd_hess, 2e-3);
    }

    #[test]
    fn pc_prior_golden_internal_log_density() {
        // Closed forms on internal θ, matching R-INLA PRIOR_EVAL / inla.pc.d* after Jacobian.
        let prec = PriorSpec::pc_prec(1.0, 0.01).unwrap();
        let lam_p = -0.01_f64.ln();
        approx(
            prec.log_density(&[0.0]).unwrap(),
            lam_p.ln() - std::f64::consts::LN_2 - lam_p,
            1e-14,
        );

        let cor0 = PriorSpec::pc_cor0(0.5, 0.05).unwrap();
        let d_u = (-(1.0 - 0.25_f64).ln()).sqrt();
        let lam0 = -0.05_f64.ln() / d_u;
        approx(
            cor0.log_density(&[0.0]).unwrap(),
            lam0.ln() - 2.0 * std::f64::consts::LN_2,
            1e-14,
        );

        let cor1 = PriorSpec::pc_cor1(0.5, 0.75).unwrap();
        approx(
            cor1.log_density(&[0.0]).unwrap(),
            -2.381_562_305_990_987,
            1e-8,
        );

        let bym2 = PriorSpec::pc_bym2(0.5, 0.5).unwrap();
        approx(
            bym2.log_density(&[0.0]).unwrap(),
            -1.486_482_918_057_251,
            1e-8,
        );
    }

    #[test]
    fn pc_spde_rejects_non_2d_and_nonpositive_lambda() {
        assert!(PriorSpec::pc_spde(1.0, 1.0, 1.0).is_err());
        assert!(PriorSpec::pc_spde(0.0, 1.0, 2.0).is_err());
        assert!(PriorSpec::pc_spde(-1.0, 1.0, 2.0).is_err());
        assert!(PriorSpec::pc_spde_quantiles(1.0, 0.05, 1.0, 0.01, 1.0).is_err());
        assert!(PriorSpec::from_name_params("pc.spde", &[1.0, 1.0, 3.0]).is_err());
    }

    #[test]
    fn pc_bym2_eval_and_analytic_derivatives() {
        let p = PriorSpec::pc_bym2(0.5, 0.5).unwrap();
        let lp0 = p.log_density(&[0.0]).unwrap();
        assert!(lp0.is_finite());
        let e = p.eval1d(0.5).unwrap();
        assert!(e.grad.is_finite() && e.hess.is_finite());

        // Check analytic grad against finite differences
        let eps = 1e-6;
        let theta = 0.5;
        let fd_grad = (p.log_density(&[theta + eps]).unwrap()
            - p.log_density(&[theta - eps]).unwrap())
            / (2.0 * eps);
        approx(e.grad, fd_grad, 1e-5);
    }

    #[test]
    fn pc_range_eval_matches_fd() {
        let p = PriorSpec::pc_range(20.0, 0.05, 2.0).unwrap();
        let theta = (20.0_f64).ln();
        let e = p.eval1d(theta).unwrap();
        assert!(e.logp.is_finite());
        let eps = 1e-6;
        let fd_grad = (p.log_density(&[theta + eps]).unwrap()
            - p.log_density(&[theta - eps]).unwrap())
            / (2.0 * eps);
        approx(e.grad, fd_grad, 1e-5);
    }

    #[test]
    fn pc_spde_mapping_from_theta() {
        let p = PriorSpec::pc_spde_quantiles(50.0, 0.05, 2.0, 0.01, 2.0).unwrap();
        assert_eq!(p.theta_dim(), 2);
        // At theta = (log_tau, log_kappa) = (0.0, 0.0)
        let lp = p.log_density(&[0.0, 0.0]).unwrap();
        assert!(lp.is_finite());

        // Compute expected directly on (log_rho, log_sigma)
        let log_rho = 0.5 * 8.0_f64.ln();
        let log_sigma = -0.5 * (4.0 * std::f64::consts::PI).ln();
        let lambda1 = -0.05_f64.ln() * 50.0;
        let lambda2 = -0.01_f64.ln() / 2.0;
        let expect_rho = lambda1.ln() - log_rho - lambda1 * (-log_rho).exp();
        let expect_sigma = lambda2.ln() + log_sigma - lambda2 * log_sigma.exp();
        approx(lp, expect_rho + expect_sigma, 1e-12);
    }

    #[test]
    fn loggamma_rate_convention() {
        let shape = 3.0_f64;
        let rate = 0.5_f64;
        let theta = 0.0_f64;
        let expect = shape * rate.ln() - log_gamma(shape) + shape * theta - rate * theta.exp();
        let p = PriorSpec::loggamma(shape, rate);
        approx(p.log_density(&[theta]).unwrap(), expect, 1e-12);
        let e = p.eval1d(theta).unwrap();
        approx(e.grad, shape - rate, 1e-12);
    }

    #[test]
    fn gaussian_and_flat() {
        let g = PriorSpec::gaussian(0.0, 4.0).eval1d(0.2).unwrap();
        approx(g.grad, -0.8, 1e-12);
        approx(g.hess, -4.0, 1e-12);
        approx(PriorSpec::flat().log_density(&[1.23]).unwrap(), 0.0, 1e-15);
    }

    #[test]
    fn logitbeta_uniform_at_zero() {
        let p = PriorSpec::logitbeta(1.0, 1.0);
        let got = p.log_density(&[0.0]).unwrap();
        approx(got, -2.0 * std::f64::consts::LN_2, 1e-10);
    }

    #[test]
    fn hyper_stack_ar1_defaults() {
        let s = HyperPriorStack::default_for_effect("ar1").unwrap();
        assert_eq!(s.theta_dim(), 2);
        let lp = s.log_density(&[0.0, 0.0]).unwrap();
        assert!(lp.is_finite());
    }

    #[test]
    fn besag_with_pc_prec_inference() {
        use crate::{
            GaussianObs, Link, MarginalOptions, Obs, besag_precision_csc, run_inla_inference_a,
            sum_to_zero_constraint,
        };
        let adj = vec![vec![1usize], vec![0, 2], vec![1, 3], vec![2]];
        let n = adj.len();
        let y: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.3 })
            .collect();
        let obs: Vec<Obs> = y
            .iter()
            .map(|&yi| {
                Obs::Gaussian(GaussianObs {
                    y: yi,
                    precision: 10.0,
                    link: Link::Identity,
                })
            })
            .collect();
        let build_prior = |theta: &[f64]| besag_precision_csc(&adj, theta[0].exp());
        let stack = HyperPriorStack::default_for_effect("besag").unwrap();
        let log_prior = move |theta: &[f64]| stack.log_density(theta).unwrap_or(f64::NEG_INFINITY);
        let constr = sum_to_zero_constraint(n, 1).unwrap();
        let result = run_inla_inference_a(
            &[0.0],
            &build_prior,
            &log_prior,
            &obs,
            None,
            Some(&constr),
            "ccd",
            1.0,
            &MarginalOptions::default(),
            true,
        )
        .expect("besag+pc.prec");
        assert!(result.marginal_log_lik.is_finite());
        let s: f64 = result.latent_means.iter().sum();
        assert!(s.abs() < 1e-4, "sum={s}");
    }
}
