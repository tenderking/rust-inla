//! R-INLA-compatible hyperparameter priors on the **internal θ scale**.
//!
//! Densities match `inlaprog/src/inla-priors.c` (`PRIOR_EVAL`): log π(θ) already
//! includes the natural→internal Jacobian. Callers must not add another Jacobian.

use inla_math::Eval1D;

use crate::inference::log_gamma;

const LOG_NORMC_GAUSSIAN: f64 = -0.918_938_533_204_672_8; // -½ log(2π)

/// Named prior family with R-INLA parameter conventions.
#[derive(Debug, Clone, PartialEq)]
pub enum PriorFamily {
    /// PC prior on precision: `P(σ > u) = α`, θ = log τ, σ = τ^{-1/2}.
    PcPrec { u: f64, alpha: f64 },
    /// PC prior on AR(1) correlation (base ρ=1): `P(ρ > u) = α`.
    /// θ = logit((1+ρ)/2). R name: `pc.cor1` / `pc.rho1`.
    PcCor1 { u: f64, alpha: f64 },
    /// PC Matérn on (log range, log σ) after λ packing: param `(λ1, λ2, d)`.
    PcMatern {
        lambda1: f64,
        lambda2: f64,
        d: f64,
    },
    /// Gamma on τ = e^θ with **rate** `b` (R-INLA): mean = a/b.
    LogGamma { shape: f64, rate: f64 },
    /// Gaussian directly on θ: mean μ, precision τ (`τ=0` ⇒ flat).
    Gaussian { mean: f64, precision: f64 },
    /// Flat on θ: log π = 0.
    Flat,
    /// Beta(a,b) on p ∈ (0,1); θ = logit(p).
    LogitBeta { a: f64, b: f64 },
}

impl PriorFamily {
    /// Dimension of θ this prior expects.
    pub fn theta_dim(&self) -> usize {
        match self {
            PriorFamily::PcMatern { .. } => 2,
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

    pub fn pc_prec(u: f64, alpha: f64) -> Self {
        Self::new(PriorFamily::PcPrec { u, alpha })
    }

    pub fn pc_cor1(u: f64, alpha: f64) -> Self {
        Self::new(PriorFamily::PcCor1 { u, alpha })
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
        let family = match key.as_str() {
            "pcprec" => {
                let (u, alpha) = take2(param, 1.0, 0.01)?;
                PriorFamily::PcPrec { u, alpha }
            }
            "pccor1" | "pcrho1" => {
                let (u, alpha) = take2(param, 0.5, 0.75)?;
                PriorFamily::PcCor1 { u, alpha }
            }
            "pcmatern" => {
                if param.len() < 3 {
                    return Err("pc.matern requires param=c(lambda1, lambda2, d)".into());
                }
                PriorFamily::PcMatern {
                    lambda1: param[0],
                    lambda2: param[1],
                    d: param[2],
                }
            }
            "loggamma" => {
                let (shape, rate) = take2(param, 1.0, 5e-5)?;
                PriorFamily::LogGamma { shape, rate }
            }
            "gaussian" | "normal" => {
                let (mean, precision) = take2(param, 0.0, 0.001)?;
                PriorFamily::Gaussian { mean, precision }
            }
            "flat" | "uniform" => PriorFamily::Flat,
            "logitbeta" => {
                let (a, b) = take2(param, 1.0, 1.0)?;
                PriorFamily::LogitBeta { a, b }
            }
            other => return Err(format!("unknown prior '{other}' (from '{name}')")),
        };
        Ok(Self {
            name: key,
            family,
        })
    }

    pub fn theta_dim(&self) -> usize {
        self.family.theta_dim()
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
            PriorFamily::PcPrec { u, alpha } => Ok(pc_prec_log_dens(theta[0], *u, *alpha)?),
            PriorFamily::PcCor1 { u, alpha } => Ok(pc_cor1_log_dens(theta[0], *u, *alpha)?),
            PriorFamily::PcMatern {
                lambda1,
                lambda2,
                d,
            } => Ok(pc_matern_log_dens(theta[0], theta[1], *lambda1, *lambda2, *d)?),
            PriorFamily::LogGamma { shape, rate } => {
                Ok(loggamma_log_dens(theta[0], *shape, *rate)?)
            }
            PriorFamily::Gaussian { mean, precision } => {
                Ok(gaussian_log_dens(theta[0], *mean, *precision)?)
            }
            PriorFamily::Flat => Ok(0.0),
            PriorFamily::LogitBeta { a, b } => Ok(logitbeta_log_dens(theta[0], *a, *b)?),
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
            PriorFamily::PcPrec { u, alpha } => pc_prec_eval(theta, *u, *alpha),
            PriorFamily::PcCor1 { u, alpha } => {
                // Analytic hess is messy; FD hess from analytic grad.
                let logp = pc_cor1_log_dens(theta, *u, *alpha)?;
                let eps = 1e-5;
                let g0 = pc_cor1_log_dens(theta - eps, *u, *alpha)?;
                let g1 = pc_cor1_log_dens(theta + eps, *u, *alpha)?;
                let grad = (g1 - g0) / (2.0 * eps);
                let hess = (g1 - 2.0 * logp + g0) / (eps * eps);
                Ok(Eval1D { logp, grad, hess })
            }
            PriorFamily::LogGamma { shape, rate } => loggamma_eval(theta, *shape, *rate),
            PriorFamily::Gaussian { mean, precision } => {
                gaussian_eval(theta, *mean, *precision)
            }
            PriorFamily::Flat => Ok(Eval1D {
                logp: 0.0,
                grad: 0.0,
                hess: 0.0,
            }),
            PriorFamily::LogitBeta { a, b } => logitbeta_eval(theta, *a, *b),
            PriorFamily::PcMatern { .. } => unreachable!(),
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
            "iid" | "rw1" | "rw2" | "rw2d" | "besag" | "besag2" | "seasonal" | "crw1"
            | "crw2" => Ok(Self::new(vec![PriorSpec::pc_prec(1.0, 0.01)])),
            "bym" => Ok(Self::new(vec![
                PriorSpec::pc_prec(1.0, 0.01),
                PriorSpec::pc_prec(1.0, 0.01),
            ])),
            // θ = [log_tau, logit_phi]
            "bym2" => Ok(Self::new(vec![
                PriorSpec::pc_prec(1.0, 0.01),
                PriorSpec::gaussian(0.0, 0.5),
            ])),
            // θ = [log_prec, log_range]
            "matern2d" => Ok(Self::new(vec![
                PriorSpec::pc_prec(1.0, 0.01),
                PriorSpec::gaussian(0.0, 0.1),
            ])),
            "ar1" => Ok(Self::new(vec![
                PriorSpec::pc_prec(1.0, 0.01),
                PriorSpec::pc_cor1(0.5, 0.75),
            ])),
            "ar" | "arp" => Ok(Self::new(vec![
                PriorSpec::pc_prec(1.0, 0.01),
                PriorSpec::gaussian(0.0, 0.1),
                PriorSpec::gaussian(0.0, 0.1),
            ])),
            "fgn" => Ok(Self::new(vec![
                PriorSpec::pc_prec(1.0, 0.01),
                // Hurst on internal scale: weak Gaussian (no dedicated PC yet)
                PriorSpec::gaussian(0.0, 0.1),
            ])),
            // θ = [log_tau, log_kappa]; PC Matérn on range/σ scale (d=2).
            "spde" => Ok(Self::new(vec![PriorSpec::from_name_params(
                "pc.matern",
                &[1.0, 1.0, 2.0],
            )?])),
            "fixed" => Ok(Self::new(vec![])),
            other => Err(format!(
                "no default hyperprior for effect type '{other}'"
            )),
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
        self.priors
            .iter()
            .map(|p| (p.name.clone(), family_param_vec(&p.family)))
            .collect()
    }
}

fn family_param_vec(f: &PriorFamily) -> Vec<f64> {
    match f {
        PriorFamily::PcPrec { u, alpha } => vec![*u, *alpha],
        PriorFamily::PcCor1 { u, alpha } => vec![*u, *alpha],
        PriorFamily::PcMatern {
            lambda1,
            lambda2,
            d,
        } => vec![*lambda1, *lambda2, *d],
        PriorFamily::LogGamma { shape, rate } => vec![*shape, *rate],
        PriorFamily::Gaussian { mean, precision } => vec![*mean, *precision],
        PriorFamily::Flat => vec![],
        PriorFamily::LogitBeta { a, b } => vec![*a, *b],
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

fn family_canonical_name(f: &PriorFamily) -> &'static str {
    match f {
        PriorFamily::PcPrec { .. } => "pcprec",
        PriorFamily::PcCor1 { .. } => "pccor1",
        PriorFamily::PcMatern { .. } => "pcmatern",
        PriorFamily::LogGamma { .. } => "loggamma",
        PriorFamily::Gaussian { .. } => "gaussian",
        PriorFamily::Flat => "flat",
        PriorFamily::LogitBeta { .. } => "logitbeta",
    }
}

fn take2(param: &[f64], d0: f64, d1: f64) -> Result<(f64, f64), String> {
    match param.len() {
        0 => Ok((d0, d1)),
        1 => Ok((param[0], d1)),
        _ => Ok((param[0], param[1])),
    }
}

fn pc_prec_lambda(u: f64, alpha: f64) -> Result<f64, String> {
    if !(u > 0.0 && alpha > 0.0 && alpha < 1.0) {
        return Err(format!(
            "pc.prec: need u>0 and 0<alpha<1, got u={u} alpha={alpha}"
        ));
    }
    Ok(-alpha.ln() / u)
}

fn pc_prec_log_dens(theta: f64, u: f64, alpha: f64) -> Result<f64, String> {
    if !theta.is_finite() {
        return Err("pc.prec: θ must be finite".into());
    }
    let lambda = pc_prec_lambda(u, alpha)?;
    // log(λ/2) - λ e^{-θ/2} - θ/2
    Ok(lambda.ln() - std::f64::consts::LN_2 - lambda * (-0.5 * theta).exp() - 0.5 * theta)
}

fn pc_prec_eval(theta: f64, u: f64, alpha: f64) -> Result<Eval1D, String> {
    let lambda = pc_prec_lambda(u, alpha)?;
    let e = (-0.5 * theta).exp();
    let logp = lambda.ln() - std::f64::consts::LN_2 - lambda * e - 0.5 * theta;
    // d/dθ [-λ e^{-θ/2}] = λ*(1/2)*e^{-θ/2}
    let grad = 0.5 * lambda * e - 0.5;
    let hess = -0.25 * lambda * e;
    Ok(Eval1D { logp, grad, hess })
}

/// Solve λ from (1-e^{-λ√(1-u)})/(1-e^{-λ√2}) = α (R `inla.pc.cor1.lambda`).
fn pc_cor1_lambda(u: f64, alpha: f64) -> Result<f64, String> {
    if !(-1.0..1.0).contains(&u) {
        return Err(format!("pc.cor1: u must be in (-1,1), got {u}"));
    }
    let alpha_min = ((1.0 - u) / 2.0).sqrt();
    // R: alpha > alpha.min (strict)
    if !(alpha > alpha_min && alpha < 1.0) {
        return Err(format!(
            "pc.cor1: need alpha_min < alpha < 1 with alpha_min={alpha_min}, got {alpha}"
        ));
    }
    let fun = |lam: f64| -> f64 {
        let ff = (1.0 - (-lam * (1.0 - u).sqrt()).exp()) / (1.0 - (-lam * std::f64::consts::SQRT_2).exp());
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
    // Local golden-section refine
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

fn pc_cor1_log_dens(theta: f64, u: f64, alpha: f64) -> Result<f64, String> {
    if !theta.is_finite() {
        return Err("pc.cor1: θ must be finite".into());
    }
    let lambda = pc_cor1_lambda(u, alpha)?;
    // ρ = 2/(1+e^{-θ}) - 1 = tanh(θ/2) style via logit
    let e = theta.exp();
    let rho = 2.0 * e / (1.0 + e) - 1.0;
    let log_pi_rho = pc_cor1_log_dens_rho(rho, lambda)?;
    // log|dρ/dθ| = log 2 + θ - 2 log(1+e^θ)
    let log_j = std::f64::consts::LN_2 + theta - 2.0 * (1.0 + e).ln();
    Ok(log_pi_rho + log_j)
}

fn pc_matern_log_dens(
    theta1: f64,
    theta2: f64,
    lambda1: f64,
    lambda2: f64,
    d: f64,
) -> Result<f64, String> {
    if !(d > 0.0) {
        return Err(format!("pc.matern: d must be > 0, got {d}"));
    }
    let mut s = 0.0;
    if lambda1 > 0.0 && theta1.is_finite() {
        // log(λ1 d / 2) - (d/2) θ1 - λ1 exp(-(d/2) θ1)
        s += (lambda1 * d / 2.0).ln() - (d / 2.0) * theta1 - lambda1 * (-(d / 2.0) * theta1).exp();
    }
    if lambda2 > 0.0 && theta2.is_finite() {
        // log λ2 + θ2 - λ2 exp(θ2)
        s += lambda2.ln() + theta2 - lambda2 * theta2.exp();
    }
    Ok(s)
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
    fn trim_aliases() {
        let p = PriorSpec::from_name_params("pc.prec", &[1.0, 0.01]).unwrap();
        assert_eq!(p.name, "pcprec");
        let g = PriorSpec::from_name_params("normal", &[]).unwrap();
        assert!(matches!(
            g.family,
            PriorFamily::Gaussian {
                mean: 0.0,
                precision: 0.001
            }
        ));
    }

    #[test]
    fn pc_prec_matches_closed_form() {
        let u = 1.0_f64;
        let alpha = 0.01_f64;
        let lambda = -alpha.ln() / u;
        let theta = 0.5_f64;
        let expect = lambda.ln() - std::f64::consts::LN_2 - lambda * (-0.5 * theta).exp()
            - 0.5 * theta;
        let got = PriorSpec::pc_prec(u, alpha).log_density(&[theta]).unwrap();
        approx(got, expect, 1e-12);
        let e = PriorSpec::pc_prec(u, alpha).eval1d(theta).unwrap();
        approx(e.logp, expect, 1e-12);
        // FD grad
        let eps = 1e-6;
        let fd = (PriorSpec::pc_prec(u, alpha)
            .log_density(&[theta + eps])
            .unwrap()
            - PriorSpec::pc_prec(u, alpha)
                .log_density(&[theta - eps])
                .unwrap())
            / (2.0 * eps);
        approx(e.grad, fd, 1e-5);
    }

    #[test]
    fn loggamma_rate_convention() {
        // shape=3, rate=0.5 ⇒ scale=2 in old API; θ=0 → τ=1
        // logp = 3*log(0.5) - logΓ(3) + 2*0 - 0.5*1
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
        // Beta(1,1) flat on p; at θ=0, p=0.5, Jacobian log|J|=log(p(1-p))= -log(4)
        // log dens = 0 + 0 - 2 log(2) = -2 ln 2  (Beta dens=1)
        let p = PriorSpec::logitbeta(1.0, 1.0);
        let got = p.log_density(&[0.0]).unwrap();
        approx(got, -2.0 * std::f64::consts::LN_2, 1e-10);
    }

    #[test]
    fn pc_cor1_lambda_and_density_finite() {
        let lam = pc_cor1_lambda(0.5, 0.75).unwrap();
        assert!(lam > 0.0 && lam.is_finite(), "lam={lam}");
        let p = PriorSpec::pc_cor1(0.5, 0.75);
        let lp = p.log_density(&[0.0]).unwrap();
        assert!(lp.is_finite(), "lp={lp}");
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
        let y: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 0.5 } else { -0.3 }).collect();
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

    #[test]
    fn pc_matern_two_theta() {
        let p = PriorSpec::from_name_params("pc.matern", &[1.0, 1.0, 2.0]).unwrap();
        assert_eq!(p.theta_dim(), 2);
        let lp = p.log_density(&[0.0, 0.0]).unwrap();
        assert!(lp.is_finite());
    }
}
