use rayon::prelude::*;

use inla_math::{
    ConstraintMethod, ConstraintSpec, CscMatrix, Eval1D, FaerCpuSolver, HARD_CONSTRAINT_KAPPA,
    InlaSolver, LdltFactor, add_csc, augment_precision_csc, ccd_design, grid_design, identity_csc,
    invert_symmetric_matrix, jacobi_eigen, laplace_newton_step_a_solver, laplace_newton_system_a,
    matvec_csc, predictor_variances_diag, project_constraints,
};

use crate::options::ComputeOptions;
use crate::priors::PriorSpec;

#[cfg(test)]
use inla_math::{block_diag_csc, csc_from_triplets_0based};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Link {
    Identity,
    Log,
    Logit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroInflationType {
    Type0,
    Type1,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianPrior {
    pub mean: f64,
    pub precision: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaPrior {
    pub shape: f64,
    pub scale: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoissonObs {
    pub y: f64,
    pub exposure: f64,
    pub link: Link,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianObs {
    pub y: f64,
    pub precision: f64,
    pub link: Link,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinomialObs {
    pub y: f64,
    pub n: f64,
    pub link: Link,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NegativeBinomialObs {
    pub y: f64,
    pub exposure: f64,
    pub size: f64,
    pub link: Link,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZeroInflatedPoissonObs {
    pub y: f64,
    pub exposure: f64,
    pub zero_prob: f64,
    pub link: Link,
    pub inflation: ZeroInflationType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZeroInflatedBinomialObs {
    pub y: f64,
    pub n: f64,
    pub zero_prob: f64,
    pub link: Link,
    pub inflation: ZeroInflationType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LaplaceObs {
    pub y: f64,
    pub alpha: f64,
    pub gamma: f64,
    pub link: Link,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExponentialSurvivalObs {
    pub y: f64,
    /// R-INLA `inla.surv` codes: 0 right, 1 event, 2 left, 3 interval.
    pub event: f64,
    /// Upper time when `event == 3`; ignored otherwise.
    pub y_upper: f64,
    pub link: Link,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeibullSurvivalObs {
    pub y: f64,
    /// R-INLA `inla.surv` codes: 0 right, 1 event, 2 left, 3 interval.
    pub event: f64,
    /// Upper time when `event == 3`; ignored otherwise.
    pub y_upper: f64,
    pub shape: f64,
    /// `0` = proportional hazards `H=λ t^α` (R-INLA default); `1` = AFT `H=(λ t)^α`.
    pub variant: i32,
    pub link: Link,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoglogisticSurvivalObs {
    pub y: f64,
    pub event: f64,
    pub y_upper: f64,
    pub shape: f64,
    pub link: Link,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LognormalSurvivalObs {
    pub y: f64,
    pub event: f64,
    pub y_upper: f64,
    /// Precision of `log T` (R-INLA `lognormal.surv` hyperparameter, here observation-level).
    pub prec: f64,
    pub link: Link,
}

const LOG_NORMC_GAUSSIAN: f64 = -0.918_938_533_204_672_8;

pub fn eval_prior_gaussian(theta: f64, p: GaussianPrior) -> Result<Eval1D, String> {
    PriorSpec::gaussian(p.mean, p.precision).eval1d(theta)
}

pub fn eval_prior_gamma(x: f64, p: GammaPrior) -> Result<Eval1D, String> {
    if x <= 0.0 || !x.is_finite() {
        return Err("gamma prior is defined for finite x > 0".to_string());
    }
    if p.shape <= 0.0 || p.scale <= 0.0 || !p.shape.is_finite() || !p.scale.is_finite() {
        return Err("gamma prior shape/scale must be finite and > 0".to_string());
    }
    // Shape–scale parameterization (mean = shape * scale).
    let a = p.shape;
    let b = p.scale;
    let logp = (a - 1.0) * (x / b).ln() - x / b - log_gamma(a) - b.ln();
    let grad = (a - 1.0) / x - 1.0 / b;
    let hess = -(a - 1.0) / (x * x);
    Ok(Eval1D { logp, grad, hess })
}

/// Log-gamma prior on θ = log x with Gamma(shape, **scale**) on x (legacy API).
/// Prefer [`crate::priors::PriorSpec::loggamma`] which uses the R-INLA **rate** convention.
pub fn eval_prior_loggamma(theta: f64, p: GammaPrior) -> Result<Eval1D, String> {
    if p.scale <= 0.0 || !p.scale.is_finite() {
        return Err("log-gamma prior scale must be finite and > 0".to_string());
    }
    let rate = 1.0 / p.scale;
    PriorSpec::loggamma(p.shape, rate).eval1d(theta)
}

pub fn eval_likelihood_gaussian(eta: f64, o: GaussianObs) -> Result<Eval1D, String> {
    if o.precision <= 0.0 || !o.precision.is_finite() {
        return Err("gaussian observation precision must be finite and > 0".to_string());
    }
    if !eta.is_finite() || !o.y.is_finite() {
        return Err("gaussian observation eta/y must be finite".to_string());
    }

    let (mu, dmu, d2mu) = link_forward(eta, o.link)?;
    let resid = mu - o.y;
    let a = -0.5 * o.precision * resid * resid;
    Ok(Eval1D {
        logp: LOG_NORMC_GAUSSIAN + 0.5 * o.precision.ln() + a,
        grad: -o.precision * resid * dmu,
        hess: -o.precision * (dmu * dmu + resid * d2mu),
    })
}

pub fn eval_likelihood_poisson(eta: f64, o: PoissonObs) -> Result<Eval1D, String> {
    if o.exposure < 0.0 || !o.exposure.is_finite() {
        return Err("poisson exposure must be finite and >= 0".to_string());
    }
    if o.y < 0.0 || !o.y.is_finite() {
        return Err("poisson y must be finite and >= 0".to_string());
    }
    if !eta.is_finite() {
        return Err("poisson eta must be finite".to_string());
    }
    let (lambda, dl, d2l) = link_forward(eta, o.link)?;
    if lambda < 0.0 {
        return Err("poisson rate must be >= 0".to_string());
    }
    let mu = o.exposure * lambda;
    let dmu = o.exposure * dl;
    let logp = if mu > 0.0 {
        o.y * mu.ln() - mu - log_factorial(o.y)?
    } else if o.y == 0.0 {
        0.0
    } else {
        f64::NEG_INFINITY
    };
    let grad = if mu > 0.0 {
        (o.y / mu - 1.0) * dmu
    } else {
        -dmu
    };
    // Fisher scoring (expected Hessian): more stable than observed for Newton.
    let hess = if mu > 0.0 {
        -(dmu * dmu) / mu
    } else {
        -(o.exposure * d2l).abs() // fallback near mu=0
    };
    Ok(Eval1D { logp, grad, hess })
}

pub fn eval_likelihood_binomial(eta: f64, o: BinomialObs) -> Result<Eval1D, String> {
    if o.n < 0.0 || o.y < 0.0 || o.y > o.n || !o.n.is_finite() || !o.y.is_finite() {
        return Err("binomial requires finite 0 <= y <= n".to_string());
    }
    if !eta.is_finite() {
        return Err("binomial eta must be finite".to_string());
    }
    let (p, dp, _d2p) = link_forward(eta, o.link)?;
    if !(0.0..=1.0).contains(&p) {
        return Err("binomial probability out of [0,1]".to_string());
    }
    let n = o.n;
    let y = o.y;
    let log_choose = log_gamma(n + 1.0) - log_gamma(y + 1.0) - log_gamma(n - y + 1.0);
    let logp = if p == 0.0 {
        if y == 0.0 {
            log_choose
        } else {
            f64::NEG_INFINITY
        }
    } else if p == 1.0 {
        if y == n {
            log_choose
        } else {
            f64::NEG_INFINITY
        }
    } else {
        log_choose + y * p.ln() + (n - y) * (1.0 - p).ln()
    };

    let a = if p > 0.0 && p < 1.0 {
        y / p - (n - y) / (1.0 - p)
    } else {
        0.0
    };
    // Fisher scoring: E[hess] = -n * p * (1-p) = -n * dp for logit.
    let hess = -n * dp;
    Ok(Eval1D {
        logp,
        grad: a * dp,
        hess,
    })
}

pub fn eval_likelihood_negative_binomial(
    eta: f64,
    o: NegativeBinomialObs,
) -> Result<Eval1D, String> {
    if !eta.is_finite() {
        return Err("negative-binomial eta must be finite".to_string());
    }
    if o.y < 0.0 || !o.y.is_finite() {
        return Err("negative-binomial y must be finite and >= 0".to_string());
    }
    if o.exposure < 0.0 || !o.exposure.is_finite() {
        return Err("negative-binomial exposure must be finite and >= 0".to_string());
    }
    if o.size <= 0.0 || !o.size.is_finite() {
        return Err("negative-binomial size must be finite and > 0".to_string());
    }

    let (base_mu, dbase, d2base) = link_forward(eta, o.link)?;
    if base_mu <= 0.0 {
        return Err("negative-binomial mean must be > 0".to_string());
    }
    let mu = o.exposure * base_mu;
    if mu <= 0.0 {
        return Err("negative-binomial effective mean must be > 0".to_string());
    }
    let dmu = o.exposure * dbase;
    let d2mu = o.exposure * d2base;
    let r = o.size;
    let y = o.y;
    let mu_r = mu + r;

    let logp = log_gamma(y + r) - log_gamma(r) - log_gamma(y + 1.0)
        + r * (r / mu_r).ln()
        + y * (mu / mu_r).ln();
    let dlog_dmu = y / mu - (y + r) / mu_r;
    let d2log_dmu2 = -y / (mu * mu) + (y + r) / (mu_r * mu_r);

    Ok(Eval1D {
        logp,
        grad: dlog_dmu * dmu,
        hess: d2log_dmu2 * dmu * dmu + dlog_dmu * d2mu,
    })
}

pub fn eval_likelihood_zero_inflated_poisson(
    eta: f64,
    o: ZeroInflatedPoissonObs,
) -> Result<Eval1D, String> {
    if !eta.is_finite() {
        return Err("zero-inflated poisson eta must be finite".to_string());
    }
    if o.y < 0.0 || !o.y.is_finite() {
        return Err("zero-inflated poisson y must be finite and >= 0".to_string());
    }
    if o.exposure < 0.0 || !o.exposure.is_finite() {
        return Err("zero-inflated poisson exposure must be finite and >= 0".to_string());
    }
    if !(0.0..1.0).contains(&o.zero_prob) || !o.zero_prob.is_finite() {
        return Err("zero-inflation probability must be finite and in [0, 1)".to_string());
    }

    let (base_mu, dbase, d2base) = link_forward(eta, o.link)?;
    if base_mu < 0.0 {
        return Err("poisson mean must be >= 0".to_string());
    }
    let adjusted_mu = match o.inflation {
        ZeroInflationType::Type0 => base_mu,
        ZeroInflationType::Type1 => base_mu / (1.0 - o.zero_prob),
    };
    let lambda = o.exposure * adjusted_mu;
    let dlambda = o.exposure
        * match o.inflation {
            ZeroInflationType::Type0 => dbase,
            ZeroInflationType::Type1 => dbase / (1.0 - o.zero_prob),
        };
    let d2lambda = o.exposure
        * match o.inflation {
            ZeroInflationType::Type0 => d2base,
            ZeroInflationType::Type1 => d2base / (1.0 - o.zero_prob),
        };
    if lambda < 0.0 {
        return Err("poisson rate must be >= 0".to_string());
    }

    let f0 = (-lambda).exp();
    if o.y == 0.0 {
        let s = o.zero_prob + (1.0 - o.zero_prob) * f0;
        let ds_dlambda = -(1.0 - o.zero_prob) * f0;
        let d2s_dlambda2 = (1.0 - o.zero_prob) * f0;
        let dlog_dlambda = ds_dlambda / s;
        let d2log_dlambda2 = d2s_dlambda2 / s - dlog_dlambda * dlog_dlambda;
        return Ok(Eval1D {
            logp: s.ln(),
            grad: dlog_dlambda * dlambda,
            hess: d2log_dlambda2 * dlambda * dlambda + dlog_dlambda * d2lambda,
        });
    }

    if lambda <= 0.0 {
        return Ok(Eval1D {
            logp: f64::NEG_INFINITY,
            grad: f64::NAN,
            hess: f64::NAN,
        });
    }
    let y = o.y;
    let logp = (1.0 - o.zero_prob).ln() + y * lambda.ln() - lambda - log_gamma(y + 1.0);
    let dlog_dlambda = y / lambda - 1.0;
    let d2log_dlambda2 = -y / (lambda * lambda);
    Ok(Eval1D {
        logp,
        grad: dlog_dlambda * dlambda,
        hess: d2log_dlambda2 * dlambda * dlambda + dlog_dlambda * d2lambda,
    })
}

pub fn eval_likelihood_zero_inflated_binomial(
    eta: f64,
    o: ZeroInflatedBinomialObs,
) -> Result<Eval1D, String> {
    if !eta.is_finite() {
        return Err("zero-inflated binomial eta must be finite".to_string());
    }
    if o.n < 0.0 || o.y < 0.0 || o.y > o.n || !o.n.is_finite() || !o.y.is_finite() {
        return Err("zero-inflated binomial requires finite 0 <= y <= n".to_string());
    }
    if !(0.0..1.0).contains(&o.zero_prob) || !o.zero_prob.is_finite() {
        return Err("zero-inflation probability must be finite and in [0, 1)".to_string());
    }
    let (base_p, dp_base, d2p_base) = link_forward(eta, o.link)?;
    if !(0.0..=1.0).contains(&base_p) {
        return Err("binomial probability out of [0,1]".to_string());
    }

    let (p, dp, d2p) = match o.inflation {
        ZeroInflationType::Type0 => (base_p, dp_base, d2p_base),
        ZeroInflationType::Type1 => {
            if base_p > (1.0 - o.zero_prob) {
                return Err(
                    "type1 zero-inflated binomial requires base probability <= 1 - zero_prob"
                        .to_string(),
                );
            }
            (
                base_p / (1.0 - o.zero_prob),
                dp_base / (1.0 - o.zero_prob),
                d2p_base / (1.0 - o.zero_prob),
            )
        }
    };
    if !(0.0..=1.0).contains(&p) {
        return Err("adjusted binomial probability out of [0,1]".to_string());
    }

    let n = o.n;
    let y = o.y;
    let log_choose = log_gamma(n + 1.0) - log_gamma(y + 1.0) - log_gamma(n - y + 1.0);

    if y == 0.0 {
        let f0 = (1.0 - p).powf(n);
        let s = o.zero_prob + (1.0 - o.zero_prob) * f0;
        let df0_dp = -n * (1.0 - p).powf(n - 1.0);
        let d2f0_dp2 = if n >= 1.0 {
            n * (n - 1.0) * (1.0 - p).powf(n - 2.0)
        } else {
            0.0
        };
        let ds_dp = (1.0 - o.zero_prob) * df0_dp;
        let d2s_dp2 = (1.0 - o.zero_prob) * d2f0_dp2;
        let dlog_dp = ds_dp / s;
        let d2log_dp2 = d2s_dp2 / s - dlog_dp * dlog_dp;
        return Ok(Eval1D {
            logp: s.ln(),
            grad: dlog_dp * dp,
            hess: d2log_dp2 * dp * dp + dlog_dp * d2p,
        });
    }

    if p == 0.0 || p == 1.0 {
        return Ok(Eval1D {
            logp: f64::NEG_INFINITY,
            grad: f64::NAN,
            hess: f64::NAN,
        });
    }
    let log_base = log_choose + y * p.ln() + (n - y) * (1.0 - p).ln();
    let dlog_dp = y / p - (n - y) / (1.0 - p);
    let d2log_dp2 = -y / (p * p) - (n - y) / ((1.0 - p) * (1.0 - p));
    Ok(Eval1D {
        logp: (1.0 - o.zero_prob).ln() + log_base,
        grad: dlog_dp * dp,
        hess: d2log_dp2 * dp * dp + dlog_dp * d2p,
    })
}

pub fn eval_likelihood_laplace(eta: f64, o: LaplaceObs) -> Result<Eval1D, String> {
    if !eta.is_finite() || !o.y.is_finite() {
        return Err("laplace eta/y must be finite".to_string());
    }
    if !(0.0..1.0).contains(&o.alpha) || !o.alpha.is_finite() {
        return Err("laplace quantile alpha must be finite and in (0,1)".to_string());
    }
    if o.gamma <= 0.0 || !o.gamma.is_finite() {
        return Err("laplace smoothing gamma must be finite and > 0".to_string());
    }

    let (mu, dmu, d2mu) = link_forward(eta, o.link)?;
    let x = o.y - mu;
    let s = (x * x + o.gamma * o.gamma).sqrt();
    let rho = 0.5 * ((2.0 * o.alpha - 1.0) * x + s);
    let rho_dx = 0.5 * ((2.0 * o.alpha - 1.0) + x / s);
    let rho_d2x = 0.5 * o.gamma * o.gamma / (s * s * s);
    Ok(Eval1D {
        logp: -rho,
        grad: rho_dx * dmu,
        hess: -rho_d2x * dmu * dmu + rho_dx * d2mu,
    })
}

pub fn eval_likelihood_exponential_survival(
    eta: f64,
    o: ExponentialSurvivalObs,
) -> Result<Eval1D, String> {
    if !eta.is_finite() || !o.y.is_finite() {
        return Err("exponential survival eta/y must be finite".to_string());
    }
    if o.y < 0.0 {
        return Err("exponential survival time must be >= 0".to_string());
    }
    validate_event_indicator(o.event, "exponential survival event")?;

    let (rate, drate, d2rate) = link_forward(eta, o.link)?;
    if rate <= 0.0 {
        return Err("exponential survival rate must be > 0".to_string());
    }

    let (logp, dlog_drate, d2log_drate2) = match o.event as i32 {
        0 => (-rate * o.y, -o.y, 0.0),
        1 => (
            o.event * rate.ln() - rate * o.y,
            1.0 / rate - o.y,
            -1.0 / (rate * rate),
        ),
        2 => {
            // F(t) = 1 - exp(-λ t)
            let u = rate * o.y;
            let logp = log1mexp(u)?;
            let emu = u.exp();
            let d1 = o.y / (emu - 1.0);
            let d2 = -o.y * o.y * emu / ((emu - 1.0) * (emu - 1.0));
            (logp, d1, d2)
        }
        3 => {
            if !o.y_upper.is_finite() || o.y_upper <= o.y {
                return Err("interval-censored exponential survival needs y_upper > y".into());
            }
            let delta = o.y_upper - o.y;
            let u = rate * delta;
            let logp = -rate * o.y + log1mexp(u)?;
            let emu = u.exp();
            let d1 = -o.y + delta / (emu - 1.0);
            let d2 = -delta * delta * emu / ((emu - 1.0) * (emu - 1.0));
            (logp, d1, d2)
        }
        _ => unreachable!(),
    };

    Ok(Eval1D {
        logp,
        grad: dlog_drate * drate,
        hess: d2log_drate2 * drate * drate + dlog_drate * d2rate,
    })
}

pub fn eval_likelihood_weibull_survival(eta: f64, o: WeibullSurvivalObs) -> Result<Eval1D, String> {
    if !eta.is_finite() || !o.y.is_finite() {
        return Err("weibull survival eta/y must be finite".to_string());
    }
    if o.y <= 0.0 {
        return Err("weibull survival time must be > 0".to_string());
    }
    if o.shape <= 0.0 || !o.shape.is_finite() {
        return Err("weibull shape must be finite and > 0".to_string());
    }
    validate_event_indicator(o.event, "weibull survival event")?;

    let (lambda, dlambda, d2lambda) = link_forward(eta, o.link)?;
    if lambda <= 0.0 {
        return Err("weibull scale/rate parameter must be > 0".to_string());
    }

    let k = o.shape;
    let (h, dh, d2h) = weibull_cumhaz(lambda, k, o.y, o.variant)?;
    let log_f = weibull_log_density(lambda, k, o.y, h, o.variant)?;
    let d_log_f = weibull_d_log_density(lambda, k, o.y, h, dh, o.variant);
    let d2_log_f = weibull_d2_log_density(lambda, k, o.y, h, dh, d2h, o.variant);

    let (logp, dlog_dlambda, d2log_dlambda2) = match o.event as i32 {
        0 => (-h, -dh, -d2h),
        1 => (log_f, d_log_f, d2_log_f),
        2 => survival_left_from_cumhaz(h, dh, d2h)?,
        3 => {
            if !o.y_upper.is_finite() || o.y_upper <= o.y {
                return Err("interval-censored weibull survival needs y_upper > y".into());
            }
            let (hu, dhu, d2hu) = weibull_cumhaz(lambda, k, o.y_upper, o.variant)?;
            survival_interval_from_cumhaz(h, dh, d2h, hu, dhu, d2hu)?
        }
        _ => unreachable!(),
    };

    Ok(Eval1D {
        logp,
        grad: dlog_dlambda * dlambda,
        hess: d2log_dlambda2 * dlambda * dlambda + dlog_dlambda * d2lambda,
    })
}

pub fn eval_likelihood_loglogistic_survival(
    eta: f64,
    o: LoglogisticSurvivalObs,
) -> Result<Eval1D, String> {
    if !eta.is_finite() || !o.y.is_finite() {
        return Err("loglogistic survival eta/y must be finite".to_string());
    }
    if o.y <= 0.0 {
        return Err("loglogistic survival time must be > 0".to_string());
    }
    if o.shape <= 0.0 || !o.shape.is_finite() {
        return Err("loglogistic shape must be finite and > 0".to_string());
    }
    validate_event_indicator(o.event, "loglogistic survival event")?;
    let (scale, dscale, d2scale) = link_forward(eta, o.link)?;
    if scale <= 0.0 {
        return Err("loglogistic scale must be > 0".to_string());
    }
    let (logp, d1, d2) = match o.event as i32 {
        0 => loglogistic_right(o.y, scale, o.shape)?,
        1 => loglogistic_event(o.y, scale, o.shape)?,
        2 => loglogistic_left(o.y, scale, o.shape)?,
        3 => {
            if !o.y_upper.is_finite() || o.y_upper <= o.y {
                return Err("interval-censored loglogistic survival needs y_upper > y".into());
            }
            loglogistic_interval(o.y, o.y_upper, scale, o.shape)?
        }
        _ => unreachable!(),
    };
    Ok(Eval1D {
        logp,
        grad: d1 * dscale,
        hess: d2 * dscale * dscale + d1 * d2scale,
    })
}

pub fn eval_likelihood_lognormal_survival(
    eta: f64,
    o: LognormalSurvivalObs,
) -> Result<Eval1D, String> {
    if !eta.is_finite() || !o.y.is_finite() {
        return Err("lognormal survival eta/y must be finite".to_string());
    }
    if o.y <= 0.0 {
        return Err("lognormal survival time must be > 0".to_string());
    }
    if o.prec <= 0.0 || !o.prec.is_finite() {
        return Err("lognormal precision must be finite and > 0".to_string());
    }
    validate_event_indicator(o.event, "lognormal survival event")?;
    let (mu, dmu, d2mu) = link_forward(eta, o.link)?;
    let sigma = o.prec.sqrt().recip();
    let (logp, d1, d2) = match o.event as i32 {
        0 => lognormal_right(o.y, mu, sigma, o.prec)?,
        1 => lognormal_event(o.y, mu, sigma, o.prec),
        2 => lognormal_left(o.y, mu, sigma)?,
        3 => {
            if !o.y_upper.is_finite() || o.y_upper <= o.y {
                return Err("interval-censored lognormal survival needs y_upper > y".into());
            }
            lognormal_interval(o.y, o.y_upper, mu, sigma)?
        }
        _ => unreachable!(),
    };
    Ok(Eval1D {
        logp,
        grad: d1 * dmu,
        hess: d2 * dmu * dmu + d1 * d2mu,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Obs {
    Poisson(PoissonObs),
    Gaussian(GaussianObs),
    Binomial(BinomialObs),
    NegativeBinomial(NegativeBinomialObs),
    ZeroInflatedPoisson(ZeroInflatedPoissonObs),
    ZeroInflatedBinomial(ZeroInflatedBinomialObs),
    Laplace(LaplaceObs),
    ExponentialSurvival(ExponentialSurvivalObs),
    WeibullSurvival(WeibullSurvivalObs),
    LoglogisticSurvival(LoglogisticSurvivalObs),
    LognormalSurvival(LognormalSurvivalObs),
    None,
}

pub fn eval_likelihood(eta: f64, o: &Obs) -> Result<Eval1D, String> {
    match o {
        Obs::Poisson(po) => eval_likelihood_poisson(eta, *po),
        Obs::Gaussian(go) => eval_likelihood_gaussian(eta, *go),
        Obs::Binomial(bo) => eval_likelihood_binomial(eta, *bo),
        Obs::NegativeBinomial(nbo) => eval_likelihood_negative_binomial(eta, *nbo),
        Obs::ZeroInflatedPoisson(zip) => eval_likelihood_zero_inflated_poisson(eta, *zip),
        Obs::ZeroInflatedBinomial(zib) => eval_likelihood_zero_inflated_binomial(eta, *zib),
        Obs::Laplace(lo) => eval_likelihood_laplace(eta, *lo),
        Obs::ExponentialSurvival(eso) => eval_likelihood_exponential_survival(eta, *eso),
        Obs::WeibullSurvival(wso) => eval_likelihood_weibull_survival(eta, *wso),
        Obs::LoglogisticSurvival(lo) => eval_likelihood_loglogistic_survival(eta, *lo),
        Obs::LognormalSurvival(lo) => eval_likelihood_lognormal_survival(eta, *lo),
        Obs::None => Ok(Eval1D {
            logp: 0.0,
            grad: 0.0,
            hess: 0.0,
        }),
    }
}

pub fn find_latent_mode(
    q_prior: &CscMatrix,
    obs: &[Obs],
    max_iter: usize,
    tol: f64,
) -> Result<(Vec<f64>, LdltFactor, f64), String> {
    find_latent_mode_a(q_prior, obs, None, None, max_iter, tol)
}

fn latent_objective(
    q_prior: &CscMatrix,
    obs: &[Obs],
    a: Option<&CscMatrix>,
    x: &[f64],
) -> Result<f64, String> {
    let eta = match a {
        None => x.to_vec(),
        Some(a_mat) => matvec_csc(a_mat, x)?,
    };
    let mut log_lik = 0.0;
    for i in 0..obs.len() {
        let e = eval_likelihood(eta[i], &obs[i])?;
        if !e.logp.is_finite() {
            return Err("non-finite log-likelihood".to_string());
        }
        log_lik += e.logp;
    }
    let mut q_x = vec![0.0; x.len()];
    for (col, colvec) in q_prior.outer_iterator().enumerate() {
        for (row, value) in colvec.iter() {
            q_x[row] += value * x[col];
        }
    }
    let quad = q_x.iter().zip(x).map(|(a, b)| a * b).sum::<f64>();
    Ok(log_lik - 0.5 * quad)
}

/// Mode of π(x | θ, y) with optional projector `A` (`η = A x`) and linear constraints.
///
/// Uses [`FaerCpuSolver`] internally; prefer [`find_latent_mode_a_with_solver`] to inject
/// a custom [`InlaSolver`] (e.g. future GPU / sTiles backends).
pub fn find_latent_mode_a(
    q_prior: &CscMatrix,
    obs: &[Obs],
    a: Option<&CscMatrix>,
    constraints: Option<&ConstraintSpec>,
    max_iter: usize,
    tol: f64,
) -> Result<(Vec<f64>, LdltFactor, f64), String> {
    let mut solver = FaerCpuSolver::new();
    let (x, mlik) =
        find_latent_mode_a_with_solver(q_prior, obs, a, constraints, max_iter, tol, &mut solver)?;
    let factor = solver
        .into_factor()
        .ok_or_else(|| "Failed to factorize posterior precision".to_string())?;
    Ok((x, factor, mlik))
}

/// Like [`find_latent_mode_a`], but threads an [`InlaSolver`] through Newton and prior log-det.
///
/// On success, `solver` holds the posterior precision factorization (for `diag_inv` / `log_abs_det`).
pub fn find_latent_mode_a_with_solver(
    q_prior: &CscMatrix,
    obs: &[Obs],
    a: Option<&CscMatrix>,
    constraints: Option<&ConstraintSpec>,
    max_iter: usize,
    tol: f64,
    solver: &mut dyn InlaSolver,
) -> Result<(Vec<f64>, f64), String> {
    let n = q_prior.rows();
    if q_prior.cols() != n {
        return Err("prior precision must be square".to_string());
    }
    match a {
        None => {
            if obs.len() != n {
                return Err("Observations length must match prior precision dimension".to_string());
            }
        }
        Some(a_mat) => {
            if a_mat.cols() != n {
                return Err("A.ncols must equal latent dimension".to_string());
            }
            if a_mat.rows() != obs.len() {
                return Err("A.nrows must equal number of observations".to_string());
            }
        }
    }
    if let Some(c) = constraints {
        c.validate()?;
        if c.n != n {
            return Err(format!(
                "constraints n={} does not match latent dimension {n}",
                c.n
            ));
        }
        if c.method == ConstraintMethod::LagrangeElimination {
            return Err(
                "ConstraintMethod::LagrangeElimination is not yet implemented; use Augmented"
                    .into(),
            );
        }
    }

    // Working prior: Q + κ AᵀA when constrained (R-INLA extraconstr-style).
    let q_aug;
    let q_work = match constraints {
        Some(c) => {
            debug_assert_eq!(c.method, ConstraintMethod::Augmented);
            q_aug = augment_precision_csc(q_prior, c, HARD_CONSTRAINT_KAPPA)?;
            &q_aug
        }
        None => q_prior,
    };

    // Prior log-det first (solver will be overwritten by Newton posterior factors).
    let log_det_prior = if constraints.is_none() {
        let eye = identity_csc(n, 1e-12).map_err(|e| e.to_string())?;
        let q_jitter = add_csc(q_work, &eye).map_err(|e| e.to_string())?;
        solver.factorize(&q_jitter).map_err(|e| e.to_string())?;
        solver.log_abs_det().map_err(|e| e.to_string())?
    } else {
        solver.factorize(q_work).map_err(|e| e.to_string())?;
        solver.log_abs_det().map_err(|e| e.to_string())?
    };

    let mut x = vec![0.0; n];
    let mut converged = false;

    for iter in 0..max_iter {
        let eta = match a {
            None => x.clone(),
            Some(a_mat) => matvec_csc(a_mat, &x)?,
        };
        let mut evals = Vec::with_capacity(obs.len());
        for i in 0..obs.len() {
            evals.push(eval_likelihood(eta[i], &obs[i])?);
        }

        let step = laplace_newton_step_a_solver(q_work, &evals, a, &x, solver)
            .map_err(|e| e.to_string())?;
        if step.iter().any(|s| !s.is_finite()) {
            return Err("Newton-Raphson step is not finite (contains NaN or Inf)".to_string());
        }

        // Cap large steps (GLM Newton can overshoot from x=0 with weak curvature).
        let max_step = step.iter().fold(0.0_f64, |m, s| m.max(s.abs()));
        let mut alpha = if max_step > 10.0 {
            10.0 / max_step
        } else {
            1.0
        };

        // Backtrack only when the trial point yields a non-finite objective.
        let mut x_trial = x.clone();
        for _ in 0..12 {
            for i in 0..n {
                x_trial[i] = x[i] + alpha * step[i];
            }
            if let Some(c) = constraints {
                project_constraints(&mut x_trial, c)?;
            }
            if latent_objective(q_work, obs, a, &x_trial).is_ok() {
                break;
            }
            alpha *= 0.5;
            if alpha < 1e-8 {
                return Err("Newton-Raphson step is not finite (contains NaN or Inf)".to_string());
            }
        }

        let mut max_diff = 0.0;
        for i in 0..n {
            let dx = x_trial[i] - x[i];
            max_diff = f64::max(max_diff, dx.abs());
            x[i] = x_trial[i];
        }

        if max_diff < tol || max_step < tol {
            // Recompute Newton system at the converged point so solver holds Q_post(x*).
            let eta = match a {
                None => x.clone(),
                Some(a_mat) => matvec_csc(a_mat, &x)?,
            };
            let mut evals = Vec::with_capacity(obs.len());
            for i in 0..obs.len() {
                evals.push(eval_likelihood(eta[i], &obs[i])?);
            }
            let _ = laplace_newton_step_a_solver(q_work, &evals, a, &x, solver)
                .map_err(|e| e.to_string())?;
            converged = true;
            break;
        }

        if iter == max_iter - 1 {
            return Err("Newton-Raphson did not converge".to_string());
        }
    }

    if !converged {
        return Err("Failed to factorize posterior precision".to_string());
    }

    if let Some(c) = constraints {
        project_constraints(&mut x, c)?;
    }

    let log_det_post = solver.log_abs_det().map_err(|e| e.to_string())?;

    let mut q_x = vec![0.0; n];
    for (col, colvec) in q_work.outer_iterator().enumerate() {
        for (row, value) in colvec.iter() {
            q_x[row] += value * x[col];
        }
    }
    let quad_prior = q_x.iter().zip(&x).map(|(a, b)| a * b).sum::<f64>();

    let eta = match a {
        None => x.clone(),
        Some(a_mat) => matvec_csc(a_mat, &x)?,
    };
    let mut log_lik = 0.0;
    for i in 0..obs.len() {
        log_lik += eval_likelihood(eta[i], &obs[i])?.logp;
    }

    let marginal_log_lik = log_lik - 0.5 * quad_prior + 0.5 * log_det_prior - 0.5 * log_det_post;

    Ok((x, marginal_log_lik))
}

#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub mode: Vec<f64>,
    pub hessian: Vec<f64>,
    pub latent_means: Vec<f64>,
    pub latent_variances: Vec<f64>,
    /// Linear predictor means η̄ = A x̄ (length = n_obs)
    pub predictor_means: Vec<f64>,
    /// Approx predictor variances diag(A Σ Aᵀ) using diagonal Σ
    pub predictor_variances: Vec<f64>,
    pub marginal_log_lik: f64,
    /// Gaussian approximation to the log-marginal-likelihood
    pub marginal_log_lik_gaussian: f64,
    /// DIC = D̄ + p_D
    pub dic: f64,
    /// Posterior mean deviance D̄
    pub mean_deviance: f64,
    /// Effective number of parameters p_D = D̄ − D(θ*)
    pub effective_params: f64,
    /// WAIC = −2 (lppd − p_waic)
    pub waic: f64,
    /// Log pointwise predictive density used in WAIC
    pub waic_lppd: f64,
    /// WAIC effective number of parameters p_waic
    pub waic_effective_params: f64,
    /// CPO_i = π(y_i | y_{-i}), None when computation fails
    pub cpo: Vec<Option<f64>>,
    /// PIT_i = Pr(y^new_i ≤ y_i | y_{-i}), None when fails or unsupported family
    pub pit: Vec<Option<f64>>,
    /// Number of CPO failures
    pub cpo_n_failures: usize,
    /// Integration nodes θ_k (internal scale).
    pub theta_nodes: Vec<Vec<f64>>,
    /// Normalized integration weights Δ_k π̃(θ_k|y).
    pub node_weights: Vec<f64>,
    /// Internal-scale hyperparameter 1D marginals (one per θ_j). Empty if disabled.
    pub internal_marginals_hyperpar: Vec<crate::marginals::Marginal1D>,
    /// Opt-in latent mixture density grids (order matches `marginals_latent_indices`).
    pub marginals_latent: Vec<crate::marginals::Marginal1D>,
    /// Opt-in predictor mixture density grids.
    pub marginals_predictor: Vec<crate::marginals::Marginal1D>,
    /// Indices used for `marginals_latent`.
    pub marginals_latent_indices: Vec<usize>,
    /// Indices used for `marginals_predictor`.
    pub marginals_predictor_indices: Vec<usize>,
    /// Posterior precision \(Q(\hat\theta)\) at the Laplace mode (for lincomb / sampling).
    pub posterior_precision: Option<CscMatrix>,
}

/// Assemble \(Q_{\mathrm{post}} = Q_{\mathrm{prior}} + A^\top (-\ell'') A\) at a latent mode.
fn posterior_precision_at(
    q_prior: &CscMatrix,
    obs: &[Obs],
    a: Option<&CscMatrix>,
    constraints: Option<&ConstraintSpec>,
    x: &[f64],
) -> Result<CscMatrix, String> {
    let q_aug;
    let q_work = if let Some(c) = constraints {
        q_aug = augment_precision_csc(q_prior, c, HARD_CONSTRAINT_KAPPA)?;
        &q_aug
    } else {
        q_prior
    };
    let eta = match a {
        None => x.to_vec(),
        Some(am) => matvec_csc(am, x)?,
    };
    if eta.len() != obs.len() {
        return Err(format!(
            "posterior Q: eta length {} != n_obs {}",
            eta.len(),
            obs.len()
        ));
    }
    let mut evals = Vec::with_capacity(obs.len());
    for i in 0..obs.len() {
        evals.push(eval_likelihood(eta[i], &obs[i])?)
    }
    let (q_post, _) = laplace_newton_system_a(q_work, &evals, a, x).map_err(|e| e.to_string())?;
    Ok(q_post)
}

pub fn run_inla_inference(
    initial_theta: &[f64],
    build_prior: &(dyn Fn(&[f64]) -> Result<CscMatrix, String> + Sync),
    log_prior_density: &(dyn Fn(&[f64]) -> f64 + Sync),
    obs: &[Obs],
    strategy: &str,
    step_or_f0: f64,
) -> Result<InferenceResult, String> {
    run_inla_inference_a(
        initial_theta,
        build_prior,
        log_prior_density,
        obs,
        None,
        None,
        strategy,
        step_or_f0,
        &crate::marginals::MarginalOptions::default(),
        false,
    )
}

/// End-to-end INLA with optional observation projector `A` (`η = A x`) and constraints.
///
/// When `deterministic` is true, CCD/grid node evaluation is sequential (stable ordering /
/// bit-reproducible on a given machine) instead of Rayon-parallel.
pub fn run_inla_inference_a(
    initial_theta: &[f64],
    build_prior: &(dyn Fn(&[f64]) -> Result<CscMatrix, String> + Sync),
    log_prior_density: &(dyn Fn(&[f64]) -> f64 + Sync),
    obs: &[Obs],
    a: Option<&CscMatrix>,
    constraints: Option<&ConstraintSpec>,
    strategy: &str,
    step_or_f0: f64,
    marginal_opts: &crate::marginals::MarginalOptions,
    deterministic: bool,
) -> Result<InferenceResult, String> {
    run_inla_inference_a_cancellable(
        initial_theta,
        build_prior,
        log_prior_density,
        obs,
        a,
        constraints,
        strategy,
        step_or_f0,
        marginal_opts,
        deterministic,
        None,
        None,
        None,
    )
}

/// End-to-end INLA with optional observation projector `A` (`η = A x`), constraints, and cancellation callback.
pub fn run_inla_inference_a_cancellable(
    initial_theta: &[f64],
    build_prior: &(dyn Fn(&[f64]) -> Result<CscMatrix, String> + Sync),
    log_prior_density: &(dyn Fn(&[f64]) -> f64 + Sync),
    obs: &[Obs],
    a: Option<&CscMatrix>,
    constraints: Option<&ConstraintSpec>,
    strategy: &str,
    step_or_f0: f64,
    marginal_opts: &crate::marginals::MarginalOptions,
    deterministic: bool,
    check_cancel: Option<&(dyn Fn() -> Result<(), String> + Sync)>,
    build_obs: Option<&(dyn Fn(&[f64]) -> Vec<Obs> + Sync)>,
    compute: Option<&ComputeOptions>,
) -> Result<InferenceResult, String> {
    let compute = compute.cloned().unwrap_or_default();
    let m = initial_theta.len();
    let n_obs = obs.len();

    if m == 0 {
        // Pure fixed-effects / no hyperparameters: single Laplace node (no NM/CCD).
        let mut solver = FaerCpuSolver::new();
        let q_prior = build_prior(&[])?;
        let (x_star, marginal_log_lik) =
            find_latent_mode_a_with_solver(&q_prior, obs, a, constraints, 200, 1e-5, &mut solver)?;
        let variances = solver.diag_inv().map_err(|e| e.to_string())?;
        let eta = match a {
            None => x_star.clone(),
            Some(a_mat) => inla_math::matvec_csc(a_mat, &x_star)?,
        };
        let eta_var = match a {
            None => variances.clone(),
            Some(a_mat) => predictor_variances_diag(a_mat, &variances)?,
        };

        // Single-node model selection (weight 1 at the Laplace mode).
        let cond_eta = vec![eta.clone()];
        let cond_eta_var = vec![eta_var.clone()];
        let norm_weights = vec![1.0];
        let dic_result = if compute.dic {
            crate::model_selection::compute_dic(obs, &cond_eta, &norm_weights, 0)?
        } else {
            crate::model_selection::DicResult {
                dic: f64::NAN,
                mean_deviance: f64::NAN,
                effective_params: f64::NAN,
            }
        };
        let waic_result = if compute.waic {
            crate::model_selection::compute_waic(obs, &cond_eta, &norm_weights)?
        } else {
            crate::model_selection::WaicResult {
                waic: f64::NAN,
                lppd: f64::NAN,
                effective_params: f64::NAN,
            }
        };
        let cpo_result = if compute.cpo {
            crate::model_selection::compute_cpo_pit(obs, &cond_eta, &cond_eta_var, &norm_weights)?
        } else {
            crate::model_selection::CpoResult {
                cpo: vec![None; n_obs],
                pit: vec![None; n_obs],
                n_failures: 0,
            }
        };

        let mut marginals_latent = Vec::with_capacity(marginal_opts.latent_indices.len());
        for &idx in &marginal_opts.latent_indices {
            if idx >= x_star.len() {
                return Err(format!(
                    "latent marginal index {idx} out of range (n={})",
                    x_star.len()
                ));
            }
            marginals_latent.push(crate::marginals::gaussian_mixture_marginal(
                &[x_star[idx]],
                &[variances[idx]],
                &norm_weights,
                marginal_opts.n_points,
                marginal_opts.n_sd,
            )?);
        }
        let mut marginals_predictor = Vec::with_capacity(marginal_opts.predictor_indices.len());
        for &idx in &marginal_opts.predictor_indices {
            if idx >= n_obs {
                return Err(format!(
                    "predictor marginal index {idx} out of range (n_obs={n_obs})"
                ));
            }
            marginals_predictor.push(crate::marginals::gaussian_mixture_marginal(
                &[eta[idx]],
                &[eta_var[idx]],
                &norm_weights,
                marginal_opts.n_points,
                marginal_opts.n_sd,
            )?);
        }

        let q_post = posterior_precision_at(&q_prior, obs, a, constraints, &x_star)?;
        return Ok(InferenceResult {
            mode: Vec::new(),
            hessian: Vec::new(),
            latent_means: x_star,
            latent_variances: variances,
            predictor_means: eta,
            predictor_variances: eta_var,
            marginal_log_lik,
            marginal_log_lik_gaussian: marginal_log_lik,
            dic: dic_result.dic,
            mean_deviance: dic_result.mean_deviance,
            effective_params: dic_result.effective_params,
            waic: waic_result.waic,
            waic_lppd: waic_result.lppd,
            waic_effective_params: waic_result.effective_params,
            cpo: cpo_result.cpo,
            pit: cpo_result.pit,
            cpo_n_failures: cpo_result.n_failures,
            theta_nodes: Vec::new(),
            node_weights: Vec::new(),
            internal_marginals_hyperpar: Vec::new(),
            marginals_latent,
            marginals_predictor,
            marginals_latent_indices: marginal_opts.latent_indices.clone(),
            marginals_predictor_indices: marginal_opts.predictor_indices.clone(),
            posterior_precision: Some(q_post),
        });
    }

    let cancelled = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    let config = crate::hyper_opt::ModelConfig {
        build_prior,
        log_prior_density,
        obs,
        a,
        constraints,
        check_cancel,
        build_obs,
    };

    let mode = crate::hyper_opt::nelder_mead(initial_theta, 0.1, 200, 1e-6, &config)?;

    // Adaptive FD step: a tiny absolute h yields exploding curvature for sharp
    // precision posteriors (especially intrinsic lattice models). Scale with the
    // mode's magnitude but keep the original floor, since raising it makes nearly
    // flat directions (e.g. Kronecker space⊗time models) look singular.
    let hess_h = mode.iter().map(|t| 0.05 * t.abs()).fold(1e-4, f64::max);
    let hessian = crate::hyper_opt::compute_hessian(&mode, &config, hess_h)?;

    let neg_hessian = hessian.iter().map(|&x| -x).collect::<Vec<_>>();
    let sigma = invert_symmetric_matrix(&neg_hessian, m)?;

    let (lambdas, v) = jacobi_eigen(&sigma, m, 100)?;

    let (z_points, z_weights) = match strategy.to_lowercase().as_str() {
        "grid" => {
            let evaluator = |z: &[f64]| -> f64 {
                if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                    return f64::NEG_INFINITY;
                }
                let mut theta = mode.clone();
                for i in 0..m {
                    let mut diff = 0.0;
                    for j in 0..m {
                        diff += v[i * m + j] * lambdas[j].abs().sqrt() * z[j];
                    }
                    theta[i] += diff;
                }
                match crate::hyper_opt::evaluate_neg_log_posterior(&theta, &config) {
                    Ok(val) => -val,
                    Err(_) => f64::NEG_INFINITY,
                }
            };
            grid_design(m, step_or_f0, 4.0, &evaluator)?
        }
        _ => ccd_design(m, step_or_f0)?,
    };

    let mut mode_index = 0;
    let mut min_z_norm = f64::INFINITY;
    for (k, z) in z_points.iter().enumerate() {
        let z_norm: f64 = z.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if z_norm < min_z_norm {
            min_z_norm = z_norm;
            mode_index = k;
        }
    }

    let eval_node =
        |z: &Vec<f64>| -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, f64), String> {
            if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                return Err("Operation cancelled by user".to_string());
            }
            if let Some(cancel) = check_cancel
                && let Err(err) = cancel()
            {
                cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
                return Err(err);
            }

            let mut theta = mode.clone();
            for i in 0..m {
                let mut diff = 0.0;
                for j in 0..m {
                    diff += v[i * m + j] * lambdas[j].abs().sqrt() * z[j];
                }
                theta[i] += diff;
            }

            let q_prior = build_prior(&theta)?;
            let obs_buf;
            let obs_slice = match build_obs {
                Some(f) => {
                    obs_buf = f(&theta);
                    &obs_buf[..]
                }
                None => obs,
            };
            // Per-node solver factory: CCD Rayon workers must not share &mut InlaSolver.
            let mut solver = FaerCpuSolver::new();
            let (x_star, marginal_log_lik) = match find_latent_mode_a_with_solver(
                &q_prior,
                obs_slice,
                a,
                constraints,
                200,
                1e-5,
                &mut solver,
            ) {
                Ok(v) => v,
                Err(_e) => {
                    if let Some(cancel) = check_cancel
                        && let Err(err) = cancel()
                    {
                        cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
                        return Err(err);
                    }
                    if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                        return Err("Operation cancelled by user".to_string());
                    }
                    let n_lat = q_prior.rows();
                    return Ok((
                        theta,
                        vec![0.0; n_lat],
                        vec![1.0; n_lat],
                        vec![0.0; n_obs],
                        vec![1.0; n_obs],
                        f64::NEG_INFINITY,
                    ));
                }
            };

            let variances = solver.diag_inv().map_err(|e| e.to_string())?;
            let eta = match a {
                None => x_star.clone(),
                Some(a_mat) => matvec_csc(a_mat, &x_star)?,
            };
            let eta_var = match a {
                None => variances.clone(),
                Some(a_mat) => predictor_variances_diag(a_mat, &variances)?,
            };

            let log_prior = log_prior_density(&theta);
            let log_post = marginal_log_lik + log_prior;

            Ok((theta, x_star, variances, eta, eta_var, log_post))
        };

    let results: Vec<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, f64)> = if deterministic {
        z_points
            .iter()
            .map(eval_node)
            .collect::<Result<Vec<_>, String>>()?
    } else {
        z_points
            .par_iter()
            .map(eval_node)
            .collect::<Result<Vec<_>, String>>()?
    };

    if !results.iter().any(|r| r.5.is_finite()) {
        return Err("Newton-Raphson did not converge at any integration node".to_string());
    }
    let mut theta_nodes = Vec::with_capacity(results.len());
    let mut cond_means = Vec::with_capacity(results.len());
    let mut cond_vars = Vec::with_capacity(results.len());
    let mut cond_eta = Vec::with_capacity(results.len());
    let mut cond_eta_var = Vec::with_capacity(results.len());
    let mut log_posts = Vec::with_capacity(results.len());

    let n_lat = results[0].1.len();
    for (theta, x_star, variances, eta, eta_var, log_post) in results {
        theta_nodes.push(theta);
        cond_means.push(x_star);
        cond_vars.push(variances);
        cond_eta.push(eta);
        cond_eta_var.push(eta_var);
        log_posts.push(log_post);
    }

    let max_log_post = log_posts.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let scaled_densities: Vec<f64> = log_posts
        .iter()
        .map(|&lp| (lp - max_log_post).exp())
        .collect();

    let mut sum_w_dens = 0.0;
    for k in 0..cond_means.len() {
        sum_w_dens += z_weights[k] * scaled_densities[k];
    }

    let norm_weights: Vec<f64> = z_weights
        .iter()
        .zip(&scaled_densities)
        .map(|(&w, &d)| w * d / sum_w_dens)
        .collect();

    let mut latent_means = vec![0.0; n_lat];
    let mut latent_variances = vec![0.0; n_lat];
    for i in 0..n_lat {
        let mut mean_i = 0.0;
        for k in 0..norm_weights.len() {
            mean_i += norm_weights[k] * cond_means[k][i];
        }
        latent_means[i] = mean_i;
        let mut var_i = 0.0;
        for k in 0..norm_weights.len() {
            let diff = cond_means[k][i] - mean_i;
            var_i += norm_weights[k] * (cond_vars[k][i] + diff * diff);
        }
        latent_variances[i] = var_i;
    }

    let mut predictor_means = vec![0.0; n_obs];
    let mut predictor_variances = vec![0.0; n_obs];
    for i in 0..n_obs {
        let mut mean_i = 0.0;
        for k in 0..norm_weights.len() {
            mean_i += norm_weights[k] * cond_eta[k][i];
        }
        predictor_means[i] = mean_i;
        let mut var_i = 0.0;
        for k in 0..norm_weights.len() {
            let diff = cond_eta[k][i] - mean_i;
            var_i += norm_weights[k] * (cond_eta_var[k][i] + diff * diff);
        }
        predictor_variances[i] = var_i;
    }

    let marginal_log_lik = max_log_post + sum_w_dens.abs().ln();

    let neg_hessian: Vec<f64> = hessian.iter().map(|&x| -x).collect();
    let marginal_log_lik_gaussian = crate::model_selection::compute_marginal_log_lik_gaussian(
        log_posts[mode_index],
        &neg_hessian,
        m,
    )
    .unwrap_or(f64::NAN);

    let mode_obs_buf;
    let mode_obs = match build_obs {
        Some(f) => {
            mode_obs_buf = f(&mode);
            &mode_obs_buf[..]
        }
        None => obs,
    };

    let dic_result = if compute.dic {
        crate::model_selection::compute_dic(mode_obs, &cond_eta, &norm_weights, mode_index)?
    } else {
        crate::model_selection::DicResult {
            dic: f64::NAN,
            mean_deviance: f64::NAN,
            effective_params: f64::NAN,
        }
    };
    let waic_result = if compute.waic {
        crate::model_selection::compute_waic(mode_obs, &cond_eta, &norm_weights)?
    } else {
        crate::model_selection::WaicResult {
            waic: f64::NAN,
            lppd: f64::NAN,
            effective_params: f64::NAN,
        }
    };

    let cpo_result = if compute.cpo {
        crate::model_selection::compute_cpo_pit(mode_obs, &cond_eta, &cond_eta_var, &norm_weights)?
    } else {
        crate::model_selection::CpoResult {
            cpo: vec![None; n_obs],
            pit: vec![None; n_obs],
            n_failures: 0,
        }
    };

    let internal_marginals_hyperpar = if marginal_opts.hyperpar && m > 0 {
        crate::marginals::hyperpar_marginals(
            &theta_nodes,
            &norm_weights,
            marginal_opts.n_points,
            marginal_opts.n_sd,
        )?
    } else {
        Vec::new()
    };

    let mut marginals_latent = Vec::with_capacity(marginal_opts.latent_indices.len());
    for &idx in &marginal_opts.latent_indices {
        if idx >= n_lat {
            return Err(format!(
                "latent marginal index {idx} out of range (n={n_lat})"
            ));
        }
        let means: Vec<f64> = cond_means.iter().map(|v| v[idx]).collect();
        let vars: Vec<f64> = cond_vars.iter().map(|v| v[idx]).collect();
        marginals_latent.push(crate::marginals::gaussian_mixture_marginal(
            &means,
            &vars,
            &norm_weights,
            marginal_opts.n_points,
            marginal_opts.n_sd,
        )?);
    }

    let mut marginals_predictor = Vec::with_capacity(marginal_opts.predictor_indices.len());
    for &idx in &marginal_opts.predictor_indices {
        if idx >= n_obs {
            return Err(format!(
                "predictor marginal index {idx} out of range (n_obs={n_obs})"
            ));
        }
        let means: Vec<f64> = cond_eta.iter().map(|v| v[idx]).collect();
        let vars: Vec<f64> = cond_eta_var.iter().map(|v| v[idx]).collect();
        marginals_predictor.push(crate::marginals::gaussian_mixture_marginal(
            &means,
            &vars,
            &norm_weights,
            marginal_opts.n_points,
            marginal_opts.n_sd,
        )?);
    }

    let q_post = posterior_precision_at(
        &build_prior(&mode)?,
        obs,
        a,
        constraints,
        &cond_means[mode_index],
    )?;
    Ok(InferenceResult {
        mode,
        hessian,
        latent_means,
        latent_variances,
        predictor_means,
        predictor_variances,
        marginal_log_lik,
        marginal_log_lik_gaussian,
        dic: dic_result.dic,
        mean_deviance: dic_result.mean_deviance,
        effective_params: dic_result.effective_params,
        waic: waic_result.waic,
        waic_lppd: waic_result.lppd,
        waic_effective_params: waic_result.effective_params,
        cpo: cpo_result.cpo,
        pit: cpo_result.pit,
        cpo_n_failures: cpo_result.n_failures,
        theta_nodes,
        node_weights: norm_weights,
        internal_marginals_hyperpar,
        marginals_latent,
        marginals_predictor,
        marginals_latent_indices: marginal_opts.latent_indices.clone(),
        marginals_predictor_indices: marginal_opts.predictor_indices.clone(),
        posterior_precision: Some(q_post),
    })
}

/// Trait-driven INLA entry point (`LatentModel` + optional `ProjectionMapper`).
///
/// Closure-based APIs (`run_inla_inference_a*`) remain available and wrap the same engine.
pub fn run_inla_inference_model(
    initial_theta: &[f64],
    latent: &dyn crate::latent::LatentModel,
    obs: &[Obs],
    mapper: Option<&dyn crate::projection::ProjectionMapper>,
    constraints: Option<&ConstraintSpec>,
    strategy: &str,
    step_or_f0: f64,
    marginal_opts: &crate::marginals::MarginalOptions,
    deterministic: bool,
) -> Result<InferenceResult, String> {
    if !initial_theta.is_empty() && initial_theta.len() != latent.num_hyperparameters() {
        return Err(format!(
            "initial_theta length {} != latent.num_hyperparameters() {}",
            initial_theta.len(),
            latent.num_hyperparameters()
        ));
    }
    let build_prior = |theta: &[f64]| latent.build_precision(theta);
    let log_prior_density = |theta: &[f64]| latent.log_prior_density(theta);
    let a = mapper.and_then(|m| m.projection_matrix());
    let constr = constraints.or_else(|| latent.constraints());
    run_inla_inference_a_cancellable(
        initial_theta,
        &build_prior,
        &log_prior_density,
        obs,
        a,
        constr,
        strategy,
        step_or_f0,
        marginal_opts,
        deterministic,
        None,
        None,
        None,
    )
}

fn link_forward(eta: f64, link: Link) -> Result<(f64, f64, f64), String> {
    match link {
        Link::Identity => Ok((eta, 1.0, 0.0)),
        Link::Log => {
            let mu = eta.exp();
            Ok((mu, mu, mu))
        }
        Link::Logit => {
            let p = logistic(eta);
            let dp = p * (1.0 - p);
            Ok((p, dp, dp * (1.0 - 2.0 * p)))
        }
    }
}

fn logistic(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

fn weibull_cumhaz(lambda: f64, k: f64, y: f64, variant: i32) -> Result<(f64, f64, f64), String> {
    match variant {
        0 => {
            let ya = y.powf(k);
            Ok((lambda * ya, ya, 0.0))
        }
        1 => {
            let h = (lambda * y).powf(k);
            Ok((h, k * h / lambda, k * (k - 1.0) * h / (lambda * lambda)))
        }
        _ => Err("weibull survival variant must be 0 (PH) or 1 (AFT)".into()),
    }
}

fn weibull_log_density(lambda: f64, k: f64, y: f64, h: f64, variant: i32) -> Result<f64, String> {
    match variant {
        0 => Ok(k.ln() + (k - 1.0) * y.ln() + lambda.ln() - h),
        1 => Ok(k.ln() + k * lambda.ln() + (k - 1.0) * y.ln() - h),
        _ => Err("weibull survival variant must be 0 (PH) or 1 (AFT)".into()),
    }
}

fn weibull_d_log_density(lambda: f64, k: f64, _y: f64, _h: f64, dh: f64, variant: i32) -> f64 {
    match variant {
        0 => 1.0 / lambda - dh,
        _ => k / lambda - dh,
    }
}

fn weibull_d2_log_density(
    lambda: f64,
    k: f64,
    _y: f64,
    _h: f64,
    _dh: f64,
    d2h: f64,
    variant: i32,
) -> f64 {
    match variant {
        0 => -1.0 / (lambda * lambda) - d2h,
        _ => -k / (lambda * lambda) - d2h,
    }
}

fn survival_left_from_cumhaz(h: f64, dh: f64, d2h: f64) -> Result<(f64, f64, f64), String> {
    let logp = log1mexp(h)?;
    let e_h = h.exp();
    let dll_dh = 1.0 / (e_h - 1.0);
    let d2ll_dh2 = -e_h / ((e_h - 1.0) * (e_h - 1.0));
    Ok((logp, dll_dh * dh, d2ll_dh2 * dh * dh + dll_dh * d2h))
}

fn survival_interval_from_cumhaz(
    hl: f64,
    dhl: f64,
    d2hl: f64,
    hu: f64,
    dhu: f64,
    d2hu: f64,
) -> Result<(f64, f64, f64), String> {
    if hu <= hl {
        return Err("interval-censored survival: upper cumulative hazard must exceed lower".into());
    }
    let logp = -hl + log1mexp(hu - hl)?;
    let eml = (-hl).exp();
    let emu = (-hu).exp();
    let g = eml - emu;
    let gp = -eml * dhl + emu * dhu;
    let gpp = eml * (dhl * dhl - d2hl) - emu * (dhu * dhu - d2hu);
    Ok((logp, gp / g, (gpp * g - gp * gp) / (g * g)))
}

fn loglogistic_u(t: f64, scale: f64, shape: f64) -> (f64, f64, f64) {
    let u = (t / scale).powf(shape);
    let du = -shape * u / scale;
    let d2u = shape * (shape + 1.0) * u / (scale * scale);
    (u, du, d2u)
}

fn chain_u(ell_u: f64, ell_uu: f64, du: f64, d2u: f64) -> (f64, f64) {
    (ell_u * du, ell_uu * du * du + ell_u * d2u)
}

fn loglogistic_event(t: f64, scale: f64, shape: f64) -> Result<(f64, f64, f64), String> {
    let (u, du, d2u) = loglogistic_u(t, scale, shape);
    let logp = shape.ln() - t.ln() + u.ln() - 2.0 * (1.0 + u).ln();
    let ell_u = 1.0 / u - 2.0 / (1.0 + u);
    let ell_uu = -1.0 / (u * u) + 2.0 / ((1.0 + u) * (1.0 + u));
    let (d1, d2) = chain_u(ell_u, ell_uu, du, d2u);
    Ok((logp, d1, d2))
}

fn loglogistic_right(t: f64, scale: f64, shape: f64) -> Result<(f64, f64, f64), String> {
    let (u, du, d2u) = loglogistic_u(t, scale, shape);
    let logp = -(1.0 + u).ln();
    let ell_u = -1.0 / (1.0 + u);
    let ell_uu = 1.0 / ((1.0 + u) * (1.0 + u));
    let (d1, d2) = chain_u(ell_u, ell_uu, du, d2u);
    Ok((logp, d1, d2))
}

fn loglogistic_left(t: f64, scale: f64, shape: f64) -> Result<(f64, f64, f64), String> {
    let (u, du, d2u) = loglogistic_u(t, scale, shape);
    let logp = u.ln() - (1.0 + u).ln();
    let ell_u = 1.0 / u - 1.0 / (1.0 + u);
    let ell_uu = -1.0 / (u * u) + 1.0 / ((1.0 + u) * (1.0 + u));
    let (d1, d2) = chain_u(ell_u, ell_uu, du, d2u);
    Ok((logp, d1, d2))
}

fn loglogistic_interval(
    t: f64,
    t_upper: f64,
    scale: f64,
    shape: f64,
) -> Result<(f64, f64, f64), String> {
    let (ul, dul, d2ul) = loglogistic_u(t, scale, shape);
    let (uu, duu, d2uu) = loglogistic_u(t_upper, scale, shape);
    let sl = 1.0 / (1.0 + ul);
    let su = 1.0 / (1.0 + uu);
    let g = sl - su;
    if !(g > 0.0) {
        return Err("interval-censored loglogistic: survival mass must be positive".into());
    }
    let dsl_du = -1.0 / ((1.0 + ul) * (1.0 + ul));
    let dsu_du = -1.0 / ((1.0 + uu) * (1.0 + uu));
    let d2sl = 2.0 / (1.0 + ul).powi(3);
    let d2su = 2.0 / (1.0 + uu).powi(3);
    let gp = dsl_du * dul - dsu_du * duu;
    let gpp = d2sl * dul * dul + dsl_du * d2ul - (d2su * duu * duu + dsu_du * d2uu);
    Ok((g.ln(), gp / g, (gpp * g - gp * gp) / (g * g)))
}

fn lognormal_event(t: f64, mu: f64, _sigma: f64, prec: f64) -> (f64, f64, f64) {
    let z = (t.ln() - mu) * prec.sqrt();
    let logp = LOG_NORMC_GAUSSIAN + 0.5 * prec.ln() - 0.5 * z * z - t.ln();
    (logp, prec * (t.ln() - mu), -prec)
}

fn lognormal_right(t: f64, mu: f64, sigma: f64, _prec: f64) -> Result<(f64, f64, f64), String> {
    let z = (t.ln() - mu) / sigma;
    let sf = standard_normal_cdf(-z);
    if !(sf > 0.0) {
        return Err("lognormal right-censor survival is numerically 0".into());
    }
    let m = standard_normal_pdf(-z) / sf;
    let logp = sf.ln();
    let d1 = m / sigma;
    let d2 = (-(-z) * m - m * m) / (sigma * sigma);
    Ok((logp, d1, d2))
}

fn lognormal_left(t: f64, mu: f64, sigma: f64) -> Result<(f64, f64, f64), String> {
    let z = (t.ln() - mu) / sigma;
    let f = standard_normal_cdf(z);
    if !(f > 0.0) {
        return Err("lognormal left-censor CDF is numerically 0".into());
    }
    let m = standard_normal_pdf(z) / f;
    let logp = f.ln();
    let d1 = -m / sigma;
    let d2 = (-z * m - m * m) / (sigma * sigma);
    Ok((logp, d1, d2))
}

fn lognormal_interval(
    t: f64,
    t_upper: f64,
    mu: f64,
    sigma: f64,
) -> Result<(f64, f64, f64), String> {
    let zl = (t.ln() - mu) / sigma;
    let zu = (t_upper.ln() - mu) / sigma;
    let g = standard_normal_cdf(zu) - standard_normal_cdf(zl);
    if !(g > 0.0) {
        return Err("interval-censored lognormal: probability mass must be positive".into());
    }
    let pl = standard_normal_pdf(zl);
    let pu = standard_normal_pdf(zu);
    let gp = (pl - pu) / sigma;
    let gpp = (zl * pl - zu * pu) / (sigma * sigma);
    Ok((g.ln(), gp / g, (gpp * g - gp * gp) / (g * g)))
}

pub(crate) fn standard_normal_cdf(x: f64) -> f64 {
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + P * ax);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-ax * ax / 2.0).exp();
    0.5 * (1.0 + sign * y)
}

fn standard_normal_pdf(z: f64) -> f64 {
    (LOG_NORMC_GAUSSIAN - 0.5 * z * z).exp()
}

fn log1mexp(x: f64) -> Result<f64, String> {
    // log(1 - exp(-x)) for x > 0
    if !(x > 0.0) || !x.is_finite() {
        return Err("survival CDF argument must be finite and > 0".into());
    }
    if x < 1e-8 {
        Ok(x.ln())
    } else {
        Ok((-(-x).exp()).ln_1p())
    }
}

fn validate_event_indicator(v: f64, label: &str) -> Result<(), String> {
    if !v.is_finite() || !(0.0..=3.0).contains(&v) || v.fract() != 0.0 {
        return Err(format!(
            "{label} must be 0 (right), 1 (event), 2 (left), or 3 (interval)"
        ));
    }
    Ok(())
}

fn log_factorial(y: f64) -> Result<f64, String> {
    if y < 0.0 {
        return Err("factorial undefined for y < 0".to_string());
    }
    Ok(log_gamma(y + 1.0))
}

pub(crate) fn log_gamma(z: f64) -> f64 {
    const COEFF: [f64; 9] = [
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
        0.0,
    ];
    if z < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * z).sin().ln()
            - log_gamma(1.0 - z);
    }
    let z1 = z - 1.0;
    let mut x = 0.999_999_999_999_809_9_f64;
    for (i, c) in COEFF.iter().enumerate().take(8) {
        x += c / (z1 + (i as f64) + 1.0);
    }
    let t = z1 + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z1 + 0.5) * t.ln() - t + x.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) {
        assert!((a - b).abs() < eps, "left={a}, right={b}, eps={eps}");
    }

    #[test]
    fn evaluates_gaussian_prior() {
        let out = eval_prior_gaussian(
            0.2,
            GaussianPrior {
                mean: 0.0,
                precision: 4.0,
            },
        )
        .expect("ok");
        approx(out.grad, -0.8, 1e-12);
        approx(out.hess, -4.0, 1e-12);
    }

    #[test]
    fn evaluates_gamma_and_loggamma_priors() {
        let g = eval_prior_gamma(
            2.0,
            GammaPrior {
                shape: 3.0,
                scale: 2.0,
            },
        )
        .expect("ok");
        approx(g.grad, 0.5, 1e-12);

        let lg = eval_prior_loggamma(
            0.0,
            GammaPrior {
                shape: 3.0,
                scale: 2.0,
            },
        )
        .expect("ok");
        approx(lg.grad, 2.5, 1e-12);
    }

    #[test]
    fn evaluates_gaussian_likelihood() {
        let out = eval_likelihood_gaussian(
            1.5,
            GaussianObs {
                y: 1.0,
                precision: 2.0,
                link: Link::Identity,
            },
        )
        .expect("ok");
        approx(out.grad, -1.0, 1e-12);
        approx(out.hess, -2.0, 1e-12);
    }

    #[test]
    fn evaluates_poisson_likelihood() {
        let out = eval_likelihood_poisson(
            0.2,
            PoissonObs {
                y: 3.0,
                exposure: 2.0,
                link: Link::Log,
            },
        )
        .expect("ok");
        assert!(out.logp.is_finite());
        assert!(out.hess.is_finite());
    }

    #[test]
    fn evaluates_binomial_likelihood() {
        let out = eval_likelihood_binomial(
            0.3,
            BinomialObs {
                y: 4.0,
                n: 10.0,
                link: Link::Logit,
            },
        )
        .expect("ok");
        assert!(out.logp.is_finite());
        assert!(out.grad.is_finite());
        assert!(out.hess.is_finite());
    }

    #[test]
    fn evaluates_negative_binomial_likelihood() {
        let eta = (2.0_f64).ln();
        let out = eval_likelihood_negative_binomial(
            eta,
            NegativeBinomialObs {
                y: 4.0,
                exposure: 1.0,
                size: 3.0,
                link: Link::Log,
            },
        )
        .expect("ok");
        let mu = 2.0;
        let r = 3.0;
        let y = 4.0;
        let dlog_dmu = y / mu - (y + r) / (mu + r);
        let d2log_dmu2 = -y / (mu * mu) + (y + r) / ((mu + r) * (mu + r));
        approx(out.grad, dlog_dmu * mu, 1e-12);
        approx(out.hess, d2log_dmu2 * mu * mu + dlog_dmu * mu, 1e-12);
    }

    #[test]
    fn evaluates_zero_inflated_poisson_type0() {
        let out = eval_likelihood_zero_inflated_poisson(
            (1.2_f64).ln(),
            ZeroInflatedPoissonObs {
                y: 0.0,
                exposure: 1.0,
                zero_prob: 0.2,
                link: Link::Log,
                inflation: ZeroInflationType::Type0,
            },
        )
        .expect("ok");
        let lambda: f64 = 1.2;
        let f0 = (-lambda).exp();
        let s = 0.2 + 0.8 * f0;
        let dlog_dlambda = -0.8 * f0 / s;
        let d2log_dlambda2 = 0.8 * f0 / s - dlog_dlambda * dlog_dlambda;
        approx(out.grad, dlog_dlambda * lambda, 1e-12);
        approx(
            out.hess,
            d2log_dlambda2 * lambda * lambda + dlog_dlambda * lambda,
            1e-12,
        );
    }

    #[test]
    fn evaluates_zero_inflated_binomial_type1() {
        let eta = 0.0;
        let out = eval_likelihood_zero_inflated_binomial(
            eta,
            ZeroInflatedBinomialObs {
                y: 2.0,
                n: 5.0,
                zero_prob: 0.25,
                link: Link::Logit,
                inflation: ZeroInflationType::Type1,
            },
        )
        .expect("ok");
        assert!(out.logp.is_finite());
        assert!(out.grad.is_finite());
        assert!(out.hess.is_finite());
    }

    #[test]
    fn evaluates_laplace_likelihood() {
        let eta = 0.3;
        let out = eval_likelihood_laplace(
            eta,
            LaplaceObs {
                y: 1.0,
                alpha: 0.7,
                gamma: 0.2,
                link: Link::Identity,
            },
        )
        .expect("ok");
        let x = 1.0 - eta;
        let s = (x * x + 0.04_f64).sqrt();
        let rho_dx = 0.5 * ((2.0 * 0.7 - 1.0) + x / s);
        let rho_d2x = 0.5 * 0.04 / (s * s * s);
        approx(out.grad, rho_dx, 1e-12);
        approx(out.hess, -rho_d2x, 1e-12);
    }

    #[test]
    fn evaluates_exponential_survival_likelihood() {
        let eta = (1.4_f64).ln();
        let out = eval_likelihood_exponential_survival(
            eta,
            ExponentialSurvivalObs {
                y: 2.0,
                event: 1.0,
                y_upper: f64::NAN,
                link: Link::Log,
            },
        )
        .expect("ok");
        let rate = 1.4;
        let dlog_drate = 1.0 / rate - 2.0;
        let d2log_drate2 = -1.0 / (rate * rate);
        approx(out.grad, dlog_drate * rate, 1e-12);
        approx(
            out.hess,
            d2log_drate2 * rate * rate + dlog_drate * rate,
            1e-12,
        );
    }

    #[test]
    fn evaluates_weibull_survival_likelihood() {
        let eta = (1.1_f64).ln();
        let out = eval_likelihood_weibull_survival(
            eta,
            WeibullSurvivalObs {
                y: 1.5,
                event: 1.0,
                y_upper: f64::NAN,
                shape: 1.8,
                variant: 1,
                link: Link::Log,
            },
        )
        .expect("ok");
        assert!(out.logp.is_finite());
        assert!(out.grad.is_finite());
        assert!(out.hess.is_finite());
    }

    #[test]
    fn exponential_left_and_interval_match_finite_differences() {
        let eta = (0.8_f64).ln();
        let left = eval_likelihood_exponential_survival(
            eta,
            ExponentialSurvivalObs {
                y: 1.5,
                event: 2.0,
                y_upper: f64::NAN,
                link: Link::Log,
            },
        )
        .expect("left");
        let rate = 0.8;
        let f = 1.0 - (-rate * 1.5_f64).exp();
        approx(left.logp, f.ln(), 1e-12);

        let interval = eval_likelihood_exponential_survival(
            eta,
            ExponentialSurvivalObs {
                y: 0.5,
                event: 3.0,
                y_upper: 2.0,
                link: Link::Log,
            },
        )
        .expect("interval");
        let sl = (-rate * 0.5_f64).exp();
        let su = (-rate * 2.0_f64).exp();
        approx(interval.logp, (sl - su).ln(), 1e-12);

        let h = 1e-6;
        let bump = |e: f64| {
            eval_likelihood_exponential_survival(
                e,
                ExponentialSurvivalObs {
                    y: 0.5,
                    event: 3.0,
                    y_upper: 2.0,
                    link: Link::Log,
                },
            )
            .unwrap()
            .logp
        };
        let fd_g = (bump(eta + h) - bump(eta - h)) / (2.0 * h);
        approx(interval.grad, fd_g, 1e-6);
    }

    #[test]
    fn weibull_left_and_interval_logp_finite() {
        let eta = (1.1_f64).ln();
        let left = eval_likelihood_weibull_survival(
            eta,
            WeibullSurvivalObs {
                y: 1.2,
                event: 2.0,
                y_upper: f64::NAN,
                shape: 1.5,
                variant: 1,
                link: Link::Log,
            },
        )
        .expect("left");
        assert!(left.logp.is_finite() && left.grad.is_finite() && left.hess.is_finite());

        let interval = eval_likelihood_weibull_survival(
            eta,
            WeibullSurvivalObs {
                y: 0.8,
                event: 3.0,
                y_upper: 2.5,
                shape: 1.5,
                variant: 1,
                link: Link::Log,
            },
        )
        .expect("interval");
        let h = 1e-6;
        let bump = |e: f64| {
            eval_likelihood_weibull_survival(
                e,
                WeibullSurvivalObs {
                    y: 0.8,
                    event: 3.0,
                    y_upper: 2.5,
                    shape: 1.5,
                    variant: 1,
                    link: Link::Log,
                },
            )
            .unwrap()
            .logp
        };
        let fd_g = (bump(eta + h) - bump(eta - h)) / (2.0 * h);
        approx(interval.grad, fd_g, 1e-5);
    }

    #[test]
    fn weibull_ph_and_aft_variants_differ() {
        let eta = 0.2_f64;
        let ph = eval_likelihood_weibull_survival(
            eta,
            WeibullSurvivalObs {
                y: 1.4,
                event: 1.0,
                y_upper: f64::NAN,
                shape: 1.7,
                variant: 0,
                link: Link::Log,
            },
        )
        .unwrap();
        let aft = eval_likelihood_weibull_survival(
            eta,
            WeibullSurvivalObs {
                y: 1.4,
                event: 1.0,
                y_upper: f64::NAN,
                shape: 1.7,
                variant: 1,
                link: Link::Log,
            },
        )
        .unwrap();
        assert!((ph.logp - aft.logp).abs() > 1e-6);
        let lambda = eta.exp();
        let k = 1.7_f64;
        let y = 1.4_f64;
        let h = lambda * y.powf(k);
        let expect = k.ln() + (k - 1.0) * y.ln() + lambda.ln() - h;
        approx(ph.logp, expect, 1e-12);
    }

    #[test]
    fn loglogistic_and_lognormal_survival_finite() {
        let ll = eval_likelihood_loglogistic_survival(
            0.0,
            LoglogisticSurvivalObs {
                y: 1.3,
                event: 1.0,
                y_upper: f64::NAN,
                shape: 2.0,
                link: Link::Log,
            },
        )
        .unwrap();
        assert!(ll.logp.is_finite() && ll.grad.is_finite() && ll.hess.is_finite());
        let scale = 1.0_f64;
        let t = 1.3_f64;
        let b = 2.0_f64;
        let u = (t / scale).powf(b);
        let expect = b.ln() - t.ln() + u.ln() - 2.0 * (1.0 + u).ln();
        approx(ll.logp, expect, 1e-12);

        let ln = eval_likelihood_lognormal_survival(
            0.1,
            LognormalSurvivalObs {
                y: 1.2,
                event: 1.0,
                y_upper: f64::NAN,
                prec: 4.0,
                link: Link::Identity,
            },
        )
        .unwrap();
        let mu = 0.1_f64;
        let prec = 4.0_f64;
        let t = 1.2_f64;
        let z = (t.ln() - mu) * prec.sqrt();
        let expect = LOG_NORMC_GAUSSIAN + 0.5 * prec.ln() - 0.5 * z * z - t.ln();
        approx(ln.logp, expect, 1e-12);
        approx(ln.grad, prec * (t.ln() - mu), 1e-12);
    }

    #[test]
    fn test_run_inla_inference_iid_gaussian() {
        // Latent field dimension
        let n = 5;
        // Observations: y = [1.0, 1.2, 0.9, 1.1, 0.8]
        let y_obs = vec![1.0, 1.2, 0.9, 1.1, 0.8];
        let obs_precision = 2.0;

        let mut obs = Vec::new();
        for &y in &y_obs {
            obs.push(Obs::Gaussian(GaussianObs {
                y,
                precision: obs_precision,
                link: Link::Identity,
            }));
        }

        // Prior builder for theta = log(tau_latent)
        let build_prior = |theta: &[f64]| -> Result<CscMatrix, String> {
            let tau = theta[0].exp();
            let mut tri = sprs::TriMatI::<f64, usize>::with_capacity((n, n), n);
            for i in 0..n {
                tri.add_triplet(i, i, tau);
            }
            Ok(tri.to_csc())
        };

        // Prior on theta: Gaussian prior on log(tau_latent) with mean 0.0 and precision 0.1
        let log_prior_density = |theta: &[f64]| -> f64 {
            let val = theta[0];
            // Gaussian density
            -0.5 * 0.1 * val * val
        };

        // Run inference using CCD strategy
        let result = run_inla_inference(
            &[0.0], // initial theta = 0.0 (tau = 1.0)
            &build_prior,
            &log_prior_density,
            &obs,
            "ccd",
            1.0,
        )
        .expect("inference should succeed");

        assert_eq!(result.mode.len(), 1);
        assert!(result.mode[0].is_finite());
        assert_eq!(result.hessian.len(), 1);
        assert!(result.hessian[0] < 0.0); // should be negative curvature (objective is negative of log-posterior, so hessian of log-posterior is negative)

        assert_eq!(result.latent_means.len(), n);
        assert_eq!(result.latent_variances.len(), n);

        for i in 0..n {
            assert!(result.latent_means[i].is_finite());
            assert!(result.latent_variances[i] > 0.0);
        }
    }

    #[test]
    fn test_run_inla_inference_iid_gaussian_grid() {
        let n = 3;
        let y_obs = vec![1.0, 1.1, 0.9];
        let obs_precision = 2.0;

        let mut obs = Vec::new();
        for &y in &y_obs {
            obs.push(Obs::Gaussian(GaussianObs {
                y,
                precision: obs_precision,
                link: Link::Identity,
            }));
        }

        let build_prior = |theta: &[f64]| -> Result<CscMatrix, String> {
            let tau = theta[0].exp();
            let mut tri = sprs::TriMatI::<f64, usize>::with_capacity((n, n), n);
            for i in 0..n {
                tri.add_triplet(i, i, tau);
            }
            Ok(tri.to_csc())
        };

        let log_prior_density = |theta: &[f64]| -> f64 {
            let val = theta[0];
            -0.5 * 0.1 * val * val
        };

        // Run inference using Grid strategy (step size 1.0)
        let result =
            run_inla_inference(&[0.0], &build_prior, &log_prior_density, &obs, "grid", 1.0)
                .expect("inference should succeed");

        assert_eq!(result.mode.len(), 1);
        assert!(result.mode[0].is_finite());
        assert_eq!(result.hessian.len(), 1);
        assert!(result.hessian[0] < 0.0);

        assert_eq!(result.latent_means.len(), n);
        assert_eq!(result.latent_variances.len(), n);
    }

    #[test]
    fn test_run_inla_inference_with_a_matrix() {
        // 2 latent region effects + intercept, 6 observations (3 per region).
        let n_lat = 2usize;
        let y = [1.0, 1.1, 0.9, 2.0, 2.1, 1.9];
        let region = [0usize, 0, 0, 1, 1, 1];
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
        let mut a_rows = Vec::new();
        let mut a_cols = Vec::new();
        let mut a_vals = Vec::new();
        for (i, &r) in region.iter().enumerate() {
            a_rows.push(i);
            a_cols.push(r);
            a_vals.push(1.0);
        }
        let n_lat_tot = n_lat + 1;
        for i in 0..y.len() {
            a_rows.push(i);
            a_cols.push(n_lat);
            a_vals.push(1.0);
        }
        let a = csc_from_triplets_0based(y.len(), n_lat_tot, &a_rows, &a_cols, &a_vals).unwrap();

        let build_prior = |theta: &[f64]| -> Result<CscMatrix, String> {
            let tau = theta[0].exp();
            let q_u = crate::latent_models::iid_precision_csc(n_lat, tau)?;
            let q_b = identity_csc(1, 1e-4)?;
            block_diag_csc(&[q_u, q_b])
        };
        let log_prior = |theta: &[f64]| -> f64 { -0.5 * 0.1 * theta[0] * theta[0] };

        let result = run_inla_inference_a(
            &[0.0],
            &build_prior,
            &log_prior,
            &obs,
            Some(&a),
            None,
            "ccd",
            1.0,
            &crate::marginals::MarginalOptions::default(),
            false,
        )
        .expect("A-matrix inference");
        assert!(!result.internal_marginals_hyperpar.is_empty());
        assert_eq!(result.theta_nodes.len(), result.node_weights.len());
        assert_eq!(result.latent_means.len(), n_lat_tot);
        assert_eq!(result.predictor_means.len(), y.len());
        assert!(result.marginal_log_lik.is_finite());
        let pred_r0: f64 = result.predictor_means[0..3].iter().sum::<f64>() / 3.0;
        let pred_r1: f64 = result.predictor_means[3..6].iter().sum::<f64>() / 3.0;
        assert!(pred_r1 > pred_r0);
    }

    #[test]
    fn binomial_identity_mode_converges() {
        use inla_math::design::identity_csc;
        let n = 20usize;
        let q = identity_csc(n, 1.0).unwrap();
        let mut obs = Vec::new();
        for i in 0..n {
            let y = if i % 2 == 0 { 4.0 } else { 6.0 };
            obs.push(Obs::Binomial(BinomialObs {
                y,
                n: 10.0,
                link: Link::Logit,
            }));
        }
        let (x, _f, mlik) = find_latent_mode(&q, &obs, 100, 1e-6).expect("mode");
        assert!(mlik.is_finite(), "mlik={mlik}");
        assert!(x.iter().all(|v| v.is_finite()));
        let mean: f64 = x.iter().sum::<f64>() / n as f64;
        assert!(mean.abs() < 1.0, "mean eta={mean}");
    }

    #[test]
    fn binomial_mode_across_prior_strengths() {
        use inla_math::design::identity_csc;
        let n = 54usize;
        let ys = [
            3., 5., 4., 2., 5., 6., 3., 4., 5., 3., 4., 6., 2., 5., 4., 3., 5., 4., 6., 3., 4., 5.,
            3., 4., 5., 2., 6., 4., 3., 5., 4., 3., 5., 4., 6., 3., 4., 5., 3., 4., 5., 2., 6., 4.,
            3., 5., 4., 3., 5., 4., 6., 3., 4., 5.,
        ];
        let mut obs = Vec::new();
        for y in ys {
            obs.push(Obs::Binomial(BinomialObs {
                y,
                n: 10.0,
                link: Link::Logit,
            }));
        }
        for tau in [1e-4, 1e-2, 1.0, 1e2, 1e4] {
            let q = identity_csc(n, tau).unwrap();
            find_latent_mode(&q, &obs, 100, 1e-5).unwrap_or_else(|e| panic!("tau={tau}: {e}"));
        }
        let q = identity_csc(n, 1.0).unwrap();
        let a = identity_csc(n, 1.0).unwrap();
        find_latent_mode_a(&q, &obs, Some(&a), None, 100, 1e-5).expect("A=I");
    }

    #[test]
    fn rw1_hard_sum_to_zero_holds() {
        use inla_math::{identity_csc, sum_to_zero_constraint};
        let n = 8usize;
        let y: Vec<f64> = (0..n).map(|i| (i as f64 - 3.5) * 0.3).collect();
        let mut obs = Vec::new();
        for &yi in &y {
            obs.push(Obs::Gaussian(GaussianObs {
                y: yi,
                precision: 4.0,
                link: Link::Identity,
            }));
        }
        // Latent layout: [rw1(n), intercept(1)]; constrain RW1 block only.
        let constr = sum_to_zero_constraint(n, 1)
            .unwrap()
            .embed(n + 1, 0)
            .unwrap();

        let a = {
            let mut rows = Vec::new();
            let mut cols = Vec::new();
            let mut vals = Vec::new();
            for i in 0..n {
                rows.push(i);
                cols.push(i);
                vals.push(1.0);
                rows.push(i);
                cols.push(n);
                vals.push(1.0);
            }
            csc_from_triplets_0based(n, n + 1, &rows, &cols, &vals).unwrap()
        };

        let build_prior = |theta: &[f64]| -> Result<CscMatrix, String> {
            let tau = theta[0].exp();
            let q_u = crate::latent_models::rw1_precision_csc(n, tau)?;
            let q_b = identity_csc(1, 1e-4)?;
            block_diag_csc(&[q_u, q_b])
        };
        let log_prior = |theta: &[f64]| -> f64 { -0.5 * 0.1 * theta[0] * theta[0] };

        let result = run_inla_inference_a(
            &[0.0],
            &build_prior,
            &log_prior,
            &obs,
            Some(&a),
            Some(&constr),
            "ccd",
            1.0,
            &crate::marginals::MarginalOptions::default(),
            false,
        )
        .expect("constrained rw1");
        let s: f64 = result.latent_means[..n].iter().sum();
        assert!(
            s.abs() < 1e-4,
            "sum of RW1 posterior means should be ~0, got {s}"
        );
        assert!(result.marginal_log_lik.is_finite());
    }

    #[test]
    fn lagrange_elimination_not_implemented() {
        use inla_math::sum_to_zero_constraint;
        let n = 4usize;
        let q = identity_csc(n, 1.0).unwrap();
        let obs: Vec<Obs> = (0..n)
            .map(|i| {
                Obs::Gaussian(GaussianObs {
                    y: i as f64,
                    precision: 1.0,
                    link: Link::Identity,
                })
            })
            .collect();
        let mut c = sum_to_zero_constraint(n, 1).unwrap();
        c.method = ConstraintMethod::LagrangeElimination;
        let err = find_latent_mode_a(&q, &obs, None, Some(&c), 50, 1e-5).unwrap_err();
        assert!(
            err.contains("LagrangeElimination"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn zero_theta_single_node_dic_cpo() {
        let n = 5usize;
        let obs: Vec<Obs> = (0..n)
            .map(|i| {
                Obs::Gaussian(GaussianObs {
                    y: 1.0 + 0.1 * i as f64,
                    precision: 4.0,
                    link: Link::Identity,
                })
            })
            .collect();
        // Fixed prior precision — no hyperparameters to integrate.
        let build_prior = |_theta: &[f64]| -> Result<CscMatrix, String> { identity_csc(n, 1.0) };
        let result = run_inla_inference_a(
            &[],
            &build_prior,
            &|_| 0.0,
            &obs,
            None,
            None,
            "ccd",
            1.0,
            &crate::marginals::MarginalOptions::default(),
            true,
        )
        .expect("zero-theta inference");
        assert!(result.mode.is_empty());
        assert!(result.dic.is_finite(), "dic={}", result.dic);
        assert!(result.mean_deviance.is_finite());
        assert_eq!(result.cpo.len(), n);
        // Single-node CPO is computed (not the old NaN/all-None bypass).
        assert!(
            result.cpo.iter().any(|c| c.is_some()) || result.cpo_n_failures == n,
            "expected CPO evaluation path to run"
        );
        assert!(result.marginal_log_lik.is_finite());
    }

    #[test]
    fn trait_entry_point_matches_closure_api() {
        use crate::latent::ClosureLatentModel;
        use crate::projection::IdentityProjection;

        let n = 4usize;
        let obs: Vec<Obs> = (0..n)
            .map(|i| {
                Obs::Gaussian(GaussianObs {
                    y: 0.5 + 0.2 * i as f64,
                    precision: 2.0,
                    link: Link::Identity,
                })
            })
            .collect();

        let latent = ClosureLatentModel::new(
            |theta| {
                let tau = theta[0].exp();
                identity_csc(n, tau)
            },
            |theta| -0.5 * 0.1 * theta[0] * theta[0],
            1,
        );
        let mapper = IdentityProjection::new(n).unwrap();
        let via_trait = run_inla_inference_model(
            &[0.0],
            &latent,
            &obs,
            Some(&mapper),
            None,
            "ccd",
            1.0,
            &crate::marginals::MarginalOptions::default(),
            true,
        )
        .expect("trait API");

        let build_prior = |theta: &[f64]| -> Result<CscMatrix, String> {
            let tau = theta[0].exp();
            identity_csc(n, tau)
        };
        let via_closure = run_inla_inference_a(
            &[0.0],
            &build_prior,
            &|theta| -0.5 * 0.1 * theta[0] * theta[0],
            &obs,
            None,
            None,
            "ccd",
            1.0,
            &crate::marginals::MarginalOptions::default(),
            true,
        )
        .expect("closure API");

        assert_eq!(via_trait.mode.len(), via_closure.mode.len());
        assert!((via_trait.marginal_log_lik - via_closure.marginal_log_lik).abs() < 1e-8);
        for i in 0..n {
            assert!((via_trait.latent_means[i] - via_closure.latent_means[i]).abs() < 1e-8);
        }
    }

    #[test]
    fn mock_solver_used_by_mode_find() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct CountingSolver {
            inner: FaerCpuSolver,
            factorize_calls: AtomicUsize,
        }

        impl InlaSolver for CountingSolver {
            fn factorize(&mut self, q: &CscMatrix) -> Result<(), inla_math::SolverError> {
                self.factorize_calls.fetch_add(1, Ordering::Relaxed);
                self.inner.factorize(q)
            }
            fn solve(&mut self, rhs: &[f64]) -> Result<Vec<f64>, inla_math::SolverError> {
                self.inner.solve(rhs)
            }
            fn diag_inv(&mut self) -> Result<Vec<f64>, inla_math::SolverError> {
                self.inner.diag_inv()
            }
            fn log_abs_det(&self) -> Result<f64, inla_math::SolverError> {
                self.inner.log_abs_det()
            }
        }

        let n = 3usize;
        let q = identity_csc(n, 2.0).unwrap();
        let obs: Vec<Obs> = (0..n)
            .map(|_| {
                Obs::Gaussian(GaussianObs {
                    y: 1.0,
                    precision: 1.0,
                    link: Link::Identity,
                })
            })
            .collect();
        let mut solver = CountingSolver {
            inner: FaerCpuSolver::new(),
            factorize_calls: AtomicUsize::new(0),
        };
        let (x, mlik) =
            find_latent_mode_a_with_solver(&q, &obs, None, None, 50, 1e-6, &mut solver).unwrap();
        assert!(mlik.is_finite());
        assert!(x.iter().all(|v| v.is_finite()));
        assert!(
            solver.factorize_calls.load(Ordering::Relaxed) >= 2,
            "expected prior + at least one Newton factorize"
        );
        let vars = solver.diag_inv().unwrap();
        assert_eq!(vars.len(), n);
    }
}

#[cfg(test)]
mod sparse_path_smoke {
    use super::*;
    use inla_math::{identity_csc, ldlt_factorize, ldlt_solve};

    #[test]
    fn ar1_and_fgn_sparse_factorize_no_panic() {
        let n = 20;
        let q = crate::ar1::ar1_precision_csc(n, 0.7, 4.0).unwrap();
        let f = ldlt_factorize(&q).expect("ar1 factorize");
        let x = ldlt_solve(&f, &vec![1.0; n]).expect("ar1 solve");
        assert!(x.iter().all(|v| v.is_finite()));

        let n = 30;
        let q = crate::latent_models::fgn_precision_csc(n, 0.7, 1.0).unwrap();
        let f = ldlt_factorize(&q).expect("fgn factorize");
        let x = ldlt_solve(&f, &vec![1.0; n]).expect("fgn solve");
        assert!(x.iter().all(|v| v.is_finite()));
        assert!(f.log_abs_det().is_finite());
    }

    #[test]
    fn fgn_gaussian_inference_no_panic() {
        let n = 30;
        let build = move |theta: &[f64]| -> Result<CscMatrix, String> {
            let tau = theta[0].exp();
            let hurst = 1.0 / (1.0 + (-theta[1]).exp());
            crate::latent_models::fgn_precision_csc(n, hurst, tau)
        };
        let obs: Vec<Obs> = (0..n)
            .map(|i| {
                Obs::Gaussian(GaussianObs {
                    y: 0.1 * (i as f64).sin(),
                    precision: 1000.0,
                    link: Link::Identity,
                })
            })
            .collect();
        let a = identity_csc(n, 1.0).unwrap();
        let res = run_inla_inference_a(
            &[0.0, 0.0],
            &build,
            &|_| 0.0,
            &obs,
            Some(&a),
            None,
            "ccd",
            1.0,
            &crate::marginals::MarginalOptions::default(),
            true,
        );
        assert!(res.is_ok(), "{res:?}");
    }

    #[test]
    fn compute_flags_skip_dic_waic_cpo() {
        let n = 8;
        let build = |theta: &[f64]| crate::latent_models::iid_precision_csc(n, theta[0].exp());
        let obs: Vec<Obs> = (0..n)
            .map(|i| {
                Obs::Gaussian(GaussianObs {
                    y: 0.05 * i as f64,
                    precision: 25.0,
                    link: Link::Identity,
                })
            })
            .collect();
        let a = identity_csc(n, 1.0).unwrap();
        let compute = ComputeOptions {
            dic: false,
            waic: false,
            cpo: false,
            ..ComputeOptions::default()
        };
        let res = run_inla_inference_a_cancellable(
            &[0.0],
            &build,
            &|_| 0.0,
            &obs,
            Some(&a),
            None,
            "ccd",
            1.0,
            &crate::marginals::MarginalOptions::default(),
            true,
            None,
            None,
            Some(&compute),
        )
        .expect("inference");
        assert!(res.dic.is_nan());
        assert!(res.waic.is_nan());
        assert!(res.cpo.iter().all(|v| v.is_none()));
    }
}
