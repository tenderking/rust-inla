use rayon::prelude::*;

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
pub struct Eval1D {
    pub logp: f64,
    pub grad: f64,
    pub hess: f64,
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
    pub event: f64,
    pub link: Link,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeibullSurvivalObs {
    pub y: f64,
    pub event: f64,
    pub shape: f64,
    pub link: Link,
}

const LOG_NORMC_GAUSSIAN: f64 = -0.918_938_533_204_672_8;

pub fn eval_prior_gaussian(theta: f64, p: GaussianPrior) -> Result<Eval1D, String> {
    if p.precision < 0.0 || !p.precision.is_finite() {
        return Err("gaussian prior precision must be finite and >= 0".to_string());
    }
    if p.precision == 0.0 {
        return Ok(Eval1D {
            logp: 0.0,
            grad: 0.0,
            hess: 0.0,
        });
    }
    let d = theta - p.mean;
    Ok(Eval1D {
        logp: LOG_NORMC_GAUSSIAN + 0.5 * p.precision.ln() - 0.5 * p.precision * d * d,
        grad: -p.precision * d,
        hess: -p.precision,
    })
}

pub fn eval_prior_gamma(x: f64, p: GammaPrior) -> Result<Eval1D, String> {
    if x <= 0.0 || !x.is_finite() {
        return Err("gamma prior is defined for finite x > 0".to_string());
    }
    if p.shape <= 0.0 || p.scale <= 0.0 || !p.shape.is_finite() || !p.scale.is_finite() {
        return Err("gamma prior shape/scale must be finite and > 0".to_string());
    }
    let a = p.shape;
    let b = p.scale;
    let logp = (a - 1.0) * (x / b).ln() - x / b - log_gamma(a) - b.ln();
    let grad = (a - 1.0) / x - 1.0 / b;
    let hess = -(a - 1.0) / (x * x);
    Ok(Eval1D { logp, grad, hess })
}

pub fn eval_prior_loggamma(theta: f64, p: GammaPrior) -> Result<Eval1D, String> {
    if !theta.is_finite() {
        return Err("log-gamma prior input must be finite".to_string());
    }
    if p.shape <= 0.0 || p.scale <= 0.0 || !p.shape.is_finite() || !p.scale.is_finite() {
        return Err("log-gamma prior shape/scale must be finite and > 0".to_string());
    }
    let x = theta.exp();
    let base = eval_prior_gamma(x, p)?;
    Ok(Eval1D {
        logp: base.logp + theta,
        grad: base.grad * x + 1.0,
        hess: base.hess * x * x + base.grad * x,
    })
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
    let d2mu = o.exposure * d2l;
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
    let hess = if mu > 0.0 {
        -(o.y / (mu * mu)) * dmu * dmu + (o.y / mu - 1.0) * d2mu
    } else {
        -d2mu
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
    let (p, dp, d2p) = link_forward(eta, o.link)?;
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

    let a = y / p - (n - y) / (1.0 - p);
    let b = -y / (p * p) - (n - y) / ((1.0 - p) * (1.0 - p));
    Ok(Eval1D {
        logp,
        grad: a * dp,
        hess: b * dp * dp + a * d2p,
    })
}

pub fn eval_likelihood_negative_binomial(eta: f64, o: NegativeBinomialObs) -> Result<Eval1D, String> {
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

    let logp = log_gamma(y + r) - log_gamma(r) - log_gamma(y + 1.0) + r * (r / mu_r).ln() + y * (mu / mu_r).ln();
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
                return Err("type1 zero-inflated binomial requires base probability <= 1 - zero_prob".to_string());
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
    let dlog_drate = o.event / rate - o.y;
    let d2log_drate2 = -o.event / (rate * rate);
    Ok(Eval1D {
        logp: o.event * rate.ln() - rate * o.y,
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
    let t = (lambda * o.y).powf(k);
    let dlog_dlambda = k * (o.event - t) / lambda;
    let d2log_dlambda2 = k * ((1.0 - k) * t - o.event) / (lambda * lambda);

    Ok(Eval1D {
        logp: o.event * (k.ln() + k * lambda.ln() + (k - 1.0) * o.y.ln()) - t,
        grad: dlog_dlambda * dlambda,
        hess: d2log_dlambda2 * dlambda * dlambda + dlog_dlambda * d2lambda,
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
        Obs::None => Ok(Eval1D { logp: 0.0, grad: 0.0, hess: 0.0 }),
    }
}

pub fn find_latent_mode(
    q_prior: &crate::sparse::CscMatrix,
    obs: &[Obs],
    max_iter: usize,
    tol: f64,
) -> Result<(Vec<f64>, crate::ldlt::LdltFactor, f64), String> {
    let n = q_prior.rows();
    if obs.len() != n {
        return Err("Observations length must match prior precision dimension".to_string());
    }

    let mut x = vec![0.0; n];
    let mut ldlt = None;

    for iter in 0..max_iter {
        let mut evals = Vec::with_capacity(n);
        for i in 0..n {
            evals.push(eval_likelihood(x[i], &obs[i])?);
        }

        let (step, factor) = crate::ldlt::laplace_newton_step(q_prior, &evals)?;
        let mut max_diff = 0.0;
        for i in 0..n {
            if !step[i].is_finite() {
                return Err("Newton-Raphson step is not finite (contains NaN or Inf)".to_string());
            }
            x[i] += step[i];
            max_diff = f64::max(max_diff, step[i].abs());
        }

        // Newton system is the current posterior precision. On convergence
        // (exact for Gaussian after the confirming step) reuse that factor.
        if max_diff < tol {
            ldlt = Some(factor);
            break;
        }

        if iter == max_iter - 1 {
            return Err("Newton-Raphson did not converge".to_string());
        }
    }

    let factor = ldlt.ok_or_else(|| "Failed to factorize posterior precision".to_string())?;

    // log|Q_prior| via dense factor of Q_prior + εI (avoids CSC rebuild).
    let mut a_prior = crate::ldlt::csc_to_dense(q_prior)?;
    for i in 0..n {
        a_prior[i * n + i] += 1e-12;
    }
    let factor_prior = crate::ldlt::ldlt_factorize_dense(&a_prior, n)?;

    let log_det_prior = factor_prior.d.iter().map(|&v| v.abs().ln()).sum::<f64>();
    let log_det_post = factor.d.iter().map(|&v| v.abs().ln()).sum::<f64>();

    let mut q_x = vec![0.0; n];
    for (col, colvec) in q_prior.outer_iterator().enumerate() {
        for (row, value) in colvec.iter() {
            q_x[row] += value * x[col];
        }
    }
    let quad_prior = q_x.iter().zip(&x).map(|(a, b)| a * b).sum::<f64>();

    let mut log_lik = 0.0;
    for i in 0..n {
        log_lik += eval_likelihood(x[i], &obs[i])?.logp;
    }

    let marginal_log_lik = log_lik - 0.5 * quad_prior + 0.5 * log_det_prior - 0.5 * log_det_post;

    Ok((x, factor, marginal_log_lik))
}

#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub mode: Vec<f64>,
    pub hessian: Vec<f64>,
    pub latent_means: Vec<f64>,
    pub latent_variances: Vec<f64>,
    pub marginal_log_lik: f64,
    /// Gaussian approximation to the log-marginal-likelihood
    pub marginal_log_lik_gaussian: f64,
    /// DIC = D̄ + p_D
    pub dic: f64,
    /// Posterior mean deviance D̄
    pub mean_deviance: f64,
    /// Effective number of parameters p_D = D̄ − D(θ*)
    pub effective_params: f64,
    /// CPO_i = π(y_i | y_{-i}), None when computation fails
    pub cpo: Vec<Option<f64>>,
    /// PIT_i = Pr(y^new_i ≤ y_i | y_{-i}), None when fails or unsupported family
    pub pit: Vec<Option<f64>>,
    /// Number of CPO failures
    pub cpo_n_failures: usize,
}

pub fn run_inla_inference(
    initial_theta: &[f64],
    build_prior: &(dyn Fn(&[f64]) -> Result<crate::sparse::CscMatrix, String> + Sync),
    log_prior_density: &(dyn Fn(&[f64]) -> f64 + Sync),
    obs: &[Obs],
    strategy: &str,
    step_or_f0: f64,
) -> Result<InferenceResult, String> {
    let m = initial_theta.len();
    let n = obs.len();

    let config = crate::hyper_opt::ModelConfig {
        build_prior,
        log_prior_density,
        obs,
    };

    let mode = crate::hyper_opt::nelder_mead(initial_theta, 0.1, 200, 1e-6, &config)?;

    let hessian = crate::hyper_opt::compute_hessian(&mode, &config, 1e-4)?;

    let neg_hessian = hessian.iter().map(|&x| -x).collect::<Vec<_>>();
    let sigma = crate::integration::invert_symmetric_matrix(&neg_hessian, m)?;

    let (lambdas, v) = crate::integration::jacobi_eigen(&sigma, m, 100)?;

    let (z_points, z_weights) = match strategy.to_lowercase().as_str() {
        "grid" => {
            let evaluator = |z: &[f64]| -> f64 {
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
            crate::integration::grid_design(m, step_or_f0, 4.0, &evaluator)?
        }
        _ => {
            crate::integration::ccd_design(m, step_or_f0)?
        }
    };

    // Track which integration point is closest to the mode (z = 0)
    let mut mode_index = 0;
    let mut min_z_norm = f64::INFINITY;
    for (k, z) in z_points.iter().enumerate() {
        let z_norm: f64 = z.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if z_norm < min_z_norm {
            min_z_norm = z_norm;
            mode_index = k;
        }
    }

    // Parallel integration loop using Rayon
    let results: Vec<(Vec<f64>, Vec<f64>, f64)> = z_points
        .par_iter()
        .map(|z| {
            let mut theta = mode.clone();
            for i in 0..m {
                let mut diff = 0.0;
                for j in 0..m {
                    diff += v[i * m + j] * lambdas[j].abs().sqrt() * z[j];
                }
                theta[i] += diff;
            }

            let q_prior = build_prior(&theta)?;
            let (x_star, ldlt, marginal_log_lik) =
                find_latent_mode(&q_prior, obs, 50, 1e-5)?;

            let variances = crate::ldlt::ldlt_diagonal_inverse(&ldlt)?;

            let log_prior = log_prior_density(&theta);
            let log_post = marginal_log_lik + log_prior;

            Ok((x_star, variances, log_post))
        })
        .collect::<Result<Vec<_>, String>>()?;

    let mut cond_means = Vec::with_capacity(results.len());
    let mut cond_vars = Vec::with_capacity(results.len());
    let mut log_posts = Vec::with_capacity(results.len());

    for (x_star, variances, log_post) in results {
        cond_means.push(x_star);
        cond_vars.push(variances);
        log_posts.push(log_post);
    }

    let max_log_post = log_posts.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let scaled_densities: Vec<f64> = log_posts.iter().map(|&lp| (lp - max_log_post).exp()).collect();

    let mut sum_w_dens = 0.0;
    for k in 0..cond_means.len() {
        sum_w_dens += z_weights[k] * scaled_densities[k];
    }

    let norm_weights: Vec<f64> = z_weights.iter().zip(&scaled_densities)
        .map(|(&w, &d)| w * d / sum_w_dens)
        .collect();

    let mut latent_means = vec![0.0; n];
    let mut latent_variances = vec![0.0; n];

    for i in 0..n {
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

    let marginal_log_lik = max_log_post + sum_w_dens.abs().ln();

    // --- Model selection criteria ---

    // 1. Gaussian approximation to marginal likelihood
    let neg_hessian: Vec<f64> = hessian.iter().map(|&x| -x).collect();
    let marginal_log_lik_gaussian = crate::model_selection::compute_marginal_log_lik_gaussian(
        log_posts[mode_index],
        &neg_hessian,
        m,
    ).unwrap_or(f64::NAN);

    // 2. DIC
    let dic_result = crate::model_selection::compute_dic(
        obs,
        &cond_means,
        &norm_weights,
        mode_index,
    )?;

    // 3. CPO / PIT
    let cpo_result = crate::model_selection::compute_cpo_pit(
        obs,
        &cond_means,
        &cond_vars,
        &norm_weights,
    )?;

    Ok(InferenceResult {
        mode,
        hessian,
        latent_means,
        latent_variances,
        marginal_log_lik,
        marginal_log_lik_gaussian,
        dic: dic_result.dic,
        mean_deviance: dic_result.mean_deviance,
        effective_params: dic_result.effective_params,
        cpo: cpo_result.cpo,
        pit: cpo_result.pit,
        cpo_n_failures: cpo_result.n_failures,
    })
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

fn validate_event_indicator(v: f64, label: &str) -> Result<(), String> {
    if !v.is_finite() || (v != 0.0 && v != 1.0) {
        return Err(format!("{label} must be 0 or 1"));
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
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
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
                shape: 1.8,
                link: Link::Log,
            },
        )
        .expect("ok");
        assert!(out.logp.is_finite());
        assert!(out.grad.is_finite());
        assert!(out.hess.is_finite());
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
        let build_prior = |theta: &[f64]| -> Result<crate::sparse::CscMatrix, String> {
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

        let build_prior = |theta: &[f64]| -> Result<crate::sparse::CscMatrix, String> {
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
        let result = run_inla_inference(
            &[0.0],
            &build_prior,
            &log_prior_density,
            &obs,
            "grid",
            1.0,
        )
        .expect("inference should succeed");

        assert_eq!(result.mode.len(), 1);
        assert!(result.mode[0].is_finite());
        assert_eq!(result.hessian.len(), 1);
        assert!(result.hessian[0] < 0.0);

        assert_eq!(result.latent_means.len(), n);
        assert_eq!(result.latent_variances.len(), n);
    }
}
