use crate::inference::{eval_likelihood, log_gamma, Obs};

/// Default log-scale difference threshold for CPO failure detection.
/// From the R-INLA manual: if the difference between the maximum of the
/// leave-one-out density and its border value is less than this, the
/// computation is flagged as unreliable.
const CPO_DIFF: f64 = 3.0;

// ────────────────────────────────────────────────────────────────────
// 1. Marginal Likelihood (Gaussian approximation)
// ────────────────────────────────────────────────────────────────────

/// Compute the Gaussian approximation to the log-marginal-likelihood:
///
///   log π(y) ≈ log π̃(θ*|y) + m/2 log(2π) − ½ log det(−H)
///
/// where θ* is the mode, H is the Hessian of the log-posterior at the mode,
/// and π̃(θ*|y) is the unnormalised log-posterior at the mode.
///
/// Arguments:
/// - `log_post_at_mode`: unnormalised log-posterior evaluated at the mode θ*
/// - `neg_hessian`: the *negative* Hessian of the log-posterior at θ*,
///    stored row-major as m×m (i.e. −H, which should be positive-definite)
/// - `m`: number of hyperparameters
pub fn compute_marginal_log_lik_gaussian(
    log_post_at_mode: f64,
    neg_hessian: &[f64],
    m: usize,
) -> Result<f64, String> {
    if m == 0 {
        // No hyperparameters: the Laplace marginal from find_latent_mode is already exact.
        return Ok(log_post_at_mode);
    }
    if neg_hessian.len() != m * m {
        return Err("neg_hessian length must be m*m".to_string());
    }

    // log det(−H) via Cholesky-like decomposition (dense, m is small)
    let mut a = neg_hessian.to_vec();
    let mut log_det = 0.0;
    for j in 0..m {
        for k in 0..j {
            let ljk = a[j * m + k];
            a[j * m + j] -= ljk * ljk;
        }
        if a[j * m + j] <= 0.0 {
            return Err("negative Hessian is not positive-definite".to_string());
        }
        log_det += a[j * m + j].ln();
        let djj = a[j * m + j].sqrt();
        a[j * m + j] = djj;
        for i in (j + 1)..m {
            for k in 0..j {
                a[i * m + j] -= a[i * m + k] * a[j * m + k];
            }
            a[i * m + j] /= djj;
        }
    }

    Ok(log_post_at_mode + 0.5 * (m as f64) * (2.0 * std::f64::consts::PI).ln() - 0.5 * log_det)
}

// ────────────────────────────────────────────────────────────────────
// 2. DIC
// ────────────────────────────────────────────────────────────────────

/// Result of a DIC computation.
#[derive(Debug, Clone, PartialEq)]
pub struct DicResult {
    /// The DIC value: D̄ + p_D
    pub dic: f64,
    /// Posterior mean deviance: D̄ = Σ_k w_k D(θ_k)
    pub mean_deviance: f64,
    /// Effective number of parameters: p_D = D̄ − D(θ*)
    pub effective_params: f64,
}

/// Compute the deviance −2 Σ_i log π(y_i | x*_i) for a single configuration.
fn deviance_at_config(obs: &[Obs], latent_mode: &[f64]) -> Result<f64, String> {
    let n = obs.len();
    if latent_mode.len() != n {
        return Err("latent_mode length must match obs length".to_string());
    }
    let mut log_lik = 0.0;
    for i in 0..n {
        log_lik += eval_likelihood(latent_mode[i], &obs[i])?.logp;
    }
    Ok(-2.0 * log_lik)
}

/// Compute DIC from the CCD integration results.
///
/// Arguments:
/// - `obs`: the observation vector
/// - `cond_modes`: for each integration point k, the conditional latent mode x*(θ_k)
/// - `norm_weights`: normalised integration weights w_k (sum to 1)
/// - `mode_index`: index of the integration point closest to θ* (used for D(θ*))
pub fn compute_dic(
    obs: &[Obs],
    cond_modes: &[Vec<f64>],
    norm_weights: &[f64],
    mode_index: usize,
) -> Result<DicResult, String> {
    let k_total = cond_modes.len();
    if norm_weights.len() != k_total {
        return Err("norm_weights length must match cond_modes length".to_string());
    }
    if mode_index >= k_total {
        return Err("mode_index out of range".to_string());
    }

    let mut mean_deviance = 0.0;
    for k in 0..k_total {
        let d_k = deviance_at_config(obs, &cond_modes[k])?;
        mean_deviance += norm_weights[k] * d_k;
    }

    let d_at_mode = deviance_at_config(obs, &cond_modes[mode_index])?;
    let effective_params = mean_deviance - d_at_mode;

    Ok(DicResult {
        dic: mean_deviance + effective_params,
        mean_deviance,
        effective_params,
    })
}

// ────────────────────────────────────────────────────────────────────
// 3. CPO / PIT
// ────────────────────────────────────────────────────────────────────

/// Result of CPO/PIT computation for all observations.
#[derive(Debug, Clone, PartialEq)]
pub struct CpoResult {
    /// CPO_i = π(y_i | y_{-i}). None when the computation fails.
    pub cpo: Vec<Option<f64>>,
    /// PIT_i = Pr(y^new_i ≤ y_i | y_{-i}). None when the computation fails
    /// or when the CDF is not implemented for the observation family.
    pub pit: Vec<Option<f64>>,
    /// Number of observations where CPO failed.
    pub n_failures: usize,
}

/// Compute CPO and PIT for all observations.
///
/// CPO_i = 1 / Σ_k w_k / π(y_i | x*_i(θ_k), θ_k)
///
/// The failure heuristic follows the R-INLA manual:
/// 1. **Monotonicity**: If the conditional density π̃(x_i | y_{-i}, θ_k) is
///    monotonically increasing or decreasing across integration points, flag as None.
/// 2. **CPO.DIFF threshold**: If the difference between the maximum log-density
///    and the border (first or last) value is less than CPO_DIFF (default 3),
///    flag as None.
///
/// Arguments:
/// - `obs`: observation vector
/// - `cond_modes`: for each integration point k, the conditional latent mode x*(θ_k)
/// - `cond_vars`: for each integration point k, the conditional marginal variances
/// - `norm_weights`: normalised integration weights (sum to 1)
pub fn compute_cpo_pit(
    obs: &[Obs],
    cond_modes: &[Vec<f64>],
    cond_vars: &[Vec<f64>],
    norm_weights: &[f64],
) -> Result<CpoResult, String> {
    let n = obs.len();
    let k_total = cond_modes.len();
    if norm_weights.len() != k_total || cond_vars.len() != k_total {
        return Err("cond_modes, cond_vars, and norm_weights must have equal length".to_string());
    }
    for k in 0..k_total {
        if cond_modes[k].len() != n || cond_vars[k].len() != n {
            return Err("each cond_modes/cond_vars entry must have length n".to_string());
        }
    }

    let mut cpo = vec![None; n];
    let mut pit = vec![None; n];
    let mut n_failures = 0;

    for i in 0..n {
        // --- Failure heuristic ---
        let mut failed = false;
        for k in 0..k_total {
            if check_cpo_failure_for_obs(&obs[i], cond_modes[k][i], cond_vars[k][i]) {
                failed = true;
                break;
            }
        }
        if failed {
            n_failures += 1;
            continue;
        }

        // Collect log-likelihoods across integration points for CPO calculation
        let mut log_liks = Vec::with_capacity(k_total);
        let mut any_bad = false;
        for k in 0..k_total {
            match eval_likelihood(cond_modes[k][i], &obs[i]) {
                Ok(e) => log_liks.push(e.logp),
                Err(_) => {
                    any_bad = true;
                    break;
                }
            }
        }

        if any_bad || log_liks.len() < 2 {
            n_failures += 1;
            continue;
        }

        // --- CPO computation ---
        // CPO_i = 1 / Σ_k w_k / π(y_i | x*_i(θ_k))
        //       = 1 / Σ_k w_k · exp(-log_lik_ik)
        let max_neg = log_liks.iter().map(|&v| -v).fold(f64::NEG_INFINITY, f64::max);
        let inv_cpo: f64 = norm_weights
            .iter()
            .zip(log_liks.iter())
            .map(|(&w, &ll)| w * (-ll - max_neg).exp())
            .sum();
        let inv_cpo_scaled = inv_cpo * max_neg.exp();

        if !inv_cpo_scaled.is_finite() || inv_cpo_scaled <= 0.0 {
            n_failures += 1;
            continue;
        }

        cpo[i] = Some(1.0 / inv_cpo_scaled);

        // --- PIT computation ---
        // PIT_i = Σ_k w_k · F(y_i | μ_ik, σ²_ik)
        let mut pit_sum = 0.0;
        let mut pit_ok = true;
        for k in 0..k_total {
            match observation_cdf(&obs[i], cond_modes[k][i], cond_vars[k][i]) {
                Some(f) => pit_sum += norm_weights[k] * f,
                None => {
                    pit_ok = false;
                    break;
                }
            }
        }
        if pit_ok && pit_sum.is_finite() {
            pit[i] = Some(pit_sum.clamp(0.0, 1.0));
        }
    }

    Ok(CpoResult {
        cpo,
        pit,
        n_failures,
    })
}

/// Check whether the CPO computation for a single observation should be
/// flagged as a failure by assessing the tail behavior of its leave-one-out density.
///
/// Returns `true` if the computation is unreliable.
pub fn check_cpo_failure_for_obs(
    obs_i: &Obs,
    mu_ik: f64,
    var_ik: f64,
) -> bool {
    if var_ik <= 0.0 || !var_ik.is_finite() {
        return true;
    }

    // 1. Get gradient and hessian of log-likelihood at full posterior mode mu_ik
    let ev = match eval_likelihood(mu_ik, obs_i) {
        Ok(e) => e,
        Err(_) => return true,
    };
    let g = ev.grad;
    let c = -ev.hess;

    // 2. Compute leave-one-out precision and variance
    let tau_loo = 1.0 / var_ik - c;
    if tau_loo <= 0.0 || !tau_loo.is_finite() {
        return true;
    }
    let var_loo = 1.0 / tau_loo;
    let std_loo = var_loo.sqrt();

    // 3. Compute leave-one-out mean
    let mu_loo = var_loo * (mu_ik / var_ik - c * mu_ik + g);

    // 4. Evaluate unnormalized leave-one-out log-density at 5 grid points of x_i
    let steps = [-3.0, -1.5, 0.0, 1.5, 3.0];
    let mut log_densities = Vec::with_capacity(5);
    for &step in &steps {
        let x = mu_loo + step * std_loo;
        let log_prior_part = -0.5 * (x - mu_ik) * (x - mu_ik) / var_ik;
        let log_lik_part = match eval_likelihood(x, obs_i) {
            Ok(e) => e.logp,
            Err(_) => return true,
        };
        log_densities.push(log_prior_part + log_lik_part);
    }

    // 5. Monotonicity check: are the values monotonically increasing or decreasing?
    let mut increasing = true;
    let mut decreasing = true;
    for j in 1..5 {
        if log_densities[j] < log_densities[j - 1] {
            increasing = false;
        }
        if log_densities[j] > log_densities[j - 1] {
            decreasing = false;
        }
    }
    if increasing || decreasing {
        return true;
    }

    // 6. CPO.DIFF check: max log-density must exceed both borders by at least CPO_DIFF (3.0)
    let max_log_d = log_densities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let border_left = log_densities[0];
    let border_right = log_densities[4];

    if max_log_d - border_left < CPO_DIFF || max_log_d - border_right < CPO_DIFF {
        return true;
    }

    false
}

/// Compute the CDF F(y | η, σ²) for supported observation families.
///
/// Returns `None` for families where the CDF is not yet implemented.
fn observation_cdf(obs: &Obs, eta: f64, variance: f64) -> Option<f64> {
    match obs {
        Obs::Gaussian(g) => {
            // Gaussian CDF: Φ((y - μ) / σ) where μ = η (identity link) and
            // σ² combines observation variance and latent variance.
            let mu = eta; // identity link assumed for CDF; more general handling later
            let total_var = 1.0 / g.precision + variance;
            if total_var <= 0.0 {
                return None;
            }
            let z = (g.y - mu) / total_var.sqrt();
            Some(normal_cdf(z))
        }
        Obs::Poisson(p) => {
            // Poisson CDF: Pr(Y ≤ y) = Σ_{j=0}^{floor(y)} e^{-λ} λ^j / j!
            // where λ = exposure · exp(η) (log link assumed)
            let lambda = p.exposure * eta.exp();
            if lambda < 0.0 || !lambda.is_finite() {
                return None;
            }
            let y_int = p.y.floor() as i64;
            if y_int < 0 {
                return Some(0.0);
            }
            Some(poisson_cdf(lambda, y_int as u64))
        }
        Obs::Binomial(b) => {
            // Binomial CDF: Pr(Y ≤ y) = Σ_{j=0}^{floor(y)} C(n,j) p^j (1-p)^{n-j}
            // where p = logistic(η) (logit link assumed)
            let prob = logistic(eta);
            let n = b.n as u64;
            let y_int = b.y.floor() as i64;
            if y_int < 0 {
                return Some(0.0);
            }
            let y_u = (y_int as u64).min(n);
            Some(binomial_cdf(n, prob, y_u))
        }
        // Families not yet implemented: return None
        _ => None,
    }
}

/// Standard normal CDF via the Abramowitz & Stegun rational approximation.
fn normal_cdf(x: f64) -> f64 {
    // Hart's approximation (|error| < 7.5e-8)
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

/// Poisson CDF: Pr(X ≤ k) for X ~ Poisson(lambda).
/// Uses the regularised incomplete gamma function identity:
///   Pr(X ≤ k) = Q(k+1, λ) = 1 − P(k+1, λ)
/// For small k we sum directly; for larger k we use the series expansion.
fn poisson_cdf(lambda: f64, k: u64) -> f64 {
    if lambda <= 0.0 {
        return 1.0;
    }
    // Direct summation for moderate k
    let mut cdf = 0.0;
    let mut log_pmf = -lambda; // log(e^{-λ} λ^0 / 0!) = -λ
    for j in 0..=k {
        cdf += log_pmf.exp();
        if j < k {
            log_pmf += lambda.ln() - ((j + 1) as f64).ln();
        }
    }
    cdf.min(1.0)
}

/// Binomial CDF: Pr(X ≤ k) for X ~ Binomial(n, p).
/// Direct summation using log-space for numerical stability.
fn binomial_cdf(n: u64, p: f64, k: u64) -> f64 {
    if p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return if k >= n { 1.0 } else { 0.0 };
    }
    let mut cdf = 0.0;
    for j in 0..=k {
        let log_choose = log_gamma((n as f64) + 1.0)
            - log_gamma((j as f64) + 1.0)
            - log_gamma((n - j) as f64 + 1.0);
        let log_pmf = log_choose + (j as f64) * p.ln() + ((n - j) as f64) * (1.0 - p).ln();
        cdf += log_pmf.exp();
    }
    cdf.min(1.0)
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

// ────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{
        GaussianObs, Link, Obs, PoissonObs,
    };

    fn approx(a: f64, b: f64, eps: f64) {
        assert!(
            (a - b).abs() < eps,
            "left={a}, right={b}, diff={}, eps={eps}",
            (a - b).abs()
        );
    }

    // --- Marginal likelihood ---

    #[test]
    fn gaussian_mlik_trivial_1d() {
        // For a 1-D Gaussian posterior with precision τ,
        // −H = τ, so log det(−H) = ln(τ).
        // Gaussian approx: log_post + 0.5 ln(2π) − 0.5 ln(τ)
        let tau = 4.0;
        let log_post = -3.0;
        let neg_hessian = vec![tau];
        let result = compute_marginal_log_lik_gaussian(log_post, &neg_hessian, 1).unwrap();
        let expected = log_post + 0.5 * (2.0 * std::f64::consts::PI).ln() - 0.5 * tau.ln();
        approx(result, expected, 1e-12);
    }

    #[test]
    fn gaussian_mlik_2d() {
        // 2×2 diagonal: −H = diag(2, 8)
        let neg_hessian = vec![2.0, 0.0, 0.0, 8.0];
        let log_post = -5.0;
        let result = compute_marginal_log_lik_gaussian(log_post, &neg_hessian, 2).unwrap();
        let expected = log_post + (2.0 * std::f64::consts::PI).ln() - 0.5 * (2.0_f64 * 8.0).ln();
        approx(result, expected, 1e-12);
    }

    #[test]
    fn gaussian_mlik_zero_hyperparams() {
        let result = compute_marginal_log_lik_gaussian(-2.0, &[], 0).unwrap();
        approx(result, -2.0, 1e-12);
    }

    // --- DIC ---

    #[test]
    fn dic_sanity_gaussian() {
        let obs = vec![
            Obs::Gaussian(GaussianObs { y: 1.0, precision: 2.0, link: Link::Identity }),
            Obs::Gaussian(GaussianObs { y: 2.0, precision: 2.0, link: Link::Identity }),
        ];
        // Two integration points: one at the mode, one slightly off
        let modes = vec![
            vec![1.0, 2.0],   // point 0: "mode" — perfect fit
            vec![1.1, 1.9],   // point 1: slightly off
        ];
        let weights = vec![0.6, 0.4];

        let dic = compute_dic(&obs, &modes, &weights, 0).unwrap();

        // D(θ*) at mode should be minimal (perfect fit → deviance = const)
        assert!(dic.mean_deviance.is_finite());
        assert!(dic.effective_params >= 0.0, "p_D should be non-negative for this case");
        // DIC = D̄ + p_D = D̄ + (D̄ − D(θ*)) = 2·D̄ − D(θ*)
        approx(dic.dic, dic.mean_deviance + dic.effective_params, 1e-12);
    }

    // --- CPO failure detection ---

    #[test]
    fn cpo_failure_detection_stable() {
        let obs = Obs::Gaussian(GaussianObs { y: 1.0, precision: 2.0, link: Link::Identity });
        // Full posterior mode and variance: tau_post = 5.0, var = 0.2
        // tau_loo = 5.0 - 2.0 = 3.0 > 0.0 (stable)
        assert!(!check_cpo_failure_for_obs(&obs, 1.0, 0.2));
    }

    #[test]
    fn cpo_failure_detection_unstable_variance() {
        let obs = Obs::Gaussian(GaussianObs { y: 1.0, precision: 2.0, link: Link::Identity });
        // Full posterior variance = 0.6 (tau_post = 1.67)
        // tau_loo = 1.67 - 2.0 = -0.33 <= 0.0 (unstable, should fail)
        assert!(check_cpo_failure_for_obs(&obs, 1.0, 0.6));
    }

    #[test]
    fn cpo_failure_detection_monotonic() {
        // High precision and far from mode -> linear gradient dominates -> monotonic
        let obs = Obs::Gaussian(GaussianObs { y: 0.0, precision: 5.0, link: Link::Identity });
        assert!(check_cpo_failure_for_obs(&obs, 10.0, 0.1));
    }

    #[test]
    fn cpo_failure_detection_small_diff() {
        // Highly asymmetric Poisson likelihood -> one side decays slowly -> diff < 3.0
        let obs = Obs::Poisson(PoissonObs { y: 1.0, exposure: 10.0, link: Link::Log });
        assert!(check_cpo_failure_for_obs(&obs, 0.0, 0.05));
    }

    // --- CPO/PIT integration ---

    #[test]
    fn cpo_pit_gaussian_well_specified() {
        let n = 3;
        let obs: Vec<Obs> = (0..n)
            .map(|i| {
                Obs::Gaussian(GaussianObs {
                    y: 1.0 + 0.1 * i as f64,
                    precision: 2.0,
                    link: Link::Identity,
                })
            })
            .collect();

        // Simulate 5 integration points with well-separated log-likelihoods
        let k_total = 5;
        let mut cond_modes = Vec::new();
        let mut cond_vars = Vec::new();
        for k in 0..k_total {
            let offset = (k as f64 - 2.0) * 0.5; // -1.0, -0.5, 0.0, 0.5, 1.0
            let modes: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * i as f64 + offset).collect();
            let vars = vec![0.5; n];
            cond_modes.push(modes);
            cond_vars.push(vars);
        }
        let norm_weights = vec![0.1, 0.2, 0.4, 0.2, 0.1];
        let result = compute_cpo_pit(&obs, &cond_modes, &cond_vars, &norm_weights).unwrap();

        // For this well-specified model, CPO should produce Some values > 0
        for i in 0..n {
            if let Some(c) = result.cpo[i] {
                assert!(c > 0.0, "CPO[{i}] should be positive, got {c}");
            }
            // PIT should be in (0, 1) for Gaussian
            if let Some(p) = result.pit[i] {
                assert!(
                    p > 0.0 && p < 1.0,
                    "PIT[{i}] should be in (0,1), got {p}"
                );
            }
        }
    }

    // --- CDF helpers ---

    #[test]
    fn normal_cdf_basic() {
        approx(normal_cdf(0.0), 0.5, 1e-6);
        assert!(normal_cdf(3.0) > 0.998);
        assert!(normal_cdf(-3.0) < 0.002);
    }

    #[test]
    fn poisson_cdf_basic() {
        // Poisson(1): Pr(X ≤ 0) = e^{-1} ≈ 0.3679
        approx(poisson_cdf(1.0, 0), (-1.0_f64).exp(), 1e-10);
        // Poisson(1): Pr(X ≤ 10) should be very close to 1
        assert!(poisson_cdf(1.0, 10) > 0.9999);
    }

    #[test]
    fn binomial_cdf_basic() {
        // Binomial(10, 0.5): Pr(X ≤ 10) = 1
        approx(binomial_cdf(10, 0.5, 10), 1.0, 1e-10);
        // Binomial(10, 0.5): Pr(X ≤ 4) ≈ 0.3770
        let cdf_4 = binomial_cdf(10, 0.5, 4);
        approx(cdf_4, 0.376953125, 1e-8);
    }

    // --- Full integration test: marginal likelihoods in the same ballpark ---

    #[test]
    fn marginal_likelihoods_agree_roughly() {
        // IID Gaussian model: n=5, obs precision=2, prior precision=1
        let n = 5;
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

        let build_prior = |theta: &[f64]| -> Result<crate::sparse::CscMatrix, String> {
            let tau = theta[0].exp();
            let mut tri = sprs::TriMatI::<f64, usize>::with_capacity((n, n), n);
            for i in 0..n {
                tri.add_triplet(i, i, tau);
            }
            Ok(tri.to_csc())
        };

        let log_prior_density = |theta: &[f64]| -> f64 { -0.5 * 0.1 * theta[0] * theta[0] };

        let result = crate::inference::run_inla_inference(
            &[0.0],
            &build_prior,
            &log_prior_density,
            &obs,
            "ccd",
            1.0,
        )
        .expect("inference should succeed");

        // The numerical and Gaussian marginal log-likelihoods should be
        // in the same ballpark (within ~1 unit on log-scale for this simple model).
        let diff = (result.marginal_log_lik - result.marginal_log_lik_gaussian).abs();
        assert!(
            diff < 2.0,
            "Numerical ({}) and Gaussian ({}) marginal log-likelihoods differ by {diff}",
            result.marginal_log_lik,
            result.marginal_log_lik_gaussian,
        );
    }

    #[test]
    fn dic_from_inference_result() {
        let n = 5;
        let y_obs = vec![1.0, 1.2, 0.9, 1.1, 0.8];

        let mut obs = Vec::new();
        for &y in &y_obs {
            obs.push(Obs::Gaussian(GaussianObs {
                y,
                precision: 2.0,
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

        let log_prior_density = |theta: &[f64]| -> f64 { -0.5 * 0.1 * theta[0] * theta[0] };

        let result = crate::inference::run_inla_inference(
            &[0.0],
            &build_prior,
            &log_prior_density,
            &obs,
            "ccd",
            1.0,
        )
        .expect("inference should succeed");

        assert!(result.dic.is_finite(), "DIC should be finite");
        assert!(result.effective_params >= 0.0, "p_D should be non-negative");
        approx(
            result.dic,
            result.mean_deviance + result.effective_params,
            1e-12,
        );
    }
}
