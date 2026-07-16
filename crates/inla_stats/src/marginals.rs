//! 1D posterior marginal grids and quantile helpers (classic INLA-style).

/// Discrete 1D density on an evaluation grid (classic `cbind(x, y)` shape).
#[derive(Debug, Clone, PartialEq)]
pub struct Marginal1D {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

impl Marginal1D {
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Trapezoidal integral of the density (should be ≈ 1 after normalization).
    pub fn integrate(&self) -> f64 {
        if self.x.len() < 2 {
            return 0.0;
        }
        let mut s = 0.0;
        for i in 0..self.x.len() - 1 {
            let dx = self.x[i + 1] - self.x[i];
            s += 0.5 * (self.y[i] + self.y[i + 1]) * dx;
        }
        s
    }

    /// Renormalize so ∫ y dx = 1 (no-op if mass is non-positive).
    pub fn normalize(&mut self) {
        let mass = self.integrate();
        if mass > 0.0 && mass.is_finite() {
            for v in &mut self.y {
                *v /= mass;
            }
        }
    }
}

/// Controls which 1D density grids are materialised after integration.
///
/// Hyperparameter marginals are cheap (dim = m). Latent/predictor grids are
/// opt-in so large spatial fields (n ≫ 10⁴) do not allocate `n × n_points`.
#[derive(Debug, Clone)]
pub struct MarginalOptions {
    /// Evaluation points per 1D grid (classic default ≈ 201).
    pub n_points: usize,
    /// Always build internal-scale hyperparameter marginals when true.
    pub hyperpar: bool,
    /// Latent indices (0-based) for mixture density grids. Empty = skip.
    pub latent_indices: Vec<usize>,
    /// Predictor indices (0-based) for mixture density grids. Empty = skip.
    pub predictor_indices: Vec<usize>,
    /// Half-width of the evaluation window in mixture SDs.
    pub n_sd: f64,
}

impl Default for MarginalOptions {
    fn default() -> Self {
        Self {
            n_points: 201,
            hyperpar: true,
            latent_indices: Vec::new(),
            predictor_indices: Vec::new(),
            n_sd: 4.0,
        }
    }
}

fn phi(x: f64, mu: f64, var: f64) -> f64 {
    if !(var > 0.0 && var.is_finite()) {
        return 0.0;
    }
    let s = var.sqrt();
    let z = (x - mu) / s;
    (-0.5 * z * z).exp() / (s * (2.0 * std::f64::consts::PI).sqrt())
}

/// Gaussian mixture marginal on a uniform grid over `[μ ± n_sd·σ]`.
pub fn gaussian_mixture_marginal(
    means: &[f64],
    vars: &[f64],
    weights: &[f64],
    n_points: usize,
    n_sd: f64,
) -> Result<Marginal1D, String> {
    if means.len() != vars.len() || means.len() != weights.len() {
        return Err("mixture means/vars/weights length mismatch".into());
    }
    if means.is_empty() {
        return Err("empty mixture".into());
    }
    if n_points < 5 {
        return Err("n_points must be >= 5".into());
    }

    let mut mu = 0.0;
    for k in 0..means.len() {
        mu += weights[k] * means[k];
    }
    let mut var = 0.0;
    for k in 0..means.len() {
        let d = means[k] - mu;
        var += weights[k] * (vars[k] + d * d);
    }
    if !(var > 0.0 && var.is_finite()) {
        var = 1e-8;
    }
    let sd = var.sqrt();
    let half = n_sd.max(1.0) * sd;
    let lo = mu - half;
    let hi = mu + half;
    let n = n_points;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let xi = lo + (hi - lo) * (i as f64) / ((n - 1) as f64);
        let mut yi = 0.0;
        for k in 0..means.len() {
            yi += weights[k] * phi(xi, means[k], vars[k]);
        }
        x.push(xi);
        y.push(yi);
    }
    let mut m = Marginal1D { x, y };
    m.normalize();
    Ok(m)
}

/// 1D hyperparameter marginal from discrete weighted nodes (internal scale).
///
/// Uses a Gaussian kernel mixture centred at each `theta_k[j]` with bandwidth
/// from the weighted sample SD (Silverman-like floor).
pub fn hyperpar_marginal_from_nodes(
    theta_j: &[f64],
    weights: &[f64],
    n_points: usize,
    n_sd: f64,
) -> Result<Marginal1D, String> {
    if theta_j.len() != weights.len() || theta_j.is_empty() {
        return Err("theta_j/weights mismatch or empty".into());
    }
    let vars: Vec<f64> = {
        let mut mu = 0.0;
        for k in 0..theta_j.len() {
            mu += weights[k] * theta_j[k];
        }
        let mut v = 0.0;
        for k in 0..theta_j.len() {
            let d = theta_j[k] - mu;
            v += weights[k] * d * d;
        }
        // Bandwidth: avoid zero when all mass is at one CCD node.
        let h2 = (v.max(1e-8)) * 1.0;
        vec![h2; theta_j.len()]
    };
    gaussian_mixture_marginal(theta_j, &vars, weights, n_points, n_sd)
}

/// Build internal hyperparameter marginals for each θ dimension.
pub fn hyperpar_marginals(
    theta_nodes: &[Vec<f64>],
    weights: &[f64],
    n_points: usize,
    n_sd: f64,
) -> Result<Vec<Marginal1D>, String> {
    if theta_nodes.is_empty() {
        return Ok(Vec::new());
    }
    let m = theta_nodes[0].len();
    let mut out = Vec::with_capacity(m);
    for j in 0..m {
        let col: Vec<f64> = theta_nodes.iter().map(|t| t[j]).collect();
        out.push(hyperpar_marginal_from_nodes(&col, weights, n_points, n_sd)?);
    }
    Ok(out)
}

/// CDF via cumulative trapezoid; returns `(x, F)` with F(x_0)=0, F(x_end)≈1.
pub fn marginal_cdf(m: &Marginal1D) -> Result<(Vec<f64>, Vec<f64>), String> {
    if m.x.len() < 2 || m.x.len() != m.y.len() {
        return Err("marginal too short or x/y length mismatch".into());
    }
    let mut f = vec![0.0; m.x.len()];
    for i in 0..m.x.len() - 1 {
        let dx = m.x[i + 1] - m.x[i];
        f[i + 1] = f[i] + 0.5 * (m.y[i] + m.y[i + 1]) * dx;
    }
    let mass = f[f.len() - 1];
    if mass > 0.0 {
        for v in &mut f {
            *v /= mass;
        }
    }
    Ok((m.x.clone(), f))
}

/// Invert the CDF for probabilities in (0, 1).
pub fn marginal_quantiles(m: &Marginal1D, probs: &[f64]) -> Result<Vec<f64>, String> {
    let (x, f) = marginal_cdf(m)?;
    let mut out = Vec::with_capacity(probs.len());
    for &p in probs {
        if !(p > 0.0 && p < 1.0) {
            return Err(format!("probability {p} must be in (0, 1)"));
        }
        // Find first i with F[i] >= p
        let mut q = x[x.len() - 1];
        for i in 1..f.len() {
            if f[i] >= p {
                let t = if (f[i] - f[i - 1]).abs() < 1e-15 {
                    0.0
                } else {
                    (p - f[i - 1]) / (f[i] - f[i - 1])
                };
                q = x[i - 1] + t * (x[i] - x[i - 1]);
                break;
            }
        }
        out.push(q);
    }
    Ok(out)
}

/// Standard summary quantiles: 2.5%, 50%, 97.5%.
pub fn marginal_summary_quantiles(m: &Marginal1D) -> Result<(f64, f64, f64), String> {
    let q = marginal_quantiles(m, &[0.025, 0.5, 0.975])?;
    Ok((q[0], q[1], q[2]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_normal_quantiles() {
        let n = 401;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let xi = -5.0 + 10.0 * (i as f64) / ((n - 1) as f64);
            let yi = (-0.5 * xi * xi).exp() / (2.0 * std::f64::consts::PI).sqrt();
            x.push(xi);
            y.push(yi);
        }
        let mut m = Marginal1D { x, y };
        m.normalize();
        assert!((m.integrate() - 1.0).abs() < 1e-3);
        let (q025, q50, q975) = marginal_summary_quantiles(&m).unwrap();
        assert!((q50).abs() < 0.05);
        assert!((q025 + 1.96).abs() < 0.08);
        assert!((q975 - 1.96).abs() < 0.08);
    }

    #[test]
    fn mixture_matches_closed_form_moments() {
        let means = [0.0, 2.0];
        let vars = [1.0, 1.0];
        let weights = [0.5, 0.5];
        let m = gaussian_mixture_marginal(&means, &vars, &weights, 201, 5.0).unwrap();
        assert!((m.integrate() - 1.0).abs() < 1e-3);
        // Mean ≈ 1
        let mut ex = 0.0;
        for i in 0..m.x.len() - 1 {
            let dx = m.x[i + 1] - m.x[i];
            ex += 0.5 * (m.x[i] * m.y[i] + m.x[i + 1] * m.y[i + 1]) * dx;
        }
        assert!((ex - 1.0).abs() < 0.05);
    }
}
