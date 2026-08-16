//! Post-inference linear combinations and joint latent sampling.
//!
//! Both need the posterior precision \(Q\) at \(\hat\theta\) (or a single Laplace
//! node). Means use the mixture `latent_means` already on [`InferenceResult`].
//! Variances / draws use \(Q^{-1}\) at the stored factor — the Gaussian-at-mode
//! approximation, not a full CCD mixture of factors.

use inla_math::{CscMatrix, FaerCpuSolver, InlaSolver, csc_to_dense, ldlt_factorize_dense};

use crate::inference::InferenceResult;
use crate::marginals::Marginal1D;

/// Tightness of the soft copy constraint \(x_{\mathrm{copy}} \approx \beta x_{\mathrm{src}}\).
pub const COPY_PRECISION: f64 = 1e6;

/// One linear combination \(v = a^\top x\).
#[derive(Debug, Clone, PartialEq)]
pub struct LinComb {
    pub name: String,
    /// `(latent_index, weight)` pairs. Indices are 0-based in the stacked field.
    pub weights: Vec<(usize, f64)>,
}

/// Gaussian summary of one linear combination.
#[derive(Debug, Clone, PartialEq)]
pub struct LinCombSummary {
    pub name: String,
    pub mean: f64,
    pub sd: f64,
    pub q025: f64,
    pub q50: f64,
    pub q975: f64,
}

impl LinComb {
    fn dense_vector(&self, n: usize) -> Result<Vec<f64>, String> {
        let mut a = vec![0.0; n];
        for &(i, w) in &self.weights {
            if i >= n {
                return Err(format!(
                    "lincomb '{}': index {i} out of range [0, {n})",
                    self.name
                ));
            }
            if !w.is_finite() {
                return Err(format!("lincomb '{}': non-finite weight", self.name));
            }
            a[i] += w;
        }
        Ok(a)
    }
}

/// \(E[a^\top x]\) from the mixture mean and \(\mathrm{Var}(a^\top x)=a^\top Q^{-1}a\).
pub fn lincomb_summaries(
    mean: &[f64],
    q_post: &CscMatrix,
    combs: &[LinComb],
) -> Result<Vec<LinCombSummary>, String> {
    let n = mean.len();
    if q_post.rows() != n || q_post.cols() != n {
        return Err(format!(
            "lincomb: Q is {}×{}, latent mean has length {n}",
            q_post.rows(),
            q_post.cols()
        ));
    }
    let mut solver = FaerCpuSolver::new();
    solver.factorize(q_post).map_err(|e| e.to_string())?;
    let mut out = Vec::with_capacity(combs.len());
    for comb in combs {
        let a = comb.dense_vector(n)?;
        let mean_v: f64 = a.iter().zip(mean).map(|(ai, mi)| ai * mi).sum();
        let w = solver.solve(&a).map_err(|e| e.to_string())?;
        let var: f64 = a.iter().zip(&w).map(|(ai, wi)| ai * wi).sum();
        let sd = var.max(0.0).sqrt();
        out.push(LinCombSummary {
            name: comb.name.clone(),
            mean: mean_v,
            sd,
            q025: mean_v - 1.96 * sd,
            q50: mean_v,
            q975: mean_v + 1.96 * sd,
        });
    }
    Ok(out)
}

/// Draw `n_samples` latent fields from \(\mathcal{N}(\mu, Q^{-1})\) (row-major, length `n_samples * n`).
pub fn sample_latent_gaussian(
    mean: &[f64],
    q_post: &CscMatrix,
    n_samples: usize,
    seed: u64,
) -> Result<Vec<f64>, String> {
    let n = mean.len();
    if q_post.rows() != n || q_post.cols() != n {
        return Err(format!(
            "sample: Q is {}×{}, mean has length {n}",
            q_post.rows(),
            q_post.cols()
        ));
    }
    if n_samples == 0 {
        return Ok(Vec::new());
    }
    let dense = csc_to_dense(q_post).map_err(|e| e.to_string())?;
    let factor = ldlt_factorize_dense(&dense, n).map_err(|e| e.to_string())?;
    let l = match &factor {
        inla_math::LdltFactor::Dense(f) => f,
        _ => return Err("sample: expected dense LDLᵀ factor".into()),
    };
    let mut rng = SplitMix64::new(seed);
    let mut out = vec![0.0; n_samples * n];
    for s in 0..n_samples {
        let mut z = vec![0.0; n];
        for zi in &mut z {
            *zi = rng.standard_normal();
        }
        // y = D^{-1/2} z, then solve Lᵀ x = y (L unit lower).
        let mut y = vec![0.0; n];
        for i in 0..n {
            let d = l.d[i];
            if d <= 0.0 || !d.is_finite() {
                return Err(format!("sample: non-positive D[{i}]={d}"));
            }
            y[i] = z[i] / d.sqrt();
        }
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut acc = y[i];
            for k in (i + 1)..n {
                acc -= l.l_row_major[k * n + i] * x[k];
            }
            x[i] = acc;
        }
        let dest = &mut out[s * n..(s + 1) * n];
        for i in 0..n {
            dest[i] = mean[i] + x[i];
        }
    }
    Ok(out)
}

/// \(\mathbb{E}[g(X)] = \int g(x)\,\pi(x)\,dx\) given \(g\) evaluated on the marginal grid.
pub fn emarginal(m: &Marginal1D, g_of_x: &[f64]) -> Result<f64, String> {
    if m.x.len() < 2 || m.x.len() != m.y.len() {
        return Err("emarginal: marginal too short or x/y mismatch".into());
    }
    if g_of_x.len() != m.x.len() {
        return Err(format!(
            "emarginal: g(x) length {} != marginal length {}",
            g_of_x.len(),
            m.x.len()
        ));
    }
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..m.x.len() - 1 {
        let dx = m.x[i + 1] - m.x[i];
        if dx <= 0.0 {
            continue;
        }
        num += 0.5 * (g_of_x[i] * m.y[i] + g_of_x[i + 1] * m.y[i + 1]) * dx;
        den += 0.5 * (m.y[i] + m.y[i + 1]) * dx;
    }
    if den <= 0.0 {
        return Err("emarginal: marginal has non-positive mass".into());
    }
    Ok(num / den)
}

/// Convenience: lincombs + samples from a fitted [`InferenceResult`].
impl InferenceResult {
    pub fn lincomb(&self, combs: &[LinComb]) -> Result<Vec<LinCombSummary>, String> {
        let q = self
            .posterior_precision
            .as_ref()
            .ok_or("lincomb: posterior precision was not stored")?;
        lincomb_summaries(&self.latent_means, q, combs)
    }

    pub fn posterior_sample(&self, n_samples: usize, seed: u64) -> Result<Vec<f64>, String> {
        let q = self
            .posterior_precision
            .as_ref()
            .ok_or("posterior_sample: posterior precision was not stored")?;
        sample_latent_gaussian(&self.latent_means, q, n_samples, seed)
    }
}

/// Small SplitMix64 + Box–Muller so we do not take a `rand` dependency.
struct SplitMix64 {
    state: u64,
    spare: Option<f64>,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed | 1,
            spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        let u = self.next_u64() >> 11;
        (u as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    fn standard_normal(&mut self) -> f64 {
        if let Some(s) = self.spare.take() {
            return s;
        }
        let mut u1 = self.next_f64();
        let u2 = self.next_f64();
        if u1 < 1e-12 {
            u1 = 1e-12;
        }
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        self.spare = Some(r * theta.sin());
        r * theta.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inla_math::identity_csc;

    #[test]
    fn lincomb_identity_precision() {
        let q = identity_csc(3, 4.0).unwrap();
        let mean = vec![1.0, 2.0, 3.0];
        let comb = LinComb {
            name: "sum".into(),
            weights: vec![(0, 1.0), (1, 1.0), (2, 1.0)],
        };
        let s = lincomb_summaries(&mean, &q, &[comb]).unwrap();
        assert!((s[0].mean - 6.0).abs() < 1e-12);
        // Var(1+1+1) = 3/4
        assert!((s[0].sd - (0.75_f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn sample_moments_match_identity() {
        let q = identity_csc(4, 1.0).unwrap();
        let mean = vec![0.0; 4];
        let draws = sample_latent_gaussian(&mean, &q, 4000, 42).unwrap();
        let n = 4usize;
        let ns = 4000usize;
        let mut emp_mean = vec![0.0; n];
        let mut emp_var = vec![0.0; n];
        for s in 0..ns {
            for i in 0..n {
                emp_mean[i] += draws[s * n + i];
            }
        }
        for i in 0..n {
            emp_mean[i] /= ns as f64;
        }
        for s in 0..ns {
            for i in 0..n {
                let d = draws[s * n + i] - emp_mean[i];
                emp_var[i] += d * d;
            }
        }
        for i in 0..n {
            emp_var[i] /= (ns - 1) as f64;
            assert!(emp_mean[i].abs() < 0.08, "mean[{i}]={}", emp_mean[i]);
            assert!((emp_var[i] - 1.0).abs() < 0.12, "var[{i}]={}", emp_var[i]);
        }
    }

    #[test]
    fn emarginal_identity_on_unit_gaussian() {
        let x: Vec<f64> = (-40..=40).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&t| (-0.5 * t * t).exp() / (2.0 * std::f64::consts::PI).sqrt())
            .collect();
        let m = Marginal1D { x: x.clone(), y };
        let g: Vec<f64> = x.iter().map(|&t| t * t).collect();
        let e = emarginal(&m, &g).unwrap();
        assert!((e - 1.0).abs() < 0.02, "E[X^2]={e}");
    }
}
