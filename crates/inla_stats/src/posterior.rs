//! Post-inference linear combinations.
//!
//! Needs the posterior precision \(Q\) at \(\hat\theta\) (or a single Laplace
//! node). Means use the mixture `latent_means` already on [`InferenceResult`].
//! Variances use \(Q^{-1}\) at the stored factor — the Gaussian-at-mode
//! approximation, not a full CCD mixture of factors.

use inla_math::{CscMatrix, FaerCpuSolver, InlaSolver};

use crate::inference::InferenceResult;

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

impl InferenceResult {
    pub fn lincomb(&self, combs: &[LinComb]) -> Result<Vec<LinCombSummary>, String> {
        let q = self
            .posterior_precision
            .as_ref()
            .ok_or("lincomb: posterior precision was not stored")?;
        lincomb_summaries(&self.latent_means, q, combs)
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
}
