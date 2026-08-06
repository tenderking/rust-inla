use inla_math::{CscMatrix, matvec_csc, matvec_transpose_csc, predictor_variances_diag};

/// Maps latent field `x` to linear predictor `η` (observation index space).
///
/// Identity mappers do not materialize an `n × n` matrix; sparse mappers expose `A`.
pub trait ProjectionMapper: Send + Sync {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;

    /// Sparse projector `A` when materialized; `None` for pure identity (`η = x`).
    fn projection_matrix(&self) -> Option<&CscMatrix>;

    /// `η = A x` (or `η = x` for identity).
    fn project(&self, latent_vector: &[f64]) -> Result<Vec<f64>, String>;

    /// `z = Aᵀ y` (or `z = y` for identity).
    fn project_transpose(&self, obs_vector: &[f64]) -> Result<Vec<f64>, String>;

    /// Diagonal predictor variances `diag(A Σ Aᵀ)` under diagonal latent covariance.
    fn predictor_variances(&self, var_x: &[f64]) -> Result<Vec<f64>, String>;
}

/// Concrete implementation wrapping a pre-built sparse projection matrix.
pub struct SparseProjectionMapper {
    a: CscMatrix,
}

impl SparseProjectionMapper {
    pub fn new(a: CscMatrix) -> Self {
        Self { a }
    }
}

impl ProjectionMapper for SparseProjectionMapper {
    fn nrows(&self) -> usize {
        self.a.rows()
    }

    fn ncols(&self) -> usize {
        self.a.cols()
    }

    fn projection_matrix(&self) -> Option<&CscMatrix> {
        Some(&self.a)
    }

    fn project(&self, latent_vector: &[f64]) -> Result<Vec<f64>, String> {
        matvec_csc(&self.a, latent_vector)
    }

    fn project_transpose(&self, obs_vector: &[f64]) -> Result<Vec<f64>, String> {
        matvec_transpose_csc(&self.a, obs_vector)
    }

    fn predictor_variances(&self, var_x: &[f64]) -> Result<Vec<f64>, String> {
        predictor_variances_diag(&self.a, var_x)
    }
}

/// Identity projection `η = x` without allocating an identity CSC.
pub struct IdentityProjection {
    n: usize,
}

impl IdentityProjection {
    pub fn new(n: usize) -> Result<Self, String> {
        if n == 0 {
            return Err("IdentityProjection: n must be > 0".into());
        }
        Ok(Self { n })
    }
}

impl ProjectionMapper for IdentityProjection {
    fn nrows(&self) -> usize {
        self.n
    }

    fn ncols(&self) -> usize {
        self.n
    }

    fn projection_matrix(&self) -> Option<&CscMatrix> {
        None
    }

    fn project(&self, latent_vector: &[f64]) -> Result<Vec<f64>, String> {
        if latent_vector.len() != self.n {
            return Err(format!(
                "IdentityProjection: expected length {}, got {}",
                self.n,
                latent_vector.len()
            ));
        }
        Ok(latent_vector.to_vec())
    }

    fn project_transpose(&self, obs_vector: &[f64]) -> Result<Vec<f64>, String> {
        self.project(obs_vector)
    }

    fn predictor_variances(&self, var_x: &[f64]) -> Result<Vec<f64>, String> {
        if var_x.len() != self.n {
            return Err(format!(
                "IdentityProjection: expected length {}, got {}",
                self.n,
                var_x.len()
            ));
        }
        Ok(var_x.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inla_math::identity_csc;

    #[test]
    fn test_sparse_projection_mapper() {
        let eye = identity_csc(3, 1.0).unwrap();
        let mapper = SparseProjectionMapper::new(eye);
        assert_eq!(mapper.nrows(), 3);
        assert!(mapper.projection_matrix().is_some());
        let proj = mapper.project(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(proj, vec![1.0, 2.0, 3.0]);
        let back = mapper.project_transpose(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(back, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_identity_projection_no_matrix() {
        let mapper = IdentityProjection::new(4).unwrap();
        assert!(mapper.projection_matrix().is_none());
        assert_eq!(mapper.project(&[1.0, 2.0, 3.0, 4.0]).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            mapper.predictor_variances(&[0.1, 0.2, 0.3, 0.4]).unwrap(),
            vec![0.1, 0.2, 0.3, 0.4]
        );
    }
}
