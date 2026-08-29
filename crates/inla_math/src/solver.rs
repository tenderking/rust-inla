use thiserror::Error;

use crate::backend::LdltBackend;
use crate::error::MathError;
use crate::ldlt::LdltFactor;
use crate::scratch::LdltScratch;
use crate::sparse::CscMatrix;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum SolverError {
    #[error("Matrix is singular or structurally rank-deficient")]
    Singular,
    #[error("Matrix is not positive-definite during factorization")]
    NotPositiveDefinite,
    #[error("Linear system solve failed: {0}")]
    SolveFailure(String),
}

impl From<MathError> for SolverError {
    fn from(err: MathError) -> Self {
        match err {
            MathError::Singular => SolverError::Singular,
            MathError::NotPositiveDefinite => SolverError::NotPositiveDefinite,
            other => SolverError::SolveFailure(other.to_string()),
        }
    }
}

pub trait InlaSolver: Send + Sync {
    /// Performs symbolic and numeric factorization on the sparse precision matrix Q.
    fn factorize(&mut self, q: &CscMatrix) -> Result<(), SolverError>;

    /// Solves Q * x = rhs. Must reuse the cached factorization if factorize() was already called.
    fn solve(&mut self, rhs: &[f64]) -> Result<Vec<f64>, SolverError>;

    /// Diagonal of the selected inverse (marginal variances) of Q.
    ///
    /// Sparse factors use Takahashi on the filled pattern of `L`. Dense factors
    /// (exact FGN) keep the dense LDLᵀ diagonal inverse.
    fn diag_inv(&mut self) -> Result<Vec<f64>, SolverError>;

    /// Log absolute determinant of Q based on current factorization.
    fn log_abs_det(&self) -> Result<f64, SolverError>;
}

/// CPU reference implementation using `faer` (via `DefaultBackend`).
#[derive(Debug, Default)]
pub struct FaerCpuSolver {
    factor: Option<LdltFactor>,
    scratch: LdltScratch,
}

impl FaerCpuSolver {
    pub fn new() -> Self {
        Self::default()
    }

    /// Borrow the cached factorization (if any).
    pub fn factor(&self) -> Option<&LdltFactor> {
        self.factor.as_ref()
    }

    /// Take ownership of the cached factorization, leaving the solver empty.
    pub fn into_factor(mut self) -> Option<LdltFactor> {
        self.factor.take()
    }
}

impl InlaSolver for FaerCpuSolver {
    fn factorize(&mut self, q: &CscMatrix) -> Result<(), SolverError> {
        let factor = crate::backend::DefaultBackend.factorize(q, &mut self.scratch)?;
        self.factor = Some(factor);
        Ok(())
    }

    fn solve(&mut self, rhs: &[f64]) -> Result<Vec<f64>, SolverError> {
        let factor = self.factor.as_ref().ok_or_else(|| {
            SolverError::SolveFailure("Factorization not computed yet".to_string())
        })?;
        let mut x = rhs.to_vec();
        crate::backend::DefaultBackend.solve_in_place(factor, &mut x, &mut self.scratch)?;
        Ok(x)
    }

    fn diag_inv(&mut self) -> Result<Vec<f64>, SolverError> {
        let factor = self.factor.as_ref().ok_or_else(|| {
            SolverError::SolveFailure("Factorization not computed yet".to_string())
        })?;
        let diag = crate::backend::DefaultBackend.diagonal_inverse(factor, &mut self.scratch)?;
        Ok(diag)
    }

    fn log_abs_det(&self) -> Result<f64, SolverError> {
        let factor = self.factor.as_ref().ok_or_else(|| {
            SolverError::SolveFailure("Factorization not computed yet".to_string())
        })?;
        Ok(factor.log_abs_det())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::design::identity_csc;

    #[test]
    fn test_faer_cpu_solver() {
        let q = identity_csc(3, 2.0).unwrap();
        let mut solver = FaerCpuSolver::new();
        solver.factorize(&q).unwrap();

        let rhs = vec![2.0, 4.0, 6.0];
        let sol = solver.solve(&rhs).unwrap();
        assert_eq!(sol, vec![1.0, 2.0, 3.0]);

        let diag = solver.diag_inv().unwrap();
        assert_eq!(diag, vec![0.5, 0.5, 0.5]);

        let log_det = solver.log_abs_det().unwrap();
        assert!((log_det - (2.0_f64.ln() * 3.0)).abs() < 1e-10);
    }
}
