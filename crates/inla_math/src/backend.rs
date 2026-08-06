//! Thin Factorize / Solve / DiagInv backend trait (mdarray-linalg-style).

use crate::error::MathError;
use crate::ldlt::{DenseLdltFactor, LdltFactor};
use crate::scratch::LdltScratch;
use crate::sparse::CscMatrix;

/// Backend for SPD / symmetric indefinite precision systems `Q = L D Lᵀ`.
pub trait LdltBackend {
    /// Factorize a sparse symmetric matrix (CSC, lower+upper or full).
    fn factorize(&self, q: &CscMatrix, scratch: &mut LdltScratch) -> Result<LdltFactor, MathError>;

    /// Factorize a dense row-major symmetric matrix.
    fn factorize_dense(
        &self,
        a: &[f64],
        n: usize,
        scratch: &mut LdltScratch,
    ) -> Result<LdltFactor, MathError>;

    fn solve_in_place(
        &self,
        factor: &LdltFactor,
        x: &mut [f64],
        scratch: &mut LdltScratch,
    ) -> Result<(), MathError>;

    fn diagonal_inverse(
        &self,
        factor: &LdltFactor,
        scratch: &mut LdltScratch,
    ) -> Result<Vec<f64>, MathError>;
}

/// Dense LDLᵀ backend (always available).
#[derive(Debug, Clone, Copy, Default)]
pub struct DenseBackend;

impl LdltBackend for DenseBackend {
    fn factorize(&self, q: &CscMatrix, scratch: &mut LdltScratch) -> Result<LdltFactor, MathError> {
        let n = q.rows();
        if q.cols() != n {
            return Err(MathError::NotSquare {
                rows: q.rows(),
                cols: q.cols(),
            });
        }
        scratch.ensure_dense(n);
        scratch.dense.fill(0.0);
        for (col, colvec) in q.outer_iterator().enumerate() {
            for (row, value) in colvec.iter() {
                scratch.dense[row * n + col] = *value;
            }
        }
        let factor = crate::ldlt::ldlt_factorize_dense_inner(&scratch.dense[..n * n], n)?;
        Ok(LdltFactor::Dense(factor))
    }

    fn factorize_dense(
        &self,
        a: &[f64],
        n: usize,
        _scratch: &mut LdltScratch,
    ) -> Result<LdltFactor, MathError> {
        Ok(LdltFactor::Dense(crate::ldlt::ldlt_factorize_dense_inner(
            a, n,
        )?))
    }

    fn solve_in_place(
        &self,
        factor: &LdltFactor,
        x: &mut [f64],
        _scratch: &mut LdltScratch,
    ) -> Result<(), MathError> {
        match factor {
            LdltFactor::Dense(f) => crate::ldlt::dense_solve_in_place(f, x),
            #[cfg(feature = "sparse-ldlt")]
            LdltFactor::Sparse(f) => crate::sparse_ldlt::sparse_solve_in_place(f, x, _scratch),
            #[cfg(not(feature = "sparse-ldlt"))]
            LdltFactor::Sparse(_) => {
                Err(MathError::Message("sparse LDLᵀ backend not enabled".into()))
            }
        }
    }

    fn diagonal_inverse(
        &self,
        factor: &LdltFactor,
        scratch: &mut LdltScratch,
    ) -> Result<Vec<f64>, MathError> {
        match factor {
            LdltFactor::Dense(f) => crate::ldlt::dense_diagonal_inverse(f),
            #[cfg(feature = "sparse-ldlt")]
            LdltFactor::Sparse(f) => crate::sparse_ldlt::sparse_diagonal_inverse(f, scratch),
            #[cfg(not(feature = "sparse-ldlt"))]
            LdltFactor::Sparse(_) => {
                Err(MathError::Message("sparse LDLᵀ backend not enabled".into()))
            }
        }
    }
}

/// Default backend: sparse faer LDLᵀ when enabled, else dense.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultBackend;

impl LdltBackend for DefaultBackend {
    fn factorize(&self, q: &CscMatrix, scratch: &mut LdltScratch) -> Result<LdltFactor, MathError> {
        #[cfg(feature = "sparse-ldlt")]
        {
            match crate::sparse_ldlt::factorize_sparse(q, scratch) {
                Ok(f) => Ok(LdltFactor::Sparse(f)),
                // Fall back to dense for tiny systems or unexpected faer errors
                // that are not structural PD failures.
                Err(MathError::OutOfMemory) => DenseBackend.factorize(q, scratch),
                Err(e) => Err(e),
            }
        }
        #[cfg(not(feature = "sparse-ldlt"))]
        {
            DenseBackend.factorize(q, scratch)
        }
    }

    fn factorize_dense(
        &self,
        a: &[f64],
        n: usize,
        scratch: &mut LdltScratch,
    ) -> Result<LdltFactor, MathError> {
        DenseBackend.factorize_dense(a, n, scratch)
    }

    fn solve_in_place(
        &self,
        factor: &LdltFactor,
        x: &mut [f64],
        scratch: &mut LdltScratch,
    ) -> Result<(), MathError> {
        DenseBackend.solve_in_place(factor, x, scratch)
    }

    fn diagonal_inverse(
        &self,
        factor: &LdltFactor,
        scratch: &mut LdltScratch,
    ) -> Result<Vec<f64>, MathError> {
        DenseBackend.diagonal_inverse(factor, scratch)
    }
}

/// Factorize `q` with the default backend and the thread-local scratch pool.
pub fn factorize_csc(q: &CscMatrix) -> Result<LdltFactor, MathError> {
    crate::scratch::with_thread_scratch(|scratch| DefaultBackend.factorize(q, scratch))
}

/// Solve `factor x = b` in place using the default backend.
pub fn solve_in_place(factor: &LdltFactor, x: &mut [f64]) -> Result<(), MathError> {
    crate::scratch::with_thread_scratch(|scratch| DefaultBackend.solve_in_place(factor, x, scratch))
}

/// Diagonal of `Q⁻¹` given an LDLᵀ factor.
pub fn diagonal_inverse(factor: &LdltFactor) -> Result<Vec<f64>, MathError> {
    crate::scratch::with_thread_scratch(|scratch| DefaultBackend.diagonal_inverse(factor, scratch))
}

/// Convenience: dense factorize without an external scratch.
pub fn factorize_dense(a: &[f64], n: usize) -> Result<DenseLdltFactor, MathError> {
    crate::ldlt::ldlt_factorize_dense_inner(a, n)
}
