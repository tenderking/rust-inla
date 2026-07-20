//! Typed errors for sparse linear algebra and design helpers.

use thiserror::Error;

/// Errors from factorization, solves, and matrix assembly in `inla_math`.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum MathError {
    #[error("matrix must be square (got {rows}×{cols})")]
    NotSquare { rows: usize, cols: usize },

    #[error("dimension mismatch: {context} (expected {expected}, got {got})")]
    DimensionMismatch {
        context: &'static str,
        expected: usize,
        got: usize,
    },

    #[error("matrix is singular or numerically unstable in LDLᵀ")]
    Singular,

    #[error("matrix is not positive definite")]
    NotPositiveDefinite,

    #[error("LDLᵀ requires a symmetric matrix")]
    NotSymmetric,

    #[error("empty or invalid matrix: {0}")]
    InvalidMatrix(&'static str),

    #[error("index out of bounds: {0}")]
    IndexOutOfBounds(&'static str),

    #[error("out of memory during sparse factorization")]
    OutOfMemory,

    #[error("{0}")]
    Message(String),
}

impl MathError {
    /// True for factorization / solve failures that SciPy maps to `LinAlgError`.
    pub fn is_linalg(&self) -> bool {
        matches!(
            self,
            Self::Singular | Self::NotPositiveDefinite | Self::NotSymmetric
        )
    }
}

impl From<MathError> for String {
    fn from(value: MathError) -> Self {
        value.to_string()
    }
}
