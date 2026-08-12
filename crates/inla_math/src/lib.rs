//! Sparse engine: CSC matrices, LDLT, integration designs, and generic hyper-optimization.

pub mod backend;
pub mod constraints;
pub mod design;
pub mod error;
pub mod hyper_opt;
pub mod integration;
pub mod ldlt;
pub mod scratch;
pub mod solver;
pub mod sparse;

#[cfg(feature = "sparse-ldlt")]
pub mod dense_faer;
#[cfg(feature = "sparse-ldlt")]
pub mod sparse_ldlt;

pub use backend::{
    DefaultBackend, DenseBackend, LdltBackend, diagonal_inverse, factorize_csc, factorize_dense,
    solve_in_place,
};
pub use constraints::{
    ConstraintMethod, ConstraintSpec, HARD_CONSTRAINT_KAPPA, augment_precision_csc,
    model_rank_deficiency, plane_constraint_2d, project_constraints, seasonal_constraint,
    sum_to_zero_constraint,
};
pub use design::{
    add_csc, at_diag_a, block_diag_csc, csc_from_triplets_0based, identity_csc, matvec_csc,
    matvec_transpose_csc, predictor_variances_diag, scale_csc, scale_model_csc,
};
pub use error::MathError;
pub use hyper_opt::{
    compute_hessian, compute_hessian_cancellable, nelder_mead, nelder_mead_cancellable,
};
pub use integration::{ccd_design, grid_design, invert_symmetric_matrix, jacobi_eigen};
pub use ldlt::{
    DenseLdltFactor, Eval1D, LdltFactor, csc_to_dense, laplace_newton_step, laplace_newton_step_a,
    laplace_newton_step_a_solver, laplace_newton_system_a, ldlt_diagonal_inverse, ldlt_factorize,
    ldlt_factorize_dense, ldlt_solve, ldlt_solve_in_place,
};
pub use scratch::{LdltScratch, with_thread_scratch};
pub use solver::{FaerCpuSolver, InlaSolver, SolverError};
pub use sparse::{
    CscForR, CscMatrix, csc_for_r_dgcmatrix, kronecker_csc, sparse_from_triplets, triplets_to_csc,
};

#[cfg(feature = "sparse-ldlt")]
pub use dense_faer::selfadjoint_eigen;
#[cfg(feature = "sparse-ldlt")]
pub use sparse_ldlt::{
    SparseLdltFactor, factorize_sparse, refactorize_numeric_arc, symbolic_pattern,
};

/// Invert an SPD matrix via Cholesky (`A = L Lᵀ ⇒ A⁻¹ = L⁻ᵀ L⁻¹`).
///
/// Uses faer SIMD kernels when the `sparse-ldlt` feature is enabled.
pub fn invert_spd_cholesky(a: &[f64], n: usize) -> Result<Vec<f64>, MathError> {
    #[cfg(feature = "sparse-ldlt")]
    {
        dense_faer::invert_spd_cholesky(a, n)
    }
    #[cfg(not(feature = "sparse-ldlt"))]
    {
        invert_spd_cholesky_scalar(a, n)
    }
}

#[cfg(not(feature = "sparse-ldlt"))]
fn invert_spd_cholesky_scalar(a: &[f64], n: usize) -> Result<Vec<f64>, MathError> {
    if a.len() != n * n {
        return Err(MathError::DimensionMismatch {
            context: "SPD invert matrix length",
            expected: n * n,
            got: a.len(),
        });
    }
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 1e-15 {
                    return Err(MathError::NotPositiveDefinite);
                }
                l[i * n + j] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    let mut y = vec![0.0; n * n];
    for i in 0..n {
        y[i * n + i] = 1.0 / l[i * n + i];
        for j in 0..i {
            let mut s = 0.0;
            for k in j..i {
                s -= l[i * n + k] * y[k * n + j];
            }
            y[i * n + j] = s / l[i * n + i];
        }
    }
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = 0.0;
            for k in i..n {
                s += y[k * n + i] * y[k * n + j];
            }
            inv[i * n + j] = s;
            inv[j * n + i] = s;
        }
    }
    Ok(inv)
}
