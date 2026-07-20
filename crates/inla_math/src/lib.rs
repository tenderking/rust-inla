//! Sparse engine: CSC matrices, LDLT, integration designs, and generic hyper-optimization.

pub mod backend;
pub mod constraints;
pub mod design;
pub mod error;
pub mod hyper_opt;
pub mod integration;
pub mod ldlt;
pub mod scratch;
pub mod sparse;

#[cfg(feature = "sparse-ldlt")]
pub mod sparse_ldlt;

pub use backend::{
    DefaultBackend, DenseBackend, LdltBackend, diagonal_inverse, factorize_csc, factorize_dense,
    solve_in_place,
};
pub use constraints::{
    ConstraintSpec, HARD_CONSTRAINT_KAPPA, augment_precision_csc, model_rank_deficiency,
    project_constraints, sum_to_zero_constraint,
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
    ldlt_diagonal_inverse, ldlt_factorize, ldlt_factorize_dense, ldlt_solve, ldlt_solve_in_place,
};
pub use scratch::{LdltScratch, with_thread_scratch};
pub use sparse::{CscForR, CscMatrix, csc_for_r_dgcmatrix, sparse_from_triplets, triplets_to_csc};

#[cfg(feature = "sparse-ldlt")]
pub use sparse_ldlt::{
    SparseLdltFactor, factorize_sparse, refactorize_numeric_arc, symbolic_pattern,
};
