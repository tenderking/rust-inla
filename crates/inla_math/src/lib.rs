//! Sparse engine: CSC matrices, LDLT, integration designs, and generic hyper-optimization.

pub mod design;
pub mod hyper_opt;
pub mod integration;
pub mod ldlt;
pub mod sparse;

pub use design::{
    add_csc, at_diag_a, block_diag_csc, csc_from_triplets_0based, identity_csc, matvec_csc,
    matvec_transpose_csc, predictor_variances_diag, scale_csc, scale_model_csc,
};
pub use hyper_opt::{compute_hessian, nelder_mead};
pub use integration::{ccd_design, grid_design, invert_symmetric_matrix, jacobi_eigen};
pub use ldlt::{
    Eval1D, LdltFactor, csc_to_dense, laplace_newton_step, laplace_newton_step_a,
    ldlt_diagonal_inverse, ldlt_factorize, ldlt_factorize_dense, ldlt_solve, ldlt_solve_in_place,
};
pub use sparse::{CscForR, CscMatrix, csc_for_r_dgcmatrix, sparse_from_triplets, triplets_to_csc};
