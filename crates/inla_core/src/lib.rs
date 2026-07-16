pub mod ar1;
pub mod arp;
pub mod besag;
pub mod crw;
pub mod fmesher;

pub mod inference;
pub mod latent_models;
pub mod ldlt;
pub mod mesh;
pub mod matern2d;
pub mod rw2d;
pub mod spde;
pub mod sparse;
pub mod hyper_opt;
pub mod integration;
pub mod model_selection;




pub use ar1::{Ar1Precision, ar1_precision};
pub use fmesher::{
    BoundaryInput, EdgeRef, FemBlocks, Mesh2D, PathStep, PathTrace, PointLocation, SparseTriplet,
    Triangle, Vertex2, build_boundary_segments, build_mesh2d, load_fmesher_boundary_input,
    load_fmesher_raw_boundary_input, read_boundary_indices, read_positions_xy,
};
pub use inference::{
    BinomialObs, Eval1D, ExponentialSurvivalObs, GammaPrior, GaussianObs, GaussianPrior,
    LaplaceObs, Link, NegativeBinomialObs, PoissonObs, WeibullSurvivalObs, ZeroInflatedBinomialObs,
    ZeroInflatedPoissonObs, ZeroInflationType, eval_likelihood_binomial,
    eval_likelihood_exponential_survival, eval_likelihood_gaussian, eval_likelihood_laplace,
    eval_likelihood_negative_binomial, eval_likelihood_poisson,
    eval_likelihood_weibull_survival, eval_likelihood_zero_inflated_binomial,
    eval_likelihood_zero_inflated_poisson, eval_prior_gamma, eval_prior_gaussian,
    eval_prior_loggamma, Obs, find_latent_mode, InferenceResult, run_inla_inference,
};
pub use hyper_opt::{ModelConfig, nelder_mead, compute_hessian, evaluate_neg_log_posterior};
pub use integration::{ccd_design, grid_design, invert_symmetric_matrix, jacobi_eigen};
pub use model_selection::{CpoResult, DicResult, compute_marginal_log_lik_gaussian, compute_dic, compute_cpo_pit};
pub use ldlt::{
    LdltFactor, csc_to_dense, laplace_newton_step, ldlt_diagonal_inverse, ldlt_factorize,
    ldlt_factorize_dense, ldlt_solve, ldlt_solve_in_place,
};
pub use latent_models::{
    rw1_precision_csc, rw2_precision_csc, rw1_cyclic_precision_csc, rw2_cyclic_precision_csc,
    seasonal_precision_csc, two_diid_precision_csc, iid_precision_csc, fgn_precision_csc,
};
pub use arp::arp_precision_csc;
pub use besag::{besag_precision_csc, bym_precision_csc, read_graph_file};
pub use crw::{crw1_precision_csc, crw2_precision_csc};
pub use matern2d::matern2d_precision_csc;
pub use rw2d::rw2d_precision_csc;
pub use spde::spde_precision_csc;
pub use mesh::{MeshSummary, read_mesh_summary};
pub use sparse::{CscForR, CscMatrix, ar1_precision_csc, csc_for_r_dgcmatrix, triplets_to_csc, sparse_triplet_to_csc};

#[cfg(test)]
mod tests {
    use super::{ar1_precision, read_mesh_summary};
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_file(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards")
            .as_nanos();
        env::temp_dir().join(format!("{}_{}.txt", name, nanos))
    }

    #[test]
    fn builds_ar1_precision_matrix() {
        let q = ar1_precision(4, 0.5, 2.0).expect("ar1 precision should be valid");
        assert_eq!(q.n, 4);
        assert_eq!(q.row_major_values.len(), 16);
        assert_eq!(q.row_major_values[0], 2.0);
        assert_eq!(q.row_major_values[1], -1.0);
        assert_eq!(q.row_major_values[5], 2.5);
        assert_eq!(q.row_major_values[15], 2.0);
    }

    #[test]
    fn reads_simple_mesh_file() {
        let path = unique_temp_file("rinla_core_mesh");
        let mesh_data = "# x y\n0.0 0.0\n1.0 0.0\n1.0 2.0\n";
        fs::write(&path, mesh_data).expect("write mesh file");

        let summary = read_mesh_summary(&path).expect("parse mesh");
        assert_eq!(summary.n_vertices, 3);
        assert_eq!(summary.xmin, 0.0);
        assert_eq!(summary.xmax, 1.0);
        assert_eq!(summary.ymin, 0.0);
        assert_eq!(summary.ymax, 2.0);

        fs::remove_file(path).expect("remove temporary mesh file");
    }
}
