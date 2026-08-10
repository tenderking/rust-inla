//! Mesh I/O and SPDE / FEM exports for R.

use crate::convert::csc_to_dgcmatrix;
use extendr_api::prelude::*;
use inla_core::mesh::read_mesh_summary;

#[extendr]
fn inla_rs_read_mesh(path: &str) -> std::result::Result<List, Error> {
    let summary = read_mesh_summary(path).map_err(Error::Other)?;
    Ok(list!(
        n_vertices = summary.n_vertices as i32,
        xmin = summary.xmin,
        xmax = summary.xmax,
        ymin = summary.ymin,
        ymax = summary.ymax
    ))
}

fn parse_mesh2d_from_r(
    vertices_mat: &Robj,
    triangles_mat: &Robj,
) -> std::result::Result<inla_core::fmesher::Mesh2D, Error> {
    if !vertices_mat.is_matrix() || !triangles_mat.is_matrix() {
        return Err(Error::Other(
            "vertices and triangles must be matrices".to_string(),
        ));
    }
    let dim_v = vertices_mat
        .dim()
        .ok_or_else(|| Error::Other("could not get vertices dims".to_string()))?;
    let dim_t = triangles_mat
        .dim()
        .ok_or_else(|| Error::Other("could not get triangles dims".to_string()))?;
    if dim_v.len() != 2 || dim_v[1] != 2 {
        return Err(Error::Other("vertices must be an N x 2 matrix".to_string()));
    }
    if dim_t.len() != 2 || dim_t[1] != 3 {
        return Err(Error::Other(
            "triangles must be an M x 3 matrix".to_string(),
        ));
    }
    let n_vertices = dim_v[0].0 as usize;
    let n_triangles = dim_t[0].0 as usize;
    let vertices_data = vertices_mat
        .as_real_vector()
        .ok_or_else(|| Error::Other("vertices must be real".to_string()))?;
    let triangles_data = triangles_mat
        .as_integer_vector()
        .ok_or_else(|| Error::Other("triangles must be integers".to_string()))?;
    let mut vertices = Vec::with_capacity(n_vertices);
    for r in 0..n_vertices {
        let x = vertices_data[r];
        let y = vertices_data[r + n_vertices];
        vertices.push(inla_core::fmesher::Vertex2 { x, y });
    }
    let mut triangles = Vec::with_capacity(n_triangles);
    for r in 0..n_triangles {
        let i0 = (triangles_data[r] - 1) as usize;
        let i1 = (triangles_data[r + n_triangles] - 1) as usize;
        let i2 = (triangles_data[r + 2 * n_triangles] - 1) as usize;
        triangles.push(inla_core::fmesher::Triangle([i0, i1, i2]));
    }
    inla_core::fmesher::build_mesh2d(vertices, triangles).map_err(Error::Other)
}

#[extendr]
fn inla_rs_spde_precision_mesh_csc(
    vertices_mat: Robj,
    triangles_mat: Robj,
    kappa: f64,
    tau: f64,
) -> std::result::Result<Robj, Error> {
    let mesh = parse_mesh2d_from_r(&vertices_mat, &triangles_mat)?;
    let fem = mesh.assemble_fem_blocks();
    let csc = inla_core::spde::spde_precision_csc(&fem, kappa, tau).map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

/// Export FEM mass (`c0` / C) and stiffness (`g1` / G) as `dgCMatrix`.
///
/// Analogous to classic INLA `spde$param.inla$M0` / `M1` (lumped-mass style `c0`
/// is the mass matrix assembled by our fmesher).
#[extendr]
fn inla_rs_fem_blocks_mesh(
    vertices_mat: Robj,
    triangles_mat: Robj,
) -> std::result::Result<List, Error> {
    let mesh = parse_mesh2d_from_r(&vertices_mat, &triangles_mat)?;
    let fem = mesh.assemble_fem_blocks();
    let c0 = inla_core::sparse_from_triplets(fem.c0.rows, fem.c0.cols, &fem.c0.entries);
    let g1 = inla_core::sparse_from_triplets(fem.g1.rows, fem.g1.cols, &fem.g1.entries);
    let c0_m = csc_to_dgcmatrix(&c0)?;
    let g1_m = csc_to_dgcmatrix(&g1)?;
    Ok(list!(
        c0 = c0_m,
        g1 = g1_m,
        n_vertices = mesh.vertices.len() as i32,
        n_triangles = mesh.triangles.len() as i32
    ))
}

/// Piecewise-linear FEM projector `A` (`length(loc_x) × n_vertices`).
#[extendr]
fn inla_rs_spde_projector_csc(
    vertices_mat: Robj,
    triangles_mat: Robj,
    loc_x: Vec<f64>,
    loc_y: Vec<f64>,
) -> std::result::Result<Robj, Error> {
    let mesh = parse_mesh2d_from_r(&vertices_mat, &triangles_mat)?;
    let a = inla_core::spde_projector_from_xy(&mesh, &loc_x, &loc_y).map_err(Error::Other)?;
    csc_to_dgcmatrix(&a)
}

/// End-to-end SPDE Gaussian INLA: mesh + locations → A, θ=`[log_tau, log_kappa]`.
#[extendr]
fn inla_rs_run_spde(
    initial_theta: Vec<f64>,
    y_obs: Vec<f64>,
    obs_precision: f64,
    strategy: &str,
    step_or_f0: f64,
    vertices_mat: Robj,
    triangles_mat: Robj,
    loc_x: Vec<f64>,
    loc_y: Vec<f64>,
    constrain: bool,
    deterministic: bool,
) -> std::result::Result<List, Error> {
    if initial_theta.len() != 2 {
        return Err(Error::Other(
            "SPDE initial_theta must be length 2: [log_tau, log_kappa]".into(),
        ));
    }
    if y_obs.len() != loc_x.len() || loc_x.len() != loc_y.len() {
        return Err(Error::Other(
            "y, loc_x, loc_y must have the same length".into(),
        ));
    }
    let mesh = parse_mesh2d_from_r(&vertices_mat, &triangles_mat)?;
    let fem = mesh.assemble_fem_blocks();
    let a = inla_core::spde_projector_from_xy(&mesh, &loc_x, &loc_y).map_err(Error::Other)?;
    let n_latent = mesh.vertices.len();
    let obs: Vec<inla_core::Obs> = y_obs
        .iter()
        .map(|&y| {
            inla_core::Obs::Gaussian(inla_core::GaussianObs {
                y,
                precision: obs_precision,
                link: inla_core::Link::Identity,
            })
        })
        .collect();
    let constr = if constrain {
        Some(inla_core::sum_to_zero_constraint(n_latent, 1).map_err(Error::Other)?)
    } else {
        None
    };
    let build_prior = move |theta: &[f64]| {
        let (tau, kappa) = inla_core::spde_params_from_theta(theta)?;
        inla_core::spde_precision_csc(&fem, kappa, tau)
    };
    let log_prior = |theta: &[f64]| -> f64 {
        match inla_core::HyperPriorStack::default_for_effect("spde") {
            Ok(stack) => stack.log_density(theta).unwrap_or(f64::NEG_INFINITY),
            Err(_) => theta.iter().map(|&v| -0.5 * 0.1 * v * v).sum(),
        }
    };
    let result = inla_core::run_inla_inference_a(
        &initial_theta,
        &build_prior,
        &log_prior,
        &obs,
        Some(&a),
        constr.as_ref(),
        strategy,
        step_or_f0,
        &inla_core::MarginalOptions::default(),
        deterministic,
    )
    .map_err(Error::Other)?;
    Ok(list!(
        mode = result.mode,
        hessian = result.hessian,
        latent_means = result.latent_means,
        latent_variances = result.latent_variances,
        marginal_log_lik = result.marginal_log_lik,
        n_latent = n_latent as i32,
        n_obs = y_obs.len() as i32
    ))
}

extendr_module! {
    mod mesh;
    fn inla_rs_read_mesh;
    fn inla_rs_spde_precision_mesh_csc;
    fn inla_rs_fem_blocks_mesh;
    fn inla_rs_spde_projector_csc;
    fn inla_rs_run_spde;
}
