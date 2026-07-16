use inla_core::ar1::ar1_precision;
use inla_core::mesh::read_mesh_summary;
use inla_core::sparse::{ar1_precision_csc, csc_for_r_dgcmatrix};
use extendr_api::prelude::*;

fn map_r_error(e: Error) -> Error {
    Error::Other(e.to_string())
}

#[extendr]
fn rinla_read_mesh(path: &str) -> std::result::Result<List, Error> {
    let summary = read_mesh_summary(path).map_err(Error::Other)?;
    Ok(list!(
        n_vertices = summary.n_vertices as i32,
        xmin = summary.xmin,
        xmax = summary.xmax,
        ymin = summary.ymin,
        ymax = summary.ymax
    ))
}

#[extendr]
fn rinla_ar1_precision(n: i32, rho: f64, tau: f64) -> std::result::Result<List, Error> {
    let n_usize =
        usize::try_from(n).map_err(|_| Error::Other("n must be non-negative".to_string()))?;
    let precision = ar1_precision(n_usize, rho, tau).map_err(Error::Other)?;

    let mut q = r!(precision.row_major_values);
    q.set_attrib(sym!(dim), r!([n, n]))?;

    let i: Vec<i32> = precision.i.into_iter().map(|v| v as i32).collect();
    let j: Vec<i32> = precision.j.into_iter().map(|v| v as i32).collect();

    Ok(list!(n = n, i = i, j = j, x = precision.x, q = q))
}

#[extendr]
fn rinla_ar1_precision_csc_dgcmatrix(
    n: i32,
    rho: f64,
    tau: f64,
) -> std::result::Result<Robj, Error> {
    let n_usize =
        usize::try_from(n).map_err(|_| Error::Other("n must be non-negative".to_string()))?;
    let csc = ar1_precision_csc(n_usize, rho, tau).map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

fn csc_to_dgcmatrix(csc: &inla_core::sparse::CscMatrix) -> std::result::Result<Robj, Error> {
    R!(r#"if (!requireNamespace("Matrix", quietly = TRUE)) stop("Matrix package is required")"#)
        .map_err(map_r_error)?;

    let slots = csc_for_r_dgcmatrix(csc).map_err(Error::Other)?;

    let mut m = S4::new("dgCMatrix").map_err(map_r_error)?;
    m.set_slot("i", r!(slots.i)).map_err(map_r_error)?;
    m.set_slot("p", r!(slots.p)).map_err(map_r_error)?;
    m.set_slot("x", r!(slots.x)).map_err(map_r_error)?;
    m.set_slot("Dim", r!([slots.nrow as i32, slots.ncol as i32])).map_err(map_r_error)?;
    m.set_slot("Dimnames", list!(NULL, NULL))
        .map_err(map_r_error)?;
    m.set_slot("factors", list!()).map_err(map_r_error)?;

    Ok(m.into())
}

#[extendr]
fn rinla_rw1_precision_csc(n: i32, tau: f64) -> std::result::Result<Robj, Error> {
    let n_usize = usize::try_from(n).map_err(|_| Error::Other("n must be non-negative".to_string()))?;
    let csc = inla_core::rw1_precision_csc(n_usize, tau).map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

#[extendr]
fn rinla_rw2_precision_csc(n: i32, tau: f64) -> std::result::Result<Robj, Error> {
    let n_usize = usize::try_from(n).map_err(|_| Error::Other("n must be non-negative".to_string()))?;
    let csc = inla_core::rw2_precision_csc(n_usize, tau).map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

#[extendr]
fn rinla_rw1_cyclic_precision_csc(n: i32, tau: f64) -> std::result::Result<Robj, Error> {
    let n_usize = usize::try_from(n).map_err(|_| Error::Other("n must be non-negative".to_string()))?;
    let csc = inla_core::rw1_cyclic_precision_csc(n_usize, tau).map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

#[extendr]
fn rinla_rw2_cyclic_precision_csc(n: i32, tau: f64) -> std::result::Result<Robj, Error> {
    let n_usize = usize::try_from(n).map_err(|_| Error::Other("n must be non-negative".to_string()))?;
    let csc = inla_core::rw2_cyclic_precision_csc(n_usize, tau).map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

#[extendr]
fn rinla_seasonal_precision_csc(n: i32, s: i32, tau: f64, cyclic: bool) -> std::result::Result<Robj, Error> {
    let n_usize = usize::try_from(n).map_err(|_| Error::Other("n must be non-negative".to_string()))?;
    let s_usize = usize::try_from(s).map_err(|_| Error::Other("s must be non-negative".to_string()))?;
    let csc = inla_core::seasonal_precision_csc(n_usize, s_usize, tau, cyclic).map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

#[extendr]
fn rinla_two_diid_precision_csc(n_pairs: i32, rho: f64, tau: f64) -> std::result::Result<Robj, Error> {
    let n_pairs_usize = usize::try_from(n_pairs).map_err(|_| Error::Other("n_pairs must be non-negative".to_string()))?;
    let csc = inla_core::two_diid_precision_csc(n_pairs_usize, rho, tau).map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

#[extendr]
fn rinla_iid_precision_csc(n: i32, tau: f64) -> std::result::Result<Robj, Error> {
    let n_usize = usize::try_from(n).map_err(|_| Error::Other("n must be non-negative".to_string()))?;
    let csc = inla_core::iid_precision_csc(n_usize, tau).map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

#[extendr]
fn rinla_arp_precision_csc(n: i32, pacf: Vec<f64>, tau: f64) -> std::result::Result<Robj, Error> {
    let n_usize = usize::try_from(n).map_err(|_| Error::Other("n must be non-negative".to_string()))?;
    let csc = inla_core::arp_precision_csc(n_usize, &pacf, tau).map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

#[extendr]
fn rinla_matern2d_precision_csc(
    nrow: i32,
    ncol: i32,
    nu: i32,
    range: f64,
    prec: f64,
    cyclic: bool,
) -> std::result::Result<Robj, Error> {
    let nrow_u = usize::try_from(nrow).map_err(|_| Error::Other("nrow must be non-negative".to_string()))?;
    let ncol_u = usize::try_from(ncol).map_err(|_| Error::Other("ncol must be non-negative".to_string()))?;
    let nu_u = usize::try_from(nu).map_err(|_| Error::Other("nu must be non-negative".to_string()))?;
    let csc = inla_core::matern2d_precision_csc(nrow_u, ncol_u, nu_u, range, prec, cyclic).map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

#[extendr]
fn rinla_besag_precision_csc(adj_list: List, tau: f64) -> std::result::Result<Robj, Error> {
    let mut adj = Vec::with_capacity(adj_list.len());
    for item in adj_list.values() {
        let nbs: Vec<usize> = item.as_integer_vector()
            .ok_or_else(|| Error::Other("adj_list must contain integer vectors".to_string()))?
            .into_iter()
            .map(|val| (val - 1) as usize)
            .collect();
        adj.push(nbs);
    }
    let csc = inla_core::besag_precision_csc(&adj, tau).map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

#[extendr]
fn rinla_bym_precision_csc(adj_list: List, tau_spatial: f64, tau_iid: f64) -> std::result::Result<Robj, Error> {
    let mut adj = Vec::with_capacity(adj_list.len());
    for item in adj_list.values() {
        let nbs: Vec<usize> = item.as_integer_vector()
            .ok_or_else(|| Error::Other("adj_list must contain integer vectors".to_string()))?
            .into_iter()
            .map(|val| (val - 1) as usize)
            .collect();
        adj.push(nbs);
    }
    let csc = inla_core::bym_precision_csc(&adj, tau_spatial, tau_iid).map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

#[extendr]
fn rinla_spde_precision_mesh_csc(
    vertices_mat: Robj,
    triangles_mat: Robj,
    kappa: f64,
    tau: f64,
) -> std::result::Result<Robj, Error> {
    if !vertices_mat.is_matrix() || !triangles_mat.is_matrix() {
        return Err(Error::Other("vertices and triangles must be matrices".to_string()));
    }
    
    let dim_v = vertices_mat.dim().ok_or_else(|| Error::Other("could not get vertices dims".to_string()))?;
    let dim_t = triangles_mat.dim().ok_or_else(|| Error::Other("could not get triangles dims".to_string()))?;
    
    if dim_v.len() != 2 || dim_v[1] != 2 {
        return Err(Error::Other("vertices must be an N x 2 matrix".to_string()));
    }
    if dim_t.len() != 2 || dim_t[1] != 3 {
        return Err(Error::Other("triangles must be an M x 3 matrix".to_string()));
    }

    let n_vertices = dim_v[0].0 as usize;
    let n_triangles = dim_t[0].0 as usize;

    let vertices_data = vertices_mat.as_real_vector()
        .ok_or_else(|| Error::Other("vertices must be real".to_string()))?;
    let triangles_data = triangles_mat.as_integer_vector()
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

    let mesh = inla_core::fmesher::build_mesh2d(vertices, triangles).map_err(Error::Other)?;
    let fem = mesh.assemble_fem_blocks();
    let csc = inla_core::spde::spde_precision_csc(&fem, kappa, tau).map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

#[extendr]
fn rinla_crw1_precision_csc(positions: Vec<f64>, tau: f64) -> std::result::Result<Robj, Error> {
    let csc = inla_core::crw1_precision_csc(&positions, tau).map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

#[extendr]
fn rinla_crw2_precision_csc(positions: Vec<f64>, tau: f64, layout: &str) -> std::result::Result<Robj, Error> {
    let csc = inla_core::crw2_precision_csc(&positions, tau, layout).map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

#[extendr]
fn rinla_fgn_precision_csc(n: i32, hurst: f64, tau: f64) -> std::result::Result<Robj, Error> {
    let n_usize = usize::try_from(n).map_err(|_| Error::Other("n must be non-negative".to_string()))?;
    let csc = inla_core::fgn_precision_csc(n_usize, hurst, tau).map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

#[extendr]
fn rinla_run_inla_inference(
    initial_theta: Vec<f64>,
    model_type: &str,
    y_obs: Vec<f64>,
    obs_precision: f64,
    strategy: &str,
    step_or_f0: f64,
) -> std::result::Result<List, Error> {
    let n = y_obs.len();
    
    // 1. Build observation array (Gaussian observations)
    let mut obs = Vec::with_capacity(n);
    for &y in &y_obs {
        obs.push(inla_core::Obs::Gaussian(inla_core::GaussianObs {
            y,
            precision: obs_precision,
            link: inla_core::Link::Identity,
        }));
    }

    let model_type_str = model_type.to_lowercase();
    
    let build_prior = move |theta: &[f64]| -> std::result::Result<inla_core::CscMatrix, String> {
        match model_type_str.as_str() {
            "fgn" => {
                if theta.len() < 2 {
                    return Err("FGN requires 2 hyperparameters: theta = [log(tau), logit(H)]".to_string());
                }
                let tau = theta[0].exp();
                let hurst = 1.0 / (1.0 + (-theta[1]).exp());
                inla_core::fgn_precision_csc(n, hurst, tau)
            }
            "ar1" => {
                if theta.len() < 2 {
                    return Err("AR1 requires 2 hyperparameters: theta = [log(tau), logit((rho+1)/2)]".to_string());
                }
                let tau = theta[0].exp();
                let rho = 2.0 / (1.0 + (-theta[1]).exp()) - 1.0;
                inla_core::ar1_precision_csc(n, rho, tau)
            }
            "rw2" => {
                if theta.is_empty() {
                    return Err("RW2 requires 1 hyperparameter: theta = [log(tau)]".to_string());
                }
                let tau = theta[0].exp();
                inla_core::rw2_precision_csc(n, tau)
            }
            _ => Err(format!("Unsupported model type for formula solving: {}", model_type_str)),
        }
    };

    let log_prior_density = |theta: &[f64]| -> f64 {
        let mut lprior = 0.0;
        for &val in theta {
            lprior += -0.5 * 0.1 * val * val;
        }
        lprior
    };

    // 3. Run solver
    let result = inla_core::run_inla_inference(
        &initial_theta,
        &build_prior,
        &log_prior_density,
        &obs,
        strategy,
        step_or_f0,
    ).map_err(Error::Other)?;

    // 4. Return results as an extendr List
    Ok(list!(
        mode = result.mode,
        hessian = result.hessian,
        latent_means = result.latent_means,
        latent_variances = result.latent_variances,
        marginal_log_lik = result.marginal_log_lik,
        marginal_log_lik_gaussian = result.marginal_log_lik_gaussian,
        dic = result.dic,
        mean_deviance = result.mean_deviance,
        effective_params = result.effective_params
    ))
}

extendr_module! {
    mod rinla_core;
    fn rinla_read_mesh;
    fn rinla_ar1_precision;
    fn rinla_ar1_precision_csc_dgcmatrix;
    fn rinla_rw1_precision_csc;
    fn rinla_rw2_precision_csc;
    fn rinla_rw1_cyclic_precision_csc;
    fn rinla_rw2_cyclic_precision_csc;
    fn rinla_seasonal_precision_csc;
    fn rinla_two_diid_precision_csc;
    fn rinla_iid_precision_csc;
    fn rinla_arp_precision_csc;
    fn rinla_matern2d_precision_csc;
    fn rinla_besag_precision_csc;
    fn rinla_bym_precision_csc;
    fn rinla_spde_precision_mesh_csc;
    fn rinla_crw1_precision_csc;
    fn rinla_crw2_precision_csc;
    fn rinla_fgn_precision_csc;
    fn rinla_run_inla_inference;
}

