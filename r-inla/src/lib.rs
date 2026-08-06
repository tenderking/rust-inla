use extendr_api::prelude::*;
use inla_core::ar1::ar1_precision;
use inla_core::mesh::read_mesh_summary;
use inla_core::sparse::{ar1_precision_csc, csc_for_r_dgcmatrix};

fn map_r_error(e: Error) -> Error {
    Error::Other(e.to_string())
}

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

#[extendr]
fn inla_rs_ar1_precision(n: i32, rho: f64, tau: f64) -> std::result::Result<List, Error> {
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
fn inla_rs_ar1_precision_csc_dgcmatrix(
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
    m.set_slot("Dim", r!([slots.nrow as i32, slots.ncol as i32]))
        .map_err(map_r_error)?;
    m.set_slot("Dimnames", list!(NULL, NULL))
        .map_err(map_r_error)?;
    m.set_slot("factors", list!()).map_err(map_r_error)?;

    Ok(m.into())
}

#[extendr]
fn inla_rs_rw1_precision_csc(n: i32, tau: f64) -> std::result::Result<Robj, Error> {
    let n_usize =
        usize::try_from(n).map_err(|_| Error::Other("n must be non-negative".to_string()))?;
    let csc = inla_core::rw1_precision_csc(n_usize, tau).map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

#[extendr]
fn inla_rs_rw2_precision_csc(n: i32, tau: f64) -> std::result::Result<Robj, Error> {
    let n_usize =
        usize::try_from(n).map_err(|_| Error::Other("n must be non-negative".to_string()))?;
    let csc = inla_core::rw2_precision_csc(n_usize, tau).map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

#[extendr]
fn inla_rs_rw1_cyclic_precision_csc(n: i32, tau: f64) -> std::result::Result<Robj, Error> {
    let n_usize =
        usize::try_from(n).map_err(|_| Error::Other("n must be non-negative".to_string()))?;
    let csc = inla_core::rw1_cyclic_precision_csc(n_usize, tau).map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

#[extendr]
fn inla_rs_rw2_cyclic_precision_csc(n: i32, tau: f64) -> std::result::Result<Robj, Error> {
    let n_usize =
        usize::try_from(n).map_err(|_| Error::Other("n must be non-negative".to_string()))?;
    let csc = inla_core::rw2_cyclic_precision_csc(n_usize, tau).map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

#[extendr]
fn inla_rs_seasonal_precision_csc(
    n: i32,
    s: i32,
    tau: f64,
    cyclic: bool,
) -> std::result::Result<Robj, Error> {
    let n_usize =
        usize::try_from(n).map_err(|_| Error::Other("n must be non-negative".to_string()))?;
    let s_usize =
        usize::try_from(s).map_err(|_| Error::Other("s must be non-negative".to_string()))?;
    let csc =
        inla_core::seasonal_precision_csc(n_usize, s_usize, tau, cyclic).map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

#[extendr]
fn inla_rs_two_diid_precision_csc(
    n_pairs: i32,
    rho: f64,
    tau: f64,
) -> std::result::Result<Robj, Error> {
    let n_pairs_usize = usize::try_from(n_pairs)
        .map_err(|_| Error::Other("n_pairs must be non-negative".to_string()))?;
    let csc = inla_core::two_diid_precision_csc(n_pairs_usize, rho, tau).map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

#[extendr]
fn inla_rs_iid_precision_csc(n: i32, tau: f64) -> std::result::Result<Robj, Error> {
    let n_usize =
        usize::try_from(n).map_err(|_| Error::Other("n must be non-negative".to_string()))?;
    let csc = inla_core::iid_precision_csc(n_usize, tau).map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

#[extendr]
fn inla_rs_arp_precision_csc(n: i32, pacf: Vec<f64>, tau: f64) -> std::result::Result<Robj, Error> {
    let n_usize =
        usize::try_from(n).map_err(|_| Error::Other("n must be non-negative".to_string()))?;
    let csc = inla_core::arp_precision_csc(n_usize, &pacf, tau).map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

#[extendr]
fn inla_rs_matern2d_precision_csc(
    nrow: i32,
    ncol: i32,
    nu: i32,
    range: f64,
    prec: f64,
    cyclic: bool,
) -> std::result::Result<Robj, Error> {
    let nrow_u =
        usize::try_from(nrow).map_err(|_| Error::Other("nrow must be non-negative".to_string()))?;
    let ncol_u =
        usize::try_from(ncol).map_err(|_| Error::Other("ncol must be non-negative".to_string()))?;
    let nu_u =
        usize::try_from(nu).map_err(|_| Error::Other("nu must be non-negative".to_string()))?;
    let csc = inla_core::matern2d_precision_csc(nrow_u, ncol_u, nu_u, range, prec, cyclic)
        .map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

#[extendr]
fn inla_rs_rw2d_precision_csc(
    nrow: i32,
    ncol: i32,
    tau: f64,
    cyclic: bool,
    bvalue_zero: bool,
) -> std::result::Result<Robj, Error> {
    let nrow_u =
        usize::try_from(nrow).map_err(|_| Error::Other("nrow must be non-negative".to_string()))?;
    let ncol_u =
        usize::try_from(ncol).map_err(|_| Error::Other("ncol must be non-negative".to_string()))?;
    let csc = inla_core::rw2d_precision_csc(nrow_u, ncol_u, tau, cyclic, bvalue_zero)
        .map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

#[extendr]
fn inla_rs_besag_precision_csc(adj_list: List, tau: f64) -> std::result::Result<Robj, Error> {
    let mut adj = Vec::with_capacity(adj_list.len());
    for item in adj_list.values() {
        let nbs: Vec<usize> = item
            .as_integer_vector()
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
fn inla_rs_bym_precision_csc(
    adj_list: List,
    tau_spatial: f64,
    tau_iid: f64,
) -> std::result::Result<Robj, Error> {
    let mut adj = Vec::with_capacity(adj_list.len());
    for item in adj_list.values() {
        let nbs: Vec<usize> = item
            .as_integer_vector()
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
fn inla_rs_bym2_precision_csc(
    adj_list: List,
    tau: f64,
    phi: f64,
) -> std::result::Result<Robj, Error> {
    let mut adj = Vec::with_capacity(adj_list.len());
    for item in adj_list.values() {
        let nbs: Vec<usize> = item
            .as_integer_vector()
            .ok_or_else(|| Error::Other("adj_list must contain integer vectors".to_string()))?
            .into_iter()
            .map(|val| (val - 1) as usize)
            .collect();
        adj.push(nbs);
    }
    let csc = inla_core::bym2_precision_csc(&adj, tau, phi).map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

#[extendr]
fn inla_rs_spde_precision_mesh_csc(
    vertices_mat: Robj,
    triangles_mat: Robj,
    kappa: f64,
    tau: f64,
) -> std::result::Result<Robj, Error> {
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

    let mesh = inla_core::fmesher::build_mesh2d(vertices, triangles).map_err(Error::Other)?;
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

#[extendr]
fn inla_rs_crw1_precision_csc(positions: Vec<f64>, tau: f64) -> std::result::Result<Robj, Error> {
    let csc = inla_core::crw1_precision_csc(&positions, tau).map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

#[extendr]
fn inla_rs_crw2_precision_csc(
    positions: Vec<f64>,
    tau: f64,
    layout: &str,
) -> std::result::Result<Robj, Error> {
    let csc = inla_core::crw2_precision_csc(&positions, tau, layout).map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

#[extendr]
fn inla_rs_fgn_precision_csc(n: i32, hurst: f64, tau: f64) -> std::result::Result<Robj, Error> {
    let n_usize =
        usize::try_from(n).map_err(|_| Error::Other("n must be non-negative".to_string()))?;
    let csc = inla_core::fgn_precision_csc(n_usize, hurst, tau).map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

#[extendr]
fn inla_rs_fgn_approx_precision_csc(
    n: i32,
    hurst: f64,
    tau: f64,
    order: i32,
    prec_eps: f64,
) -> std::result::Result<Robj, Error> {
    let n_usize =
        usize::try_from(n).map_err(|_| Error::Other("n must be non-negative".to_string()))?;
    let order_usize = usize::try_from(order)
        .map_err(|_| Error::Other("order must be non-negative".to_string()))?;
    let csc = inla_core::fgn_approx_precision_csc(n_usize, hurst, tau, order_usize, prec_eps)
        .map_err(Error::Other)?;
    csc_to_dgcmatrix(&csc)
}

fn parse_link(link: &str, family: &str) -> std::result::Result<inla_core::Link, Error> {
    let link = link.trim().to_lowercase();
    if link.is_empty() || link == "default" {
        return Ok(match family {
            "gaussian" | "laplace" => inla_core::Link::Identity,
            "poisson"
            | "nbinomial"
            | "negative_binomial"
            | "zeroinflatedpoisson0"
            | "zeroinflatedpoisson1"
            | "zero_inflated_poisson"
            | "exponential"
            | "exponential_survival"
            | "weibull"
            | "weibull_survival" => inla_core::Link::Log,
            "binomial"
            | "zeroinflatedbinomial0"
            | "zeroinflatedbinomial1"
            | "zero_inflated_binomial" => inla_core::Link::Logit,
            _ => inla_core::Link::Identity,
        });
    }
    match link.as_str() {
        "identity" => Ok(inla_core::Link::Identity),
        "log" => Ok(inla_core::Link::Log),
        "logit" => Ok(inla_core::Link::Logit),
        other => Err(Error::Other(format!("unknown link function: {other}"))),
    }
}

fn pad_or_default(values: &[f64], n: usize, default: f64) -> Vec<f64> {
    if values.is_empty() {
        return vec![default; n];
    }
    if values.len() == 1 {
        return vec![values[0]; n];
    }
    if values.len() == n {
        return values.to_vec();
    }
    // Truncate or pad with last/default so callers can pass partial vectors safely.
    let mut out = vec![default; n];
    let m = values.len().min(n);
    out[..m].copy_from_slice(&values[..m]);
    out
}

fn build_observations(
    family: &str,
    y_obs: &[f64],
    n_latent: usize,
    link: inla_core::Link,
    obs_precision: f64,
    exposure: &[f64],
    ntrials: &[f64],
    event: &[f64],
    size: f64,
    zero_prob: f64,
    inflation: &str,
    alpha: f64,
    gamma: f64,
    shape: f64,
) -> std::result::Result<Vec<inla_core::Obs>, Error> {
    let n = y_obs.len();
    let fam = family.trim().to_lowercase();
    let inflation_ty = match inflation.trim().to_lowercase().as_str() {
        "type1" | "1" => inla_core::ZeroInflationType::Type1,
        _ => inla_core::ZeroInflationType::Type0,
    };
    let e = pad_or_default(exposure, n, 1.0);
    let nt = pad_or_default(ntrials, n, 1.0);
    let ev = pad_or_default(event, n, 1.0);

    let mut obs = Vec::with_capacity(n_latent);
    for i in 0..n {
        let y = y_obs[i];
        let one = match fam.as_str() {
            "gaussian" => inla_core::Obs::Gaussian(inla_core::GaussianObs {
                y,
                precision: obs_precision,
                link,
            }),
            "poisson" => inla_core::Obs::Poisson(inla_core::PoissonObs {
                y,
                exposure: e[i],
                link,
            }),
            "binomial" => inla_core::Obs::Binomial(inla_core::BinomialObs { y, n: nt[i], link }),
            "nbinomial" | "negative_binomial" => {
                inla_core::Obs::NegativeBinomial(inla_core::NegativeBinomialObs {
                    y,
                    exposure: e[i],
                    size,
                    link,
                })
            }
            "zeroinflatedpoisson0" | "zero_inflated_poisson" => {
                inla_core::Obs::ZeroInflatedPoisson(inla_core::ZeroInflatedPoissonObs {
                    y,
                    exposure: e[i],
                    zero_prob,
                    link,
                    inflation: inflation_ty,
                })
            }
            "zeroinflatedpoisson1" => {
                inla_core::Obs::ZeroInflatedPoisson(inla_core::ZeroInflatedPoissonObs {
                    y,
                    exposure: e[i],
                    zero_prob,
                    link,
                    inflation: inla_core::ZeroInflationType::Type1,
                })
            }
            "zeroinflatedbinomial0" | "zero_inflated_binomial" => {
                inla_core::Obs::ZeroInflatedBinomial(inla_core::ZeroInflatedBinomialObs {
                    y,
                    n: nt[i],
                    zero_prob,
                    link,
                    inflation: inflation_ty,
                })
            }
            "zeroinflatedbinomial1" => {
                inla_core::Obs::ZeroInflatedBinomial(inla_core::ZeroInflatedBinomialObs {
                    y,
                    n: nt[i],
                    zero_prob,
                    link,
                    inflation: inla_core::ZeroInflationType::Type1,
                })
            }
            "laplace" => inla_core::Obs::Laplace(inla_core::LaplaceObs {
                y,
                alpha,
                gamma,
                link,
            }),
            "exponential" | "exponential_survival" => {
                inla_core::Obs::ExponentialSurvival(inla_core::ExponentialSurvivalObs {
                    y,
                    event: ev[i],
                    link,
                })
            }
            "weibull" | "weibull_survival" => {
                inla_core::Obs::WeibullSurvival(inla_core::WeibullSurvivalObs {
                    y,
                    event: ev[i],
                    shape,
                    link,
                })
            }
            other => {
                return Err(Error::Other(format!(
                    "unsupported observation family: {other}"
                )));
            }
        };
        obs.push(one);
    }
    for _ in n..n_latent {
        obs.push(inla_core::Obs::None);
    }
    Ok(obs)
}

#[extendr]
fn inla_rs_run_inla_inference(
    initial_theta: Vec<f64>,
    model_type: &str,
    y_obs: Vec<f64>,
    obs_precision: f64,
    strategy: &str,
    step_or_f0: f64,
    order: i32,
    family: &str,
    link: &str,
    exposure: Vec<f64>,
    ntrials: Vec<f64>,
    event: Vec<f64>,
    size: f64,
    zero_prob: f64,
    inflation: &str,
    alpha: f64,
    gamma: f64,
    shape: f64,
    adj_list: List,
    deterministic: bool,
) -> std::result::Result<List, Error> {
    let n = y_obs.len();
    let model_type_str = model_type.to_lowercase();
    let family_str = family.trim().to_lowercase();
    let use_fgn_approx = model_type_str == "fgn" && order > 0;
    let order_usize = if use_fgn_approx {
        usize::try_from(order)
            .map_err(|_| Error::Other("order must be non-negative".to_string()))?
    } else {
        0
    };
    if use_fgn_approx && order_usize != 3 && order_usize != 4 {
        return Err(Error::Other(
            "FGN order must be 3 or 4 (R-INLA AR mixture), or 0 for exact dense FGN".to_string(),
        ));
    }

    let adj: Option<Vec<Vec<usize>>> = if model_type_str == "besag" {
        let mut out = Vec::with_capacity(adj_list.len());
        for item in adj_list.values() {
            let nbs: Vec<usize> = item
                .as_integer_vector()
                .ok_or_else(|| Error::Other("adj_list must contain integer vectors".to_string()))?
                .into_iter()
                .map(|val| (val - 1) as usize)
                .collect();
            out.push(nbs);
        }
        if out.len() != n {
            return Err(Error::Other(format!(
                "besag adj_list length ({}) must equal length(y) ({n})",
                out.len()
            )));
        }
        Some(out)
    } else {
        None
    };

    // Observations: for approx FGN, only the first n (z-block) are observed.
    let n_latent = if use_fgn_approx {
        inla_core::fgn_approx_latent_len(n, order_usize)
    } else {
        n
    };
    let link_ty = parse_link(link, &family_str)?;
    let obs = build_observations(
        &family_str,
        &y_obs,
        n_latent,
        link_ty,
        obs_precision,
        &exposure,
        &ntrials,
        &event,
        size,
        zero_prob,
        inflation,
        alpha,
        gamma,
        shape,
    )?;

    let model_type_owned = model_type_str.clone();
    let adj_owned = adj.clone();
    let build_prior = move |theta: &[f64]| -> std::result::Result<inla_core::CscMatrix, String> {
        match model_type_owned.as_str() {
            "fgn" if use_fgn_approx => {
                if theta.len() < 2 {
                    return Err(
                        "FGN requires 2 hyperparameters: theta = [log(tau), H_intern]".to_string(),
                    );
                }
                let tau = theta[0].exp();
                let hurst = inla_core::fgn_hurst_from_intern(theta[1]);
                // R-INLA default f(..., precision = 1e8)
                inla_core::fgn_approx_precision_csc(n, hurst, tau, order_usize, 1e8)
            }
            "fgn" => {
                if theta.len() < 2 {
                    return Err(
                        "FGN requires 2 hyperparameters: theta = [log(tau), logit(H)]".to_string(),
                    );
                }
                let tau = theta[0].exp();
                let hurst = 1.0 / (1.0 + (-theta[1]).exp());
                inla_core::fgn_precision_csc(n, hurst, tau)
            }
            "ar1" => {
                if theta.len() < 2 {
                    return Err(
                        "AR1 requires 2 hyperparameters: theta = [log(tau), logit((rho+1)/2)]"
                            .to_string(),
                    );
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
            "iid" => {
                if theta.is_empty() {
                    return Err("IID requires 1 hyperparameter: theta = [log(tau)]".to_string());
                }
                let tau = theta[0].exp();
                inla_core::iid_precision_csc(n, tau)
            }
            "besag" => {
                if theta.is_empty() {
                    return Err("Besag requires 1 hyperparameter: theta = [log(tau)]".to_string());
                }
                let tau = theta[0].exp();
                let adj = adj_owned
                    .as_ref()
                    .ok_or_else(|| "besag requires adj_list".to_string())?;
                inla_core::besag_precision_csc(adj, tau)
            }
            _ => Err(format!(
                "Unsupported model type for formula solving: {}",
                model_type_owned
            )),
        }
    };

    let constr = match inla_core::model_rank_deficiency(&model_type_str) {
        0 => None,
        k => Some(inla_core::sum_to_zero_constraint(n_latent, k).map_err(Error::Other)?),
    };

    let prior_stack = match inla_core::HyperPriorStack::default_for_effect(&model_type_str) {
        Ok(s) => s,
        Err(_) => inla_core::HyperPriorStack::new(vec![inla_core::PriorSpec::gaussian(0.0, 0.1)]),
    };
    let log_prior_density =
        move |theta: &[f64]| -> f64 { prior_stack.log_density(theta).unwrap_or(f64::NEG_INFINITY) };

    let result = inla_core::run_inla_inference_a(
        &initial_theta,
        &build_prior,
        &log_prior_density,
        &obs,
        None,
        constr.as_ref(),
        strategy,
        step_or_f0,
        &inla_core::MarginalOptions::default(),
        deterministic,
    )
    .map_err(Error::Other)?;

    // Report FGN of interest (z-block) and a Hurst summary matching R-INLA scale when approx.
    let latent_means: Vec<f64> = result.latent_means.iter().take(n).copied().collect();
    let latent_variances: Vec<f64> = result.latent_variances.iter().take(n).copied().collect();
    let hurst_est = if model_type_str == "fgn" && result.mode.len() >= 2 {
        if use_fgn_approx {
            inla_core::fgn_hurst_from_intern(result.mode[1])
        } else {
            1.0 / (1.0 + (-result.mode[1]).exp())
        }
    } else {
        f64::NAN
    };

    Ok(list!(
        mode = result.mode,
        hessian = result.hessian,
        latent_means = latent_means,
        latent_variances = latent_variances,
        predictor_means = result.predictor_means,
        predictor_variances = result.predictor_variances,
        marginal_log_lik = result.marginal_log_lik,
        marginal_log_lik_gaussian = result.marginal_log_lik_gaussian,
        dic = result.dic,
        mean_deviance = result.mean_deviance,
        effective_params = result.effective_params,
        hurst = hurst_est,
        order = order
    ))
}

fn parse_adj_list_1based(adj_list: &List) -> std::result::Result<Vec<Vec<usize>>, Error> {
    let mut adj = Vec::with_capacity(adj_list.len());
    for item in adj_list.values() {
        let nbs: Vec<usize> = item
            .as_integer_vector()
            .ok_or_else(|| Error::Other("adj_list must contain integer vectors".to_string()))?
            .into_iter()
            .map(|val| (val - 1) as usize)
            .collect();
        adj.push(nbs);
    }
    Ok(adj)
}

fn scale_csc_entries(
    q: &inla_core::CscMatrix,
    scale: f64,
) -> std::result::Result<inla_core::CscMatrix, String> {
    inla_core::scale_csc(q, scale)
}

fn marginal_to_r_matrix(m: &inla_core::Marginal1D) -> std::result::Result<Robj, Error> {
    let n = m.x.len();
    let mut data = Vec::with_capacity(n * 2);
    // column-major: first column x, second column y
    data.extend_from_slice(&m.x);
    data.extend_from_slice(&m.y);
    let mut mat = r!(data);
    mat.set_attrib(sym!(dim), r!([n as i32, 2i32]))?;
    Ok(mat)
}

fn marginals_to_r_list(ms: &[inla_core::Marginal1D]) -> std::result::Result<List, Error> {
    let mut items: Vec<Robj> = Vec::with_capacity(ms.len());
    for m in ms {
        items.push(marginal_to_r_matrix(m)?);
    }
    Ok(List::from_values(items))
}

/// Structured INLA fit: A-matrix projector + block-diagonal multi-effect prior.
///
/// `effect_types`: "iid"|"ar1"|"rw2"|"besag"|"fixed"|"fgn"
/// `adj_lists`: list of the same length as effects; non-besag entries may be `list()`.
#[extendr]
fn inla_rs_run_inla_structured(
    initial_theta: Vec<f64>,
    y_obs: Vec<f64>,
    obs_precision: f64,
    strategy: &str,
    step_or_f0: f64,
    family: &str,
    link: &str,
    a_i: Vec<i32>,
    a_j: Vec<i32>,
    a_x: Vec<f64>,
    a_nrow: i32,
    a_ncol: i32,
    effect_types: Vec<String>,
    effect_ns: Vec<i32>,
    effect_scales: Vec<i32>,
    effect_theta_lens: Vec<i32>,
    effect_orders: Vec<i32>,
    adj_lists: List,
    fixed_prec: f64,
    exposure: Vec<f64>,
    ntrials: Vec<f64>,
    event: Vec<f64>,
    size: f64,
    zero_prob: f64,
    inflation: &str,
    alpha: f64,
    gamma: f64,
    shape: f64,
    deterministic: bool,
) -> std::result::Result<List, Error> {
    let n_obs = y_obs.len();
    let a_nrow_u = usize::try_from(a_nrow).map_err(|_| Error::Other("a_nrow".into()))?;
    let a_ncol_u = usize::try_from(a_ncol).map_err(|_| Error::Other("a_ncol".into()))?;
    if a_nrow_u != n_obs {
        return Err(Error::Other(format!(
            "A.nrows ({a_nrow_u}) must equal length(y) ({n_obs})"
        )));
    }
    if effect_types.len() != effect_ns.len()
        || effect_types.len() != effect_scales.len()
        || effect_types.len() != effect_theta_lens.len()
        || effect_types.len() != effect_orders.len()
    {
        return Err(Error::Other(
            "effect_* vectors must have equal length".to_string(),
        ));
    }
    if adj_lists.len() != effect_types.len() {
        return Err(Error::Other(
            "adj_lists length must match number of effects".to_string(),
        ));
    }

    let rows: Vec<usize> = a_i.iter().map(|&v| v as usize).collect();
    let cols: Vec<usize> = a_j.iter().map(|&v| v as usize).collect();
    let a = inla_core::csc_from_triplets_0based(a_nrow_u, a_ncol_u, &rows, &cols, &a_x)
        .map_err(Error::Other)?;

    let family_str = family.trim().to_lowercase();
    let link_ty = parse_link(link, &family_str)?;
    let obs = build_observations(
        &family_str,
        &y_obs,
        n_obs,
        link_ty,
        obs_precision,
        &exposure,
        &ntrials,
        &event,
        size,
        zero_prob,
        inflation,
        alpha,
        gamma,
        shape,
    )?;

    // Parse adjacency once per besag effect
    let mut adjs: Vec<Option<Vec<Vec<usize>>>> = Vec::with_capacity(effect_types.len());
    for (ei, item) in adj_lists.values().enumerate() {
        let typ = effect_types[ei].to_lowercase();
        if typ == "besag" || typ == "bym" || typ == "bym2" {
            let sub: List = item.try_into().map_err(|e| {
                Error::Other(format!(
                    "adj_lists[{ei}] must be a list of integer vectors: {e}"
                ))
            })?;
            adjs.push(Some(parse_adj_list_1based(&sub)?));
        } else {
            adjs.push(None);
        }
    }

    let effect_types_owned = effect_types.clone();
    let effect_ns_u: Vec<usize> = effect_ns
        .iter()
        .map(|&v| usize::try_from(v).unwrap_or(0))
        .collect();
    let effect_scales_b: Vec<bool> = effect_scales.iter().map(|&v| v != 0).collect();
    let effect_theta_lens_u: Vec<usize> = effect_theta_lens
        .iter()
        .map(|&v| usize::try_from(v).unwrap_or(0))
        .collect();
    let effect_orders_u: Vec<usize> = effect_orders
        .iter()
        .map(|&v| usize::try_from(v).unwrap_or(0))
        .collect();
    let effect_orders_i: Vec<i32> = effect_orders.clone();
    let expected_theta: usize = effect_theta_lens_u.iter().sum();
    if initial_theta.len() != expected_theta {
        return Err(Error::Other(format!(
            "initial_theta length {} != sum(effect_theta_lens) {}",
            initial_theta.len(),
            expected_theta
        )));
    }
    let n_lat_expected: usize = effect_ns_u.iter().sum();
    if n_lat_expected != a_ncol_u {
        return Err(Error::Other(format!(
            "sum(effect_ns)={n_lat_expected} != A.ncols={a_ncol_u}"
        )));
    }

    // Hard sum-to-zero on intrinsic random fields (before moving effect_* into build_prior).
    let constr_opt = {
        let mut stacked: Option<inla_core::ConstraintSpec> = None;
        let mut offset = 0usize;
        for (ei, typ) in effect_types.iter().enumerate() {
            let n_e = effect_ns_u[ei];
            let mut k = inla_core::model_rank_deficiency(typ);
            if typ.eq_ignore_ascii_case("rw2d") || typ.eq_ignore_ascii_case("matern2d") {
                // effect_orders: ±nrow (negative ⇒ cyclic). matern2d is proper → k=0.
                if typ.eq_ignore_ascii_case("rw2d") {
                    let raw = effect_orders[ei];
                    k = if raw < 0 { 1 } else { 2 };
                } else {
                    k = 0;
                }
            }
            if typ.eq_ignore_ascii_case("bym") {
                // Constrain spatial block only (first n = adj.len()).
                let n_sp = adjs
                    .get(ei)
                    .and_then(|a| a.as_ref())
                    .map(|a| a.len())
                    .unwrap_or(n_e / 2);
                let block = inla_core::sum_to_zero_constraint(n_sp, 1).map_err(Error::Other)?;
                let embedded = block.embed(a_ncol_u, offset).map_err(Error::Other)?;
                stacked = Some(match stacked {
                    None => embedded,
                    Some(prev) => prev.vstack(&embedded).map_err(Error::Other)?,
                });
                offset += n_e;
                continue;
            }
            if k > 0 {
                let block = inla_core::sum_to_zero_constraint(n_e, k).map_err(Error::Other)?;
                let embedded = block.embed(a_ncol_u, offset).map_err(Error::Other)?;
                stacked = Some(match stacked {
                    None => embedded,
                    Some(prev) => prev.vstack(&embedded).map_err(Error::Other)?,
                });
            }
            offset += n_e;
        }
        stacked
    };

    let build_prior = move |theta: &[f64]| -> std::result::Result<inla_core::CscMatrix, String> {
        let mut blocks = Vec::with_capacity(effect_types_owned.len());
        let mut off = 0usize;
        for ei in 0..effect_types_owned.len() {
            let typ = effect_types_owned[ei].to_lowercase();
            let n_e = effect_ns_u[ei];
            let tlen = effect_theta_lens_u[ei];
            let th = if tlen == 0 {
                &[][..]
            } else {
                &theta[off..off + tlen]
            };
            off += tlen;

            let q = match typ.as_str() {
                "fixed" => inla_core::identity_csc(n_e, fixed_prec)?,
                "iid" => {
                    let tau = th.first().copied().unwrap_or(0.0).exp();
                    let q0 = inla_core::iid_precision_csc(n_e, 1.0)?;
                    let q0 = if effect_scales_b[ei] {
                        inla_core::scale_model_csc(&q0)?
                    } else {
                        q0
                    };
                    scale_csc_entries(&q0, tau)?
                }
                "rw1" => {
                    let tau = th.first().copied().unwrap_or(0.0).exp();
                    let q0 = inla_core::rw1_precision_csc(n_e, 1.0)?;
                    let q0 = if effect_scales_b[ei] {
                        inla_core::scale_model_csc(&q0)?
                    } else {
                        q0
                    };
                    scale_csc_entries(&q0, tau)?
                }
                "rw2" => {
                    let tau = th.first().copied().unwrap_or(0.0).exp();
                    let q0 = inla_core::rw2_precision_csc(n_e, 1.0)?;
                    let q0 = if effect_scales_b[ei] {
                        inla_core::scale_model_csc(&q0)?
                    } else {
                        q0
                    };
                    scale_csc_entries(&q0, tau)?
                }
                "rw2d" => {
                    let tau = th.first().copied().unwrap_or(0.0).exp();
                    // effect_orders encodes ±nrow (negative ⇒ cyclic); ncol = n_e / nrow.
                    let raw = effect_orders_i[ei];
                    let cyclic = raw < 0;
                    let nrow = raw.unsigned_abs() as usize;
                    if nrow == 0 || n_e % nrow != 0 {
                        return Err(format!(
                            "rw2d: effect_orders (±nrow)={raw} incompatible with n={n_e}"
                        ));
                    }
                    let ncol = n_e / nrow;
                    let q0 = inla_core::rw2d_precision_csc(nrow, ncol, 1.0, cyclic, false)?;
                    scale_csc_entries(&q0, tau)?
                }
                "matern2d" => {
                    if th.len() < 2 {
                        return Err("matern2d needs theta=[log_prec, log_range]".into());
                    }
                    let prec = th[0].exp();
                    let range = th[1].exp();
                    let raw = effect_orders_i[ei];
                    let cyclic = raw < 0;
                    let nrow = raw.unsigned_abs() as usize;
                    if nrow == 0 || n_e % nrow != 0 {
                        return Err(format!(
                            "matern2d: effect_orders (±nrow)={raw} incompatible with n={n_e}"
                        ));
                    }
                    let ncol = n_e / nrow;
                    inla_core::matern2d_precision_csc(nrow, ncol, 1, range, prec, cyclic)?
                }
                "bym" => {
                    if th.len() < 2 {
                        return Err("bym needs theta=[log_tau_spatial, log_tau_iid]".into());
                    }
                    let adj = adjs[ei]
                        .as_ref()
                        .ok_or_else(|| "bym missing adj".to_string())?;
                    if adj.len() * 2 != n_e {
                        return Err(format!(
                            "bym adj length {} ⇒ latent {}, got n={}",
                            adj.len(),
                            adj.len() * 2,
                            n_e
                        ));
                    }
                    inla_core::bym_precision_csc(adj, th[0].exp(), th[1].exp())?
                }
                "bym2" => {
                    if th.len() < 2 {
                        return Err("bym2 needs theta=[log_tau, logit_phi]".into());
                    }
                    let adj = adjs[ei]
                        .as_ref()
                        .ok_or_else(|| "bym2 missing adj".to_string())?;
                    if adj.len() != n_e {
                        return Err(format!("bym2 adj length {} != effect n {}", adj.len(), n_e));
                    }
                    let tau = th[0].exp();
                    let phi = 1.0 / (1.0 + (-th[1]).exp());
                    let phi = phi.clamp(1e-6, 1.0 - 1e-6);
                    inla_core::bym2_precision_csc(adj, tau, phi)?
                }
                "seasonal" => {
                    let tau = th.first().copied().unwrap_or(0.0).exp();
                    let season = if effect_orders_u[ei] > 0 {
                        effect_orders_u[ei]
                    } else {
                        4
                    };
                    let q0 = inla_core::seasonal_precision_csc(n_e, season, 1.0, true)?;
                    let q0 = if effect_scales_b[ei] {
                        inla_core::scale_model_csc(&q0)?
                    } else {
                        q0
                    };
                    scale_csc_entries(&q0, tau)?
                }
                "ar1" => {
                    if th.len() < 2 {
                        return Err("ar1 needs theta=[log_tau, logit_rho]".into());
                    }
                    let tau = th[0].exp();
                    let rho = 2.0 / (1.0 + (-th[1]).exp()) - 1.0;
                    let q0 = inla_core::ar1_precision_csc(n_e, rho, 1.0)?;
                    let q0 = if effect_scales_b[ei] {
                        inla_core::scale_model_csc(&q0)?
                    } else {
                        q0
                    };
                    scale_csc_entries(&q0, tau)?
                }
                "ar" | "arp" => {
                    let tau = th.first().copied().unwrap_or(0.0).exp();
                    let pacf: Vec<f64> = if th.len() > 1 {
                        th[1..].iter().map(|&v| (v * 0.5).tanh()).collect()
                    } else {
                        vec![0.0]
                    };
                    let q0 = inla_core::arp_precision_csc(n_e, &pacf, 1.0)?;
                    let q0 = if effect_scales_b[ei] {
                        inla_core::scale_model_csc(&q0)?
                    } else {
                        q0
                    };
                    scale_csc_entries(&q0, tau)?
                }
                "crw1" => {
                    let tau = th.first().copied().unwrap_or(0.0).exp();
                    let positions: Vec<f64> = (0..n_e).map(|i| i as f64).collect();
                    let q0 = inla_core::crw1_precision_csc(&positions, 1.0)?;
                    let q0 = if effect_scales_b[ei] {
                        inla_core::scale_model_csc(&q0)?
                    } else {
                        q0
                    };
                    scale_csc_entries(&q0, tau)?
                }
                "crw2" => {
                    let tau = th.first().copied().unwrap_or(0.0).exp();
                    let positions: Vec<f64> = (0..n_e).map(|i| i as f64).collect();
                    let q0 = inla_core::crw2_precision_csc(&positions, 1.0, "simple")?;
                    let q0 = if effect_scales_b[ei] {
                        inla_core::scale_model_csc(&q0)?
                    } else {
                        q0
                    };
                    scale_csc_entries(&q0, tau)?
                }
                "besag" => {
                    let tau = th.first().copied().unwrap_or(0.0).exp();
                    let adj = adjs[ei]
                        .as_ref()
                        .ok_or_else(|| "besag missing adj".to_string())?;
                    if adj.len() != n_e {
                        return Err(format!(
                            "besag adj length {} != effect n {}",
                            adj.len(),
                            n_e
                        ));
                    }
                    // Scale the τ=1 structure first; then apply τ (otherwise
                    // scale.model cancels the precision hyperparameter).
                    let q0 = inla_core::besag_precision_csc(adj, 1.0)?;
                    let q0 = if effect_scales_b[ei] {
                        inla_core::scale_model_csc(&q0)?
                    } else {
                        q0
                    };
                    scale_csc_entries(&q0, tau)?
                }
                "fgn" => {
                    let order = effect_orders_u[ei];
                    if order == 3 || order == 4 {
                        if th.len() < 2 {
                            return Err("fgn approx needs [log_tau, H_intern]".into());
                        }
                        let tau = th[0].exp();
                        let hurst = inla_core::fgn_hurst_from_intern(th[1]);
                        let n_time = n_e / (order + 1);
                        let q0 =
                            inla_core::fgn_approx_precision_csc(n_time, hurst, 1.0, order, 1e8)?;
                        let q0 = if effect_scales_b[ei] {
                            inla_core::scale_model_csc(&q0)?
                        } else {
                            q0
                        };
                        scale_csc_entries(&q0, tau)?
                    } else {
                        if th.len() < 2 {
                            return Err("fgn needs [log_tau, logit_H]".into());
                        }
                        let tau = th[0].exp();
                        let hurst = 1.0 / (1.0 + (-th[1]).exp());
                        let q0 = inla_core::fgn_precision_csc(n_e, hurst, 1.0)?;
                        let q0 = if effect_scales_b[ei] {
                            inla_core::scale_model_csc(&q0)?
                        } else {
                            q0
                        };
                        scale_csc_entries(&q0, tau)?
                    }
                }
                other => return Err(format!("unsupported effect type: {other}")),
            };
            let _ = effect_scales_b[ei]; // scaling already applied above for random effects
            blocks.push(q);
        }
        inla_core::block_diag_csc(&blocks)
    };

    let log_prior_density = {
        let mut priors = Vec::new();
        for typ in &effect_types {
            let m = typ.to_lowercase();
            if m == "fixed" {
                continue;
            }
            match inla_core::HyperPriorStack::default_for_effect(&m) {
                Ok(s) => priors.extend(s.priors),
                Err(_) => priors.push(inla_core::PriorSpec::gaussian(0.0, 0.1)),
            }
        }
        let stack = inla_core::HyperPriorStack::new(priors);
        move |theta: &[f64]| -> f64 { stack.log_density(theta).unwrap_or(f64::NEG_INFINITY) }
    };

    let result = inla_core::run_inla_inference_a(
        &initial_theta,
        &build_prior,
        &log_prior_density,
        &obs,
        Some(&a),
        constr_opt.as_ref(),
        strategy,
        step_or_f0,
        &inla_core::MarginalOptions::default(),
        deterministic,
    )
    .map_err(Error::Other)?;

    let internal_marginals_hyperpar = marginals_to_r_list(&result.internal_marginals_hyperpar)?;

    Ok(list!(
        mode = result.mode,
        hessian = result.hessian,
        latent_means = result.latent_means,
        latent_variances = result.latent_variances,
        predictor_means = result.predictor_means,
        predictor_variances = result.predictor_variances,
        marginal_log_lik = result.marginal_log_lik,
        marginal_log_lik_gaussian = result.marginal_log_lik_gaussian,
        dic = result.dic,
        mean_deviance = result.mean_deviance,
        effective_params = result.effective_params,
        cpo_n_failures = result.cpo_n_failures as i32,
        node_weights = result.node_weights,
        internal_marginals_hyperpar = internal_marginals_hyperpar
    ))
}

#[extendr]
fn inla_rs_scale_model_csc(adj_list: List, tau: f64) -> std::result::Result<Robj, Error> {
    let adj = parse_adj_list_1based(&adj_list)?;
    let q = inla_core::besag_precision_csc(&adj, tau).map_err(Error::Other)?;
    let qs = inla_core::scale_model_csc(&q).map_err(Error::Other)?;
    csc_to_dgcmatrix(&qs)
}

/// Evaluate named prior on internal θ. `param` may be empty (use defaults).
#[extendr]
fn inla_rs_prior_log_density(
    name: &str,
    param: Vec<f64>,
    theta: Vec<f64>,
) -> std::result::Result<f64, Error> {
    let spec = inla_core::PriorSpec::from_name_params(name, &param).map_err(Error::Other)?;
    spec.log_density(&theta).map_err(Error::Other)
}

/// Default hyperpriors for an effect model: list(names=..., params=list of numeric vectors).
#[extendr]
fn inla_rs_default_hyper_priors(model: &str) -> std::result::Result<List, Error> {
    let stack = inla_core::HyperPriorStack::default_for_effect(model).map_err(Error::Other)?;
    let pairs = stack.to_names_params();
    let names: Vec<String> = pairs.iter().map(|(n, _)| n.clone()).collect();
    let mut param_items: Vec<Robj> = Vec::with_capacity(pairs.len());
    for (_, p) in &pairs {
        param_items.push(Robj::from(p.clone()));
    }
    Ok(list!(
        names = names,
        params = List::from_values(param_items)
    ))
}

/// Sum log-density for a prior stack. `param_list` is a list of numeric vectors.
#[extendr]
fn inla_rs_hyper_prior_stack_log_density(
    names: Vec<String>,
    param_list: List,
    theta: Vec<f64>,
) -> std::result::Result<f64, Error> {
    if param_list.len() != names.len() {
        return Err(Error::Other(
            "param_list length must match names length".into(),
        ));
    }
    let mut params: Vec<Vec<f64>> = Vec::with_capacity(names.len());
    for item in param_list.values() {
        let v: Vec<f64> = item
            .as_real_vector()
            .ok_or_else(|| Error::Other("each params entry must be numeric".into()))?
            .iter()
            .map(|&x| x)
            .collect();
        params.push(v);
    }
    let stack =
        inla_core::HyperPriorStack::from_names_params(&names, &params).map_err(Error::Other)?;
    stack.log_density(&theta).map_err(Error::Other)
}

extendr_module! {
    mod inla_rs;
    fn inla_rs_read_mesh;
    fn inla_rs_ar1_precision;
    fn inla_rs_ar1_precision_csc_dgcmatrix;
    fn inla_rs_rw1_precision_csc;
    fn inla_rs_rw2_precision_csc;
    fn inla_rs_rw1_cyclic_precision_csc;
    fn inla_rs_rw2_cyclic_precision_csc;
    fn inla_rs_seasonal_precision_csc;
    fn inla_rs_two_diid_precision_csc;
    fn inla_rs_iid_precision_csc;
    fn inla_rs_arp_precision_csc;
    fn inla_rs_matern2d_precision_csc;
    fn inla_rs_rw2d_precision_csc;
    fn inla_rs_besag_precision_csc;
    fn inla_rs_bym_precision_csc;
    fn inla_rs_bym2_precision_csc;
    fn inla_rs_spde_precision_mesh_csc;
    fn inla_rs_fem_blocks_mesh;
    fn inla_rs_spde_projector_csc;
    fn inla_rs_run_spde;
    fn inla_rs_crw1_precision_csc;
    fn inla_rs_crw2_precision_csc;
    fn inla_rs_fgn_precision_csc;
    fn inla_rs_fgn_approx_precision_csc;
    fn inla_rs_run_inla_inference;
    fn inla_rs_run_inla_structured;
    fn inla_rs_scale_model_csc;
    fn inla_rs_prior_log_density;
    fn inla_rs_default_hyper_priors;
    fn inla_rs_hyper_prior_stack_log_density;
}
