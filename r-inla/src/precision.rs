//! Latent-model precision matrix exports for R (`dgCMatrix`).

use crate::convert::{csc_to_dgcmatrix, parse_adj_list_1based};
use extendr_api::prelude::*;
use inla_core::ar1::ar1_precision;
use inla_core::sparse::ar1_precision_csc;

fn as_usize(n: i32, name: &str) -> std::result::Result<usize, Error> {
    usize::try_from(n).map_err(|_| Error::Other(format!("{name} must be non-negative")))
}

fn export_csc(
    build: impl FnOnce() -> std::result::Result<inla_core::CscMatrix, String>,
) -> std::result::Result<Robj, Error> {
    csc_to_dgcmatrix(&build().map_err(Error::Other)?)
}

#[extendr]
fn inla_rs_ar1_precision(n: i32, rho: f64, tau: f64) -> std::result::Result<List, Error> {
    let n_usize = as_usize(n, "n")?;
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
    let n = as_usize(n, "n")?;
    export_csc(|| ar1_precision_csc(n, rho, tau))
}

#[extendr]
fn inla_rs_rw1_precision_csc(n: i32, tau: f64) -> std::result::Result<Robj, Error> {
    let n = as_usize(n, "n")?;
    export_csc(|| inla_core::rw1_precision_csc(n, tau))
}

#[extendr]
fn inla_rs_rw2_precision_csc(n: i32, tau: f64) -> std::result::Result<Robj, Error> {
    let n = as_usize(n, "n")?;
    export_csc(|| inla_core::rw2_precision_csc(n, tau))
}

#[extendr]
fn inla_rs_rw1_cyclic_precision_csc(n: i32, tau: f64) -> std::result::Result<Robj, Error> {
    let n = as_usize(n, "n")?;
    export_csc(|| inla_core::rw1_cyclic_precision_csc(n, tau))
}

#[extendr]
fn inla_rs_rw2_cyclic_precision_csc(n: i32, tau: f64) -> std::result::Result<Robj, Error> {
    let n = as_usize(n, "n")?;
    export_csc(|| inla_core::rw2_cyclic_precision_csc(n, tau))
}

#[extendr]
fn inla_rs_seasonal_precision_csc(
    n: i32,
    s: i32,
    tau: f64,
    cyclic: bool,
) -> std::result::Result<Robj, Error> {
    let n = as_usize(n, "n")?;
    let s = as_usize(s, "s")?;
    export_csc(|| inla_core::seasonal_precision_csc(n, s, tau, cyclic))
}

#[extendr]
fn inla_rs_two_diid_precision_csc(
    n_pairs: i32,
    rho: f64,
    tau: f64,
) -> std::result::Result<Robj, Error> {
    let n_pairs = as_usize(n_pairs, "n_pairs")?;
    export_csc(|| inla_core::two_diid_precision_csc(n_pairs, rho, tau))
}

#[extendr]
fn inla_rs_iid_precision_csc(n: i32, tau: f64) -> std::result::Result<Robj, Error> {
    let n = as_usize(n, "n")?;
    export_csc(|| inla_core::iid_precision_csc(n, tau))
}

#[extendr]
fn inla_rs_arp_precision_csc(n: i32, pacf: Vec<f64>, tau: f64) -> std::result::Result<Robj, Error> {
    let n = as_usize(n, "n")?;
    export_csc(|| inla_core::arp_precision_csc(n, &pacf, tau))
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
    let nrow = as_usize(nrow, "nrow")?;
    let ncol = as_usize(ncol, "ncol")?;
    let nu = as_usize(nu, "nu")?;
    export_csc(|| inla_core::matern2d_precision_csc(nrow, ncol, nu, range, prec, cyclic))
}

#[extendr]
fn inla_rs_rw2d_precision_csc(
    nrow: i32,
    ncol: i32,
    tau: f64,
    cyclic: bool,
    bvalue_zero: bool,
) -> std::result::Result<Robj, Error> {
    let nrow = as_usize(nrow, "nrow")?;
    let ncol = as_usize(ncol, "ncol")?;
    export_csc(|| inla_core::rw2d_precision_csc(nrow, ncol, tau, cyclic, bvalue_zero))
}

#[extendr]
fn inla_rs_besag_precision_csc(adj_list: List, tau: f64) -> std::result::Result<Robj, Error> {
    let adj = parse_adj_list_1based(&adj_list)?;
    export_csc(|| inla_core::besag_precision_csc(&adj, tau))
}

#[extendr]
fn inla_rs_bym_precision_csc(
    adj_list: List,
    tau_spatial: f64,
    tau_iid: f64,
) -> std::result::Result<Robj, Error> {
    let adj = parse_adj_list_1based(&adj_list)?;
    export_csc(|| inla_core::bym_precision_csc(&adj, tau_spatial, tau_iid))
}

#[extendr]
fn inla_rs_bym2_precision_csc(
    adj_list: List,
    tau: f64,
    phi: f64,
) -> std::result::Result<Robj, Error> {
    let adj = parse_adj_list_1based(&adj_list)?;
    export_csc(|| inla_core::bym2_precision_csc(&adj, tau, phi))
}

#[extendr]
fn inla_rs_crw1_precision_csc(positions: Vec<f64>, tau: f64) -> std::result::Result<Robj, Error> {
    export_csc(|| inla_core::crw1_precision_csc(&positions, tau))
}

#[extendr]
fn inla_rs_crw2_precision_csc(
    positions: Vec<f64>,
    tau: f64,
    layout: &str,
) -> std::result::Result<Robj, Error> {
    export_csc(|| inla_core::crw2_precision_csc(&positions, tau, layout))
}

#[extendr]
fn inla_rs_fgn_precision_csc(n: i32, hurst: f64, tau: f64) -> std::result::Result<Robj, Error> {
    let n = as_usize(n, "n")?;
    export_csc(|| inla_core::fgn_precision_csc(n, hurst, tau))
}

#[extendr]
fn inla_rs_fgn_approx_precision_csc(
    n: i32,
    hurst: f64,
    tau: f64,
    order: i32,
    prec_eps: f64,
) -> std::result::Result<Robj, Error> {
    let n = as_usize(n, "n")?;
    let order = as_usize(order, "order")?;
    export_csc(|| inla_core::fgn_approx_precision_csc(n, hurst, tau, order, prec_eps))
}

#[extendr]
fn inla_rs_scale_model_csc(adj_list: List, tau: f64) -> std::result::Result<Robj, Error> {
    let adj = parse_adj_list_1based(&adj_list)?;
    export_csc(|| {
        let q = inla_core::besag_precision_csc(&adj, tau)?;
        inla_core::scale_model_csc(&q)
    })
}

extendr_module! {
    mod precision;
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
    fn inla_rs_crw1_precision_csc;
    fn inla_rs_crw2_precision_csc;
    fn inla_rs_fgn_precision_csc;
    fn inla_rs_fgn_approx_precision_csc;
    fn inla_rs_scale_model_csc;
}
