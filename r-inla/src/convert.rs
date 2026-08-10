//! R ↔ Rust conversion helpers shared by the extendr wrappers.

use extendr_api::prelude::*;
use inla_core::sparse::csc_for_r_dgcmatrix;

pub(crate) fn map_r_error(e: Error) -> Error {
    Error::Other(e.to_string())
}

pub(crate) fn csc_to_dgcmatrix(
    csc: &inla_core::sparse::CscMatrix,
) -> std::result::Result<Robj, Error> {
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

pub(crate) fn parse_adj_list_1based(adj_list: &List) -> std::result::Result<Vec<Vec<usize>>, Error> {
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

pub(crate) fn scale_csc_entries(
    q: &inla_core::CscMatrix,
    scale: f64,
) -> std::result::Result<inla_core::CscMatrix, String> {
    inla_core::scale_csc(q, scale)
}

pub(crate) fn marginal_to_r_matrix(m: &inla_core::Marginal1D) -> std::result::Result<Robj, Error> {
    let n = m.x.len();
    let mut data = Vec::with_capacity(n * 2);
    // column-major: first column x, second column y
    data.extend_from_slice(&m.x);
    data.extend_from_slice(&m.y);
    let mut mat = r!(data);
    mat.set_attrib(sym!(dim), r!([n as i32, 2i32]))?;
    Ok(mat)
}

pub(crate) fn marginals_to_r_list(ms: &[inla_core::Marginal1D]) -> std::result::Result<List, Error> {
    let mut items: Vec<Robj> = Vec::with_capacity(ms.len());
    for m in ms {
        items.push(marginal_to_r_matrix(m)?);
    }
    Ok(List::from_values(items))
}
