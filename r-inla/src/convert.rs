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

pub(crate) fn parse_effect_positions(
    lists: &List,
    ns: &[usize],
) -> std::result::Result<Vec<Option<Vec<f64>>>, Error> {
    if lists.len() != ns.len() {
        return Err(Error::Other(
            "effect_positions length must match number of effects".to_string(),
        ));
    }
    let mut out = Vec::with_capacity(ns.len());
    for (ei, item) in lists.values().enumerate() {
        let vals: Vec<f64> = if let Some(v) = item.as_real_vector() {
            v
        } else if let Some(v) = item.as_integer_vector() {
            v.into_iter().map(|i| i as f64).collect()
        } else {
            Vec::new()
        };
        if vals.len() == ns[ei] && vals.iter().all(|x| x.is_finite()) {
            out.push(Some(vals));
        } else {
            out.push(None);
        }
    }
    Ok(out)
}

pub(crate) fn parse_adj_list_1based(
    adj_list: &List,
) -> std::result::Result<Vec<Vec<usize>>, Error> {
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

pub(crate) fn posterior_q_slots(
    q: &Option<inla_core::CscMatrix>,
) -> std::result::Result<(Vec<i32>, Vec<i32>, Vec<f64>, i32), Error> {
    let Some(q) = q else {
        return Ok((Vec::new(), Vec::new(), Vec::new(), 0));
    };
    let slots = csc_for_r_dgcmatrix(q).map_err(Error::Other)?;
    Ok((slots.i, slots.p, slots.x, slots.nrow as i32))
}

pub(crate) fn csc_from_r_slots(
    n: usize,
    i: &[i32],
    p: &[i32],
    x: &[f64],
) -> std::result::Result<inla_core::CscMatrix, Error> {
    if p.len() != n + 1 {
        return Err(Error::Other(format!(
            "CSC p length {} != n+1={}",
            p.len(),
            n + 1
        )));
    }
    let mut rows = Vec::with_capacity(x.len());
    let mut cols = Vec::with_capacity(x.len());
    let mut vals = Vec::with_capacity(x.len());
    for col in 0..n {
        let start = usize::try_from(p[col]).map_err(|_| Error::Other("CSC p".into()))?;
        let end = usize::try_from(p[col + 1]).map_err(|_| Error::Other("CSC p".into()))?;
        if end > i.len() || end > x.len() || start > end {
            return Err(Error::Other("CSC pointer out of range".into()));
        }
        for k in start..end {
            rows.push(usize::try_from(i[k]).map_err(|_| Error::Other("CSC i".into()))?);
            cols.push(col);
            vals.push(x[k]);
        }
    }
    inla_core::csc_from_triplets_0based(n, n, &rows, &cols, &vals).map_err(Error::Other)
}

pub(crate) fn marginals_to_r_list(
    ms: &[inla_core::Marginal1D],
) -> std::result::Result<List, Error> {
    let mut items: Vec<Robj> = Vec::with_capacity(ms.len());
    for m in ms {
        items.push(marginal_to_r_matrix(m)?);
    }
    Ok(List::from_values(items))
}
