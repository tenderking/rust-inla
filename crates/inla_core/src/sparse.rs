use crate::ar1::ar1_precision;
use crate::fmesher::SparseTriplet;
use sprs::{CsMat, TriMatI};

pub type CscMatrix = CsMat<f64>;

pub fn sparse_triplet_to_csc(st: &SparseTriplet) -> CscMatrix {
    let mut tri = TriMatI::<f64, usize>::with_capacity((st.rows, st.cols), st.entries.len());
    for &(r, c, v) in &st.entries {
        tri.add_triplet(r, c, v);
    }
    tri.to_csc()
}


#[derive(Debug, Clone, PartialEq)]
pub struct CscForR {
    pub nrow: usize,
    pub ncol: usize,
    pub p: Vec<i32>,
    pub i: Vec<i32>,
    pub x: Vec<f64>,
}

pub fn triplets_to_csc(
    nrow: usize,
    ncol: usize,
    i_1based: &[usize],
    j_1based: &[usize],
    x: &[f64],
) -> Result<CscMatrix, String> {
    if i_1based.len() != j_1based.len() || i_1based.len() != x.len() {
        return Err("triplet vectors i/j/x must have the same length".to_string());
    }
    let mut tri = TriMatI::<f64, usize>::with_capacity((nrow, ncol), x.len());
    for k in 0..x.len() {
        let r = i_1based[k];
        let c = j_1based[k];
        if r == 0 || c == 0 {
            return Err("triplet indices must be 1-based positive integers".to_string());
        }
        if r > nrow || c > ncol {
            return Err("triplet index exceeds matrix dimensions".to_string());
        }
        tri.add_triplet(r - 1, c - 1, x[k]);
    }
    Ok(tri.to_csc())
}

pub fn ar1_precision_csc(n: usize, rho: f64, tau: f64) -> Result<CscMatrix, String> {
    let q = ar1_precision(n, rho, tau)?;
    triplets_to_csc(q.n, q.n, &q.i, &q.j, &q.x)
}

pub fn csc_for_r_dgcmatrix(csc: &CscMatrix) -> Result<CscForR, String> {
    let p = csc
        .indptr()
        .raw_storage()
        .iter()
        .copied()
        .map(|v| i32::try_from(v).map_err(|_| "column pointer exceeds i32 range".to_string()))
        .collect::<Result<Vec<_>, _>>()?;

    let i = csc
        .indices()
        .iter()
        .copied()
        .map(|v| i32::try_from(v).map_err(|_| "row index exceeds i32 range".to_string()))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(CscForR {
        nrow: csc.rows(),
        ncol: csc.cols(),
        p,
        i,
        x: csc.data().to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_ar1_triplets_to_csc() {
        let csc = ar1_precision_csc(5, 0.7, 1.0).expect("build csc");
        assert_eq!(csc.rows(), 5);
        assert_eq!(csc.cols(), 5);
        assert_eq!(csc.nnz(), 13);
        assert_eq!(csc.indptr().raw_storage(), &[0, 2, 5, 8, 11, 13]);
    }

    #[test]
    fn exports_dgcmatrix_slots() {
        let csc = ar1_precision_csc(4, 0.5, 1.0).expect("build csc");
        let slots = csc_for_r_dgcmatrix(&csc).expect("export slots");
        assert_eq!(slots.nrow, 4);
        assert_eq!(slots.ncol, 4);
        assert_eq!(slots.p.len(), 5);
        assert_eq!(slots.i.len(), slots.x.len());
    }
}
