use sprs::{CsMat, TriMatI};

pub type CscMatrix = CsMat<f64>;

/// Build CSC from 0-based (row, col, value) triplets.
pub fn sparse_from_triplets(
    rows: usize,
    cols: usize,
    entries: &[(usize, usize, f64)],
) -> CscMatrix {
    let mut tri = TriMatI::<f64, usize>::with_capacity((rows, cols), entries.len());
    for &(r, c, v) in entries {
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

/// Build CSC from 1-based triplet index vectors (R/Matrix convention).
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
    fn converts_triplets_to_csc() {
        let csc = triplets_to_csc(3, 3, &[1, 2, 3], &[1, 2, 3], &[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(csc.rows(), 3);
        assert_eq!(csc.nnz(), 3);
    }

    #[test]
    fn exports_dgcmatrix_slots() {
        let csc = sparse_from_triplets(2, 2, &[(0, 0, 1.0), (1, 1, 2.0)]);
        let slots = csc_for_r_dgcmatrix(&csc).unwrap();
        assert_eq!(slots.nrow, 2);
        assert_eq!(slots.i.len(), slots.x.len());
    }
}
