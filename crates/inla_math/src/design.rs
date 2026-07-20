//! Design-matrix helpers: A-projection, block-diagonal assembly, scale.model.

use sprs::{CsMat, TriMatI};

use crate::sparse::CscMatrix;

/// Identity (or scaled identity) CSC matrix of size `n × n`.
pub fn identity_csc(n: usize, value: f64) -> Result<CscMatrix, String> {
    if n == 0 {
        return Err("identity matrix size must be > 0".to_string());
    }
    if !value.is_finite() {
        return Err("identity value must be finite".to_string());
    }
    let mut tri = TriMatI::<f64, usize>::with_capacity((n, n), n);
    for i in 0..n {
        tri.add_triplet(i, i, value);
    }
    Ok(tri.to_csc())
}

/// Assemble a block-diagonal CSC from square blocks.
pub fn block_diag_csc(blocks: &[CscMatrix]) -> Result<CscMatrix, String> {
    if blocks.is_empty() {
        return Err("block_diag requires at least one block".to_string());
    }
    let mut nrow = 0usize;
    let mut nnz = 0usize;
    for b in blocks {
        if b.rows() != b.cols() {
            return Err("block_diag blocks must be square".to_string());
        }
        nrow += b.rows();
        nnz += b.nnz();
    }
    let mut tri = TriMatI::<f64, usize>::with_capacity((nrow, nrow), nnz);
    let mut offset = 0usize;
    for b in blocks {
        for (col, colvec) in b.outer_iterator().enumerate() {
            for (row, &val) in colvec.iter() {
                if val != 0.0 {
                    tri.add_triplet(offset + row, offset + col, val);
                }
            }
        }
        offset += b.rows();
    }
    Ok(tri.to_csc())
}

/// y = A x for CSC A (`nrow × ncol`).
pub fn matvec_csc(a: &CscMatrix, x: &[f64]) -> Result<Vec<f64>, String> {
    if a.cols() != x.len() {
        return Err(format!(
            "matvec: A has {} cols but x has length {}",
            a.cols(),
            x.len()
        ));
    }
    let mut y = vec![0.0; a.rows()];
    for (col, colvec) in a.outer_iterator().enumerate() {
        let xv = x[col];
        if xv == 0.0 {
            continue;
        }
        for (row, &val) in colvec.iter() {
            y[row] += val * xv;
        }
    }
    Ok(y)
}

/// z = Aᵀ y for CSC A.
pub fn matvec_transpose_csc(a: &CscMatrix, y: &[f64]) -> Result<Vec<f64>, String> {
    if a.rows() != y.len() {
        return Err(format!(
            "matvec_transpose: A has {} rows but y has length {}",
            a.rows(),
            y.len()
        ));
    }
    let mut z = vec![0.0; a.cols()];
    for (col, colvec) in a.outer_iterator().enumerate() {
        let mut s = 0.0;
        for (row, &val) in colvec.iter() {
            s += val * y[row];
        }
        z[col] = s;
    }
    Ok(z)
}

/// Compute Aᵀ diag(d) A (symmetric), returned as CSC.
pub fn at_diag_a(a: &CscMatrix, d: &[f64]) -> Result<CscMatrix, String> {
    if a.rows() != d.len() {
        return Err("at_diag_a: d length must equal A.nrows".to_string());
    }
    let a_csr: CsMat<f64> = a.to_csr();
    let n = a.cols();
    let mut tri = TriMatI::<f64, usize>::new((n, n));
    for i in 0..a.rows() {
        let di = d[i];
        if di == 0.0 || !di.is_finite() {
            continue;
        }
        let row = a_csr
            .outer_view(i)
            .ok_or_else(|| "CSR row view failed".to_string())?;
        let entries: Vec<(usize, f64)> = row.iter().map(|(j, &v)| (j, v)).collect();
        for &(j, aj) in &entries {
            for &(k, ak) in &entries {
                let v = di * aj * ak;
                if v != 0.0 {
                    tri.add_triplet(j, k, v);
                }
            }
        }
    }
    Ok(tri.to_csc())
}

/// C = A + B (same shape).
pub fn add_csc(a: &CscMatrix, b: &CscMatrix) -> Result<CscMatrix, String> {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return Err("add_csc: shape mismatch".to_string());
    }
    let mut tri = TriMatI::<f64, usize>::with_capacity((a.rows(), a.cols()), a.nnz() + b.nnz());
    for (col, colvec) in a.outer_iterator().enumerate() {
        for (row, &val) in colvec.iter() {
            tri.add_triplet(row, col, val);
        }
    }
    for (col, colvec) in b.outer_iterator().enumerate() {
        for (row, &val) in colvec.iter() {
            tri.add_triplet(row, col, val);
        }
    }
    Ok(tri.to_csc())
}

/// Scale an intrinsic GMRF precision so the geometric mean of the marginal
/// variances is one — matching R-INLA `inla.scale.model` /
/// `inla.rw(..., scale.model=TRUE)`:
///
/// ```text
/// fac = exp(mean(log(diag(ginv(Q)))))
/// Q_scaled = fac * Q
/// ```
///
/// where `ginv` is the Moore–Penrose inverse (zero eigenvalues dropped).
pub fn scale_model_csc(q: &CscMatrix) -> Result<CscMatrix, String> {
    if q.rows() != q.cols() {
        return Err("scale_model: Q must be square".to_string());
    }
    let n = q.rows();
    if n == 0 {
        return Err("scale_model: empty Q".to_string());
    }
    if n == 1 {
        // R-INLA: singleton component → Q = 1
        return identity_csc(1, 1.0);
    }

    let a = crate::ldlt::csc_to_dense(q).map_err(|e| e.to_string())?;
    let (evals, evecs) = crate::integration::jacobi_eigen(&a, n, 500)?;
    let max_lam = evals.iter().copied().fold(0.0_f64, f64::max).abs().max(1.0);
    // Same spirit as inla.ginv / MASS::ginv default tol
    let tol = f64::EPSILON.sqrt() * max_lam;

    // diag(ginv)_i = Σ_{k: λ_k > tol} V_{ik}^2 / λ_k
    // (evecs stored column-wise: evecs[row * n + col])
    let mut log_sum = 0.0;
    let mut count = 0usize;
    for i in 0..n {
        let mut vi = 0.0;
        for k in 0..n {
            if evals[k] > tol {
                let vik = evecs[i * n + k];
                vi += vik * vik / evals[k];
            }
        }
        if vi > 0.0 && vi.is_finite() {
            log_sum += vi.ln();
            count += 1;
        }
    }
    if count == 0 {
        return Err("scale_model: ginv has no positive diagonal".to_string());
    }
    let fac = (log_sum / count as f64).exp();
    if fac <= 0.0 || !fac.is_finite() {
        return Err(format!("scale_model: invalid scale factor {fac}"));
    }
    scale_csc(q, fac)
}

/// Diagonal approximation to diag(A Σ Aᵀ) using only diag(Σ)=`var_x`.
pub fn predictor_variances_diag(a: &CscMatrix, var_x: &[f64]) -> Result<Vec<f64>, String> {
    if a.cols() != var_x.len() {
        return Err("predictor_variances_diag: size mismatch".to_string());
    }
    let mut out = vec![0.0; a.rows()];
    for (col, colvec) in a.outer_iterator().enumerate() {
        let v = var_x[col];
        for (row, &val) in colvec.iter() {
            out[row] += val * val * v;
        }
    }
    Ok(out)
}

/// Multiply all entries of a CSC matrix by a positive scalar.
pub fn scale_csc(q: &CscMatrix, scale: f64) -> Result<CscMatrix, String> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err("scale_csc: scale must be finite and > 0".to_string());
    }
    let mut tri = TriMatI::<f64, usize>::with_capacity((q.rows(), q.cols()), q.nnz());
    for (col, colvec) in q.outer_iterator().enumerate() {
        for (row, &val) in colvec.iter() {
            tri.add_triplet(row, col, val * scale);
        }
    }
    Ok(tri.to_csc())
}

/// Build CSC from 0-based triplets (i, j, x).
pub fn csc_from_triplets_0based(
    nrow: usize,
    ncol: usize,
    rows: &[usize],
    cols: &[usize],
    vals: &[f64],
) -> Result<CscMatrix, String> {
    if rows.len() != cols.len() || rows.len() != vals.len() {
        return Err("triplet length mismatch".to_string());
    }
    let mut tri = TriMatI::<f64, usize>::with_capacity((nrow, ncol), vals.len());
    for k in 0..vals.len() {
        let r = rows[k];
        let c = cols[k];
        if r >= nrow || c >= ncol {
            return Err(format!(
                "triplet index ({r},{c}) out of bounds ({nrow}x{ncol})"
            ));
        }
        tri.add_triplet(r, c, vals[k]);
    }
    Ok(tri.to_csc())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_diag_two_identities() {
        let a = identity_csc(2, 2.0).unwrap();
        let b = identity_csc(3, 3.0).unwrap();
        let q = block_diag_csc(&[a, b]).unwrap();
        assert_eq!(q.rows(), 5);
        assert_eq!(q.nnz(), 5);
    }

    #[test]
    fn at_diag_a_selection() {
        let a = identity_csc(2, 1.0).unwrap();
        let m = at_diag_a(&a, &[4.0, 9.0]).unwrap();
        let d = crate::ldlt::csc_to_dense(&m).unwrap();
        assert!((d[0] - 4.0).abs() < 1e-12);
        assert!((d[3] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn scale_model_identity_geom_mean_near_one() {
        // Proper diagonal Q = I → ginv = I → geom mean of diag = 1 → scale = 1
        let q = identity_csc(4, 1.0).unwrap();
        let qs = scale_model_csc(&q).unwrap();
        let a = crate::ldlt::csc_to_dense(&qs).unwrap();
        for i in 0..4 {
            assert!((a[i * 4 + i] - 1.0).abs() < 1e-8);
        }
    }
}
