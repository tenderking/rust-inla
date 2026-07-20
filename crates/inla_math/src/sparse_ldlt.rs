//! Sparse LDLᵀ via faer (no `csc_to_dense` on the hot path).

use std::sync::Arc;

use faer::{Conj, Par, Side};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::ldlt::factor::LdltRegularization;
use faer::prelude::*;
use faer::sparse::linalg::SupernodalThreshold;
use faer::sparse::linalg::cholesky::{
    self, CholeskySymbolicParams, LdltRef, SymbolicCholesky, SymbolicCholeskyRaw,
};
use faer::sparse::{SparseColMat, Triplet};

use crate::error::MathError;
use crate::scratch::LdltScratch;
use crate::sparse::CscMatrix;

/// Owned sparse LDLᵀ factor (symbolic pattern + numeric values + D diagonal).
#[derive(Clone)]
pub struct SparseLdltFactor {
    pub n: usize,
    /// Diagonal of D in `Q = L D Lᵀ` (for log|Q| = Σ log|Dᵢ|).
    pub d: Vec<f64>,
    symbolic: Arc<SymbolicCholesky<usize>>,
    l_values: Vec<f64>,
}

impl std::fmt::Debug for SparseLdltFactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SparseLdltFactor")
            .field("n", &self.n)
            .field("nnz_L", &self.l_values.len())
            .finish()
    }
}

impl PartialEq for SparseLdltFactor {
    fn eq(&self, other: &Self) -> bool {
        self.n == other.n && self.d == other.d && self.l_values == other.l_values
    }
}

impl SparseLdltFactor {
    /// Shared symbolic pattern (for cache / refactorize checks).
    pub fn symbolic_arc(&self) -> Arc<SymbolicCholesky<usize>> {
        Arc::clone(&self.symbolic)
    }
}

fn csc_to_faer(q: &CscMatrix) -> Result<SparseColMat<usize, f64>, MathError> {
    let n = q.rows();
    if q.cols() != n {
        return Err(MathError::NotSquare {
            rows: q.rows(),
            cols: q.cols(),
        });
    }
    // Lower triangle only (incl. diagonal), including structural zeros.
    // faer simplicial LDLᵀ with Side::Lower expects this; passing a full
    // symmetric matrix makes A_nnz (symbolic) disagree with permute scratch.
    let mut trips = Vec::with_capacity(q.nnz());
    for (col, colvec) in q.outer_iterator().enumerate() {
        for (row, &val) in colvec.iter() {
            if row >= col {
                trips.push(Triplet::new(row, col, val));
            }
        }
    }
    SparseColMat::try_new_from_triplets(n, n, &trips).map_err(|e| match e {
        faer::sparse::CreationError::Generic(faer::sparse::FaerError::OutOfMemory) => {
            MathError::OutOfMemory
        }
        faer::sparse::CreationError::OutOfBounds { .. } => {
            MathError::IndexOutOfBounds("faer triplet index out of bounds")
        }
        other => MathError::Message(format!("faer sparse create: {other:?}")),
    })
}

fn extract_d(symbolic: &SymbolicCholesky<usize>, l_values: &[f64]) -> Result<Vec<f64>, MathError> {
    let n = symbolic.nrows();
    let mut d = vec![0.0; n];
    match symbolic.raw() {
        SymbolicCholeskyRaw::Simplicial(sym) => {
            let col_ptr = sym.col_ptr();
            let row_idx = sym.row_idx();
            for j in 0..n {
                let start = col_ptr[j];
                let end = col_ptr[j + 1];
                let mut found = false;
                for p in start..end {
                    if row_idx[p] == j {
                        let dj = l_values[p];
                        if !dj.is_finite() || dj.abs() < 1e-14 {
                            return Err(MathError::Singular);
                        }
                        d[j] = dj;
                        found = true;
                        break;
                    }
                }
                if !found {
                    return Err(MathError::Singular);
                }
            }
        }
        SymbolicCholeskyRaw::Supernodal(_) => {
            return Err(MathError::Message(
                "sparse LDLᵀ: expected simplicial factor (FORCE_SIMPLICIAL)".into(),
            ));
        }
    }
    Ok(d)
}

fn symbolic_params() -> CholeskySymbolicParams<'static> {
    CholeskySymbolicParams {
        supernodal_flop_ratio_threshold: SupernodalThreshold::FORCE_SIMPLICIAL,
        ..Default::default()
    }
}

/// Sparse LDLᵀ of a CSC precision matrix (AMD ordering, simplicial numeric).
///
/// When `scratch` already holds a matching CSC pattern, only the numeric
/// factor is recomputed (Symbolica-style factorize-once / evaluate-many).
pub fn factorize_sparse(
    q: &CscMatrix,
    scratch: &mut LdltScratch,
) -> Result<SparseLdltFactor, MathError> {
    if scratch.pattern_matches(q) {
        let symbolic = scratch
            .symbolic_cache
            .as_ref()
            .expect("pattern_matches implies cache")
            .symbolic
            .clone();
        return refactorize_numeric_arc(symbolic, q, scratch);
    }

    let a = csc_to_faer(q)?;
    let symbolic = cholesky::factorize_symbolic_cholesky(
        a.symbolic(),
        Side::Lower,
        Default::default(),
        symbolic_params(),
    )
    .map_err(|e| match e {
        faer::sparse::FaerError::OutOfMemory => MathError::OutOfMemory,
        other => MathError::Message(format!("sparse symbolic LDLᵀ failed: {other:?}")),
    })?;

    let len = symbolic.len_val();
    scratch.ensure_l_values(len);
    let mut l_values = vec![0.0; len];
    let mut mem = MemBuffer::new(
        symbolic.factorize_numeric_ldlt_scratch::<f64>(Par::Seq, Default::default()),
    );
    symbolic
        .factorize_numeric_ldlt(
            &mut l_values,
            a.rb(),
            Side::Lower,
            LdltRegularization::default(),
            Par::Seq,
            MemStack::new(&mut mem),
            Default::default(),
        )
        .map_err(|_| MathError::NotPositiveDefinite)?;

    let d = extract_d(&symbolic, &l_values)?;
    let symbolic = Arc::new(symbolic);
    scratch.store_pattern(q, Arc::clone(&symbolic));
    Ok(SparseLdltFactor {
        n: symbolic.nrows(),
        d,
        symbolic,
        l_values,
    })
}

pub fn sparse_solve_in_place(
    factor: &SparseLdltFactor,
    x: &mut [f64],
    _scratch: &mut LdltScratch,
) -> Result<(), MathError> {
    if x.len() != factor.n {
        return Err(MathError::DimensionMismatch {
            context: "sparse LDLᵀ solve RHS",
            expected: factor.n,
            got: x.len(),
        });
    }
    let mut rhs = Mat::from_fn(factor.n, 1, |i, _| x[i]);
    let mut mem =
        MemBuffer::new(factor.symbolic.solve_in_place_scratch::<f64>(1, Par::Seq));
    let ldlt = LdltRef::new(factor.symbolic.as_ref(), &factor.l_values);
    ldlt.solve_in_place_with_conj(Conj::No, rhs.as_mut(), Par::Seq, MemStack::new(&mut mem));
    for i in 0..factor.n {
        x[i] = rhs[(i, 0)];
    }
    Ok(())
}

pub fn sparse_diagonal_inverse(
    factor: &SparseLdltFactor,
    _scratch: &mut LdltScratch,
) -> Result<Vec<f64>, MathError> {
    let n = factor.n;
    let mut diag = vec![0.0; n];
    let mut mem =
        MemBuffer::new(factor.symbolic.solve_in_place_scratch::<f64>(1, Par::Seq));
    let ldlt = LdltRef::new(factor.symbolic.as_ref(), &factor.l_values);
    for i in 0..n {
        let mut rhs = Mat::from_fn(n, 1, |r, _| if r == i { 1.0 } else { 0.0 });
        ldlt.solve_in_place_with_conj(Conj::No, rhs.as_mut(), Par::Seq, MemStack::new(&mut mem));
        diag[i] = rhs[(i, 0)];
    }
    Ok(diag)
}

/// Symbolic pattern only (factorize-once / numeric-many).
pub fn symbolic_pattern(q: &CscMatrix) -> Result<Arc<SymbolicCholesky<usize>>, MathError> {
    let a = csc_to_faer(q)?;
    let symbolic = cholesky::factorize_symbolic_cholesky(
        a.symbolic(),
        Side::Lower,
        Default::default(),
        symbolic_params(),
    )
    .map_err(|e| match e {
        faer::sparse::FaerError::OutOfMemory => MathError::OutOfMemory,
        other => MathError::Message(format!("sparse symbolic LDLᵀ failed: {other:?}")),
    })?;
    Ok(Arc::new(symbolic))
}

/// Numeric refactorize reusing a shared symbolic pattern (same sparsity as `q`).
pub fn refactorize_numeric_arc(
    symbolic: Arc<SymbolicCholesky<usize>>,
    q: &CscMatrix,
    _scratch: &mut LdltScratch,
) -> Result<SparseLdltFactor, MathError> {
    let a = csc_to_faer(q)?;
    if a.nrows() != symbolic.nrows() {
        return Err(MathError::DimensionMismatch {
            context: "sparse refactorize",
            expected: symbolic.nrows(),
            got: a.nrows(),
        });
    }
    let len = symbolic.len_val();
    let mut l_values = vec![0.0; len];
    let mut mem = MemBuffer::new(
        symbolic.factorize_numeric_ldlt_scratch::<f64>(Par::Seq, Default::default()),
    );
    symbolic
        .factorize_numeric_ldlt(
            &mut l_values,
            a.rb(),
            Side::Lower,
            LdltRegularization::default(),
            Par::Seq,
            MemStack::new(&mut mem),
            Default::default(),
        )
        .map_err(|_| MathError::NotPositiveDefinite)?;
    let d = extract_d(symbolic.as_ref(), &l_values)?;
    Ok(SparseLdltFactor {
        n: symbolic.nrows(),
        d,
        symbolic,
        l_values,
    })
}
