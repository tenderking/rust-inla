//! Sparse LDLᵀ via faer (no `csc_to_dense` on the hot path).

use std::sync::Arc;

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::ldlt::factor::LdltRegularization;
use faer::perm::PermRef;
use faer::prelude::*;
use faer::sparse::linalg::SupernodalThreshold;
use faer::sparse::linalg::cholesky::{
    self, CholeskySymbolicParams, LdltRef, SymbolicCholesky, SymbolicCholeskyRaw, SymmetricOrdering,
};
use faer::sparse::{SparseColMat, Triplet};
use faer::{Conj, Par, Side};

use crate::error::MathError;
use crate::ordering::{CholeskyOrder, choose_symmetric_order};
use crate::scratch::LdltScratch;
use crate::sparse::CscMatrix;

/// RHS block width for multi-column marginal-variance solves.
const DIAG_INV_BLOCK: usize = 64;
/// Prefer Rayon once the system is large enough that thread overhead pays off.
const PAR_N_THRESHOLD: usize = 128;

/// Owned sparse LDLᵀ factor (symbolic pattern + numeric values + D diagonal).
#[derive(Clone)]
pub struct SparseLdltFactor {
    pub n: usize,
    /// Diagonal of D in `Q = L D Lᵀ` (for log|Q| = Σ log|Dᵢ|).
    /// Indexed in the factor's (possibly AMD-permuted) order.
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

    /// Number of stored entries in `L` (including `D` on the diagonal).
    ///
    /// For a time-major / banded FGN approx this is `O(n · order²)`, not `Θ(n²)`.
    pub fn nnz_l(&self) -> usize {
        self.l_values.len()
    }
}

fn par_for(n: usize) -> Par {
    if n >= PAR_N_THRESHOLD {
        Par::rayon(0)
    } else {
        Par::Seq
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
    // faer LDLᵀ with Side::Lower expects this; passing a full symmetric
    // matrix makes A_nnz (symbolic) disagree with permute scratch.
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
        SymbolicCholeskyRaw::Supernodal(sym) => {
            let ldlt = cholesky::supernodal::SupernodalLdltRef::new(sym, l_values);
            for s in 0..sym.n_supernodes() {
                let sn = ldlt.supernode(s);
                let size = sn.val().ncols();
                let ds = sn.val().diagonal().column_vector();
                let start = sn.start();
                for idx in 0..size {
                    let dj = ds[idx];
                    if !dj.is_finite() || dj.abs() < 1e-14 {
                        return Err(MathError::Singular);
                    }
                    d[start + idx] = dj;
                }
            }
        }
    }
    Ok(d)
}

fn symbolic_params() -> CholeskySymbolicParams<'static> {
    CholeskySymbolicParams {
        // AUTO picks simplicial vs supernodal from flop-ratio heuristics.
        supernodal_flop_ratio_threshold: SupernodalThreshold::AUTO,
        ..Default::default()
    }
}

fn factorize_symbolic(
    q: &CscMatrix,
    a: &SparseColMat<usize, f64>,
) -> Result<SymbolicCholesky<usize>, MathError> {
    let order = choose_symmetric_order(q);
    let symbolic = match &order {
        CholeskyOrder::Amd => cholesky::factorize_symbolic_cholesky(
            a.symbolic(),
            Side::Lower,
            SymmetricOrdering::Amd,
            symbolic_params(),
        ),
        CholeskyOrder::Custom { fwd, inv } => {
            let perm = PermRef::<usize>::new_checked(fwd, inv, a.nrows());
            cholesky::factorize_symbolic_cholesky(
                a.symbolic(),
                Side::Lower,
                SymmetricOrdering::Custom(perm),
                symbolic_params(),
            )
        }
    }
    .map_err(|e| match e {
        faer::sparse::FaerError::OutOfMemory => MathError::OutOfMemory,
        other => MathError::Message(format!("sparse symbolic LDLᵀ failed: {other:?}")),
    })?;
    Ok(symbolic)
}

/// Sparse LDLᵀ of a CSC precision matrix.
///
/// Ordering: time-major for mixture-major Kronecker / FGN-approx graphs, else
/// RCM when it cuts the envelope, else AMD. The stored CSC (and A-matrix
/// indexing) is not rewritten.
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
    let symbolic = factorize_symbolic(q, &a)?;

    let n = symbolic.nrows();
    let par = par_for(n);
    let len = symbolic.len_val();
    scratch.ensure_l_values(len);
    let mut l_values = vec![0.0; len];
    let mut mem =
        MemBuffer::new(symbolic.factorize_numeric_ldlt_scratch::<f64>(par, Default::default()));
    symbolic
        .factorize_numeric_ldlt(
            &mut l_values,
            a.rb(),
            Side::Lower,
            LdltRegularization::default(),
            par,
            MemStack::new(&mut mem),
            Default::default(),
        )
        .map_err(|_| MathError::NotPositiveDefinite)?;

    let d = extract_d(&symbolic, &l_values)?;
    let symbolic = Arc::new(symbolic);
    scratch.store_pattern(q, Arc::clone(&symbolic));
    Ok(SparseLdltFactor {
        n,
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
    let n = factor.n;
    let par = par_for(n);
    let mut rhs = Mat::from_fn(n, 1, |i, _| x[i]);
    let mut mem = MemBuffer::new(factor.symbolic.solve_in_place_scratch::<f64>(1, par));
    let ldlt = LdltRef::new(factor.symbolic.as_ref(), &factor.l_values);
    ldlt.solve_in_place_with_conj(Conj::No, rhs.as_mut(), par, MemStack::new(&mut mem));
    for i in 0..n {
        x[i] = rhs[(i, 0)];
    }
    Ok(())
}

/// Lower CSC of unit-`L` plus `D` on the diagonal, in factor (AMD) order.
/// Pattern is the filled `L` (including structural zeros).
struct FilledUnitLdlt {
    n: usize,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    /// CSC values: `D_j` on the diagonal, unit-`L` below.
    vals: Vec<f64>,
    /// `orig_of_factor[j]` is the original index of factor index `j`.
    orig_of_factor: Vec<usize>,
}

fn orig_of_factor(symbolic: &SymbolicCholesky<usize>, n: usize) -> Vec<usize> {
    match symbolic.perm() {
        Some(perm) => perm.arrays().0.to_vec(),
        None => (0..n).collect(),
    }
}

fn packed_lower_csc(
    n: usize,
    cols: &mut [Vec<(usize, f64)>],
    orig_of_factor: Vec<usize>,
) -> FilledUnitLdlt {
    let mut col_ptr = Vec::with_capacity(n + 1);
    col_ptr.push(0);
    let mut row_idx = Vec::new();
    let mut vals = Vec::new();
    for col in cols.iter_mut() {
        col.sort_unstable_by_key(|&(r, _)| r);
        for &(r, v) in col.iter() {
            row_idx.push(r);
            vals.push(v);
        }
        col_ptr.push(row_idx.len());
    }
    FilledUnitLdlt {
        n,
        col_ptr,
        row_idx,
        vals,
        orig_of_factor,
    }
}

fn filled_l_from_factor(factor: &SparseLdltFactor) -> Result<FilledUnitLdlt, MathError> {
    let n = factor.n;
    let orig = orig_of_factor(factor.symbolic.as_ref(), n);
    match factor.symbolic.raw() {
        SymbolicCholeskyRaw::Simplicial(sym) => {
            let col_ptr = sym.col_ptr().to_vec();
            let row_idx = sym.row_idx().to_vec();
            let mut cols: Vec<Vec<(usize, f64)>> = (0..n)
                .map(|j| {
                    let a = col_ptr[j];
                    let b = col_ptr[j + 1];
                    row_idx[a..b]
                        .iter()
                        .zip(&factor.l_values[a..b])
                        .map(|(&r, &v)| (r, v))
                        .collect()
                })
                .collect();
            Ok(packed_lower_csc(n, &mut cols, orig))
        }
        SymbolicCholeskyRaw::Supernodal(sym) => {
            let ldlt = cholesky::supernodal::SupernodalLdltRef::new(sym, &factor.l_values);
            let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
            for s in 0..sym.n_supernodes() {
                let sn = ldlt.supernode(s);
                let start = sn.start();
                let size = sn.val().ncols();
                let val = sn.val();
                let pattern = sn.pattern();
                for js in 0..size {
                    let j = start + js;
                    cols[j].push((j, val[(js, js)]));
                    for is in (js + 1)..size {
                        cols[j].push((start + is, val[(is, js)]));
                    }
                    for (p, row) in pattern.iter().enumerate() {
                        cols[j].push((*row, val[(size + p, js)]));
                    }
                }
            }
            Ok(packed_lower_csc(n, &mut cols, orig))
        }
    }
}

fn lookup_s(l: &FilledUnitLdlt, s: &[f64], i: usize, j: usize) -> Result<f64, MathError> {
    let (row, col) = if i >= j { (i, j) } else { (j, i) };
    debug_assert!(row >= col);
    let a = l.col_ptr[col];
    let b = l.col_ptr[col + 1];
    match l.row_idx[a..b].binary_search(&row) {
        Ok(k) => Ok(s[a + k]),
        Err(_) => Err(MathError::Message(format!(
            "Takahashi: missing selected-inverse entry ({row},{col})"
        ))),
    }
}

/// Takahashi selected inverse on the filled pattern of `L`: only `S_{ij}` with
/// `L_{ij} ≠ 0` (structurally), which is enough for `diag(Q⁻¹)`.
///
/// For `Q = L D Lᵀ` with unit lower `L`:
/// `S_{ij} = −∑_{k>j} L_{kj} S_{ki}` (`i > j` in the pattern),
/// `S_{jj} = 1/D_j − ∑_{k>j} L_{kj} S_{kj}`.
fn takahashi_selected_diag(l: &FilledUnitLdlt) -> Result<Vec<f64>, MathError> {
    let n = l.n;
    let mut s = vec![0.0; l.vals.len()];
    for j in (0..n).rev() {
        let start = l.col_ptr[j];
        let end = l.col_ptr[j + 1];
        if start >= end {
            return Err(MathError::Singular);
        }
        let Some(diag_off) = l.row_idx[start..end].iter().position(|&r| r == j) else {
            return Err(MathError::Singular);
        };
        let diag_p = start + diag_off;
        let dj = l.vals[diag_p];
        if !dj.is_finite() || dj.abs() < 1e-14 {
            return Err(MathError::Singular);
        }

        // Largest `i > j` first so `S_{ii}` (and larger columns) are already done.
        for p in (start..end).rev() {
            let i = l.row_idx[p];
            if i <= j {
                continue;
            }
            let mut acc = 0.0;
            for pk in start..end {
                let k = l.row_idx[pk];
                if k <= j {
                    continue;
                }
                acc += l.vals[pk] * lookup_s(l, &s, k, i)?;
            }
            s[p] = -acc;
        }

        let mut sjj = 1.0 / dj;
        for pk in start..end {
            if l.row_idx[pk] <= j {
                continue;
            }
            sjj -= l.vals[pk] * s[pk];
        }
        s[diag_p] = sjj;
    }

    let mut diag = vec![0.0; n];
    for j in 0..n {
        let start = l.col_ptr[j];
        let end = l.col_ptr[j + 1];
        let diag_off = l.row_idx[start..end]
            .iter()
            .position(|&r| r == j)
            .ok_or(MathError::Singular)?;
        diag[l.orig_of_factor[j]] = s[start + diag_off];
    }
    Ok(diag)
}

/// Marginal variances `diag(Q⁻¹)` by Takahashi selected inversion on `L`'s fill.
///
/// Dense fallback is [`crate::ldlt::dense_diagonal_inverse`] for `LdltFactor::Dense`
/// (exact FGN and other densified systems). This path never forms `Q⁻¹`.
pub fn sparse_diagonal_inverse(
    factor: &SparseLdltFactor,
    _scratch: &mut LdltScratch,
) -> Result<Vec<f64>, MathError> {
    let filled = filled_l_from_factor(factor)?;
    takahashi_selected_diag(&filled)
}

/// Reference `diag(Q⁻¹)` via blocked multi-column solves (`Q X = I`).
///
/// Kept for tests and the AR1 / Besag timing comparison in `benches/ar1_ldlt.rs`.
/// Typical cost is `Θ(n)` triangular solves vs Takahashi `Θ(nnz(L))`.
pub fn sparse_diagonal_inverse_by_solves(
    factor: &SparseLdltFactor,
    _scratch: &mut LdltScratch,
) -> Result<Vec<f64>, MathError> {
    let n = factor.n;
    let par = par_for(n);
    let mut diag = vec![0.0; n];
    let ldlt = LdltRef::new(factor.symbolic.as_ref(), &factor.l_values);
    let mut start = 0;
    while start < n {
        let k = (n - start).min(DIAG_INV_BLOCK);
        let mut rhs = Mat::from_fn(n, k, |r, c| if r == start + c { 1.0 } else { 0.0 });
        let mut mem = MemBuffer::new(factor.symbolic.solve_in_place_scratch::<f64>(k, par));
        ldlt.solve_in_place_with_conj(Conj::No, rhs.as_mut(), par, MemStack::new(&mut mem));
        for c in 0..k {
            diag[start + c] = rhs[(start + c, c)];
        }
        start += k;
    }
    Ok(diag)
}

/// Symbolic pattern only (factorize-once / numeric-many).
pub fn symbolic_pattern(q: &CscMatrix) -> Result<Arc<SymbolicCholesky<usize>>, MathError> {
    let a = csc_to_faer(q)?;
    let symbolic = factorize_symbolic(q, &a)?;
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
    let n = symbolic.nrows();
    let par = par_for(n);
    let len = symbolic.len_val();
    let mut l_values = vec![0.0; len];
    let mut mem =
        MemBuffer::new(symbolic.factorize_numeric_ldlt_scratch::<f64>(par, Default::default()));
    symbolic
        .factorize_numeric_ldlt(
            &mut l_values,
            a.rb(),
            Side::Lower,
            LdltRegularization::default(),
            par,
            MemStack::new(&mut mem),
            Default::default(),
        )
        .map_err(|_| MathError::NotPositiveDefinite)?;
    let d = extract_d(symbolic.as_ref(), &l_values)?;
    Ok(SparseLdltFactor {
        n,
        d,
        symbolic,
        l_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{DenseBackend, LdltBackend};
    use crate::scratch::LdltScratch;
    use crate::sparse::sparse_from_triplets;
    use sprs::TriMatI;

    fn mixture_major_fgn_like(n_time: usize, n_comp: usize) -> CscMatrix {
        let n = n_time * n_comp;
        let mut tri = TriMatI::<f64, usize>::new((n, n));
        for t in 0..n_time {
            for c in 0..n_comp {
                let i = c * n_time + t;
                tri.add_triplet(i, i, 6.0);
                for d in (c + 1)..n_comp {
                    let j = d * n_time + t;
                    tri.add_triplet(i, j, -0.4);
                    tri.add_triplet(j, i, -0.4);
                }
                if t + 1 < n_time && c > 0 {
                    let i2 = c * n_time + t + 1;
                    tri.add_triplet(i, i2, -1.2);
                    tri.add_triplet(i2, i, -1.2);
                }
            }
        }
        tri.to_csc()
    }

    #[test]
    fn fgn_like_factor_is_sparse_and_not_cubic() {
        let n_time = 64;
        let n_comp = 5;
        let n = n_time * n_comp;
        let q = mixture_major_fgn_like(n_time, n_comp);
        let mut scratch = LdltScratch::default();
        let f = factorize_sparse(&q, &mut scratch).expect("sparse factor");
        assert_eq!(f.n, n);
        assert!(
            f.nnz_l() < n * n / 8,
            "nnz_L={} looks Θ(n²) for n={n}",
            f.nnz_l()
        );
        assert!(
            scratch.dense.len() < n * n,
            "sparse path must not allocate a dense n×n workspace"
        );
        let mut x = vec![1.0; n];
        sparse_solve_in_place(&f, &mut x, &mut scratch).expect("solve");
        assert!(x.iter().all(|v| v.is_finite()));
    }

    fn ar1_q(n: usize, rho: f64, tau: f64) -> CscMatrix {
        let mut trips = Vec::with_capacity(3 * n);
        for i in 0..n {
            let diag = if i == 0 || i == n - 1 {
                tau
            } else {
                tau * (1.0 + rho * rho)
            };
            trips.push((i, i, diag));
            if i + 1 < n {
                trips.push((i, i + 1, -tau * rho));
                trips.push((i + 1, i, -tau * rho));
            }
        }
        sparse_from_triplets(n, n, &trips)
    }

    /// Intrinsic Besag (graph Laplacian) plus a ridge so `Q` is SPD.
    fn besag_ridge_q(adj: &[Vec<usize>], tau: f64, ridge: f64) -> CscMatrix {
        let n = adj.len();
        let mut trips = Vec::new();
        for i in 0..n {
            let deg = adj[i].len() as f64;
            trips.push((i, i, tau * deg + ridge));
            for &j in &adj[i] {
                if j > i {
                    trips.push((i, j, -tau));
                    trips.push((j, i, -tau));
                }
            }
        }
        sparse_from_triplets(n, n, &trips)
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    fn compare_takahashi_vs_dense(q: &CscMatrix, tol: f64) {
        let mut scratch = LdltScratch::default();
        let sparse = factorize_sparse(q, &mut scratch).expect("sparse LDLᵀ");
        let taka = sparse_diagonal_inverse(&sparse, &mut scratch).expect("Takahashi");
        let by_solves = sparse_diagonal_inverse_by_solves(&sparse, &mut scratch).expect("solves");

        let mut dense_scratch = LdltScratch::default();
        let dense = DenseBackend
            .factorize(q, &mut dense_scratch)
            .expect("dense LDLᵀ");
        let dense_diag = DenseBackend
            .diagonal_inverse(&dense, &mut dense_scratch)
            .expect("dense diag");

        assert_eq!(taka.len(), q.rows());
        assert!(
            max_abs_diff(&taka, &dense_diag) < tol,
            "Takahashi vs dense max|Δ|={} tol={tol}\n taka={taka:?}\n dense={dense_diag:?}",
            max_abs_diff(&taka, &dense_diag)
        );
        assert!(
            max_abs_diff(&taka, &by_solves) < tol,
            "Takahashi vs blocked solves max|Δ|={}",
            max_abs_diff(&taka, &by_solves)
        );
    }

    #[test]
    fn takahashi_matches_dense_diag_ar1() {
        compare_takahashi_vs_dense(&ar1_q(12, 0.7, 2.5), 1e-9);
        compare_takahashi_vs_dense(&ar1_q(3, 0.2, 1.0), 1e-10);
    }

    #[test]
    fn takahashi_matches_dense_diag_besag() {
        // 4-cycle plus a chord (small connected Besag graph).
        let adj = vec![vec![1, 3], vec![0, 2, 3], vec![1, 3], vec![0, 1, 2]];
        compare_takahashi_vs_dense(&besag_ridge_q(&adj, 1.5, 0.25), 1e-9);
        // Path of 6 (tree-like ICAR).
        let path: Vec<Vec<usize>> = (0..6)
            .map(|i| {
                let mut nbs = Vec::new();
                if i > 0 {
                    nbs.push(i - 1);
                }
                if i + 1 < 6 {
                    nbs.push(i + 1);
                }
                nbs
            })
            .collect();
        compare_takahashi_vs_dense(&besag_ridge_q(&path, 1.0, 0.1), 1e-9);
    }

    // Timing note (do not treat as a CI benchmark): on AR1 n≈400, Takahashi is
    // Θ(nnz(L)) ≈ O(n) while `sparse_diagonal_inverse_by_solves` does Θ(n)
    // triangular solves and the dense path densifies then inverts. See
    // `benches/ar1_ldlt.rs` (`diag_inv_ar1`).
}
