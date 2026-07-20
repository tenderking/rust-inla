use crate::error::MathError;
use crate::scratch::{with_thread_scratch, LdltScratch};
use crate::sparse::CscMatrix;

#[cfg(feature = "sparse-ldlt")]
use crate::sparse_ldlt::SparseLdltFactor;

/// Scalar log-density evaluation with gradient and Hessian w.r.t. the linear predictor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eval1D {
    pub logp: f64,
    pub grad: f64,
    pub hess: f64,
}

/// Dense LDLᵀ factor `A = L D Lᵀ` in row-major `L` storage.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseLdltFactor {
    pub n: usize,
    pub l_row_major: Vec<f64>,
    pub d: Vec<f64>,
}

/// LDLᵀ factor of a precision matrix (dense or sparse backend).
#[derive(Debug, Clone, PartialEq)]
pub enum LdltFactor {
    Dense(DenseLdltFactor),
    #[cfg(feature = "sparse-ldlt")]
    Sparse(SparseLdltFactor),
    /// Placeholder when the sparse feature is off (never constructed).
    #[cfg(not(feature = "sparse-ldlt"))]
    Sparse(SparseStub),
}

#[cfg(not(feature = "sparse-ldlt"))]
#[derive(Debug, Clone, PartialEq)]
pub struct SparseStub {
    pub n: usize,
    pub d: Vec<f64>,
}

impl LdltFactor {
    pub fn n(&self) -> usize {
        match self {
            Self::Dense(f) => f.n,
            #[cfg(feature = "sparse-ldlt")]
            Self::Sparse(f) => f.n,
            #[cfg(not(feature = "sparse-ldlt"))]
            Self::Sparse(f) => f.n,
        }
    }

    /// Diagonal of D in `Q = L D Lᵀ`.
    pub fn diagonal(&self) -> &[f64] {
        match self {
            Self::Dense(f) => &f.d,
            #[cfg(feature = "sparse-ldlt")]
            Self::Sparse(f) => &f.d,
            #[cfg(not(feature = "sparse-ldlt"))]
            Self::Sparse(f) => &f.d,
        }
    }

    /// `log|Q| = Σᵢ log|Dᵢ|`.
    pub fn log_abs_det(&self) -> f64 {
        self.diagonal().iter().map(|&v| v.abs().ln()).sum()
    }
}

/// Densify a square CSC matrix into row-major storage.
pub fn csc_to_dense(csc: &CscMatrix) -> Result<Vec<f64>, MathError> {
    if csc.rows() != csc.cols() {
        return Err(MathError::NotSquare {
            rows: csc.rows(),
            cols: csc.cols(),
        });
    }
    let n = csc.rows();
    let mut a = vec![0.0; n * n];
    for (col, colvec) in csc.outer_iterator().enumerate() {
        for (row, value) in colvec.iter() {
            a[row * n + col] = *value;
        }
    }
    Ok(a)
}

pub(crate) fn ldlt_factorize_dense_inner(a: &[f64], n: usize) -> Result<DenseLdltFactor, MathError> {
    if a.len() != n * n {
        return Err(MathError::DimensionMismatch {
            context: "dense LDLᵀ matrix length",
            expected: n * n,
            got: a.len(),
        });
    }

    let mut l = vec![0.0; n * n];
    let mut d = vec![0.0; n];
    for i in 0..n {
        l[i * n + i] = 1.0;
    }

    for j in 0..n {
        let mut dj = a[j * n + j];
        for k in 0..j {
            let ljk = l[j * n + k];
            dj -= ljk * ljk * d[k];
        }
        if !dj.is_finite() || dj.abs() < 1e-14 {
            return Err(MathError::Singular);
        }
        d[j] = dj;

        for i in (j + 1)..n {
            let mut lij = a[i * n + j];
            for k in 0..j {
                lij -= l[i * n + k] * l[j * n + k] * d[k];
            }
            l[i * n + j] = lij / d[j];
        }
    }

    Ok(DenseLdltFactor {
        n,
        l_row_major: l,
        d,
    })
}

/// LDLᵀ factorize a dense symmetric matrix stored row-major.
pub fn ldlt_factorize_dense(a: &[f64], n: usize) -> Result<LdltFactor, MathError> {
    Ok(LdltFactor::Dense(ldlt_factorize_dense_inner(a, n)?))
}

/// Factorize CSC via the default backend (sparse when enabled).
pub fn ldlt_factorize(csc: &CscMatrix) -> Result<LdltFactor, MathError> {
    crate::backend::factorize_csc(csc)
}

pub fn ldlt_solve(f: &LdltFactor, b: &[f64]) -> Result<Vec<f64>, MathError> {
    let mut x = b.to_vec();
    ldlt_solve_in_place(f, &mut x)?;
    Ok(x)
}

pub fn ldlt_solve_in_place(f: &LdltFactor, x: &mut [f64]) -> Result<(), MathError> {
    crate::backend::solve_in_place(f, x)
}

pub(crate) fn dense_solve_in_place(f: &DenseLdltFactor, x: &mut [f64]) -> Result<(), MathError> {
    if x.len() != f.n {
        return Err(MathError::DimensionMismatch {
            context: "LDLᵀ solve RHS",
            expected: f.n,
            got: x.len(),
        });
    }
    let n = f.n;
    let l = &f.l_row_major;

    for i in 0..n {
        let mut acc = x[i];
        for k in 0..i {
            acc -= l[i * n + k] * x[k];
        }
        x[i] = acc;
    }

    for i in 0..n {
        if f.d[i].abs() < 1e-14 {
            return Err(MathError::Singular);
        }
        x[i] /= f.d[i];
    }

    for i in (0..n).rev() {
        let mut acc = x[i];
        for k in (i + 1)..n {
            acc -= l[k * n + i] * x[k];
        }
        x[i] = acc;
    }
    Ok(())
}

/// Diagonal of A⁻¹ given A = L D Lᵀ, via Y = L⁻¹ then (A⁻¹)ᵢᵢ = Σₖ Yₖᵢ² / Dₖ.
pub fn ldlt_diagonal_inverse(f: &LdltFactor) -> Result<Vec<f64>, MathError> {
    crate::backend::diagonal_inverse(f)
}

pub(crate) fn dense_diagonal_inverse(f: &DenseLdltFactor) -> Result<Vec<f64>, MathError> {
    let n = f.n;
    let l = &f.l_row_major;

    let mut y = vec![0.0; n * n];
    for i in 0..n {
        y[i * n + i] = 1.0;
        for j in 0..i {
            let mut s = 0.0;
            for k in j..i {
                s -= l[i * n + k] * y[k * n + j];
            }
            y[i * n + j] = s;
        }
    }

    let mut diag = vec![0.0; n];
    for i in 0..n {
        let mut s = 0.0;
        for k in i..n {
            let yki = y[k * n + i];
            s += yki * yki / f.d[k];
        }
        diag[i] = s;
    }
    Ok(diag)
}

/// One Newton step for the latent mode: solve `(Q − H) δ = g − Qx`.
pub fn laplace_newton_step(
    q_prior: &CscMatrix,
    evals: &[Eval1D],
    x: &[f64],
) -> Result<(Vec<f64>, LdltFactor), MathError> {
    laplace_newton_step_a(q_prior, evals, None, x)
}

/// Newton step with optional observation projector `A` (`n_obs × n_latent`).
pub fn laplace_newton_step_a(
    q_prior: &CscMatrix,
    evals: &[Eval1D],
    a: Option<&CscMatrix>,
    x: &[f64],
) -> Result<(Vec<f64>, LdltFactor), MathError> {
    with_thread_scratch(|scratch| laplace_newton_step_a_scratch(q_prior, evals, a, x, scratch))
}

fn laplace_newton_step_a_scratch(
    q_prior: &CscMatrix,
    evals: &[Eval1D],
    a: Option<&CscMatrix>,
    x: &[f64],
    scratch: &mut LdltScratch,
) -> Result<(Vec<f64>, LdltFactor), MathError> {
    use crate::backend::{DefaultBackend, LdltBackend};

    if q_prior.rows() != q_prior.cols() {
        return Err(MathError::NotSquare {
            rows: q_prior.rows(),
            cols: q_prior.cols(),
        });
    }
    let n = q_prior.rows();
    if x.len() != n {
        return Err(MathError::DimensionMismatch {
            context: "latent vector",
            expected: n,
            got: x.len(),
        });
    }

    let qx = crate::design::matvec_csc(q_prior, x).map_err(MathError::Message)?;
    let backend = DefaultBackend;

    match a {
        None => {
            if evals.len() != n {
                return Err(MathError::DimensionMismatch {
                    context: "eval vector",
                    expected: n,
                    got: evals.len(),
                });
            }
            // Build Q − H in CSC without densifying: diagonal update stays sparse.
            let mut tri = sprs::TriMatI::<f64, usize>::with_capacity((n, n), q_prior.nnz() + n);
            for (col, colvec) in q_prior.outer_iterator().enumerate() {
                for (row, &val) in colvec.iter() {
                    tri.add_triplet(row, col, val);
                }
            }
            for i in 0..n {
                let h = -evals[i].hess;
                if h != 0.0 {
                    tri.add_triplet(i, i, h);
                }
            }
            let q_post = tri.to_csc();
            let factor = backend.factorize(&q_post, scratch)?;
            let mut rhs: Vec<f64> = evals
                .iter()
                .enumerate()
                .map(|(i, e)| e.grad - qx[i])
                .collect();
            backend.solve_in_place(&factor, &mut rhs, scratch)?;
            Ok((rhs, factor))
        }
        Some(a_mat) => {
            if a_mat.cols() != n {
                return Err(MathError::DimensionMismatch {
                    context: "A columns",
                    expected: n,
                    got: a_mat.cols(),
                });
            }
            if a_mat.rows() != evals.len() {
                return Err(MathError::DimensionMismatch {
                    context: "eval vs A.nrows",
                    expected: a_mat.rows(),
                    got: evals.len(),
                });
            }
            let neg_hess: Vec<f64> = evals.iter().map(|e| -e.hess).collect();
            let like_prec = crate::design::at_diag_a(a_mat, &neg_hess).map_err(MathError::Message)?;
            let q_post = crate::design::add_csc(q_prior, &like_prec).map_err(MathError::Message)?;
            let factor = backend.factorize(&q_post, scratch)?;
            let g_eta: Vec<f64> = evals.iter().map(|e| e.grad).collect();
            let mut rhs =
                crate::design::matvec_transpose_csc(a_mat, &g_eta).map_err(MathError::Message)?;
            for i in 0..n {
                rhs[i] -= qx[i];
            }
            backend.solve_in_place(&factor, &mut rhs, scratch)?;
            Ok((rhs, factor))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::design::identity_csc;

    #[test]
    fn factors_and_solves_identity() {
        let csc = identity_csc(5, 2.0).unwrap();
        let f = ldlt_factorize(&csc).expect("factorize");
        assert_eq!(f.n(), 5);
        let rhs = vec![1.0; 5];
        let x = ldlt_solve(&f, &rhs).expect("solve");
        assert!(x.iter().all(|v| (v - 0.5).abs() < 1e-10));
    }

    #[test]
    fn computes_laplace_newton_step() {
        let q = identity_csc(4, 1.0).unwrap();
        let evals = vec![
            Eval1D {
                logp: 0.0,
                grad: 0.4,
                hess: -1.2,
            };
            4
        ];
        let (step, factor) = laplace_newton_step(&q, &evals, &[0.0; 4]).expect("step");
        assert_eq!(step.len(), 4);
        assert_eq!(factor.n(), 4);
        assert!(step.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn diagonal_inverse_matches_solves() {
        let csc = identity_csc(6, 1.5).unwrap();
        let f = ldlt_factorize(&csc).expect("factorize");
        let diag = ldlt_diagonal_inverse(&f).expect("diag");
        for i in 0..f.n() {
            let mut e = vec![0.0; f.n()];
            e[i] = 1.0;
            let sol = ldlt_solve(&f, &e).expect("solve");
            assert!(
                (diag[i] - sol[i]).abs() < 1e-10,
                "i={i}: diag={} sol={}",
                diag[i],
                sol[i]
            );
        }
    }

    #[test]
    fn sparse_matches_dense_on_tridiagonal() {
        let n = 8;
        let mut tri = sprs::TriMatI::<f64, usize>::new((n, n));
        for i in 0..n {
            tri.add_triplet(i, i, 2.5);
            if i + 1 < n {
                tri.add_triplet(i + 1, i, -0.7);
                tri.add_triplet(i, i + 1, -0.7);
            }
        }
        let csc = tri.to_csc();
        let sparse = ldlt_factorize(&csc).expect("sparse");
        let dense_mat = csc_to_dense(&csc).unwrap();
        let dense = ldlt_factorize_dense(&dense_mat, n).unwrap();
        assert!((sparse.log_abs_det() - dense.log_abs_det()).abs() < 1e-9);
        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let xs = ldlt_solve(&sparse, &b).unwrap();
        let xd = ldlt_solve(&dense, &b).unwrap();
        for i in 0..n {
            assert!((xs[i] - xd[i]).abs() < 1e-9, "i={i}");
        }
    }

    #[cfg(feature = "sparse-ldlt")]
    #[test]
    fn symbolic_cache_reuses_pattern_across_value_updates() {
        use crate::scratch::LdltScratch;
        let n = 6;
        let build = |diag: f64| {
            let mut tri = sprs::TriMatI::<f64, usize>::new((n, n));
            for i in 0..n {
                tri.add_triplet(i, i, diag);
                if i + 1 < n {
                    tri.add_triplet(i + 1, i, -0.4);
                    tri.add_triplet(i, i + 1, -0.4);
                }
            }
            tri.to_csc()
        };
        let q1 = build(2.0);
        let q2 = build(3.5);
        let mut scratch = LdltScratch::default();
        let f1 = crate::sparse_ldlt::factorize_sparse(&q1, &mut scratch).unwrap();
        assert!(scratch.symbolic_cache.is_some());
        let f2 = crate::sparse_ldlt::factorize_sparse(&q2, &mut scratch).unwrap();
        // Same symbolic Arc means numeric-only refactorize.
        assert!(std::sync::Arc::ptr_eq(
            &f1.symbolic_arc(),
            &f2.symbolic_arc()
        ));
        let b = vec![1.0; n];
        let x1 = ldlt_solve(&LdltFactor::Sparse(f1), &b).unwrap();
        let x2 = ldlt_solve(&LdltFactor::Sparse(f2), &b).unwrap();
        // Stronger diagonal → smaller solution magnitude.
        assert!(x2.iter().map(|v| v.abs()).sum::<f64>() < x1.iter().map(|v| v.abs()).sum::<f64>());
    }
}
