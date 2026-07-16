use crate::sparse::CscMatrix;

/// Scalar log-density evaluation with gradient and Hessian w.r.t. the linear predictor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eval1D {
    pub logp: f64,
    pub grad: f64,
    pub hess: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LdltFactor {
    pub n: usize,
    pub l_row_major: Vec<f64>,
    pub d: Vec<f64>,
}

/// Densify a square CSC matrix into row-major storage.
pub fn csc_to_dense(csc: &CscMatrix) -> Result<Vec<f64>, String> {
    if csc.rows() != csc.cols() {
        return Err("matrix must be square".to_string());
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

/// LDLᵀ factorize a dense symmetric matrix stored row-major.
///
/// Consumes `a` as workspace (upper triangle is overwritten during elimination
/// bookkeeping via the lower-triangle algorithm on the copy in `a`).
pub fn ldlt_factorize_dense(a: &[f64], n: usize) -> Result<LdltFactor, String> {
    if a.len() != n * n {
        return Err(format!(
            "dense matrix length {} does not match n² = {}",
            a.len(),
            n * n
        ));
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
            return Err("matrix is singular or numerically unstable in LDLT".to_string());
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

    Ok(LdltFactor {
        n,
        l_row_major: l,
        d,
    })
}

pub fn ldlt_factorize(csc: &CscMatrix) -> Result<LdltFactor, String> {
    let a = csc_to_dense(csc)?;
    let n = csc.rows();
    // Light symmetry check on a sample of entries (full O(n²) check is too
    // expensive on the dense FGN hot path).
    for r in 0..n {
        let c = (r * 7 + 3) % n;
        if (a[r * n + c] - a[c * n + r]).abs() > 1e-10 {
            return Err("LDLT requires a symmetric matrix".to_string());
        }
    }
    ldlt_factorize_dense(&a, n)
}

pub fn ldlt_solve(f: &LdltFactor, b: &[f64]) -> Result<Vec<f64>, String> {
    let mut x = b.to_vec();
    ldlt_solve_in_place(f, &mut x)?;
    Ok(x)
}

pub fn ldlt_solve_in_place(f: &LdltFactor, x: &mut [f64]) -> Result<(), String> {
    if x.len() != f.n {
        return Err("right-hand side length must match matrix dimension".to_string());
    }
    let n = f.n;
    let l = &f.l_row_major;

    // Forward: L y = b
    for i in 0..n {
        let mut acc = x[i];
        for k in 0..i {
            acc -= l[i * n + k] * x[k];
        }
        x[i] = acc;
    }

    // Diagonal: D z = y
    for i in 0..n {
        if f.d[i].abs() < 1e-14 {
            return Err("singular diagonal in LDLT solve".to_string());
        }
        x[i] /= f.d[i];
    }

    // Back: Lᵀ x = z
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
pub fn ldlt_diagonal_inverse(f: &LdltFactor) -> Result<Vec<f64>, String> {
    let n = f.n;
    let l = &f.l_row_major;

    // Invert unit lower-triangular L into Y (lower).
    let mut y = vec![0.0; n * n];
    for i in 0..n {
        y[i * n + i] = 1.0;
        for j in 0..i {
            let mut s = 0.0;
            for k in j..i {
                s -= l[i * n + k] * y[k * n + j];
            }
            y[i * n + j] = s; // L has unit diagonal
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

/// One Newton step for the latent mode: solve (Q + (−hess)) δ = grad.
///
/// Returns `(step, posterior_factor)` so the caller can reuse the factor when
/// the step has converged (Gaussian case: exact after one iteration).
pub fn laplace_newton_step(
    q_prior: &CscMatrix,
    evals: &[Eval1D],
) -> Result<(Vec<f64>, LdltFactor), String> {
    laplace_newton_step_a(q_prior, evals, None)
}

/// Newton step with optional observation projector `A` (`n_obs × n_latent`).
///
/// When `A` is `None`, observations map 1:1 onto the latent field.
/// Otherwise η = A x, and the system is
/// `(Q − Aᵀ H A) δ = Aᵀ g` with `H = diag(hess)`, `g = grad` w.r.t. η.
pub fn laplace_newton_step_a(
    q_prior: &CscMatrix,
    evals: &[Eval1D],
    a: Option<&CscMatrix>,
) -> Result<(Vec<f64>, LdltFactor), String> {
    if q_prior.rows() != q_prior.cols() {
        return Err("prior precision must be square".to_string());
    }
    let n = q_prior.rows();

    match a {
        None => {
            if q_prior.rows() != evals.len() {
                return Err("eval vector length must match precision dimension".to_string());
            }
            let mut dens = csc_to_dense(q_prior)?;
            for i in 0..n {
                dens[i * n + i] += -evals[i].hess;
            }
            let factor = ldlt_factorize_dense(&dens, n)?;
            let mut grad: Vec<f64> = evals.iter().map(|e| e.grad).collect();
            ldlt_solve_in_place(&factor, &mut grad)?;
            Ok((grad, factor))
        }
        Some(a_mat) => {
            if a_mat.cols() != n {
                return Err(format!(
                    "A has {} cols but Q is {}×{}",
                    a_mat.cols(),
                    n,
                    n
                ));
            }
            if a_mat.rows() != evals.len() {
                return Err("eval vector length must match A.nrows (n_obs)".to_string());
            }
            let neg_hess: Vec<f64> = evals.iter().map(|e| -e.hess).collect();
            let like_prec = crate::design::at_diag_a(a_mat, &neg_hess)?;
            let q_post = crate::design::add_csc(q_prior, &like_prec)?;
            let dens = csc_to_dense(&q_post)?;
            let factor = ldlt_factorize_dense(&dens, n)?;
            let g_eta: Vec<f64> = evals.iter().map(|e| e.grad).collect();
            let mut grad_x = crate::design::matvec_transpose_csc(a_mat, &g_eta)?;
            ldlt_solve_in_place(&factor, &mut grad_x)?;
            Ok((grad_x, factor))
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
        assert_eq!(f.n, 5);
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
        let (step, factor) = laplace_newton_step(&q, &evals).expect("step");
        assert_eq!(step.len(), 4);
        assert_eq!(factor.n, 4);
        assert!(step.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn diagonal_inverse_matches_solves() {
        let csc = identity_csc(6, 1.5).unwrap();
        let f = ldlt_factorize(&csc).expect("factorize");
        let diag = ldlt_diagonal_inverse(&f).expect("diag");
        for i in 0..f.n {
            let mut e = vec![0.0; f.n];
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
}
