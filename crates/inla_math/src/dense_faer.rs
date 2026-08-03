//! Dense linear algebra via faer (SIMD / cache-blocked Cholesky, EVD).

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::{ldlt, llt};
use faer::linalg::evd::{self, ComputeEigenvectors};
use faer::prelude::*;
use faer::Par;

use crate::error::MathError;
use crate::ldlt::DenseLdltFactor;

const PAR_N_THRESHOLD: usize = 64;

fn par_for(n: usize) -> Par {
    if n >= PAR_N_THRESHOLD {
        Par::rayon(0)
    } else {
        Par::Seq
    }
}

/// Dense LDLᵀ factorization `A = L D Lᵀ` using faer's blocked SIMD kernel.
///
/// `a` is row-major symmetric; only the lower triangle is required.
pub fn ldlt_factorize(a: &[f64], n: usize) -> Result<DenseLdltFactor, MathError> {
    if a.len() != n * n {
        return Err(MathError::DimensionMismatch {
            context: "dense LDLᵀ matrix length",
            expected: n * n,
            got: a.len(),
        });
    }
    if n == 0 {
        return Ok(DenseLdltFactor {
            n: 0,
            l_row_major: Vec::new(),
            d: Vec::new(),
        });
    }

    let par = par_for(n);
    let mut mat = Mat::from_fn(n, n, |i, j| a[i * n + j]);
    let mut mem = MemBuffer::new(ldlt::factor::cholesky_in_place_scratch::<f64>(
        n,
        par,
        Default::default(),
    ));
    ldlt::factor::cholesky_in_place(
        mat.as_mut(),
        ldlt::factor::LdltRegularization::default(),
        par,
        MemStack::new(&mut mem),
        Default::default(),
    )
    .map_err(|_| MathError::Singular)?;

    // After factor: diagonal holds D; strict lower holds unit-triangular L.
    let mut l = vec![0.0; n * n];
    let mut d = vec![0.0; n];
    for i in 0..n {
        d[i] = mat[(i, i)];
        if !d[i].is_finite() || d[i].abs() < 1e-14 {
            return Err(MathError::Singular);
        }
        l[i * n + i] = 1.0;
        for j in 0..i {
            l[i * n + j] = mat[(i, j)];
        }
    }
    Ok(DenseLdltFactor {
        n,
        l_row_major: l,
        d,
    })
}

/// Invert an SPD matrix via faer LLᵀ Cholesky: `A = L Lᵀ ⇒ A⁻¹ = L⁻ᵀ L⁻¹`.
///
/// Input/output are row-major. Used for exact FGN precision construction.
pub fn invert_spd_cholesky(a: &[f64], n: usize) -> Result<Vec<f64>, MathError> {
    if a.len() != n * n {
        return Err(MathError::DimensionMismatch {
            context: "SPD invert matrix length",
            expected: n * n,
            got: a.len(),
        });
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    let par = par_for(n);
    let mut mat = Mat::from_fn(n, n, |i, j| a[i * n + j]);
    let mut mem_f = MemBuffer::new(llt::factor::cholesky_in_place_scratch::<f64>(
        n,
        par,
        Default::default(),
    ));
    llt::factor::cholesky_in_place(
        mat.as_mut(),
        llt::factor::LltRegularization::default(),
        par,
        MemStack::new(&mut mem_f),
        Default::default(),
    )
    .map_err(|_| MathError::NotPositiveDefinite)?;

    let mut inv = Mat::zeros(n, n);
    let mut mem_i = MemBuffer::new(llt::inverse::inverse_scratch::<f64>(n, par));
    llt::inverse::inverse(
        inv.as_mut(),
        mat.as_ref(),
        par,
        MemStack::new(&mut mem_i),
    );

    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            // inverse writes the lower triangle; symmetrize.
            let v = if i >= j { inv[(i, j)] } else { inv[(j, i)] };
            out[i * n + j] = v;
        }
    }
    Ok(out)
}

/// Self-adjoint eigendecomposition via faer (divide-and-conquer / QR).
///
/// Returns eigenvalues and eigenvectors in the same layout as `jacobi_eigen`:
/// `evecs[row * n + col]` stores column `col` of `V`.
pub fn selfadjoint_eigen(matrix: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), MathError> {
    if matrix.len() != n * n {
        return Err(MathError::DimensionMismatch {
            context: "selfadjoint EVD matrix length",
            expected: n * n,
            got: matrix.len(),
        });
    }
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let par = par_for(n);
    let a = Mat::from_fn(n, n, |i, j| matrix[i * n + j]);
    let mut s = Mat::zeros(n, 1);
    let mut u = Mat::zeros(n, n);
    let mut mem = MemBuffer::new(evd::self_adjoint_evd_scratch::<f64>(
        n,
        ComputeEigenvectors::Yes,
        par,
        Default::default(),
    ));
    evd::self_adjoint_evd(
        a.as_ref(),
        s.as_mut().col_mut(0).as_diagonal_mut(),
        Some(u.as_mut()),
        par,
        MemStack::new(&mut mem),
        Default::default(),
    )
    .map_err(|_| MathError::Message("self-adjoint EVD failed to converge".into()))?;

    let mut evals = vec![0.0; n];
    let mut evecs = vec![0.0; n * n];
    for i in 0..n {
        evals[i] = s[(i, 0)];
        for j in 0..n {
            evecs[i * n + j] = u[(i, j)];
        }
    }
    Ok((evals, evecs))
}
