//! SPD Toeplitz inversion via Durbin–Levinson and Trench (Gohberg–Semencul).
//!
//! Model-independent: first row `r` defines the symmetric Toeplitz matrix
//! `T_{ij} = r[|i-j|]`. The inverse is generally dense (persymmetric, not
//! Toeplitz). Cost is Θ(n²) versus Θ(n³) for unstructured Cholesky inversion.
//!
//! Circulant/FFT embeddings are not used here: they invert a *different*
//! (periodic) covariance unless the embedding is undone with extra solves,
//! and they are not exact for the finite non-circulant FGN Gram matrix.

use crate::error::MathError;

/// Invert a symmetric positive-definite Toeplitz matrix given its first row.
///
/// Returns a row-major dense inverse. Fails if a Levinson reflection
/// coefficient leaves the leading minors non-positive (`|κ| ≥ 1`).
pub fn invert_spd_toeplitz(first_row: &[f64]) -> Result<Vec<f64>, MathError> {
    let n = first_row.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let r0 = first_row[0];
    if !r0.is_finite() || r0 <= 0.0 {
        return Err(MathError::NotPositiveDefinite);
    }
    for &rk in first_row {
        if !rk.is_finite() {
            return Err(MathError::InvalidMatrix(
                "Toeplitz first row must be finite",
            ));
        }
    }

    // Durbin: T [1, a₁, …, a_{n-1}]ᵀ = [E, 0, …, 0]ᵀ with E > 0.
    let mut a = vec![0.0; n.saturating_sub(1)];
    let mut e = r0;
    for k in 0..n.saturating_sub(1) {
        let mut kappa = first_row[k + 1];
        for j in 0..k {
            kappa += a[j] * first_row[k - j];
        }
        kappa = -kappa / e;
        let omk2 = 1.0 - kappa * kappa;
        if !omk2.is_finite() || omk2 <= 0.0 {
            return Err(MathError::NotPositiveDefinite);
        }
        if k > 0 {
            let mut updated = vec![0.0; k];
            for j in 0..k {
                updated[j] = a[j] + kappa * a[k - 1 - j];
            }
            a[..k].copy_from_slice(&updated);
        }
        a[k] = kappa;
        e *= omk2;
        if !e.is_finite() || e <= 0.0 {
            return Err(MathError::NotPositiveDefinite);
        }
    }

    // First column of T⁻¹ (symmetric ⇒ first row).
    let mut x = vec![0.0; n];
    x[0] = 1.0 / e;
    for i in 1..n {
        x[i] = a[i - 1] / e;
    }

    // Trench / Gohberg–Semencul: B_{i,j} = B_{i-1,j-1} + (x_i x_j − x_{n-i} x_{n-j}) / x_0
    // with z_0 = 0 and z_k = x_{n-k} for k ≥ 1.
    let mut inv = vec![0.0; n * n];
    for j in 0..n {
        inv[j] = x[j];
        inv[j * n] = x[j];
    }
    let x0 = x[0];
    for i in 1..n {
        for j in i..n {
            let val = inv[(i - 1) * n + (j - 1)] + (x[i] * x[j] - x[n - i] * x[n - j]) / x0;
            inv[i * n + j] = val;
            inv[j * n + i] = val;
        }
    }
    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::invert_spd_cholesky;

    fn toeplitz_dense(r: &[f64]) -> Vec<f64> {
        let n = r.len();
        let mut t = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                t[i * n + j] = r[i.abs_diff(j)];
            }
        }
        t
    }

    fn fgn_acf(n: usize, hurst: f64, tau: f64) -> Vec<f64> {
        let h2 = 2.0 * hurst;
        let mut acf = vec![0.0; n];
        acf[0] = 1.0 / tau;
        for k in 1..n {
            let kf = k as f64;
            acf[k] = 0.5 * ((kf + 1.0).powf(h2) - 2.0 * kf.powf(h2) + (kf - 1.0).powf(h2)) / tau;
        }
        acf
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn trench_matches_cholesky_iid() {
        let r = vec![0.5, 0.0, 0.0, 0.0];
        let inv = invert_spd_toeplitz(&r).expect("toeplitz");
        let chol = invert_spd_cholesky(&toeplitz_dense(&r), 4).expect("chol");
        assert!(max_abs_diff(&inv, &chol) < 1e-12);
        for i in 0..4 {
            for j in 0..4 {
                let v = inv[i * 4 + j];
                if i == j {
                    assert!((v - 2.0).abs() < 1e-12);
                } else {
                    assert!(v.abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn trench_matches_cholesky_fgn_small_n() {
        for n in [8usize, 16] {
            for h in [0.5, 0.7, 0.9] {
                let r = fgn_acf(n, h, 2.0);
                let inv = invert_spd_toeplitz(&r).expect("toeplitz");
                let chol = invert_spd_cholesky(&toeplitz_dense(&r), n).expect("chol");
                let scale = chol.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1.0);
                assert!(
                    max_abs_diff(&inv, &chol) < 1e-9 * scale,
                    "n={n} H={h} max|Δ|={}",
                    max_abs_diff(&inv, &chol)
                );
            }
        }
    }

    #[test]
    fn rejects_indefinite_row() {
        assert!(invert_spd_toeplitz(&[1.0, 2.0]).is_err());
    }
}
