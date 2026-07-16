use inla_math::CscMatrix;
use sprs::TriMatI;

pub fn rw1_precision_csc(n: usize, tau: f64) -> Result<CscMatrix, String> {
    if n < 2 {
        return Err("rw1 requires n >= 2".to_string());
    }
    if tau <= 0.0 || !tau.is_finite() {
        return Err("rw1 tau must be finite and > 0".to_string());
    }

    let mut tri = TriMatI::<f64, usize>::with_capacity((n, n), 3 * n - 2);
    for i in 0..n {
        let diag = if i == 0 || i == n - 1 { tau } else { 2.0 * tau };
        tri.add_triplet(i, i, diag);
        if i + 1 < n {
            tri.add_triplet(i, i + 1, -tau);
            tri.add_triplet(i + 1, i, -tau);
        }
    }
    Ok(tri.to_csc())
}

pub fn rw2_precision_csc(n: usize, tau: f64) -> Result<CscMatrix, String> {
    if n < 3 {
        return Err("rw2 requires n >= 3".to_string());
    }
    if tau <= 0.0 || !tau.is_finite() {
        return Err("rw2 tau must be finite and > 0".to_string());
    }

    // Q = tau * D2' D2, where each row of D2 is [1, -2, 1] over consecutive entries.
    let mut dense = vec![0.0; n * n];
    for r in 0..(n - 2) {
        let idx = [r, r + 1, r + 2];
        let w = [1.0_f64, -2.0, 1.0];
        for a in 0..3 {
            for b in 0..3 {
                dense[idx[a] * n + idx[b]] += tau * w[a] * w[b];
            }
        }
    }

    let mut tri = TriMatI::<f64, usize>::with_capacity((n, n), 5 * n - 6);
    for i in 0..n {
        for j in 0..n {
            let v = dense[i * n + j];
            if v != 0.0 {
                tri.add_triplet(i, j, v);
            }
        }
    }
    Ok(tri.to_csc())
}

pub fn rw1_cyclic_precision_csc(n: usize, tau: f64) -> Result<CscMatrix, String> {
    if n < 3 {
        return Err("rw1 cyclic requires n >= 3".to_string());
    }
    if tau <= 0.0 || !tau.is_finite() {
        return Err("rw1 cyclic tau must be finite and > 0".to_string());
    }

    let mut tri = TriMatI::<f64, usize>::with_capacity((n, n), 3 * n);
    for i in 0..n {
        for j in 0..n {
            let diff = (i as isize - j as isize).abs();
            let idiff = std::cmp::min(diff, n as isize - diff);
            if idiff == 0 {
                tri.add_triplet(i, j, 2.0 * tau);
            } else if idiff == 1 {
                tri.add_triplet(i, j, -1.0 * tau);
            }
        }
    }
    Ok(tri.to_csc())
}

pub fn rw2_cyclic_precision_csc(n: usize, tau: f64) -> Result<CscMatrix, String> {
    if n < 5 {
        return Err("rw2 cyclic requires n >= 5".to_string());
    }
    if tau <= 0.0 || !tau.is_finite() {
        return Err("rw2 cyclic tau must be finite and > 0".to_string());
    }

    let mut tri = TriMatI::<f64, usize>::with_capacity((n, n), 5 * n);
    for i in 0..n {
        for j in 0..n {
            let diff = (i as isize - j as isize).abs();
            let idiff = std::cmp::min(diff, n as isize - diff);
            if idiff == 0 {
                tri.add_triplet(i, j, 6.0 * tau);
            } else if idiff == 1 {
                tri.add_triplet(i, j, -4.0 * tau);
            } else if idiff == 2 {
                tri.add_triplet(i, j, 1.0 * tau);
            }
        }
    }
    Ok(tri.to_csc())
}

pub fn seasonal_precision_csc(n: usize, s: usize, tau: f64, cyclic: bool) -> Result<CscMatrix, String> {
    if s == 0 {
        return Err("seasonal period s must be >= 1".to_string());
    }
    if n < s {
        return Err("seasonal requires n >= s".to_string());
    }
    if tau <= 0.0 || !tau.is_finite() {
        return Err("seasonal tau must be finite and > 0".to_string());
    }

    let mut tri = TriMatI::<f64, usize>::with_capacity((n, n), n * s);
    for i in 0..n {
        for j in 0..n {
            let val = if cyclic {
                let diff = (i as isize - j as isize).abs();
                let idiff = std::cmp::min(diff, n as isize - diff) as usize;
                if idiff >= s {
                    0.0
                } else {
                    (s - idiff) as f64
                }
            } else {
                let imin = std::cmp::min(i, j);
                let imax = std::cmp::max(i, j);
                let diff = imax - imin;

                if diff >= s {
                    0.0
                } else if imin <= (s - diff - 1) {
                    (imin + 1) as f64
                } else if imin > (s - diff - 1) && imin < (n - s) {
                    (s - diff) as f64
                } else if imin >= (n - s) {
                    (n - diff - imin) as f64
                } else {
                    0.0
                }
            };

            if val > 0.0 {
                tri.add_triplet(i, j, val * tau);
            }
        }
    }
    Ok(tri.to_csc())
}

pub fn iid_precision_csc(n: usize, tau: f64) -> Result<CscMatrix, String> {
    if n == 0 {
        return Err("iid requires n >= 1".to_string());
    }
    if tau <= 0.0 || !tau.is_finite() {
        return Err("iid tau must be finite and > 0".to_string());
    }
    let mut tri = TriMatI::<f64, usize>::with_capacity((n, n), n);
    for i in 0..n {
        tri.add_triplet(i, i, tau);
    }
    Ok(tri.to_csc())
}

pub fn two_diid_precision_csc(n_pairs: usize, rho: f64, tau: f64) -> Result<CscMatrix, String> {
    if n_pairs == 0 {
        return Err("2diid requires n_pairs >= 1".to_string());
    }
    if !rho.is_finite() || rho.abs() >= 1.0 {
        return Err("2diid rho must be finite and satisfy |rho| < 1".to_string());
    }
    if tau <= 0.0 || !tau.is_finite() {
        return Err("2diid tau must be finite and > 0".to_string());
    }

    let n = 2 * n_pairs;
    let mut tri = TriMatI::<f64, usize>::with_capacity((n, n), 4 * n_pairs);
    let s = tau / (1.0 - rho * rho);
    let a = s;
    let b = -rho * s;

    for g in 0..n_pairs {
        let i = 2 * g;
        let j = i + 1;
        tri.add_triplet(i, i, a);
        tri.add_triplet(j, j, a);
        tri.add_triplet(i, j, b);
        tri.add_triplet(j, i, b);
    }
    Ok(tri.to_csc())
}

pub fn fgn_precision_csc(n: usize, hurst: f64, tau: f64) -> Result<CscMatrix, String> {
    if n == 0 {
        return Err("fgn requires n >= 1".to_string());
    }
    if hurst <= 0.0 || hurst >= 1.0 {
        return Err("hurst parameter must be in (0, 1)".to_string());
    }
    if tau <= 0.0 || !tau.is_finite() {
        return Err("tau must be finite and > 0".to_string());
    }

    // Toeplitz FGN covariance: Sigma_{i,j} = gamma_H(|i-j|) / tau
    // gamma_H(0) = 1, gamma_H(k) = 0.5 * ((k+1)^{2H} - 2k^{2H} + (k-1)^{2H})
    let h2 = 2.0 * hurst;
    let mut acf = vec![0.0; n];
    acf[0] = 1.0 / tau;
    for k in 1..n {
        let k_f = k as f64;
        acf[k] = 0.5 * ((k_f + 1.0).powf(h2) - 2.0 * k_f.powf(h2) + (k_f - 1.0).powf(h2)) / tau;
    }

    let mut cov = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let v = acf[i - j];
            cov[i * n + j] = v;
            cov[j * n + i] = v;
        }
    }

    // Covariance is SPD: Cholesky inversion is O(n³) and more stable than GE.
    let prec = invert_spd_cholesky(&cov, n)?;

    let mut tri = TriMatI::<f64, usize>::with_capacity((n, n), n * n);
    for i in 0..n {
        for j in 0..n {
            let val = prec[i * n + j];
            if val.is_finite() {
                tri.add_triplet(i, j, val);
            }
        }
    }

    Ok(tri.to_csc())
}

/// Invert an SPD matrix via Cholesky: A = L Lᵀ ⇒ A⁻¹ = L⁻ᵀ L⁻¹.
fn invert_spd_cholesky(a: &[f64], n: usize) -> Result<Vec<f64>, String> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if !(s > 1e-15) {
                    return Err("FGN covariance is not positive definite".to_string());
                }
                l[i * n + j] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }

    // Y = L⁻¹ (lower triangular)
    let mut y = vec![0.0; n * n];
    for i in 0..n {
        y[i * n + i] = 1.0 / l[i * n + i];
        for j in 0..i {
            let mut s = 0.0;
            for k in j..i {
                s -= l[i * n + k] * y[k * n + j];
            }
            y[i * n + j] = s / l[i * n + i];
        }
    }

    // A⁻¹ = Yᵀ Y
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = 0.0;
            for k in i..n {
                s += y[k * n + i] * y[k * n + j];
            }
            inv[i * n + j] = s;
            inv[j * n + i] = s;
        }
    }
    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_from_csc(csc: &CscMatrix) -> Vec<f64> {
        let nrow = csc.rows();
        let ncol = csc.cols();
        let mut out = vec![0.0; nrow * ncol];
        for (col, vec) in csc.outer_iterator().enumerate() {
            for (row, v) in vec.iter() {
                out[row * ncol + col] = *v;
            }
        }
        out
    }

    fn get(m: &[f64], n: usize, i: usize, j: usize) -> f64 {
        m[i * n + j]
    }

    fn approx(a: f64, b: f64, eps: f64) {
        assert!((a - b).abs() < eps, "left={a}, right={b}, eps={eps}");
    }

    #[test]
    fn builds_rw1_precision_csc() {
        let q = rw1_precision_csc(5, 1.0).expect("rw1");
        assert_eq!(q.rows(), 5);
        assert_eq!(q.cols(), 5);
        assert_eq!(q.nnz(), 13);
        let d = dense_from_csc(&q);
        approx(get(&d, 5, 0, 0), 1.0, 1e-12);
        approx(get(&d, 5, 1, 1), 2.0, 1e-12);
        approx(get(&d, 5, 4, 4), 1.0, 1e-12);
        approx(get(&d, 5, 0, 1), -1.0, 1e-12);
        approx(get(&d, 5, 1, 0), -1.0, 1e-12);
    }

    #[test]
    fn builds_rw2_precision_csc() {
        let q = rw2_precision_csc(6, 1.0).expect("rw2");
        assert_eq!(q.rows(), 6);
        assert_eq!(q.cols(), 6);
        assert_eq!(q.nnz(), 24);
        let d = dense_from_csc(&q);
        let diag = [1.0, 5.0, 6.0, 6.0, 5.0, 1.0];
        for (k, want) in diag.iter().enumerate() {
            approx(get(&d, 6, k, k), *want, 1e-12);
        }
        approx(get(&d, 6, 0, 1), -2.0, 1e-12);
        approx(get(&d, 6, 1, 2), -4.0, 1e-12);
        approx(get(&d, 6, 0, 2), 1.0, 1e-12);
        approx(get(&d, 6, 3, 5), 1.0, 1e-12);
    }

    #[test]
    fn builds_rw1_cyclic_precision_csc() {
        let q = rw1_cyclic_precision_csc(12, 1.0).expect("rw1 cyclic");
        assert_eq!(q.rows(), 12);
        assert_eq!(q.cols(), 12);
        assert_eq!(q.nnz(), 36);
        let d = dense_from_csc(&q);
        approx(get(&d, 12, 0, 0), 2.0, 1e-12);
        approx(get(&d, 12, 0, 1), -1.0, 1e-12);
        approx(get(&d, 12, 0, 11), -1.0, 1e-12);
        approx(get(&d, 12, 11, 0), -1.0, 1e-12);
    }

    #[test]
    fn builds_rw2_cyclic_precision_csc() {
        let q = rw2_cyclic_precision_csc(8, 1.0).expect("rw2 cyclic");
        assert_eq!(q.rows(), 8);
        assert_eq!(q.cols(), 8);
        assert_eq!(q.nnz(), 40); // 5 elements per row for n=8
        let d = dense_from_csc(&q);
        approx(get(&d, 8, 0, 0), 6.0, 1e-12);
        approx(get(&d, 8, 0, 1), -4.0, 1e-12);
        approx(get(&d, 8, 0, 7), -4.0, 1e-12);
        approx(get(&d, 8, 0, 2), 1.0, 1e-12);
        approx(get(&d, 8, 0, 6), 1.0, 1e-12);
    }

    #[test]
    fn builds_seasonal_precision_csc() {
        // Cyclic seasonal: s=3, cyclic=true
        let q_cyclic = seasonal_precision_csc(6, 3, 1.0, true).expect("seasonal cyclic");
        assert_eq!(q_cyclic.rows(), 6);
        assert_eq!(q_cyclic.cols(), 6);
        let d_cyclic = dense_from_csc(&q_cyclic);
        // diag: s-0 = 3
        approx(get(&d_cyclic, 6, 0, 0), 3.0, 1e-12);
        // off-1: s-1 = 2
        approx(get(&d_cyclic, 6, 0, 1), 2.0, 1e-12);
        approx(get(&d_cyclic, 6, 0, 5), 2.0, 1e-12);
        // off-2: s-2 = 1
        approx(get(&d_cyclic, 6, 0, 2), 1.0, 1e-12);
        approx(get(&d_cyclic, 6, 0, 4), 1.0, 1e-12);
        // off-3: s-3 = 0 -> 0.0
        approx(get(&d_cyclic, 6, 0, 3), 0.0, 1e-12);

        // Non-cyclic seasonal: s=3, cyclic=false, n=6
        let q_noncyclic = seasonal_precision_csc(6, 3, 1.0, false).expect("seasonal non-cyclic");
        assert_eq!(q_noncyclic.rows(), 6);
        let d_noncyclic = dense_from_csc(&q_noncyclic);
        // Diagonals (i==j, diff=0, imin=i):
        // i=0: imin=0 <= s-diff-1 (3-0-1 = 2) -> val = imin+1 = 1
        approx(get(&d_noncyclic, 6, 0, 0), 1.0, 1e-12);
        // i=1: imin=1 <= 2 -> val = imin+1 = 2
        approx(get(&d_noncyclic, 6, 1, 1), 2.0, 1e-12);
        // i=2: imin=2 <= 2 -> val = imin+1 = 3
        approx(get(&d_noncyclic, 6, 2, 2), 3.0, 1e-12);
        // i=3: imin=3 > 2, imin < n-s (6-3 = 3) -> false (imin==3), imin >= n-s (3>=3) -> true -> val = n - diff - imin = 6 - 0 - 3 = 3
        approx(get(&d_noncyclic, 6, 3, 3), 3.0, 1e-12);
        // i=4: imin=4 >= 3 -> true -> val = 6 - 0 - 4 = 2
        approx(get(&d_noncyclic, 6, 4, 4), 2.0, 1e-12);
        // i=5: imin=5 >= 3 -> true -> val = 6 - 0 - 5 = 1
        approx(get(&d_noncyclic, 6, 5, 5), 1.0, 1e-12);
    }

    #[test]
    fn builds_two_diid_precision_csc() {
        let q = two_diid_precision_csc(3, 0.3, 2.0).expect("2diid");
        assert_eq!(q.rows(), 6);
        assert_eq!(q.cols(), 6);
        assert_eq!(q.nnz(), 12);
        let d = dense_from_csc(&q);
        let s = 2.0 / (1.0 - 0.09);
        approx(get(&d, 6, 0, 0), s, 1e-12);
        approx(get(&d, 6, 0, 1), -0.3 * s, 1e-12);
        approx(get(&d, 6, 2, 3), -0.3 * s, 1e-12);
        approx(get(&d, 6, 1, 2), 0.0, 1e-12);
    }

    #[test]
    fn builds_iid_precision_csc() {
        let q = iid_precision_csc(5, 3.5).expect("iid");
        assert_eq!(q.rows(), 5);
        assert_eq!(q.cols(), 5);
        assert_eq!(q.nnz(), 5);
        let d = dense_from_csc(&q);
        approx(get(&d, 5, 0, 0), 3.5, 1e-12);
        approx(get(&d, 5, 1, 1), 3.5, 1e-12);
        approx(get(&d, 5, 0, 1), 0.0, 1e-12);
    }

    #[test]
    fn builds_fgn_precision_csc() {
        // H = 0.5 behaves exactly like independent observations
        let q_ind = fgn_precision_csc(4, 0.5, 2.0).expect("fgn H=0.5");
        assert_eq!(q_ind.rows(), 4);
        assert_eq!(q_ind.cols(), 4);
        let d_ind = dense_from_csc(&q_ind);
        for i in 0..4 {
            for j in 0..4 {
                let val = get(&d_ind, 4, i, j);
                if i == j {
                    approx(val, 2.0, 1e-12);
                } else {
                    approx(val, 0.0, 1e-12);
                }
            }
        }

        // H = 0.7 has positive correlations, so off-diagonals of precision will be non-zero
        let q_dep = fgn_precision_csc(3, 0.7, 1.0).expect("fgn H=0.7");
        assert_eq!(q_dep.rows(), 3);
        assert_eq!(q_dep.cols(), 3);
    }
}
