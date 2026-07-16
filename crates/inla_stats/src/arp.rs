use inla_math::CscMatrix;
use sprs::TriMatI;

pub fn ar_pacf2phi(pacf: &[f64]) -> Result<Vec<f64>, String> {
    let p = pacf.len();
    if p == 0 {
        return Err("ar_pacf2phi requires p > 0".to_string());
    }
    let mut phi = pacf.to_vec();
    let mut work = pacf.to_vec();

    for j in 1..p {
        let a = phi[j];
        for k in 0..j {
            work[k] -= a * phi[j - k - 1];
        }
        for k in 0..j {
            phi[k] = work[k];
        }
    }
    Ok(phi)
}

pub fn ar_phi2pacf(phi: &[f64]) -> Result<Vec<f64>, String> {
    let p = phi.len();
    if p == 0 {
        return Err("ar_phi2pacf requires p > 0".to_string());
    }
    let mut pacf = phi.to_vec();
    let mut work = phi.to_vec();

    for j in (1..p).rev() {
        let a = pacf[j];
        let denom = 1.0 - a * a;
        if denom.abs() < 1e-14 {
            return Err("Singular denominator in ar_phi2pacf (stationary constraint violated)".to_string());
        }
        for k in 0..j {
            work[k] = (pacf[k] + a * pacf[j - k - 1]) / denom;
        }
        for k in 0..j {
            pacf[k] = work[k];
        }
    }
    Ok(pacf)
}

fn solve_linear_system(a: &mut [f64], b: &mut [f64], n: usize) -> Result<Vec<f64>, String> {
    for i in 0..n {
        let mut max_row = i;
        for r in (i + 1)..n {
            if a[r * n + i].abs() > a[max_row * n + i].abs() {
                max_row = r;
            }
        }
        if a[max_row * n + i].abs() < 1e-14 {
            return Err("Singular matrix in Yule-Walker solver".to_string());
        }
        if max_row != i {
            for c in 0..n {
                a.swap(i * n + c, max_row * n + c);
            }
            b.swap(i, max_row);
        }
        for r in (i + 1)..n {
            let factor = a[r * n + i] / a[i * n + i];
            b[r] -= factor * b[i];
            for c in i..n {
                a[r * n + c] -= factor * a[i * n + c];
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i * n + j] * x[j];
        }
        x[i] = sum / a[i * n + i];
    }
    Ok(x)
}

fn invert_matrix(matrix: &[f64], n: usize) -> Result<Vec<f64>, String> {
    let mut inv = vec![0.0; n * n];
    for col in 0..n {
        let mut b = vec![0.0; n];
        b[col] = 1.0;
        let mut temp_a = matrix.to_vec();
        let x = solve_linear_system(&mut temp_a, &mut b, n)?;
        for row in 0..n {
            inv[row * n + col] = x[row];
        }
    }
    Ok(inv)
}

pub fn ar_marginal_distribution(pacf: &[f64]) -> Result<(f64, Vec<f64>), String> {
    let p = pacf.len();
    if p == 0 {
        return Err("ar_marginal_distribution requires p > 0".to_string());
    }
    let phi = ar_pacf2phi(pacf)?;

    let mut a = vec![0.0; p * p];
    let mut b = vec![0.0; p];

    for i in 0..p {
        for j in 0..p {
            if i == j {
                a[i * p + j] = -1.0;
            } else {
                let lag = (i as isize - j as isize).abs() as usize;
                let lag_idx = lag - 1;
                a[i * p + lag_idx] += phi[j];
            }
        }
        b[i] = -phi[i];
    }

    let x = solve_linear_system(&mut a, &mut b, p)?;

    let mut sigma = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            if i == j {
                sigma[i * p + j] = 1.0;
            } else {
                let lag = (i as isize - j as isize).abs() as usize;
                let lag_idx = lag - 1;
                sigma[i * p + j] = x[lag_idx];
            }
        }
    }

    let sigma_inv = invert_matrix(&sigma, p)?;

    let mut prec = 1.0;
    for i in 0..p {
        prec -= phi[i] * x[i];
    }

    Ok((prec, sigma_inv))
}

pub fn arp_precision_csc(n: usize, pacf: &[f64], tau: f64) -> Result<CscMatrix, String> {
    let p = pacf.len();
    if p == 0 {
        return Err("AR(p) requires at least one PACF parameter".to_string());
    }
    if n < p {
        return Err(format!("n ({}) must be >= p ({})", n, p));
    }
    if tau <= 0.0 || !tau.is_finite() {
        return Err("AR(p) tau must be finite and > 0".to_string());
    }
    for &val in pacf {
        if !val.is_finite() || val.abs() >= 1.0 {
            return Err("PACF parameters must be in (-1, 1)".to_string());
        }
    }

    let phi = ar_pacf2phi(pacf)?;
    let (prec, r_p_inv) = ar_marginal_distribution(pacf)?;

    let mut tri = TriMatI::<f64, usize>::with_capacity((n, n), n * (p + 1));

    for i in 0..p {
        for j in 0..p {
            tri.add_triplet(i, j, r_p_inv[i * p + j] * tau);
        }
    }

    for t in p..n {
        let mut idx = vec![0; p + 1];
        let mut coeffs = vec![0.0; p + 1];
        idx[0] = t;
        coeffs[0] = 1.0;
        for k in 1..=p {
            idx[k] = t - k;
            coeffs[k] = -phi[k - 1];
        }

        for a in 0..=p {
            for b in 0..=p {
                tri.add_triplet(idx[a], idx[b], (coeffs[a] * coeffs[b] / prec) * tau);
            }
        }
    }

    Ok(tri.to_csc())
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
    fn test_pacf_phi_conversion() {
        let pacf = [0.5, -0.3];
        let phi = ar_pacf2phi(&pacf).unwrap();
        let recon_pacf = ar_phi2pacf(&phi).unwrap();
        approx(recon_pacf[0], pacf[0], 1e-12);
        approx(recon_pacf[1], pacf[1], 1e-12);
    }

    #[test]
    fn test_arp_vs_ar1() {
        // ar1_precision is parameterized by innovation precision
        // arp_precision is parameterized by marginal precision
        // innovation_prec = marginal_prec / (1 - rho^2)
        let rho = 0.7;
        let tau_marginal = 2.0;
        let tau_innovation = tau_marginal / (1.0 - rho * rho);
        let q_ar1 = crate::ar1_precision_csc(10, rho, tau_innovation).unwrap();
        let q_arp = arp_precision_csc(10, &[rho], tau_marginal).unwrap();
        assert_eq!(q_ar1.rows(), q_arp.rows());
        assert_eq!(q_ar1.cols(), q_arp.cols());
        assert_eq!(q_ar1.nnz(), q_arp.nnz());

        let d_ar1 = dense_from_csc(&q_ar1);
        let d_arp = dense_from_csc(&q_arp);
        for i in 0..10 {
            for j in 0..10 {
                approx(get(&d_ar1, 10, i, j), get(&d_arp, 10, i, j), 1e-12);
            }
        }
    }

    #[test]
    fn test_ar2_properties() {
        let q = arp_precision_csc(5, &[0.5, -0.3], 1.0).unwrap();
        assert_eq!(q.rows(), 5);
        assert_eq!(q.cols(), 5);
        let d = dense_from_csc(&q);
        
        // Assert symmetry
        for i in 0..5 {
            for j in 0..5 {
                approx(get(&d, 5, i, j), get(&d, 5, j, i), 1e-12);
            }
        }

        // Bandwidth is p = 2 (so entries with distance > 2 are 0)
        approx(get(&d, 5, 0, 3), 0.0, 1e-12);
        approx(get(&d, 5, 0, 4), 0.0, 1e-12);
        approx(get(&d, 5, 1, 4), 0.0, 1e-12);
    }
}
