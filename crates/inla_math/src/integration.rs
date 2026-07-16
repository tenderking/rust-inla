pub fn ccd_design(m: usize, f0: f64) -> Result<(Vec<Vec<f64>>, Vec<f64>), String> {
    if m == 0 {
        return Err("CCD requires at least 1 factor".to_string());
    }

    let mut points = Vec::new();
    points.push(vec![0.0; m]);

    for i in 0..m {
        let mut p1 = vec![0.0; m];
        p1[i] = f0;
        let mut p2 = vec![0.0; m];
        p2[i] = -f0;
        points.push(p1);
        points.push(p2);
    }

    if m == 1 {
        // No factorial points for m=1
    } else if m <= 4 {
        let num_fact = 1 << m;
        let val = f0 / (m as f64).sqrt();
        for k in 0..num_fact {
            let mut p = vec![0.0; m];
            for i in 0..m {
                let sign = if (k >> i) & 1 == 1 { 1.0 } else { -1.0 };
                p[i] = sign * val;
            }
            points.push(p);
        }
    } else if m == 5 {
        let num_fact = 1 << 4;
        let val = f0 / 5.0f64.sqrt();
        for k in 0..num_fact {
            let mut p = vec![0.0; m];
            let mut sign_prod = 1.0;
            for i in 0..4 {
                let sign = if (k >> i) & 1 == 1 { 1.0 } else { -1.0 };
                p[i] = sign * val;
                sign_prod *= sign;
            }
            p[4] = sign_prod * val;
            points.push(p);
        }
    } else {
        return Err(format!("CCD strategy for m={} is not supported", m));
    }

    let n_exp = points.len();
    
    let w = if n_exp > 1 {
        1.0 / ((n_exp - 1) as f64 * (1.0 + (-0.5 * f0 * f0).exp() * (f0 * f0 / m as f64 - 1.0)))
    } else {
        1.0
    };
    let w_origo = 1.0 - (n_exp - 1) as f64 * w;

    let mut weights = vec![w; n_exp];
    weights[0] = w_origo;

    Ok((points, weights))
}

pub fn grid_design(
    m: usize,
    step: f64,
    threshold: f64,
    evaluator: &dyn Fn(&[f64]) -> f64,
) -> Result<(Vec<Vec<f64>>, Vec<f64>), String> {
    if m == 0 {
        return Err("Grid requires at least 1 factor".to_string());
    }
    if m > 2 {
        return ccd_design(m, 2.5);
    }

    let mut points = Vec::new();
    let center = vec![0.0; m];
    let g0 = evaluator(&center);

    if m == 1 {
        points.push(center);
        let mut k = 1;
        loop {
            let z = vec![k as f64 * step];
            let g = evaluator(&z);
            if g0 - g > threshold {
                break;
            }
            points.push(z);
            k += 1;
        }
        let mut k = 1;
        loop {
            let z = vec![-(k as f64 * step)];
            let g = evaluator(&z);
            if g0 - g > threshold {
                break;
            }
            points.push(z);
            k += 1;
        }
    } else {
        let mut max_i = 0;
        loop {
            let z = vec![(max_i + 1) as f64 * step, 0.0];
            let g = evaluator(&z);
            if g0 - g > threshold {
                break;
            }
            max_i += 1;
        }
        let mut min_i = 0;
        loop {
            let z = vec![-(min_i + 1) as f64 * step, 0.0];
            let g = evaluator(&z);
            if g0 - g > threshold {
                break;
            }
            min_i += 1;
        }
        let mut max_j = 0;
        loop {
            let z = vec![0.0, (max_j + 1) as f64 * step];
            let g = evaluator(&z);
            if g0 - g > threshold {
                break;
            }
            max_j += 1;
        }
        let mut min_j = 0;
        loop {
            let z = vec![0.0, -(min_j + 1) as f64 * step];
            let g = evaluator(&z);
            if g0 - g > threshold {
                break;
            }
            min_j += 1;
        }

        for i in -(min_i as isize)..= (max_i as isize) {
            for j in -(min_j as isize)..= (max_j as isize) {
                let z = vec![i as f64 * step, j as f64 * step];
                let g = evaluator(&z);
                if g0 - g <= threshold {
                    points.push(z);
                }
            }
        }
    }

    let n_exp = points.len();
    let weights = vec![1.0; n_exp];

    Ok((points, weights))
}

/// Invert an `m × m` matrix stored row-major in `h`.
///
/// Uses Gauss–Jordan elimination with partial pivoting on the augmented
/// system `[A | I]`. This is O(m³); the previous implementation re-factorized
/// for every right-hand side and was accidentally O(m⁴).
pub fn invert_symmetric_matrix(h: &[f64], m: usize) -> Result<Vec<f64>, String> {
    if h.len() != m * m {
        return Err(format!(
            "matrix length {} does not match m² = {}",
            h.len(),
            m * m
        ));
    }
    if m == 0 {
        return Ok(Vec::new());
    }

    let cols = 2 * m;
    let mut a = vec![0.0; m * cols];
    for i in 0..m {
        for j in 0..m {
            a[i * cols + j] = h[i * m + j];
        }
        a[i * cols + m + i] = 1.0;
    }

    for i in 0..m {
        let mut max_row = i;
        for r in (i + 1)..m {
            if a[r * cols + i].abs() > a[max_row * cols + i].abs() {
                max_row = r;
            }
        }
        if a[max_row * cols + i].abs() < 1e-14 {
            return Err("Singular Hessian matrix".to_string());
        }
        if max_row != i {
            for c in 0..cols {
                a.swap(i * cols + c, max_row * cols + c);
            }
        }

        let piv = a[i * cols + i];
        for c in 0..cols {
            a[i * cols + c] /= piv;
        }

        for r in 0..m {
            if r == i {
                continue;
            }
            let factor = a[r * cols + i];
            if factor == 0.0 {
                continue;
            }
            for c in 0..cols {
                a[r * cols + c] -= factor * a[i * cols + c];
            }
        }
    }

    let mut inv = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            inv[i * m + j] = a[i * cols + m + j];
        }
    }
    Ok(inv)
}

pub fn jacobi_eigen(matrix: &[f64], m: usize, max_iter: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    let mut a = matrix.to_vec();
    let mut v = vec![0.0; m * m];
    for i in 0..m {
        v[i * m + i] = 1.0;
    }

    for _sweep in 0..max_iter {
        let mut off_diag = 0.0;
        for i in 0..m {
            for j in (i + 1)..m {
                off_diag += a[i * m + j].abs();
            }
        }
        if off_diag < 1e-12 {
            let mut eigenvalues = vec![0.0; m];
            for i in 0..m {
                eigenvalues[i] = a[i * m + i];
            }
            return Ok((eigenvalues, v));
        }

        for p in 0..m {
            for q in (p + 1)..m {
                let apq = a[p * m + q];
                if apq.abs() < 1e-15 {
                    continue;
                }
                let app = a[p * m + p];
                let aqq = a[q * m + q];
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                a[p * m + p] = app - t * apq;
                a[q * m + q] = aqq + t * apq;
                a[p * m + q] = 0.0;
                a[q * m + p] = 0.0;

                for r in 0..m {
                    if r != p && r != q {
                        let arp = a[r * m + p];
                        let arq = a[r * m + q];
                        a[r * m + p] = c * arp - s * arq;
                        a[p * m + r] = a[r * m + p];
                        a[r * m + q] = s * arp + c * arq;
                        a[q * m + r] = a[r * m + q];
                    }
                }

                for r in 0..m {
                    let vrp = v[r * m + p];
                    let vrq = v[r * m + q];
                    v[r * m + p] = c * vrp - s * vrq;
                    v[r * m + q] = s * vrp + c * vrq;
                }
            }
        }
    }
    Err("Jacobi eigenvalue algorithm did not converge".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) {
        assert!((a - b).abs() < eps, "left={a}, right={b}, eps={eps}");
    }

    #[test]
    fn test_ccd_design_points() {
        // m = 1
        let (p1, w1) = ccd_design(1, 2.5).unwrap();
        assert_eq!(p1.len(), 3);
        assert_eq!(p1[0], vec![0.0]);
        approx(p1[1][0], 2.5, 1e-12);
        approx(p1[2][0], -2.5, 1e-12);
        approx(w1.iter().sum(), 1.0, 1e-12);

        // m = 2
        let (p2, w2) = ccd_design(2, 2.5).unwrap();
        assert_eq!(p2.len(), 9);
        approx(w2.iter().sum(), 1.0, 1e-12);

        // m = 3
        let (p3, w3) = ccd_design(3, 2.5).unwrap();
        assert_eq!(p3.len(), 15);
        approx(w3.iter().sum(), 1.0, 1e-12);
    }

    #[test]
    fn test_grid_design_points() {
        let evaluator = |z: &[f64]| -> f64 {
            // spherical Gaussian: -0.5 * z^2
            -0.5 * z.iter().map(|&x| x * x).sum::<f64>()
        };

        // m = 1, step = 1.0, threshold = 2.0 (so z^2 <= 4.0 -> z in -2..2)
        let (p1, _) = grid_design(1, 1.0, 2.0, &evaluator).unwrap();
        // points should be 0.0, 1.0, 2.0, -1.0, -2.0 (5 points)
        assert_eq!(p1.len(), 5);

        // m = 2, step = 1.0, threshold = 2.0 (so z1^2 + z2^2 <= 4.0)
        let (p2, _) = grid_design(2, 1.0, 2.0, &evaluator).unwrap();
        // evaluated points should satisfy z1^2 + z2^2 <= 4.0
        for pt in p2 {
            assert!(pt[0] * pt[0] + pt[1] * pt[1] <= 4.0001);
        }
    }
}
