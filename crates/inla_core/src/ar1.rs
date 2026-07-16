pub struct Ar1Precision {
    pub n: usize,
    pub row_major_values: Vec<f64>,
    pub i: Vec<usize>,
    pub j: Vec<usize>,
    pub x: Vec<f64>,
}

pub fn ar1_precision(n: usize, rho: f64, tau: f64) -> Result<Ar1Precision, String> {
    if n < 2 {
        return Err("n must be at least 2".to_string());
    }
    if !rho.is_finite() || !tau.is_finite() {
        return Err("rho and tau must be finite".to_string());
    }
    if rho.abs() >= 1.0 {
        return Err("rho must satisfy |rho| < 1".to_string());
    }
    if tau <= 0.0 {
        return Err("tau must be positive".to_string());
    }

    let mut row_major_values = vec![0.0; n * n];
    let mut i = Vec::with_capacity(3 * n - 2);
    let mut j = Vec::with_capacity(3 * n - 2);
    let mut x = Vec::with_capacity(3 * n - 2);

    let off_diag = -tau * rho;
    let diag_inner = tau * (1.0 + rho * rho);

    for r in 0..n {
        let diag = if r == 0 || r == n - 1 {
            tau
        } else {
            diag_inner
        };
        row_major_values[r * n + r] = diag;
        i.push(r + 1);
        j.push(r + 1);
        x.push(diag);

        if r + 1 < n {
            row_major_values[r * n + (r + 1)] = off_diag;
            row_major_values[(r + 1) * n + r] = off_diag;

            i.push(r + 1);
            j.push(r + 2);
            x.push(off_diag);

            i.push(r + 2);
            j.push(r + 1);
            x.push(off_diag);
        }
    }

    Ok(Ar1Precision {
        n,
        row_major_values,
        i,
        j,
        x,
    })
}
