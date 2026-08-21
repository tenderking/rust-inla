//! Correlated iid random effects (`iid2d` … `iid5d`).
//!
//! Layout matches R-INLA: latent length `n = d * m` is stacked by component
//!
//! `(u_1,…,u_m, v_1,…,v_m, …)` so `Q = W ⊗ I_m` with `W = Σ^{-1}`.
//!
//! `Σ` is parameterized by **marginal** precisions `τ_i` and correlations `ρ_ij`
//! (not the entries of `W`). Internal θ is
//! `(log τ_1,…,log τ_d, ρ̃_12, ρ̃_13, …)` with `ρ = 2/(1+e^{-ρ̃}) − 1`.

use inla_math::{CscMatrix, invert_spd_cholesky, sparse_from_triplets};

use crate::inference::log_gamma;

/// Dimension `d ∈ {2,3,4,5}` for `iid2d`…`iid5d`.
pub fn iidkd_dim(model: &str) -> Option<usize> {
    match model.trim().to_ascii_lowercase().as_str() {
        "iid2d" => Some(2),
        "iid3d" => Some(3),
        "iid4d" => Some(4),
        "iid5d" => Some(5),
        _ => None,
    }
}

/// Number of unique `(τ, ρ)` coordinates: `d(d+1)/2`.
pub fn iidkd_nparam(dim: usize) -> usize {
    dim * (dim + 1) / 2
}

/// Default Wishart `(r, vech(R))` for dimension `d` (R-INLA `param=`).
///
/// Packed `R` is diagonals first, then upper triangle `i < j`.
pub fn iidkd_default_wishart_param(dim: usize) -> Vec<f64> {
    let n = iidkd_nparam(dim);
    let mut p = vec![0.0; n + 1];
    // df: 4, 7, 11, 16 for d = 2,3,4,5 (R-INLA defaults).
    p[0] = match dim {
        2 => 4.0,
        3 => 7.0,
        4 => 11.0,
        5 => 16.0,
        _ => (dim * (dim + 1)) as f64 / 2.0 + 1.0,
    };
    for i in 0..dim {
        p[1 + i] = 1.0;
    }
    p
}

/// `Q(θ) = W(θ) ⊗ I_m` for a length-`n` field (`n` must be divisible by `dim`).
pub fn iidkd_precision_csc(n: usize, dim: usize, theta: &[f64]) -> Result<CscMatrix, String> {
    if !(2..=5).contains(&dim) {
        return Err(format!("iidkd: dim must be 2..=5, got {dim}"));
    }
    if n == 0 || !n.is_multiple_of(dim) {
        return Err(format!(
            "iid{dim}d: latent length n={n} must be positive and divisible by {dim}"
        ));
    }
    let need = iidkd_nparam(dim);
    if theta.len() != need {
        return Err(format!(
            "iid{dim}d: θ length {} != expected {need}",
            theta.len()
        ));
    }
    let w = precision_from_theta(dim, theta)?;
    let n_units = n / dim;
    let mut entries = Vec::with_capacity(dim * dim * n_units);
    for u in 0..n_units {
        for a in 0..dim {
            for b in 0..dim {
                let val = w[a * dim + b];
                if val != 0.0 {
                    entries.push((a * n_units + u, b * n_units + u, val));
                }
            }
        }
    }
    Ok(sparse_from_triplets(n, n, &entries))
}

/// Build `W = Σ^{-1}` from internal θ. `Σ_ii = 1/τ_i`, `Σ_ij = ρ_ij / √(τ_i τ_j)`.
pub fn precision_from_theta(dim: usize, theta: &[f64]) -> Result<Vec<f64>, String> {
    let mut natural = natural_from_theta(dim, theta)?;
    adjust_correlations_spd(dim, &mut natural)?;
    covariance_from_natural(dim, &natural).and_then(|s| invert_spd(dim, &s))
}

fn natural_from_theta(dim: usize, theta: &[f64]) -> Result<Vec<f64>, String> {
    let n = iidkd_nparam(dim);
    if theta.len() != n {
        return Err(format!(
            "iidkd: θ length {} != {n} for dim={dim}",
            theta.len()
        ));
    }
    let mut x = vec![0.0; n];
    for i in 0..dim {
        let tau = theta[i].exp();
        if !(tau > 0.0 && tau.is_finite()) {
            return Err(format!("iidkd: non-finite precision τ_{}", i + 1));
        }
        x[i] = tau;
    }
    for k in dim..n {
        // ρ = 2/(1+e^{-θ}) − 1 = tanh(θ/2)
        let rho = 2.0 / (1.0 + (-theta[k]).exp()) - 1.0;
        if !rho.is_finite() || rho.abs() >= 1.0 {
            return Err("iidkd: correlation must be in (-1, 1)".into());
        }
        x[k] = rho;
    }
    Ok(x)
}

/// Shrink correlations by 0.95 until `Σ` is SPD (R-INLA `inla_iid_wishart_adjust`).
fn adjust_correlations_spd(dim: usize, x: &mut [f64]) -> Result<(), String> {
    const F: f64 = 0.95;
    let n = iidkd_nparam(dim);
    for _ in 0..80 {
        match covariance_from_natural(dim, x).and_then(|s| invert_spd(dim, &s)) {
            Ok(_) => return Ok(()),
            Err(_) => {
                for v in x.iter_mut().skip(dim).take(n - dim) {
                    *v *= F;
                }
            }
        }
    }
    Err("iidkd: correlation matrix is not positive definite".into())
}

fn covariance_from_natural(dim: usize, x: &[f64]) -> Result<Vec<f64>, String> {
    let mut s = vec![0.0; dim * dim];
    let mut k = 0usize;
    for i in 0..dim {
        if !(x[k] > 0.0 && x[k].is_finite()) {
            return Err(format!("iidkd: τ_{} must be positive", i + 1));
        }
        s[i * dim + i] = 1.0 / x[k];
        k += 1;
    }
    for i in 0..dim {
        for j in (i + 1)..dim {
            let rho = x[k];
            k += 1;
            let cov = rho / (x[i] * x[j]).sqrt();
            s[i * dim + j] = cov;
            s[j * dim + i] = cov;
        }
    }
    Ok(s)
}

fn invert_spd(dim: usize, a: &[f64]) -> Result<Vec<f64>, String> {
    invert_spd_cholesky(a, dim).map_err(|e| e.to_string())
}

fn spd_logdet(a: &[f64], n: usize) -> Result<f64, String> {
    // In-place Cholesky; logdet = 2 ∑ log L_ii.
    let mut l = a.to_vec();
    for i in 0..n {
        for j in 0..=i {
            let mut sum = l[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    return Err("matrix is not positive definite".into());
                }
                l[i * n + i] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
        for j in (i + 1)..n {
            l[i * n + j] = 0.0;
        }
    }
    Ok(2.0 * (0..n).map(|i| l[i * n + i].ln()).sum::<f64>())
}

fn trace_prod(a: &[f64], b: &[f64], n: usize) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).take(n * n).sum()
}

fn unpack_scale_r(dim: usize, param: &[f64]) -> Result<(f64, Vec<f64>), String> {
    let need = iidkd_nparam(dim) + 1;
    if param.len() < need {
        return Err(format!(
            "wishart{dim}d: param length {} < {need} (r plus packed R)",
            param.len()
        ));
    }
    let r = param[0];
    let mut rmat = vec![0.0; dim * dim];
    let mut k = 1usize;
    for i in 0..dim {
        rmat[i * dim + i] = param[k];
        k += 1;
    }
    for i in 0..dim {
        for j in (i + 1)..dim {
            rmat[i * dim + j] = param[k];
            rmat[j * dim + i] = param[k];
            k += 1;
        }
    }
    Ok((r, rmat))
}

/// `log π(Q)` for `Q ~ Wishart_p(r, R^{-1})` (R-INLA / iid.pdf).
fn wishart_logdens_q(q: &[f64], dim: usize, r: f64, rmat: &[f64]) -> Result<f64, String> {
    let logdet_q = spd_logdet(q, dim)?;
    let logdet_r = spd_logdet(rmat, dim)?;
    let tr = trace_prod(q, rmat, dim);
    let p = dim as f64;
    let mut log_c = 0.5 * r * p * std::f64::consts::LN_2 - 0.5 * r * logdet_r
        + 0.25 * p * (p - 1.0) * std::f64::consts::PI.ln();
    for j in 1..=dim {
        log_c += log_gamma((r + 1.0 - j as f64) / 2.0);
    }
    Ok(0.5 * (r - p - 1.0) * logdet_q - 0.5 * tr - log_c)
}

fn vech_q(q: &[f64], dim: usize) -> Vec<f64> {
    let mut v = Vec::with_capacity(iidkd_nparam(dim));
    for i in 0..dim {
        v.push(q[i * dim + i]);
    }
    for i in 0..dim {
        for j in (i + 1)..dim {
            v.push(q[i * dim + j]);
        }
    }
    v
}

fn logabsdet(mat: &[f64], n: usize) -> Result<f64, String> {
    // LU with partial pivoting (in-place copy).
    let mut a = mat.to_vec();
    let mut logdet = 0.0;
    for i in 0..n {
        let mut piv = i;
        let mut best = a[i * n + i].abs();
        for r in (i + 1)..n {
            let v = a[r * n + i].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-18 {
            return Err("jacobian is singular".into());
        }
        if piv != i {
            for c in 0..n {
                a.swap(i * n + c, piv * n + c);
            }
        }
        let diag = a[i * n + i];
        logdet += diag.abs().ln();
        for r in (i + 1)..n {
            let f = a[r * n + i] / diag;
            a[r * n + i] = f;
            for c in (i + 1)..n {
                a[r * n + c] -= f * a[i * n + c];
            }
        }
    }
    Ok(logdet)
}

/// Log-density of the Wishart prior on **natural** `(τ, ρ)`, including `|d vech(W)/d(τ,ρ)|`.
///
/// Matches `priorfunc_wishart_generic` in `inla-priors.c`.
pub fn wishart_logdens_natural(dim: usize, natural: &[f64], param: &[f64]) -> Result<f64, String> {
    let n = iidkd_nparam(dim);
    if natural.len() != n {
        return Err(format!(
            "wishart{dim}d: natural length {} != {n}",
            natural.len()
        ));
    }
    let mut x = natural.to_vec();
    let fail = adjust_correlations_spd(dim, &mut x).is_err();
    let sigma = covariance_from_natural(dim, &x)?;
    let q = invert_spd(dim, &sigma)?;
    let (r, rmat) = unpack_scale_r(dim, param)?;
    let mut val = wishart_logdens_q(&q, dim, r, &rmat)?;
    if fail {
        val -= 1.0e8;
    }

    // Numerical Jacobian of vech(W) wrt (τ, ρ), as in R-INLA.
    let logdet_q = spd_logdet(&q, dim)?;
    let mut f = 1.0e-6 * (-logdet_q / dim as f64).exp();
    for i in 0..dim {
        f = f.min(0.5 * x[i]);
    }
    if !(f > 0.0 && f.is_finite()) {
        f = 1e-8;
    }

    let mut jac = vec![0.0; n * n];
    for ii in 0..n {
        let save = x[ii];
        x[ii] = save + f;
        let q_plus = invert_spd(dim, &covariance_from_natural(dim, &x)?)?;
        x[ii] = save - f;
        let q_minus = invert_spd(dim, &covariance_from_natural(dim, &x)?)?;
        x[ii] = save;
        let vp = vech_q(&q_plus, dim);
        let vm = vech_q(&q_minus, dim);
        for k in 0..n {
            jac[ii * n + k] = (vp[k] - vm[k]) / (2.0 * f);
        }
    }
    val += logabsdet(&jac, n)?;
    Ok(val)
}

/// Log-density on **internal** θ, including the natural→internal Jacobian.
pub fn wishart_logdens_theta(dim: usize, theta: &[f64], param: &[f64]) -> Result<f64, String> {
    let natural = natural_from_theta(dim, theta)?;
    let mut ld = wishart_logdens_natural(dim, &natural, param)?;
    // τ = e^θ ⇒ log|dτ/dθ| = θ
    for i in 0..dim {
        ld += theta[i];
    }
    // ρ = 2/(1+e^{-θ})−1 ⇒ dρ/dθ = (1−ρ²)/2
    for k in dim..iidkd_nparam(dim) {
        let rho = natural[k];
        ld += ((1.0 - rho * rho) * 0.5).ln();
    }
    Ok(ld)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get(m: &[f64], n: usize, i: usize, j: usize) -> f64 {
        m[i * n + j]
    }

    fn dense(csc: &CscMatrix) -> Vec<f64> {
        let n = csc.rows();
        let mut out = vec![0.0; n * n];
        for (col, vec) in csc.outer_iterator().enumerate() {
            for (row, v) in vec.iter() {
                out[row * n + col] = *v;
            }
        }
        out
    }

    #[test]
    fn nparam_and_dim() {
        assert_eq!(iidkd_dim("iid2d"), Some(2));
        assert_eq!(iidkd_dim("IID3D"), Some(3));
        assert_eq!(iidkd_nparam(2), 3);
        assert_eq!(iidkd_nparam(3), 6);
        assert_eq!(iidkd_nparam(5), 15);
        assert_eq!(iidkd_default_wishart_param(2), vec![4.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn precision_matches_closed_form_2d() {
        // τa=1, τb=4, ρ=0.5 ⇒ θ = (0, log 4, logit((1+0.5)/2))
        let tau_a = 1.0_f64;
        let tau_b = 4.0_f64;
        let rho = 0.5_f64;
        let t3 = ((1.0 + rho) / (1.0 - rho)).ln();
        let theta = [tau_a.ln(), tau_b.ln(), t3];
        let w = precision_from_theta(2, &theta).unwrap();
        let s = 1.0 / (1.0 - rho * rho);
        let w11 = tau_a * s;
        let w22 = tau_b * s;
        let w12 = -rho * (tau_a * tau_b).sqrt() * s;
        assert!((w[0] - w11).abs() < 1e-10, "W11 {} vs {w11}", w[0]);
        assert!((w[3] - w22).abs() < 1e-10, "W22 {} vs {w22}", w[3]);
        assert!((w[1] - w12).abs() < 1e-10, "W12 {} vs {w12}", w[1]);
        assert!((w[2] - w12).abs() < 1e-10);
    }

    #[test]
    fn kronecker_layout_two_units() {
        let theta = [0.0, 0.0, 0.0]; // τ=1,1 ρ=0 ⇒ W = I_2
        let q = iidkd_precision_csc(4, 2, &theta).unwrap();
        let d = dense(&q);
        // Identity 4×4
        for i in 0..4 {
            for j in 0..4 {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (get(&d, 4, i, j) - expect).abs() < 1e-12,
                    "Q[{i},{j}] = {}",
                    get(&d, 4, i, j)
                );
            }
        }
    }

    #[test]
    fn correlated_block_repeats_across_units() {
        let rho = 0.3_f64;
        let t3 = ((1.0 + rho) / (1.0 - rho)).ln();
        let theta = [0.0, 0.0, t3];
        let w = precision_from_theta(2, &theta).unwrap();
        let q = iidkd_precision_csc(6, 2, &theta).unwrap(); // m=3
        let d = dense(&q);
        // Q[u, u] = W11, Q[u+3, u+3] = W22, Q[u, u+3] = W12
        for u in 0..3 {
            assert!((get(&d, 6, u, u) - w[0]).abs() < 1e-10);
            assert!((get(&d, 6, u + 3, u + 3) - w[3]).abs() < 1e-10);
            assert!((get(&d, 6, u, u + 3) - w[1]).abs() < 1e-10);
            assert!((get(&d, 6, u + 3, u) - w[1]).abs() < 1e-10);
        }
        // units do not couple
        assert!(get(&d, 6, 0, 1).abs() < 1e-12);
    }

    #[test]
    fn wishart_density_finite_at_default() {
        let theta = [4.0, 4.0, 0.0];
        let p = iidkd_default_wishart_param(2);
        let ld = wishart_logdens_theta(2, &theta, &p).unwrap();
        assert!(ld.is_finite(), "logdens={ld}");
    }

    #[test]
    fn wishart_prefers_uncorrelated_when_r_offdiag_zero() {
        // Default R = I ⇒ E[W] ∝ I, so ρ=0 should beat |ρ| large.
        let p = iidkd_default_wishart_param(2);
        let uncorr = wishart_logdens_theta(2, &[0.0, 0.0, 0.0], &p).unwrap();
        let t_hi = ((1.0_f64 + 0.9) / (1.0 - 0.9)).ln();
        let corr = wishart_logdens_theta(2, &[0.0, 0.0, t_hi], &p).unwrap();
        assert!(
            uncorr > corr,
            "uncorrelated {uncorr} should exceed rho=0.9 {corr}"
        );
    }
}
