//! R-INLA-compatible FGN via AR(1) mixture (`order` = 3 or 4).
//!
//! Matches `inlaprog/src/fgn.c` (`Qfunc_fgn`) and coefficient tables from
//! `fgn-tables-{3,4}.h` / `fgn-code.h`. Latent field is length `(order+1)*n` in
//! **mixture-major** order `(z, x_1, …, x_order)` with soft constraint `z ≈ Σ x_i`.
//! Observations / `A` therefore index the first `n` entries (`z`). Sparse LDLᵀ
//! applies a time-major permutation at factorize time so the Cholesky envelope
//! is `O(order)`, without changing this stored layout.

#[path = "fgn_tables.rs"]
mod fgn_tables;

use inla_math::CscMatrix;
use sprs::TriMatI;

/// Map H_intern → H ∈ (1/2, 1), as in R-INLA `from.theta` for FGN.
pub fn fgn_hurst_from_intern(h_intern: f64) -> f64 {
    0.5 + 0.5 / (1.0 + (-h_intern).exp())
}

/// Map H ∈ (1/2, 1) → H_intern.
pub fn fgn_intern_from_hurst(h: f64) -> Result<f64, String> {
    if !(h > 0.5 && h < 1.0) {
        return Err("FGN Hurst (R-INLA param) must be in (0.5, 1)".to_string());
    }
    Ok(((2.0 * h - 1.0) / (2.0 * (1.0 - h))).ln())
}

/// φ and weights for the order-K AR(1) mixture at a given H_intern.
pub fn fgn_ar_coeffs(h_intern: f64, order: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    let (start, by, end, k, len_par, param) = match order {
        3 => (
            fgn_tables::FGN_K3_START,
            fgn_tables::FGN_K3_BY,
            fgn_tables::FGN_K3_END,
            fgn_tables::FGN_K3_ORDER,
            fgn_tables::FGN_K3_LEN_PAR,
            fgn_tables::FGN_K3_PARAM,
        ),
        4 => (
            fgn_tables::FGN_K4_START,
            fgn_tables::FGN_K4_BY,
            fgn_tables::FGN_K4_END,
            fgn_tables::FGN_K4_ORDER,
            fgn_tables::FGN_K4_LEN_PAR,
            fgn_tables::FGN_K4_PARAM,
        ),
        _ => return Err("FGN order must be 3 or 4".to_string()),
    };
    debug_assert_eq!(k, order);

    // Clamp into table range (same hack as fgn-code.h).
    let mut h = h_intern;
    h = h.max(start + by);
    h = h.min(end - by);

    let mut idx = ((h - start) / by).floor() as usize;
    let weight = (h - (start + idx as f64 * by)) / by;
    idx *= len_par;

    let mut fit_par = vec![0.0; len_par];
    for i in 0..len_par {
        fit_par[i] = (1.0 - weight) * param[idx + i] + weight * param[idx + len_par + i];
    }

    let mut phi = vec![0.0; order];
    let mut tmp = 0.0;
    for i in 0..order {
        tmp += (-fit_par[i]).exp();
        phi[i] = 1.0 / (1.0 + tmp);
    }

    let mut par = vec![0.0; order];
    par[0] = 1.0;
    let mut psum = 1.0;
    for i in 1..order {
        par[i] = fit_par[order + (i - 1)].exp();
        psum += par[i];
    }
    let w: Vec<f64> = par.iter().map(|p| p / psum).collect();
    Ok((phi, w))
}

/// Sparse FGN precision of size `(order+1)*n`, R-INLA `model="fgn"`.
///
/// Stored CSC is mixture-major (`z` then each AR component) so projectors that
/// observe the first `n` latents stay valid. Factorization reorders to
/// time-major internally (see `inla_math` sparse LDLᵀ).
///
/// `hurst` must be in (0.5, 1). `prec_eps` defaults to `1e8` (R-INLA `f(..., precision=)`).
pub fn fgn_approx_precision_csc(
    n: usize,
    hurst: f64,
    tau: f64,
    order: usize,
    prec_eps: f64,
) -> Result<CscMatrix, String> {
    if n < 2 {
        return Err("fgn approx requires n >= 2".to_string());
    }
    if !(hurst > 0.5 && hurst < 1.0) {
        return Err("fgn approx hurst must be in (0.5, 1)".to_string());
    }
    if !(tau > 0.0 && tau.is_finite()) {
        return Err("tau must be finite and > 0".to_string());
    }
    if !(prec_eps > 0.0 && prec_eps.is_finite()) {
        return Err("prec_eps must be finite and > 0".to_string());
    }

    let h_intern = fgn_intern_from_hurst(hurst)?;
    let (phi, w) = fgn_ar_coeffs(h_intern, order)?;
    let kappa = prec_eps * tau;
    let n_tot = n * (order + 1);

    // nnz: z diag n; z–x couplings 2*k*n; each AR1 ~ 3n-2 plus kappa on diag
    let nnz_est = n + 2 * order * n + order * (3 * n);
    let mut tri = TriMatI::<f64, usize>::with_capacity((n_tot, n_tot), nnz_est);

    // z-block diagonal: kappa
    for t in 0..n {
        tri.add_triplet(t, t, kappa);
    }

    // Coupling z_t ↔ x_{i,t}: -kappa (symmetric)
    for i in 0..order {
        let base = (i + 1) * n;
        for t in 0..n {
            tri.add_triplet(t, base + t, -kappa);
            tri.add_triplet(base + t, t, -kappa);
        }
    }

    // Cross-coupling x_{i,t} ↔ x_{j,t}: +kappa (from (z - Σ x)^2)
    for i in 0..order {
        for j in (i + 1)..order {
            let bi = (i + 1) * n;
            let bj = (j + 1) * n;
            for t in 0..n {
                tri.add_triplet(bi + t, bj + t, kappa);
                tri.add_triplet(bj + t, bi + t, kappa);
            }
        }
    }

    // AR(1) blocks for each component, scaled by prec/w, plus kappa on diagonal
    for i in 0..order {
        let base = (i + 1) * n;
        let p = phi[i];
        let p2 = p * p;
        let prec_cond = 1.0 / (1.0 - p2);
        let scale = tau / w[i];
        let a = scale * prec_cond; // multiplies the unit-innovation AR1 pattern

        for t in 0..n {
            let diag = if t == 0 || t == n - 1 {
                a * 1.0
            } else {
                a * (1.0 + p2)
            };
            tri.add_triplet(base + t, base + t, diag + kappa);
            if t + 1 < n {
                let off = -a * p;
                tri.add_triplet(base + t, base + t + 1, off);
                tri.add_triplet(base + t + 1, base + t, off);
            }
        }
    }

    Ok(tri.to_csc())
}

/// Total latent dimension for approx FGN.
pub fn fgn_approx_latent_len(n_obs: usize, order: usize) -> usize {
    n_obs * (order + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coeffs_order4_at_h08() {
        let h_intern = fgn_intern_from_hurst(0.8).unwrap();
        let (phi, w) = fgn_ar_coeffs(h_intern, 4).unwrap();
        assert_eq!(phi.len(), 4);
        assert_eq!(w.len(), 4);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        for p in &phi {
            assert!(*p > 0.0 && *p < 1.0);
        }
        // phis should be decreasing
        for i in 1..4 {
            assert!(phi[i] < phi[i - 1]);
        }
    }

    #[test]
    fn builds_approx_q_shape() {
        let q = fgn_approx_precision_csc(10, 0.8, 1.0, 4, 1e8).unwrap();
        assert_eq!(q.rows(), 50);
        assert_eq!(q.cols(), 50);
        assert!(q.nnz() > 0);
    }
}
