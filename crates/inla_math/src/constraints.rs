//! Linear constraints for intrinsic GMRFs (R-INLA extraconstr / sum-to-zero).

use crate::sparse::CscMatrix;
use sprs::TriMatI;

/// Method used for applying hard linear constraints Ax = e.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConstraintMethod {
    /// Q + κ AᵀA augmentation (κ = exp(15)), combined with post-hoc projection.
    #[default]
    Augmented,
    /// Lagrange multiplier elimination inside factorization.
    LagrangeElimination,
}

/// Linear constraints `A x = e` with `A` dense `k × n` (row-major).
#[derive(Debug, Clone, PartialEq)]
pub struct ConstraintSpec {
    pub n: usize,
    pub k: usize,
    /// Row-major `k * n`
    pub a: Vec<f64>,
    pub e: Vec<f64>,
    pub method: ConstraintMethod,
}

impl ConstraintSpec {
    pub fn validate(&self) -> Result<(), String> {
        if self.k == 0 {
            return Err("ConstraintSpec: k must be > 0".into());
        }
        if self.a.len() != self.k * self.n {
            return Err(format!(
                "ConstraintSpec: a length {} != k*n={}",
                self.a.len(),
                self.k * self.n
            ));
        }
        if self.e.len() != self.k {
            return Err(format!(
                "ConstraintSpec: e length {} != k={}",
                self.e.len(),
                self.k
            ));
        }
        Ok(())
    }

    /// Embed block constraints into a larger latent of size `full_n` starting at `offset`.
    pub fn embed(&self, full_n: usize, offset: usize) -> Result<ConstraintSpec, String> {
        self.validate()?;
        if offset + self.n > full_n {
            return Err("ConstraintSpec::embed: block exceeds full dimension".into());
        }
        let mut a = vec![0.0; self.k * full_n];
        for r in 0..self.k {
            for c in 0..self.n {
                a[r * full_n + (offset + c)] = self.a[r * self.n + c];
            }
        }
        Ok(ConstraintSpec {
            n: full_n,
            k: self.k,
            a,
            e: self.e.clone(),
            method: self.method,
        })
    }

    /// Stack two constraint specs on the same `n` (row-concatenate).
    pub fn vstack(self, other: &ConstraintSpec) -> Result<ConstraintSpec, String> {
        self.validate()?;
        other.validate()?;
        if self.n != other.n {
            return Err("ConstraintSpec::vstack: n mismatch".into());
        }
        let k = self.k + other.k;
        let mut a = Vec::with_capacity(k * self.n);
        a.extend_from_slice(&self.a);
        a.extend_from_slice(&other.a);
        let mut e = self.e;
        e.extend_from_slice(&other.e);
        Ok(ConstraintSpec {
            n: self.n,
            k,
            a,
            e,
            method: self.method,
        })
    }
}

/// Rank deficiency / number of sum-to-zero style constraints for a latent model.
pub fn model_rank_deficiency(model: &str) -> usize {
    match model.to_ascii_lowercase().as_str() {
        "rw1" | "besag" | "besag2" | "bym2" => 1,
        "rw2" => 2,
        // Intrinsic (non-cyclic) RW2D: kill the discrete plane (const + row + col);
        // cyclic uses k=1. Callers that know cyclic=true should request k=1 explicitly.
        "rw2d" => 3,
        "seasonal" => 1, // sum-to-zero over the seasonal contrast (common default)
        "bym" => 1,      // spatial ICAR block only (caller embeds on that block)
        "crw1" => 1,
        "crw2" => 2,
        _ => 0,
    }
}

/// Build orthonormal-ish sum-to-zero constraints for a length-`n` field.
///
/// - `k=1`: constant (ones)
/// - `k=2`: constant + linear trend (centered index), matching RW2 null space
pub fn sum_to_zero_constraint(n: usize, k: usize) -> Result<ConstraintSpec, String> {
    if n == 0 {
        return Err("sum_to_zero_constraint: n must be > 0".into());
    }
    if k == 0 {
        return Err("sum_to_zero_constraint: k must be > 0".into());
    }
    if k > 2 {
        return Err("sum_to_zero_constraint: only k=1 or k=2 supported".into());
    }
    let mut a = vec![0.0; k * n];
    let inv_sqrt_n = 1.0 / (n as f64).sqrt();
    for c in 0..n {
        a[c] = inv_sqrt_n;
    }
    if k == 2 {
        let mean = (n - 1) as f64 / 2.0;
        let mut ss = 0.0;
        for c in 0..n {
            let v = c as f64 - mean;
            a[n + c] = v;
            ss += v * v;
        }
        let scale = ss.sqrt();
        if scale < 1e-14 {
            return Err("sum_to_zero_constraint: degenerate linear row".into());
        }
        for c in 0..n {
            a[n + c] /= scale;
        }
    }
    Ok(ConstraintSpec {
        n,
        k,
        a,
        e: vec![0.0; k],
        method: ConstraintMethod::default(),
    })
}

/// Orthonormal basis of the seasonal null space for period `season`.
///
/// The seasonal model penalizes every window sum `x_i + … + x_{i+s-1}`, so its
/// null space is the `s-1` dimensional space of period-`s` patterns summing to
/// zero over one period — not the 1-dimensional constant space. Constraining
/// only the constant leaves `Q` singular.
pub fn seasonal_constraint(n: usize, season: usize) -> Result<ConstraintSpec, String> {
    if n == 0 {
        return Err("seasonal_constraint: n must be > 0".into());
    }
    if season < 2 {
        return Err("seasonal_constraint: season must be >= 2".into());
    }
    if season > n {
        return Err(format!(
            "seasonal_constraint: season {season} exceeds field length {n}"
        ));
    }
    let k = season - 1;
    // Raw contrast rows: +1 on phase 0, -1 on phase j, repeated over the field.
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(k);
    for j in 1..season {
        let mut row = vec![0.0; n];
        for (i, slot) in row.iter_mut().enumerate() {
            let phase = i % season;
            if phase == 0 {
                *slot = 1.0;
            } else if phase == j {
                *slot = -1.0;
            }
        }
        rows.push(row);
    }

    // Gram-Schmidt: the raw contrasts are independent but not orthogonal.
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(k);
    for mut row in rows {
        for b in &basis {
            let dot: f64 = row.iter().zip(b).map(|(x, y)| x * y).sum();
            for (x, y) in row.iter_mut().zip(b) {
                *x -= dot * y;
            }
        }
        let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm < 1e-10 {
            return Err(format!(
                "seasonal_constraint: degenerate contrast for season {season}, n {n}"
            ));
        }
        for x in row.iter_mut() {
            *x /= norm;
        }
        basis.push(row);
    }

    let mut a = Vec::with_capacity(k * n);
    for row in &basis {
        a.extend_from_slice(row);
    }
    Ok(ConstraintSpec {
        n,
        k,
        a,
        e: vec![0.0; k],
        method: ConstraintMethod::default(),
    })
}

/// Plane constraints for a regular lattice stored column-major (`node = j * nrow + i`).
///
/// Non-cyclic intrinsic RW2D has a 3D null space (constant + row linear + column linear).
/// Cyclic RW2D uses only the constant (`sum_to_zero_constraint(n, 1)`).
pub fn plane_constraint_2d(nrow: usize, ncol: usize) -> Result<ConstraintSpec, String> {
    if nrow == 0 || ncol == 0 {
        return Err("plane_constraint_2d: nrow and ncol must be > 0".into());
    }
    let n = nrow * ncol;
    let k = 3usize;
    let mut a = vec![0.0; k * n];
    let inv_sqrt_n = 1.0 / (n as f64).sqrt();
    let mean_i = (nrow - 1) as f64 / 2.0;
    let mean_j = (ncol - 1) as f64 / 2.0;
    let mut ss_i = 0.0;
    let mut ss_j = 0.0;
    for j in 0..ncol {
        for i in 0..nrow {
            let node = j * nrow + i;
            a[node] = inv_sqrt_n;
            let vi = i as f64 - mean_i;
            let vj = j as f64 - mean_j;
            a[n + node] = vi;
            a[2 * n + node] = vj;
            ss_i += vi * vi;
            ss_j += vj * vj;
        }
    }
    let scale_i = ss_i.sqrt();
    let scale_j = ss_j.sqrt();
    if scale_i < 1e-14 || scale_j < 1e-14 {
        return Err("plane_constraint_2d: degenerate row/col trend".into());
    }
    for node in 0..n {
        a[n + node] /= scale_i;
        a[2 * n + node] /= scale_j;
    }
    Ok(ConstraintSpec {
        n,
        k,
        a,
        e: vec![0.0; k],
        method: ConstraintMethod::default(),
    })
}

/// Default precision weight for “hard” extraconstr (R-INLA-like).
pub const HARD_CONSTRAINT_KAPPA: f64 = 3.059_023_205_018_258e6; // exp(15)

/// Return `Q + κ Aᵀ A` as CSC (makes intrinsic Q SPD for LDLT).
///
/// Assembled from triplets without densifying `Q`. For full sum-to-zero
/// constraints `AᵀA` is dense, so the result may still be dense — but sparse
/// `Q` is never copied into an `n²` buffer first.
pub fn augment_precision_csc(
    q: &CscMatrix,
    constr: &ConstraintSpec,
    kappa: f64,
) -> Result<CscMatrix, String> {
    constr.validate()?;
    if q.rows() != constr.n || q.cols() != constr.n {
        return Err("augment_precision_csc: Q dimension mismatch".into());
    }
    if !kappa.is_finite() || kappa <= 0.0 {
        return Err("augment_precision_csc: kappa must be finite and > 0".into());
    }
    let n = constr.n;
    let k = constr.k;
    // Upper bound: Q nnz + full AᵀA (k may make AᵀA dense).
    let nnz_hint = q.nnz() + n * n;
    let mut tri = TriMatI::<f64, usize>::with_capacity((n, n), nnz_hint);
    for (col, colvec) in q.outer_iterator().enumerate() {
        for (row, &val) in colvec.iter() {
            if val != 0.0 {
                tri.add_triplet(row, col, val);
            }
        }
    }
    // κ (Aᵀ A)_{ij} = κ Σ_r A_{ri} A_{rj}
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for r in 0..k {
                s += constr.a[r * n + i] * constr.a[r * n + j];
            }
            let v = kappa * s;
            if v != 0.0 {
                tri.add_triplet(i, j, v);
            }
        }
    }
    Ok(tri.to_csc())
}

/// Project `x` onto `{ z : A z = e }` via `x <- x - Aᵀ (A Aᵀ)^{-1} (A x - e)`.
pub fn project_constraints(x: &mut [f64], constr: &ConstraintSpec) -> Result<(), String> {
    constr.validate()?;
    if x.len() != constr.n {
        return Err("project_constraints: length mismatch".into());
    }
    let n = constr.n;
    let k = constr.k;
    // r = A x - e
    let mut r = vec![0.0; k];
    for row in 0..k {
        let mut s = 0.0;
        for c in 0..n {
            s += constr.a[row * n + c] * x[c];
        }
        r[row] = s - constr.e[row];
    }
    // G = A Aᵀ (k×k)
    let mut g = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            let mut s = 0.0;
            for c in 0..n {
                s += constr.a[i * n + c] * constr.a[j * n + c];
            }
            g[i * k + j] = s;
        }
    }
    // Solve G λ = r (dense Gaussian elimination for tiny k)
    let lambda = solve_dense_symmetric(&g, k, &r)?;
    for c in 0..n {
        let mut corr = 0.0;
        for row in 0..k {
            corr += constr.a[row * n + c] * lambda[row];
        }
        x[c] -= corr;
    }
    Ok(())
}

fn solve_dense_symmetric(a: &[f64], k: usize, b: &[f64]) -> Result<Vec<f64>, String> {
    let mut m = a.to_vec();
    let mut x = b.to_vec();
    for i in 0..k {
        let mut piv = i;
        for r in (i + 1)..k {
            if m[r * k + i].abs() > m[piv * k + i].abs() {
                piv = r;
            }
        }
        if m[piv * k + i].abs() < 1e-14 {
            return Err("project_constraints: singular A Aᵀ".into());
        }
        if piv != i {
            for c in 0..k {
                m.swap(i * k + c, piv * k + c);
            }
            x.swap(i, piv);
        }
        let diag = m[i * k + i];
        for c in i..k {
            m[i * k + c] /= diag;
        }
        x[i] /= diag;
        for r in 0..k {
            if r == i {
                continue;
            }
            let f = m[r * k + i];
            for c in i..k {
                m[r * k + c] -= f * m[i * k + c];
            }
            x[r] -= f * x[i];
        }
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::design::identity_csc;

    #[test]
    fn sum_to_zero_projects() {
        let c = sum_to_zero_constraint(5, 1).unwrap();
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        project_constraints(&mut x, &c).unwrap();
        let s: f64 = x.iter().sum();
        assert!(s.abs() < 1e-10, "sum={s}");
    }

    #[test]
    fn rw2_constraints_kill_constant_and_linear() {
        let c = sum_to_zero_constraint(6, 2).unwrap();
        let mut x: Vec<f64> = (0..6).map(|i| 2.0 + 0.5 * i as f64).collect();
        project_constraints(&mut x, &c).unwrap();
        let s: f64 = x.iter().sum();
        let mean = 2.5;
        let mut lin = 0.0;
        for (i, &v) in x.iter().enumerate() {
            lin += (i as f64 - mean) * v;
        }
        assert!(s.abs() < 1e-9, "sum={s}");
        assert!(lin.abs() < 1e-9, "lin={lin}");
    }

    #[test]
    fn augment_makes_identity_stiffer() {
        let q = identity_csc(4, 1.0).unwrap();
        let c = sum_to_zero_constraint(4, 1).unwrap();
        let qa = augment_precision_csc(&q, &c, HARD_CONSTRAINT_KAPPA).unwrap();
        let d = crate::ldlt::csc_to_dense(&qa).unwrap();
        // Diagonal should grow by κ/n
        let expect = 1.0 + HARD_CONSTRAINT_KAPPA / 4.0;
        assert!((d[0] - expect).abs() < 1e-6);
    }

    #[test]
    fn plane_2d_kills_const_row_col_trends() {
        let nrow = 4usize;
        let ncol = 5usize;
        let c = plane_constraint_2d(nrow, ncol).unwrap();
        assert_eq!(c.k, 3);
        let mut x = vec![0.0; nrow * ncol];
        for j in 0..ncol {
            for i in 0..nrow {
                x[j * nrow + i] = 1.0 + 0.3 * i as f64 + 0.7 * j as f64;
            }
        }
        project_constraints(&mut x, &c).unwrap();
        let s: f64 = x.iter().sum();
        let mean_i = (nrow - 1) as f64 / 2.0;
        let mean_j = (ncol - 1) as f64 / 2.0;
        let mut row_mom = 0.0;
        let mut col_mom = 0.0;
        for j in 0..ncol {
            for i in 0..nrow {
                let v = x[j * nrow + i];
                row_mom += (i as f64 - mean_i) * v;
                col_mom += (j as f64 - mean_j) * v;
            }
        }
        assert!(s.abs() < 1e-9, "sum={s}");
        assert!(row_mom.abs() < 1e-9, "row_mom={row_mom}");
        assert!(col_mom.abs() < 1e-9, "col_mom={col_mom}");
    }
}
