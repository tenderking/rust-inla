use inla_fmesher::FemBlocks;
use inla_math::{CscMatrix, sparse_from_triplets};
use sprs::TriMatI;

pub fn spde_precision_csc(
    fem: &FemBlocks,
    kappa: f64,
    tau: f64,
) -> Result<CscMatrix, String> {
    let n = fem.c0.rows;
    if n == 0 {
        return Err("SPDE requires non-empty FEM blocks".to_string());
    }
    if kappa <= 0.0 || !kappa.is_finite() {
        return Err("SPDE kappa must be finite and > 0".to_string());
    }
    if tau <= 0.0 || !tau.is_finite() {
        return Err("SPDE tau must be finite and > 0".to_string());
    }

    let mut c0_diag = vec![0.0; n];
    for &(r, c, v) in &fem.c0.entries {
        if r == c {
            c0_diag[r] += v;
        }
    }

    let mut c0_inv_tri = TriMatI::<f64, usize>::with_capacity((n, n), n);
    let mut k4_c0_tri = TriMatI::<f64, usize>::with_capacity((n, n), n);
    
    let k2 = kappa * kappa;
    let k4 = k2 * k2;

    for i in 0..n {
        let v = c0_diag[i];
        if v.abs() < 1e-15 {
            return Err("Zero entry in mass matrix c0 diagonal".to_string());
        }
        c0_inv_tri.add_triplet(i, i, 1.0 / v);
        k4_c0_tri.add_triplet(i, i, k4 * v);
    }

    let c0_inv: CscMatrix = c0_inv_tri.to_csc();
    let k4_c0: CscMatrix = k4_c0_tri.to_csc();

    let g1 = sparse_from_triplets(fem.g1.rows, fem.g1.cols, &fem.g1.entries);

    let mut k2_2_g1_tri = TriMatI::<f64, usize>::with_capacity((n, n), g1.nnz());
    for (val, (r, c)) in g1.iter() {
        k2_2_g1_tri.add_triplet(r, c, 2.0 * k2 * *val);
    }
    let k2_2_g1: CscMatrix = k2_2_g1_tri.to_csc();

    let g1_c0_inv = &g1 * &c0_inv;
    let g1_c0_inv_g1 = &g1_c0_inv * &g1;

    let mut q_unscaled_tri = TriMatI::<f64, usize>::with_capacity((n, n), k4_c0.nnz() + k2_2_g1.nnz() + g1_c0_inv_g1.nnz());
    
    for (val, (r, c)) in k4_c0.iter() {
        q_unscaled_tri.add_triplet(r, c, *val);
    }
    for (val, (r, c)) in k2_2_g1.iter() {
        q_unscaled_tri.add_triplet(r, c, *val);
    }
    for (val, (r, c)) in g1_c0_inv_g1.iter() {
        q_unscaled_tri.add_triplet(r, c, *val);
    }

    let q_unscaled: CscMatrix = q_unscaled_tri.to_csc();

    let tau2 = tau * tau;
    let mut q_scaled_tri = TriMatI::<f64, usize>::with_capacity((n, n), q_unscaled.nnz());
    for (val, (r, c)) in q_unscaled.iter() {
        q_scaled_tri.add_triplet(r, c, *val * tau2);
    }

    Ok(q_scaled_tri.to_csc())
}

#[cfg(test)]
mod tests {
    use super::*;
    use inla_fmesher::{Vertex2, Triangle, build_mesh2d};

    #[test]
    fn test_spde_precision() {
        // Small 2D square mesh with 2 triangles
        // 0 ---- 1
        // |   /  |
        // |  /   |
        // 2 ---- 3
        let vertices = vec![
            Vertex2 { x: 0.0, y: 1.0 },
            Vertex2 { x: 1.0, y: 1.0 },
            Vertex2 { x: 0.0, y: 0.0 },
            Vertex2 { x: 1.0, y: 0.0 },
        ];
        let triangles = vec![
            Triangle([0, 2, 1]),
            Triangle([1, 2, 3]),
        ];
        let mesh = build_mesh2d(vertices, triangles).unwrap();
        let fem = mesh.assemble_fem_blocks();

        let q = spde_precision_csc(&fem, 1.0, 2.0).unwrap();
        assert_eq!(q.rows(), 4);
        assert_eq!(q.cols(), 4);
        assert!(q.nnz() > 0);

        // Check diagonal elements are positive
        for i in 0..4 {
            let diag_val = *q.get(i, i).unwrap_or(&0.0);
            assert!(diag_val > 0.0, "diagonal value at {} is {}", i, diag_val);
        }
    }
}
