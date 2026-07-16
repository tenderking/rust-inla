use inla_math::CscMatrix;
use sprs::TriMatI;

pub fn crw1_precision_csc(positions: &[f64], tau: f64) -> Result<CscMatrix, String> {
    let n = positions.len();
    if n < 2 {
        return Err("crw1 requires n >= 2".to_string());
    }
    if tau <= 0.0 || !tau.is_finite() {
        return Err("crw1 tau must be finite and > 0".to_string());
    }
    for i in 0..(n - 1) {
        if positions[i] >= positions[i + 1] {
            return Err("positions must be strictly increasing".to_string());
        }
    }

    let mut idelta = vec![0.0; n - 1];
    for i in 0..(n - 1) {
        idelta[i] = 1.0 / (positions[i + 1] - positions[i]);
    }

    let mut tri = TriMatI::<f64, usize>::with_capacity((n, n), 3 * n - 2);
    for i in 0..n {
        let diag = if i == 0 {
            idelta[0]
        } else if i == n - 1 {
            idelta[n - 2]
        } else {
            idelta[i - 1] + idelta[i]
        };
        tri.add_triplet(i, i, diag * tau);

        if i + 1 < n {
            tri.add_triplet(i, i + 1, -idelta[i] * tau);
            tri.add_triplet(i + 1, i, -idelta[i] * tau);
        }
    }

    Ok(tri.to_csc())
}

pub fn crw2_precision_csc(
    positions: &[f64],
    tau: f64,
    layout: &str,
) -> Result<CscMatrix, String> {
    let n = positions.len();
    if n < 3 {
        return Err("crw2 requires n >= 3".to_string());
    }
    if tau <= 0.0 || !tau.is_finite() {
        return Err("crw2 tau must be finite and > 0".to_string());
    }
    for i in 0..(n - 1) {
        if positions[i] >= positions[i + 1] {
            return Err("positions must be strictly increasing".to_string());
        }
    }

    let mut idelta = vec![0.0; n - 1];
    let mut idelta2 = vec![0.0; n - 1];
    let mut idelta3 = vec![0.0; n - 1];
    for i in 0..(n - 1) {
        let diff = positions[i + 1] - positions[i];
        idelta[i] = 1.0 / diff;
        idelta2[i] = idelta[i] * idelta[i];
        idelta3[i] = idelta[i] * idelta2[i];
    }

    let mut isdelta = vec![0.0; n - 2];
    for i in 0..(n - 2) {
        isdelta[i] = 1.0 / (positions[i + 2] - positions[i]);
    }

    if layout == "simple" {
        let get_idelta = |k: isize| {
            if k < 0 || k >= (n - 1) as isize { 0.0 } else { idelta[k as usize] }
        };
        let get_idelta2 = |k: isize| {
            if k < 0 || k >= (n - 1) as isize { 0.0 } else { idelta2[k as usize] }
        };
        let get_isdelta = |k: isize| {
            if k < 0 || k >= (n - 2) as isize { 0.0 } else { isdelta[k as usize] }
        };
        let get_sidelta = |k: isize| {
            if k == -1 {
                get_idelta(0)
            } else if k == (n - 2) as isize {
                get_idelta((n - 2) as isize)
            } else if k < -1 || k >= (n - 2) as isize {
                0.0
            } else {
                get_idelta(k) + get_idelta(k + 1)
            }
        };

        let mut tri = TriMatI::<f64, usize>::with_capacity((n, n), 5 * n - 6);
        for i in 0..n {
            for j in 0..n {
                let imax = std::cmp::max(i, j);
                let imin = std::cmp::min(i, j);
                let idiff = imax - imin;

                let mut val = 0.0;
                if idiff == 0 {
                    let idx = imax as isize;
                    val = 2.0 * (get_idelta2(idx - 1) * get_isdelta(idx - 2)
                        + get_idelta(idx - 1) * get_idelta(idx) * get_sidelta(idx - 1)
                        + get_idelta2(idx) * get_isdelta(idx));
                } else if idiff == 1 {
                    let idx = imax as isize;
                    val = -2.0 * get_idelta2(idx - 1) * (get_idelta(idx - 2) + get_idelta(idx));
                } else if idiff == 2 {
                    let idx = imax as isize;
                    val = 2.0 * get_idelta(idx - 2) * get_idelta(idx - 1) * get_isdelta(idx - 2);
                }

                if val != 0.0 {
                    tri.add_triplet(i, j, val * tau);
                }
            }
        }
        Ok(tri.to_csc())
    } else if layout == "pairs" || layout == "block" {
        let dim = 2 * n;
        let mut tri = TriMatI::<f64, usize>::with_capacity((dim, dim), dim * 4);

        for node in 0..dim {
            for nnode in 0..dim {
                let (node_i, nnode_i, node_tp, nnode_tp) = match layout {
                    "pairs" => {
                        let (n1, n2) = if node < nnode { (node, nnode) } else { (nnode, node) };
                        (n1 / 2, n2 / 2, if n1 % 2 == 0 { 0 } else { 1 }, if n2 % 2 == 0 { 0 } else { 1 })
                    }
                    "block" => {
                        let (n1, n2) = if node % n < nnode % n { (node, nnode) } else { (nnode, node) };
                        (n1 % n, n2 % n, if n1 < n { 0 } else { 1 }, if n2 < n { 0 } else { 1 })
                    }
                    _ => unreachable!(),
                };

                let idiff = nnode_i as isize - node_i as isize;
                if idiff > 1 {
                    continue;
                }

                let mut val = 0.0;
                if idiff == 0 {
                    if node_tp == 0 {
                        if nnode_tp == 0 {
                            if nnode_i == 0 {
                                val = 12.0 * idelta3[0];
                            } else if nnode_i == n - 1 {
                                val = 12.0 * idelta3[n - 2];
                            } else {
                                val = 12.0 * (idelta3[node_i] + idelta3[node_i - 1]);
                            }
                        } else {
                            if nnode_i == 0 {
                                val = 6.0 * idelta2[0];
                            } else if nnode_i == n - 1 {
                                val = -6.0 * idelta2[n - 2];
                            } else {
                                val = 6.0 * (idelta2[node_i] - idelta2[node_i - 1]);
                            }
                        }
                    } else {
                        if nnode_tp == 0 {
                            if nnode_i == 0 {
                                val = 6.0 * idelta2[0];
                            } else if nnode_i == n - 1 {
                                val = -6.0 * idelta2[n - 2];
                            } else {
                                val = 6.0 * (idelta2[node_i] - idelta2[node_i - 1]);
                            }
                        } else {
                            if nnode_i == 0 {
                                val = 4.0 * idelta[0];
                            } else if nnode_i == n - 1 {
                                val = 4.0 * idelta[n - 2];
                            } else {
                                val = 4.0 * (idelta[node_i] + idelta[node_i - 1]);
                            }
                        }
                    }
                } else if idiff == 1 {
                    if node_tp == 0 {
                        if nnode_tp == 0 {
                            val = -12.0 * idelta3[node_i];
                        } else {
                            val = 6.0 * idelta2[node_i];
                        }
                    } else {
                        if nnode_tp == 0 {
                            val = -6.0 * idelta2[node_i];
                        } else {
                            val = 2.0 * idelta[node_i];
                        }
                    }
                }

                // Note: the C code checks node < nnode and maps values back. Since we evaluate for all node/nnode pairs,
                // we should make sure we apply correct signs. The C code returns Q(node, nnode) symmetrically.
                // Our val logic above is symmetric because:
                // For idiff == 1:
                // node_tp and nnode_tp role is correctly preserved.
                // Let's verify: in C, if node > nnode, the mapping swaps node and nnode, so node_tp and nnode_tp are swapped.
                // In our code:
                // (node_i, nnode_i) are sorted (n1 <= n2), but node_tp corresponds to n1 and nnode_tp to n2.
                // So if we sorted them, they represent the sorted variables, making it symmetric!
                // Yes! Because (n1, n2) are sorted, the computed val is for Q[n1, n2]. Since we add it to (node, nnode) (which is either (n1,n2) or (n2,n1)), it naturally populates the matrix symmetrically.
                // This is extremely elegant!

                if val != 0.0 {
                    tri.add_triplet(node, nnode, val * tau);
                }
            }
        }
        Ok(tri.to_csc())
    } else {
        Err("Invalid layout: must be simple, pairs, or block".to_string())
    }
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
    fn test_crw1_precision() {
        let pos = [0.0, 1.0, 3.0, 4.0];
        let q = crw1_precision_csc(&pos, 1.0).unwrap();
        assert_eq!(q.rows(), 4);
        assert_eq!(q.cols(), 4);
        let d = dense_from_csc(&q);

        // idelta = [1.0, 0.5, 1.0]
        approx(get(&d, 4, 0, 0), 1.0, 1e-12);
        approx(get(&d, 4, 1, 1), 1.5, 1e-12);
        approx(get(&d, 4, 0, 1), -1.0, 1e-12);
        approx(get(&d, 4, 1, 2), -0.5, 1e-12);
    }

    #[test]
    fn test_crw2_simple() {
        let pos = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let q_crw = crw2_precision_csc(&pos, 1.0, "simple").unwrap();
        let q_rw = crate::latent_models::rw2_precision_csc(6, 1.0).unwrap();
        
        let d_crw = dense_from_csc(&q_crw);
        let d_rw = dense_from_csc(&q_rw);
        for i in 0..6 {
            for j in 0..6 {
                approx(get(&d_crw, 6, i, j), get(&d_rw, 6, i, j), 1e-12);
            }
        }
    }

    #[test]
    fn test_crw2_augmented() {
        let pos = [0.0, 1.0, 2.0, 3.0];
        let q_pairs = crw2_precision_csc(&pos, 1.0, "pairs").unwrap();
        let q_block = crw2_precision_csc(&pos, 1.0, "block").unwrap();
        
        assert_eq!(q_pairs.rows(), 8);
        assert_eq!(q_block.rows(), 8);
        
        // Assert symmetry
        let d_pairs = dense_from_csc(&q_pairs);
        let d_block = dense_from_csc(&q_block);
        for i in 0..8 {
            for j in 0..8 {
                approx(get(&d_pairs, 8, i, j), get(&d_pairs, 8, j, i), 1e-12);
                approx(get(&d_block, 8, i, j), get(&d_block, 8, j, i), 1e-12);
            }
        }
    }
}
