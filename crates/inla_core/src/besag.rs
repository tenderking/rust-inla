use crate::sparse::CscMatrix;
use sprs::TriMatI;
use std::fs;
use std::path::Path;

pub fn read_graph_file<P: AsRef<Path>>(path: P) -> Result<Vec<Vec<usize>>, String> {
    let content = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut tokens = content.split_whitespace();
    let n_str = tokens.next().ok_or_else(|| "Graph file is empty".to_string())?;
    let n: usize = n_str.parse().map_err(|e| format!("Invalid node count: {e}"))?;

    let mut raw_nodes = vec![Vec::new(); n + 1];
    let mut min_node = usize::MAX;
    let mut max_node = 0;

    for _ in 0..n {
        let tnode_str = tokens.next().ok_or_else(|| "Expected node index".to_string())?;
        let tnode: usize = tnode_str.parse().map_err(|e| format!("Invalid node index: {e}"))?;
        
        let degree_str = tokens.next().ok_or_else(|| "Expected degree".to_string())?;
        let degree: usize = degree_str.parse().map_err(|e| format!("Invalid degree: {e}"))?;

        if tnode > n {
            return Err(format!("Node index {} exceeds n={}", tnode, n));
        }
        min_node = std::cmp::min(min_node, tnode);
        max_node = std::cmp::max(max_node, tnode);

        let mut nbs = Vec::with_capacity(degree);
        for _ in 0..degree {
            let nb_str = tokens.next().ok_or_else(|| "Expected neighbor index".to_string())?;
            let nb: usize = nb_str.parse().map_err(|e| format!("Invalid neighbor index: {e}"))?;
            nbs.push(nb);
            min_node = std::cmp::min(min_node, nb);
            max_node = std::cmp::max(max_node, nb);
        }
        raw_nodes[tnode] = nbs;
    }

    let mut adj = vec![Vec::new(); n];
    if min_node == 1 && max_node == n {
        for i in 1..=n {
            let mut shifted_nbs = raw_nodes[i].clone();
            for val in &mut shifted_nbs {
                *val -= 1;
            }
            adj[i - 1] = shifted_nbs;
        }
    } else if min_node == 0 && max_node == n - 1 {
        for i in 0..n {
            adj[i] = raw_nodes[i].clone();
        }
    } else {
        return Err(format!("Graph indexing is inconsistent: min={}, max={}, n={}", min_node, max_node, n));
    }

    Ok(adj)
}

pub fn besag_precision_csc(adj: &[Vec<usize>], tau: f64) -> Result<CscMatrix, String> {
    let n = adj.len();
    if n == 0 {
        return Err("Besag requires at least 1 node".to_string());
    }
    if tau <= 0.0 || !tau.is_finite() {
        return Err("Besag tau must be finite and > 0".to_string());
    }

    let mut tri = TriMatI::<f64, usize>::with_capacity((n, n), n * 4);
    for i in 0..n {
        let degree = adj[i].len();
        tri.add_triplet(i, i, (degree as f64) * tau);
        for &j in &adj[i] {
            if j >= n {
                return Err(format!("Neighbor index {} exceeds node count {}", j, n));
            }
            tri.add_triplet(i, j, -tau);
        }
    }
    Ok(tri.to_csc())
}

pub fn bym_precision_csc(
    adj: &[Vec<usize>],
    tau_spatial: f64,
    tau_iid: f64,
) -> Result<CscMatrix, String> {
    let n = adj.len();
    if n == 0 {
        return Err("BYM requires at least 1 node".to_string());
    }
    if tau_spatial <= 0.0 || !tau_spatial.is_finite() {
        return Err("BYM tau_spatial must be finite and > 0".to_string());
    }
    if tau_iid <= 0.0 || !tau_iid.is_finite() {
        return Err("BYM tau_iid must be finite and > 0".to_string());
    }

    let mut tri = TriMatI::<f64, usize>::with_capacity((2 * n, 2 * n), n * 5);
    for i in 0..n {
        let degree = adj[i].len();
        tri.add_triplet(i, i, (degree as f64) * tau_spatial);
        for &j in &adj[i] {
            if j >= n {
                return Err(format!("Neighbor index {} exceeds node count {}", j, n));
            }
            tri.add_triplet(i, j, -tau_spatial);
        }
    }
    for i in 0..n {
        tri.add_triplet(n + i, n + i, tau_iid);
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
    fn test_besag_and_bym() {
        // Small cycle graph of size 4: 0-1, 1-2, 2-3, 3-0
        let adj = vec![
            vec![1, 3],
            vec![0, 2],
            vec![1, 3],
            vec![2, 0],
        ];

        let q_besag = besag_precision_csc(&adj, 1.5).unwrap();
        assert_eq!(q_besag.rows(), 4);
        assert_eq!(q_besag.cols(), 4);
        let d_besag = dense_from_csc(&q_besag);

        approx(get(&d_besag, 4, 0, 0), 3.0, 1e-12); // degree 2 * 1.5 = 3.0
        approx(get(&d_besag, 4, 0, 1), -1.5, 1e-12);
        approx(get(&d_besag, 4, 0, 2), 0.0, 1e-12);

        let q_bym = bym_precision_csc(&adj, 1.5, 2.5).unwrap();
        assert_eq!(q_bym.rows(), 8);
        assert_eq!(q_bym.cols(), 8);
        let d_bym = dense_from_csc(&q_bym);

        approx(get(&d_bym, 8, 0, 0), 3.0, 1e-12);
        approx(get(&d_bym, 8, 4, 4), 2.5, 1e-12); // IID diagonal
        approx(get(&d_bym, 8, 0, 4), 0.0, 1e-12); // Block isolation
    }
}
