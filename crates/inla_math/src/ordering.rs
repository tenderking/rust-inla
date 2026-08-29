//! Fill-reducing / band-reducing symmetric orderings for sparse LDLᵀ.
//!
//! Used only at factorize time so caller indexing (A, latent layout) is unchanged.

use crate::sparse::CscMatrix;

/// Lower-triangular envelope: `max(row − col)` over stored entries with `row ≥ col`.
pub fn lower_bandwidth(q: &CscMatrix) -> usize {
    let mut bw = 0usize;
    for (col, colvec) in q.outer_iterator().enumerate() {
        for (row, _) in colvec.iter() {
            if row >= col {
                bw = bw.max(row - col);
            }
        }
    }
    bw
}

/// Bandwidth after `new_index = fwd[old_index]`.
pub fn bandwidth_with_fwd(q: &CscMatrix, fwd: &[usize]) -> usize {
    let mut bw = 0usize;
    for (col, colvec) in q.outer_iterator().enumerate() {
        let jc = fwd[col];
        for (row, _) in colvec.iter() {
            bw = bw.max(fwd[row].abs_diff(jc));
        }
    }
    bw
}

/// Map mixture-major `c * n_time + t` → time-major `t * n_comp + c`.
///
/// FGN approx stores `(z, x_1, …, x_{k})` as `n_comp = k+1` blocks of length
/// `n_time`. Time-major numbering makes the Cholesky envelope `O(n_comp)`.
pub fn mixture_to_time_major_perm(n_time: usize, n_comp: usize) -> (Vec<usize>, Vec<usize>) {
    let n = n_time.saturating_mul(n_comp);
    let mut fwd = vec![0usize; n];
    let mut inv = vec![0usize; n];
    for c in 0..n_comp {
        for t in 0..n_time {
            let old = c * n_time + t;
            let new = t * n_comp + c;
            fwd[old] = new;
            inv[new] = old;
        }
    }
    (fwd, inv)
}

/// If `Q` looks like `n_comp` consecutive length-`n_time` blocks with long-range
/// couplings of distance `Θ(n_time)`, return the time-major permutation.
///
/// This is the FGN-approx (and similar Kronecker) case: we do **not** rewrite
/// the stored CSC, only the factorize permutation.
pub fn try_time_major_kronecker(q: &CscMatrix) -> Option<(Vec<usize>, Vec<usize>)> {
    let n = q.rows();
    if n != q.cols() || n < 16 {
        return None;
    }
    let bw = lower_bandwidth(q);
    if bw < n / 8 {
        return None;
    }
    let mut best: Option<(usize, Vec<usize>, Vec<usize>)> = None;
    for n_comp in 2..=8 {
        if !n.is_multiple_of(n_comp) {
            continue;
        }
        let n_time = n / n_comp;
        if n_time < 8 {
            continue;
        }
        let (fwd, inv) = mixture_to_time_major_perm(n_time, n_comp);
        let bw_new = bandwidth_with_fwd(q, &fwd);
        if bw_new < bw {
            match &best {
                Some((b, _, _)) if *b <= bw_new => {}
                _ => best = Some((bw_new, fwd, inv)),
            }
        }
    }
    let (bw_new, fwd, inv) = best?;
    if bw_new * 4 < bw || bw_new <= 4 * ((q.nnz() / n.max(1)).max(2)) {
        Some((fwd, inv))
    } else {
        None
    }
}

/// Reverse Cuthill–McKee (RCM) on the undirected sparsity graph.
pub fn reverse_cuthill_mckee(q: &CscMatrix) -> (Vec<usize>, Vec<usize>) {
    let n = q.rows();
    let mut adj = vec![Vec::new(); n];
    for (col, colvec) in q.outer_iterator().enumerate() {
        for (row, _) in colvec.iter() {
            if row != col {
                adj[col].push(row);
                adj[row].push(col);
            }
        }
    }
    for nbrs in &mut adj {
        nbrs.sort_unstable();
        nbrs.dedup();
    }
    let deg: Vec<usize> = adj.iter().map(|nbrs| nbrs.len()).collect();
    for nbrs in &mut adj {
        nbrs.sort_by_key(|&j| (deg[j], j));
    }

    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);
    loop {
        let start = (0..n).filter(|&i| !visited[i]).min_by_key(|&i| (deg[i], i));
        let Some(start) = start else {
            break;
        };
        let mut queue = std::collections::VecDeque::new();
        visited[start] = true;
        queue.push_back(start);
        while let Some(u) = queue.pop_front() {
            order.push(u);
            for &v in &adj[u] {
                if !visited[v] {
                    visited[v] = true;
                    queue.push_back(v);
                }
            }
        }
    }
    order.reverse();
    let mut fwd = vec![0usize; n];
    let mut inv = vec![0usize; n];
    for (new, &old) in order.iter().enumerate() {
        fwd[old] = new;
        inv[new] = old;
    }
    (fwd, inv)
}

pub enum CholeskyOrder {
    Amd,
    Custom { fwd: Vec<usize>, inv: Vec<usize> },
}

/// Prefer time-major (Kronecker / FGN approx), else RCM when it cuts the envelope,
/// else AMD (2-D graphs, already-banded AR/RW).
pub fn choose_symmetric_order(q: &CscMatrix) -> CholeskyOrder {
    let bw = lower_bandwidth(q);
    if let Some((fwd, inv)) = try_time_major_kronecker(q) {
        return CholeskyOrder::Custom { fwd, inv };
    }
    if bw > 32 {
        let (fwd, inv) = reverse_cuthill_mckee(q);
        let bw_new = bandwidth_with_fwd(q, &fwd);
        if bw_new.saturating_mul(2) < bw {
            return CholeskyOrder::Custom { fwd, inv };
        }
    }
    CholeskyOrder::Amd
}

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::TriMatI;

    fn mixture_major_path_cliques(n_time: usize, n_comp: usize) -> CscMatrix {
        let n = n_time * n_comp;
        let mut tri = TriMatI::<f64, usize>::new((n, n));
        for t in 0..n_time {
            for c in 0..n_comp {
                let i = c * n_time + t;
                tri.add_triplet(i, i, 4.0);
                for d in (c + 1)..n_comp {
                    let j = d * n_time + t;
                    tri.add_triplet(i, j, -0.3);
                    tri.add_triplet(j, i, -0.3);
                }
                if t + 1 < n_time && c > 0 {
                    let i2 = c * n_time + t + 1;
                    tri.add_triplet(i, i2, -1.0);
                    tri.add_triplet(i2, i, -1.0);
                }
            }
        }
        tri.to_csc()
    }

    #[test]
    fn time_major_cuts_fgn_like_envelope() {
        let n_time = 64;
        let n_comp = 5;
        let q = mixture_major_path_cliques(n_time, n_comp);
        let bw = lower_bandwidth(&q);
        assert!(bw >= n_time - 1, "mixture-major envelope {bw}");
        let (fwd, _) = try_time_major_kronecker(&q).expect("detect Kronecker layout");
        let bw_tm = bandwidth_with_fwd(&q, &fwd);
        assert!(
            bw_tm <= 2 * n_comp,
            "time-major envelope {bw_tm} should be O(n_comp)={n_comp}"
        );
    }
}
