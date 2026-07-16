use crate::sparse::CscMatrix;
use sprs::TriMatI;

pub fn rw2d_precision_csc(
    nrow: usize,
    ncol: usize,
    tau: f64,
    cyclic: bool,
    bvalue_zero: bool,
) -> Result<CscMatrix, String> {
    if nrow < 3 || ncol < 3 {
        return Err("rw2d requires nrow >= 3 and ncol >= 3".to_string());
    }
    if tau <= 0.0 || !tau.is_finite() {
        return Err("rw2d tau must be finite and > 0".to_string());
    }

    let n = nrow * ncol;
    let mut tri = TriMatI::<f64, usize>::with_capacity((n, n), n * 13); // maximum elements per node is 13 (stencil size)

    let map = [0, 1, 2, 1, 0];

    for i in 0..nrow {
        for j in 0..ncol {
            let node = j * nrow + i;

            // Stencil is at most distance 2
            for di in -2..=2 {
                for dj in -2..=2 {
                    let ii_signed = i as isize + di;
                    let jj_signed = j as isize + dj;

                    let (ii, jj) = if cyclic {
                        (
                            ii_signed.rem_euclid(nrow as isize) as usize,
                            jj_signed.rem_euclid(ncol as isize) as usize,
                        )
                    } else {
                        if ii_signed < 0
                            || ii_signed >= nrow as isize
                            || jj_signed < 0
                            || jj_signed >= ncol as isize
                        {
                            continue;
                        }
                        (ii_signed as usize, jj_signed as usize)
                    };

                    let nnode = jj * nrow + ii;

                    let mut val = 0.0;

                    if cyclic {
                        let mut dx = (i as isize - ii as isize).abs();
                        let mut dy = (j as isize - jj as isize).abs();
                        dx = std::cmp::min(dx, nrow as isize - dx);
                        dy = std::cmp::min(dy, ncol as isize - dy);

                        match dx {
                            0 => match dy {
                                0 => val = 20.0,
                                1 => val = -8.0,
                                2 => val = 1.0,
                                _ => {}
                            },
                            1 => match dy {
                                0 => val = -8.0,
                                1 => val = 2.0,
                                _ => {}
                            },
                            2 => match dy {
                                0 => val = 1.0,
                                _ => {}
                            },
                            _ => {}
                        }
                    } else if bvalue_zero {
                        let dx = (i as isize - ii as isize).abs();
                        let dy = (j as isize - jj as isize).abs();

                        match dx {
                            0 => match dy {
                                0 => val = 20.0,
                                1 => val = -8.0,
                                2 => val = 1.0,
                                _ => {}
                            },
                            1 => match dy {
                                0 => val = -8.0,
                                1 => val = 2.0,
                                _ => {}
                            },
                            2 => match dy {
                                0 => val = 1.0,
                                _ => {}
                            },
                            _ => {}
                        }
                    } else {
                        let dx = (i as isize - ii as isize).abs();
                        let dy = (j as isize - jj as isize).abs();

                        if std::cmp::max(dx, dy) > 2 || (std::cmp::max(dx, dy) == 2 && std::cmp::min(dx, dy) >= 1) {
                            continue;
                        }

                        if std::cmp::max(dx, dy) == 2 {
                            val = 1.0;
                        } else if dx == 1 && dy == 1 {
                            val = 2.0;
                        } else if (i > 1 && i < nrow - 2 && j > 1 && j < ncol - 2)
                            || (ii > 1 && ii < nrow - 2 && jj > 1 && jj < ncol - 2)
                        {
                            if dx == 0 {
                                val = if dy == 0 { 20.0 } else { -8.0 };
                            } else {
                                val = -8.0;
                            }
                        } else if (i == ii) && (j == jj) {
                            let itmp = if i > 1 { map[std::cmp::max(2, i as isize + 5 - nrow as isize) as usize] } else { i };
                            let jtmp = if j > 1 { map[std::cmp::max(2, j as isize + 5 - ncol as isize) as usize] } else { j };

                            let iref = std::cmp::max(itmp, jtmp);
                            let jref = std::cmp::min(itmp, jtmp);

                            if iref == 1 {
                                val = if jref == 1 { 18.0 } else { 10.0 };
                            } else if iref == 2 {
                                val = if jref == 1 { 19.0 } else { 11.0 };
                            } else {
                                val = 4.0;
                            }
                        } else {
                            let imax = std::cmp::max(i, ii);
                            let imin = std::cmp::min(i, ii);
                            let jmax = std::cmp::max(j, jj);
                            let jmin = std::cmp::min(j, jj);

                            if (imin == 0 || imax == nrow - 1) && (jmin == 0 || jmax == ncol - 1) {
                                val = -4.0;
                            } else if imin == 0 || imax == nrow - 1 || jmin == 0 || jmax == ncol - 1 {
                                val = -6.0;
                            } else {
                                val = -8.0;
                            }
                        }
                    }

                    if val != 0.0 {
                        tri.add_triplet(node, nnode, val * tau);
                    }
                }
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
    fn test_rw2d_cyclic() {
        let q = rw2d_precision_csc(5, 5, 2.0, true, false).unwrap();
        assert_eq!(q.rows(), 25);
        assert_eq!(q.cols(), 25);
        let d = dense_from_csc(&q);

        // Center element: 20 * tau = 40.0
        approx(get(&d, 25, 0, 0), 40.0, 1e-12);
        // Off-1 adjacent: -8 * tau = -16.0
        approx(get(&d, 25, 0, 1), -16.0, 1e-12);
        // Diagonal adjacent: 2 * tau = 4.0
        approx(get(&d, 25, 0, 6), 4.0, 1e-12);
    }
}
