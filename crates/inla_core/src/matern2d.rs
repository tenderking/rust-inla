use crate::sparse::CscMatrix;
use sprs::TriMatI;

pub fn matern2d_precision_csc(
    nrow: usize,
    ncol: usize,
    nu: usize,
    range: f64,
    prec: f64,
    cyclic: bool,
) -> Result<CscMatrix, String> {
    if nrow == 0 || ncol == 0 {
        return Err("matern2d requires nrow > 0 and ncol > 0".to_string());
    }
    if nu == 0 || nu > 3 {
        return Err("matern2d nu must be 1, 2, or 3".to_string());
    }
    if range <= 0.0 || !range.is_finite() {
        return Err("matern2d range must be finite and > 0".to_string());
    }
    if prec <= 0.0 || !prec.is_finite() {
        return Err("matern2d prec must be finite and > 0".to_string());
    }

    let n = nrow * ncol;
    let kappa = 2.0 * (2.0 * nu as f64).sqrt() / range;
    let a = 4.0 + kappa * kappa;
    let std_variance = kappa.powf(-2.0 * nu as f64) / (4.0 * std::f64::consts::PI * nu as f64);

    let mut tri = TriMatI::<f64, usize>::with_capacity((n, n), n * (2 * nu + 3) * (2 * nu + 3));

    let stencil_limit = (nu + 1) as isize;

    for irow in 0..nrow {
        for icol in 0..ncol {
            let node = icol * nrow + irow;

            for drow_offset in -stencil_limit..=stencil_limit {
                for dcol_offset in -stencil_limit..=stencil_limit {
                    let jrow_signed = irow as isize + drow_offset;
                    let jcol_signed = icol as isize + dcol_offset;

                    let (jrow, jcol) = if cyclic {
                        (
                            jrow_signed.rem_euclid(nrow as isize) as usize,
                            jcol_signed.rem_euclid(ncol as isize) as usize,
                        )
                    } else {
                        if jrow_signed < 0
                            || jrow_signed >= nrow as isize
                            || jcol_signed < 0
                            || jcol_signed >= ncol as isize
                        {
                            continue;
                        }
                        (jrow_signed as usize, jcol_signed as usize)
                    };

                    let nnode = jcol * nrow + jrow;

                    let mut drow = (irow as isize - jrow as isize).abs();
                    let mut dcol = (icol as isize - jcol as isize).abs();

                    if cyclic {
                        drow = std::cmp::min(drow, nrow as isize - drow);
                        dcol = std::cmp::min(dcol, ncol as isize - dcol);
                    }

                    let dmax = std::cmp::max(drow, dcol) as usize;
                    let dmin = std::cmp::min(drow, dcol) as usize;

                    let mut val = 0.0;
                    if node == nnode {
                        val = match nu {
                            1 => 4.0 + a * a,
                            2 => a * (a * a + 12.0),
                            3 => (a * a + 6.0).powi(2) + 12.0 * a * a,
                            _ => 0.0,
                        };
                    } else {
                        match nu {
                            1 => {
                                match dmin {
                                    0 => match dmax {
                                        1 => val = -2.0 * a,
                                        2 => val = 1.0,
                                        _ => {}
                                    },
                                    1 => {
                                        if dmax == 1 {
                                            val = 2.0;
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            2 => {
                                match dmin {
                                    0 => match dmax {
                                        1 => val = -3.0 * (a * a + 3.0),
                                        2 => val = 3.0 * a,
                                        3 => val = -1.0,
                                        _ => {}
                                    },
                                    1 => match dmax {
                                        1 => val = 6.0 * a,
                                        2 => val = -3.0,
                                        _ => {}
                                    },
                                    _ => {}
                                }
                            }
                            3 => {
                                match dmin {
                                    0 => match dmax {
                                        1 => val = -4.0 * a * (a * a + 9.0),
                                        2 => val = 2.0 * (3.0 * a * a + 8.0),
                                        3 => val = -4.0 * a,
                                        4 => val = 1.0,
                                        _ => {}
                                    },
                                    1 => match dmax {
                                        1 => val = 12.0 * (a * a + 2.0),
                                        2 => val = -12.0 * a,
                                        3 => val = 4.0,
                                        _ => {}
                                    },
                                    2 => {
                                        if dmax == 2 {
                                            val = 6.0;
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            _ => {}
                        }
                    }

                    if val != 0.0 {
                        tri.add_triplet(node, nnode, val * prec * std_variance);
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
    fn test_matern2d_nu1() {
        let q = matern2d_precision_csc(4, 4, 1, 2.0, 1.0, false).unwrap();
        assert_eq!(q.rows(), 16);
        assert_eq!(q.cols(), 16);
        let d = dense_from_csc(&q);

        // Verify diagonal elements
        let kappa = 2.0 * (2.0f64).sqrt() / 2.0; // sqrt(2)
        let a = 4.0 + kappa * kappa; // 6.0
        let std_var = (kappa * kappa).recip() / (4.0 * std::f64::consts::PI);
        let want_diag = (4.0 + a * a) * std_var; // 40 * std_var
        
        approx(get(&d, 16, 0, 0), want_diag, 1e-12);
        
        // Symmetry test
        for i in 0..16 {
            for j in 0..16 {
                approx(get(&d, 16, i, j), get(&d, 16, j, i), 1e-12);
            }
        }
    }
}
