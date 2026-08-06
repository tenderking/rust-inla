//! Generic Nelder–Mead and finite-difference Hessian for scalar objectives.

/// Minimize `f` with Nelder–Mead (downhill simplex).
pub fn nelder_mead(
    initial: &[f64],
    step_size: f64,
    max_iter: usize,
    tol: f64,
    f: &dyn Fn(&[f64]) -> Result<f64, String>,
) -> Result<Vec<f64>, String> {
    nelder_mead_cancellable(initial, step_size, max_iter, tol, f, None)
}

/// Minimize `f` with Nelder–Mead, supporting an optional `check_cancel` callback.
pub fn nelder_mead_cancellable(
    initial: &[f64],
    step_size: f64,
    max_iter: usize,
    tol: f64,
    f: &dyn Fn(&[f64]) -> Result<f64, String>,
    check_cancel: Option<&dyn Fn() -> Result<(), String>>,
) -> Result<Vec<f64>, String> {
    let m = initial.len();
    let mut vertices = vec![vec![0.0; m]; m + 1];
    let mut f_vals = vec![0.0; m + 1];

    if let Some(cancel) = check_cancel {
        cancel()?;
    }

    vertices[0] = initial.to_vec();
    f_vals[0] = f(&vertices[0])?;

    for i in 1..=m {
        if let Some(cancel) = check_cancel {
            cancel()?;
        }
        let mut v = initial.to_vec();
        v[i - 1] += step_size;
        vertices[i] = v;
        f_vals[i] = f(&vertices[i])?;
    }

    for _iter in 0..max_iter {
        if let Some(cancel) = check_cancel {
            cancel()?;
        }

        let mut idxs: Vec<usize> = (0..=m).collect();
        idxs.sort_by(|&a, &b| f_vals[a].partial_cmp(&f_vals[b]).unwrap_or(std::cmp::Ordering::Equal));

        let mut sorted_vertices = vec![vec![0.0; m]; m + 1];
        let mut sorted_f = vec![0.0; m + 1];
        for i in 0..=m {
            sorted_vertices[i] = vertices[idxs[i]].clone();
            sorted_f[i] = f_vals[idxs[i]];
        }
        vertices = sorted_vertices;
        f_vals = sorted_f;

        let mean_f: f64 = f_vals.iter().sum::<f64>() / (m + 1) as f64;
        let variance_f: f64 =
            f_vals.iter().map(|&v| (v - mean_f).powi(2)).sum::<f64>() / (m + 1) as f64;
        if variance_f.sqrt() < tol {
            return Ok(vertices[0].clone());
        }

        let mut centroid = vec![0.0; m];
        for i in 0..m {
            for j in 0..m {
                centroid[j] += vertices[i][j];
            }
        }
        for j in 0..m {
            centroid[j] /= m as f64;
        }

        let mut reflected = vec![0.0; m];
        for j in 0..m {
            reflected[j] = centroid[j] + 1.0 * (centroid[j] - vertices[m][j]);
        }
        let f_r = match f(&reflected) {
            Ok(v) => v,
            Err(e) => {
                if let Some(cancel) = check_cancel {
                    cancel()?;
                }
                return Err(e);
            }
        };

        if f_vals[0] <= f_r && f_r < f_vals[m - 1] {
            vertices[m] = reflected;
            f_vals[m] = f_r;
            continue;
        }

        if f_r < f_vals[0] {
            let mut expanded = vec![0.0; m];
            for j in 0..m {
                expanded[j] = centroid[j] + 2.0 * (reflected[j] - centroid[j]);
            }
            let f_e = match f(&expanded) {
                Ok(v) => v,
                Err(e) => {
                    if let Some(cancel) = check_cancel {
                        cancel()?;
                    }
                    return Err(e);
                }
            };
            if f_e < f_r {
                vertices[m] = expanded;
                f_vals[m] = f_e;
            } else {
                vertices[m] = reflected;
                f_vals[m] = f_r;
            }
            continue;
        }

        let mut contract = false;
        if f_r < f_vals[m] {
            let mut contracted = vec![0.0; m];
            for j in 0..m {
                contracted[j] = centroid[j] + 0.5 * (reflected[j] - centroid[j]);
            }
            let f_c = match f(&contracted) {
                Ok(v) => v,
                Err(e) => {
                    if let Some(cancel) = check_cancel {
                        cancel()?;
                    }
                    return Err(e);
                }
            };
            if f_c < f_r {
                vertices[m] = contracted;
                f_vals[m] = f_c;
                contract = true;
            }
        } else {
            let mut contracted = vec![0.0; m];
            for j in 0..m {
                contracted[j] = centroid[j] + 0.5 * (vertices[m][j] - centroid[j]);
            }
            let f_c = match f(&contracted) {
                Ok(v) => v,
                Err(e) => {
                    if let Some(cancel) = check_cancel {
                        cancel()?;
                    }
                    return Err(e);
                }
            };
            if f_c < f_vals[m] {
                vertices[m] = contracted;
                f_vals[m] = f_c;
                contract = true;
            }
        }

        if contract {
            continue;
        }

        for i in 1..=m {
            if let Some(cancel) = check_cancel {
                cancel()?;
            }
            for j in 0..m {
                vertices[i][j] = vertices[0][j] + 0.5 * (vertices[i][j] - vertices[0][j]);
            }
            f_vals[i] = f(&vertices[i])?;
        }
    }

    Ok(vertices[0].clone())
}

/// Finite-difference Hessian of `g = -f` (so `f` is a negate-log-posterior style objective).
///
/// Returns the Hessian of `g` at `mode` (row-major m×m).
pub fn compute_hessian(
    mode: &[f64],
    f: &dyn Fn(&[f64]) -> Result<f64, String>,
    h: f64,
) -> Result<Vec<f64>, String> {
    compute_hessian_cancellable(mode, f, h, None)
}

/// Finite-difference Hessian of `g = -f` with optional cancellation check.
pub fn compute_hessian_cancellable(
    mode: &[f64],
    f: &dyn Fn(&[f64]) -> Result<f64, String>,
    h: f64,
    check_cancel: Option<&dyn Fn() -> Result<(), String>>,
) -> Result<Vec<f64>, String> {
    let m = mode.len();
    let mut hessian = vec![0.0; m * m];

    let eval_g = |theta: &[f64]| -> Result<f64, String> {
        if let Some(cancel) = check_cancel {
            cancel()?;
        }
        let v = f(theta)?;
        Ok(-v)
    };

    let g_mode = eval_g(mode)?;

    for i in 0..m {
        for j in 0..m {
            if i == j {
                let mut theta_plus = mode.to_vec();
                theta_plus[i] += h;
                let mut theta_minus = mode.to_vec();
                theta_minus[i] -= h;

                let g_plus = eval_g(&theta_plus)?;
                let g_minus = eval_g(&theta_minus)?;

                let val = (g_plus - 2.0 * g_mode + g_minus) / (h * h);
                hessian[i * m + j] = val;
            } else if i < j {
                let mut tp_pp = mode.to_vec();
                tp_pp[i] += h;
                tp_pp[j] += h;

                let mut tp_pm = mode.to_vec();
                tp_pm[i] += h;
                tp_pm[j] -= h;

                let mut tp_mp = mode.to_vec();
                tp_mp[i] -= h;
                tp_mp[j] += h;

                let mut tp_mm = mode.to_vec();
                tp_mm[i] -= h;
                tp_mm[j] -= h;

                let g_pp = eval_g(&tp_pp)?;
                let g_pm = eval_g(&tp_pm)?;
                let g_mp = eval_g(&tp_mp)?;
                let g_mm = eval_g(&tp_mm)?;

                let val = (g_pp - g_pm - g_mp + g_mm) / (4.0 * h * h);
                hessian[i * m + j] = val;
                hessian[j * m + i] = val;
            }
        }
    }

    Ok(hessian)
}

