use crate::sparse::CscMatrix;

pub struct ModelConfig<'a> {
    pub build_prior: &'a dyn Fn(&[f64]) -> Result<CscMatrix, String>,
    pub log_prior_density: &'a dyn Fn(&[f64]) -> f64,
    pub obs: &'a [crate::inference::Obs],
}

pub fn evaluate_neg_log_posterior(
    theta: &[f64],
    config: &ModelConfig,
) -> Result<f64, String> {
    let q_prior = (config.build_prior)(theta)?;
    let (_x_star, _factor, marginal_log_lik) =
        crate::inference::find_latent_mode(&q_prior, config.obs, 50, 1e-5)?;
    let log_prior = (config.log_prior_density)(theta);
    Ok(-(marginal_log_lik + log_prior))
}

pub fn nelder_mead(
    initial: &[f64],
    step_size: f64,
    max_iter: usize,
    tol: f64,
    config: &ModelConfig,
) -> Result<Vec<f64>, String> {
    let m = initial.len();
    let mut vertices = vec![vec![0.0; m]; m + 1];
    let mut f_vals = vec![0.0; m + 1];

    vertices[0] = initial.to_vec();
    f_vals[0] = evaluate_neg_log_posterior(&vertices[0], config)?;

    for i in 1..=m {
        let mut v = initial.to_vec();
        v[i - 1] += step_size;
        vertices[i] = v;
        f_vals[i] = evaluate_neg_log_posterior(&vertices[i], config)?;
    }

    for _iter in 0..max_iter {
        let mut idxs: Vec<usize> = (0..=m).collect();
        idxs.sort_by(|&a, &b| f_vals[a].partial_cmp(&f_vals[b]).unwrap());

        let mut sorted_vertices = vec![vec![0.0; m]; m + 1];
        let mut sorted_f = vec![0.0; m + 1];
        for i in 0..=m {
            sorted_vertices[i] = vertices[idxs[i]].clone();
            sorted_f[i] = f_vals[idxs[i]];
        }
        vertices = sorted_vertices;
        f_vals = sorted_f;

        let mean_f: f64 = f_vals.iter().sum::<f64>() / (m + 1) as f64;
        let variance_f: f64 = f_vals.iter().map(|&v| (v - mean_f).powi(2)).sum::<f64>() / (m + 1) as f64;
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
        let f_r = evaluate_neg_log_posterior(&reflected, config).unwrap_or(f64::INFINITY);

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
            let f_e = evaluate_neg_log_posterior(&expanded, config).unwrap_or(f64::INFINITY);
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
            let f_c = evaluate_neg_log_posterior(&contracted, config).unwrap_or(f64::INFINITY);
            if f_c < f_r {
                vertices[m] = contracted;
                f_vals[m] = f_c;
                contract = true;
            }
        } else {
            let mut contracted = vec![0.0; m];
            for j in 0..m {
                contracted[j] = centroid[j] - 0.5 * (centroid[j] - vertices[m][j]);
            }
            let f_c = evaluate_neg_log_posterior(&contracted, config).unwrap_or(f64::INFINITY);
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
            for j in 0..m {
                vertices[i][j] = vertices[0][j] + 0.5 * (vertices[i][j] - vertices[0][j]);
            }
            f_vals[i] = evaluate_neg_log_posterior(&vertices[i], config)?;
        }
    }

    Ok(vertices[0].clone())
}

pub fn compute_hessian(
    mode: &[f64],
    config: &ModelConfig,
    h: f64,
) -> Result<Vec<f64>, String> {
    let m = mode.len();
    let mut hessian = vec![0.0; m * m];

    let g = |theta: &[f64]| -> f64 {
        match evaluate_neg_log_posterior(theta, config) {
            Ok(v) => -v,
            Err(_) => f64::NEG_INFINITY,
        }
    };

    let g_mode = g(mode);

    for i in 0..m {
        for j in 0..m {
            if i == j {
                let mut theta_plus = mode.to_vec();
                theta_plus[i] += h;
                let mut theta_minus = mode.to_vec();
                theta_minus[i] -= h;

                let val = (g(&theta_plus) - 2.0 * g_mode + g(&theta_minus)) / (h * h);
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

                let val = (g(&tp_pp) - g(&tp_pm) - g(&tp_mp) + g(&tp_mm)) / (4.0 * h * h);
                hessian[i * m + j] = val;
                hessian[j * m + i] = val;
            }
        }
    }

    Ok(hessian)
}
