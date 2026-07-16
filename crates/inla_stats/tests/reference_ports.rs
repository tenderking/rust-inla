//! Ports of scenarios from `reference/r-inla-tests/` onto rust-inla.
//! These call `inla_stats::run_inla_inference`, not classic R-INLA.

use inla_stats::{
    BinomialObs, GaussianObs, Link, Obs, PoissonObs, ar1_precision_csc, arp_precision_csc,
    besag_precision_csc, fgn_approx_latent_len, fgn_approx_precision_csc, fgn_hurst_from_intern,
    fgn_precision_csc, iid_precision_csc, run_inla_inference, rw1_precision_csc,
    rw2_precision_csc, seasonal_precision_csc,
};

fn log_prior_flatish(theta: &[f64]) -> f64 {
    theta.iter().map(|&v| -0.5 * 0.1 * v * v).sum()
}

fn assert_finite_result(result: &inla_stats::InferenceResult, n: usize, m: usize) {
    assert_eq!(result.mode.len(), m);
    assert!(result.mode.iter().all(|v| v.is_finite()));
    assert_eq!(result.latent_means.len(), n);
    assert_eq!(result.latent_variances.len(), n);
    assert!(result.latent_means.iter().all(|v| v.is_finite()));
    assert!(result.latent_variances.iter().all(|&v| v > 0.0 && v.is_finite()));
    assert!(result.marginal_log_lik.is_finite());
}

fn gaussian_obs(y: &[f64], precision: f64) -> Vec<Obs> {
    y.iter()
        .map(|&yi| {
            Obs::Gaussian(GaussianObs {
                y: yi,
                precision,
                link: Link::Identity,
            })
        })
        .collect()
}

#[test]
fn port_iid_gaussian() {
    let y = [1.0, 1.2, 0.9, 1.1, 0.8];
    let n = y.len();
    let obs = gaussian_obs(&y, 4.0);
    let build_prior = |theta: &[f64]| iid_precision_csc(n, theta[0].exp());
    let result = run_inla_inference(&[0.0], &build_prior, &log_prior_flatish, &obs, "ccd", 1.0)
        .expect("iid gaussian");
    assert_finite_result(&result, n, 1);
}

#[test]
fn port_iid_gaussian_model_selection() {
    let y = [0.5, 1.0, 1.5, 0.8, 1.2, 0.9];
    let n = y.len();
    let obs = gaussian_obs(&y, 2.0);
    let build_prior = |theta: &[f64]| iid_precision_csc(n, theta[0].exp());
    let result = run_inla_inference(&[0.0], &build_prior, &log_prior_flatish, &obs, "ccd", 1.0)
        .expect("model selection");
    assert!(result.marginal_log_lik.is_finite());
    assert!(result.marginal_log_lik_gaussian.is_finite() || result.marginal_log_lik_gaussian.is_nan());
    assert!(result.dic.is_finite());
    assert_eq!(result.cpo.len(), n);
    assert!(result.cpo.iter().all(|v| matches!(v, Some(c) if c.is_finite() && *c > 0.0)));
}

#[test]
fn port_ar1_gaussian() {
    let n = 12;
    let y: Vec<f64> = (0..n).map(|i| 0.3 * ((i as f64) * 0.5).sin()).collect();
    let obs = gaussian_obs(&y, 100.0);
    let build_prior = |theta: &[f64]| {
        let tau = if theta[0].is_finite() {
            theta[0].exp().clamp(0.1, 100.0)
        } else {
            1.0
        };
        ar1_precision_csc(n, 0.5, tau)
    };
    let result = run_inla_inference(&[0.0], &build_prior, &log_prior_flatish, &obs, "ccd", 1.0)
        .expect("ar1");
    assert_finite_result(&result, n, 1);
}

#[test]
fn port_arp_gaussian() {
    let n = 16;
    let y: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.4).sin() * 0.5).collect();
    let obs = gaussian_obs(&y, 16.0);
    let build_prior = |theta: &[f64]| {
        let tau = theta[0].exp();
        // Fixed mild AR(2) PACF; only tau free (keeps m=1 for speed/stability).
        let _ = theta;
        arp_precision_csc(n, &[0.4, 0.2], tau)
    };
    let result = run_inla_inference(&[0.0], &build_prior, &log_prior_flatish, &obs, "ccd", 1.0)
        .expect("arp");
    assert_finite_result(&result, n, 1);
}

#[test]
fn port_fgn_gaussian() {
    let n = 30;
    let y: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.15).sin()).collect();
    let obs = gaussian_obs(&y, 100.0);
    let build_prior = |theta: &[f64]| {
        let tau = theta[0].exp();
        let hurst = 1.0 / (1.0 + (-theta[1]).exp());
        fgn_precision_csc(n, hurst, tau)
    };
    let result =
        run_inla_inference(&[0.0, 0.0], &build_prior, &log_prior_flatish, &obs, "ccd", 1.0)
            .expect("fgn");
    assert_finite_result(&result, n, 2);
    let est_h = 1.0 / (1.0 + (-result.mode[1]).exp());
    assert!(est_h > 0.0 && est_h < 1.0);
}

#[test]
fn port_fgn_approx_order4_gaussian() {
    // R-INLA-style AR-mixture FGN (order=4). Keep n small: latent = 5n.
    let n = 20;
    let y: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.2).sin() * 0.5).collect();
    let n_lat = fgn_approx_latent_len(n, 4);
    let mut obs: Vec<Obs> = y
        .iter()
        .map(|&yi| {
            Obs::Gaussian(GaussianObs {
                y: yi,
                precision: 1e4,
                link: Link::Identity,
            })
        })
        .collect();
    obs.extend(std::iter::repeat(Obs::None).take(n_lat - n));

    let build_prior = |theta: &[f64]| {
        let tau = theta[0].exp().clamp(1e-3, 1e4);
        let hurst = fgn_hurst_from_intern(theta[1].clamp(-6.0, 6.0));
        fgn_approx_precision_csc(n, hurst, tau, 4, 1e8)
    };
    let result =
        run_inla_inference(&[1.0, 1.0], &build_prior, &log_prior_flatish, &obs, "ccd", 1.0)
            .expect("fgn approx");
    assert_eq!(result.mode.len(), 2);
    assert!(result.mode.iter().all(|v| v.is_finite()));
    let h = fgn_hurst_from_intern(result.mode[1]);
    assert!(h > 0.5 && h < 1.0, "H={h}");
    assert_eq!(result.latent_means.len(), n_lat);
}

#[test]
fn port_rw1_gaussian() {
    let n = 15;
    let y: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();
    let obs = gaussian_obs(&y, 40.0);
    let build_prior = |theta: &[f64]| rw1_precision_csc(n, theta[0].exp());
    let result = run_inla_inference(&[0.0], &build_prior, &log_prior_flatish, &obs, "ccd", 1.0)
        .expect("rw1");
    assert_finite_result(&result, n, 1);
}

#[test]
fn port_rw2_gaussian() {
    let n = 15;
    let y: Vec<f64> = (0..n).map(|i| {
        let t = i as f64 / n as f64;
        t * t
    }).collect();
    let obs = gaussian_obs(&y, 40.0);
    let build_prior = |theta: &[f64]| rw2_precision_csc(n, theta[0].exp());
    let result = run_inla_inference(&[0.0], &build_prior, &log_prior_flatish, &obs, "ccd", 1.0)
        .expect("rw2");
    assert_finite_result(&result, n, 1);
}

#[test]
fn port_seasonal_gaussian() {
    let n = 24;
    let s = 4usize;
    let y: Vec<f64> = (0..n)
        .map(|i| ((i % s) as f64) * 0.2 + ((i as f64) * 0.05).sin() * 0.1)
        .collect();
    let obs = gaussian_obs(&y, 25.0);
    let build_prior = |theta: &[f64]| seasonal_precision_csc(n, s, theta[0].exp(), true);
    let result = run_inla_inference(&[0.0], &build_prior, &log_prior_flatish, &obs, "ccd", 1.0)
        .expect("seasonal");
    assert_finite_result(&result, n, 1);
}

#[test]
fn port_besag_gaussian() {
    // Cycle graph of size 6 (same style as besag unit test).
    let adj = vec![
        vec![1usize, 5],
        vec![0, 2],
        vec![1, 3],
        vec![2, 4],
        vec![3, 5],
        vec![4, 0],
    ];
    let n = adj.len();
    let y: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 0.5 } else { -0.3 }).collect();
    let obs = gaussian_obs(&y, 10.0);
    let build_prior = |theta: &[f64]| besag_precision_csc(&adj, theta[0].exp());
    let result = run_inla_inference(&[0.0], &build_prior, &log_prior_flatish, &obs, "ccd", 1.0)
        .expect("besag");
    assert_finite_result(&result, n, 1);
}

#[test]
fn port_iid_poisson() {
    // Mild counts; stronger latent prior helps Newton on the log-link.
    let counts = [2.0, 3.0, 2.0, 4.0, 3.0, 2.0, 3.0, 2.0];
    let n = counts.len();
    let obs: Vec<Obs> = counts
        .iter()
        .map(|&y| {
            Obs::Poisson(PoissonObs {
                y,
                exposure: 1.0,
                link: Link::Log,
            })
        })
        .collect();
    let build_prior = |theta: &[f64]| {
        let tau = theta[0].exp().clamp(1e-3, 1e6);
        iid_precision_csc(n, tau)
    };
    let result = run_inla_inference(&[1.0], &build_prior, &log_prior_flatish, &obs, "ccd", 1.0)
        .expect("poisson");
    assert_finite_result(&result, n, 1);
}

#[test]
fn port_iid_binomial() {
    let ys = [2.0, 5.0, 3.0, 7.0, 4.0, 6.0];
    let ntrials = 10.0;
    let n = ys.len();
    let obs: Vec<Obs> = ys
        .iter()
        .map(|&y| {
            Obs::Binomial(BinomialObs {
                y,
                n: ntrials,
                link: Link::Logit,
            })
        })
        .collect();
    let build_prior = |theta: &[f64]| iid_precision_csc(n, theta[0].exp());
    let result = run_inla_inference(&[0.0], &build_prior, &log_prior_flatish, &obs, "ccd", 1.0)
        .expect("binomial");
    assert_finite_result(&result, n, 1);
}
