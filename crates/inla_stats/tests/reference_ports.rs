//! Ports of scenarios from `reference/r-inla-tests/` onto rust-inla.
//! These call `inla_stats::run_inla_inference`, not classic R-INLA.

use inla_fmesher::{Triangle, Vertex2, build_mesh2d};
use inla_math::{csc_from_triplets_0based, kronecker_csc};
use inla_stats::{
    BinomialObs, ExponentialSurvivalObs, GaussianObs, LaplaceObs, Link, MarginalOptions,
    NegativeBinomialObs, Obs, PoissonObs, WeibullSurvivalObs, ZeroInflatedBinomialObs,
    ZeroInflatedPoissonObs, ZeroInflationType, ar1_precision_csc, arp_precision_csc,
    besag_precision_csc, bym_precision_csc, bym2_precision_csc, crw1_precision_csc,
    crw2_precision_csc, fgn_approx_latent_len, fgn_approx_precision_csc, fgn_hurst_from_intern,
    fgn_precision_csc, iid_precision_csc, matern2d_precision_csc, run_inla_inference,
    run_inla_inference_a, rw1_precision_csc, rw2_precision_csc, rw2d_precision_csc,
    seasonal_precision_csc, spde_params_from_theta, spde_precision_csc, spde_projector_csc,
    sum_to_zero_constraint,
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
    assert!(
        result
            .latent_variances
            .iter()
            .all(|&v| v > 0.0 && v.is_finite())
    );
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
    assert!(
        result.marginal_log_lik_gaussian.is_finite() || result.marginal_log_lik_gaussian.is_nan()
    );
    assert!(result.dic.is_finite());
    assert_eq!(result.cpo.len(), n);
    assert!(
        result
            .cpo
            .iter()
            .all(|v| matches!(v, Some(c) if c.is_finite() && *c > 0.0))
    );
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
    // AR(2): theta length = 1 + p = 3 (theta[0]=log(tau), theta[1..]=logit PACF)
    let build_prior = |theta: &[f64]| {
        let tau = theta[0].exp();
        let pacf1 = (theta[1] * 0.5).tanh();
        let pacf2 = (theta[2] * 0.5).tanh();
        arp_precision_csc(n, &[pacf1, pacf2], tau)
    };
    let result = run_inla_inference(
        &[0.0, 0.5, 0.2],
        &build_prior,
        &log_prior_flatish,
        &obs,
        "ccd",
        1.0,
    )
    .expect("arp");
    assert_finite_result(&result, n, 3);
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
    let result = run_inla_inference(
        &[0.0, 0.0],
        &build_prior,
        &log_prior_flatish,
        &obs,
        "ccd",
        1.0,
    )
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
    let result = run_inla_inference(
        &[1.0, 1.0],
        &build_prior,
        &log_prior_flatish,
        &obs,
        "ccd",
        1.0,
    )
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
    let constr = sum_to_zero_constraint(n, 1).unwrap();
    let result = run_inla_inference_a(
        &[0.0],
        &build_prior,
        &log_prior_flatish,
        &obs,
        None,
        Some(&constr),
        "ccd",
        1.0,
        &MarginalOptions::default(),
        false,
    )
    .expect("rw1");
    assert_finite_result(&result, n, 1);
    let s: f64 = result.latent_means.iter().sum();
    assert!(s.abs() < 1e-4, "rw1 sum-to-zero violated: {s}");
}

#[test]
fn port_rw2_gaussian() {
    let n = 15;
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            t * t
        })
        .collect();
    let obs = gaussian_obs(&y, 40.0);
    let build_prior = |theta: &[f64]| rw2_precision_csc(n, theta[0].exp());
    let constr = sum_to_zero_constraint(n, 2).unwrap();
    let result = run_inla_inference_a(
        &[0.0],
        &build_prior,
        &log_prior_flatish,
        &obs,
        None,
        Some(&constr),
        "ccd",
        1.0,
        &MarginalOptions::default(),
        false,
    )
    .expect("rw2");
    assert_finite_result(&result, n, 1);
    let s: f64 = result.latent_means.iter().sum();
    assert!(s.abs() < 1e-4, "rw2 sum violated: {s}");
    let mean = (n - 1) as f64 / 2.0;
    let lin: f64 = result
        .latent_means
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f64 - mean) * v)
        .sum();
    assert!(lin.abs() < 1e-3, "rw2 linear constraint violated: {lin}");
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
    let y: Vec<f64> = (0..n)
        .map(|i| if i % 2 == 0 { 0.5 } else { -0.3 })
        .collect();
    let obs = gaussian_obs(&y, 10.0);
    let build_prior = |theta: &[f64]| besag_precision_csc(&adj, theta[0].exp());
    let constr = sum_to_zero_constraint(n, 1).unwrap();
    let result = run_inla_inference_a(
        &[0.0],
        &build_prior,
        &log_prior_flatish,
        &obs,
        None,
        Some(&constr),
        "ccd",
        1.0,
        &MarginalOptions::default(),
        false,
    )
    .expect("besag");
    assert_finite_result(&result, n, 1);
    let s: f64 = result.latent_means.iter().sum();
    assert!(s.abs() < 1e-4, "besag sum-to-zero violated: {s}");
}

#[test]
fn port_bym_gaussian() {
    let adj = vec![vec![1usize, 3], vec![0, 2], vec![1, 3], vec![0, 2]];
    let n = adj.len();
    let y: Vec<f64> = (0..n)
        .map(|i| if i % 2 == 0 { 0.4 } else { -0.2 })
        .collect();
    let obs = gaussian_obs(&y, 15.0);
    // A maps obs i → u_i + v_i
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for i in 0..n {
        rows.push(i);
        cols.push(i);
        vals.push(1.0);
        rows.push(i);
        cols.push(n + i);
        vals.push(1.0);
    }
    let a = csc_from_triplets_0based(n, 2 * n, &rows, &cols, &vals).unwrap();
    let build_prior = |theta: &[f64]| bym_precision_csc(&adj, theta[0].exp(), theta[1].exp());
    let constr = sum_to_zero_constraint(n, 1)
        .unwrap()
        .embed(2 * n, 0)
        .unwrap();
    let result = run_inla_inference_a(
        &[1.0, 1.0],
        &build_prior,
        &log_prior_flatish,
        &obs,
        Some(&a),
        Some(&constr),
        "ccd",
        1.0,
        &MarginalOptions::default(),
        false,
    )
    .expect("bym");
    assert_finite_result(&result, 2 * n, 2);
}

#[test]
fn port_bym2_gaussian() {
    let adj = vec![vec![1usize, 3], vec![0, 2], vec![1, 3], vec![0, 2]];
    let n = adj.len();
    let y: Vec<f64> = (0..n)
        .map(|i| if i % 2 == 0 { 0.4 } else { -0.2 })
        .collect();
    let obs = gaussian_obs(&y, 15.0);
    let build_prior = |theta: &[f64]| {
        let tau = theta[0].exp();
        let phi = (1.0 / (1.0 + (-theta[1]).exp())).clamp(1e-6, 1.0 - 1e-6);
        bym2_precision_csc(&adj, tau, phi)
    };
    let constr = sum_to_zero_constraint(n, 1).unwrap();
    let result = run_inla_inference_a(
        &[1.0, 0.0],
        &build_prior,
        &log_prior_flatish,
        &obs,
        None,
        Some(&constr),
        "ccd",
        1.0,
        &MarginalOptions::default(),
        false,
    )
    .expect("bym2");
    assert_finite_result(&result, n, 2);
}

#[test]
fn port_rw2d_gaussian() {
    let nrow = 5;
    let ncol = 5;
    let n = nrow * ncol;
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let r = (i % nrow) as f64;
            let c = (i / nrow) as f64;
            0.2 * r + 0.1 * c + 0.05 * ((r + c) * 0.4).sin()
        })
        .collect();
    let obs = gaussian_obs(&y, 30.0);
    let build_prior = |theta: &[f64]| rw2d_precision_csc(nrow, ncol, theta[0].exp(), false, false);
    let constr = sum_to_zero_constraint(n, 2).unwrap();
    let result = run_inla_inference_a(
        &[0.0],
        &build_prior,
        &log_prior_flatish,
        &obs,
        None,
        Some(&constr),
        "ccd",
        1.0,
        &MarginalOptions::default(),
        false,
    )
    .expect("rw2d");
    assert_finite_result(&result, n, 1);
}

#[test]
fn port_group_besag_ar1_gaussian() {
    // Spatio-temporal: Q = Q_ar1 ⊗ Q_besag (main=space fastest).
    let adj = vec![vec![1usize, 3], vec![0, 2], vec![1, 3], vec![0, 2]];
    let n_space = adj.len();
    let n_time = 4usize;
    let n = n_space * n_time;
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let s = (i % n_space) as f64;
            let t = (i / n_space) as f64;
            0.4 * (s * 0.8).sin() + 0.3 * (t * 0.5).cos()
        })
        .collect();
    let obs = gaussian_obs(&y, 20.0);
    let build_prior = |theta: &[f64]| {
        let tau_s = theta[0].exp();
        let tau_t = theta[1].exp();
        let mut rho = 2.0 / (1.0 + (-theta[2]).exp()) - 1.0;
        rho = rho.clamp(-0.999, 0.999);
        let q_main = besag_precision_csc(&adj, tau_s)?;
        let q_group = ar1_precision_csc(n_time, rho, tau_t)?;
        Ok(kronecker_csc(&q_group, &q_main))
    };
    let constr = sum_to_zero_constraint(n, 1).unwrap();
    let result = run_inla_inference_a(
        &[0.0, 0.0, 0.0],
        &build_prior,
        &log_prior_flatish,
        &obs,
        None,
        Some(&constr),
        "ccd",
        1.0,
        &MarginalOptions::default(),
        false,
    )
    .expect("group besag⊗ar1");
    assert_finite_result(&result, n, 3);
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

#[test]
fn port_iid_nbinom() {
    let ys = [1.0, 3.0, 2.0, 6.0, 4.0, 5.0];
    let n = ys.len();
    let obs: Vec<Obs> = ys
        .iter()
        .map(|&y| {
            Obs::NegativeBinomial(NegativeBinomialObs {
                y,
                exposure: 1.0,
                size: 2.0,
                link: Link::Log,
            })
        })
        .collect();
    let build_prior = |theta: &[f64]| iid_precision_csc(n, theta[0].exp());
    let result = run_inla_inference(&[1.0], &build_prior, &log_prior_flatish, &obs, "ccd", 1.0)
        .expect("nbinom");
    assert_finite_result(&result, n, 1);
}

#[test]
fn port_iid_zip() {
    let ys = [0.0, 2.0, 0.0, 4.0, 1.0, 0.0, 3.0];
    let n = ys.len();
    let obs: Vec<Obs> = ys
        .iter()
        .map(|&y| {
            Obs::ZeroInflatedPoisson(ZeroInflatedPoissonObs {
                y,
                exposure: 1.0,
                zero_prob: 0.2,
                link: Link::Log,
                inflation: ZeroInflationType::Type0,
            })
        })
        .collect();
    let build_prior = |theta: &[f64]| iid_precision_csc(n, theta[0].exp());
    let result = run_inla_inference(&[1.0], &build_prior, &log_prior_flatish, &obs, "ccd", 1.0)
        .expect("zip");
    assert_finite_result(&result, n, 1);
}

#[test]
fn port_iid_zib() {
    let ys = [0.0, 3.0, 0.0, 5.0, 2.0, 0.0];
    let n = ys.len();
    let obs: Vec<Obs> = ys
        .iter()
        .map(|&y| {
            Obs::ZeroInflatedBinomial(ZeroInflatedBinomialObs {
                y,
                n: 10.0,
                zero_prob: 0.15,
                link: Link::Logit,
                inflation: ZeroInflationType::Type0,
            })
        })
        .collect();
    let build_prior = |theta: &[f64]| iid_precision_csc(n, theta[0].exp());
    let result = run_inla_inference(&[0.0], &build_prior, &log_prior_flatish, &obs, "ccd", 1.0)
        .expect("zib");
    assert_finite_result(&result, n, 1);
}

#[test]
fn port_iid_exponential_survival() {
    // Survival times with right-censoring indicator (1=observed, 0=censored)
    let times = [1.2, 2.5, 0.8, 3.1, 1.9, 2.0];
    let events = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    let n = times.len();
    let obs: Vec<Obs> = times
        .iter()
        .zip(events.iter())
        .map(|(&t, &e)| {
            Obs::ExponentialSurvival(ExponentialSurvivalObs {
                y: t,
                event: e,
                link: Link::Log,
            })
        })
        .collect();
    let build_prior = |theta: &[f64]| iid_precision_csc(n, theta[0].exp());
    let result = run_inla_inference(&[0.0], &build_prior, &log_prior_flatish, &obs, "ccd", 1.0)
        .expect("exponential survival");
    assert_finite_result(&result, n, 1);
}

#[test]
fn port_iid_weibull_survival() {
    let times = [0.9, 1.8, 2.3, 0.5, 1.4, 2.1];
    let events = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let n = times.len();
    let obs: Vec<Obs> = times
        .iter()
        .zip(events.iter())
        .map(|(&t, &e)| {
            Obs::WeibullSurvival(WeibullSurvivalObs {
                y: t,
                event: e,
                shape: 1.5,
                link: Link::Log,
            })
        })
        .collect();
    let build_prior = |theta: &[f64]| iid_precision_csc(n, theta[0].exp());
    let result = run_inla_inference(&[0.0], &build_prior, &log_prior_flatish, &obs, "ccd", 1.0)
        .expect("weibull survival");
    assert_finite_result(&result, n, 1);
}

#[test]
fn port_iid_laplace() {
    let y = [0.2, -0.5, 0.8, 0.1, -0.3, 0.4];
    let n = y.len();
    let obs: Vec<Obs> = y
        .iter()
        .map(|&yi| {
            Obs::Laplace(LaplaceObs {
                y: yi,
                alpha: 0.5,
                gamma: 0.5,
                link: Link::Identity,
            })
        })
        .collect();
    let build_prior = |theta: &[f64]| iid_precision_csc(n, theta[0].exp());
    let result = run_inla_inference(&[0.0], &build_prior, &log_prior_flatish, &obs, "ccd", 1.0)
        .expect("laplace");
    assert_finite_result(&result, n, 1);
}

#[test]
fn port_crw1_gaussian() {
    let pos = [0.0, 1.2, 2.5, 4.0, 5.5, 7.0];
    let n = pos.len();
    let y: Vec<f64> = pos.iter().map(|&p| p * 0.2 + (p * 0.5_f64).sin()).collect();
    let obs = gaussian_obs(&y, 25.0);
    let build_prior = |theta: &[f64]| crw1_precision_csc(&pos, theta[0].exp());
    let constr = sum_to_zero_constraint(n, 1).unwrap();
    let result = run_inla_inference_a(
        &[0.0],
        &build_prior,
        &log_prior_flatish,
        &obs,
        None,
        Some(&constr),
        "ccd",
        1.0,
        &MarginalOptions::default(),
        false,
    )
    .expect("crw1");
    assert_finite_result(&result, n, 1);
}

#[test]
fn port_crw2_gaussian() {
    let pos = [0.0, 1.0, 2.5, 4.0, 6.0, 8.0];
    let n = pos.len();
    let y: Vec<f64> = pos.iter().map(|&p| (p * 0.3_f64).sin()).collect();
    let obs = gaussian_obs(&y, 30.0);
    let build_prior = |theta: &[f64]| crw2_precision_csc(&pos, theta[0].exp(), "simple");
    let constr = sum_to_zero_constraint(n, 2).unwrap();
    let result = run_inla_inference_a(
        &[0.0],
        &build_prior,
        &log_prior_flatish,
        &obs,
        None,
        Some(&constr),
        "ccd",
        1.0,
        &MarginalOptions::default(),
        false,
    )
    .expect("crw2 simple");
    assert_finite_result(&result, n, 1);
}

#[test]
fn port_crw2_pairs_gaussian() {
    let pos = [0.0, 1.0, 2.5, 4.0, 6.0];
    let n = pos.len();
    let y: Vec<f64> = pos.iter().map(|&p| (p * 0.3_f64).sin()).collect();
    let obs = gaussian_obs(&y, 30.0);
    // Observe value component only (pairs layout: cols 0,2,4,...)
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for i in 0..n {
        rows.push(i);
        cols.push(2 * i);
        vals.push(1.0);
    }
    let a = csc_from_triplets_0based(n, 2 * n, &rows, &cols, &vals).unwrap();
    let build_prior = |theta: &[f64]| crw2_precision_csc(&pos, theta[0].exp(), "pairs");
    let result = run_inla_inference_a(
        &[0.0],
        &build_prior,
        &log_prior_flatish,
        &obs,
        Some(&a),
        None,
        "ccd",
        1.0,
        &MarginalOptions::default(),
        false,
    )
    .expect("crw2 pairs");
    assert_finite_result(&result, 2 * n, 1);
}

#[test]
fn port_matern2d_gaussian() {
    // Lattice Matérn: observations at every grid node ⇒ A = I.
    let nrow = 4;
    let ncol = 4;
    let n = nrow * ncol;
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let r = (i % nrow) as f64;
            let c = (i / nrow) as f64;
            0.3 * (r * 0.7).sin() + 0.2 * (c * 0.5).cos()
        })
        .collect();
    let obs = gaussian_obs(&y, 40.0);
    let build_prior = |theta: &[f64]| {
        let prec = theta[0].exp();
        matern2d_precision_csc(nrow, ncol, 1, 2.0, prec, false)
    };
    let result = run_inla_inference(&[0.0], &build_prior, &log_prior_flatish, &obs, "ccd", 1.0)
        .expect("matern2d");
    assert_finite_result(&result, n, 1);
}

#[test]
fn port_spde_gaussian() {
    // Unit square mesh (2 triangles) + interior observation locations.
    // Latent dim = #vertices; projector A maps field → η at locs.
    let vertices = vec![
        Vertex2 { x: 0.0, y: 0.0 },
        Vertex2 { x: 1.0, y: 0.0 },
        Vertex2 { x: 1.0, y: 1.0 },
        Vertex2 { x: 0.0, y: 1.0 },
        Vertex2 { x: 0.5, y: 0.5 },
    ];
    let triangles = vec![
        Triangle([0, 1, 4]),
        Triangle([1, 2, 4]),
        Triangle([2, 3, 4]),
        Triangle([3, 0, 4]),
    ];
    let mesh = build_mesh2d(vertices, triangles).expect("mesh");
    let fem = mesh.assemble_fem_blocks();
    let n_latent = mesh.vertices.len();

    let locs = [
        Vertex2 { x: 0.25, y: 0.25 },
        Vertex2 { x: 0.75, y: 0.25 },
        Vertex2 { x: 0.75, y: 0.75 },
        Vertex2 { x: 0.25, y: 0.75 },
        Vertex2 { x: 0.5, y: 0.5 },
        Vertex2 { x: 0.4, y: 0.6 },
    ];
    let a = spde_projector_csc(&mesh, &locs).expect("projector A");
    assert_eq!(a.cols(), n_latent);
    assert_eq!(a.rows(), locs.len());

    let y: Vec<f64> = locs
        .iter()
        .map(|p| 0.4 * (p.x * 2.0).sin() + 0.3 * (p.y * 1.5).cos())
        .collect();
    let obs = gaussian_obs(&y, 50.0);

    let build_prior = move |theta: &[f64]| {
        let (tau, kappa) = spde_params_from_theta(theta)?;
        spde_precision_csc(&fem, kappa, tau)
    };
    // Optional sum-to-zero when an intercept is present; here field-only.
    let result = run_inla_inference_a(
        &[0.0, 0.0], // log_tau, log_kappa
        &build_prior,
        &log_prior_flatish,
        &obs,
        Some(&a),
        None,
        "ccd",
        1.0,
        &MarginalOptions::default(),
        false,
    )
    .expect("spde");
    assert_finite_result(&result, n_latent, 2);
}
