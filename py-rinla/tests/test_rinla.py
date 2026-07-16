"""Python binding smoke / integration tests.

NumPy/SciPy are used only for:
  - SciPy CSC interop checks (`to_scipy`)
  - simulating synthetic data (RNG + Cholesky of a *rinla* precision)

Hyperparameter maps, latent sizes, and precision matrices come from `rinla`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import rinla
import scipy.sparse as sp


def _ar1_rho_from_intern(theta1: float) -> float:
    """Match R-INLA / r-inla AR(1) `from.theta` for rho ∈ (-1, 1)."""
    return 2.0 / (1.0 + math.exp(-theta1)) - 1.0


def _sample_gmrf(q_csc: rinla.PyCscMatrix, rng: np.random.Generator) -> np.ndarray:
    """Draw x ~ N(0, Q^{-1}) using rinla's precision (via dense Cholesky)."""
    q = q_csc.to_scipy().toarray()
    # Q = L Lᵀ  ⇒  x = L^{-T} z has covariance Q^{-1}
    l = np.linalg.cholesky(q)
    z = rng.standard_normal(q.shape[0])
    return np.linalg.solve(l.T, z)


def test_scipy_conversion():
    triplets = rinla.ar1_precision_matrix(5, 0.7, 1.0)
    assert len(triplets) == 3

    mat = rinla.PyCscMatrix(5, 5, triplets[0], triplets[1], triplets[2])
    assert mat.shape == (5, 5)

    sp_mat = mat.to_scipy()
    assert sp.isspmatrix_csc(sp_mat)
    assert sp_mat.shape == (5, 5)
    assert hasattr(sp_mat, "_base_matrix")

    dense = sp_mat.toarray()
    assert dense[0, 0] == pytest.approx(1.0)
    assert dense[0, 1] == pytest.approx(-0.7)

    csc_mat = rinla.ar1_precision_matrix_csc(5, 0.7, 1.0)
    assert csc_mat.to_scipy().shape == (5, 5)


def test_fgn_matrices():
    q_fgn = rinla.fgn_precision_matrix(5, 0.7, 1.5)
    assert q_fgn.shape == (5, 5)
    assert q_fgn.to_scipy().nnz == 25  # exact FGN Q is dense

    n = 5
    order = 4
    q_approx = rinla.fgn_approx_precision_matrix(n, 0.7, 1.0, order=order)
    assert q_approx.shape == (rinla.fgn_approx_latent_len(n, order),) * 2

    # Round-trip Hurst ↔ internal θ through the Rust helpers
    h = 0.7
    h_int = rinla.fgn_intern_from_hurst(h)
    assert rinla.fgn_hurst_from_intern(h_int) == pytest.approx(h, rel=1e-12)


def test_inference_ar1():
    rng = np.random.default_rng(42)
    n = 20
    rho_true, tau_true = 0.7, 4.0  # innovation precision ≈ 1/0.5²
    x = _sample_gmrf(rinla.ar1_precision_matrix_csc(n, rho_true, tau_true), rng)
    y = x + rng.normal(0.0, 0.2, n)
    obs_prec = 1.0 / (0.2**2)
    obs = [{"family": "gaussian", "y": float(yi), "precision": obs_prec} for yi in y]

    def build_prior(theta):
        tau = math.exp(theta[0])
        rho = _ar1_rho_from_intern(theta[1])
        return rinla.ar1_precision_matrix_csc(n, rho, tau)

    def log_prior_density(theta):
        return -0.5 * 0.1 * (theta[0] ** 2 + theta[1] ** 2)

    res = rinla.run_inla_inference(
        initial_theta=[0.0, 0.0],
        build_prior=build_prior,
        log_prior_density=log_prior_density,
        obs=obs,
        strategy="ccd",
    )

    assert len(res.mode) == 2
    assert 0.0 < res.mode[0] < 2.0
    assert 0.5 < res.mode[1] < 3.0
    assert res.marginal_log_lik < 0.0
    assert len(res.predictor_means) == n
    assert len(res.latent_means) == n
    assert len(res.internal_marginals_hyperpar) == 2
    q = res.internal_marginals_hyperpar[0].quantiles([0.025, 0.5, 0.975])
    assert q[0] < q[1] < q[2]


def test_inference_fgn():
    """Exact FGN: θ₁ uses logistic → H ∈ (0, 1) (same as r-inla structured path)."""
    rng = np.random.default_rng(123)
    n = 30
    h_true = 0.7
    x = _sample_gmrf(rinla.fgn_precision_matrix(n, h_true, 1.0), rng)
    noise_sd = 0.0316  # ≈ 1/√1000
    y = x + rng.normal(0.0, noise_sd, n)
    obs = [{"family": "gaussian", "y": float(yi), "precision": 1000.0} for yi in y]

    def build_prior(theta):
        tau = math.exp(theta[0])
        # Exact dense FGN in Rust takes H ∈ (0, 1); keep the logistic map used by r-inla.
        hurst = 1.0 / (1.0 + math.exp(-theta[1]))
        return rinla.fgn_precision_matrix(n, hurst, tau)

    def log_prior_density(theta):
        return -0.5 * 0.1 * (theta[0] ** 2 + theta[1] ** 2)

    res = rinla.run_inla_inference(
        initial_theta=[0.0, 0.0],
        build_prior=build_prior,
        log_prior_density=log_prior_density,
        obs=obs,
        strategy="ccd",
    )

    est_h = 1.0 / (1.0 + math.exp(-res.mode[1]))
    assert 0.5 < est_h < 0.9


def test_inference_fgn_approx():
    """Approx FGN: θ₁ is H_intern via rinla.fgn_hurst_from_intern (H ∈ (1/2, 1))."""
    rng = np.random.default_rng(123)
    n = 30
    order = 4
    h_true = 0.7
    # Truth from exact FGN Q (engine under test is the AR-mixture approx).
    x = _sample_gmrf(rinla.fgn_precision_matrix(n, h_true, 1.0), rng)
    obs_prec = math.exp(8.0)
    y = x + rng.normal(0.0, 1.0 / math.sqrt(obs_prec), n)

    n_latent = rinla.fgn_approx_latent_len(n, order)
    obs = [{"family": "gaussian", "y": float(yi), "precision": obs_prec} for yi in y]
    obs += [None] * (n_latent - n)

    def build_prior(theta):
        tau = math.exp(theta[0])
        hurst = rinla.fgn_hurst_from_intern(theta[1])
        return rinla.fgn_approx_precision_matrix(n, hurst, tau, order=order)

    def log_prior_density(theta):
        return -0.5 * 0.1 * (theta[0] ** 2 + theta[1] ** 2)

    res = rinla.run_inla_inference(
        initial_theta=[1.0, rinla.fgn_intern_from_hurst(h_true)],
        build_prior=build_prior,
        log_prior_density=log_prior_density,
        obs=obs,
        strategy="ccd",
    )

    est_h = rinla.fgn_hurst_from_intern(res.mode[1])
    assert 0.5 < est_h < 0.9
    assert len(res.latent_means) == n_latent


def test_inference_rw2():
    rng = np.random.default_rng(42)
    n = 20
    t = np.linspace(1.0 / n, 1.0, n)
    y = t**2 + rng.normal(0.0, 0.05, n)
    obs = [{"family": "gaussian", "y": float(yi), "precision": 100.0} for yi in y]

    def build_prior(theta):
        return rinla.rw2_precision_matrix(n, math.exp(theta[0]))

    def log_prior_density(theta):
        return -0.5 * 0.1 * theta[0] ** 2

    res = rinla.run_inla_inference(
        initial_theta=[0.0],
        build_prior=build_prior,
        log_prior_density=log_prior_density,
        obs=obs,
        strategy="ccd",
    )
    assert len(res.mode) == 1
    assert res.mode[0] > 0.0


def test_non_gaussian_families():
    def log_prior_iid(theta):
        return -0.5 * 0.1 * theta[0] ** 2

    # Poisson + IID
    counts = [2, 3, 2, 4, 3, 2, 3, 2]
    n = len(counts)
    obs_pois = [
        {"family": "poisson", "y": float(c), "exposure": 1.0} for c in counts
    ]
    res_pois = rinla.run_inla_inference(
        initial_theta=[1.0],
        build_prior=lambda th: rinla.iid_precision_matrix(n, math.exp(th[0])),
        log_prior_density=log_prior_iid,
        obs=obs_pois,
        strategy="ccd",
    )
    assert len(res_pois.mode) == 1
    assert res_pois.marginal_log_lik < 0.0

    # Binomial + IID
    ys_b = [2, 5, 3, 7, 4, 6]
    n_b = len(ys_b)
    obs_bin = [{"family": "binomial", "y": float(y), "n": 10.0} for y in ys_b]
    res_bin = rinla.run_inla_inference(
        initial_theta=[0.0],
        build_prior=lambda th: rinla.iid_precision_matrix(n_b, math.exp(th[0])),
        log_prior_density=log_prior_iid,
        obs=obs_bin,
        strategy="ccd",
    )
    assert len(res_bin.mode) == 1

    # Laplace + IID
    y_lap = [0.2, -0.1, 0.4, 0.0, -0.3, 0.1, 0.2, -0.2]
    n_l = len(y_lap)
    obs_lap = [
        {"family": "laplace", "y": float(y), "alpha": 0.5, "gamma": 0.2} for y in y_lap
    ]
    res_lap = rinla.run_inla_inference(
        initial_theta=[1.0],
        build_prior=lambda th: rinla.iid_precision_matrix(n_l, math.exp(th[0])),
        log_prior_density=log_prior_iid,
        obs=obs_lap,
        strategy="ccd",
    )
    assert len(res_lap.mode) == 1


def test_family_aliases():
    """R-INLA-style family names accepted by the Python binder."""
    n = 4
    obs = [
        {"family": "nbinomial", "y": 1.0, "exposure": 1.0, "size": 5.0}
        for _ in range(n)
    ]
    res = rinla.run_inla_inference(
        initial_theta=[0.0],
        build_prior=lambda th: rinla.iid_precision_matrix(n, math.exp(th[0])),
        log_prior_density=lambda th: -0.5 * 0.1 * th[0] ** 2,
        obs=obs,
        strategy="ccd",
    )
    assert len(res.mode) == 1
    assert math.isfinite(res.marginal_log_lik)
