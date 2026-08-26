"""High-level ``inla()`` smoke / integration tests.

Matrix helpers are used only to simulate GMRF draws for synthetic data.
Inference goes through ``import inla; inla(...)`` only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sp

import inla


def _sample_gmrf(q_csc: inla.PyCscMatrix, rng: np.random.Generator) -> np.ndarray:
    q = q_csc.to_scipy().toarray()
    chol = np.linalg.cholesky(q)
    z = rng.standard_normal(q.shape[0])
    return np.linalg.solve(chol.T, z)


def test_scipy_conversion():
    triplets = inla.ar1_precision_matrix(5, 0.7, 1.0)
    assert len(triplets) == 3

    mat = inla.PyCscMatrix(5, 5, triplets[0], triplets[1], triplets[2])
    assert mat.shape == (5, 5)

    sp_mat = mat.to_scipy()
    assert sp.isspmatrix_csc(sp_mat)
    assert sp_mat.shape == (5, 5)

    dense = sp_mat.toarray()
    assert dense[0, 0] == pytest.approx(1.0)
    assert dense[0, 1] == pytest.approx(-0.7)

    csc_mat = inla.ar1_precision_matrix_csc(5, 0.7, 1.0)
    assert csc_mat.to_scipy().shape == (5, 5)


def test_fgn_matrices():
    q_fgn = inla.fgn_precision_matrix(5, 0.7, 1.5)
    assert q_fgn.shape == (5, 5)
    assert q_fgn.to_scipy().nnz == 25

    n = 5
    order = 4
    q_approx = inla.fgn_approx_precision_matrix(n, 0.7, 1.0, order=order)
    assert q_approx.shape == (inla.fgn_approx_latent_len(n, order),) * 2

    h = 0.7
    h_int = inla.fgn_intern_from_hurst(h)
    assert inla.fgn_hurst_from_intern(h_int) == pytest.approx(h, rel=1e-12)


def test_inference_ar1():
    rng = np.random.default_rng(42)
    n = 20
    rho_true, tau_true = 0.7, 4.0
    x = _sample_gmrf(inla.ar1_precision_matrix_csc(n, rho_true, tau_true), rng)
    y = x + rng.normal(0.0, 0.2, n)
    obs_prec = 1.0 / (0.2**2)

    res = inla(
        "y ~ -1 + f(idx, model='ar1')",
        data={"y": y, "idx": np.arange(n)},
        family="gaussian",
        control_family={"hyper": {"prec": {"initial": math.log(obs_prec), "fixed": True}}},
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
    rng = np.random.default_rng(123)
    n = 30
    h_true = 0.7
    x = _sample_gmrf(inla.fgn_precision_matrix(n, h_true, 1.0), rng)
    noise_sd = 0.0316
    y = x + rng.normal(0.0, noise_sd, n)

    res = inla(
        "y ~ -1 + f(idx, model='fgn')",
        data={"y": y, "idx": np.arange(n)},
        family="gaussian",
        control_family={"hyper": {"prec": {"initial": math.log(1000.0), "fixed": True}}},
    )

    est_h = inla.fgn_hurst_from_intern(res.mode[1])
    assert 0.5 < est_h < 0.9


def test_inference_fgn_approx():
    rng = np.random.default_rng(123)
    n = 30
    order = 4
    h_true = 0.7
    x = _sample_gmrf(inla.fgn_precision_matrix(n, h_true, 1.0), rng)
    obs_prec = math.exp(8.0)
    y = x + rng.normal(0.0, 1.0 / math.sqrt(obs_prec), n)
    n_latent = inla.fgn_approx_latent_len(n, order)

    res = inla(
        f"y ~ -1 + f(idx, model='fgn', order={order})",
        data={"y": y, "idx": np.arange(n)},
        family="gaussian",
        control_family={"hyper": {"prec": {"initial": math.log(obs_prec), "fixed": True}}},
        initial_theta=[1.0, inla.fgn_intern_from_hurst(h_true)],
    )

    est_h = inla.fgn_hurst_from_intern(res.mode[1])
    assert 0.5 < est_h < 0.9
    assert len(res.latent_means) == n_latent


def test_inference_rw2():
    rng = np.random.default_rng(42)
    n = 20
    t = np.linspace(1.0 / n, 1.0, n)
    y = t**2 + rng.normal(0.0, 0.05, n)

    res = inla(
        "y ~ -1 + f(idx, model='rw2')",
        data={"y": y, "idx": np.arange(n)},
        family="gaussian",
        control_family={"hyper": {"prec": {"initial": math.log(100.0), "fixed": True}}},
    )
    assert len(res.mode) == 1
    assert res.mode[0] > 0.0


def test_non_gaussian_families():
    counts = np.array([2, 3, 2, 4, 3, 2, 3, 2], dtype=float)
    n = len(counts)
    res_pois = inla(
        "y ~ -1 + f(idx, model='iid')",
        data={"y": counts, "idx": np.arange(n)},
        family="poisson",
        E=1.0,
        initial_theta=[1.0],
    )
    assert len(res_pois.mode) == 1
    assert res_pois.marginal_log_lik < 0.0

    ys_b = np.array([2, 5, 3, 7, 4, 6], dtype=float)
    n_b = len(ys_b)
    res_bin = inla(
        "y ~ -1 + f(idx, model='iid')",
        data={"y": ys_b, "idx": np.arange(n_b)},
        family="binomial",
        Ntrials=10.0,
    )
    assert len(res_bin.mode) == 1

    y_lap = np.array([0.2, -0.1, 0.4, 0.0, -0.3, 0.1, 0.2, -0.2])
    n_l = len(y_lap)
    res_lap = inla(
        "y ~ -1 + f(idx, model='iid')",
        data={"y": y_lap, "idx": np.arange(n_l)},
        family="laplace",
        alpha=0.5,
        gamma=0.2,
        initial_theta=[1.0],
    )
    assert len(res_lap.mode) == 1


def test_family_aliases():
    n = 4
    y = np.ones(n)
    res = inla(
        "y ~ -1 + f(idx, model='iid')",
        data={"y": y, "idx": np.arange(n)},
        family="nbinomial",
        E=1.0,
        size=5.0,
    )
    assert len(res.mode) == 1
    assert math.isfinite(res.marginal_log_lik)


def test_run_inla_inference_not_public():
    assert not hasattr(inla, "run_inla_inference")
    assert "run_inla_inference" not in inla.__all__
