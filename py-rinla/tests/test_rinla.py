import numpy as np
import scipy.sparse as sp
import pytest
import rinla

def test_scipy_conversion():
    # Construct AR1 precision matrix triplets
    triplets = rinla.ar1_precision_matrix(5, 0.7, 1.0)
    assert len(triplets) == 3
    
    # Construct PyCscMatrix wrapper
    mat = rinla.PyCscMatrix(5, 5, triplets[0], triplets[1], triplets[2])
    assert mat.shape == (5, 5)
    
    # Convert to scipy.sparse.csc_matrix
    sp_mat = mat.to_scipy()
    assert sp.isspmatrix_csc(sp_mat)
    assert sp_mat.shape == (5, 5)
    
    # Ensure memory safety attributes are attached
    assert hasattr(sp_mat, "_base_matrix")
    
    # Verify CSC values
    dense = sp_mat.toarray()
    assert np.allclose(dense[0, 0], 1.0) # Innovation scale tau=1.0 at boundary
    assert np.allclose(dense[0, 1], -0.7) # Off-diagonal: -tau * rho
    
    # Check directly built CSC wrapper
    csc_mat = rinla.ar1_precision_matrix_csc(5, 0.7, 1.0)
    sp_csc = csc_mat.to_scipy()
    assert sp_csc.shape == (5, 5)

def test_fgn_matrices():
    # Exact dense matrix
    q_fgn = rinla.fgn_precision_matrix(5, 0.7, 1.5)
    assert q_fgn.shape == (5, 5)
    sp_fgn = q_fgn.to_scipy()
    assert sp_fgn.nnz == 25 # dense matrix has 25 elements
    
    # Approx sparse FGN matrix (order=4)
    n = 5
    q_approx = rinla.fgn_approx_precision_matrix(n, 0.7, 1.0, order=4, prec_eps=1e8)
    # Latent size for approx FGN is (order + 1) * n = 5 * 5 = 25
    assert q_approx.shape == (25, 25)

def test_inference_ar1():
    np.random.seed(42)
    n = 20
    x = np.zeros(n)
    x[0] = np.random.normal(0, 1.0)
    for i in range(1, n):
        x[i] = 0.7 * x[i-1] + np.random.normal(0, 0.5)
    # y = x + noise, noise sd = 0.2
    y = x + np.random.normal(0, 0.2, n)
    
    # Obs precision = 1 / 0.2^2 = 25.0
    obs = [{"family": "gaussian", "y": float(y[i]), "precision": 25.0} for i in range(n)]
    
    def build_prior(theta):
        tau = np.exp(theta[0])
        rho = 2.0 / (1.0 + np.exp(-theta[1])) - 1.0
        return rinla.ar1_precision_matrix_csc(n, rho, tau)
        
    def log_prior_density(theta):
        return -0.5 * 0.1 * (theta[0]**2 + theta[1]**2)
        
    res = rinla.run_inla_inference(
        initial_theta=[0.0, 0.0],
        build_prior=build_prior,
        log_prior_density=log_prior_density,
        obs=obs,
        strategy="ccd"
    )
    
    assert len(res.mode) == 2
    # Verify estimated params are in correct statistical range
    # log_tau (marginal precision) should be positive/reasonable
    assert 0.0 < res.mode[0] < 2.0
    # logit_rho (rho ~ 0.7 -> logit(rho) ~ 1.7)
    assert 0.5 < res.mode[1] < 3.0
    assert res.marginal_log_lik < 0.0

def test_inference_fgn():
    np.random.seed(123)
    n = 30
    H_true = 0.7
    
    def fgn_autocov(k, H):
        return 0.5 * (abs(k + 1)**(2 * H) - 2 * abs(k)**(2 * H) + abs(k - 1)**(2 * H))
        
    Sigma = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Sigma[i, j] = fgn_autocov(i - j, H_true)
    Sigma += np.eye(n) * 1e-9
    L = np.linalg.cholesky(Sigma)
    x = L @ np.random.normal(size=n)
    y = x + np.random.normal(0, 0.0316, n) # noise prec = 1000
    
    obs = [{"family": "gaussian", "y": float(y[i]), "precision": 1000.0} for i in range(n)]
    
    def build_prior(theta):
        tau = np.exp(theta[0])
        hurst = 1.0 / (1.0 + np.exp(-theta[1]))
        return rinla.fgn_precision_matrix(n, hurst, tau)
        
    def log_prior_density(theta):
        return -0.5 * 0.1 * (theta[0]**2 + theta[1]**2)
        
    res = rinla.run_inla_inference(
        initial_theta=[0.0, 0.0],
        build_prior=build_prior,
        log_prior_density=log_prior_density,
        obs=obs,
        strategy="ccd"
    )
    
    est_H = 1.0 / (1.0 + np.exp(-res.mode[1]))
    assert 0.5 < est_H < 0.9

def test_inference_fgn_approx():
    np.random.seed(123)
    n = 30
    H_true = 0.7
    
    def fgn_autocov(k, H):
        return 0.5 * (abs(k + 1)**(2 * H) - 2 * abs(k)**(2 * H) + abs(k - 1)**(2 * H))
        
    Sigma = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Sigma[i, j] = fgn_autocov(i - j, H_true)
    Sigma += np.eye(n) * 1e-9
    L = np.linalg.cholesky(Sigma)
    x = L @ np.random.normal(size=n)
    y = x + np.random.normal(0, np.exp(-4.0), n) # fixed precision parameter exp(8.0)
    
    order = 4
    n_latent = (order + 1) * n
    # For approx FGN, only the first n variables are observed
    obs = [{"family": "gaussian", "y": float(y[i]), "precision": float(np.exp(8.0))} for i in range(n)]
    obs += [None] * (n_latent - n)
    
    def build_prior(theta):
        tau = np.exp(theta[0])
        hurst = 0.5 + 0.5 / (1.0 + np.exp(-theta[1]))
        return rinla.fgn_approx_precision_matrix(n, hurst, tau, order=4, prec_eps=1e8)
        
    def log_prior_density(theta):
        return -0.5 * 0.1 * (theta[0]**2 + theta[1]**2)
        
    res = rinla.run_inla_inference(
        initial_theta=[1.0, 2.0],
        build_prior=build_prior,
        log_prior_density=log_prior_density,
        obs=obs,
        strategy="ccd"
    )
    
    est_H = 0.5 + 0.5 / (1.0 + np.exp(-res.mode[1]))
    assert 0.5 < est_H < 0.9

def test_inference_rw2():
    np.random.seed(42)
    n = 20
    t = np.arange(1, n + 1) / n
    y = t**2 + np.random.normal(0, 0.05, n)
    
    obs = [{"family": "gaussian", "y": float(y[i]), "precision": 100.0} for i in range(n)]
    
    def build_prior(theta):
        tau = np.exp(theta[0])
        return rinla.rw2_precision_matrix(n, tau)
        
    def log_prior_density(theta):
        return -0.5 * 0.1 * theta[0]**2
        
    res = rinla.run_inla_inference(
        initial_theta=[0.0],
        build_prior=build_prior,
        log_prior_density=log_prior_density,
        obs=obs,
        strategy="ccd"
    )
    assert len(res.mode) == 1
    assert res.mode[0] > 0.0

def test_non_gaussian_families():
    # 1. Poisson with IID latent model
    np.random.seed(3)
    counts = [2, 3, 2, 4, 3, 2, 3, 2]
    n = len(counts)
    obs_pois = [{"family": "poisson", "y": float(counts[i]), "exposure": 1.0} for i in range(n)]
    
    def build_prior_iid(theta):
        tau = np.exp(theta[0])
        return rinla.iid_precision_matrix(n, tau)
        
    def log_prior_iid(theta):
        return -0.5 * 0.1 * theta[0]**2
        
    res_pois = rinla.run_inla_inference(
        initial_theta=[1.0],
        build_prior=build_prior_iid,
        log_prior_density=log_prior_iid,
        obs=obs_pois,
        strategy="ccd"
    )
    assert len(res_pois.mode) == 1
    assert res_pois.marginal_log_lik < 0.0
    
    # 2. Binomial
    ys_b = [2, 5, 3, 7, 4, 6]
    n_b = len(ys_b)
    obs_bin = [{"family": "binomial", "y": float(ys_b[i]), "n": 10.0} for i in range(n_b)]
    
    def build_prior_bin(theta):
        tau = np.exp(theta[0])
        return rinla.iid_precision_matrix(n_b, tau)
        
    res_bin = rinla.run_inla_inference(
        initial_theta=[0.0],
        build_prior=build_prior_bin,
        log_prior_density=log_prior_iid,
        obs=obs_bin,
        strategy="ccd"
    )
    assert len(res_bin.mode) == 1
    
    # 3. Laplace
    y_lap = [0.2, -0.1, 0.4, 0.0, -0.3, 0.1, 0.2, -0.2]
    n_l = len(y_lap)
    obs_lap = [{"family": "laplace", "y": float(y_lap[i]), "alpha": 0.5, "gamma": 0.2} for i in range(n_l)]
    
    def build_prior_lap(theta):
        tau = np.exp(theta[0])
        return rinla.iid_precision_matrix(n_l, tau)
        
    res_lap = rinla.run_inla_inference(
        initial_theta=[1.0],
        build_prior=build_prior_lap,
        log_prior_density=log_prior_iid,
        obs=obs_lap,
        strategy="ccd"
    )
    assert len(res_lap.mode) == 1
