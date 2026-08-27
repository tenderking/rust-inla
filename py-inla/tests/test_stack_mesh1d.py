"""1D SPDE helpers, inla.Stack, and multi-likelihood fits."""

from __future__ import annotations

import numpy as np
from scipy import sparse

import inla
from inla import Gaussian, Poisson, Stack
from inla.spde import make_A, matern, mesh_1d


def test_mesh_1d_projector_and_precision():
    mesh = mesh_1d(np.linspace(0.0, 1.0, 5))
    a = make_A(mesh, loc=np.array([0.0, 0.25, 1.0]))
    scipy_a = a.to_scipy().tocsr()
    assert scipy_a.shape == (3, 5)
    assert abs(scipy_a[0, 0] - 1.0) < 1e-12
    q = inla.spde.precision_matrix(loc=mesh["loc"], kappa=1.0, tau=1.0)
    qq = q.to_scipy() if hasattr(q, "to_scipy") else q
    assert sparse.csc_matrix(qq).shape == (5, 5)


def test_stack_join_and_index():
    a_est = sparse.eye(4, 6, format="csc")
    a_pred = sparse.csr_matrix(np.ones((2, 6)))
    est = Stack(
        data={"y": np.arange(4, dtype=float)},
        A=[1, a_est],
        effects=[{"Intercept": 1}, {"x": np.arange(6)}],
        tag="est",
    )
    pred = Stack(
        data={"y": np.full(2, np.nan)},
        A=[1, a_pred],
        effects=[{"Intercept": 1}, {"x": np.arange(6)}],
        tag="pred",
    )
    joint = Stack.join(est, pred)
    assert joint.A.shape == (6, 7)
    assert joint.index("est").tolist() == [0, 1, 2, 3]
    assert joint.index("pred").tolist() == [4, 5]
    assert np.isnan(joint.data["y"][4])


def test_lidar_style_1d_spde_smoothing():
    rng = np.random.default_rng(0)
    x = np.linspace(390.0, 720.0, 40)
    y = 0.2 * np.sin((x - 390.0) / 50.0) + rng.normal(0.0, 0.05, size=x.size)
    mesh = mesh_1d(np.linspace(390.0, 720.0, 12))
    data = {"y": y, "range": x, "field": np.zeros(x.size)}
    res = inla.fit(
        data=data,
        response="y",
        intercept=True,
        random=[inla.SPDE("field", spde_model=matern(mesh), loc="range")],
        family=Gaussian(
            control_family={"hyper": {"prec": {"initial": float(np.log(400.0)), "fixed": True}}}
        ),
    )
    assert np.isfinite(res.marginal_log_lik)
    assert len(res.latent_means) == 1 + mesh["n"]


def test_multi_likelihood_gaussian_poisson_na_pattern():
    rng = np.random.default_rng(1)
    n = 24
    eta = 0.3 + 0.4 * np.linspace(-1.0, 1.0, n)
    y_g = eta + rng.normal(0.0, 0.15, size=n)
    y_p = rng.poisson(np.exp(eta))
    y = np.column_stack([y_g, y_p.astype(float)])
    y[::2, 1] = np.nan
    y[1::2, 0] = np.nan
    data = {"y": y}
    res = inla.fit(
        data=data,
        response="y",
        intercept=True,
        family=[
            Gaussian(control_family={"hyper": {"prec": {"initial": np.log(25.0), "fixed": True}}}),
            Poisson(),
        ],
    )
    assert np.isfinite(res.marginal_log_lik)
    assert len(res.predictor_means) == n
