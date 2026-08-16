"""SPDE / Matérn 2D structural path: Q(κ,τ) + projector A."""

from __future__ import annotations

import numpy as np

from inla import _native as core


def test_matern2d_precision_shape():
    q = core.matern2d_precision_matrix(4, 4, nu=1, range=2.0, prec=1.0)
    assert q.shape == (16, 16)
    assert q.to_scipy().nnz > 0


def test_spde_projector_and_precision():
    vertices = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)]
    triangles = [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]
    loc_x = [0.25, 0.75, 0.5]
    loc_y = [0.25, 0.25, 0.5]

    a = core.spde_projector_matrix(vertices, triangles, loc_x, loc_y)
    assert a.shape == (3, 5)
    assert a.to_scipy().nnz >= 3

    q = core.spde_precision_matrix(vertices, triangles, kappa=1.0, tau=1.0)
    assert q.shape == (5, 5)

    fem = core.fem_blocks_mesh(vertices, triangles)
    assert fem["c0"].shape == (5, 5)
    assert fem["g1"].shape == (5, 5)
    assert fem["n_vertices"] == 5
    assert fem["n_triangles"] == 4

    # Vertex observation → unit row of A
    a0 = core.spde_projector_matrix(vertices, triangles, [0.0], [0.0])
    row = np.asarray(a0.to_scipy().toarray()).ravel()
    assert abs(row[0] - 1.0) < 1e-10


def test_spde_gaussian_inference():
    vertices = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)]
    triangles = [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]
    loc_x = np.array([0.25, 0.75, 0.75, 0.25, 0.5, 0.4])
    loc_y = np.array([0.25, 0.25, 0.75, 0.75, 0.5, 0.6])
    y = 0.4 * np.sin(loc_x * 2) + 0.3 * np.cos(loc_y * 1.5)

    a = core.spde_projector_matrix(vertices, triangles, loc_x.tolist(), loc_y.tolist())

    def build_prior(theta):
        tau = float(np.exp(theta[0]))
        kappa = float(np.exp(theta[1]))
        return core.spde_precision_matrix(vertices, triangles, kappa=kappa, tau=tau)

    def log_prior(theta):
        return float(-0.05 * (theta[0] ** 2 + theta[1] ** 2))

    obs = [{"family": "gaussian", "y": float(yi), "precision": 50.0} for yi in y]
    res = core.run_inla_inference(
        [0.0, 0.0],
        build_prior,
        log_prior,
        obs,
        strategy="ccd",
        a=a,
    )
    assert len(res.mode) == 2
    assert len(res.latent_means) == 5
    assert np.isfinite(res.marginal_log_lik)
