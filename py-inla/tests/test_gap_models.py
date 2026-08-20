"""Formula wiring for BYM/BYM2, matern2d, SPDE, and CRW2 layouts."""

from __future__ import annotations

import numpy as np

import inla
from inla import _native as core
from inla.spde import lattice_mesh


def _cycle_adj(n: int) -> list[list[int]]:
    return [[(i - 1) % n, (i + 1) % n] for i in range(n)]


def test_bym_formula():
    n = 6
    adj = _cycle_adj(n)
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 0.5, size=n)
    data = {"y": y, "region": np.arange(n), "adj_list": adj}
    res = inla(
        "y ~ -1 + f(region, model='bym')",
        data=data,
        family="gaussian",
    )
    assert len(res.mode) == 3
    assert len(res.latent_means) == 2 * n
    assert np.isfinite(res.marginal_log_lik)


def test_bym2_formula():
    n = 6
    adj = _cycle_adj(n)
    rng = np.random.default_rng(1)
    y = rng.normal(0.0, 0.5, size=n)
    data = {"y": y, "region": np.arange(n), "adj_list": adj}
    res = inla(
        "y ~ -1 + f(region, model='bym2')",
        data=data,
        family="gaussian",
    )
    assert len(res.mode) == 3
    assert len(res.latent_means) == n
    assert np.isfinite(res.marginal_log_lik)


def test_matern2d_formula():
    nrow, ncol = 4, 4
    n = nrow * ncol
    rng = np.random.default_rng(2)
    idx = np.arange(n)
    y = 0.2 * np.sin(idx * 0.3) + rng.normal(0, 0.2, size=n)
    data = {"y": y, "idx": idx}
    res = inla(
        "y ~ -1 + f(idx, model='matern2d', nrow=4, ncol=4, nu=1, cyclic=False)",
        data=data,
        family="gaussian",
    )
    assert len(res.mode) == 3
    assert len(res.latent_means) == n
    assert np.isfinite(res.marginal_log_lik)


def test_spde_formula():
    mesh = lattice_mesh(xlim=(0, 1), ylim=(0, 1), nx=5, ny=5)
    verts = mesh["vertices"]
    tris = mesh["triangles"]
    rng = np.random.default_rng(3)
    n_obs = 20
    loc = rng.uniform(0.05, 0.95, size=(n_obs, 2))
    y = 0.3 * np.sin(loc[:, 0] * 3) + rng.normal(0, 0.15, size=n_obs)
    data = {
        "y": y,
        "field": np.zeros(n_obs),  # unused index placeholder
        "loc_x": loc[:, 0],
        "loc_y": loc[:, 1],
        "vertices": verts,
        "triangles": tris,
    }
    res = inla(
        "y ~ -1 + f(field, model='spde', vertices='vertices', triangles='triangles', loc_x='loc_x', loc_y='loc_y')",
        data=data,
        family="gaussian",
    )
    assert len(res.mode) == 3
    assert len(res.latent_means) == verts.shape[0]
    assert np.isfinite(res.marginal_log_lik)


def test_crw2_pairs_formula():
    pos = np.array([0.0, 1.0, 2.0, 3.5, 5.0])
    n = len(pos)
    rng = np.random.default_rng(4)
    y = np.sin(pos * 0.4) + rng.normal(0, 0.15, size=n)
    data = {"y": y, "t": np.arange(n), "positions": pos}
    res = inla(
        "y ~ -1 + f(t, model='crw2', positions='positions', layout='pairs')",
        data=data,
        family="gaussian",
    )
    assert len(res.latent_means) == 2 * n
    assert np.isfinite(res.marginal_log_lik)


def test_bym_precision_bindings():
    adj = _cycle_adj(4)
    q = core.bym_precision_matrix(adj, 1.0, 2.0)
    assert q.shape == (8, 8)
    q2 = core.bym2_precision_matrix(adj, 1.5, 0.4)
    assert q2.shape == (4, 4)
