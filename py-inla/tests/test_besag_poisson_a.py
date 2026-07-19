"""Besag/ICAR + Poisson + fixed effects via high-level ``inla()``."""

from __future__ import annotations

import numpy as np
import pytest
import inla


def _connected_erdos_renyi(n: int, p: float, seed: int):
    """Build a connected undirected ER graph without requiring networkx."""
    rng = np.random.default_rng(seed)
    for _attempt in range(200):
        adj = rng.random((n, n)) < p
        adj = np.triu(adj, 1)
        adj = adj | adj.T
        for i in range(n - 1):
            adj[i, i + 1] = True
            adj[i + 1, i] = True
        np.fill_diagonal(adj, False)
        seen = {0}
        stack = [0]
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(adj[u]):
                if int(v) not in seen:
                    seen.add(int(v))
                    stack.append(int(v))
        if len(seen) == n:
            return adj.astype(float)
        p = min(0.95, p + 0.05)
    raise RuntimeError("failed to build connected graph")


def test_besag_poisson_fixed_effects():
    n_regions = 54
    adj = _connected_erdos_renyi(n_regions, 0.15, seed=42)

    rng = np.random.default_rng(42)
    observed_y = rng.poisson(lam=12, size=n_regions).astype(float)
    expected_e = rng.uniform(5.0, 15.0, size=n_regions)
    covariate_x = rng.normal(0.0, 1.0, size=n_regions)

    data = {
        "y": observed_y,
        "x": covariate_x,
        "region": np.arange(n_regions),
        "adj_matrix": adj,
    }

    result = inla(
        "y ~ x + f(region, model='besag')",
        data=data,
        family="poisson",
        E=expected_e,
        latent_marginal_indices=[0, 1, 2, 10, 20],
    )

    assert len(result.latent_means) == 2 + n_regions
    assert len(result.predictor_means) == n_regions
    assert len(result.mode) == 1
    assert np.isfinite(result.marginal_log_lik)
    assert np.isfinite(result.dic)
    assert len(result.marginals_latent) == 5

    u = np.asarray(result.summary_random["region"]["mean"])
    assert np.std(u) > 0.0

    beta0, beta1 = float(result.latent_means[0]), float(result.latent_means[1])
    assert np.isfinite(beta0) and np.isfinite(beta1)

    tau_hat = float(np.exp(result.mode[0]))
    assert tau_hat > 0.0
