"""Tests for typed effect components, likelihood families, and declarative ModelSpec API."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

import inla
from inla import (
    AR1,
    Besag,
    Binomial,
    Gaussian,
    IID,
    Intercept,
    Linear,
    ModelSpec,
    Poisson,
    RW1,
    RW2,
)


def _make_grid_adj(n_side: int = 3) -> np.ndarray:
    """Simple 2D grid adjacency matrix (n_side x n_side)."""
    n = n_side * n_side
    adj = np.zeros((n, n), dtype=int)
    for r in range(n_side):
        for c in range(n_side):
            i = r * n_side + c
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n_side and 0 <= nc < n_side:
                    j = nr * n_side + nc
                    adj[i, j] = 1
    return adj


def test_functional_and_modelspec_equivalence_gaussian_ar1():
    """Verify that formula, functional kwargs, and ModelSpec yield identical results."""
    np.random.seed(42)
    n = 20
    t = np.arange(n, dtype=int)
    x = np.random.randn(n)
    y = 0.5 + 1.2 * x + np.sin(t / 2.0) + np.random.normal(0, 0.2, size=n)

    data = {"y": y, "x": x, "t": t}

    # 1. Formula string
    res_formula = inla(
        "y ~ x + f(t, model='ar1')",
        data=data,
        family="gaussian",
        deterministic=True,
    )

    # 2. Functional kwargs
    res_functional = inla.fit(
        data=data,
        response="y",
        fixed=["x"],
        random=[AR1("t")],
        family=Gaussian(),
        deterministic=True,
    )

    # 3. ModelSpec class
    class TimeSeriesModel(ModelSpec):
        response = "y"
        family = Gaussian()
        fixed = [Linear("x")]
        temporal = AR1("t")

    res_spec_cls = inla.fit(TimeSeriesModel, data=data, deterministic=True)

    # 4. ModelSpec instance
    spec_inst = TimeSeriesModel()
    res_spec_inst = inla.fit(spec_inst, data=data, deterministic=True)

    # Check identical latent means and marginal log-likelihood
    np.testing.assert_allclose(
        res_formula.latent_means, res_functional.latent_means, rtol=1e-5
    )
    np.testing.assert_allclose(
        res_formula.latent_means, res_spec_cls.latent_means, rtol=1e-5
    )
    np.testing.assert_allclose(
        res_formula.latent_means, res_spec_inst.latent_means, rtol=1e-5
    )

    assert pytest.approx(res_formula.marginal_log_lik, rel=1e-4) == res_functional.marginal_log_lik
    assert pytest.approx(res_formula.marginal_log_lik, rel=1e-4) == res_spec_cls.marginal_log_lik


def test_binomial_besag_with_direct_graph_objects():
    """Test Binomial Besag model with numpy and scipy sparse graph objects."""
    n_side = 3
    n = n_side * n_side
    adj = _make_grid_adj(n_side)
    adj_sparse = sparse.csr_matrix(adj)

    np.random.seed(123)
    spatial_idx = np.arange(n, dtype=int)
    trials = np.full(n, 20, dtype=int)
    successes = np.random.binomial(n=trials, p=0.4)
    x = np.random.randn(n)

    data = {
        "y": successes,
        "n_trials": trials,
        "x": x,
        "region": spatial_idx,
    }

    # Pass dense numpy adjacency directly into Besag
    res_dense = inla.fit(
        data=data,
        response="y",
        fixed=["x"],
        random=[Besag("region", graph=adj, scale_model=True)],
        family=Binomial(Ntrials="n_trials"),
        deterministic=True,
    )

    # Pass scipy sparse adjacency directly into Besag
    res_sparse = inla.fit(
        data=data,
        response="y",
        fixed=["x"],
        random=[Besag("region", graph=adj_sparse, scale_model=True)],
        family=Binomial(Ntrials="n_trials"),
        deterministic=True,
    )

    np.testing.assert_allclose(res_dense.latent_means, res_sparse.latent_means, rtol=1e-6)
    assert res_dense.summary_random is not None
    assert "region" in res_dense.summary_random


def test_modelspec_inheritance_and_overrides():
    """Test subclassing ModelSpec and overriding fields in instances."""
    n = 15
    data = {
        "y": np.random.randn(n),
        "x1": np.random.randn(n),
        "x2": np.random.randn(n),
        "idx": np.arange(n, dtype=int),
    }

    class BaseModel(ModelSpec):
        response = "y"
        family = "gaussian"
        fixed = ["x1"]
        rw = RW1("idx")

    class ExtendedModel(BaseModel):
        fixed = ["x1", "x2"]

    res_base = inla.fit(BaseModel, data=data, deterministic=True)
    res_ext = inla.fit(ExtendedModel, data=data, deterministic=True)

    assert "x1" in res_base.summary_fixed["names"]
    assert "x2" not in res_base.summary_fixed["names"]
    assert "x1" in res_ext.summary_fixed["names"]
    assert "x2" in res_ext.summary_fixed["names"]
    assert len(res_base.summary_fixed["mean"]) == 2  # (Intercept), x1
    assert len(res_ext.summary_fixed["mean"]) == 3   # (Intercept), x1, x2


def test_modelspec_without_intercept():
    """Test intercept=False / Intercept(enabled=False)."""
    n = 20
    data = {
        "y": np.random.randn(n) + 5.0,
        "x": np.ones(n),
        "idx": np.arange(n, dtype=int),
    }

    class NoInterceptModel(ModelSpec):
        response = "y"
        intercept = False
        fixed = ["x"]
        spatial = IID("idx")

    res = inla.fit(NoInterceptModel, data=data, deterministic=True)
    assert "(Intercept)" not in res.summary_fixed["names"]
    assert "x" in res.summary_fixed["names"]
    assert len(res.summary_fixed["mean"]) == 1


def test_poisson_family_with_exposure():
    """Test Poisson family with E parameter."""
    n = 20
    data = {
        "count": np.random.poisson(lam=5, size=n),
        "e": np.ones(n) * 2.0,
        "idx": np.arange(n, dtype=int),
    }

    res = inla.fit(
        data=data,
        response="count",
        family=Poisson(E="e"),
        random=[IID("idx")],
        deterministic=True,
    )
    assert res.latent_means is not None
    assert len(res.latent_means) > 0
