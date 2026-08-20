"""R-parity high-level ``inla()`` formula API."""

from __future__ import annotations

import numpy as np
from scipy import sparse

import inla


def test_parse_formula_besag():
    p = inla.parse_formula("successes ~ covariate_x + f(spatial_idx, model='besag')")
    assert p.response == "successes"
    assert p.intercept is True
    assert p.fixed_terms == ["covariate_x"]
    assert len(p.f_terms) == 1
    assert p.f_terms[0].index == "spatial_idx"
    assert p.f_terms[0].model == "besag"


def test_parse_formula_assignment_arrow():
    p = inla.parse_formula("successes <- covariate_x + f(spatial_idx, model='besag')")
    assert p.response == "successes"
    assert p.fixed_terms == ["covariate_x"]
    assert p.f_terms[0].model == "besag"


def test_inla_binomial_besag_formula():
    rng = np.random.default_rng(0)
    n_areas = 54
    successes = rng.binomial(n=10, p=0.4, size=n_areas).astype(float)
    total_trials = np.full(n_areas, 10.0)
    covariate_x = rng.normal(0, 1, size=n_areas)
    spatial_idx = np.arange(n_areas)

    adj = np.zeros((n_areas, n_areas))
    for i in range(n_areas - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0

    data = {
        "successes": successes,
        "covariate_x": covariate_x,
        "spatial_idx": spatial_idx,
        "adj_matrix": adj,
    }

    result = inla(
        formula="successes <- covariate_x + f(spatial_idx, model='besag')",
        family="cbinomial",
        data=data,
        Ntrials=np.column_stack([successes, total_trials]),
        verbose=False,
    )

    assert np.isfinite(result.marginal_log_lik)
    assert np.isfinite(result.dic)
    assert len(result.latent_means) == 56
    assert result.summary_fixed is not None
    assert "spatial_idx" in result.summary_random
    intercept_mean = float(result.latent_means[0])
    assert np.isfinite(intercept_mean)
    assert np.isclose(intercept_mean, float(result.summary_fixed["mean"][0]))
    # Hard sum-to-zero on Besag block (after intercept + covariate).
    spatial = np.asarray(result.summary_random["spatial_idx"]["mean"], dtype=float)
    assert abs(spatial.sum()) < 1e-3


def test_control_compute_latent_marginals():
    rng = np.random.default_rng(1)
    n = 12
    y = rng.normal(0, 1, n)
    data = {"y": y, "idx": np.arange(n)}
    fit = inla(
        "y ~ f(idx, model='rw1')",
        data=data,
        deterministic=True,
        control_compute={"return_marginals_latent": [0, 1]},
    )
    assert len(fit.marginals_latent) == 2
    assert list(fit.marginals_latent_indices) == [0, 1]
    rw = np.asarray(fit.summary_random["idx"]["mean"], dtype=float)
    assert abs(rw.sum()) < 1e-3


def test_inla_generic_define_iid():
    rng = np.random.default_rng(0)
    n = 20
    y = rng.normal(0, 1, n)
    data = {"y": y, "idx": np.arange(n)}

    model = inla.generic.define(
        n=n,
        n_theta=1,
        initial=[0.0],
        Q=lambda th: sparse.eye(n, format="csc") * float(np.exp(th[0])),
        name="myiid",
    )

    r1 = inla(
        "y ~ -1 + f(idx, model='rgeneric')",
        data=data,
        family="gaussian",
        rgeneric=model,
        control_family={"hyper": {"prec": {"initial": np.log(4.0)}}},
    )
    assert np.isfinite(r1.marginal_log_lik)
    assert len(r1.latent_means) == n

    r2 = inla(
        "y ~ -1 + f(idx, model='myiid')",
        data=data,
        family="gaussian",
        models={"myiid": model},
        control_family={"hyper": {"prec": {"initial": np.log(4.0)}}},
    )
    assert np.isfinite(r2.marginal_log_lik)


def test_inla_generic_model_subclass():
    class MyIID(inla.generic.Model):
        def __init__(self, n):
            super().__init__(n=n, n_theta=1, initial=[1.0], name="cls_iid")

        def Q(self, theta):
            return sparse.eye(self.n, format="csc") * float(np.exp(theta[0]))

    rng = np.random.default_rng(1)
    n = 15
    y = rng.normal(0, 0.5, n)
    m = MyIID(n)
    r = inla(
        "y <- -1 + f(idx, model='cls_iid')",
        data={"y": y, "idx": np.arange(n)},
        family="gaussian",
        models={"cls_iid": m},
        control_family={"hyper": {"prec": {"initial": np.log(10.0)}}},
    )
    assert np.isfinite(r.marginal_log_lik)
    assert "idx" in r.summary_random


def test_parse_formula_copy():
    p = inla.parse_formula("y ~ -1 + f(i, model='iid') + f(j, copy='i')")
    assert p.f_terms[0].model == "iid"
    assert p.f_terms[1].model == "copy"
    assert p.f_terms[1].kwargs["copy"] == "i"


def test_copy_shared_latent():
    rng = np.random.default_rng(4)
    n = 12
    u = 0.4 * (np.arange(n) - 5.5)
    y = u + rng.normal(0, 0.05, n)
    fit = inla(
        "y ~ -1 + f(i, model='iid') + f(j, copy='i')",
        data={"y": y, "i": np.arange(n), "j": np.arange(n)},
        family="gaussian",
        deterministic=True,
        control_family={"hyper": {"prec": {"initial": np.log(400.0), "fixed": True}}},
        control_compute={"return_marginals_latent": [0]},
    )
    assert np.isfinite(fit.marginal_log_lik)
    assert len(fit.mode) == 2
    assert len(fit.latent_means) == 2 * n
    lc = fit.lincomb([("u0", [(0, 1.0)])])
    assert abs(lc[0]["mean"] - fit.latent_means[0]) < 1e-10
    assert lc[0]["sd"] > 0

    draws = fit.posterior_sample(16, seed=3)
    assert draws.shape == (16, 2 * n)
    assert np.isfinite(draws).all()

    m = fit.marginals_latent[0]
    g = [xi * xi for xi in m.x]
    e = m.emarginal(g)
    assert np.isfinite(e)
