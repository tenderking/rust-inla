"""iid2d / iid3d correlated random effects (GitHub issue #19)."""

from __future__ import annotations

import numpy as np

import inla
from inla.formula import parse_formula


def test_parse_iid2d_weights_list():
    p = parse_formula("y ~ f(id, model='iid2d', weights=['1', 'time'])")
    assert p.f_terms[0].model == "iid2d"
    assert p.f_terms[0].kwargs["weights"] == ["1", "time"]


def test_parse_copy_positional_weights():
    p = parse_formula("y ~ f(i, model='iid2d', n=20) + f(j, z, copy='i')")
    assert p.f_terms[0].kwargs["n"] == 20
    assert p.f_terms[1].model == "copy"
    assert p.f_terms[1].kwargs["weights"] == "z"


def test_iid2d_stacked_bivariate():
    """Each component of the pair is its own observation row (iid.pdf example 1)."""
    rng = np.random.default_rng(19)
    m = 40
    tau_a, tau_b, rho = 1.0, 4.0, 0.5
    s11, s22 = 1.0 / tau_a, 1.0 / tau_b
    s12 = rho * np.sqrt(s11 * s22)
    cov = np.array([[s11, s12], [s12, s22]])
    pairs = rng.multivariate_normal(np.zeros(2), cov, size=m)
    y = np.concatenate([pairs[:, 0], pairs[:, 1]])
    idx = np.arange(1, 2 * m + 1)
    res = inla.fit(
        "y ~ -1 + f(idx, model='iid2d', n=80, initial=[0, 1.4, 1.1])",
        data={"y": y, "idx": idx},
        family="gaussian",
        control_family={"hyper": {"prec": {"initial": np.log(400.0), "fixed": True}}},
        deterministic=True,
    )
    assert np.isfinite(res.marginal_log_lik)
    assert len(res.mode) == 3
    assert len(res.latent_means) == 2 * m
    labels = list(res.summary_hyperpar["names"])
    assert any("Precision (component 1)" in s for s in labels)
    assert any("Rho1:2" in s for s in labels)
    # Recovered natural-scale correlation should stay in (-1, 1).
    rho_hat = float(res.summary_hyperpar["mean"][-1])
    assert -0.99 < rho_hat < 0.99


def test_iid2d_random_intercept_slope():
    """Chapter 4 style: subject-level intercept + slope on time via weights=."""
    rng = np.random.default_rng(7)
    n_subj = 25
    n_rep = 6
    t = np.tile(np.linspace(-1.0, 1.0, n_rep), n_subj)
    subj = np.repeat(np.arange(1, n_subj + 1), n_rep)
    b = rng.multivariate_normal([0.0, 0.0], [[0.4, 0.15], [0.15, 0.25]], size=n_subj)
    eta = b[subj - 1, 0] + b[subj - 1, 1] * t
    y = eta + rng.normal(0.0, 0.2, n_subj * n_rep)
    res = inla.fit(
        data={"y": y, "id": subj, "time": t},
        response="y",
        intercept=False,
        random=[inla.IID2D("id", weights=["1", "time"], initial=[1.0, 1.5, 0.5])],
        family=inla.Gaussian(),
        control_family={"hyper": {"prec": {"initial": np.log(25.0), "fixed": True}}},
        deterministic=True,
    )
    assert np.isfinite(res.marginal_log_lik)
    assert len(res.mode) == 3
    assert len(res.latent_means) == 2 * n_subj
    names = list(res.summary_hyperpar["names"])
    rho_i = [i for i, s in enumerate(names) if "Rho1:2" in s][0]
    rho_hat = float(res.summary_hyperpar["mean"][rho_i])
    assert rho_hat > 0.0


def test_iid3d_and_iid5d_registry():
    from inla import _native as core

    m3 = core.model_metadata("iid3d")
    assert m3["theta_len"] == 6
    assert m3["hyper_labels"][-1] == "Rho2:3"
    assert m3["default_priors"][0][0] == "wishart3d"
    m5 = core.model_metadata("iid5d")
    assert m5["theta_len"] == 15
    assert m5["default_priors"][0][0] == "wishart5d"
