"""Pytest suite for rw2d and group spatio-temporal models in py-inla."""

import numpy as np
import pytest

import inla
from inla import _native as core


def test_rw2d_formula():
    nrow, ncol = 4, 4
    n = nrow * ncol
    idx = np.arange(n)
    data = {
        "y": 0.2 * (idx % nrow) + 0.1 * (idx // nrow) + np.random.normal(0, 0.1, n),
        "idx": idx,
    }
    res = inla.fit("y ~ f(idx, model='rw2d', nrow=4, ncol=4, cyclic=False)", data=data)
    assert res.marginal_log_lik is not None
    assert np.isfinite(res.marginal_log_lik)
    assert len(res.latent_means) >= n


def test_kronecker_csc_native():
    a = core.iid_precision_matrix(2, 1.0)
    b = core.iid_precision_matrix(3, 2.0)
    kron = core.kronecker_csc(a, b)
    assert kron.shape == (6, 6)


def test_group_formula():
    n_space = 3
    n_time = 4
    n = n_space * n_time
    space_idx = np.tile(np.arange(n_space), n_time)
    time_idx = np.repeat(np.arange(n_time), n_space)
    data = {
        "y": 0.5 * space_idx + 0.3 * time_idx + np.random.normal(0, 0.1, n),
        "s": space_idx,
        "t": time_idx,
    }
    res = inla.fit(
        "y ~ f(s, model='iid', group=t, control_group=dict(model='ar1'))",
        data=data,
    )
    assert res.marginal_log_lik is not None
    assert np.isfinite(res.marginal_log_lik)
    assert len(res.latent_means) >= n
