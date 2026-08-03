"""Pytest suite for expanded latent models and family arms in py-inla."""

import numpy as np
import pytest

import inla


def test_rw1_formula():
    n = 15
    data = {
        "y": np.linspace(0, 1.5, n) + np.random.normal(0, 0.1, n),
        "t": np.arange(n),
    }
    res = inla.fit("y ~ f(t, model='rw1')", data=data)
    assert res.marginal_log_lik is not None
    assert np.isfinite(res.marginal_log_lik)
    assert len(res.latent_means) >= n


def test_seasonal_formula():
    n = 24
    t = np.arange(n)
    s_effect = (t % 4) * 0.5
    data = {
        "y": s_effect + np.random.normal(0, 0.1, n),
        "t": t,
    }
    res = inla.fit("y ~ f(t, model='seasonal', season=4)", data=data)
    assert res.marginal_log_lik is not None
    assert np.isfinite(res.marginal_log_lik)


def test_ar_formula():
    n = 20
    t = np.arange(n)
    data = {
        "y": np.sin(t * 0.3) + np.random.normal(0, 0.1, n),
        "t": t,
    }
    res = inla.fit("y ~ f(t, model='ar', order=2)", data=data)
    assert res.marginal_log_lik is not None
    assert np.isfinite(res.marginal_log_lik)
    assert len(res.mode) >= 3  # log_tau + 2 PACF hyperparameters


def test_crw1_formula():
    n = 10
    pos = np.sort(np.random.uniform(0, 10, n))
    data = {
        "y": np.sin(pos) + np.random.normal(0, 0.1, n),
        "idx": np.arange(n),
        "pos": pos,
    }
    res = inla.fit("y ~ f(idx, model='crw1', positions=pos)", data=data)
    assert res.marginal_log_lik is not None
    assert np.isfinite(res.marginal_log_lik)


def test_crw2_simple_formula():
    n = 12
    pos = np.sort(np.random.uniform(0, 10, n))
    data = {
        "y": np.cos(pos) + np.random.normal(0, 0.1, n),
        "idx": np.arange(n),
        "pos": pos,
    }
    res = inla.fit("y ~ f(idx, model='crw2', positions=pos, layout='simple')", data=data)
    assert res.marginal_log_lik is not None
    assert np.isfinite(res.marginal_log_lik)


def test_negative_binomial_family():
    data = {
        "y": np.array([1, 3, 2, 6, 4, 5], dtype=float),
        "x": np.array([0, 1, 2, 3, 4, 5], dtype=int),
    }
    res = inla.fit("y ~ f(x, model='iid')", data=data, family="negative_binomial", size=2.0, initial_theta=[1.0])
    assert res.marginal_log_lik is not None
    assert np.isfinite(res.marginal_log_lik)


def test_zero_inflated_poisson_family():
    data = {
        "y": np.array([0, 2, 0, 4, 1, 0, 3], dtype=float),
        "x": np.array([0, 1, 2, 3, 4, 5, 6], dtype=int),
    }
    res = inla.fit("y ~ f(x, model='iid')", data=data, family="zero_inflated_poisson", zero_prob=0.2, initial_theta=[1.0])
    assert res.marginal_log_lik is not None
    assert np.isfinite(res.marginal_log_lik)


def test_survival_families_with_event():
    times = np.array([1.2, 2.5, 0.8, 3.1, 1.9, 2.0])
    events = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    data = {
        "y": times,
        "event": events,
        "x": np.arange(len(times), dtype=int),
    }
    res_exp = inla.fit("y ~ f(x, model='iid')", data=data, family="exponential_survival", event=events, initial_theta=[0.0])
    assert res_exp.marginal_log_lik is not None
    assert np.isfinite(res_exp.marginal_log_lik)

    res_weib = inla.fit("y ~ f(x, model='iid')", data=data, family="weibull_survival", event=events, shape=1.5, initial_theta=[0.0])
    assert res_weib.marginal_log_lik is not None
    assert np.isfinite(res_weib.marginal_log_lik)


def test_laplace_family():
    data = {
        "y": np.array([0.2, -0.5, 0.8, 0.1, -0.3, 0.4]),
        "x": np.arange(6, dtype=int),
    }
    res = inla.fit("y ~ f(x, model='iid')", data=data, family="laplace", alpha=0.5, gamma=0.5, initial_theta=[1.0])
    assert res.marginal_log_lik is not None
    assert np.isfinite(res.marginal_log_lik)
