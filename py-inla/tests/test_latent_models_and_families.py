"""Pytest suite for expanded latent models and family arms in py-inla."""

import numpy as np

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
    res = inla.fit(
        "y ~ f(x, model='iid')",
        data=data,
        family="negative_binomial",
        size=2.0,
        initial_theta=[1.0],
    )
    assert res.marginal_log_lik is not None
    assert np.isfinite(res.marginal_log_lik)


def test_zero_inflated_poisson_family():
    data = {
        "y": np.array([0, 2, 0, 4, 1, 0, 3], dtype=float),
        "x": np.array([0, 1, 2, 3, 4, 5, 6], dtype=int),
    }
    res = inla.fit(
        "y ~ f(x, model='iid')",
        data=data,
        family="zero_inflated_poisson",
        zero_prob=0.2,
        initial_theta=[1.0],
    )
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
    res_exp = inla.fit(
        "y ~ f(x, model='iid')",
        data=data,
        family="exponential_survival",
        event=events,
        initial_theta=[0.0],
    )
    assert res_exp.marginal_log_lik is not None
    assert np.isfinite(res_exp.marginal_log_lik)

    res_weib = inla.fit(
        "y ~ f(x, model='iid')",
        data=data,
        family="weibull_survival",
        event=events,
        shape=1.5,
        initial_theta=[0.0],
    )
    assert res_weib.marginal_log_lik is not None
    assert np.isfinite(res_weib.marginal_log_lik)


def test_survival_left_and_interval_censoring():
    times = np.array([0.5, 1.2, 0.8, 1.5, 2.0, 0.4])
    events = np.array([1.0, 0.0, 2.0, 3.0, 1.0, 3.0])
    y_upper = np.array([np.nan, np.nan, np.nan, 2.4, np.nan, 1.1])
    data = {"y": times, "event": events, "y_upper": y_upper, "x": np.arange(6, dtype=int)}
    res = inla.fit(
        data=data,
        response=inla.Surv(time="y", event="event", time2="y_upper"),
        random=[inla.IID("x")],
        intercept=False,
        family=inla.ExponentialSurvival(),
        initial_theta=[0.0],
    )
    assert np.isfinite(res.marginal_log_lik)

    res_w = inla.fit(
        "y ~ f(x, model='iid')",
        data=data,
        family=inla.WeibullSurvival(shape=1.2),
        initial_theta=[0.0],
    )
    assert np.isfinite(res_w.marginal_log_lik)


def test_coxph_expand_and_poisson_fit():
    raw = {
        "time": np.array([1.0, 2.0, 3.0, 2.5]),
        "status": np.array([1.0, 0.0, 1.0, 1.0]),
        "treat": np.array([0.0, 1.0, 0.0, 1.0]),
    }
    exp = inla.coxph_expand(raw, time="time", event="status", cutpoints=4)
    assert exp["y_events"].size == exp["exposure"].size
    assert exp["y_events"].sum() == 3.0
    assert np.all(exp["exposure"] > 0)
    res = inla.fit(
        data=exp,
        response="y_events",
        family=inla.Poisson(E="exposure"),
        random=[inla.RW1("time_bin")],
        intercept=False,
        initial_theta=[0.0],
    )
    assert np.isfinite(res.marginal_log_lik)


def test_loglogistic_lognormal_and_weibull_ph():
    times = np.array([0.8, 1.1, 1.4, 0.6, 2.0, 1.7])
    events = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    data = {"y": times, "event": events, "x": np.arange(6, dtype=int)}
    res_ll = inla.fit(
        "y ~ f(x, model='iid')",
        data=data,
        family=inla.LoglogisticSurvival(shape=1.5),
        initial_theta=[0.0],
    )
    assert np.isfinite(res_ll.marginal_log_lik)
    res_ln = inla.fit(
        "y ~ f(x, model='iid')",
        data=data,
        family=inla.LognormalSurvival(prec=2.0),
        initial_theta=[0.0],
    )
    assert np.isfinite(res_ln.marginal_log_lik)
    res_ph = inla.fit(
        "y ~ f(x, model='iid')",
        data=data,
        family=inla.WeibullSurvival(shape=1.4, variant=0),
        initial_theta=[0.0],
    )
    assert np.isfinite(res_ph.marginal_log_lik)


def test_kidney_like_frailty_and_leukemia_like_spatial():
    # Kidney-style clustered recurrent times with iid frailty.
    rng = np.random.default_rng(16)
    n_subj = 12
    subject = np.repeat(np.arange(n_subj), 2)
    frailty = np.repeat(rng.normal(0.0, 0.3, n_subj), 2)
    time = rng.exponential(np.exp(-frailty))
    event = np.ones(time.size)
    data = {"time": time, "event": event, "subject": subject}
    res_k = inla.fit(
        data=data,
        response=inla.Surv(time="time", event="event"),
        random=[inla.IID("subject")],
        intercept=False,
        family=inla.ExponentialSurvival(),
        initial_theta=[0.0],
    )
    assert np.isfinite(res_k.marginal_log_lik)

    # Leukemia-style spatial survival on a small lattice.
    nrow = ncol = 4
    n = nrow * ncol
    adj = np.zeros((n, n))
    for i in range(nrow):
        for j in range(ncol):
            u = i * ncol + j
            if j + 1 < ncol:
                v = i * ncol + (j + 1)
                adj[u, v] = adj[v, u] = 1.0
            if i + 1 < nrow:
                v = (i + 1) * ncol + j
                adj[u, v] = adj[v, u] = 1.0
    region = np.arange(n)
    time = 0.4 + 0.05 * region
    event = np.ones(n)
    data_l = {"time": time, "event": event, "region": region, "adj_matrix": adj}
    res_l = inla.fit(
        "time ~ f(region, model='besag')",
        data=data_l,
        family=inla.WeibullSurvival(shape=1.2, variant=0),
        initial_theta=[0.0],
    )
    assert np.isfinite(res_l.marginal_log_lik)


def test_coxph_family_and_competing_risks():
    raw = {
        "time": np.array([1.0, 2.0, 3.0, 2.5, 1.5, 2.2]),
        "status": np.array([1.0, 0.0, 1.0, 1.0, 2.0, 1.0]),
        "idx": np.arange(6, dtype=int),
    }
    res = inla.fit(
        data=raw,
        response="time",
        family=inla.CoxPH(event="status", cutpoints=4),
        intercept=False,
        initial_theta=[0.0],
    )
    assert np.isfinite(res.marginal_log_lik)
    cause1 = inla.competing_event(raw["status"], 1)
    assert cause1.sum() == 4.0
    res_c = inla.fit(
        data={**raw, "event": cause1},
        response="time",
        family=inla.ExponentialSurvival(),
        random=[inla.IID("idx")],
        intercept=False,
        initial_theta=[0.0],
    )
    assert np.isfinite(res_c.marginal_log_lik)


def test_laplace_family():
    data = {
        "y": np.array([0.2, -0.5, 0.8, 0.1, -0.3, 0.4]),
        "x": np.arange(6, dtype=int),
    }
    res = inla.fit(
        "y ~ f(x, model='iid')",
        data=data,
        family="laplace",
        alpha=0.5,
        gamma=0.5,
        initial_theta=[1.0],
    )
    assert res.marginal_log_lik is not None
    assert np.isfinite(res.marginal_log_lik)
