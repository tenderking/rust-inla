"""Tests for Penalized Complexity (PC) prior framework and catalog (Issue #20)."""

import math

import numpy as np
import pytest

import inla
from inla import (
    AR1,
    BYM2,
    SPDE,
    Flat,
    Gaussian,
    GaussianPrior,
    LogGamma,
    LogitBeta,
    ModelSpec,
    Normal,
    PCBym2,
    PCCor0,
    PCCor1,
    PCMatern,
    PCPhi,
    PCPrec,
    PCRange,
    PCRho0,
    PCRho1,
    PCSpde,
    Uniform,
)


def test_pc_prec_density_and_tuple():
    prior = PCPrec(u=1.0, alpha=0.01)
    name, param = prior.to_tuple()
    assert name == "pc.prec"
    assert param == [1.0, 0.01]
    d = prior.to_dict()
    assert d == {"prior": "pc.prec", "param": [1.0, 0.01]}

    # theta = log(tau) = 0 => tau = 1, sigma = 1
    # lambda = -ln(0.01) / 1 = 4.605170185988092
    # log pi(theta) = ln(lambda/2) - lambda*exp(-theta/2) - theta/2
    lam = -math.log(0.01) / 1.0
    expected = math.log(lam / 2.0) - lam * math.exp(0.0) - 0.0
    lp = prior.log_density(0.0)
    assert pytest.approx(lp, 1e-10) == expected


def test_pc_cor0_density_symmetry_and_zero():
    prior = PCCor0(u=0.5, alpha=0.05)
    assert prior.to_tuple() == ("pc.cor0", [0.5, 0.05])
    # At theta = 0 (rho = 0):
    # d_u = sqrt(-ln(1 - 0.5^2)) = sqrt(-ln(0.75)) = 0.53634289
    # lambda = -ln(0.05) / d_u = 5.58548177
    # log pi(0) = ln(lambda) - 2 ln(2)
    d_u = math.sqrt(-math.log(1.0 - 0.25))
    lam = -math.log(0.05) / d_u
    expect_0 = math.log(lam) - 2.0 * math.log(2.0)
    assert pytest.approx(prior.log_density(0.0), 1e-6) == expect_0

    # Symmetry about theta=0 (rho=0)
    lp_pos = prior.log_density(1.2)
    lp_neg = prior.log_density(-1.2)
    assert pytest.approx(lp_pos, 1e-10) == lp_neg
    assert PCRho0 is PCCor0


def test_pc_cor1_density():
    prior = PCCor1(u=0.5, alpha=0.75)
    assert prior.to_tuple() == ("pc.cor1", [0.5, 0.75])
    # Internal θ=0 (ρ=0); λ from R-INLA inla.pc.cor1.lambda / PRIOR_EVAL
    assert pytest.approx(prior.log_density(0.0), rel=1e-8) == -2.381562305990987
    assert PCRho1 is PCCor1


def test_pc_bym2_density():
    prior = PCBym2(u=0.5, alpha=0.5)
    assert prior.to_tuple() == ("pc.bym2", [0.5, 0.5])
    assert pytest.approx(prior.log_density(0.0), rel=1e-8) == -1.486482918057251
    assert PCPhi is PCBym2


def test_pc_range_and_spde():
    prior_r = PCRange(r0=20.0, alpha_r=0.05, d=2.0)
    assert prior_r.to_tuple() == ("pc.range", [20.0, 0.05, 2.0])
    # lambda = -ln(0.05) * 20^(2/2) = 2.99573227 * 20 = 59.914645
    # theta = ln(20)
    # log pi(theta) = ln(lambda * 1) - theta - lambda * exp(-theta) = ln(59.9146) - ln(20) - 59.9146/20
    lam = -math.log(0.05) * 20.0
    theta = math.log(20.0)
    expected_r = math.log(lam) - theta - lam * math.exp(-theta)
    assert pytest.approx(prior_r.log_density(theta), 1e-10) == expected_r

    prior_spde = PCSpde(r0=50.0, alpha_r=0.05, s0=2.0, alpha_s=0.01, d=2.0)
    assert prior_spde.to_tuple() == ("pc.spde", [50.0, 0.05, 2.0, 0.01, 2.0])
    assert PCMatern is PCSpde
    assert math.isfinite(prior_spde.log_density([0.0, 0.0]))


def test_standard_priors_and_aliases():
    lg = LogGamma(shape=2.0, rate=0.5)
    assert lg.to_tuple() == ("loggamma", [2.0, 0.5])
    assert math.isfinite(lg.log_density(0.0))

    gp = GaussianPrior(mean=0.0, precision=1.0)
    assert gp.to_tuple() == ("gaussian", [0.0, 1.0])
    assert Normal is GaussianPrior

    fl = Flat()
    assert fl.to_tuple() == ("flat", [])
    assert fl.log_density(1.23) == 0.0
    assert Uniform is Flat

    lb = LogitBeta(a=1.0, b=1.0)
    assert lb.to_tuple() == ("logitbeta", [1.0, 1.0])
    assert math.isfinite(lb.log_density(0.0))


def test_gaussian_family_preserves_initial_and_prior():
    gf = Gaussian(obs_precision=4.0, prior_prec=PCPrec(u=1.0, alpha=0.01))
    assert gf.control_family is not None
    hyper = gf.control_family["hyper"]["prec"]
    assert hyper["initial"] == 4.0
    assert hyper["prior"] == "pc.prec"
    assert hyper["param"] == [1.0, 0.01]


def test_fit_ar1_with_pc_priors():
    np.random.seed(42)
    n = 40
    # Simulate AR1 series
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.6 * x[i - 1] + np.random.normal(0, 0.5)
    y = x + np.random.normal(0, 0.2, size=n)
    time_idx = np.arange(n)

    data = {"y": y, "t": time_idx}

    # 1. Functional API with typed PC priors (no intercept)
    res1 = inla.fit(
        data=data,
        response="y",
        intercept=False,
        family=Gaussian(prior_prec=PCPrec(u=1.0, alpha=0.01)),
        random=[
            AR1("t", prior_prec=PCPrec(u=1.0, alpha=0.01), prior_rho=PCCor0(u=0.5, alpha=0.05))
        ],
    )
    assert res1.latent_means is not None
    assert len(res1.latent_means) == n

    # 2. Formula API with hyper string dictionaries
    res2 = inla.fit(
        "y ~ 0 + f(t, model='ar1', hyper={'prec': {'prior': 'pc.prec', 'param': [1.0, 0.01]}, 'rho': {'prior': 'pc.cor0', 'param': [0.5, 0.05]}})",
        data=data,
        control_family={"hyper": {"prec": {"prior": "pc.prec", "param": [1.0, 0.01]}}},
    )
    assert res2.latent_means is not None
    # Both formulations should yield equivalent latent means
    np.testing.assert_allclose(res1.latent_means, res2.latent_means, rtol=1e-3, atol=1e-3)


def test_fit_bym2_with_pc_priors():
    # 4-node ring graph
    adj = [[1, 3], [0, 2], [1, 3], [0, 2]]
    y = np.array([1.2, -0.8, 0.9, -1.1])
    region = np.array([0, 1, 2, 3])
    data = {"y": y, "region": region}

    # Declarative ModelSpec with PC priors
    class DiseaseMappingModel(ModelSpec):
        response = "y"
        intercept = False
        family = Gaussian(prior_prec=PCPrec(u=1.0, alpha=0.01))
        spatial = BYM2(
            "region",
            graph=adj,
            scale_model=True,
            prior_prec=PCPrec(u=1.0, alpha=0.01),
            prior_phi=PCBym2(u=0.5, alpha=0.5),
        )

    res = inla.fit(DiseaseMappingModel, data=data)
    assert res.latent_means is not None
    assert len(res.latent_means) == 4


def test_fit_spde_with_pc_priors():
    mesh = inla.spde.lattice_mesh(xlim=(0.0, 1.0), ylim=(0.0, 1.0), nx=3, ny=3)
    verts = mesh["vertices"]
    n_nodes = len(verts)
    y = np.array([0.5, 0.8, 0.3, -0.2, 0.0, 0.4, 0.1, -0.5, -0.1])
    data = {"y": y, "idx": np.arange(n_nodes), "x": verts[:, 0], "y_coord": verts[:, 1]}

    res = inla.fit(
        data=data,
        response="y",
        intercept=False,
        family=Gaussian(prior_prec=PCPrec(u=1.0, alpha=0.01)),
        random=[
            SPDE(
                "idx",
                spde_model=mesh,
                loc_x="x",
                loc_y="y_coord",
                prior_range=PCRange(r0=1.0, alpha_r=0.05),
                prior_sigma=PCPrec(u=1.0, alpha=0.01),
            )
        ],
    )
    assert res.latent_means is not None
    assert len(res.latent_means) == n_nodes


def test_pc_spde_defaults_and_rejects_non_2d():
    prior = PCSpde()
    assert prior.to_tuple() == ("pc.spde", [1.0, 0.05, 1.0, 0.01, 2.0])
    with pytest.raises(ValueError, match="d=2"):
        PCSpde(d=1.0)


def test_hyper_overlay_uses_registry_slot_labels():
    from inla.api import _resolve_effect_priors

    specs = _resolve_effect_priors(
        "ar1",
        {
            "hyper": {
                "rho": {"prior": "pc.cor0", "param": [0.5, 0.05]},
            }
        },
    )
    assert specs[0][0] == "pc.prec"
    assert specs[1] == ("pc.cor0", [0.5, 0.05])

    phi = _resolve_effect_priors("bym2", {"prior_phi": PCBym2(u=0.25, alpha=0.4)})
    assert phi[0][0] == "pc.prec"
    assert phi[1] == ("pc.bym2", [0.25, 0.4])

    with pytest.raises(ValueError, match="unknown hyper slot"):
        _resolve_effect_priors("ar1", {"hyper": {"not_a_slot": {"prior": "flat"}}})
