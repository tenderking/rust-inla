"""Python front-end for rust-inla.

Usage::

    import inla
    from inla import Besag, Binomial, ModelSpec

    # Formula interface
    result = inla("y ~ x + f(idx, model='besag')", data=..., family=...)

    # Functional interface
    result = inla.fit(
        data=df,
        response="y",
        fixed=["x"],
        random=[Besag("idx", graph=adj)],
        family=Binomial(Ntrials="n"),
    )

    # Declarative ModelSpec interface
    class DiseaseModel(ModelSpec):
        response = "y"
        family = Binomial(Ntrials="n")
        fixed = ["x"]
        spatial = Besag("idx", graph=adj)

    result = inla.fit(DiseaseModel, data=df)
"""

from __future__ import annotations

import sys
import types

from inla import generic as _generic
from inla import models as _models_mod
from inla import priors as _priors_mod
from inla import spde as _spde
from inla._native import (
    PyCscMatrix,
    PyInferenceResult,
    PyMarginal1D,
    ar1_precision_matrix,
    ar1_precision_matrix_csc,
    arp_precision_matrix,
    besag_precision_matrix,
    crw1_precision_matrix,
    crw2_precision_matrix,
    fgn_approx_latent_len,
    fgn_approx_precision_matrix,
    fgn_hurst_from_intern,
    fgn_intern_from_hurst,
    fgn_precision_matrix,
    iid_precision_matrix,
    rw1_precision_matrix,
    rw2_precision_matrix,
    seasonal_precision_matrix,
)
from inla.api import InlaResult, _fit, competing_event, coxph_expand, fit, group
from inla.formula import parse_formula
from inla.generic import GenericModel, Model, define
from inla.models import (
    AR,
    AR1,
    BYM,
    BYM2,
    CRW1,
    CRW2,
    FGN,
    IID,
    IID2D,
    IID3D,
    IID4D,
    IID5D,
    RW1,
    RW2,
    SPDE,
    Besag,
    Binomial,
    CoxPH,
    Effect,
    ExponentialSurvival,
    Family,
    Gamma,
    Gaussian,
    Generic,
    Intercept,
    Linear,
    LoglogisticSurvival,
    LognormalSurvival,
    ModelSpec,
    NegativeBinomial,
    Poisson,
    Seasonal,
    Surv,
    WeibullSurvival,
)
from inla.priors import (
    Flat,
    GaussianPrior,
    LogGamma,
    LogitBeta,
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
    Prior,
    Uniform,
)
from inla.stack import Stack


class _InlaModule(types.ModuleType):
    """Package module that is itself callable: ``inla(...)``."""

    def __call__(self, *args, **kwargs):
        return _fit(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return _fit(*args, **kwargs)

    def define(self, *args, **kwargs):
        return _generic.define(*args, **kwargs)

    @property
    def generic(self):
        return _generic

    @property
    def models(self):
        return _models_mod

    @property
    def priors(self):
        return _priors_mod

    @property
    def spde(self):
        return _spde


_mod = _InlaModule(__name__)
_mod.__dict__.update(
    {
        "__doc__": __doc__,
        "__file__": __file__,
        "__path__": __path__,
        "__package__": __package__,
        "__all__": [
            "fit",
            "coxph_expand",
            "competing_event",
            "InlaResult",
            "PyCscMatrix",
            "PyMarginal1D",
            "PyInferenceResult",
            "parse_formula",
            "group",
            "generic",
            "define",
            "GenericModel",
            "Model",
            "spde",
            "Stack",
            "models",
            "priors",
            # ModelSpec & Effects
            "ModelSpec",
            "Effect",
            "Linear",
            "Intercept",
            "IID",
            "IID2D",
            "IID3D",
            "IID4D",
            "IID5D",
            "Besag",
            "BYM",
            "BYM2",
            "RW1",
            "RW2",
            "AR1",
            "AR",
            "SPDE",
            "Generic",
            "CRW1",
            "CRW2",
            "FGN",
            "Seasonal",
            # Families
            "Family",
            "Gaussian",
            "Binomial",
            "Poisson",
            "NegativeBinomial",
            "Gamma",
            "ExponentialSurvival",
            "WeibullSurvival",
            "LoglogisticSurvival",
            "LognormalSurvival",
            "CoxPH",
            "Surv",
            # Priors
            "Prior",
            "PCPrec",
            "PCCor0",
            "PCCor1",
            "PCRho0",
            "PCRho1",
            "PCBym2",
            "PCPhi",
            "PCRange",
            "PCSpde",
            "PCMatern",
            "LogGamma",
            "Flat",
            "Uniform",
            "Normal",
            "GaussianPrior",
            "LogitBeta",
            # Precision matrix functions
            "ar1_precision_matrix",
            "ar1_precision_matrix_csc",
            "arp_precision_matrix",
            "rw1_precision_matrix",
            "rw2_precision_matrix",
            "seasonal_precision_matrix",
            "crw1_precision_matrix",
            "crw2_precision_matrix",
            "iid_precision_matrix",
            "besag_precision_matrix",
            "fgn_precision_matrix",
            "fgn_approx_precision_matrix",
            "fgn_hurst_from_intern",
            "fgn_intern_from_hurst",
            "fgn_approx_latent_len",
        ],
        "fit": fit,
        "coxph_expand": coxph_expand,
        "competing_event": competing_event,
        "InlaResult": InlaResult,
        "PyCscMatrix": PyCscMatrix,
        "PyMarginal1D": PyMarginal1D,
        "PyInferenceResult": PyInferenceResult,
        "parse_formula": parse_formula,
        "group": group,
        "generic": _generic,
        "spde": _spde,
        "Stack": Stack,
        "models": _models_mod,
        "priors": _priors_mod,
        "define": define,
        "GenericModel": GenericModel,
        "Model": Model,
        "ModelSpec": ModelSpec,
        "Effect": Effect,
        "Linear": Linear,
        "Intercept": Intercept,
        "IID": IID,
        "IID2D": IID2D,
        "IID3D": IID3D,
        "IID4D": IID4D,
        "IID5D": IID5D,
        "Besag": Besag,
        "BYM": BYM,
        "BYM2": BYM2,
        "RW1": RW1,
        "RW2": RW2,
        "AR1": AR1,
        "AR": AR,
        "SPDE": SPDE,
        "Generic": Generic,
        "CRW1": CRW1,
        "CRW2": CRW2,
        "FGN": FGN,
        "Seasonal": Seasonal,
        "Family": Family,
        "Gaussian": Gaussian,
        "Binomial": Binomial,
        "Poisson": Poisson,
        "NegativeBinomial": NegativeBinomial,
        "Gamma": Gamma,
        "ExponentialSurvival": ExponentialSurvival,
        "WeibullSurvival": WeibullSurvival,
        "LoglogisticSurvival": LoglogisticSurvival,
        "LognormalSurvival": LognormalSurvival,
        "CoxPH": CoxPH,
        "Surv": Surv,
        "Prior": Prior,
        "PCPrec": PCPrec,
        "PCCor0": PCCor0,
        "PCCor1": PCCor1,
        "PCRho0": PCRho0,
        "PCRho1": PCRho1,
        "PCBym2": PCBym2,
        "PCPhi": PCPhi,
        "PCRange": PCRange,
        "PCSpde": PCSpde,
        "PCMatern": PCMatern,
        "LogGamma": LogGamma,
        "Flat": Flat,
        "Uniform": Uniform,
        "Normal": Normal,
        "GaussianPrior": GaussianPrior,
        "LogitBeta": LogitBeta,
        "ar1_precision_matrix": ar1_precision_matrix,
        "ar1_precision_matrix_csc": ar1_precision_matrix_csc,
        "arp_precision_matrix": arp_precision_matrix,
        "rw1_precision_matrix": rw1_precision_matrix,
        "rw2_precision_matrix": rw2_precision_matrix,
        "seasonal_precision_matrix": seasonal_precision_matrix,
        "crw1_precision_matrix": crw1_precision_matrix,
        "crw2_precision_matrix": crw2_precision_matrix,
        "iid_precision_matrix": iid_precision_matrix,
        "besag_precision_matrix": besag_precision_matrix,
        "fgn_precision_matrix": fgn_precision_matrix,
        "fgn_approx_precision_matrix": fgn_approx_precision_matrix,
        "fgn_hurst_from_intern": fgn_hurst_from_intern,
        "fgn_intern_from_hurst": fgn_intern_from_hurst,
        "fgn_approx_latent_len": fgn_approx_latent_len,
    }
)
sys.modules[__name__] = _mod
