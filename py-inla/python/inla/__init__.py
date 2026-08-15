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
from inla.api import InlaResult, _fit, fit
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
    RW1,
    RW2,
    SPDE,
    Besag,
    Binomial,
    Effect,
    Family,
    Gamma,
    Gaussian,
    Generic,
    Intercept,
    Linear,
    ModelSpec,
    NegativeBinomial,
    Poisson,
    Seasonal,
)


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
            "InlaResult",
            "PyCscMatrix",
            "PyMarginal1D",
            "PyInferenceResult",
            "parse_formula",
            "generic",
            "define",
            "GenericModel",
            "Model",
            "spde",
            "models",
            # ModelSpec & Effects
            "ModelSpec",
            "Effect",
            "Linear",
            "Intercept",
            "IID",
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
        "InlaResult": InlaResult,
        "PyCscMatrix": PyCscMatrix,
        "PyMarginal1D": PyMarginal1D,
        "PyInferenceResult": PyInferenceResult,
        "parse_formula": parse_formula,
        "generic": _generic,
        "spde": _spde,
        "models": _models_mod,
        "define": define,
        "GenericModel": GenericModel,
        "Model": Model,
        "ModelSpec": ModelSpec,
        "Effect": Effect,
        "Linear": Linear,
        "Intercept": Intercept,
        "IID": IID,
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
