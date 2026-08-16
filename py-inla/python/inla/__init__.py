"""Python front-end for rust-inla.

Usage::

    import inla
    result = inla("y ~ x + f(idx, model='besag')", data=..., family=...)

    model = inla.generic.define(n=20, Q=...)
    result = inla("y ~ -1 + f(idx, model='rgeneric')", data=..., rgeneric=model)
"""

from __future__ import annotations

import sys
import types

from inla import generic as _generic
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
            "spde",
        ],
        "fit": fit,
        "InlaResult": InlaResult,
        "PyCscMatrix": PyCscMatrix,
        "PyMarginal1D": PyMarginal1D,
        "PyInferenceResult": PyInferenceResult,
        "parse_formula": parse_formula,
        "generic": _generic,
        "spde": _spde,
        "define": define,
        "GenericModel": GenericModel,
        "Model": Model,
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
