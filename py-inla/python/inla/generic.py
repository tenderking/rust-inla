"""R-INLA-style ``rgeneric`` / custom latent models for Python ``inla()``."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse


def _as_csc(q) -> sparse.csc_matrix:
    if sparse.issparse(q):
        return sparse.csc_matrix(q)
    if hasattr(q, "to_scipy"):
        return sparse.csc_matrix(q.to_scipy())
    return sparse.csc_matrix(np.asarray(q, dtype=float))


@dataclass
class GenericModel:
    """User-defined GMRF latent model (R ``inla.rgeneric`` analogue).

    Pass into ``inla(..., models={"name": model})`` and reference it from the
    formula as ``f(idx, model='name')``, or use ``model='rgeneric'`` with
    ``rgeneric=model``.
    """

    n: int
    Q: Callable[[Sequence[float]], Any]
    n_theta: int = 1
    initial: list[float] = field(default_factory=list)
    log_prior: Callable[[Sequence[float]], float] | None = None
    name: str = "rgeneric"

    def __post_init__(self):
        if self.n <= 0:
            raise ValueError("GenericModel.n must be > 0")
        if self.n_theta < 0:
            raise ValueError("GenericModel.n_theta must be >= 0")
        if not self.initial:
            self.initial = [0.0] * self.n_theta
        else:
            self.initial = [float(v) for v in np.asarray(self.initial, dtype=float).reshape(-1)]
            if len(self.initial) != self.n_theta:
                raise ValueError(f"initial length {len(self.initial)} != n_theta={self.n_theta}")

    def precision(self, theta: Sequence[float]) -> sparse.csc_matrix:
        th = list(theta)
        if len(th) != self.n_theta:
            raise ValueError(f"theta length {len(th)} != n_theta={self.n_theta}")
        q = _as_csc(self.Q(th))
        if q.shape != (self.n, self.n):
            raise ValueError(f"Q(theta) shape {q.shape} != ({self.n}, {self.n})")
        return q

    def eval_log_prior(self, theta: Sequence[float]) -> float:
        if self.log_prior is None:
            return float(-0.5 * 0.1 * sum(float(v) ** 2 for v in theta))
        return float(self.log_prior(list(theta)))


class Model:
    """Subclassable custom latent model.

    Example
    -------
    >>> class MyIID(Model):
    ...     def __init__(self, n):
    ...         super().__init__(n=n, n_theta=1, initial=[0.0], name="myiid")
    ...     def Q(self, theta):
    ...         return sparse.eye(self.n, format="csc") * float(np.exp(theta[0]))
    """

    def __init__(
        self,
        n: int,
        *,
        n_theta: int = 1,
        initial: Sequence[float] | None = None,
        name: str = "rgeneric",
    ):
        self.n = int(n)
        self.n_theta = int(n_theta)
        self.name = str(name)
        if initial is None:
            self.initial = [0.0] * self.n_theta
        else:
            self.initial = [float(v) for v in np.asarray(initial, dtype=float).reshape(-1)]
            if len(self.initial) != self.n_theta:
                raise ValueError(f"initial length {len(self.initial)} != n_theta={self.n_theta}")

    def Q(self, theta: Sequence[float]) -> Any:  # noqa: N802 — R-style name
        raise NotImplementedError(f"{type(self).__name__} must implement Q(theta)")

    def log_prior(self, theta: Sequence[float]) -> float:
        return float(-0.5 * 0.1 * sum(float(v) ** 2 for v in theta))

    def precision(self, theta: Sequence[float]) -> sparse.csc_matrix:
        th = list(theta)
        if len(th) != self.n_theta:
            raise ValueError(f"theta length {len(th)} != n_theta={self.n_theta}")
        q = _as_csc(self.Q(th))
        if q.shape != (self.n, self.n):
            raise ValueError(f"Q(theta) shape {q.shape} != ({self.n}, {self.n})")
        return q

    def eval_log_prior(self, theta: Sequence[float]) -> float:
        return float(self.log_prior(list(theta)))

    def as_generic(self) -> GenericModel:
        return GenericModel(
            n=self.n,
            Q=self.Q,
            n_theta=self.n_theta,
            initial=list(self.initial),
            log_prior=self.log_prior,
            name=self.name,
        )


def define(
    *,
    n: int,
    Q: Callable[[Sequence[float]], Any],
    n_theta: int = 1,
    initial: Sequence[float] | None = None,
    log_prior: Callable[[Sequence[float]], float] | None = None,
    name: str = "rgeneric",
) -> GenericModel:
    """Define a custom latent model (R ``inla.rgeneric.define`` analogue).

    Parameters
    ----------
    n :
        Latent dimension.
    Q :
        ``Q(theta) ->`` sparse/dense precision matrix.
    n_theta :
        Hyperparameter dimension.
    initial :
        Starting values for ``theta`` (length ``n_theta``).
    log_prior :
        ``log_prior(theta) -> float``. Default: weak Gaussian.
    name :
        Optional label; use the same string in ``f(..., model=name)``.
    """
    return GenericModel(
        n=n,
        Q=Q,
        n_theta=n_theta,
        initial=list(initial) if initial is not None else [],
        log_prior=log_prior,
        name=name,
    )


# Duck-type union for resolve helpers
GenericLike = (GenericModel, Model)
