"""Penalized Complexity (PC) and standard hyperparameter prior classes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from inla import _native


@dataclass
class Prior:
    """Base class for hyperparameter priors."""

    name: str
    params: list[float] = field(default_factory=list)

    def to_tuple(self) -> tuple[str, list[float]]:
        """Return (name, param_list) pair for the Rust core."""
        return (self.name, list(self.params))

    def to_dict(self) -> dict[str, Any]:
        """Return R-style hyper dictionary {'prior': name, 'param': params}."""
        return {"prior": self.name, "param": list(self.params)}

    def log_density(self, theta: Sequence[float] | float) -> float:
        """Evaluate the log-density of this prior on the internal θ scale."""
        if isinstance(theta, (int, float)):
            th = [float(theta)]
        else:
            th = [float(t) for t in theta]
        return float(_native.prior_log_density(self.name, self.params, th))


# =============================================================================
# Penalized Complexity (PC) Priors
# =============================================================================


@dataclass
class PCPrec(Prior):
    """Penalized Complexity (PC) prior on precision / standard deviation.

    Probability statement: P(σ > u) = α, with base model σ = 0 (no effect).

    Parameters
    ----------
    u : float
        Upper standard deviation quantile threshold (u > 0). Default: 1.0.
    alpha : float
        Tail probability P(σ > u) (0 < alpha < 1). Default: 0.01.
    """

    u: float = 1.0
    alpha: float = 0.01

    def __init__(self, u: float = 1.0, alpha: float = 0.01):
        if not (u > 0 and 0 < alpha < 1):
            raise ValueError(f"PCPrec requires u > 0 and 0 < alpha < 1, got u={u}, alpha={alpha}")
        super().__init__(name="pc.prec", params=[float(u), float(alpha)])
        self.u = float(u)
        self.alpha = float(alpha)


@dataclass
class PCCor0(Prior):
    """Penalized Complexity (PC) prior on correlation with base ρ = 0.

    Probability statement: P(|ρ| > u) = α, penalizing correlation towards independence.

    Parameters
    ----------
    u : float
        Correlation threshold in (0, 1). Default: 0.5.
    alpha : float
        Probability in (0, 1). Default: 0.05.
    """

    u: float = 0.5
    alpha: float = 0.05

    def __init__(self, u: float = 0.5, alpha: float = 0.05):
        if not (0 < u < 1 and 0 < alpha < 1):
            raise ValueError(
                f"PCCor0 requires 0 < u < 1 and 0 < alpha < 1, got u={u}, alpha={alpha}"
            )
        super().__init__(name="pc.cor0", params=[float(u), float(alpha)])
        self.u = float(u)
        self.alpha = float(alpha)


PCRho0 = PCCor0


@dataclass
class PCCor1(Prior):
    """Penalized Complexity (PC) prior on correlation with base ρ = 1.

    Probability statement: P(ρ > u) = α, penalizing correlation towards random walk.

    Parameters
    ----------
    u : float
        Correlation threshold in (-1, 1). Default: 0.5.
    alpha : float
        Probability in (0, 1). Default: 0.75.
    """

    u: float = 0.5
    alpha: float = 0.75

    def __init__(self, u: float = 0.5, alpha: float = 0.75):
        if not (-1 < u < 1 and 0 < alpha < 1):
            raise ValueError(
                f"PCCor1 requires -1 < u < 1 and 0 < alpha < 1, got u={u}, alpha={alpha}"
            )
        super().__init__(name="pc.cor1", params=[float(u), float(alpha)])
        self.u = float(u)
        self.alpha = float(alpha)


PCRho1 = PCCor1


@dataclass
class PCBym2(Prior):
    """Penalized Complexity (PC) prior on the BYM2 spatial mixing parameter φ ∈ (0, 1).

    Probability statement: P(φ < u) = α, with base model φ = 0 (pure unstructured noise).

    Parameters
    ----------
    u : float
        Mixing proportion threshold in (0, 1). Default: 0.5.
    alpha : float
        Probability in (0, 1). Default: 0.5.
    """

    u: float = 0.5
    alpha: float = 0.5

    def __init__(self, u: float = 0.5, alpha: float = 0.5):
        if not (0 < u < 1 and 0 < alpha < 1):
            raise ValueError(
                f"PCBym2 requires 0 < u < 1 and 0 < alpha < 1, got u={u}, alpha={alpha}"
            )
        super().__init__(name="pc.bym2", params=[float(u), float(alpha)])
        self.u = float(u)
        self.alpha = float(alpha)


PCPhi = PCBym2


@dataclass
class PCRange(Prior):
    """Penalized Complexity (PC) prior on spatial range ρ > 0.

    Probability statement: P(ρ < r0) = alpha_r.

    Parameters
    ----------
    r0 : float
        Range threshold (> 0). Default: 1.0.
    alpha_r : float
        Probability in (0, 1). Default: 0.05.
    d : float
        Dimension of spatial domain. Default: 2.0.
    """

    r0: float = 1.0
    alpha_r: float = 0.05
    d: float = 2.0

    def __init__(self, r0: float = 1.0, alpha_r: float = 0.05, d: float = 2.0):
        if not (r0 > 0 and 0 < alpha_r < 1 and d > 0):
            raise ValueError(
                f"PCRange requires r0 > 0, 0 < alpha_r < 1, d > 0, "
                f"got r0={r0}, alpha_r={alpha_r}, d={d}"
            )
        super().__init__(name="pc.range", params=[float(r0), float(alpha_r), float(d)])
        self.r0 = float(r0)
        self.alpha_r = float(alpha_r)
        self.d = float(d)


@dataclass
class PCSpde(Prior):
    """Penalized Complexity (PC) joint prior on Matérn / SPDE range and standard deviation.

    Probability statements:
        P(range < r0) = alpha_r
        P(σ > s0) = alpha_s

    Parameters
    ----------
    r0 : float
        Spatial range threshold. Default: 50.0.
    alpha_r : float
        Probability P(range < r0). Default: 0.05.
    s0 : float
        Marginal standard deviation threshold. Default: 1.0.
    alpha_s : float
        Probability P(σ > s0). Default: 0.01.
    d : float
        Dimension of the spatial domain. Default: 2.0.
    """

    r0: float = 50.0
    alpha_r: float = 0.05
    s0: float = 1.0
    alpha_s: float = 0.01
    d: float = 2.0

    def __init__(
        self,
        r0: float = 50.0,
        alpha_r: float = 0.05,
        s0: float = 1.0,
        alpha_s: float = 0.01,
        d: float = 2.0,
    ):
        if not (r0 > 0 and 0 < alpha_r < 1 and s0 > 0 and 0 < alpha_s < 1 and d > 0):
            raise ValueError("PCSpde requires positive thresholds and probabilities in (0, 1)")
        super().__init__(
            name="pc.spde",
            params=[float(r0), float(alpha_r), float(s0), float(alpha_s), float(d)],
        )
        self.r0 = float(r0)
        self.alpha_r = float(alpha_r)
        self.s0 = float(s0)
        self.alpha_s = float(alpha_s)
        self.d = float(d)


PCMatern = PCSpde


# =============================================================================
# Standard Conjugate & Diffuse Priors
# =============================================================================


@dataclass
class LogGamma(Prior):
    """Gamma prior on precision τ = exp(θ) with shape and rate parameters."""

    shape: float = 1.0
    rate: float = 5e-5

    def __init__(self, shape: float = 1.0, rate: float = 5e-5):
        if not (shape > 0 and rate > 0):
            raise ValueError(
                f"LogGamma requires shape > 0 and rate > 0, got shape={shape}, rate={rate}"
            )
        super().__init__(name="loggamma", params=[float(shape), float(rate)])
        self.shape = float(shape)
        self.rate = float(rate)


@dataclass
class Gaussian(Prior):
    """Gaussian prior directly on internal hyperparameter θ."""

    mean: float = 0.0
    precision: float = 0.001

    def __init__(self, mean: float = 0.0, precision: float = 0.001):
        if precision < 0:
            raise ValueError(f"Gaussian prior requires precision >= 0, got {precision}")
        super().__init__(name="gaussian", params=[float(mean), float(precision)])
        self.mean = float(mean)
        self.precision = float(precision)


Normal = Gaussian
GaussianPrior = Gaussian


@dataclass
class Flat(Prior):
    """Flat / improper uniform prior on internal hyperparameter θ."""

    def __init__(self):
        super().__init__(name="flat", params=[])


Uniform = Flat


@dataclass
class LogitBeta(Prior):
    """Beta(a, b) prior on probability parameter p ∈ (0, 1) where θ = logit(p)."""

    a: float = 1.0
    b: float = 1.0

    def __init__(self, a: float = 1.0, b: float = 1.0):
        if not (a > 0 and b > 0):
            raise ValueError(f"LogitBeta requires a > 0 and b > 0, got a={a}, b={b}")
        super().__init__(name="logitbeta", params=[float(a), float(b)])
        self.a = float(a)
        self.b = float(b)
