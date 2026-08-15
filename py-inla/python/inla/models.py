"""Typed latent effect components, likelihood families, and declarative ModelSpec."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Union

from inla.formula import FTerm, ParsedFormula


# =============================================================================
# Likelihood Families
# =============================================================================


@dataclass
class Family:
    """Base class for likelihood families."""

    name: str
    Ntrials: Any = None
    E: Any = None
    event: Any = None
    size: float = 1.0
    zero_prob: float = 0.1
    inflation: str = "type0"
    alpha: float = 0.5
    gamma: float = 1.0
    shape: float = 1.0
    control_family: Optional[Mapping[str, Any]] = None


@dataclass
class Gaussian(Family):
    """Gaussian (normal) likelihood."""

    def __init__(
        self,
        *,
        obs_precision: Optional[float] = None,
        control_family: Optional[Mapping[str, Any]] = None,
    ):
        ctrl = dict(control_family) if control_family is not None else {}
        if obs_precision is not None:
            ctrl.setdefault("hyper", {}).setdefault("prec", {})[
                "initial"
            ] = float(obs_precision)
        super().__init__(name="gaussian", control_family=ctrl if ctrl else None)


@dataclass
class Binomial(Family):
    """Binomial likelihood."""

    def __init__(
        self,
        Ntrials: Any = None,
        *,
        control_family: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__(
            name="binomial", Ntrials=Ntrials, control_family=control_family
        )


@dataclass
class Poisson(Family):
    """Poisson likelihood."""

    def __init__(
        self,
        E: Any = None,
        *,
        control_family: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__(name="poisson", E=E, control_family=control_family)


@dataclass
class NegativeBinomial(Family):
    """Negative binomial likelihood."""

    def __init__(
        self,
        *,
        size: float = 1.0,
        control_family: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__(
            name="negative_binomial", size=size, control_family=control_family
        )


@dataclass
class Gamma(Family):
    """Gamma likelihood."""

    def __init__(
        self,
        *,
        shape: float = 1.0,
        control_family: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__(name="gamma", shape=shape, control_family=control_family)


# =============================================================================
# Latent Effect Components
# =============================================================================


@dataclass
class Effect:
    """Base class for structured random / latent effect terms."""

    index: str
    model: str = "iid"
    order: int = 0
    graph: Any = None
    scale_model: Optional[bool] = None
    initial: Any = None
    weights: Any = None
    group: Optional[str] = None
    group_model: Optional[str] = None
    replicate: Optional[str] = None
    cyclic: bool = False
    prior: Any = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    def to_fterm(self) -> FTerm:
        """Convert this Effect into an internal FTerm dataclass."""
        kw = dict(self.kwargs)
        if self.weights is not None:
            kw["weights"] = self.weights
        if self.group is not None:
            kw["group"] = self.group
        if self.group_model is not None:
            kw["group_model"] = self.group_model
            kw["control_group"] = {"model": self.group_model}
        if self.replicate is not None:
            kw["replicate"] = self.replicate
        if self.cyclic:
            kw["cyclic"] = True
        if self.prior is not None:
            kw["prior"] = self.prior
        return FTerm(
            index=self.index,
            model=self.model.lower(),
            order=self.order,
            graph=self.graph,
            scale_model=self.scale_model,
            initial=self.initial,
            kwargs=kw,
        )


@dataclass
class IID(Effect):
    """Independent and identically distributed (IID) Gaussian random effect."""

    def __init__(
        self,
        index: str,
        *,
        initial: Any = None,
        prior: Any = None,
        group: Optional[str] = None,
        group_model: Optional[str] = None,
        replicate: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            index=index,
            model="iid",
            initial=initial,
            prior=prior,
            group=group,
            group_model=group_model,
            replicate=replicate,
            kwargs=kwargs,
        )


@dataclass
class Besag(Effect):
    """Intrinsic Autoregressive / Besag spatial random effect."""

    def __init__(
        self,
        index: str,
        *,
        graph: Any = None,
        scale_model: Optional[bool] = None,
        initial: Any = None,
        weights: Any = None,
        group: Optional[str] = None,
        group_model: Optional[str] = None,
        replicate: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            index=index,
            model="besag",
            graph=graph,
            scale_model=scale_model,
            initial=initial,
            weights=weights,
            group=group,
            group_model=group_model,
            replicate=replicate,
            kwargs=kwargs,
        )


@dataclass
class BYM(Effect):
    """Besag-York-Mollié (BYM) combined spatial CAR and IID random effect."""

    def __init__(
        self,
        index: str,
        *,
        graph: Any = None,
        scale_model: Optional[bool] = None,
        initial: Any = None,
        **kwargs: Any,
    ):
        super().__init__(
            index=index,
            model="bym",
            graph=graph,
            scale_model=scale_model,
            initial=initial,
            kwargs=kwargs,
        )


@dataclass
class BYM2(Effect):
    """Scaled Besag-York-Mollié 2 (BYM2) reparameterized spatial random effect."""

    def __init__(
        self,
        index: str,
        *,
        graph: Any = None,
        scale_model: Optional[bool] = None,
        initial: Any = None,
        **kwargs: Any,
    ):
        super().__init__(
            index=index,
            model="bym2",
            graph=graph,
            scale_model=scale_model,
            initial=initial,
            kwargs=kwargs,
        )


@dataclass
class RW1(Effect):
    """First-order Random Walk (RW1) latent model."""

    def __init__(
        self,
        index: str,
        *,
        cyclic: bool = False,
        scale_model: Optional[bool] = None,
        initial: Any = None,
        **kwargs: Any,
    ):
        super().__init__(
            index=index,
            model="rw1",
            cyclic=cyclic,
            scale_model=scale_model,
            initial=initial,
            kwargs=kwargs,
        )


@dataclass
class RW2(Effect):
    """Second-order Random Walk (RW2) latent model."""

    def __init__(
        self,
        index: str,
        *,
        cyclic: bool = False,
        scale_model: Optional[bool] = None,
        initial: Any = None,
        **kwargs: Any,
    ):
        super().__init__(
            index=index,
            model="rw2",
            cyclic=cyclic,
            scale_model=scale_model,
            initial=initial,
            kwargs=kwargs,
        )


@dataclass
class AR1(Effect):
    """Autoregressive process of order 1 (AR1)."""

    def __init__(
        self,
        index: str,
        *,
        initial: Any = None,
        group: Optional[str] = None,
        replicate: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            index=index,
            model="ar1",
            initial=initial,
            group=group,
            replicate=replicate,
            kwargs=kwargs,
        )


@dataclass
class AR(Effect):
    """Autoregressive process of order p (AR(p))."""

    def __init__(
        self,
        index: str,
        *,
        order: int = 1,
        initial: Any = None,
        **kwargs: Any,
    ):
        super().__init__(
            index=index,
            model="ar",
            order=order,
            initial=initial,
            kwargs=kwargs,
        )


@dataclass
class SPDE(Effect):
    """Spatial SPDE Matérn latent model."""

    def __init__(
        self,
        index: str,
        *,
        spde_model: Any = None,
        initial: Any = None,
        group: Optional[str] = None,
        replicate: Optional[str] = None,
        **kwargs: Any,
    ):
        kw = dict(kwargs)
        if spde_model is not None:
            kw["spde_model"] = spde_model
        super().__init__(
            index=index,
            model="spde",
            initial=initial,
            group=group,
            replicate=replicate,
            kwargs=kw,
        )


@dataclass
class Generic(Effect):
    """Custom / generic user-defined latent GMRF model (rgeneric / cgeneric)."""

    def __init__(
        self,
        index: str,
        *,
        model: Any = "rgeneric",
        initial: Any = None,
        **kwargs: Any,
    ):
        model_name = getattr(model, "name", str(model))
        kw = dict(kwargs)
        if hasattr(model, "precision") or hasattr(model, "Q") or hasattr(model, "as_generic"):
            kw["generic_instance"] = model
        super().__init__(
            index=index,
            model=model_name,
            initial=initial,
            kwargs=kw,
        )


@dataclass
class CRW1(Effect):
    """Continuous Random Walk 1 latent model."""

    def __init__(self, index: str, **kwargs: Any):
        super().__init__(index=index, model="crw1", kwargs=kwargs)


@dataclass
class CRW2(Effect):
    """Continuous Random Walk 2 latent model."""

    def __init__(self, index: str, **kwargs: Any):
        super().__init__(index=index, model="crw2", kwargs=kwargs)


@dataclass
class FGN(Effect):
    """Fractional Gaussian Noise (FGN) latent model."""

    def __init__(self, index: str, **kwargs: Any):
        super().__init__(index=index, model="fgn", kwargs=kwargs)


@dataclass
class Seasonal(Effect):
    """Seasonal latent model."""

    def __init__(self, index: str, *, season_length: int = 12, **kwargs: Any):
        kw = dict(kwargs)
        kw["season_length"] = int(season_length)
        super().__init__(index=index, model="seasonal", kwargs=kw)


@dataclass
class Linear:
    """Fixed-effect linear covariate term."""

    name: str


@dataclass
class Intercept:
    """Explicit intercept configuration."""

    enabled: bool = True


# =============================================================================
# Declarative ModelSpec Base Class
# =============================================================================


class ModelSpec:
    """Declarative specification for an INLA model.

    Can be used by subclassing or by instantiating directly.

    Example
    -------
    >>> class DiseaseMapping(ModelSpec):
    ...     response = "successes"
    ...     family = Binomial(Ntrials="n_trials")
    ...     fixed = ["covariate_x"]
    ...     spatial = Besag("spatial_idx", graph=adj_matrix, scale_model=True)
    ...     temporal = RW2("time_idx", cyclic=True)
    ...
    >>> result = inla.fit(DiseaseMapping, data=df)
    """

    response: Optional[str] = None
    family: Union[str, Family] = "gaussian"
    intercept: bool = True
    fixed: Optional[list[Union[str, Linear]]] = None
    fixed_effects: Optional[list[Union[str, Linear]]] = None
    random: Optional[list[Effect]] = None
    random_effects: Optional[list[Effect]] = None
    offset: Optional[str] = None
    Ntrials: Any = None
    E: Any = None

    def __init__(
        self,
        *,
        response: Optional[str] = None,
        family: Optional[Union[str, Family]] = None,
        intercept: Optional[bool] = None,
        fixed: Optional[Sequence[Union[str, Linear]]] = None,
        fixed_effects: Optional[Sequence[Union[str, Linear]]] = None,
        random: Optional[Sequence[Effect]] = None,
        random_effects: Optional[Sequence[Effect]] = None,
        offset: Optional[str] = None,
        Ntrials: Any = None,
        E: Any = None,
        **kwargs: Any,
    ):
        if response is not None:
            self.response = response
        if family is not None:
            self.family = family
        if intercept is not None:
            self.intercept = intercept
        if fixed is not None:
            self.fixed = list(fixed)
        if fixed_effects is not None:
            self.fixed_effects = list(fixed_effects)
        if random is not None:
            self.random = list(random)
        if random_effects is not None:
            self.random_effects = list(random_effects)
        if offset is not None:
            self.offset = offset
        if Ntrials is not None:
            self.Ntrials = Ntrials
        if E is not None:
            self.E = E

        # Store any additional effect attributes passed in constructor
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def compile_spec(cls_or_self: Union[type[ModelSpec], ModelSpec]) -> tuple[ParsedFormula, dict[str, Any]]:
        """Compile class or instance into a ParsedFormula and fitting kwargs."""
        inst = cls_or_self() if isinstance(cls_or_self, type) else cls_or_self

        # 1. Resolve response
        response = inst.response
        if not response:
            raise ValueError("ModelSpec must define 'response'")

        # 2. Resolve intercept
        intercept = bool(inst.intercept)

        # 3. Resolve fixed effects
        raw_fixed = inst.fixed if inst.fixed is not None else inst.fixed_effects
        fixed_terms: list[str] = []
        if raw_fixed:
            for item in raw_fixed:
                if isinstance(item, Linear):
                    fixed_terms.append(item.name)
                elif isinstance(item, str):
                    fixed_terms.append(item)
                else:
                    raise TypeError(f"expected str or Linear for fixed effect, got {type(item)}")

        # 4. Resolve random effects (from .random / .random_effects AND any attribute that is an Effect)
        f_terms: list[FTerm] = []
        seen_effects: set[int] = set()

        raw_random = inst.random if inst.random is not None else inst.random_effects
        if raw_random:
            for eff in raw_random:
                if isinstance(eff, Effect):
                    f_terms.append(eff.to_fterm())
                    seen_effects.add(id(eff))
                else:
                    raise TypeError(f"expected Effect instance in random effects, got {type(eff)}")

        # Scan class / instance attributes for assigned Effect objects
        for name in dir(inst):
            if name.startswith("_"):
                continue
            if name in ("random", "random_effects", "fixed", "fixed_effects", "family", "response"):
                continue
            val = getattr(inst, name, None)
            if isinstance(val, Effect) and id(val) not in seen_effects:
                f_terms.append(val.to_fterm())
                seen_effects.add(id(val))

        parsed = ParsedFormula(
            response=response,
            fixed_terms=fixed_terms,
            intercept=intercept,
            f_terms=f_terms,
        )

        # 5. Extra kwargs (family, Ntrials, E, etc.)
        fit_kwargs: dict[str, Any] = {}
        if isinstance(inst.family, Family):
            fam = inst.family
            fit_kwargs["family"] = fam.name
            if fam.Ntrials is not None:
                fit_kwargs["Ntrials"] = fam.Ntrials
            if fam.E is not None:
                fit_kwargs["E"] = fam.E
            if fam.event is not None:
                fit_kwargs["event"] = fam.event
            if fam.control_family is not None:
                fit_kwargs["control_family"] = fam.control_family
            fit_kwargs["size"] = fam.size
            fit_kwargs["zero_prob"] = fam.zero_prob
            fit_kwargs["inflation"] = fam.inflation
            fit_kwargs["alpha"] = fam.alpha
            fit_kwargs["gamma"] = fam.gamma
            fit_kwargs["shape"] = fam.shape
        elif isinstance(inst.family, str):
            fit_kwargs["family"] = inst.family
        else:
            raise TypeError(f"expected str or Family for family, got {type(inst.family)}")

        if inst.Ntrials is not None:
            fit_kwargs["Ntrials"] = inst.Ntrials
        if inst.E is not None:
            fit_kwargs["E"] = inst.E
        if inst.offset is not None:
            fit_kwargs["offset"] = inst.offset

        return parsed, fit_kwargs
