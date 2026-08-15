"""High-level R-INLA-like `inla()` front-end for Python."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
from scipy import sparse

from inla import _native as core
from inla import generic as generic_mod
from inla.formula import FTerm, ParsedFormula, parse_formula
from inla.generic import GenericModel, Model
from inla.models import Effect, Family, Linear, ModelSpec

SUPPORTED_F_MODELS = (
    "iid",
    "rw1",
    "rw2",
    "rw2d",
    "ar1",
    "ar",
    "arp",
    "besag",
    "bym",
    "bym2",
    "fgn",
    "seasonal",
    "crw1",
    "crw2",
    "matern2d",
    "spde",
)
GENERIC_MODEL_ALIASES = ("rgeneric", "generic", "cgeneric")

FAMILY_ALIASES = {
    "cbinomial": "binomial",
    "nbinomial": "negative_binomial",
    "negbin": "negative_binomial",
}

GenericLike = Union[GenericModel, Model]


def _as_1d(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    return arr


def _get_col(data: Mapping[str, Any], key: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"column '{key}' not found in data")
    return _as_1d(data[key], key)


def _adj_from_matrix(mat) -> list[list[int]]:
    if sparse.issparse(mat):
        mat = mat.tocsr()
        if mat.shape[0] != mat.shape[1]:
            raise ValueError("adjacency matrix must be square")
        n = mat.shape[0]
        out: list[list[int]] = []
        for i in range(n):
            row = mat.indices[mat.indptr[i] : mat.indptr[i + 1]]
            nbs = [int(j) for j in row if int(j) != i and mat[i, j] != 0]
            out.append(nbs)
        return out
    if hasattr(mat, "adjacency"):
        return [[int(nbr) for nbr in nbrs] for _, nbrs in mat.adjacency()]
    a = np.asarray(mat)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("adjacency matrix must be square")
    n = a.shape[0]
    out: list[list[int]] = []
    for i in range(n):
        nbs = [int(j) for j in np.flatnonzero(a[i] != 0) if int(j) != i]
        out.append(nbs)
    return out


def _resolve_graph(f_term, data: Mapping[str, Any]) -> list[list[int]]:
    g = f_term.graph
    if g is None:
        if "adj_matrix" in data:
            return _adj_from_matrix(data["adj_matrix"])
        if "adj_list" in data:
            raw = data["adj_list"]
            return [[int(j) for j in row] for row in raw]
        raise ValueError(
            f"f({f_term.index}, model='besag') requires graph=... or data['adj_matrix']"
        )
    if isinstance(g, str):
        if g not in data:
            raise KeyError(f"graph='{g}' not found in data")
        g = data[g]
    if isinstance(g, (list, tuple)) and (len(g) == 0 or isinstance(g[0], (list, tuple))):
        return [[int(j) for j in row] for row in g]
    return _adj_from_matrix(g)


@lru_cache(maxsize=None)
def _model_meta(
    model: str, order: int = 0, group_model: Optional[str] = None, cyclic: bool = False
) -> Mapping[str, Any]:
    """Per-model metadata from the shared Rust registry (cached; called per θ node)."""
    return core.model_metadata(
        model.lower(), order=int(order), group_model=group_model, cyclic=bool(cyclic)
    )


def _theta_len(model: str, order: int = 0, group_model: Optional[str] = None) -> int:
    return int(_model_meta(model, order, group_model)["theta_len"])


def _default_theta(
    model: str, order: int = 0, group_model: Optional[str] = None
) -> list[float]:
    return list(_model_meta(model, order, group_model)["default_theta"])


def _rank_deficiency(model: str, cyclic: bool = False) -> int:
    return int(_model_meta(model, 0, None, cyclic)["rank_deficiency"])


def _hyper_labels(
    types: Sequence[str],
    names: Sequence[str],
    orders: Sequence[int],
    group_models: Sequence[Optional[str]],
) -> tuple[list[str], list[str]]:
    """(labels, transform tags) per internal θ, in optimizer order."""
    labels: list[str] = []
    transforms: list[str] = []
    for typ, nm, order, gm in zip(types, names, orders, group_models):
        if typ == "fixed":
            continue
        meta = _model_meta(typ, order, gm)
        for lab, tr in zip(meta["hyper_labels"], meta["hyper_transforms"]):
            labels.append(f"{lab} for {nm}")
            transforms.append(tr)
    return labels, transforms


def _to_natural(tag: str, theta: float) -> float:
    if tag == "exp":
        return float(np.exp(theta))
    if tag == "rho":
        return float(2.0 / (1.0 + np.exp(-theta)) - 1.0)
    if tag == "phi":
        return float(1.0 / (1.0 + np.exp(-theta)))
    return float(theta)


def _natural_sd(tag: str, theta_mean: float, theta_sd: float) -> float:
    """Delta-method sd matching Rust `HyperTransformKind::natural_sd`."""
    if not np.isfinite(theta_sd):
        return float("nan")
    if tag == "exp":
        return float(np.exp(theta_mean) * theta_sd)
    if tag == "rho":
        r = _to_natural(tag, theta_mean)
        return float(0.5 * (1.0 - r * r) * theta_sd)
    if tag == "phi":
        p = _to_natural(tag, theta_mean)
        return float(p * (1.0 - p) * theta_sd)
    return float(theta_sd)


def _group_model_from_ft(ft) -> Optional[str]:
    """Extract control.group model name from f() kwargs, if present."""
    cg = ft.kwargs.get("control_group")
    if cg is None:
        cg = ft.kwargs.get("control.group")
    if cg is None:
        return None
    if isinstance(cg, dict):
        return str(cg.get("model", "ar1")).lower()
    if isinstance(cg, str):
        return cg.lower()
    raise ValueError("control_group must be a dict with model=... or a model name string")


def _build_group_q(n_group: int, group_model: str, theta_g: list[float]):
    g = group_model.lower()
    if g == "iid":
        tau = float(np.exp(theta_g[0])) if theta_g else 1.0
        return core.iid_precision_matrix(n_group, tau)
    if g == "rw1":
        tau = float(np.exp(theta_g[0])) if theta_g else 1.0
        return core.rw1_precision_matrix(n_group, tau)
    if g == "rw2":
        tau = float(np.exp(theta_g[0])) if theta_g else 1.0
        return core.rw2_precision_matrix(n_group, tau)
    if g == "ar1":
        if len(theta_g) < 2:
            raise ValueError("group ar1 needs [log_tau, logit_rho]")
        tau = float(np.exp(theta_g[0]))
        rho = 2.0 / (1.0 + np.exp(-theta_g[1])) - 1.0
        rho = float(np.clip(rho, -0.999, 0.999))
        return core.ar1_precision_matrix_csc(n_group, rho, tau)
    raise ValueError(f"unsupported group model '{group_model}'")


def _as_scipy_csc(q):
    if isinstance(q, core.PyCscMatrix):
        return q.to_scipy().tocsc().copy()
    return sparse.csc_matrix(q).tocsc()


def _sum_to_zero_a(n: int, k: int) -> tuple[list[float], list[float]]:
    """Orthonormal sum-to-zero rows matching Rust ``sum_to_zero_constraint``."""
    if n <= 0 or k not in (1, 2):
        raise ValueError(f"sum_to_zero requires n>0 and k in {{1,2}}, got n={n} k={k}")
    a = [0.0] * (k * n)
    inv = 1.0 / float(n) ** 0.5
    for c in range(n):
        a[c] = inv
    if k == 2:
        mean = (n - 1) / 2.0
        ss = 0.0
        for c in range(n):
            v = c - mean
            a[n + c] = v
            ss += v * v
        scale = ss**0.5
        for c in range(n):
            a[n + c] /= scale
    return a, [0.0] * k


def _embed_constraint(
    block_a: list[float], block_e: list[float], block_n: int, full_n: int, offset: int
) -> tuple[list[float], list[float]]:
    k = len(block_e)
    a = [0.0] * (k * full_n)
    for r in range(k):
        for c in range(block_n):
            a[r * full_n + (offset + c)] = block_a[r * block_n + c]
    return a, list(block_e)


def _vstack_constraints(
    parts: list[tuple[list[float], list[float]]],
) -> Optional[tuple[list[float], list[float]]]:
    if not parts:
        return None
    a_all: list[float] = []
    e_all: list[float] = []
    for a, e in parts:
        a_all.extend(a)
        e_all.extend(e)
    return a_all, e_all


def _as_param_list(val) -> list[float]:
    if val is None:
        return []
    return [float(x) for x in np.asarray(val, dtype=float).reshape(-1)]


def _resolve_effect_priors(model: str, kwargs: Optional[Mapping[str, Any]]) -> list[tuple[str, list[float]]]:
    """Build (name, param) list for an effect from f() kwargs or model defaults."""
    kw = dict(kwargs or {})
    # Flat f(..., prior=..., param=...) → first hyper slot
    if "prior" in kw:
        name = str(kw["prior"])
        param = _as_param_list(kw.get("param"))
        defaults = core.default_hyper_priors(model)
        if not defaults:
            return [(name, param)]
        out = [(name, param)]
        # keep remaining default slots (e.g. AR1 rho) if only prec overridden
        for i, (dn, dp) in enumerate(defaults):
            if i == 0:
                continue
            out.append((dn, dp))
        return out
    hyper = kw.get("hyper")
    if isinstance(hyper, Mapping):
        defaults = core.default_hyper_priors(model)
        # Map R-style keys onto default slots by order: prec, rho/rho1, ...
        key_order = []
        for k in ("prec", "rho", "rho1", "cor1", "h", "hurst"):
            if k in hyper:
                key_order.append(k)
        for k in hyper:
            if k not in key_order:
                key_order.append(k)
        out: list[tuple[str, list[float]]] = []
        for i, k in enumerate(key_order):
            h = hyper[k]
            if not isinstance(h, Mapping):
                raise ValueError(f"hyper['{k}'] must be a mapping with prior=/param=")
            name = str(h.get("prior") or (defaults[i][0] if i < len(defaults) else "gaussian"))
            param = _as_param_list(h.get("param"))
            out.append((name, param))
        # If fewer keys than default slots, append remaining defaults
        if len(out) < len(defaults):
            out.extend(defaults[len(out) :])
        return out
    return list(core.default_hyper_priors(model))


def _control_get(cc: Mapping[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in cc:
            return cc[k]
        # R-style dotted aliases
        dotted = k.replace("_", ".")
        if dotted in cc:
            return cc[dotted]
    return None


def _resolve_controls(
    control_compute: Optional[Mapping[str, Any]],
    *,
    strategy: str,
    step_or_f0: float,
    fixed_prec: float,
    deterministic: bool,
) -> dict[str, Any]:
    """Validate + default engine controls in Rust, so R and Python agree on names."""
    bag: dict[str, Any] = {
        "strategy": str(strategy),
        "step_or_f0": float(step_or_f0),
        "fixed_prec": float(fixed_prec),
        "deterministic": bool(deterministic),
    }
    for key, value in (control_compute or {}).items():
        if value is None:
            continue
        if isinstance(value, (bool, np.bool_)):
            bag[key] = bool(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            bag[key] = float(value)
        elif isinstance(value, str):
            bag[key] = value
        else:
            bag[key] = [float(v) for v in np.asarray(value).reshape(-1)]
    return dict(core.resolve_compute_options(bag))


def _resolve_marginal_indices(
    value: Any, n: int, *, name: str
) -> Optional[list[int]]:
    if value is None:
        return None
    if value is True:
        return list(range(n))
    if value is False:
        return None
    idx = [int(i) for i in value]
    for i in idx:
        if i < 0 or i >= n:
            raise ValueError(f"{name}: index {i} out of range [0, {n})")
    return idx


def _resolve_f_model(
    ft,
    *,
    models: Optional[Mapping[str, GenericLike]],
    rgeneric: Optional[GenericLike],
) -> Optional[GenericLike]:
    """Return a GenericLike if this f() term is a custom model, else None."""
    if ft.kwargs.get("generic_instance") is not None:
        inst = ft.kwargs["generic_instance"]
        if hasattr(inst, "as_generic"):
            return inst.as_generic()
        return inst
    key = str(ft.model).lower()
    if key in SUPPORTED_F_MODELS:
        return None
    models = models or {}
    if key in GENERIC_MODEL_ALIASES:
        if rgeneric is not None:
            return rgeneric
        if len(models) == 1:
            return next(iter(models.values()))
        if ft.model in models:
            return models[ft.model]
        raise ValueError(
            "f(..., model='rgeneric') requires rgeneric=... or a single entry in models="
        )
    if key in models:
        return models[key]
    # Also allow exact (case-sensitive) lookup for user-chosen names
    if ft.model in models:
        return models[ft.model]
    supported = ", ".join(SUPPORTED_F_MODELS + GENERIC_MODEL_ALIASES)
    raise ValueError(
        f"unsupported f() model '{ft.model}'. Built-ins: {supported}. "
        f"Or pass models={{'{ft.model}': ...}} / inla.generic.define(...)."
    )


def _compute_scale_factor(q_scipy) -> float:
    n = q_scipy.shape[0]
    if n <= 1:
        return 1.0
    a = q_scipy.toarray()
    evals, evecs = np.linalg.eigh(a)
    max_lam = max(np.max(np.abs(evals)), 1.0)
    tol = np.sqrt(np.finfo(float).eps) * max_lam
    
    valid = evals > tol
    if not np.any(valid):
        raise ValueError("scale_model: ginv has no positive diagonal")
    
    evals_valid = evals[valid]
    evecs_valid = evecs[:, valid]
    
    ginv_diag = np.sum((evecs_valid ** 2) / evals_valid, axis=1)
    
    valid_diag = (ginv_diag > 0.0) & np.isfinite(ginv_diag)
    if not np.any(valid_diag):
        raise ValueError("scale_model: ginv has no positive diagonal")
        
    log_sum = np.sum(np.log(ginv_diag[valid_diag]))
    count = np.sum(valid_diag)
    
    fac = np.exp(log_sum / count)
    return float(fac)



def _build_obs(
    family: str,
    y: np.ndarray,
    *,
    E=None,
    Ntrials=None,
    event=None,
    size: float = 1.0,
    zero_prob: float = 0.1,
    inflation: str = "type0",
    alpha: float = 0.5,
    gamma: float = 1.0,
    shape: float = 1.0,
    obs_precision: float = 1.0,
) -> list[dict]:
    fam = FAMILY_ALIASES.get(family.lower(), family.lower())
    n = y.size
    obs: list[dict] = []

    def _opt_arr(v, default=None):
        if v is None:
            return None
        a = np.asarray(v, dtype=float).reshape(-1)
        if a.size == 1:
            return np.full(n, float(a[0]))
        if a.size != n:
            raise ValueError(f"length mismatch vs y ({n})")
        return a

    # cbind(k, n) parity for binomial-like families
    if Ntrials is not None:
        nt = np.asarray(Ntrials, dtype=float)
        if nt.ndim == 2 and nt.shape[1] == 2:
            y = nt[:, 0].copy()
            ntrials = nt[:, 1].copy()
            n = y.size
        else:
            ntrials = _opt_arr(Ntrials)
    else:
        ntrials = None

    E_arr = _opt_arr(E)
    event_arr = _opt_arr(event)

    for i in range(n):
        d: dict[str, Any] = {"family": fam, "y": float(y[i])}
        if fam == "gaussian":
            d["precision"] = float(obs_precision)
        elif fam == "poisson":
            d["E"] = float(E_arr[i]) if E_arr is not None else 1.0
        elif fam == "binomial":
            if ntrials is None:
                raise ValueError("binomial/cbinomial requires Ntrials")
            d["n"] = float(ntrials[i])
        elif fam in ("negative_binomial", "nbinomial"):
            d["exposure"] = float(E_arr[i]) if E_arr is not None else 1.0
            d["size"] = float(size)
        elif fam in ("laplace",):
            d["alpha"] = float(alpha)
            d["gamma"] = float(gamma)
        elif fam in ("zero_inflated_poisson", "zeroinflatedpoisson0", "zeroinflatedpoisson1", "zip"):
            d["exposure"] = float(E_arr[i]) if E_arr is not None else 1.0
            d["zero_prob"] = float(zero_prob)
            d["inflation"] = str(inflation)
        elif fam in ("zero_inflated_binomial", "zeroinflatedbinomial0", "zeroinflatedbinomial1", "zib"):
            if ntrials is None:
                raise ValueError("zero_inflated_binomial requires Ntrials")
            d["n"] = float(ntrials[i])
            d["zero_prob"] = float(zero_prob)
            d["inflation"] = str(inflation)
        elif fam in ("exponential", "exponential_survival"):
            d["event"] = float(event_arr[i]) if event_arr is not None else 1.0
        elif fam in ("weibull", "weibull_survival"):
            d["event"] = float(event_arr[i]) if event_arr is not None else 1.0
            d["shape"] = float(shape)
        else:
            raise ValueError(f"unsupported family '{family}'")
        obs.append(d)
    return obs


def _design_matrix(parsed: ParsedFormula, data: Mapping[str, Any], n_obs: int) -> np.ndarray:
    cols = []
    names = []
    if parsed.intercept:
        cols.append(np.ones(n_obs))
        names.append("(Intercept)")
    for name in parsed.fixed_terms:
        cols.append(_get_col(data, name))
        names.append(name)
    if not cols:
        return np.zeros((n_obs, 0))
    x = np.column_stack(cols)
    return x


def _identity_ar1_index(idx: np.ndarray, n_obs: int) -> bool:
    """True when idx is 0..n-1 or 1..n (so η = x with no remapping)."""
    if idx.shape != (n_obs,):
        return False
    a0 = np.arange(n_obs, dtype=int)
    a1 = np.arange(1, n_obs + 1, dtype=int)
    return bool(np.array_equal(idx, a0) or np.array_equal(idx, a1))


def _try_gaussian_ar1_plan(
    *,
    formula: str,
    parsed,
    data: Mapping[str, Any],
    family: str,
    y: np.ndarray,
    n_obs: int,
    obs_precision: float,
    strategy: str,
    step_or_f0: float,
    initial_theta: Optional[Sequence[float]],
    control_compute: Optional[Mapping[str, Any]],
    latent_marginal_indices: Optional[Sequence[int]],
    predictor_marginal_indices: Optional[Sequence[int]],
    verbose: bool,
) -> Optional[Any]:
    """Fast path: single AR1 + Gaussian, no fixed effects, identity index → ModelPlan."""
    fam = str(family).lower()
    if fam not in ("gaussian", "normal"):
        return None
    if parsed.intercept or parsed.fixed_terms:
        return None
    if len(parsed.f_terms) != 1:
        return None
    ft = parsed.f_terms[0]
    if str(ft.model).lower() != "ar1":
        return None
    if ft.kwargs.get("group") is not None:
        return None
    # Opt-in full latent/predictor marginal grids still use the generic path.
    if latent_marginal_indices is not None or predictor_marginal_indices is not None:
        return None
    if control_compute:
        for k in (
            "return_marginals_latent",
            "return.marginals.random",
            "return_marginals_predictor",
            "return.marginals.predictor",
        ):
            if control_compute.get(k):
                return None
    idx = _get_col(data, ft.index).astype(int)
    if not _identity_ar1_index(idx, n_obs):
        return None
    init = None if initial_theta is None else list(np.asarray(initial_theta, dtype=float).reshape(-1))
    if verbose:
        print(f"inla: ModelPlan path (gaussian+ar1) n={n_obs}")
    result = core.run_gaussian_ar1_plan(
        y=list(np.asarray(y, dtype=float).reshape(-1)),
        name=str(ft.index),
        obs_precision=float(obs_precision),
        strategy=strategy,
        step_or_f0=float(step_or_f0),
        initial_theta=init,
    )
    means = np.asarray(result.latent_means, dtype=float)
    sds = np.sqrt(np.maximum(np.asarray(result.latent_variances, dtype=float), 0.0))
    out = InlaResult(result)
    out.summary_random = {
        str(ft.index): {
            "mean": means,
            "sd": sds,
            "0.025quant": means - 1.96 * sds,
            "0.5quant": means,
            "0.975quant": means + 1.96 * sds,
        }
    }
    out.summary_fixed = None
    labels, transforms = _hyper_labels(["ar1"], [str(ft.index)], [0], [None])
    out.summary_hyperpar_internal = _internal_hyperpar_table(result)
    out.summary_hyperpar = _natural_hyperpar_table(
        out.summary_hyperpar_internal, labels, transforms
    )
    out.effects = {"names": [str(ft.index)], "types": ["ar1"], "ns": [n_obs]}
    out.formula = formula
    if verbose:
        print(f"inla: mlik={out.marginal_log_lik:.4f} dic={out.dic:.4f}")
    return out


def _fit(
    formula: Optional[Union[str, ModelSpec, type[ModelSpec]]] = None,
    data: Optional[Any] = None,
    family: Union[str, Family] = "gaussian",
    *,
    response: Optional[str] = None,
    fixed: Optional[Sequence[Union[str, Linear]]] = None,
    fixed_effects: Optional[Sequence[Union[str, Linear]]] = None,
    random: Optional[Sequence[Union[Effect, FTerm]]] = None,
    random_effects: Optional[Sequence[Union[Effect, FTerm]]] = None,
    intercept: bool = True,
    offset: Optional[str] = None,
    E=None,
    Ntrials=None,
    event=None,
    size: float = 1.0,
    zero_prob: float = 0.1,
    inflation: str = "type0",
    alpha: float = 0.5,
    gamma: float = 1.0,
    shape: float = 1.0,
    strategy: str = "ccd",
    step_or_f0: float = 1.0,
    initial_theta: Optional[Sequence[float]] = None,
    control_family: Optional[Mapping[str, Any]] = None,
    control_compute: Optional[Mapping[str, Any]] = None,
    fixed_prec: float = 1e-4,
    latent_marginal_indices: Optional[Sequence[int]] = None,
    predictor_marginal_indices: Optional[Sequence[int]] = None,
    deterministic: bool = False,
    models: Optional[Mapping[str, GenericLike]] = None,
    rgeneric: Optional[GenericLike] = None,
    verbose: bool = False,
):
    """Fit an INLA model with formula syntax, ModelSpec class, or functional kwargs.

    Parameters
    ----------
    formula :
        e.g. ``"y ~ x + f(region, model='besag')"`` or a ``ModelSpec`` class / instance.
    data :
        Mapping of column name → array or DataFrame.
        For Besag, include ``adj_matrix`` (square) or pass ``graph=...`` in ``f()`` / ``Besag()``.
    family :
        Likelihood family name (str) or typed ``Family`` instance (e.g. ``Binomial()``).
    response :
        Response variable name (when not using formula string).
    fixed / fixed_effects :
        Sequence of fixed-effect column names (or ``Linear("x")`` instances).
    random / random_effects :
        Sequence of latent random effect terms (e.g. ``[Besag("idx", graph=adj)]``).
    intercept :
        Whether to include an intercept term (default: True).
    Ntrials :
        Binomial trials ``n``, or ``cbind``-style ``(n_obs, 2)`` array of ``(y, n)``.
    control_compute :
        Opt-in compute flags. Recognized keys (underscore or R-style dots)::

            return_marginals_latent / return.marginals.random
                True → all latent indices; or a sequence of 0-based indices.
            return_marginals_predictor / return.marginals.predictor
                True → all predictor indices; or a sequence of 0-based indices.
    deterministic :
        If True, evaluate CCD/grid nodes sequentially (reproducible ordering).
    models :
        Mapping of name → ``GenericModel`` / ``Model`` for ``f(..., model='name')``.
    rgeneric :
        Single custom model for ``f(..., model='rgeneric')`` (R-style).
    """
    # Handle data passed as first positional arg when response is passed
    if (
        data is None
        and formula is not None
        and not isinstance(formula, str)
        and not (isinstance(formula, type) and issubclass(formula, ModelSpec))
        and not isinstance(formula, ModelSpec)
        and (response is not None or hasattr(formula, "to_dict") or hasattr(formula, "__getitem__"))
    ):
        data = formula
        formula = None

    if data is not None and hasattr(data, "to_dict"):
        try:
            data = data.to_dict(orient="series")
        except TypeError:
            data = data.to_dict()

    formula_str = None
    if (isinstance(formula, type) and issubclass(formula, ModelSpec)) or isinstance(
        formula, ModelSpec
    ):
        parsed, spec_kw = ModelSpec.compile_spec(formula)
        if "family" in spec_kw and family == "gaussian":
            family = spec_kw["family"]
        if "Ntrials" in spec_kw and Ntrials is None:
            Ntrials = spec_kw["Ntrials"]
        if "E" in spec_kw and E is None:
            E = spec_kw["E"]
        if "event" in spec_kw and event is None:
            event = spec_kw["event"]
        if "control_family" in spec_kw and control_family is None:
            control_family = spec_kw["control_family"]
        if "size" in spec_kw:
            size = spec_kw["size"]
        if "zero_prob" in spec_kw:
            zero_prob = spec_kw["zero_prob"]
        if "inflation" in spec_kw:
            inflation = spec_kw["inflation"]
        if "alpha" in spec_kw:
            alpha = spec_kw["alpha"]
        if "gamma" in spec_kw:
            gamma = spec_kw["gamma"]
        if "shape" in spec_kw:
            shape = spec_kw["shape"]
        formula_str = f"{parsed.response} ~ {' + '.join(parsed.fixed_terms) or '1'}"
    elif formula is None or response is not None:
        if response is None:
            raise ValueError(
                "Must provide a formula string, a ModelSpec class/instance, or 'response=' with data."
            )
        fixed_terms: list[str] = []
        raw_fixed = fixed if fixed is not None else fixed_effects
        if raw_fixed:
            for item in raw_fixed:
                if isinstance(item, Linear):
                    fixed_terms.append(item.name)
                elif isinstance(item, str):
                    fixed_terms.append(item)
                else:
                    raise TypeError(
                        f"expected str or Linear for fixed effect, got {type(item)}"
                    )

        f_terms: list[FTerm] = []
        raw_random = random if random is not None else random_effects
        if raw_random:
            for eff in raw_random:
                if isinstance(eff, Effect):
                    f_terms.append(eff.to_fterm())
                elif isinstance(eff, FTerm):
                    f_terms.append(eff)
                else:
                    raise TypeError(
                        f"expected Effect instance in random effects, got {type(eff)}"
                    )

        parsed = ParsedFormula(
            response=response,
            fixed_terms=fixed_terms,
            intercept=bool(intercept),
            f_terms=f_terms,
        )
        formula_str = f"{response} ~ {' + '.join(fixed_terms) or ('1' if intercept else '-1')}"
    elif isinstance(formula, str):
        parsed = parse_formula(formula)
        raw_random = random if random is not None else random_effects
        if raw_random:
            for eff in raw_random:
                if isinstance(eff, Effect):
                    parsed.f_terms.append(eff.to_fterm())
                elif isinstance(eff, FTerm):
                    parsed.f_terms.append(eff)
        formula_str = formula
    else:
        raise TypeError(
            f"Unsupported formula type: {type(formula)}. Pass str, ModelSpec, or response=..."
        )

    if isinstance(family, Family):
        if family.Ntrials is not None and Ntrials is None:
            Ntrials = family.Ntrials
        if family.E is not None and E is None:
            E = family.E
        if family.event is not None and event is None:
            event = family.event
        if family.control_family is not None and control_family is None:
            control_family = family.control_family
        size = family.size
        zero_prob = family.zero_prob
        inflation = family.inflation
        alpha = family.alpha
        gamma = family.gamma
        shape = family.shape
        family = family.name

    if data is None:
        raise ValueError("data must be provided")

    formula = formula_str
    resolved_generics: list[Optional[GenericLike]] = []
    for ft in parsed.f_terms:
        g = _resolve_f_model(ft, models=models, rgeneric=rgeneric)
        resolved_generics.append(g)
        if g is None and ft.model not in SUPPORTED_F_MODELS:
            raise ValueError(f"unsupported f() model '{ft.model}'")

    y = _get_col(data, parsed.response)
    n_obs = y.size

    if isinstance(Ntrials, str):
        Ntrials = _get_col(data, Ntrials)
    if isinstance(E, str):
        E = _get_col(data, E)
    if isinstance(event, str):
        event = _get_col(data, event)

    obs_precision = 1.0
    if control_family is not None:
        try:
            obs_precision = float(np.exp(control_family["hyper"]["prec"]["initial"]))
        except Exception:
            pass
    # `f(..., obs_precision=)` overrides, matching the R front-end.
    for ft in parsed.f_terms:
        if ft.kwargs.get("obs_precision") is not None:
            obs_precision = float(ft.kwargs["obs_precision"])

    controls = _resolve_controls(
        control_compute,
        strategy=strategy,
        step_or_f0=step_or_f0,
        fixed_prec=fixed_prec,
        deterministic=deterministic,
    )
    strategy = controls["strategy"]
    step_or_f0 = controls["step_or_f0"]
    fixed_prec = controls["fixed_prec"]
    deterministic = controls["deterministic"]

    if all(g is None for g in resolved_generics):
        planned = _try_gaussian_ar1_plan(
            formula=formula,
            parsed=parsed,
            data=data,
            family=family,
            y=y,
            n_obs=n_obs,
            obs_precision=obs_precision,
            strategy=strategy,
            step_or_f0=step_or_f0,
            initial_theta=initial_theta,
            control_compute=control_compute,
            latent_marginal_indices=latent_marginal_indices,
            predictor_marginal_indices=predictor_marginal_indices,
            verbose=verbose,
        )
        if planned is not None:
            return planned

    obs = _build_obs(
        family,
        y,
        E=E,
        Ntrials=Ntrials,
        event=event,
        size=size,
        zero_prob=zero_prob,
        inflation=inflation,
        alpha=alpha,
        gamma=gamma,
        shape=shape,
        obs_precision=obs_precision,
    )
    # cbind may have rewritten y length
    n_obs = len(obs)

    x_fixed = _design_matrix(parsed, data, n_obs)
    p = x_fixed.shape[1]

    # Assemble A triplets and effect metadata.
    # Layout matches R-INLA intuition: fixed effects first, then random fields.
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    col_off = 0
    effect_types: list[str] = []
    effect_ns: list[int] = []
    effect_orders: list[int] = []
    effect_names: list[str] = []
    effect_graphs: list[Optional[list[list[int]]]] = []
    effect_scale: list[bool] = []
    effect_generics: list[Optional[GenericLike]] = []
    effect_prior_specs: list[list[tuple[str, list[float]]]] = []
    theta: list[float] = []

    effect_positions: list[Optional[list[float]]] = []
    effect_layouts: list[str] = []
    effect_seasons: list[int] = []
    effect_group_models: list[Optional[str]] = []
    effect_n_main: list[int] = []
    effect_nrow: list[int] = []
    effect_ncol: list[int] = []
    effect_cyclic: list[bool] = []
    effect_meshes: list[Optional[tuple]] = []  # (verts, tris) for SPDE
    effect_nus: list[int] = []  # matern2d nu

    # Fixed effects block first → latent_means[0] is intercept when present
    if p > 0:
        for j in range(p):
            for r in range(n_obs):
                v = float(x_fixed[r, j])
                if v != 0.0:
                    rows.append(r)
                    cols.append(col_off + j)
                    vals.append(v)
        effect_types.append("fixed")
        effect_ns.append(p)
        effect_orders.append(0)
        effect_names.append("fixed")
        effect_graphs.append(None)
        effect_scale.append(False)
        effect_generics.append(None)
        effect_prior_specs.append([])
        effect_positions.append(None)
        effect_layouts.append("simple")
        effect_seasons.append(4)
        effect_group_models.append(None)
        effect_n_main.append(p)
        effect_nrow.append(0)
        effect_ncol.append(0)
        effect_cyclic.append(False)
        effect_meshes.append(None)
        effect_nus.append(1)
        col_off += p

    for ft, gmodel in zip(parsed.f_terms, resolved_generics):
        idx = _get_col(data, ft.index).astype(int)
        if np.any(idx < 0):
            raise ValueError(f"NA/negative index in f({ft.index})")
        model = ft.model
        order = ft.order
        group_model = _group_model_from_ft(ft) if gmodel is None else None
        group_key = ft.kwargs.get("group")
        cyclic = bool(ft.kwargs.get("cyclic", False))
        nrow_kw = ft.kwargs.get("nrow")
        ncol_kw = ft.kwargs.get("ncol")

        if gmodel is not None:
            n_e = int(gmodel.n)
            umin = int(idx.min())
            zcol = idx - umin
            if int(zcol.min()) < 0 or int(zcol.max()) >= n_e:
                zcol = idx.copy()
                if int(zcol.min()) < 0 or int(zcol.max()) >= n_e:
                    raise ValueError(
                        f"generic index for '{ft.index}' out of range for n={n_e}"
                    )
            for r in range(n_obs):
                rows.append(r)
                cols.append(col_off + int(zcol[r]))
                vals.append(1.0)
            effect_graphs.append(None)
            effect_ns.append(n_e)
            effect_types.append("rgeneric")
            effect_orders.append(0)
            effect_names.append(ft.index)
            effect_scale.append(False)
            effect_generics.append(gmodel)
            effect_prior_specs.append([])
            effect_positions.append(None)
            effect_layouts.append("simple")
            effect_seasons.append(4)
            effect_group_models.append(None)
            effect_n_main.append(n_e)
            effect_nrow.append(0)
            effect_ncol.append(0)
            effect_cyclic.append(False)
            effect_meshes.append(None)
            effect_nus.append(1)
            tlen = int(gmodel.n_theta)
            if ft.initial is not None:
                init = list(np.asarray(ft.initial, dtype=float).reshape(-1))
                if len(init) != tlen:
                    raise ValueError(
                        f"f({ft.index}): initial length {len(init)} != expected {tlen}"
                    )
                theta.extend(init)
            elif initial_theta is None:
                theta.extend(list(gmodel.initial))
            col_off += n_e
            continue

        mesh_store = None
        nu_i = int(ft.kwargs.get("nu", 1) or 1)
        layout = str(ft.kwargs.get("layout", "simple"))

        if model == "rw2d" or model == "matern2d":
            if nrow_kw is None or ncol_kw is None:
                raise ValueError(f"f(..., model='{model}') requires nrow= and ncol=")
            nrow_i = int(nrow_kw)
            ncol_i = int(ncol_kw)
            n_main = nrow_i * ncol_i
            zcol = idx.copy()
            if int(zcol.min()) >= 1:
                zcol = zcol - 1
            if int(zcol.min()) < 0 or int(zcol.max()) >= n_main:
                raise ValueError(
                    f"{model} index for '{ft.index}' out of range for {nrow_i}x{ncol_i}"
                )
            adj = None
        elif model in ("besag", "bym", "bym2"):
            adj = _resolve_graph(ft, data)
            n_graph = len(adj)
            umin = int(idx.min())
            zcol = idx - umin
            if int(zcol.min()) < 0 or int(zcol.max()) >= n_graph:
                zcol = idx.copy()
                if int(zcol.min()) < 0 or int(zcol.max()) >= n_graph:
                    raise ValueError(
                        f"{model} index for '{ft.index}' out of range for graph size {n_graph}"
                    )
            n_main = n_graph  # spatial size; BYM expands latent below
            nrow_i = ncol_i = 0
        elif model == "spde":
            verts = ft.kwargs.get("vertices")
            tris = ft.kwargs.get("triangles")
            if isinstance(verts, str):
                verts = data[verts]
            if isinstance(tris, str):
                tris = data[tris]
            if verts is None or tris is None:
                raise ValueError(
                    "f(..., model='spde') requires vertices= and triangles="
                )
            verts_arr = np.asarray(verts, dtype=float)
            tris_arr = np.asarray(tris)
            if verts_arr.ndim != 2 or verts_arr.shape[1] != 2:
                raise ValueError("spde vertices must be N x 2")
            if tris_arr.ndim != 2 or tris_arr.shape[1] != 3:
                raise ValueError("spde triangles must be M x 3")
            # Accept 1-based triangles
            if int(tris_arr.min()) >= 1:
                tris_arr = tris_arr - 1
            vert_tuples = [(float(x), float(y)) for x, y in verts_arr]
            tri_tuples = [(int(a), int(b), int(c)) for a, b, c in tris_arr]
            mesh_store = (vert_tuples, tri_tuples)
            n_main = len(vert_tuples)
            loc_x_key = ft.kwargs.get("loc_x", "loc_x")
            loc_y_key = ft.kwargs.get("loc_y", "loc_y")
            if "loc" in ft.kwargs:
                loc = np.asarray(data[ft.kwargs["loc"]] if isinstance(ft.kwargs["loc"], str) else ft.kwargs["loc"], dtype=float)
                loc_x = loc[:, 0]
                loc_y = loc[:, 1]
            elif loc_x_key in data and loc_y_key in data:
                loc_x = _get_col(data, str(loc_x_key))
                loc_y = _get_col(data, str(loc_y_key))
            else:
                raise ValueError(
                    "f(..., model='spde') needs loc= or loc_x=/loc_y= columns"
                )
            a_spde = core.spde_projector_matrix(
                vert_tuples, tri_tuples, loc_x.tolist(), loc_y.tolist()
            ).to_scipy().tocsc().copy()
            # Scatter projector into global A
            coo = a_spde.tocoo()
            for r, c, v in zip(coo.row, coo.col, coo.data):
                rows.append(int(r))
                cols.append(col_off + int(c))
                vals.append(float(v))
            n_e = n_main
            adj = None
            nrow_i = ncol_i = 0
            zcol = None  # already mapped
        elif model == "fgn" and order in (3, 4):
            levels = np.sort(np.unique(idx))
            n_time = int(levels.size)
            zcol = np.searchsorted(levels, idx)
            n_main = (order + 1) * n_time
            adj = None
            nrow_i = ncol_i = 0
        else:
            raw_idx = data[ft.index]
            if hasattr(raw_idx, "cat"):
                levels = np.asarray(raw_idx.cat.categories)
                n_main = int(levels.size)
                zcol = np.asarray(raw_idx.cat.codes)
            elif hasattr(raw_idx, "categories"):
                levels = np.asarray(raw_idx.categories)
                n_main = int(levels.size)
                zcol = np.asarray(raw_idx.codes)
            else:
                levels = np.sort(np.unique(idx))
                n_main = int(levels.size)
                zcol = np.searchsorted(levels, idx)
            adj = None
            nrow_i = ncol_i = 0

        if model != "spde":
            if group_model is not None:
                if group_key is None:
                    raise ValueError(
                        f"f({ft.index}): control_group=... requires group= column"
                    )
                gcol = _get_col(data, str(group_key)).astype(int)
                g_levels = np.sort(np.unique(gcol))
                n_group = int(g_levels.size)
                g_z = np.searchsorted(g_levels, gcol)
                for r in range(n_obs):
                    rows.append(r)
                    cols.append(col_off + int(g_z[r]) * n_main + int(zcol[r]))
                    vals.append(1.0)
                n_e = n_main * n_group
            elif model == "bym":
                # Observe u + v: columns i and n+i
                for r in range(n_obs):
                    zi = int(zcol[r])
                    rows.append(r); cols.append(col_off + zi); vals.append(1.0)
                    rows.append(r); cols.append(col_off + n_main + zi); vals.append(1.0)
                n_e = 2 * n_main
            elif model == "crw2" and layout in ("pairs", "block"):
                for r in range(n_obs):
                    zi = int(zcol[r])
                    col = (2 * zi) if layout == "pairs" else zi
                    rows.append(r)
                    cols.append(col_off + col)
                    vals.append(1.0)
                n_e = 2 * n_main
            else:
                for r in range(n_obs):
                    rows.append(r)
                    cols.append(col_off + int(zcol[r]))
                    vals.append(1.0)
                n_e = n_main

        effect_graphs.append(adj)
        effect_ns.append(n_e)
        effect_n_main.append(n_main)
        effect_group_models.append(group_model)
        effect_nrow.append(nrow_i)
        effect_ncol.append(ncol_i)
        effect_cyclic.append(cyclic)
        effect_meshes.append(mesh_store)
        effect_nus.append(nu_i)

        raw_pos = ft.kwargs.get("positions")
        if raw_pos is not None:
            if isinstance(raw_pos, str):
                pos_arr = _get_col(data, raw_pos).tolist()
            else:
                pos_arr = [float(p) for p in np.asarray(raw_pos, dtype=float).reshape(-1)]
        else:
            pos_arr = [float(p) for p in range(n_main)]
        effect_positions.append(pos_arr)
        effect_layouts.append(layout)
        season = int(ft.kwargs.get("season", ft.kwargs.get("s", order if order > 0 else 4)))
        effect_seasons.append(season)
        effect_types.append(model)
        effect_orders.append(int(order))
        effect_names.append(ft.index)
        effect_scale.append(
            bool(_model_meta(model)["default_scale_model"])
            if ft.scale_model is None
            else bool(ft.scale_model)
        )
        effect_generics.append(None)
        effect_prior_specs.append(_resolve_effect_priors(model, ft.kwargs))
        tlen = _theta_len(model, order, group_model)
        if ft.initial is not None:
            init = list(np.asarray(ft.initial, dtype=float).reshape(-1))
            if len(init) != tlen:
                raise ValueError(
                    f"f({ft.index}): initial length {len(init)} != expected {tlen}"
                )
            theta.extend(init)
        elif initial_theta is None:
            theta.extend(_default_theta(model, order, group_model))
        col_off += n_e

    if not effect_types:
        raise ValueError("formula has no f() terms and no fixed effects")

    if initial_theta is not None:
        theta = list(np.asarray(initial_theta, dtype=float).reshape(-1))

    a = sparse.csc_matrix((vals, (rows, cols)), shape=(n_obs, col_off))

    types = list(effect_types)
    ns = list(effect_ns)
    orders = list(effect_orders)
    graphs = list(effect_graphs)
    generics = list(effect_generics)
    prior_specs = list(effect_prior_specs)
    positions_list = list(effect_positions)
    layouts_list = list(effect_layouts)
    seasons_list = list(effect_seasons)
    group_models = list(effect_group_models)
    n_mains = list(effect_n_main)
    nrows = list(effect_nrow)
    ncols = list(effect_ncol)
    cyclics = list(effect_cyclic)
    meshes = list(effect_meshes)
    nus = list(effect_nus)

    theta_lens: list[int] = []
    for t, o, g, gm in zip(types, orders, generics, group_models):
        if g is not None:
            theta_lens.append(int(g.n_theta))
        else:
            theta_lens.append(_theta_len(t, o, gm))
    has_intercept = bool(parsed.intercept)

    precomputed_scales = []
    for t, n_main, g, scale_flag in zip(types, n_mains, graphs, effect_scale):
        if not scale_flag:
            precomputed_scales.append(1.0)
            continue
        q_base = None
        if t == "iid":
            q_base = core.iid_precision_matrix(n_main, 1.0)
        elif t == "rw1":
            q_base = core.rw1_precision_matrix(n_main, 1.0)
        elif t == "rw2":
            q_base = core.rw2_precision_matrix(n_main, 1.0)
        elif t == "besag":
            assert g is not None
            q_base = core.besag_precision_matrix(g, 1.0)
        if q_base is not None:
            q_scipy = q_base.to_scipy() if isinstance(q_base, core.PyCscMatrix) else sparse.csc_matrix(q_base)
            precomputed_scales.append(_compute_scale_factor(q_scipy))
        else:
            precomputed_scales.append(1.0)

    use_shared_q = (
        all(g is None for g in generics)
        and all(gm is None for gm in group_models)
        and "spde" not in types
        and hasattr(core, "build_structured_precision")
    )

    def _structured_effect_dicts() -> list[dict]:
        out = []
        for ei, typ in enumerate(types):
            order_enc = int(orders[ei])
            if typ in ("rw2d", "matern2d"):
                # Match R: ±nrow (negative ⇒ cyclic)
                nrow_i = int(nrows[ei])
                order_enc = -nrow_i if cyclics[ei] else nrow_i
            elif typ == "seasonal":
                order_enc = int(seasons_list[ei])
            d = {
                "model": typ,
                "n": int(ns[ei]),
                "theta_len": int(theta_lens[ei]),
                "scale_model": bool(effect_scale[ei]),
                "order": order_enc,
                "nrow": int(nrows[ei]),
                "ncol": int(ncols[ei]),
                "cyclic": bool(cyclics[ei]),
                "matern_nu": int(nus[ei]),
                "crw2_layout": str(layouts_list[ei]),
            }
            if positions_list[ei] is not None:
                d["positions"] = positions_list[ei]
            if graphs[ei] is not None:
                d["adj"] = graphs[ei]
            out.append(d)
        return out

    def _main_precision(typ, n_main, ti, ei):
        if typ == "iid":
            tau = float(np.exp(ti[0])) if ti else 1.0
            return core.iid_precision_matrix(n_main, tau)
        if typ == "rw1":
            tau = float(np.exp(ti[0])) if ti else 1.0
            return core.rw1_precision_matrix(n_main, tau)
        if typ == "rw2":
            tau = float(np.exp(ti[0])) if ti else 1.0
            return core.rw2_precision_matrix(n_main, tau)
        if typ == "rw2d":
            tau = float(np.exp(ti[0])) if ti else 1.0
            return core.rw2d_precision_matrix(nrows[ei], ncols[ei], tau, cyclic=cyclics[ei])
        if typ == "ar1":
            if len(ti) < 2:
                raise ValueError("ar1 needs theta=[log_tau, logit_rho]")
            tau = float(np.exp(ti[0]))
            rho = float(np.clip(2.0 / (1.0 + np.exp(-ti[1])) - 1.0, -0.999, 0.999))
            return core.ar1_precision_matrix_csc(n_main, rho, tau)
        if typ in ("ar", "arp"):
            tau = float(np.exp(ti[0])) if ti else 1.0
            pacf = [float(np.tanh(v * 0.5)) for v in ti[1:]] if len(ti) > 1 else [0.0]
            return core.arp_precision_matrix(n_main, pacf, tau)
        if typ == "seasonal":
            tau = float(np.exp(ti[0])) if ti else 1.0
            return core.seasonal_precision_matrix(n_main, season=seasons_list[ei], tau=tau)
        if typ == "crw1":
            tau = float(np.exp(ti[0])) if ti else 1.0
            pos = positions_list[ei] or [float(i) for i in range(n_main)]
            if len(pos) < 2:
                pos = [float(i) for i in range(n_main)]
            return core.crw1_precision_matrix(pos, tau)
        if typ == "crw2":
            tau = float(np.exp(ti[0])) if ti else 1.0
            pos = positions_list[ei] or [float(i) for i in range(n_main)]
            if len(pos) < 3:
                pos = [float(i) for i in range(n_main)]
            return core.crw2_precision_matrix(pos, tau, layout=layouts_list[ei])
        if typ == "besag":
            tau = float(np.exp(ti[0])) if ti else 1.0
            assert graphs[ei] is not None
            return core.besag_precision_matrix(graphs[ei], tau)
        if typ == "bym":
            if len(ti) < 2:
                raise ValueError("bym needs [log_tau_spatial, log_tau_iid]")
            assert graphs[ei] is not None
            return core.bym_precision_matrix(
                graphs[ei], float(np.exp(ti[0])), float(np.exp(ti[1]))
            )
        if typ == "bym2":
            if len(ti) < 2:
                raise ValueError("bym2 needs [log_tau, logit_phi]")
            assert graphs[ei] is not None
            tau = float(np.exp(ti[0]))
            phi = float(1.0 / (1.0 + np.exp(-ti[1])))
            phi = float(np.clip(phi, 1e-6, 1.0 - 1e-6))
            return core.bym2_precision_matrix(graphs[ei], tau, phi)
        if typ == "matern2d":
            if len(ti) < 2:
                raise ValueError("matern2d needs [log_prec, log_range]")
            prec = float(np.exp(ti[0]))
            rng = float(np.exp(ti[1]))
            return core.matern2d_precision_matrix(
                nrows[ei], ncols[ei], nu=nus[ei], range=rng, prec=prec, cyclic=cyclics[ei]
            )
        if typ == "spde":
            if len(ti) < 2:
                raise ValueError("spde needs [log_tau, log_kappa]")
            mesh = meshes[ei]
            assert mesh is not None
            tau = float(np.exp(ti[0]))
            kappa = float(np.exp(ti[1]))
            return core.spde_precision_matrix(mesh[0], mesh[1], kappa=kappa, tau=tau)
        if typ == "fgn":
            if len(ti) < 2:
                raise ValueError("fgn needs two hyperparameters")
            tau = float(np.exp(ti[0]))
            if orders[ei] in (3, 4):
                hurst = core.fgn_hurst_from_intern(ti[1])
                n_time = n_main // (orders[ei] + 1)
                return core.fgn_approx_precision_matrix(n_time, hurst, tau, order=orders[ei])
            hurst = 1.0 / (1.0 + np.exp(-ti[1]))
            return core.fgn_precision_matrix(n_main, hurst, tau)
        raise ValueError(f"unsupported effect type {typ}")

    def build_prior(th):
        th = list(th)
        if use_shared_q:
            return core.build_structured_precision(
                _structured_effect_dicts(), th, float(fixed_prec)
            )
        blocks = []
        off = 0
        for ei, typ in enumerate(types):
            n_e = ns[ei]
            tlen = theta_lens[ei]
            ti = th[off : off + tlen]
            off += tlen
            if typ == "fixed":
                blocks.append(sparse.eye(n_e, format="csc") * float(fixed_prec))
            elif typ == "rgeneric":
                g = generics[ei]
                assert g is not None
                blocks.append(g.precision(ti))
            else:
                gm = group_models[ei]
                n_main = n_mains[ei]
                main_len = _theta_len(typ, orders[ei], None)
                q_main = _main_precision(typ, n_main, ti[:main_len], ei)
                if gm is not None:
                    q_group = _build_group_q(n_e // n_main, gm, ti[main_len:])
                    # Prefer native kronecker; fall back to SciPy.
                    if hasattr(core, "kronecker_csc"):
                        q = core.kronecker_csc(q_group, q_main)
                    else:
                        q = sparse.kron(_as_scipy_csc(q_group), _as_scipy_csc(q_main), format="csc")
                else:
                    q = q_main
                blocks.append(q)
            if precomputed_scales[ei] != 1.0:
                q = blocks[-1]
                if isinstance(q, core.PyCscMatrix):
                    q = q.to_scipy()
                blocks[-1] = q * precomputed_scales[ei]
        if len(blocks) == 1:
            b0 = blocks[0]
            if isinstance(b0, core.PyCscMatrix):
                return b0
            return core.PyCscMatrix.from_scipy(sparse.csc_matrix(b0).copy())
        owned = []
        for b in blocks:
            if isinstance(b, core.PyCscMatrix):
                owned.append(b.to_scipy().copy())
            else:
                owned.append(sparse.csc_matrix(b).copy())
        return core.PyCscMatrix.from_scipy(sparse.block_diag(owned, format="csc"))

    constraints_a = None
    constraints_e = None
    if use_shared_q and hasattr(core, "structured_constraints"):
        shared_c = core.structured_constraints(_structured_effect_dicts())
        if shared_c is not None:
            constraints_a, constraints_e = shared_c
        # Extra: classic INLA sum-to-zero on iid when an intercept is present.
        if has_intercept:
            off = 0
            for ei, (typ, n_e) in enumerate(zip(types, ns)):
                if typ == "iid":
                    ba, be = _sum_to_zero_a(n_e, 1)
                    part = _embed_constraint(ba, be, n_e, col_off, off)
                    if constraints_a is None:
                        constraints_a, constraints_e = part
                    else:
                        constraints_a = list(constraints_a) + list(part[0])
                        constraints_e = list(constraints_e) + list(part[1])
                off += n_e
    else:
        constr_parts: list[tuple[list[float], list[float]]] = []
        off = 0
        for ei, (typ, n_e, cyclic_flag) in enumerate(zip(types, ns, cyclics)):
            if typ == "seasonal":
                season = int(seasons_list[ei])
                k = max(season - 1, 1)
            else:
                k = _rank_deficiency(typ, cyclic=cyclic_flag)
            if k == 0 and typ == "iid" and has_intercept:
                k = 1
            if k > 0:
                # Classic BYM: constrain spatial block u only (first n_main).
                block_n = n_mains[ei] if typ == "bym" else n_e
                if typ == "seasonal":
                    ba, be = core.seasonal_constraint(block_n, min(season, block_n))
                elif typ == "rw2d" and not cyclic_flag:
                    ba, be = core.plane_constraint_2d(int(nrows[ei]), int(ncols[ei]))
                else:
                    ba, be = _sum_to_zero_a(block_n, k)
                constr_parts.append(_embed_constraint(ba, be, block_n, col_off, off))
            off += n_e
        stacked = _vstack_constraints(constr_parts)
        constraints_a = stacked[0] if stacked is not None else None
        constraints_e = stacked[1] if stacked is not None else None

    def log_prior_density(th):
        th = list(th)
        lp = 0.0
        off = 0
        for ei, typ in enumerate(types):
            tlen = theta_lens[ei]
            ti = th[off : off + tlen]
            off += tlen
            if typ == "fixed" or tlen == 0:
                continue
            g = generics[ei]
            if g is not None:
                lp += g.eval_log_prior(ti)
            else:
                specs = list(prior_specs[ei])
                gm = group_models[ei]
                if gm is not None:
                    # Append default priors for the group hyperparameter block.
                    specs = specs + list(core.default_hyper_priors(gm))
                try:
                    names = [s[0] for s in specs]
                    params = [s[1] for s in specs]
                    lp += float(core.hyper_prior_stack_log_density(names, params, ti))
                except Exception:
                    lp += float(-0.05 * sum(v * v for v in ti))
        return lp

    if verbose:
        print(f"inla: family={family} n_obs={n_obs} n_latent={col_off}")
        print(f"inla: effects={list(zip(effect_names, effect_types, effect_ns))}")
        print(f"inla: initial_theta={theta}")

    result = core.run_inla_inference(
        initial_theta=theta,
        build_prior=build_prior,
        log_prior_density=log_prior_density,
        obs=obs,
        strategy=strategy,
        step_or_f0=step_or_f0,
        a=a,
        latent_marginal_indices=(
            list(latent_marginal_indices)
            if latent_marginal_indices is not None
            else (
                _resolve_marginal_indices(
                    controls["return_marginals_latent"],
                    col_off,
                    name="return_marginals_latent",
                )
            )
        ),
        predictor_marginal_indices=(
            list(predictor_marginal_indices)
            if predictor_marginal_indices is not None
            else (
                _resolve_marginal_indices(
                    controls["return_marginals_predictor"],
                    n_obs,
                    name="return_marginals_predictor",
                )
            )
        ),
        constraints_a=constraints_a,
        constraints_e=constraints_e,
        deterministic=bool(deterministic),
    )

    # Attach R-like summary slices (Gaussian interim) via a thin wrapper
    means = np.asarray(result.latent_means, dtype=float)
    sds = np.sqrt(np.maximum(np.asarray(result.latent_variances, dtype=float), 0.0))
    summary_random = {}
    summary_fixed = None
    off = 0
    for name, typ, n_e in zip(effect_names, effect_types, effect_ns):
        sl = slice(off, off + n_e)
        tab = {
            "mean": means[sl],
            "sd": sds[sl],
            "0.025quant": means[sl] - 1.96 * sds[sl],
            "0.5quant": means[sl],
            "0.975quant": means[sl] + 1.96 * sds[sl],
        }
        if typ == "fixed":
            fixed_col_names = []
            if has_intercept:
                fixed_col_names.append("(Intercept)")
            fixed_col_names.extend(parsed.fixed_terms)
            tab["names"] = fixed_col_names
            summary_fixed = tab
        else:
            summary_random[name] = tab
        off += n_e

    # Hyperparameter tables: internal θ as optimized, plus the natural scale
    # (Precision / Rho / Phi / …) that R reports, using the shared registry labels.
    label_orders = [
        theta_lens[ei] if typ == "rgeneric" else effect_orders[ei]
        for ei, typ in enumerate(effect_types)
    ]
    hyper_labels, hyper_transforms = _hyper_labels(
        effect_types, effect_names, label_orders, group_models
    )
    summary_hyperpar_internal = _internal_hyperpar_table(result)
    summary_hyperpar = _natural_hyperpar_table(
        summary_hyperpar_internal, hyper_labels, hyper_transforms
    )

    out = InlaResult(result)
    out.summary_random = summary_random
    out.summary_fixed = summary_fixed
    out.summary_hyperpar = summary_hyperpar
    out.summary_hyperpar_internal = summary_hyperpar_internal
    out.effects = {
        "names": effect_names,
        "types": effect_types,
        "ns": effect_ns,
    }
    out.formula = formula

    if verbose:
        print(f"inla: mlik={out.marginal_log_lik:.4f} dic={out.dic:.4f}")
        if summary_fixed is not None:
            print("inla: fixed means=", list(summary_fixed["mean"]))

    return out


def _internal_hyperpar_table(result) -> Optional[dict[str, np.ndarray]]:
    """Moments/quantiles of each internal θ marginal (mirrors the R front-end)."""
    mode = np.asarray(result.mode, dtype=float)
    m = mode.size
    if m == 0:
        return None
    mean = np.full(m, np.nan)
    sd = np.full(m, np.nan)
    q025 = np.full(m, np.nan)
    q50 = mode.copy()
    q975 = np.full(m, np.nan)

    marginals = list(result.internal_marginals_hyperpar or [])
    if len(marginals) == m:
        for j, marg in enumerate(marginals):
            x = np.asarray(marg.x, dtype=float)
            y = np.asarray(marg.y, dtype=float)
            if x.size < 2:
                continue
            dx = np.diff(x)
            mass = float(np.sum(0.5 * (y[:-1] + y[1:]) * dx))
            if mass <= 0:
                continue
            y = y / mass
            ex = float(np.sum(0.5 * (x[:-1] * y[:-1] + x[1:] * y[1:]) * dx))
            ex2 = float(
                np.sum(0.5 * (x[:-1] ** 2 * y[:-1] + x[1:] ** 2 * y[1:]) * dx)
            )
            mean[j] = ex
            sd[j] = np.sqrt(max(ex2 - ex * ex, 0.0))
            cdf = np.concatenate(([0.0], np.cumsum(0.5 * (y[:-1] + y[1:]) * dx)))
            cdf = cdf / max(cdf[-1], np.finfo(float).eps)
            q025[j], q50[j], q975[j] = np.interp([0.025, 0.5, 0.975], cdf, x)

    return {
        "names": [f"theta{j + 1}" for j in range(m)],
        "mean": mean,
        "sd": sd,
        "0.025quant": q025,
        "0.5quant": q50,
        "0.975quant": q975,
        "mode": mode,
    }


def _natural_hyperpar_table(
    internal: Optional[dict[str, np.ndarray]],
    labels: Sequence[str],
    transforms: Sequence[str],
) -> Optional[dict[str, np.ndarray]]:
    """Map the internal θ table to the natural scale R prints in `summary.hyperpar`."""
    if internal is None:
        return None
    m = len(internal["mode"])
    if len(transforms) != m:
        return internal

    out = {k: np.array(v, dtype=float) for k, v in internal.items() if k != "names"}
    for j, tag in enumerate(transforms):
        theta_mean = internal["mean"][j]
        out["sd"][j] = _natural_sd(tag, theta_mean, internal["sd"][j])
        for key in ("mean", "0.025quant", "0.5quant", "0.975quant", "mode"):
            out[key][j] = _to_natural(tag, internal[key][j])
        if tag in ("exp", "rho", "phi"):
            lo, hi = sorted((out["0.025quant"][j], out["0.975quant"][j]))
            out["0.025quant"][j], out["0.975quant"][j] = lo, hi
    out["names"] = list(labels)
    return out


class InlaResult:
    """Thin wrapper around ``PyInferenceResult`` with R-like summary fields."""

    def __init__(self, native):
        self._native = native
        self.summary_random = {}
        self.summary_fixed = None
        self.summary_hyperpar = None
        self.summary_hyperpar_internal = None
        self.effects = None
        self.formula = None

    def __getattr__(self, name):
        return getattr(self._native, name)


fit = _fit
