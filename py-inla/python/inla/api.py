"""High-level R-INLA-like `inla()` front-end for Python."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
from scipy import sparse

from inla import _native as core
from inla import generic as generic_mod
from inla.formula import ParsedFormula, parse_formula
from inla.generic import GenericModel, Model

SUPPORTED_F_MODELS = ("iid", "rw2", "ar1", "besag", "fgn", "rw1")
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


def _theta_len(model: str, order: int = 0) -> int:
    model = model.lower()
    if model in ("iid", "rw1", "rw2", "besag"):
        return 1
    if model in ("ar1", "fgn"):
        return 2
    if model == "fixed":
        return 0
    return 1


def _default_theta(model: str, order: int = 0) -> list[float]:
    model = model.lower()
    if model == "fgn" and order in (3, 4):
        return [1.0, 2.0]
    if model in ("ar1", "fgn"):
        return [0.0, 0.0]
    if model == "fixed":
        return []
    if model == "besag":
        # Mildly informative spatial precision; helps Newton from x=0.
        return [1.0]
    return [0.0]


def _resolve_f_model(
    ft,
    *,
    models: Optional[Mapping[str, GenericLike]],
    rgeneric: Optional[GenericLike],
) -> Optional[GenericLike]:
    """Return a GenericLike if this f() term is a custom model, else None."""
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


def _fit(
    formula: str,
    data: Mapping[str, Any],
    family: str = "gaussian",
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
    strategy: str = "ccd",
    step_or_f0: float = 1.0,
    initial_theta: Optional[Sequence[float]] = None,
    control_family: Optional[Mapping[str, Any]] = None,
    fixed_prec: float = 1e-4,
    latent_marginal_indices: Optional[Sequence[int]] = None,
    models: Optional[Mapping[str, GenericLike]] = None,
    rgeneric: Optional[GenericLike] = None,
    verbose: bool = False,
):
    """Fit an INLA model with R-like formula syntax.

    Parameters
    ----------
    formula :
        e.g. ``"y ~ x + f(region, model='besag')"`` or ``"y <- x + f(...)"``.
    data :
        Mapping of column name → array (DataFrame ``to_dict(orient='series')`` works).
        For Besag, include ``adj_matrix`` (square) or pass ``graph=...`` in ``f()``.
    family :
        Likelihood family. ``cbinomial`` is accepted as an alias of ``binomial``.
    Ntrials :
        Binomial trials ``n``, or ``cbind``-style ``(n_obs, 2)`` array of ``(y, n)``.
    models :
        Mapping of name → ``GenericModel`` / ``Model`` for ``f(..., model='name')``.
    rgeneric :
        Single custom model for ``f(..., model='rgeneric')`` (R-style).
    """
    parsed = parse_formula(formula)
    resolved_generics: list[Optional[GenericLike]] = []
    for ft in parsed.f_terms:
        g = _resolve_f_model(ft, models=models, rgeneric=rgeneric)
        resolved_generics.append(g)
        if g is None and ft.model not in SUPPORTED_F_MODELS:
            raise ValueError(f"unsupported f() model '{ft.model}'")

    y = _get_col(data, parsed.response)
    n_obs = y.size

    obs_precision = 1.0
    if control_family is not None:
        try:
            obs_precision = float(np.exp(control_family["hyper"]["prec"]["initial"]))
        except Exception:
            pass

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
    theta: list[float] = []

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
        col_off += p

    for ft, gmodel in zip(parsed.f_terms, resolved_generics):
        idx = _get_col(data, ft.index).astype(int)
        if np.any(idx < 0):
            raise ValueError(f"NA/negative index in f({ft.index})")
        model = ft.model
        order = ft.order

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

        if model == "besag":
            adj = _resolve_graph(ft, data)
            n_e = len(adj)
            umin = int(idx.min())
            zcol = idx - umin
            if int(zcol.min()) < 0 or int(zcol.max()) >= n_e:
                zcol = idx.copy()
                if int(zcol.min()) < 0 or int(zcol.max()) >= n_e:
                    raise ValueError(
                        f"besag index for '{ft.index}' out of range for graph size {n_e}"
                    )
            for r in range(n_obs):
                rows.append(r)
                cols.append(col_off + int(zcol[r]))
                vals.append(1.0)
            effect_graphs.append(adj)
            effect_ns.append(n_e)
        elif model == "fgn" and order in (3, 4):
            levels = np.sort(np.unique(idx))
            n_time = int(levels.size)
            zcol = np.searchsorted(levels, idx)
            n_e = (order + 1) * n_time
            for r in range(n_obs):
                rows.append(r)
                cols.append(col_off + int(zcol[r]))  # z-block only
                vals.append(1.0)
            effect_graphs.append(None)
            effect_ns.append(n_e)
        else:
            levels = np.sort(np.unique(idx))
            n_e = int(levels.size)
            zcol = np.searchsorted(levels, idx)
            for r in range(n_obs):
                rows.append(r)
                cols.append(col_off + int(zcol[r]))
                vals.append(1.0)
            effect_graphs.append(None)
            effect_ns.append(n_e)

        effect_types.append(model)
        effect_orders.append(int(order))
        effect_names.append(ft.index)
        effect_scale.append(bool(ft.scale_model))
        effect_generics.append(None)
        tlen = _theta_len(model, order)
        if ft.initial is not None:
            init = list(np.asarray(ft.initial, dtype=float).reshape(-1))
            if len(init) != tlen:
                raise ValueError(
                    f"f({ft.index}): initial length {len(init)} != expected {tlen}"
                )
            theta.extend(init)
        elif initial_theta is None:
            theta.extend(_default_theta(model, order))
        col_off += effect_ns[-1]

    if not effect_types:
        raise ValueError("formula has no f() terms and no fixed effects")

    if initial_theta is not None:
        theta = list(np.asarray(initial_theta, dtype=float).reshape(-1))

    a = sparse.csc_matrix((vals, (rows, cols)), shape=(n_obs, col_off))

    # Prior builder
    types = list(effect_types)
    ns = list(effect_ns)
    orders = list(effect_orders)
    graphs = list(effect_graphs)
    generics = list(effect_generics)
    theta_lens: list[int] = []
    for t, o, g in zip(types, orders, generics):
        if g is not None:
            theta_lens.append(int(g.n_theta))
        else:
            theta_lens.append(_theta_len(t, o))
    has_intercept = bool(parsed.intercept)

    def build_prior(th):
        th = list(th)
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
            elif typ == "iid":
                tau = float(np.exp(ti[0])) if ti else 1.0
                if has_intercept:
                    q = core.iid_precision_matrix(n_e, tau).to_scipy().copy()
                    q = q + sparse.csr_matrix(np.ones((n_e, n_e)) * 1.0)
                    blocks.append(q.tocsc())
                else:
                    blocks.append(core.iid_precision_matrix(n_e, tau))
            elif typ == "rw1":
                tau = float(np.exp(ti[0])) if ti else 1.0
                if has_intercept:
                    q = core.rw1_precision_matrix(n_e, tau).to_scipy().copy()
                    q = q + sparse.csr_matrix(np.ones((n_e, n_e)) * 1.0)
                    blocks.append(q.tocsc())
                else:
                    blocks.append(core.rw1_precision_matrix(n_e, tau))
            elif typ == "rw2":
                tau = float(np.exp(ti[0])) if ti else 1.0
                if has_intercept:
                    q = core.rw2_precision_matrix(n_e, tau).to_scipy().copy()
                    q = q + sparse.csr_matrix(np.ones((n_e, n_e)) * 1.0)
                    blocks.append(q.tocsc())
                else:
                    blocks.append(core.rw2_precision_matrix(n_e, tau))
            elif typ == "ar1":
                if len(ti) < 2:
                    raise ValueError("ar1 needs theta=[log_tau, logit_rho]")
                tau = float(np.exp(ti[0]))
                rho = 2.0 / (1.0 + np.exp(-ti[1])) - 1.0
                blocks.append(core.ar1_precision_matrix_csc(n_e, rho, tau))
            elif typ == "besag":
                tau = float(np.exp(ti[0])) if ti else 1.0
                adj = graphs[ei]
                assert adj is not None
                q = core.besag_precision_matrix(adj, 1.0).to_scipy().copy()
                q = q.tolil()
                for i in range(q.shape[0]):
                    q[i, i] += 1e-5
                q = q.tocsc() * tau
                if has_intercept:
                    q = q + sparse.csr_matrix(np.ones((n_e, n_e)) * 1.0)
                blocks.append(q.tocsc())
            elif typ == "fgn":
                if len(ti) < 2:
                    raise ValueError("fgn needs two hyperparameters")
                tau = float(np.exp(ti[0]))
                if orders[ei] in (3, 4):
                    hurst = core.fgn_hurst_from_intern(ti[1])
                    n_time = n_e // (orders[ei] + 1)
                    blocks.append(
                        core.fgn_approx_precision_matrix(
                            n_time, hurst, tau, order=orders[ei]
                        )
                    )
                else:
                    hurst = 1.0 / (1.0 + np.exp(-ti[1]))
                    blocks.append(core.fgn_precision_matrix(n_e, hurst, tau))
            else:
                raise ValueError(f"unsupported effect type {typ}")
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
                lp += float(-0.5 * 0.1 * sum(float(v) ** 2 for v in ti))
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
        latent_marginal_indices=list(latent_marginal_indices)
        if latent_marginal_indices is not None
        else None,
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
            summary_fixed = tab
        else:
            summary_random[name] = tab
        off += n_e

    out = InlaResult(result)
    out.summary_random = summary_random
    out.summary_fixed = summary_fixed
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


class InlaResult:
    """Thin wrapper around ``PyInferenceResult`` with R-like summary fields."""

    def __init__(self, native):
        self._native = native
        self.summary_random = {}
        self.summary_fixed = None
        self.effects = None
        self.formula = None

    def __getattr__(self, name):
        return getattr(self._native, name)
