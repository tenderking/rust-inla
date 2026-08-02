"""R-INLA-style formula parsing for high-level `inla()`."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any


_F_CALL_RE = re.compile(
    r"f\s*\(\s*(?P<idx>[A-Za-z_][\w.]*)\s*(?P<rest>(?:,\s*.*?)?)\s*\)",
    re.DOTALL,
)


@dataclass
class FTerm:
    index: str
    model: str = "iid"
    order: int = 0
    graph: Any = None
    scale_model: bool = False
    initial: Any = None
    kwargs: dict = field(default_factory=dict)


@dataclass
class ParsedFormula:
    response: str
    fixed_terms: list[str]
    intercept: bool
    f_terms: list[FTerm]


def _parse_f_kwargs(rest: str) -> dict[str, Any]:
    """Parse `, model='besag', scale.model=True` into a dict."""
    rest = (rest or "").strip()
    if rest.startswith(","):
        rest = rest[1:].strip()
    if not rest:
        return {}

    # Allow R-style dots in kwarg names: scale.model -> scale_model for Python
    # Evaluate as a fake function call: f(idx, ...)
    # Wrap unknown identifiers as strings when they look like data keys.
    stub = f"f(__idx__, {rest})"
    # Replace dots in keyword names only: scale.model= -> scale_model=
    stub = re.sub(r"([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*=", r"\1_\2=", stub)

    tree = ast.parse(stub, mode="eval")
    assert isinstance(tree.body, ast.Call)
    out: dict[str, Any] = {}
    for kw in tree.body.keywords:
        if kw.arg is None:
            continue
        key = kw.arg
        try:
            out[key] = ast.literal_eval(kw.value)
        except Exception:
            # bare name → treat as string key into data (e.g. graph=adj_matrix)
            if isinstance(kw.value, ast.Name):
                out[key] = kw.value.id
            else:
                raise ValueError(f"unsupported f() argument: {key}=...") from None
    return out


def parse_formula(formula: str) -> ParsedFormula:
    # Accept R formula `~` and assignment-style `<-` as response separators.
    if "<-" in formula:
        lhs, rhs = formula.split("<-", 1)
    elif "~" in formula:
        lhs, rhs = formula.split("~", 1)
    else:
        raise ValueError("formula must contain '~' or '<-'")
    response = lhs.strip()
    if not response:
        raise ValueError("formula missing response")

    rhs = rhs.strip()
    f_terms: list[FTerm] = []
    stripped = rhs

    for m in _F_CALL_RE.finditer(rhs):
        idx = m.group("idx")
        kw = _parse_f_kwargs(m.group("rest") or "")
        model = str(kw.pop("model", "iid")).lower()
        order = int(kw.pop("order", 0) or 0)
        graph = kw.pop("graph", None)
        scale_model = bool(kw.pop("scale_model", False))
        initial = kw.pop("initial", None)
        f_terms.append(
            FTerm(
                index=idx,
                model=model,
                order=order,
                graph=graph,
                scale_model=scale_model,
                initial=initial,
                kwargs=kw,
            )
        )
        stripped = stripped.replace(m.group(0), " ")

    # Fixed-effects tokens
    intercept = True
    fixed_terms: list[str] = []
    # Normalize operators
    toks = re.split(r"\s*\+\s*", stripped)
    for tok in toks:
        t = tok.strip()
        if not t or t == "1":
            continue
        if t in ("-1", "0"):
            intercept = False
            continue
        # leftover punctuation from removed f()
        t = re.sub(r"[(),]", " ", t).strip()
        if not t:
            continue
        # may have multiple names glued; split on whitespace
        for name in t.split():
            if name in ("-1", "0", "1", "+"):
                if name in ("-1", "0"):
                    intercept = False
                continue
            if re.fullmatch(r"[A-Za-z_][\w.]*", name):
                fixed_terms.append(name)

    # Detect "-1 +" or "0 +" style at start of original rhs
    if re.search(r"(^|[\s+])-\s*1([\s+]|$)", rhs) or re.search(
        r"(^|[\s+])0\s*\+", rhs
    ):
        intercept = False

    return ParsedFormula(
        response=response,
        fixed_terms=fixed_terms,
        intercept=intercept,
        f_terms=f_terms,
    )
