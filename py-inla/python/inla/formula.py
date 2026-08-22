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


def _find_f_calls(rhs: str) -> list[tuple[str, str, str]]:
    """Find `f(idx, ...)` calls with balanced parentheses.

    Returns list of (full_match, idx, rest) where rest is the kwargs substring
    including a leading comma when present.
    """
    out: list[tuple[str, str, str]] = []
    i = 0
    n = len(rhs)
    while i < n:
        m = re.search(r"f\s*\(", rhs[i:], re.IGNORECASE)
        if not m:
            break
        start = i + m.start()
        # position of '(' after f
        paren = i + m.end() - 1
        depth = 0
        j = paren
        while j < n:
            ch = rhs[j]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    body = rhs[paren + 1 : j]
                    # idx is first comma-separated token
                    body_st = body.strip()
                    if not body_st:
                        break
                    # split idx from rest at top-level comma
                    idx = None
                    rest = ""
                    depth2 = 0
                    cut = None
                    for k, c in enumerate(body_st):
                        if c in "([{":
                            depth2 += 1
                        elif c in ")]}":
                            depth2 -= 1
                        elif c == "," and depth2 == 0:
                            cut = k
                            break
                    if cut is None:
                        idx = body_st
                        rest = ""
                    else:
                        idx = body_st[:cut].strip()
                        rest = body_st[cut:]  # includes leading comma
                    if idx and re.fullmatch(r"[A-Za-z_][\w.]*", idx):
                        out.append((rhs[start : j + 1], idx, rest))
                    break
            j += 1
        i = j + 1 if j < n else n
    return out


@dataclass
class FTerm:
    index: str
    model: str = "iid"
    order: int = 0
    graph: Any = None
    #: ``None`` means "use the registry default for this model".
    scale_model: bool | None = None
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
    # Positional args after the index are R-INLA weights: f(i, z, model=...) or f(j, z, copy=...).
    for arg in tree.body.args[1:]:
        if "weights" in out:
            break
        try:
            out["weights"] = ast.literal_eval(arg)
        except Exception:
            if isinstance(arg, ast.Name):
                out["weights"] = arg.id
            elif isinstance(arg, (ast.List, ast.Tuple)):
                vals = []
                elts = arg.elts
                for el in elts:
                    try:
                        vals.append(ast.literal_eval(el))
                    except Exception:
                        if isinstance(el, ast.Name):
                            vals.append(el.id)
                        elif isinstance(el, ast.Constant) and el.value == 1:
                            vals.append(1)
                        else:
                            raise ValueError("unsupported f() positional weights") from None
                out["weights"] = vals
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
            elif isinstance(kw.value, ast.Call) and isinstance(kw.value.func, ast.Name):
                # Allow dict(model='ar1') / list(...) style constructors
                fname = kw.value.func.id
                if fname == "dict":
                    d = {}
                    for skw in kw.value.keywords:
                        if skw.arg is None:
                            continue
                        try:
                            d[skw.arg] = ast.literal_eval(skw.value)
                        except Exception:
                            if isinstance(skw.value, ast.Name):
                                d[skw.arg] = skw.value.id
                            else:
                                raise ValueError(
                                    f"unsupported f() argument: {key}=dict(...)"
                                ) from None
                    out[key] = d
                else:
                    raise ValueError(f"unsupported f() argument: {key}=...") from None
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

    for full, idx, rest in _find_f_calls(rhs):
        kw = _parse_f_kwargs(rest or "")
        model = str(kw.pop("model", "iid")).lower()
        copy = kw.get("copy")
        if copy is not None:
            model = "copy"
        order = int(kw.pop("order", 0) or 0)
        graph = kw.pop("graph", None)
        raw_scale = kw.pop("scale_model", None)
        scale_model = None if raw_scale is None else bool(raw_scale)
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
        stripped = stripped.replace(full, " ", 1)

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
    if re.search(r"(^|[\s+])-\s*1([\s+]|$)", rhs) or re.search(r"(^|[\s+])0\s*\+", rhs):
        intercept = False

    return ParsedFormula(
        response=response,
        fixed_terms=fixed_terms,
        intercept=intercept,
        f_terms=f_terms,
    )
