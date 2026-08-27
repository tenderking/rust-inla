"""``inla.stack``-style alignment of responses, projectors, and effects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import sparse


def _as_csc(mat, n_obs: int, n_col: int | None = None):
    if sparse.issparse(mat):
        a = mat.tocsc()
    else:
        arr = np.asarray(mat, dtype=float)
        if arr.ndim == 0 or arr.size == 1:
            val = float(arr.reshape(-1)[0])
            a = sparse.csc_matrix(np.full((n_obs, 1), val))
        elif arr.ndim == 1:
            if arr.size == n_obs:
                a = sparse.diags(arr, format="csc")
            else:
                a = sparse.csc_matrix(arr.reshape(n_obs, -1))
        else:
            a = sparse.csc_matrix(arr)
    if a.shape[0] != n_obs:
        raise ValueError(f"projector row count {a.shape[0]} != n_obs {n_obs}")
    if n_col is not None and a.shape[1] != n_col:
        raise ValueError(f"projector col count {a.shape[1]} != effect size {n_col}")
    return a


def _ones_or_identity(spec, n_obs: int, n_col: int):
    if isinstance(spec, (bool, int, float)) and spec in (1, 1.0, True):
        if n_col == 1:
            return sparse.csc_matrix(np.ones((n_obs, 1)))
        if n_col == n_obs:
            return sparse.eye(n_obs, format="csc")
        return sparse.csc_matrix(np.ones((n_obs, n_col)))
    return _as_csc(spec, n_obs, n_col)


def _effect_size(effect: Any, n_obs: int) -> int:
    if isinstance(effect, Mapping):
        if not effect:
            return 1
        return _effect_size(next(iter(effect.values())), n_obs)
    arr = np.asarray(effect)
    if arr.ndim == 0 or arr.size == 1:
        return 1
    return int(arr.reshape(-1).size)


def _effect_name(effect: Any, fallback: str) -> str:
    if isinstance(effect, Mapping) and effect:
        return str(next(iter(effect.keys())))
    return fallback


def _broadcast_data_value(value, n_obs: int) -> np.ndarray:
    if value is None:
        return np.full(n_obs, np.nan)
    arr = np.asarray(value, dtype=float)
    if arr.ndim > 1:
        if arr.shape[0] == n_obs:
            return arr
        raise ValueError(f"data array shape {arr.shape} is incompatible with n_obs={n_obs}")
    vec = arr.reshape(-1)
    if vec.size == 1:
        return np.full(n_obs, float(vec[0]))
    if vec.size != n_obs:
        raise ValueError(f"data length {vec.size} != n_obs {n_obs}")
    return vec


def _infer_n_obs(data: Mapping[str, Any]) -> int:
    for val in data.values():
        if val is None:
            continue
        arr = np.asarray(val)
        if arr.ndim == 0:
            continue
        if arr.size > 1 or arr.ndim >= 1:
            return int(arr.shape[0])
    return 1


class Stack:
    """Align observation rows, projector blocks, and named latent effects.

    Mirrors classic ``inla.stack``: each component has ``data``, a list of
    projector blocks ``A``, and matching ``effects``. :meth:`join` row-binds
    stacks and pads missing effect columns with zeros.
    """

    def __init__(
        self,
        data: Mapping[str, Any],
        A: Sequence[Any],
        effects: Sequence[Any] | None = None,
        tag: str = "est",
    ):
        if not isinstance(data, Mapping) or not data:
            raise ValueError("Stack data must be a non-empty mapping")
        n_obs = _infer_n_obs(data)
        stored: dict[str, np.ndarray] = {}
        for key, val in data.items():
            stored[str(key)] = _broadcast_data_value(val, n_obs)
        self.n_obs = int(n_obs)
        self.data = stored
        self.tag = str(tag)
        effect_list = list(effects) if effects is not None else [None] * len(A)
        if len(effect_list) != len(A):
            raise ValueError("A and effects must have the same length")
        blocks = []
        names: list[str] = []
        sizes: list[int] = []
        for i, (a_spec, eff) in enumerate(zip(A, effect_list)):
            name = _effect_name(eff, f"e{i}")
            n_col = _effect_size(eff, self.n_obs) if eff is not None else None
            if n_col is None:
                ident = isinstance(a_spec, (bool, int, float)) and a_spec in (1, 1.0, True)
                mat = _as_csc(np.ones((self.n_obs, 1)) if ident else a_spec, self.n_obs)
                n_col = int(mat.shape[1])
            else:
                mat = _ones_or_identity(a_spec, self.n_obs, n_col)
            blocks.append(mat)
            names.append(name)
            sizes.append(int(n_col))
            if isinstance(eff, Mapping):
                for k, v in eff.items():
                    if k in self.data:
                        continue
                    vec = np.asarray(v)
                    if vec.ndim == 0 or vec.size == 1:
                        self.data[str(k)] = np.full(self.n_obs, float(vec.reshape(-1)[0]))
                    elif vec.size == self.n_obs:
                        self.data[str(k)] = np.asarray(vec, dtype=float).reshape(-1)
        self.effect_names = names
        self.effect_sizes = sizes
        self.A = (
            sparse.hstack(blocks, format="csc") if blocks else sparse.csc_matrix((self.n_obs, 0))
        )
        self._tags = [(self.tag, 0, self.n_obs)]

    def index(self, tag: str) -> np.ndarray:
        """0-based row indices belonging to ``tag`` (after :meth:`join`)."""
        rows: list[int] = []
        for name, start, stop in self._tags:
            if name == tag:
                rows.extend(range(start, stop))
        if not rows:
            raise KeyError(f"unknown stack tag '{tag}'")
        return np.asarray(rows, dtype=int)

    @classmethod
    def join(cls, *stacks: Stack) -> Stack:
        if not stacks:
            raise ValueError("Stack.join requires at least one stack")
        if len(stacks) == 1:
            return stacks[0]
        names: list[str] = []
        seen: set[str] = set()
        sizes: dict[str, int] = {}
        for stk in stacks:
            for name, sz in zip(stk.effect_names, stk.effect_sizes):
                if name not in seen:
                    names.append(name)
                    seen.add(name)
                    sizes[name] = sz
                elif sizes[name] != sz:
                    raise ValueError(f"effect '{name}' has inconsistent size {sz} vs {sizes[name]}")
        blocks = []
        data_keys: set[str] = set()
        for stk in stacks:
            data_keys.update(stk.data)
            local = {
                name: mat for name, mat in zip(stk.effect_names, _split_blocks(stk), strict=True)
            }
            parts = []
            for name in names:
                if name in local:
                    parts.append(local[name])
                else:
                    parts.append(sparse.csc_matrix((stk.n_obs, sizes[name])))
            blocks.append(sparse.hstack(parts, format="csc"))
        a = sparse.vstack(blocks, format="csc")
        n_obs = int(a.shape[0])
        merged: dict[str, np.ndarray] = {}
        for key in data_keys:
            chunks = []
            for stk in stacks:
                if key in stk.data:
                    chunks.append(_broadcast_data_value(stk.data[key], stk.n_obs))
                else:
                    chunks.append(np.full(stk.n_obs, np.nan))
            merged[key] = np.concatenate(chunks, axis=0)
        out = cls.__new__(cls)
        out.n_obs = n_obs
        out.data = merged
        out.tag = "join"
        out.effect_names = names
        out.effect_sizes = [sizes[n] for n in names]
        out.A = a
        tags = []
        row = 0
        for stk in stacks:
            for name, start, stop in stk._tags:
                n = stop - start
                tags.append((name, row, row + n))
                row += n
        out._tags = tags
        return out


def _split_blocks(stk: Stack) -> list:
    mats = []
    off = 0
    for sz in stk.effect_sizes:
        mats.append(stk.A[:, off : off + sz])
        off += sz
    return mats
