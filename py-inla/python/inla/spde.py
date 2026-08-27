"""SPDE / mesh helpers for the Python front-end."""

from __future__ import annotations

from typing import Any

import numpy as np

from inla._native import (
    fem_blocks_mesh as _fem_blocks_mesh,
)
from inla._native import (
    fem_blocks_mesh_1d as _fem_blocks_mesh_1d,
)
from inla._native import (
    spde_precision_matrix,
    spde_precision_matrix_1d,
    spde_projector_matrix,
    spde_projector_matrix_1d,
)


def lattice_mesh(
    xlim: tuple[float, float] = (0.0, 1.0),
    ylim: tuple[float, float] = (0.0, 1.0),
    nx: int = 11,
    ny: int = 11,
) -> dict[str, Any]:
    """Regular triangular lattice over a rectangle.

    Stand-in for classic ``inla.mesh.2d``: ``nx`` × ``ny`` vertices, each cell
    split into two triangles. Indices are 0-based.
    """
    nx = int(nx)
    ny = int(ny)
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must be >= 2")
    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[0], ylim[1], ny)
    # x varies fastest (column-major grid), matching R expand.grid(x, y)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    vertices = np.column_stack([xx.ravel(order="C"), yy.ravel(order="C")])

    def idx(i: int, j: int) -> int:
        return j * nx + i

    tris: list[tuple[int, int, int]] = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            v00, v10 = idx(i, j), idx(i + 1, j)
            v01, v11 = idx(i, j + 1), idx(i + 1, j + 1)
            tris.append((v00, v10, v01))
            tris.append((v10, v11, v01))
    triangles = np.asarray(tris, dtype=np.int64)
    return {
        "kind": "2d",
        "vertices": vertices,
        "triangles": triangles,
        "nx": nx,
        "ny": ny,
        "idx": np.arange(vertices.shape[0]),
        "n": int(vertices.shape[0]),
    }


def mesh_1d(loc) -> dict[str, Any]:
    """1D mesh on strictly increasing knots (classic ``inla.mesh.1d``)."""
    knots = np.asarray(loc, dtype=float).reshape(-1)
    if knots.size < 2:
        raise ValueError("mesh_1d requires at least two knots")
    if np.any(~np.isfinite(knots)):
        raise ValueError("mesh_1d knots must be finite")
    if np.any(np.diff(knots) <= 0.0):
        raise ValueError("mesh_1d knots must be strictly increasing")
    return {
        "kind": "1d",
        "loc": knots.copy(),
        "idx": np.arange(knots.size),
        "n": int(knots.size),
    }


def fem_blocks_mesh(
    vertices: np.ndarray | list | None = None,
    triangles: np.ndarray | list | None = None,
    *,
    loc=None,
    barrier_triangles=None,
    range_fraction: float | None = None,
    diffusion=None,
) -> dict[str, Any]:
    """FEM mass (``c0`` / C) and stiffness (``g1`` / G) as ``PyCscMatrix``."""
    if loc is not None:
        knots = np.asarray(loc, dtype=float).reshape(-1)
        return _fem_blocks_mesh_1d(knots.tolist())
    if vertices is None or triangles is None:
        raise ValueError("fem_blocks_mesh requires vertices and triangles, or loc=")
    verts = _as_vertex_tuples(vertices)
    tris = _as_triangle_tuples(triangles)
    kwargs: dict[str, Any] = {}
    if barrier_triangles is not None:
        kwargs["barrier_triangles"] = [int(t) for t in np.asarray(barrier_triangles).ravel()]
    if range_fraction is not None:
        kwargs["range_fraction"] = float(range_fraction)
    if diffusion is not None:
        kwargs["diffusion"] = [float(x) for x in np.asarray(diffusion).ravel()]
    return _fem_blocks_mesh(verts, tris, **kwargs)


def _as_vertex_tuples(vertices: np.ndarray | list) -> list[tuple[float, float]]:
    arr = np.asarray(vertices, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("vertices must be an N x 2 array")
    return [(float(x), float(y)) for x, y in arr]


def _as_triangle_tuples(
    triangles: np.ndarray | list,
) -> list[tuple[int, int, int]]:
    arr = np.asarray(triangles)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("triangles must be an M x 3 array")
    return [(int(a), int(b), int(c)) for a, b, c in arr]


def _mesh_kind(mesh: dict[str, Any]) -> str:
    kind = mesh.get("kind")
    if kind in ("1d", "2d"):
        return str(kind)
    if "loc" in mesh and "vertices" not in mesh:
        return "1d"
    return "2d"


def precision_matrix(
    vertices: np.ndarray | list | None = None,
    triangles: np.ndarray | list | None = None,
    kappa: float = 1.0,
    tau: float = 1.0,
    *,
    loc=None,
    barrier_triangles=None,
    range_fraction: float | None = None,
    diffusion=None,
):
    """Matérn SPDE precision Q(κ, τ) on a triangular or 1D mesh."""
    if loc is not None:
        knots = np.asarray(loc, dtype=float).reshape(-1)
        return spde_precision_matrix_1d(knots.tolist(), float(kappa), float(tau))
    if vertices is None or triangles is None:
        raise ValueError("precision_matrix requires vertices and triangles, or loc=")
    kwargs: dict[str, Any] = {}
    if barrier_triangles is not None:
        kwargs["barrier_triangles"] = [int(t) for t in np.asarray(barrier_triangles).ravel()]
    if range_fraction is not None:
        kwargs["range_fraction"] = float(range_fraction)
    if diffusion is not None:
        kwargs["diffusion"] = [float(x) for x in np.asarray(diffusion).ravel()]
    return spde_precision_matrix(
        _as_vertex_tuples(vertices),
        _as_triangle_tuples(triangles),
        float(kappa),
        float(tau),
        **kwargs,
    )


def projector_matrix(
    vertices: np.ndarray | list,
    triangles: np.ndarray | list,
    loc_x: np.ndarray | list,
    loc_y: np.ndarray | list,
):
    """Piecewise-linear 2D observation projector A (n_obs × n_vertices)."""
    return spde_projector_matrix(
        _as_vertex_tuples(vertices),
        _as_triangle_tuples(triangles),
        [float(x) for x in np.asarray(loc_x, dtype=float).ravel()],
        [float(y) for y in np.asarray(loc_y, dtype=float).ravel()],
    )


def make_A(mesh: dict[str, Any], loc, loc_y=None):
    """Build the FEM projector at observation locations.

    For a 1D mesh, ``loc`` is a 1-D coordinate vector. For a 2D mesh, ``loc``
    is an ``n × 2`` array, or pass ``loc`` / ``loc_y`` as separate vectors.
    """
    kind = _mesh_kind(mesh)
    if kind == "1d":
        knots = np.asarray(mesh["loc"], dtype=float).reshape(-1)
        pts = np.asarray(loc, dtype=float).reshape(-1)
        return spde_projector_matrix_1d(knots.tolist(), pts.tolist())
    loc_arr = np.asarray(loc, dtype=float)
    if loc_y is None:
        if loc_arr.ndim != 2 or loc_arr.shape[1] != 2:
            raise ValueError("2D make_A expects loc as n x 2, or loc and loc_y")
        xs = loc_arr[:, 0]
        ys = loc_arr[:, 1]
    else:
        xs = loc_arr.reshape(-1)
        ys = np.asarray(loc_y, dtype=float).reshape(-1)
    return projector_matrix(mesh["vertices"], mesh["triangles"], xs, ys)


def make_A_1d(mesh: dict[str, Any], loc):
    """Alias for :func:`make_A` on a 1D mesh."""
    return make_A(mesh, loc)


def matern(mesh: dict[str, Any]) -> dict[str, Any]:
    """Matérn SPDE model handle for ``f(..., model='spde', spde_model=...)``."""
    out = dict(mesh)
    out["spde"] = "matern"
    return out


def matern_1d(mesh: dict[str, Any]) -> dict[str, Any]:
    """Alias for :func:`matern` on a 1D mesh."""
    if _mesh_kind(mesh) != "1d":
        raise ValueError("matern_1d requires a 1D mesh from mesh_1d()")
    return matern(mesh)


def triangles_in_x_range(mesh: dict[str, Any], x0: float, x1: float) -> list[int]:
    """0-based indices of triangles whose centroid x lies in ``[x0, x1]``."""
    if _mesh_kind(mesh) != "2d":
        raise ValueError("triangles_in_x_range requires a 2D mesh")
    verts = np.asarray(mesh["vertices"], dtype=float)
    tris = np.asarray(mesh["triangles"])
    lo, hi = (min(x0, x1), max(x0, x1))
    out: list[int] = []
    for k, (a, b, c) in enumerate(tris):
        cx = (verts[int(a), 0] + verts[int(b), 0] + verts[int(c), 0]) / 3.0
        if lo <= cx <= hi:
            out.append(k)
    return out


def barrier_matern(
    mesh: dict[str, Any],
    barrier_triangles,
    range_fraction: float = 0.1,
) -> dict[str, Any]:
    """Matérn SPDE with a Bakka-style range barrier (still θ = [log τ, log κ])."""
    if _mesh_kind(mesh) != "2d":
        raise ValueError("barrier_matern requires a 2D triangular mesh")
    out = matern(mesh)
    out["barrier_triangles"] = [int(t) for t in np.asarray(barrier_triangles).ravel()]
    out["range_fraction"] = float(range_fraction)
    return out


def anisotropic_matern(mesh: dict[str, Any], diffusion) -> dict[str, Any]:
    """Matérn SPDE with a fixed anisotropic diffusion tensor ``[hxx, hxy, hyy]``."""
    if _mesh_kind(mesh) != "2d":
        raise ValueError("anisotropic_matern requires a 2D triangular mesh")
    hs = [float(x) for x in np.asarray(diffusion, dtype=float).ravel()]
    if len(hs) != 3:
        raise ValueError("diffusion must be length 3 [hxx, hxy, hyy]")
    out = matern(mesh)
    out["diffusion"] = hs
    return out


__all__ = [
    "lattice_mesh",
    "mesh_1d",
    "fem_blocks_mesh",
    "precision_matrix",
    "projector_matrix",
    "make_A",
    "make_A_1d",
    "matern",
    "matern_1d",
    "barrier_matern",
    "anisotropic_matern",
    "triangles_in_x_range",
    "spde_precision_matrix",
    "spde_projector_matrix",
    "spde_precision_matrix_1d",
    "spde_projector_matrix_1d",
]
