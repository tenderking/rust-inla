"""SPDE / mesh helpers for the Python front-end."""

from __future__ import annotations

from typing import Any

import numpy as np

from inla._native import (
    fem_blocks_mesh as _fem_blocks_mesh,
    spde_precision_matrix,
    spde_projector_matrix,
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
        "vertices": vertices,
        "triangles": triangles,
        "nx": nx,
        "ny": ny,
    }


def fem_blocks_mesh(
    vertices: np.ndarray | list,
    triangles: np.ndarray | list,
) -> dict[str, Any]:
    """FEM mass (``c0`` / C) and stiffness (``g1`` / G) as ``PyCscMatrix``."""
    verts = _as_vertex_tuples(vertices)
    tris = _as_triangle_tuples(triangles)
    return _fem_blocks_mesh(verts, tris)


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


def precision_matrix(
    vertices: np.ndarray | list,
    triangles: np.ndarray | list,
    kappa: float,
    tau: float = 1.0,
):
    """Matérn SPDE precision Q(κ, τ) on a triangular mesh."""
    return spde_precision_matrix(
        _as_vertex_tuples(vertices),
        _as_triangle_tuples(triangles),
        float(kappa),
        float(tau),
    )


def projector_matrix(
    vertices: np.ndarray | list,
    triangles: np.ndarray | list,
    loc_x: np.ndarray | list,
    loc_y: np.ndarray | list,
):
    """Piecewise-linear observation projector A (n_obs × n_vertices)."""
    return spde_projector_matrix(
        _as_vertex_tuples(vertices),
        _as_triangle_tuples(triangles),
        [float(x) for x in np.asarray(loc_x, dtype=float).ravel()],
        [float(y) for y in np.asarray(loc_y, dtype=float).ravel()],
    )


__all__ = [
    "lattice_mesh",
    "fem_blocks_mesh",
    "precision_matrix",
    "projector_matrix",
    "spde_precision_matrix",
    "spde_projector_matrix",
]
