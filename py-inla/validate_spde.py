#!/usr/bin/env python3
"""SPDE mesh / FEM / field demo using rust-inla only (no classic R-INLA).

Mirrors r-inla/validate_spde.R:
  coords → lattice mesh → FEM (C, G) → Q → sample field → fit → figures
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow running before install if PYTHONPATH includes python/
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "python"))

from inla import _native as core  # noqa: E402
from inla.spde import (  # noqa: E402
    fem_blocks_mesh,
    lattice_mesh,
    precision_matrix,
    projector_matrix,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for figure export; install with: pip install matplotlib"
    ) from exc


def main() -> None:
    out_dir = HERE / "spde_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    n_points = 200
    coords = rng.uniform(0.0, 10.0, size=(n_points, 2))

    pad = 0.5
    mesh = lattice_mesh(
        xlim=(coords[:, 0].min() - pad, coords[:, 0].max() + pad),
        ylim=(coords[:, 1].min() - pad, coords[:, 1].max() + pad),
        nx=21,
        ny=21,
    )
    verts = mesh["vertices"]
    tris = mesh["triangles"]
    vert_tuples = [(float(x), float(y)) for x, y in verts]
    tri_tuples = [(int(a), int(b), int(c)) for a, b, c in tris]

    fem = fem_blocks_mesh(verts, tris)
    # Keep PyCscMatrix alive: to_scipy() is a view into Rust buffers.
    c_py, g_py = fem["c0"], fem["g1"]
    c_matrix = c_py.to_scipy().tocsc().copy()
    g_matrix = g_py.to_scipy().tocsc().copy()

    kappa = 1.5
    tau = 1.0
    q_py = precision_matrix(verts, tris, kappa=kappa, tau=tau)
    q = q_py.to_scipy().tocsc().copy()

    print(f"Mesh nodes (n): {verts.shape[0]}")
    print(f"Number of triangles: {tris.shape[0]}")
    print(f"Dimensions of C (Mass): {c_matrix.shape[0]} x {c_matrix.shape[1]}")
    print(f"Dimensions of G (Stiffness): {g_matrix.shape[0]} x {g_matrix.shape[1]}")
    print(f"Q nnz: {q.nnz}")

    # Sample x ~ N(0, Q^{-1}) via dense Cholesky
    qd = q.toarray()
    qd = 0.5 * (qd + qd.T)
    qd.flat[:: qd.shape[0] + 1] += 1e-8
    rchol = np.linalg.cholesky(qd)
    sample_field = np.linalg.solve(rchol.T, rng.standard_normal(qd.shape[0]))

    a_obs_py = projector_matrix(verts, tris, coords[:, 0], coords[:, 1])
    a_obs = a_obs_py.to_scipy().tocsc().copy()
    eta_true = a_obs @ sample_field
    y_obs = eta_true + rng.normal(0.0, 0.15, size=n_points)

    def build_prior(theta):
        t = float(np.exp(theta[0]))
        k = float(np.exp(theta[1]))
        return core.spde_precision_matrix(vert_tuples, tri_tuples, kappa=k, tau=t)

    def log_prior(theta):
        return float(-0.05 * (theta[0] ** 2 + theta[1] ** 2))

    obs = [{"family": "gaussian", "y": float(yi), "precision": 1.0 / (0.15**2)} for yi in y_obs]
    a_py = core.spde_projector_matrix(
        vert_tuples, tri_tuples, coords[:, 0].tolist(), coords[:, 1].tolist()
    )
    res = core.run_inla_inference(
        [np.log(tau), np.log(kappa)],
        build_prior,
        log_prior,
        obs,
        strategy="ccd",
        a=a_py,
    )
    print(
        "SPDE fit mode (log_tau, log_kappa):",
        ", ".join(f"{m:.4f}" for m in res.mode),
    )
    print(f"SPDE mlik: {res.marginal_log_lik:.4f}")

    # --- figures ---
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    for a, b, c in tris:
        idx = [a, b, c, a]
        ax.plot(verts[idx, 0], verts[idx, 1], color="0.4", lw=0.4)
    ax.scatter(coords[:, 0], coords[:, 1], c="red", s=8, zorder=3)
    ax.set_aspect("equal")
    ax.set_title("SPDE Triangulated Mesh (rust-inla)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.tight_layout()
    fig.savefig(out_dir / "spde_mesh_validation.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    ax.spy(g_matrix, markersize=0.4, color="black")
    ax.set_title("Sparsity Pattern of Stiffness Matrix (G)")
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    fig.tight_layout()
    fig.savefig(out_dir / "spde_stiffness_sparsity.png")
    plt.close(fig)

    gx = np.linspace(verts[:, 0].min(), verts[:, 0].max(), 80)
    gy = np.linspace(verts[:, 1].min(), verts[:, 1].max(), 80)
    grid_x, grid_y = np.meshgrid(gx, gy, indexing="ij")
    a_grid_py = projector_matrix(verts, tris, grid_x.ravel(), grid_y.ravel())
    a_grid = a_grid_py.to_scipy().tocsc().copy()
    field_grid = (a_grid @ sample_field).reshape(grid_x.shape)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    im = ax.imshow(
        field_grid.T,
        origin="lower",
        extent=(gx.min(), gx.max(), gy.min(), gy.max()),
        cmap="terrain",
        aspect="equal",
    )
    ax.contour(gx, gy, field_grid.T, colors="black", linewidths=0.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Simulated Matérn Spatial Field (rust-inla)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.tight_layout()
    fig.savefig(out_dir / "spde_simulated_field.png")
    plt.close(fig)

    field_fit = (a_grid @ np.asarray(res.latent_means, dtype=float)).reshape(grid_x.shape)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    im = ax.imshow(
        field_fit.T,
        origin="lower",
        extent=(gx.min(), gx.max(), gy.min(), gy.max()),
        cmap="terrain",
        aspect="equal",
    )
    ax.scatter(coords[:, 0], coords[:, 1], c="k", s=4, alpha=0.35)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Fitted SPDE Posterior Mean Field")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.tight_layout()
    fig.savefig(out_dir / "spde_fitted_field.png")
    plt.close(fig)

    print(f"Figures exported to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
