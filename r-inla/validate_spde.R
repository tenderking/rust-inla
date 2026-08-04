# SPDE mesh / FEM / field demo using rust-inla only (no classic INLA package).
#
# Mirrors the usual R-INLA workflow:
#   coords → mesh → Matérn SPDE FEM (C, G) → Q → sample field → figures
#
# Usage (from r-inla/):
#   Rscript validate_spde.R
# or via validate_spde.sh (builds the shared library first).

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_dir <- if (length(file_arg)) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1])))
} else {
  getwd()
}
setwd(script_dir)

source("R/inla_rs.R")
lib <- file.path("..", "target", "release", "libinla_rs.so")
if (!file.exists(lib)) {
  lib <- file.path("..", "target", "debug", "libinla_rs.so")
}
.inla_rs_dynload(lib)

if (!requireNamespace("Matrix", quietly = TRUE)) {
  stop("Package 'Matrix' is required", call. = FALSE)
}

out_dir <- file.path(script_dir, "spde_validation")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# 1. Dummy observation coordinates
set.seed(42)
n_points <- 200L
coords <- matrix(runif(n_points * 2L, 0, 10), ncol = 2L)
colnames(coords) <- c("x", "y")

# 2. Triangulated mesh (lattice stand-in for inla.mesh.2d)
pad <- 0.5
mesh <- inla_rs_lattice_mesh(
  xlim = range(coords[, 1]) + c(-pad, pad),
  ylim = range(coords[, 2]) + c(-pad, pad),
  nx = 21L,
  ny = 21L
)
verts <- mesh$vertices
tris <- mesh$triangles

# 3–4. FEM blocks + Matérn SPDE precision (alpha = 2 style: τ²(κ⁴C + 2κ²G + G C⁻¹ G))
fem <- inla_rs_fem_blocks_mesh(verts, tris)
C_matrix <- fem$c0
G_matrix <- fem$g1

kappa <- 1.5
tau <- 1.0
Q <- inla_rs_spde_precision_mesh_csc(verts, tris, kappa = kappa, tau = tau)

cat("Mesh nodes (n):", nrow(verts), "\n")
cat("Number of triangles:", nrow(tris), "\n")
cat("Dimensions of C (Mass):", paste(dim(C_matrix), collapse = " x "), "\n")
cat("Dimensions of G (Stiffness):", paste(dim(G_matrix), collapse = " x "), "\n")
cat("Q nnz:", length(Q@x), "\n")

# Sample x ~ N(0, Q^{-1}) via dense Cholesky (mesh is moderate)
set.seed(42)
Qd <- as.matrix(Q)
Qd <- (Qd + t(Qd)) / 2
# Tiny ridge for numerical SPD (boundary modes can be near-singular)
diag(Qd) <- diag(Qd) + 1e-8
Rchol <- chol(Qd)
z <- rnorm(nrow(Qd))
sample_field <- as.numeric(backsolve(Rchol, z))

# Project sample onto observation locations for a quick fit check
A_obs <- inla_rs_spde_projector_csc(verts, tris, coords[, 1], coords[, 2])
eta_true <- as.numeric(A_obs %*% sample_field)
y_obs <- eta_true + rnorm(n_points, sd = 0.15)

res <- inla_rs_spde(
  y = y_obs,
  loc = coords,
  vertices = verts,
  triangles = tris,
  initial_theta = c(log(tau), log(kappa)),
  obs_precision = 1 / (0.15^2),
  constrain = FALSE
)
cat(
  "SPDE fit mode (log_tau, log_kappa):",
  paste(round(res$mode, 4), collapse = ", "),
  "\n"
)
cat("SPDE mlik:", round(res$marginal_log_lik, 4), "\n")

# ==========================================
# EXPORTING FIGURES
# ==========================================

plot_mesh <- function(verts, tris, main = "Mesh") {
  plot(
    verts,
    type = "n",
    asp = 1,
    xlab = "X",
    ylab = "Y",
    main = main
  )
  for (i in seq_len(nrow(tris))) {
    idx <- c(tris[i, ], tris[i, 1])
    lines(verts[idx, 1], verts[idx, 2], col = "grey40", lwd = 0.4)
  }
}

# Figure 1: Triangulated mesh + observation points
png(file.path(out_dir, "spde_mesh_validation.png"), width = 800, height = 800, res = 120)
plot_mesh(verts, tris, main = "SPDE Triangulated Mesh (rust-inla)")
points(coords, col = "red", pch = 16, cex = 0.55)
dev.off()

# Figure 2: Sparsity pattern of stiffness G
png(file.path(out_dir, "spde_stiffness_sparsity.png"), width = 800, height = 800, res = 120)
Matrix::image(
  G_matrix,
  main = "Sparsity Pattern of Stiffness Matrix (G)",
  sub = "Non-zero elements are highlighted",
  xlab = "column",
  ylab = "row"
)
dev.off()

# Figure 3: Simulated Matérn field on a regular projection grid
gx <- seq(min(verts[, 1]), max(verts[, 1]), length.out = 80L)
gy <- seq(min(verts[, 2]), max(verts[, 2]), length.out = 80L)
grid <- as.matrix(expand.grid(x = gx, y = gy))
A_grid <- inla_rs_spde_projector_csc(verts, tris, grid[, 1], grid[, 2])
field_vec <- as.numeric(A_grid %*% sample_field)
field_grid <- matrix(field_vec, nrow = length(gx), ncol = length(gy))

png(file.path(out_dir, "spde_simulated_field.png"), width = 800, height = 800, res = 120)
image(
  gx,
  gy,
  field_grid,
  col = terrain.colors(100),
  main = "Simulated Matérn Spatial Field (rust-inla)",
  xlab = "X",
  ylab = "Y",
  asp = 1
)
contour(gx, gy, field_grid, add = TRUE, col = "black", labcex = 0.6)
dev.off()

# Figure 4: Posterior mean field from SPDE fit
field_fit <- matrix(
  as.numeric(A_grid %*% res$latent_means),
  nrow = length(gx),
  ncol = length(gy)
)
png(file.path(out_dir, "spde_fitted_field.png"), width = 800, height = 800, res = 120)
image(
  gx,
  gy,
  field_fit,
  col = terrain.colors(100),
  main = "Fitted SPDE Posterior Mean Field",
  xlab = "X",
  ylab = "Y",
  asp = 1
)
points(coords, pch = 16, cex = 0.35, col = adjustcolor("black", 0.35))
dev.off()

cat("Figures exported to:", normalizePath(out_dir), "\n")
