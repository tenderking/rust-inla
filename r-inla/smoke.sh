#!/usr/bin/env bash
set -euo pipefail

# Find workspace root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v Rscript >/dev/null 2>&1; then
  echo "Rscript not found in PATH" >&2
  exit 1
fi

echo "[1/2] Building r-inla (release)..."
cargo build -p r-inla --release

echo "[2/2] Running R smoke test..."
Rscript - << 'EOF'
source("R/inla_rs.R")
source("R/summary.R")
source("R/plot.R")
.inla_rs_dynload("../target/release/libinla_rs.so")

cat("--- Testing basic mesh & AR1 ---\n")
m <- inla_rs_read_mesh("../crates/inla_fmesher/examples/mesh_xy.txt")
q <- inla_rs_ar1_precision(5L, 0.7, 1.0)
q_csc <- inla_rs_ar1_precision_csc(5L, 0.7, 1.0)
cat("mesh_n=", m$n_vertices,
    " q_dim=", paste(dim(q$q), collapse="x"),
    " nnz=", length(q$x),
    " csc_class=", class(q_csc)[1],
    " csc_nnz=", length(q_csc@x), "\n", sep="")

cat("\n--- Testing FGN precision ---\n")
q_fgn <- inla_rs_fgn_precision_csc(5L, 0.7, 1.5)
print(as.matrix(q_fgn))

cat("\n--- Testing Formula Parser & Inference (AR1) ---\n")
set.seed(42)
n <- 20
x <- numeric(n)
x[1] <- rnorm(1)
for (i in 2:n) {
  x[i] <- 0.7 * x[i-1] + rnorm(1, sd = 0.5)
}
y <- x + rnorm(n, sd = 0.2)
df <- data.frame(y = y, idx = 1:n)

res_ar1 <- inla_rs(y ~ -1 + f(idx, model = "ar1", obs_precision = 25.0), data = df)
cat("AR1 Hyperparameter Mode (log_tau, logit_rho):", paste(round(res_ar1$mode, 4), collapse = ", "), "\n")
cat("AR1 Marginal Log-Likelihood:", round(res_ar1$marginal_log_lik, 4), "\n")
stopifnot(inherits(res_ar1, "inla_rs"))
stopifnot(!is.null(res_ar1$summary.hyperpar))
stopifnot(!is.null(res_ar1$internal.marginals.hyperpar))
stopifnot(length(res_ar1$summary.random) >= 1L)
cat("AR1 summary.hyperpar:\n")
print(round(res_ar1$summary.hyperpar, 3))

cat("\n--- Testing Formula Parser & Inference (FGN) ---\n")
res_fgn <- inla_rs(y ~ -1 + f(idx, model = "fgn", obs_precision = 25.0), data = df)
cat("FGN Hyperparameter Mode (log_tau, logit_hurst):", paste(round(res_fgn$mode, 4), collapse = ", "), "\n")
cat("FGN Marginal Log-Likelihood:", round(res_fgn$marginal_log_lik, 4), "\n")

cat("\n--- Classic-style FGN formula (order=4 AR mixture) ---\n")
set.seed(7)
n_c <- 40
# Toy FGN-like series (exact sim not required for API smoke)
yc <- scale(as.numeric(arima.sim(list(ar = 0.8), n = n_c)))
df_c <- data.frame(y = as.numeric(yc), time = seq_len(n_c))
res_fgn4 <- inla_rs(
  y ~ -1 + f(time, model = "fgn", order = 4L),
  data = df_c,
  control.family = list(hyper = list(prec = list(initial = 8, fixed = TRUE)))
)
cat("FGN order=4 mode (log_tau, H_intern):", paste(round(res_fgn4$mode, 4), collapse = ", "), "\n")
# Structured API returns raw mode; convert H_intern → H for approx FGN
h_est <- if (!is.null(res_fgn4$hurst) && is.numeric(res_fgn4$hurst)) {
  res_fgn4$hurst
} else {
  # H = 0.5 * (1 + exp(H_intern)/(1+exp(H_intern)))  via fgn_hurst_from_intern
  # Match rust: 0.5 + 0.5 * logistic(H_intern)  — check fgn_hurst_from_intern
  hi <- res_fgn4$mode[2]
  0.5 + 0.5 / (1 + exp(-hi))
}
cat("FGN order=4 est H (R-INLA scale):", round(h_est, 4), "\n")

cat("\n--- Testing Formula Parser & Inference (RW2) ---\n")
t <- seq_len(n) / n
y_rw <- t^2 + rnorm(n, sd = 0.05)
df_rw <- data.frame(y = y_rw, idx = 1:n)
res_rw2 <- inla_rs(y ~ -1 + f(idx, model = "rw2", obs_precision = 100.0), data = df_rw)
cat("RW2 Hyperparameter Mode (log_tau):", paste(round(res_rw2$mode, 4), collapse = ", "), "\n")
cat("RW2 Marginal Log-Likelihood:", round(res_rw2$marginal_log_lik, 4), "\n")

cat("\n--- Testing Formula Parser & Inference (RW1, Seasonal, AR, CRW1, CRW2) ---\n")
res_rw1 <- inla_rs(y ~ -1 + f(idx, model = "rw1", obs_precision = 50.0), data = df)
cat("RW1 Marginal Log-Likelihood:", round(res_rw1$marginal_log_lik, 4), "\n")
stopifnot(is.finite(res_rw1$marginal_log_lik))

t_seas <- 1:24
df_seas <- data.frame(y = sin(t_seas * 0.5) + rnorm(24, sd = 0.1), idx = t_seas)
res_seas <- inla_rs(y ~ -1 + f(idx, model = "seasonal", season = 4L, obs_precision = 50.0), data = df_seas)
cat("Seasonal(4) Marginal Log-Likelihood:", round(res_seas$marginal_log_lik, 4), "\n")
stopifnot(is.finite(res_seas$marginal_log_lik))

res_ar2 <- inla_rs(y ~ -1 + f(idx, model = "ar", order = 2L, obs_precision = 50.0), data = df)
cat("AR(2) mode length:", length(res_ar2$mode), " mlik:", round(res_ar2$marginal_log_lik, 4), "\n")
stopifnot(length(res_ar2$mode) >= 3L)

pos_crw <- sort(runif(15, 0, 10))
df_crw <- data.frame(y = sin(pos_crw) + rnorm(15, sd = 0.1), idx = 1:15)
res_crw1 <- inla_rs(y ~ -1 + f(idx, model = "crw1", positions = pos_crw, obs_precision = 50.0), data = df_crw)
cat("CRW1 Marginal Log-Likelihood:", round(res_crw1$marginal_log_lik, 4), "\n")
stopifnot(is.finite(res_crw1$marginal_log_lik))

res_crw2 <- inla_rs(y ~ -1 + f(idx, model = "crw2", positions = pos_crw, layout = "simple", obs_precision = 50.0), data = df_crw)
cat("CRW2 simple Marginal Log-Likelihood:", round(res_crw2$marginal_log_lik, 4), "\n")
stopifnot(is.finite(res_crw2$marginal_log_lik))

cat("\n--- Non-Gaussian families (poisson / binomial / nbinom / zip / exp / weibull / laplace) ---\n")
set.seed(3)
counts <- c(2, 3, 2, 4, 3, 2, 3, 2)
df_p <- data.frame(y = counts, idx = seq_along(counts))
res_pois <- inla_rs(
  y ~ -1 + f(idx, model = "iid"),
  data = df_p,
  family = "poisson",
  initial_theta = 1.0
)
cat("Poisson+IID mode:", paste(round(res_pois$mode, 4), collapse = ", "), "\n")
cat("Poisson+IID mlik:", round(res_pois$marginal_log_lik, 4), "\n")

ys_b <- c(2, 5, 3, 7, 4, 6)
df_b <- data.frame(y = ys_b, idx = seq_along(ys_b))
res_bin <- inla_rs(
  y ~ -1 + f(idx, model = "iid"),
  data = df_b,
  family = "binomial",
  Ntrials = rep(10, length(ys_b)),
  initial_theta = 0.0
)
cat("Binomial+IID mode:", paste(round(res_bin$mode, 4), collapse = ", "), "\n")

res_nb <- inla_rs(
  y ~ -1 + f(idx, model = "iid"),
  data = df_p,
  family = "nbinomial",
  size = 2.0,
  initial_theta = 1.0
)
cat("NBinomial+IID mode:", paste(round(res_nb$mode, 4), collapse = ", "), "\n")

res_zip <- inla_rs(
  y ~ -1 + f(idx, model = "iid"),
  data = df_p,
  family = "zeroinflatedpoisson0",
  zero_prob = 0.2,
  initial_theta = 1.0
)
cat("ZIP+IID mode:", paste(round(res_zip$mode, 4), collapse = ", "), "\n")

df_surv <- data.frame(y = c(1.2, 2.5, 0.8, 3.1, 1.9, 2.0), event = c(1, 0, 1, 1, 0, 1), idx = 1:6)
res_exp <- inla_rs(
  y ~ -1 + f(idx, model = "iid"),
  data = df_surv,
  family = "exponential_survival",
  initial_theta = 0.0
)
cat("ExponentialSurvival+IID mode:", paste(round(res_exp$mode, 4), collapse = ", "), "\n")

res_weib <- inla_rs(
  y ~ -1 + f(idx, model = "iid"),
  data = df_surv,
  family = "weibull_survival",
  shape = 1.5,
  initial_theta = 0.0
)
cat("WeibullSurvival+IID mode:", paste(round(res_weib$mode, 4), collapse = ", "), "\n")

y_lap <- c(0.2, -0.1, 0.4, 0.0, -0.3, 0.1, 0.2, -0.2)
df_l <- data.frame(y = y_lap, idx = seq_along(y_lap))
res_lap <- inla_rs(
  y ~ -1 + f(idx, model = "iid"),
  data = df_l,
  family = "laplace",
  alpha = 0.5,
  gamma = 0.5,
  initial_theta = 1.0
)
cat("Laplace+IID mode:", paste(round(res_lap$mode, 4), collapse = ", "), "\n")

q_fgn4 <- inla_rs_fgn_approx_precision_csc(20L, 0.7, 1.0, order = 4L)
cat("FGN approx order=4 dim=", paste(dim(q_fgn4), collapse = "x"),
    " nnz=", length(q_fgn4@x), " (sparse)\n", sep = "")

cat("\n--- SPDE fit (mesh + projector A) ---\n")
verts <- matrix(as.numeric(c(0,0, 1,0, 1,1, 0,1, 0.5,0.5)), ncol=2, byrow=TRUE)
tris <- matrix(as.integer(c(1,2,5, 2,3,5, 3,4,5, 4,1,5)), ncol=3, byrow=TRUE)
loc <- rbind(c(0.25,0.25), c(0.75,0.25), c(0.75,0.75), c(0.25,0.75), c(0.5,0.5), c(0.4,0.6))
set.seed(11)
y_spde <- 0.4 * sin(loc[,1] * 2) + 0.3 * cos(loc[,2] * 1.5) + rnorm(nrow(loc), sd = 0.1)
A_spde <- inla_rs_spde_projector_csc(verts, tris, loc[,1], loc[,2])
cat("SPDE A dim=", paste(dim(A_spde), collapse="x"), " nnz=", length(A_spde@x), "\n", sep="")
res_spde <- inla_rs_spde(
  y = y_spde, loc = loc, vertices = verts, triangles = tris,
  initial_theta = c(0, 0), obs_precision = 50, constrain = FALSE
)
cat("SPDE mode (log_tau, log_kappa):", paste(round(res_spde$mode, 4), collapse = ", "), "\n")
cat("SPDE mlik:", round(res_spde$marginal_log_lik, 4), " n_latent=", res_spde$n_latent, "\n", sep="")
stopifnot(length(res_spde$latent_means) == res_spde$n_latent)
stopifnot(is.finite(res_spde$marginal_log_lik))

# Intentionally omit `copy` formula productization in this pass (shared β scaling).
# Python already covers rgeneric; R exposes inla_rs_rgeneric_define() + rw2d formula.

cat("\n--- Advanced latents: rw2d + rgeneric define ---\n")
set.seed(21)
nrow <- 5L; ncol <- 5L; n_g <- nrow * ncol
idx <- seq_len(n_g)
y_rw2d <- 0.15 * ((idx - 1L) %% nrow) + 0.1 * ((idx - 1L) %/% nrow) + rnorm(n_g, sd = 0.2)
df_rw2d <- data.frame(y = y_rw2d, idx = idx)
q_rw2d <- inla_rs_rw2d_precision_csc(nrow, ncol, tau = 1.0, cyclic = FALSE)
cat("rw2d Q dim=", paste(dim(q_rw2d), collapse = "x"), " nnz=", length(q_rw2d@x), "\n", sep = "")
res_rw2d <- inla_rs(
  y ~ -1 + f(idx, model = "rw2d", nrow = nrow, ncol = ncol, cyclic = FALSE),
  data = df_rw2d,
  family = "gaussian"
)
cat("rw2d mode:", paste(round(res_rw2d$mode, 4), collapse = ", "),
    " mlik=", round(res_rw2d$marginal_log_lik, 4), "\n", sep = "")
stopifnot(length(res_rw2d$latent_means) == n_g)
stopifnot(is.finite(res_rw2d$marginal_log_lik))

rg <- inla_rs_rgeneric_define(
  n = 5L,
  Q = function(theta) {
    Matrix::Diagonal(5, x = exp(theta[1]))
  },
  n_theta = 1L,
  initial = 0.0
)
stopifnot(identical(rg$n, 5L), is.function(rg$Q))
cat("rgeneric define ok: n=", rg$n, " n_theta=", rg$n_theta, "\n", sep = "")

cat("\n--- Gap models: BYM + matern2d ---\n")
set.seed(31)
n_reg <- 6L
adj <- lapply(seq_len(n_reg), function(i) {
  as.integer(c(((i - 2L) %% n_reg) + 1L, (i %% n_reg) + 1L))
})
y_bym <- rnorm(n_reg, sd = 0.4)
df_bym <- data.frame(y = y_bym, region = seq_len(n_reg))
res_bym <- inla_rs(
  y ~ -1 + f(region, model = "bym", graph = adj),
  data = df_bym
)
cat("BYM mode:", paste(round(res_bym$mode, 4), collapse = ", "),
    " n_latent=", length(res_bym$latent_means), "\n", sep = "")
stopifnot(length(res_bym$mode) == 2L, length(res_bym$latent_means) == 2L * n_reg)

res_bym2 <- inla_rs(
  y ~ -1 + f(region, model = "bym2", graph = adj),
  data = df_bym
)
cat("BYM2 mode:", paste(round(res_bym2$mode, 4), collapse = ", "),
    " n_latent=", length(res_bym2$latent_means), "\n", sep = "")
stopifnot(length(res_bym2$latent_means) == n_reg)

nrow_m <- 4L; ncol_m <- 4L; n_m <- nrow_m * ncol_m
y_m <- 0.2 * sin(seq_len(n_m) * 0.3) + rnorm(n_m, sd = 0.2)
df_m <- data.frame(y = y_m, idx = seq_len(n_m) - 1L)
res_m <- inla_rs(
  y ~ -1 + f(idx, model = "matern2d", nrow = nrow_m, ncol = ncol_m, cyclic = FALSE),
  data = df_m
)
cat("matern2d mode:", paste(round(res_m$mode, 4), collapse = ", "),
    " mlik=", round(res_m$marginal_log_lik, 4), "\n", sep = "")
stopifnot(length(res_m$mode) == 2L, is.finite(res_m$marginal_log_lik))

cat("\n--- FGN Parameter Estimation Validation ---\n")
sim_fgn <- function(n, H) {
  gamma <- function(k, H) {
    0.5 * (abs(k + 1)^(2 * H) - 2 * abs(k)^(2 * H) + abs(k - 1)^(2 * H))
  }
  Sigma <- outer(1:n, 1:n, function(i, j) gamma(abs(i - j), H))
  Sigma <- Sigma + diag(1e-9, n)
  L <- t(chol(Sigma))
  L %*% rnorm(n)
}

set.seed(123)
# Exact dense FGN: keep moderate n (O(n³)). For n≈500 use order=3/4 approx.
n_val <- 500
h_targets <- c(0.6, 0.7, 0.8, 0.9)

for (H_true in h_targets) {
  cat(sprintf("\n--- Starting H = %.1f ---\n", H_true))

  # 1. Profile the Simulation
  t_sim <- system.time({
    y_fgn <- sim_fgn(n_val, H_true)
  })
  cat(sprintf("  [Sim] Generated FGN data in %.3f seconds\n", t_sim["elapsed"]))

  df_fgn <- data.frame(y = y_fgn, idx = 1:n_val)

  # 2. Profile the INLA fit
  cat("  [Fit] Fitting FGN model via INLA (exact dense)...\n")
  t_fit <- system.time({
    res_val <- inla_rs(y ~ f(idx, model = "fgn", order = 0L, obs_precision = 1000.0), data = df_fgn)
  })

  cat(sprintf("  [Fit] Completed in %.3f seconds\n", t_fit["elapsed"]))
  est_H <- as.numeric(res_val$hurst)
  est_tau <- exp(res_val$mode[1])
  cat(sprintf("  [Result] Real H = %.1f | Est H = %.4f | Est tau = %.4f\n",
              H_true, est_H, est_tau))
}
EOF
