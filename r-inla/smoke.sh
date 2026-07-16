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
Rscript -e '
source("R/rinla_core.R")
.rinla_core_dynload("../target/release/librinla.so")

cat("--- Testing basic mesh & AR1 ---\n")
m <- rinla_core_read_mesh("../crates/inla_core/examples/mesh_xy.txt")
q <- rinla_core_ar1_precision(5L, 0.7, 1.0)
q_csc <- rinla_core_ar1_precision_csc(5L, 0.7, 1.0)
cat("mesh_n=", m$n_vertices,
    " q_dim=", paste(dim(q$q), collapse="x"),
    " nnz=", length(q$x),
    " csc_class=", class(q_csc)[1],
    " csc_nnz=", length(q_csc@x), "\n", sep="")

cat("\n--- Testing FGN precision ---\n")
q_fgn <- rinla_core_fgn_precision_csc(5L, 0.7, 1.5)
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

res_ar1 <- rinla_core_inla(y ~ f(idx, model = "ar1", obs_precision = 25.0), data = df)
cat("AR1 Hyperparameter Mode (log_tau, logit_rho):", paste(round(res_ar1$mode, 4), collapse = ", "), "\n")
cat("AR1 Marginal Log-Likelihood:", round(res_ar1$marginal_log_lik, 4), "\n")

cat("\n--- Testing Formula Parser & Inference (FGN) ---\n")
res_fgn <- rinla_core_inla(y ~ f(idx, model = "fgn", obs_precision = 25.0), data = df)
cat("FGN Hyperparameter Mode (log_tau, logit_hurst):", paste(round(res_fgn$mode, 4), collapse = ", "), "\n")
cat("FGN Marginal Log-Likelihood:", round(res_fgn$marginal_log_lik, 4), "\n")

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
n_val <- 250
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
  cat("  [Fit] Fitting FGN model via INLA...\n")
  t_fit <- system.time({
    res_val <- rinla_core_inla(y ~ f(idx, model = "fgn", obs_precision = 1000.0), data = df_fgn)
  })

  est_logit_H <- res_val$mode[2]
  est_H <- 1.0 / (1.0 + exp(-est_logit_H))
  est_tau <- exp(res_val$mode[1])

  cat(sprintf("  [Fit] Completed in %.3f seconds\n", t_fit["elapsed"]))
  cat(sprintf("  [Result] Real H = %.1f | Est H = %.4f | Est tau = %.4f\n",
              H_true, est_H, est_tau))
}
'
