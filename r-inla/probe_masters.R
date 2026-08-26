# Masters compatibility probe — A-matrix / fixed effects / multi-f / scale.model / inla.group

ROOT_RINLA <- "/home/george/workspace/tenderking/rust-inla/r-inla"
MASTERS <- "/mnt/c/Users/g_m/projects/masters"
LIB <- "/home/george/workspace/tenderking/rust-inla/target/release/libinla_rs.so"

cat("=== masters probe (A-matrix era) ===\n")
source(file.path(ROOT_RINLA, "R/inla_rs.R"))
source(file.path(ROOT_RINLA, "R/summary.R"))
source(file.path(ROOT_RINLA, "R/plot.R"))
.inla_rs_dynload(LIB)

house <- read.csv(file.path(MASTERS, "output/house_data.csv"), stringsAsFactors = FALSE)
house$boligtype_id <- as.integer(as.character(house$boligtype_id))
house <- house[!is.na(house$ave_kvm_pris_norm), ]

n_reg <- 19L
connections <- list(
  c(2), c(1, 4, 6, 11, 12), c(13, 16, 18), c(2, 7, 11, 12, 15, 16, 19),
  c(17), c(2, 11, 14), c(4, 13, 15, 16), c(11, 15, 14), c(10, 14), c(9, 17),
  c(2, 4, 6, 8, 12, 14, 15), c(2), c(3, 16, 18), c(6, 8, 9, 11),
  c(4, 7, 8, 11), c(3, 4, 7, 13, 19), c(5, 10), c(3, 13), c(4, 16)
)
adj_mat <- matrix(0, n_reg, n_reg)
for (i in seq_along(connections)) {
  for (j in connections[[i]]) {
    adj_mat[i, j] <- 1
    adj_mat[j, i] <- 1
  }
}

pass <- 0L; fail <- 0L
report <- function(ok, label, detail = "") {
  if (isTRUE(ok)) {
    pass <<- pass + 1L
    cat("  PASS ", label, if (nzchar(detail)) paste0(" — ", detail) else "", "\n", sep = "")
  } else {
    fail <<- fail + 1L
    cat("  FAIL ", label, if (nzchar(detail)) paste0(" — ", detail) else "", "\n", sep = "")
  }
}

# Scale responses for numerical stability (PoC has no INLA-style centering)
oslo <- house[house$region_id == 3 & house$boligtype_id == 1, ]
oslo <- oslo[order(oslo$year), ]
oslo$idx <- seq_len(nrow(oslo))
oslo$y <- as.numeric(scale(oslo$ave_kvm_pris_norm))
oslo$year_c <- as.numeric(scale(oslo$year, scale = FALSE))

nye <- house[house$boligtype_id == 1, ]
nye$y <- as.numeric(scale(nye$ave_kvm_pris_norm))
nye$year_c <- as.numeric(scale(nye$year, scale = FALSE))
nye$year_g <- inla_rs_group(nye$year, n = 10L)

cat("\n--- Fixed effects: Oslo new ~ year ---\n")
res_fe <- tryCatch(
  inla_rs(
    y ~ year_c + f(idx, model = "iid"),
    data = oslo,
    control.family = list(hyper = list(prec = list(initial = log(10), fixed = TRUE))),
    initial_theta = 1.0,
    fixed_prec = 1e-3
  ),
  error = function(e) e
)
if (inherits(res_fe, "error")) {
  report(FALSE, "Oslo ~ year_c + iid", conditionMessage(res_fe))
} else {
  report(TRUE, "Oslo ~ year_c + iid",
         sprintf("mlik=%.2f n_lat=%d n_pred=%d",
                 res_fe$marginal_log_lik, length(res_fe$latent_means),
                 length(res_fe$predictor_means)))
}

cat("\n--- Unaggregated Besag (A-matrix, many obs/region) ---\n")
res_besag0 <- tryCatch(
  inla_rs(
    y ~ -1 + f(region_id, model = "besag", graph = adj_mat),
    data = nye,
    control.family = list(hyper = list(prec = list(initial = log(10), fixed = TRUE))),
    initial_theta = 0.0
  ),
  error = function(e) e
)
if (inherits(res_besag0, "error")) {
  report(FALSE, "besag unaggregated", conditionMessage(res_besag0))
} else {
  report(TRUE, "besag unaggregated",
         sprintf("n_obs=%d log_tau=%.3f mlik=%.2f",
                 nrow(nye), res_besag0$mode[1], res_besag0$marginal_log_lik))
}

cat("\n--- Besag + year (scale.model, no intercept) ---\n")
res_besag <- tryCatch(
  inla_rs(
    # Intrinsic Besag has a constant null space — drop intercept (-1).
    y ~ -1 + year_c + f(region_id, model = "besag", graph = adj_mat, scale.model = TRUE),
    data = nye,
    control.family = list(hyper = list(prec = list(initial = log(10), fixed = TRUE))),
    initial_theta = 0.0,
    fixed_prec = 1e-3
  ),
  error = function(e) e
)
if (inherits(res_besag, "error")) {
  report(FALSE, "besag+year scale.model", conditionMessage(res_besag))
} else {
  u <- res_besag$summary.random$region_id$mean
  report(TRUE, "besag+year scale.model",
         sprintf("log_tau=%.3f mlik=%.2f max|u|=%.2f",
                 res_besag$mode[1], res_besag$marginal_log_lik, max(abs(u))))
}

cat("\n--- inla_rs_group(year) + RW2 ---\n")
res_grp <- tryCatch(
  inla_rs(
    y ~ -1 + f(year_g, model = "rw2"),
    data = nye,
    control.family = list(hyper = list(prec = list(initial = log(10), fixed = TRUE))),
    initial_theta = 1.0
  ),
  error = function(e) e
)
if (inherits(res_grp, "error")) {
  report(FALSE, "inla_rs_group + rw2", conditionMessage(res_grp))
} else {
  report(TRUE, "inla_rs_group + rw2",
         sprintf("n_lat=%d mlik=%.2f", length(res_grp$latent_means), res_grp$marginal_log_lik))
}

cat("\n--- Multi-f: besag + iid(year_g) ---\n")
res_multi <- tryCatch(
  inla_rs(
    # RW2 is also improper; pair Besag with a proper IID year effect for the PoC.
    y ~ -1 + f(region_id, model = "besag", graph = adj_mat) +
      f(year_g, model = "iid"),
    data = nye,
    control.family = list(hyper = list(prec = list(initial = log(10), fixed = TRUE))),
    initial_theta = c(0.0, 1.0)
  ),
  error = function(e) e
)
if (inherits(res_multi, "error")) {
  report(FALSE, "besag + iid multi-f", conditionMessage(res_multi))
} else {
  report(TRUE, "besag + iid multi-f",
         sprintf("theta=(%.2f,%.2f) mlik=%.2f n_lat=%d",
                 res_multi$mode[1], res_multi$mode[2],
                 res_multi$marginal_log_lik, length(res_multi$latent_means)))
}

cat("\n=== Summary: ", pass, " pass / ", fail, " fail ===\n", sep = "")
if (fail > 0) quit(status = 1)
