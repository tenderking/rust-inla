# Fit the conformance model suite in R and emit a flat key/value report.
#
# Usage: Rscript fit_models.R <repo_root> <data_csv>
#
# The Python side (tests/test_cross_language_conformance.py) fits the identical
# models and diffs this report field by field. Keep the emitted keys in sync.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) stop("usage: fit_models.R <repo_root> <data_csv>")
root <- normalizePath(args[[1]])
csv <- normalizePath(args[[2]])

source(file.path(root, "r-inla", "R", "inla_rs.R"))
source(file.path(root, "r-inla", "R", "summary.R"))
.inla_rs_dynload(file.path(root, "target", "release", "libinla_rs.so"))

df <- read.csv(csv)

emit <- function(model, field, values) {
  cat(sprintf(
    "%s\t%s\t%s\n",
    model, field, paste(format(as.numeric(values), digits = 12), collapse = ",")
  ))
}

report <- function(model, res) {
  emit(model, "mode", res$mode)
  emit(model, "mlik", res$marginal_log_lik)
  emit(model, "mlik_gaussian", res$marginal_log_lik_gaussian)
  emit(model, "dic", res$dic)
  emit(model, "waic", res$waic)
  emit(model, "effective_params", res$effective_params)
  if (!is.null(res$summary.hyperpar)) {
    emit(model, "hyper_mean", res$summary.hyperpar$mean)
    emit(model, "hyper_sd", res$summary.hyperpar$sd)
    cat(sprintf(
      "%s\t%s\t%s\n", model, "hyper_labels",
      paste(rownames(res$summary.hyperpar), collapse = ",")
    ))
  }
  if (length(res$summary.random) > 0L) {
    emit(model, "random_mean", res$summary.random[[1]]$mean)
  }
}

report("ar1", inla_rs(
  y ~ -1 + f(idx, model = "ar1", obs_precision = 25.0),
  data = df
))

report("rw2", inla_rs(
  y ~ -1 + f(idx, model = "rw2", obs_precision = 100.0),
  data = df
))

report("iid", inla_rs(
  y ~ -1 + f(idx, model = "iid", obs_precision = 25.0),
  data = df
))

report("seasonal", inla_rs(
  y ~ -1 + f(idx, model = "seasonal", season = 4L, obs_precision = 50.0),
  data = df
))

report("poisson_iid", inla_rs(
  count ~ -1 + f(idx, model = "iid"),
  data = df,
  family = "poisson",
  initial_theta = 1.0
))

report("gaussian_family_pc", inla_rs(
  y ~ -1 + f(idx, model = "iid"),
  data = df,
  control.family = list(hyper = list(
    prec = list(prior = "pc.prec", param = c(2.0, 0.1))
  ))
))

report("ar1_pc", inla_rs(
  y ~ -1 + f(idx, model = "ar1", obs_precision = 25.0, hyper = list(
    prec = list(prior = "pc.prec", param = c(2.0, 0.1)),
    rho = list(prior = "pc.cor1", param = c(0.5, 0.75))
  )),
  data = df
))

report("iid2d", inla_rs(
  y ~ -1 + f(idx, model = "iid2d", n = 24L, obs_precision = 25.0, initial = c(0, 0, 0)),
  data = df
))

report("grouped_iid_ar1", inla_rs(
  y ~ -1 + f(space, model = "iid", group = time,
             control.group = list(model = "ar1"), obs_precision = 25.0),
  data = df
))

report("crw2_pairs", inla_rs(
  y ~ -1 + f(idx, model = "crw2", layout = "pairs", positions = "idx", obs_precision = 25.0),
  data = df
))

verts_spde <- matrix(c(0, 0, 1, 0, 1, 1, 0, 1, 0.5, 0.5), ncol = 2L, byrow = TRUE)
tris_spde <- matrix(as.integer(c(1, 2, 5, 2, 3, 5, 3, 4, 5, 4, 1, 5)), ncol = 3L, byrow = TRUE)
df$loc_x <- 0.15 + 0.7 * (df$idx - 1) / 23
df$loc_y <- 0.15 + 0.7 * ((df$idx - 1) %% 5) / 4
df$field <- df$idx
report("spde_formula", inla_rs(
  y ~ -1 + f(field, model = "spde", vertices = verts_spde, triangles = tris_spde,
             loc_x = "loc_x", loc_y = "loc_y", obs_precision = 25.0),
  data = df
))
