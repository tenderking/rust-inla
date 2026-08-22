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

report("iid2d", inla_rs(
  y ~ -1 + f(idx, model = "iid2d", n = 24L, obs_precision = 25.0, initial = c(0, 0, 0)),
  data = df
))
