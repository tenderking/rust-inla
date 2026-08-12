#' Print a concise summary of a `"inla_rs"` fit (classic INLA-style tables).
#'
#' @param object A `"inla_rs"` result from [inla_rs].
#' @param digits Significant digits for printing.
#' @param ... Unused.
#' @export
summary.inla_rs <- function(object, digits = 3L, ...) {
  structure(
    list(
      call = object$call,
      fixed = object$summary.fixed,
      random = if (length(object$summary.random)) {
        lapply(object$summary.random, function(tab) {
          # compact: first few rows for large effects
          if (nrow(tab) > 8L) {
            rbind(utils::head(tab, 4L), utils::tail(tab, 4L))
          } else {
            tab
          }
        })
      } else {
        NULL
      },
      hyperpar = object$summary.hyperpar,
      mlik = c(
        integration = object$marginal_log_lik,
        gaussian = object$marginal_log_lik_gaussian
      ),
      dic = c(
        dic = object$dic,
        mean.deviance = object$mean_deviance,
        p.eff = object$effective_params
      ),
      waic = c(
        waic = object$waic,
        lppd = object$waic_lppd,
        p.eff = object$waic_effective_params
      ),
      digits = digits
    ),
    class = "summary.inla_rs"
  )
}

#' @export
print.summary.inla_rs <- function(x, ...) {
  dig <- if (!is.null(x$digits)) x$digits else 3L
  cat("Call:\n inla_rs(...)\n\n")
  if (!is.null(x$fixed)) {
    cat("Fixed effects:\n")
    print(round(x$fixed, dig))
    cat("\n")
  }
  if (!is.null(x$random) && length(x$random)) {
    cat("Random effects:\n")
    for (nm in names(x$random)) {
      cat("Name: ", nm, "  size: see full object$summary.random$", nm, "\n", sep = "")
      print(round(x$random[[nm]][, c("mean", "sd", "0.025quant", "0.975quant")], dig))
      cat("\n")
    }
  }
  if (!is.null(x$hyperpar)) {
    cat("Model hyperparameters:\n")
    print(round(x$hyperpar, dig))
    cat("\n")
  }
  if (!is.null(x$mlik)) {
    cat("Marginal log-likelihood:\n")
    print(round(x$mlik, dig))
    cat("\n")
  }
  if (!is.null(x$dic) && all(is.finite(x$dic))) {
    cat("DIC:\n")
    print(round(x$dic, dig))
    cat("\n")
  }
  if (!is.null(x$waic) && all(is.finite(x$waic))) {
    cat("WAIC:\n")
    print(round(x$waic, dig))
  }
  invisible(x)
}

#' @export
print.inla_rs <- function(x, ...) {
  print(summary(x, ...))
  invisible(x)
}
