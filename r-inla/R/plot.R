#' Plot selected panels from a `"rinla"` fit.
#'
#' Mirrors a subset of classic `plot.inla`: hyperparameter densities,
#' random-effect mean ± quantile ribbon, and linear predictor.
#'
#' @param x A `"rinla"` result.
#' @param plot.hyperparameters Plot internal hyperparameter marginals.
#' @param plot.random.effects Plot random-effect summaries.
#' @param plot.predictor Plot linear predictor mean ± 95% band.
#' @param single One plot per page when TRUE.
#' @param ... Unused.
#' @export
plot.rinla <- function(
    x,
    plot.hyperparameters = TRUE,
    plot.random.effects = TRUE,
    plot.predictor = TRUE,
    single = FALSE,
    ...) {
  ask <- isTRUE(single) && interactive()
  old_ask <- if (ask) graphics::par(ask = TRUE) else NULL
  on.exit({
    if (!is.null(old_ask)) graphics::par(old_ask)
  })

  if (isTRUE(plot.hyperparameters) && !is.null(x$internal.marginals.hyperpar)) {
    im <- x$internal.marginals.hyperpar
    for (j in seq_along(im)) {
      mat <- im[[j]]
      if (is.null(dim(mat)) || nrow(mat) < 2L) next
      graphics::plot(
        mat[, 1], mat[, 2],
        type = "l", lwd = 2,
        xlab = paste0("theta[", j, "] (internal)"),
        ylab = "density",
        main = paste("Hyperparameter", j)
      )
    }
  }

  if (isTRUE(plot.random.effects) && length(x$summary.random)) {
    for (nm in names(x$summary.random)) {
      tab <- x$summary.random[[nm]]
      id <- tab$ID
      graphics::plot(
        id, tab$mean,
        type = "l", lwd = 2,
        ylim = range(c(tab$`0.025quant`, tab$`0.975quant`), finite = TRUE),
        xlab = "index", ylab = "mean",
        main = paste("Random effect:", nm)
      )
      graphics::polygon(
        c(id, rev(id)),
        c(tab$`0.025quant`, rev(tab$`0.975quant`)),
        border = NA, col = grDevices::adjustcolor("steelblue", alpha.f = 0.25)
      )
      graphics::lines(id, tab$mean, lwd = 2)
    }
  }

  if (isTRUE(plot.predictor) && !is.null(x$summary.linear.predictor)) {
    tab <- x$summary.linear.predictor
    id <- seq_len(nrow(tab))
    graphics::plot(
      id, tab$mean,
      type = "l", lwd = 2,
      ylim = range(c(tab$`0.025quant`, tab$`0.975quant`), finite = TRUE),
      xlab = "observation", ylab = "eta",
      main = "Linear predictor"
    )
    graphics::polygon(
      c(id, rev(id)),
      c(tab$`0.025quant`, rev(tab$`0.975quant`)),
      border = NA, col = grDevices::adjustcolor("darkgreen", alpha.f = 0.25)
    )
    graphics::lines(id, tab$mean, lwd = 2)
  }

  invisible(x)
}
