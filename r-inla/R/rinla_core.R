# Minimal direct R -> Rust bridge for the PoC dynamic library.

.rinla_core_dynload <- function(path) {
  if (!file.exists(path)) {
    stop("Library not found: ", path, call. = FALSE)
  }
  dyn.load(path)
}

rinla_core_read_mesh <- function(path) {
  .Call("wrap__rinla_read_mesh", as.character(path))
}

rinla_core_ar1_precision <- function(n, rho, tau = 1) {
  .Call(
    "wrap__rinla_ar1_precision",
    as.integer(n),
    as.numeric(rho),
    as.numeric(tau)
  )
}

rinla_core_ar1_precision_csc <- function(n, rho, tau = 1) {
  if (!requireNamespace("Matrix", quietly = TRUE)) {
    stop("Package 'Matrix' is required", call. = FALSE)
  }
  .Call(
    "wrap__rinla_ar1_precision_csc_dgcmatrix",
    as.integer(n),
    as.numeric(rho),
    as.numeric(tau)
  )
}

rinla_core_rw1_precision_csc <- function(n, tau = 1) {
  .Call("wrap__rinla_rw1_precision_csc", as.integer(n), as.numeric(tau))
}

rinla_core_rw2_precision_csc <- function(n, tau = 1) {
  .Call("wrap__rinla_rw2_precision_csc", as.integer(n), as.numeric(tau))
}

rinla_core_rw1_cyclic_precision_csc <- function(n, tau = 1) {
  .Call("wrap__rinla_rw1_cyclic_precision_csc", as.integer(n), as.numeric(tau))
}

rinla_core_rw2_cyclic_precision_csc <- function(n, tau = 1) {
  .Call("wrap__rinla_rw2_cyclic_precision_csc", as.integer(n), as.numeric(tau))
}

rinla_core_seasonal_precision_csc <- function(n, s, tau = 1, cyclic = TRUE) {
  .Call("wrap__rinla_seasonal_precision_csc", as.integer(n), as.integer(s), as.numeric(tau), as.logical(cyclic))
}

rinla_core_two_diid_precision_csc <- function(n_pairs, rho, tau = 1) {
  .Call("wrap__rinla_two_diid_precision_csc", as.integer(n_pairs), as.numeric(rho), as.numeric(tau))
}

rinla_core_iid_precision_csc <- function(n, tau = 1) {
  .Call("wrap__rinla_iid_precision_csc", as.integer(n), as.numeric(tau))
}

rinla_core_arp_precision_csc <- function(n, pacf, tau = 1) {
  .Call("wrap__rinla_arp_precision_csc", as.integer(n), as.numeric(pacf), as.numeric(tau))
}

rinla_core_matern2d_precision_csc <- function(nrow, ncol, nu, range, prec = 1, cyclic = FALSE) {
  .Call("wrap__rinla_matern2d_precision_csc", as.integer(nrow), as.integer(ncol), as.integer(nu), as.numeric(range), as.numeric(prec), as.logical(cyclic))
}

rinla_core_besag_precision_csc <- function(adj_list, tau = 1) {
  .Call("wrap__rinla_besag_precision_csc", adj_list, as.numeric(tau))
}

rinla_core_bym_precision_csc <- function(adj_list, tau_spatial = 1, tau_iid = 1) {
  .Call("wrap__rinla_bym_precision_csc", adj_list, as.numeric(tau_spatial), as.numeric(tau_iid))
}

rinla_core_spde_precision_mesh_csc <- function(vertices_mat, triangles_mat, kappa, tau = 1) {
  .Call("wrap__rinla_spde_precision_mesh_csc", as.matrix(vertices_mat), as.matrix(triangles_mat), as.numeric(kappa), as.numeric(tau))
}

rinla_core_crw1_precision_csc <- function(positions, tau = 1) {
  .Call("wrap__rinla_crw1_precision_csc", as.numeric(positions), as.numeric(tau))
}

rinla_core_crw2_precision_csc <- function(positions, tau = 1, layout = "simple") {
  .Call("wrap__rinla_crw2_precision_csc", as.numeric(positions), as.numeric(tau), as.character(layout))
}

rinla_core_fgn_precision_csc <- function(n, hurst, tau = 1) {
  if (!requireNamespace("Matrix", quietly = TRUE)) {
    stop("Package 'Matrix' is required", call. = FALSE)
  }
  .Call(
    "wrap__rinla_fgn_precision_csc",
    as.integer(n),
    as.numeric(hurst),
    as.numeric(tau)
  )
}

rinla_core_fgn_approx_precision_csc <- function(n, hurst, tau = 1, order = 4L, prec_eps = 1e8) {
  if (!requireNamespace("Matrix", quietly = TRUE)) {
    stop("Package 'Matrix' is required", call. = FALSE)
  }
  .Call(
    "wrap__rinla_fgn_approx_precision_csc",
    as.integer(n),
    as.numeric(hurst),
    as.numeric(tau),
    as.integer(order),
    as.numeric(prec_eps)
  )
}

rinla_core_run_inla_inference <- function(
    initial_theta,
    model_type,
    y_obs,
    obs_precision = 1.0,
    strategy = "ccd",
    step_or_f0 = 1.0,
    order = 0L,
    family = "gaussian",
    link = "default",
    E = numeric(0),
    Ntrials = numeric(0),
    event = numeric(0),
    size = 1.0,
    zero_prob = 0.1,
    inflation = "type0",
    alpha = 0.5,
    gamma = 1.0,
    shape = 1.0,
    adj_list = list()) {
  .Call(
    "wrap__rinla_run_inla_inference",
    as.numeric(initial_theta),
    as.character(model_type),
    as.numeric(y_obs),
    as.numeric(obs_precision),
    as.character(strategy),
    as.numeric(step_or_f0),
    as.integer(order),
    as.character(family),
    as.character(link),
    as.numeric(E),
    as.numeric(Ntrials),
    as.numeric(event),
    as.numeric(size),
    as.numeric(zero_prob),
    as.character(inflation),
    as.numeric(alpha),
    as.numeric(gamma),
    as.numeric(shape),
    adj_list
  )
}

rinla_core_run_inla_structured <- function(
    initial_theta,
    y_obs,
    obs_precision = 1.0,
    strategy = "ccd",
    step_or_f0 = 1.0,
    family = "gaussian",
    link = "default",
    a_i,
    a_j,
    a_x,
    a_nrow,
    a_ncol,
    effect_types,
    effect_ns,
    effect_scales,
    effect_theta_lens,
    effect_orders,
    adj_lists,
    fixed_prec = 1e-4,
    E = numeric(0),
    Ntrials = numeric(0),
    event = numeric(0),
    size = 1.0,
    zero_prob = 0.1,
    inflation = "type0",
    alpha = 0.5,
    gamma = 1.0,
    shape = 1.0) {
  .Call(
    "wrap__rinla_run_inla_structured",
    as.numeric(initial_theta),
    as.numeric(y_obs),
    as.numeric(obs_precision),
    as.character(strategy),
    as.numeric(step_or_f0),
    as.character(family),
    as.character(link),
    as.integer(a_i),
    as.integer(a_j),
    as.numeric(a_x),
    as.integer(a_nrow),
    as.integer(a_ncol),
    as.character(effect_types),
    as.integer(effect_ns),
    as.integer(effect_scales),
    as.integer(effect_theta_lens),
    as.integer(effect_orders),
    adj_lists,
    as.numeric(fixed_prec),
    as.numeric(E),
    as.numeric(Ntrials),
    as.numeric(event),
    as.numeric(size),
    as.numeric(zero_prob),
    as.character(inflation),
    as.numeric(alpha),
    as.numeric(gamma),
    as.numeric(shape)
  )
}

rinla_core_scale_model_csc <- function(adj_list, tau = 1) {
  .Call("wrap__rinla_scale_model_csc", adj_list, as.numeric(tau))
}

#' Bin a continuous covariate into `n` groups (R-INLA `inla.group` style).
#'
#' Returns integer group indices in `1..n` (or `1..n_unique` when `n` is NULL).
rinla_group <- function(x, n = NULL, method = c("quantile", "cut")) {
  method <- match.arg(method)
  x <- as.numeric(x)
  ok <- is.finite(x)
  if (!any(ok)) {
    stop("rinla_group: no finite values", call. = FALSE)
  }
  if (is.null(n)) {
    ux <- sort(unique(x[ok]))
    out <- match(x, ux)
    return(as.integer(out))
  }
  n <- as.integer(n)[1]
  if (n < 2L) {
    stop("rinla_group: n must be >= 2", call. = FALSE)
  }
  if (method == "quantile") {
    br <- unique(as.numeric(quantile(x[ok], probs = seq(0, 1, length.out = n + 1L), na.rm = TRUE)))
  } else {
    br <- seq(min(x[ok]), max(x[ok]), length.out = n + 1L)
  }
  if (length(br) < 3L) {
    return(rep(1L, length(x)))
  }
  as.integer(cut(x, breaks = br, include.lowest = TRUE, labels = FALSE))
}

#' Scale a Besag/GMRF precision so geom-mean marginal variance ≈ 1.
rinla_scale_model <- function(adj_list, tau = 1) {
  rinla_core_scale_model_csc(adj_list, tau = tau)
}

.rinla_find_all_f_calls <- function(expr) {
  out <- list()
  walk <- function(e) {
    if (is.call(e)) {
      if (identical(e[[1]], as.symbol("f"))) {
        out[[length(out) + 1L]] <<- e
      } else {
        for (i in seq_along(e)[-1]) walk(e[[i]])
      }
    }
  }
  walk(expr)
  out
}

.rinla_strip_f <- function(expr) {
  if (!is.call(expr)) {
    return(expr)
  }
  if (identical(expr[[1]], as.symbol("f"))) {
    return(NULL)
  }
  if (identical(expr[[1]], as.symbol("+"))) {
    a <- .rinla_strip_f(expr[[2]])
    b <- .rinla_strip_f(expr[[3]])
    if (is.null(a)) return(b)
    if (is.null(b)) return(a)
    return(call("+", a, b))
  }
  if (identical(expr[[1]], as.symbol("-")) && length(expr) == 3L) {
    a <- .rinla_strip_f(expr[[2]])
    b <- .rinla_strip_f(expr[[3]])
    if (is.null(b)) return(a)
    if (is.null(a)) return(call("-", b))
    return(call("-", a, b))
  }
  expr
}

.rinla_normalize_adj <- function(graph) {
  if (is.null(graph)) {
    return(NULL)
  }
  if (is.matrix(graph)) {
    return(lapply(seq_len(nrow(graph)), function(i) {
      as.integer(which(graph[i, ] != 0))
    }))
  }
  lapply(graph, as.integer)
}

# Models accepted by `f()` in [rinla_core_inla] (must match Rust structured path).
.rinla_supported_f_models <- c("iid", "rw2", "ar1", "besag", "fgn")

.rinla_effect_theta_len <- function(model, order = 0L) {
  model <- tolower(model)
  if (model %in% c("iid", "rw2", "besag")) return(1L)
  if (model %in% c("ar1", "fgn")) return(2L)
  if (model == "fixed") return(0L)
  stop("Unknown model '", model, "'", call. = FALSE)
}

.rinla_default_theta <- function(model, order = 0L) {
  model <- tolower(model)
  if (model == "fgn" && order > 0L) return(c(1.0, 2.0))
  if (model %in% c("ar1", "fgn")) return(c(0.0, 0.0))
  if (model == "fixed") return(numeric(0))
  c(0.0)
}

#' Fit a latent GMRF model (formula API with A-matrix / multi-f / fixed effects).
#'
#' Supports:
#' - `y ~ x` fixed effects (design matrix `X`, vague prior on β)
#' - one or more `f(...)` terms (block-diagonal prior)
#' - `f(id, model="besag", graph=G, scale.model=TRUE)`
#' - `f(..., initial=c(...))` per-effect starting values for θ
#' - many observations per latent index via sparse projector `A`
#'
#' Supported `f()` models: `iid`, `rw2`, `ar1`, `besag`, `fgn`.
#'
#' Preprocessing helpers: [rinla_group], [rinla_scale_model].
rinla_core_inla <- function(
    formula,
    data,
    family = "gaussian",
    strategy = "ccd",
    step_or_f0 = 1.0,
    initial_theta = NULL,
    control.family = NULL,
    E = NULL,
    Ntrials = NULL,
    event = NULL,
    size = 1.0,
    zero_prob = 0.1,
    inflation = "type0",
    alpha = 0.5,
    gamma = 1.0,
    shape = 1.0,
    link = "default",
    adj_list = NULL,
    fixed_prec = 1e-4,
    ...) {
  supported <- c(
    "gaussian", "poisson", "binomial", "nbinomial", "negative_binomial",
    "zeroinflatedpoisson0", "zeroinflatedpoisson1", "zero_inflated_poisson",
    "zeroinflatedbinomial0", "zeroinflatedbinomial1", "zero_inflated_binomial",
    "laplace", "exponential", "exponential_survival", "weibull", "weibull_survival"
  )
  fam <- tolower(as.character(family)[1])
  if (!(fam %in% supported)) {
    stop("Unsupported family '", family, "'. Supported: ", paste(supported, collapse = ", "),
         call. = FALSE)
  }

  data <- as.data.frame(data)
  resp_var <- all.vars(formula)[1]
  y <- as.numeric(data[[resp_var]])
  n_obs <- length(y)

  f_env <- new.env(parent = parent.frame())
  f_env$f <- function(x, model = "iid", order = 0L, graph = NULL,
                      scale.model = FALSE, values = NULL, initial = NULL, ...) {
    list(
      name = deparse(substitute(x)),
      model = as.character(model)[1],
      order = as.integer(order)[1],
      graph = graph,
      scale.model = isTRUE(scale.model),
      values = values,
      initial = initial,
      args = list(...)
    )
  }
  f_env$inla.group <- rinla_group

  f_calls <- .rinla_find_all_f_calls(formula[[3]])
  f_structs <- lapply(f_calls, function(fc) eval(fc, envir = f_env))

  # Fixed-effects design matrix from stripped formula.
  # When only f() terms remain, build X ourselves so typos in covariates still
  # surface as model.matrix errors instead of being swallowed.
  rhs_fixed <- .rinla_strip_f(formula[[3]])
  if (is.null(rhs_fixed)) {
    has_minus1 <- grepl("(^|\\s)-\\s*1(\\s|$)|(^|\\s)0\\s*\\+",
                        paste(deparse(formula[[3]]), collapse = " "))
    if (length(f_structs) > 0L && !has_minus1) {
      X <- matrix(1, nrow = n_obs, ncol = 1L, dimnames = list(NULL, "(Intercept)"))
    } else {
      X <- matrix(0, nrow = n_obs, ncol = 0)
    }
  } else {
    fixed_fml <- stats::as.formula(call("~", as.symbol(resp_var), rhs_fixed))
    X <- stats::model.matrix(fixed_fml, data = data)
  }

  # Observation precision
  obs_precision <- 1.0
  if (!is.null(control.family)) {
    prec <- tryCatch(control.family$hyper$prec, error = function(e) NULL)
    if (!is.null(prec) && !is.null(prec$initial)) {
      obs_precision <- exp(as.numeric(prec$initial))
    }
  }
  for (fs in f_structs) {
    if (!is.null(fs$args$obs_precision)) {
      obs_precision <- as.numeric(fs$args$obs_precision)
    }
  }

  # Build latent blocks and A triplets (0-based)
  a_i <- integer(0)
  a_j <- integer(0)
  a_x <- numeric(0)
  col_off <- 0L
  effect_types <- character(0)
  effect_ns <- integer(0)
  effect_scales <- integer(0)
  effect_theta_lens <- integer(0)
  effect_orders <- integer(0)
  adj_lists <- list()
  theta <- numeric(0)

  add_triplets <- function(rows0, cols0, vals) {
    a_i <<- c(a_i, as.integer(rows0))
    a_j <<- c(a_j, as.integer(cols0))
    a_x <<- c(a_x, as.numeric(vals))
  }

  for (fs in f_structs) {
    model <- tolower(fs$model)
    if (!(model %in% .rinla_supported_f_models)) {
      stop("Unsupported f() model '", fs$model, "'. Supported: ",
           paste(.rinla_supported_f_models, collapse = ", "), call. = FALSE)
    }
    idx_name <- fs$name
    # Support inla.group(...) captured as name string — evaluate index from data
    if (grepl("^inla\\.group\\(", idx_name) || grepl("^rinla_group\\(", idx_name)) {
      idx <- as.integer(eval(parse(text = idx_name), envir = data))
    } else if (!is.null(fs$values)) {
      idx <- as.integer(fs$values)
    } else {
      if (is.null(data[[idx_name]])) {
        stop("Index variable '", idx_name, "' not found in data", call. = FALSE)
      }
      idx <- as.integer(data[[idx_name]])
    }
    if (any(is.na(idx))) {
      stop("NA in index for f(", idx_name, ")", call. = FALSE)
    }

    order <- fs$order
    if (is.na(order) || is.null(order)) order <- 0L

    if (model == "fgn" && order %in% c(3L, 4L)) {
      # Approx FGN: observations map to z-block only; latent length (order+1)*n_time
      n_time <- length(unique(idx))
      n_e <- as.integer((order + 1L) * n_time)
      # Map obs i -> z_{idx[i]} column (0-based within block = idx-1)
      umin <- min(idx)
      zcol <- as.integer(idx - umin) # 0..n_time-1
      add_triplets(seq_len(n_obs) - 1L, col_off + zcol, rep(1.0, n_obs))
      graph <- NULL
    } else if (model == "besag") {
      graph <- fs$graph
      if (is.null(graph)) graph <- adj_list
      graph <- .rinla_normalize_adj(graph)
      if (is.null(graph)) {
        stop("besag requires graph= or adj_list=", call. = FALSE)
      }
      n_e <- length(graph)
      umin <- 1L
      # region ids are typically 1..n_e
      zcol <- as.integer(idx - umin)
      if (any(zcol < 0L | zcol >= n_e)) {
        stop("besag index out of range for graph size ", n_e, call. = FALSE)
      }
      add_triplets(seq_len(n_obs) - 1L, col_off + zcol, rep(1.0, n_obs))
    } else {
      # Generic: unique sorted levels → latent size
      lev <- sort(unique(idx))
      n_e <- length(lev)
      zcol <- match(idx, lev) - 1L
      add_triplets(seq_len(n_obs) - 1L, col_off + zcol, rep(1.0, n_obs))
      graph <- NULL
    }

    effect_types <- c(effect_types, model)
    effect_ns <- c(effect_ns, n_e)
    effect_scales <- c(effect_scales, if (isTRUE(fs$scale.model)) 1L else 0L)
    tlen <- .rinla_effect_theta_len(model, order)
    effect_theta_lens <- c(effect_theta_lens, tlen)
    effect_orders <- c(effect_orders, as.integer(order))
    if (model == "besag") {
      adj_lists[[length(adj_lists) + 1L]] <- graph
    } else {
      adj_lists[[length(adj_lists) + 1L]] <- list()
    }
    if (is.null(initial_theta)) {
      if (!is.null(fs$initial)) {
        init <- as.numeric(fs$initial)
        if (length(init) != tlen) {
          stop("f(", idx_name, ", model=\"", model, "\"): initial has length ",
               length(init), " but expected ", tlen, call. = FALSE)
        }
        if (any(!is.finite(init))) {
          stop("f(", idx_name, "): initial must be finite", call. = FALSE)
        }
        theta <- c(theta, init)
      } else {
        theta <- c(theta, .rinla_default_theta(model, order))
      }
    }
    col_off <- col_off + n_e
  }

  # Fixed effects block
  p <- ncol(X)
  if (p > 0L) {
    for (j in seq_len(p)) {
      rows <- which(X[, j] != 0)
      if (length(rows)) {
        add_triplets(rows - 1L, rep(col_off + j - 1L, length(rows)), X[rows, j])
      }
    }
    effect_types <- c(effect_types, "fixed")
    effect_ns <- c(effect_ns, as.integer(p))
    effect_scales <- c(effect_scales, 0L)
    effect_theta_lens <- c(effect_theta_lens, 0L)
    effect_orders <- c(effect_orders, 0L)
    adj_lists[[length(adj_lists) + 1L]] <- list()
    col_off <- col_off + p
  }

  if (length(effect_types) == 0L) {
    stop("Formula has no f() terms and no fixed effects", call. = FALSE)
  }

  if (!is.null(initial_theta)) {
    theta <- as.numeric(initial_theta)
  }

  if (is.null(E)) E <- numeric(0)
  if (is.null(Ntrials)) Ntrials <- numeric(0)
  if (is.null(event)) event <- numeric(0)

  rinla_core_run_inla_structured(
    initial_theta = theta,
    y_obs = y,
    obs_precision = obs_precision,
    strategy = strategy,
    step_or_f0 = step_or_f0,
    family = fam,
    link = link,
    a_i = a_i,
    a_j = a_j,
    a_x = a_x,
    a_nrow = n_obs,
    a_ncol = col_off,
    effect_types = effect_types,
    effect_ns = effect_ns,
    effect_scales = effect_scales,
    effect_theta_lens = effect_theta_lens,
    effect_orders = effect_orders,
    adj_lists = adj_lists,
    fixed_prec = fixed_prec,
    E = E,
    Ntrials = Ntrials,
    event = event,
    size = size,
    zero_prob = zero_prob,
    inflation = inflation,
    alpha = alpha,
    gamma = gamma,
    shape = shape
  )
}
