# Minimal direct R -> Rust bridge for the PoC dynamic library.

.inla_rs_dynload <- function(path) {
  if (!file.exists(path)) {
    stop("Library not found: ", path, call. = FALSE)
  }
  dyn.load(path)
}

inla_rs_read_mesh <- function(path) {
  .Call("wrap__inla_rs_read_mesh", as.character(path))
}

inla_rs_ar1_precision <- function(n, rho, tau = 1) {
  .Call(
    "wrap__inla_rs_ar1_precision",
    as.integer(n),
    as.numeric(rho),
    as.numeric(tau)
  )
}

inla_rs_ar1_precision_csc <- function(n, rho, tau = 1) {
  if (!requireNamespace("Matrix", quietly = TRUE)) {
    stop("Package 'Matrix' is required", call. = FALSE)
  }
  .Call(
    "wrap__inla_rs_ar1_precision_csc_dgcmatrix",
    as.integer(n),
    as.numeric(rho),
    as.numeric(tau)
  )
}

inla_rs_rw1_precision_csc <- function(n, tau = 1) {
  .Call("wrap__inla_rs_rw1_precision_csc", as.integer(n), as.numeric(tau))
}

inla_rs_rw2_precision_csc <- function(n, tau = 1) {
  .Call("wrap__inla_rs_rw2_precision_csc", as.integer(n), as.numeric(tau))
}

inla_rs_rw1_cyclic_precision_csc <- function(n, tau = 1) {
  .Call("wrap__inla_rs_rw1_cyclic_precision_csc", as.integer(n), as.numeric(tau))
}

inla_rs_rw2_cyclic_precision_csc <- function(n, tau = 1) {
  .Call("wrap__inla_rs_rw2_cyclic_precision_csc", as.integer(n), as.numeric(tau))
}

inla_rs_seasonal_precision_csc <- function(n, s, tau = 1, cyclic = TRUE) {
  .Call("wrap__inla_rs_seasonal_precision_csc", as.integer(n), as.integer(s), as.numeric(tau), as.logical(cyclic))
}

inla_rs_two_diid_precision_csc <- function(n_pairs, rho, tau = 1) {
  .Call("wrap__inla_rs_two_diid_precision_csc", as.integer(n_pairs), as.numeric(rho), as.numeric(tau))
}

inla_rs_iid_precision_csc <- function(n, tau = 1) {
  .Call("wrap__inla_rs_iid_precision_csc", as.integer(n), as.numeric(tau))
}

inla_rs_arp_precision_csc <- function(n, pacf, tau = 1) {
  .Call("wrap__inla_rs_arp_precision_csc", as.integer(n), as.numeric(pacf), as.numeric(tau))
}

inla_rs_matern2d_precision_csc <- function(nrow, ncol, nu, range, prec = 1, cyclic = FALSE) {
  .Call("wrap__inla_rs_matern2d_precision_csc", as.integer(nrow), as.integer(ncol), as.integer(nu), as.numeric(range), as.numeric(prec), as.logical(cyclic))
}

inla_rs_rw2d_precision_csc <- function(nrow, ncol, tau = 1, cyclic = FALSE, bvalue_zero = FALSE) {
  .Call(
    "wrap__inla_rs_rw2d_precision_csc",
    as.integer(nrow),
    as.integer(ncol),
    as.numeric(tau),
    as.logical(cyclic),
    as.logical(bvalue_zero)
  )
}

inla_rs_besag_precision_csc <- function(adj_list, tau = 1) {
  .Call("wrap__inla_rs_besag_precision_csc", adj_list, as.numeric(tau))
}

inla_rs_bym_precision_csc <- function(adj_list, tau_spatial = 1, tau_iid = 1) {
  .Call("wrap__inla_rs_bym_precision_csc", adj_list, as.numeric(tau_spatial), as.numeric(tau_iid))
}

inla_rs_bym2_precision_csc <- function(adj_list, tau = 1, phi = 0.5) {
  .Call("wrap__inla_rs_bym2_precision_csc", adj_list, as.numeric(tau), as.numeric(phi))
}

inla_rs_spde_precision_mesh_csc <- function(vertices_mat, triangles_mat, kappa, tau = 1) {
  .Call("wrap__inla_rs_spde_precision_mesh_csc", as.matrix(vertices_mat), as.matrix(triangles_mat), as.numeric(kappa), as.numeric(tau))
}

#' FEM mass (c0 / C) and stiffness (g1 / G) for a triangular mesh.
#'
#' Corresponds to classic INLA `spde$param.inla$M0` / `M1`.
inla_rs_fem_blocks_mesh <- function(vertices_mat, triangles_mat) {
  .Call(
    "wrap__inla_rs_fem_blocks_mesh",
    as.matrix(vertices_mat),
    as.matrix(triangles_mat)
  )
}

#' Build a regular triangular lattice mesh over a rectangle.
#'
#' Stand-in for classic `inla.mesh.2d` when only rust-inla is available:
#' vertices on an `nx` × `ny` grid, each cell split into two triangles.
inla_rs_lattice_mesh <- function(xlim = c(0, 1), ylim = c(0, 1), nx = 11L, ny = 11L) {
  nx <- as.integer(nx)[1]
  ny <- as.integer(ny)[1]
  if (nx < 2L || ny < 2L) {
    stop("nx and ny must be >= 2", call. = FALSE)
  }
  xs <- seq(xlim[1], xlim[2], length.out = nx)
  ys <- seq(ylim[1], ylim[2], length.out = ny)
  # Column-major: x varies fastest (same as expand.grid(x, y))
  vertices <- as.matrix(expand.grid(x = xs, y = ys))
  colnames(vertices) <- c("x", "y")
  idx <- function(i, j) (j - 1L) * nx + i
  tris <- matrix(NA_integer_, nrow = 2L * (nx - 1L) * (ny - 1L), ncol = 3L)
  k <- 0L
  for (j in seq_len(ny - 1L)) {
    for (i in seq_len(nx - 1L)) {
      v00 <- idx(i, j)
      v10 <- idx(i + 1L, j)
      v01 <- idx(i, j + 1L)
      v11 <- idx(i + 1L, j + 1L)
      k <- k + 1L
      tris[k, ] <- c(v00, v10, v01)
      k <- k + 1L
      tris[k, ] <- c(v10, v11, v01)
    }
  }
  list(vertices = vertices, triangles = tris, nx = nx, ny = ny)
}

#' Piecewise-linear SPDE projector A (n_obs x n_vertices).
inla_rs_spde_projector_csc <- function(vertices_mat, triangles_mat, loc_x, loc_y) {
  .Call(
    "wrap__inla_rs_spde_projector_csc",
    as.matrix(vertices_mat),
    as.matrix(triangles_mat),
    as.numeric(loc_x),
    as.numeric(loc_y)
  )
}

#' Fit a Gaussian SPDE model on a triangular mesh.
#'
#' Hyperparameters are internal \eqn{\theta = (\log\tau, \log\kappa)}.
#' Observation field is \eqn{\eta = A x} with barycentric FEM weights.
#'
#' @param y Numeric response.
#' @param loc Two-column matrix / data.frame of observation coordinates, or
#'   parallel `loc_x` / `loc_y` vectors.
#' @param vertices N x 2 vertex coordinates.
#' @param triangles M x 3 triangle indices (1-based).
#' @param initial_theta Length-2 starting values `[log_tau, log_kappa]`.
#' @param obs_precision Gaussian observation precision.
#' @param constrain If TRUE, apply a sum-to-zero constraint on the field.
inla_rs_spde <- function(
    y,
    loc = NULL,
    loc_x = NULL,
    loc_y = NULL,
    vertices,
    triangles,
    initial_theta = c(0.0, 0.0),
    obs_precision = 25.0,
    strategy = "ccd",
    step_or_f0 = 1.0,
    constrain = FALSE,
    deterministic = FALSE) {
  if (!is.null(loc)) {
    loc <- as.matrix(loc)
    if (ncol(loc) != 2L) stop("loc must have 2 columns", call. = FALSE)
    loc_x <- loc[, 1]
    loc_y <- loc[, 2]
  }
  if (is.null(loc_x) || is.null(loc_y)) {
    stop("Provide loc=cbind(x,y) or loc_x= and loc_y=", call. = FALSE)
  }
  .Call(
    "wrap__inla_rs_run_spde",
    as.numeric(initial_theta),
    as.numeric(y),
    as.numeric(obs_precision)[1],
    as.character(strategy)[1],
    as.numeric(step_or_f0)[1],
    as.matrix(vertices),
    as.matrix(triangles),
    as.numeric(loc_x),
    as.numeric(loc_y),
    as.logical(constrain)[1],
    as.logical(deterministic)[1]
  )
}

inla_rs_crw1_precision_csc <- function(positions, tau = 1) {
  .Call("wrap__inla_rs_crw1_precision_csc", as.numeric(positions), as.numeric(tau))
}

inla_rs_crw2_precision_csc <- function(positions, tau = 1, layout = "simple") {
  .Call("wrap__inla_rs_crw2_precision_csc", as.numeric(positions), as.numeric(tau), as.character(layout))
}

inla_rs_fgn_precision_csc <- function(n, hurst, tau = 1) {
  if (!requireNamespace("Matrix", quietly = TRUE)) {
    stop("Package 'Matrix' is required", call. = FALSE)
  }
  .Call(
    "wrap__inla_rs_fgn_precision_csc",
    as.integer(n),
    as.numeric(hurst),
    as.numeric(tau)
  )
}

inla_rs_fgn_approx_precision_csc <- function(n, hurst, tau = 1, order = 4L, prec_eps = 1e8) {
  if (!requireNamespace("Matrix", quietly = TRUE)) {
    stop("Package 'Matrix' is required", call. = FALSE)
  }
  .Call(
    "wrap__inla_rs_fgn_approx_precision_csc",
    as.integer(n),
    as.numeric(hurst),
    as.numeric(tau),
    as.integer(order),
    as.numeric(prec_eps)
  )
}

inla_rs_run_inla_inference <- function(
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
    adj_list = list(),
    deterministic = FALSE) {
  .Call(
    "wrap__inla_rs_run_inla_inference",
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
    adj_list,
    as.logical(deterministic)
  )
}

inla_rs_run_inla_structured <- function(
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
    effect_copy_of = integer(0),
    effect_n_main,
    effect_group_models,
    effect_group_ns,
    effect_group_scales,
    adj_lists,
    effect_positions = list(),
    prior_names = character(0),
    prior_params = list(),
    fixed_prec = 1e-4,
    E = numeric(0),
    Ntrials = numeric(0),
    event = numeric(0),
    size = 1.0,
    zero_prob = 0.1,
    inflation = "type0",
    alpha = 0.5,
    gamma = 1.0,
    shape = 1.0,
    deterministic = FALSE,
    gaussian_free_prec = FALSE,
    family_prior_name = "loggamma",
    family_prior_param = c(1.0, 5e-5),
    effect_nrow = integer(0),
    effect_ncol = integer(0),
    effect_cyclic = integer(0),
    effect_season = integer(0),
    effect_layouts = character(0),
    effect_meshes = list(),
    dic = TRUE,
    waic = TRUE,
    cpo = TRUE) {
  .Call(
    "wrap__inla_rs_run_inla_structured",
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
    as.integer(effect_copy_of),
    as.integer(effect_n_main),
    as.character(effect_group_models),
    as.integer(effect_group_ns),
    as.integer(effect_group_scales),
    adj_lists,
    effect_positions,
    as.character(prior_names),
    prior_params,
    as.numeric(fixed_prec),
    as.numeric(E),
    as.numeric(Ntrials),
    as.numeric(event),
    as.numeric(size),
    as.numeric(zero_prob),
    as.character(inflation),
    as.numeric(alpha),
    as.numeric(gamma),
    as.numeric(shape),
    as.logical(deterministic),
    as.logical(gaussian_free_prec),
    as.character(family_prior_name),
    as.numeric(family_prior_param),
    as.integer(effect_nrow),
    as.integer(effect_ncol),
    as.integer(effect_cyclic),
    as.integer(effect_season),
    as.character(effect_layouts),
    effect_meshes,
    as.logical(dic),
    as.logical(waic),
    as.logical(cpo)
  )
}

#' Gaussian + single AR(1) via Rust ModelSpec / ModelPlan (identity η = x).
inla_rs_run_gaussian_ar1_plan <- function(
    y_obs,
    name = "time",
    obs_precision = 100.0,
    strategy = "ccd",
    step_or_f0 = 1.0,
    initial_theta = NULL) {
  if (is.null(initial_theta)) {
    initial_theta <- numeric(0)
  }
  .Call(
    "wrap__inla_rs_run_gaussian_ar1_plan",
    as.numeric(y_obs),
    as.character(name)[1],
    as.numeric(obs_precision)[1],
    as.character(strategy)[1],
    as.numeric(step_or_f0)[1],
    as.numeric(initial_theta)
  )
}

inla_rs_scale_model_csc <- function(adj_list, tau = 1) {
  .Call("wrap__inla_rs_scale_model_csc", adj_list, as.numeric(tau))
}

inla_rs_prior_log_density <- function(name, param = numeric(0), theta) {
  .Call(
    "wrap__inla_rs_prior_log_density",
    as.character(name),
    as.numeric(param),
    as.numeric(theta)
  )
}

inla_rs_default_hyper_priors <- function(model, order = 0L) {
  .Call("wrap__inla_rs_default_hyper_priors", as.character(model), as.integer(order))
}

inla_rs_hyper_prior_stack_log_density <- function(names, params, theta) {
  .Call(
    "wrap__inla_rs_hyper_prior_stack_log_density",
    as.character(names),
    params,
    as.numeric(theta)
  )
}

#' Classic-INLA style survival response helper.
#'
#' Returns a two-column data frame; use with `family = "exponential.surv"` (etc.)
#' and pass `event = dat$event` (or keep an `event` column in `data`).
inla_rs_surv <- function(time, event) {
  data.frame(time = as.numeric(time), event = as.numeric(event))
}

# Alias matching classic INLA spelling when this package is sourced alone.
if (!exists("inla.surv", mode = "function", inherits = TRUE)) {
  inla.surv <- inla_rs_surv
}

#' Bin a continuous covariate into `n` groups (classic R-INLA `inla.group`).
#'
#' Returns the **median** of each occupied bin (not integer codes `1..K`).
#' Those medians are the RW2/`$ID` locations, matching
#' `f(inla.group(x, n), model = "rw2")`.
inla_rs_group <- function(x, n = 25, method = c("cut", "quantile"), idx.only = FALSE) {
  method <- match.arg(method)
  x <- as.numeric(x)
  if (!any(is.finite(x))) {
    stop("inla_rs_group: no finite values", call. = FALSE)
  }
  n <- as.integer(n)[1]
  if (is.na(n) || n < 1L) {
    stop("inla_rs_group: n must be >= 1", call. = FALSE)
  }

  core <- function(xx) {
    if (n == 1L) {
      return(rep(stats::median(xx), length(xx)))
    }
    if (method == "cut") {
      a <- cut(xx, n)
    } else {
      aq <- unique(as.numeric(stats::quantile(
        xx, probs = c(0, stats::ppoints(n - 1L), 1)
      )))
      a <- cut(xx, breaks = aq, include.lowest = TRUE)
    }
    idx <- as.integer(a)
    med <- vapply(seq_len(nlevels(a)), function(i) {
      xi <- xx[idx == i]
      if (length(xi) > 0L) stats::median(xi) else NA_real_
    }, numeric(1))
    if (isTRUE(idx.only)) {
      return(as.numeric(idx))
    }
    as.numeric(med[idx])
  }

  out <- rep(NA_real_, length(x))
  ok <- is.finite(x)
  out[ok] <- core(x[ok])
  out
}

#' Scale a Besag/GMRF precision so geom-mean marginal variance ≈ 1.
inla_rs_scale_model <- function(adj_list, tau = 1) {
  inla_rs_scale_model_csc(adj_list, tau = tau)
}

.inla_rs_find_all_f_calls <- function(expr) {
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

.inla_rs_strip_f <- function(expr) {
  if (!is.call(expr)) {
    return(expr)
  }
  if (identical(expr[[1]], as.symbol("f"))) {
    return(NULL)
  }
  if (identical(expr[[1]], as.symbol("+"))) {
    a <- .inla_rs_strip_f(expr[[2]])
    b <- .inla_rs_strip_f(expr[[3]])
    if (is.null(a)) return(b)
    if (is.null(b)) return(a)
    return(call("+", a, b))
  }
  if (identical(expr[[1]], as.symbol("-")) && length(expr) == 3L) {
    a <- .inla_rs_strip_f(expr[[2]])
    b <- .inla_rs_strip_f(expr[[3]])
    if (is.null(b)) return(a)
    if (is.null(a)) return(call("-", b))
    return(call("-", a, b))
  }
  expr
}

.inla_rs_normalize_adj <- function(graph) {
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

# Models accepted by `f()` in [inla_rs]; the Rust registry is the source of truth.
.inla_rs_supported_f_models <- function() {
  inla_rs_supported_models()
}

#' Per-model metadata from the shared Rust registry.
#' @export
inla_rs_model_metadata <- function(model, order = 0L, group_model = "", cyclic = FALSE) {
  .Call(
    "wrap__inla_rs_model_metadata",
    as.character(model)[1],
    as.integer(order)[1],
    as.character(group_model)[1],
    as.logical(cyclic)[1]
  )
}

#' Latent model names accepted by `f()`.
#' @export
inla_rs_supported_models <- function() {
  .Call("wrap__inla_rs_supported_models")
}

#' Latent models with registry metadata (not necessarily formula-executable).
#' @export
inla_rs_registered_models <- function() {
  .Call("wrap__inla_rs_registered_models")
}

#' Models accepted as `control.group` factors.
#' @export
inla_rs_supported_group_models <- function() {
  .Call("wrap__inla_rs_supported_group_models")
}

#' Validate a named list of engine controls and fill Rust-side defaults.
#' @export
inla_rs_resolve_compute_options <- function(controls = list()) {
  .Call("wrap__inla_rs_resolve_compute_options", controls)
}

#' Validate + default engine controls in Rust, so R and Python agree on names.
.inla_rs_resolve_controls <- function(control.compute = NULL, strategy = "ccd",
                                      step_or_f0 = 1.0, fixed_prec = 1e-4,
                                      deterministic = FALSE) {
  bag <- list(
    strategy = as.character(strategy)[1],
    step_or_f0 = as.numeric(step_or_f0)[1],
    fixed_prec = as.numeric(fixed_prec)[1],
    deterministic = as.logical(deterministic)[1]
  )
  if (!is.null(control.compute)) {
    if (is.null(names(control.compute)) || any(!nzchar(names(control.compute)))) {
      stop("control.compute must be a named list", call. = FALSE)
    }
    for (nm in names(control.compute)) {
      v <- control.compute[[nm]]
      if (is.null(v)) next
      bag[[nm]] <- v
    }
  }
  inla_rs_resolve_compute_options(bag)
}

# Cached wrapper: metadata is looked up per theta node during optimization.
.inla_rs_meta_cache <- new.env(parent = emptyenv())

.inla_rs_model_meta <- function(model, order = 0L, group_model = NULL, cyclic = FALSE) {
  model <- tolower(model)
  gm <- if (is.null(group_model)) "" else tolower(group_model)
  key <- paste(model, as.integer(order), gm, isTRUE(cyclic), sep = "|")
  hit <- .inla_rs_meta_cache[[key]]
  if (!is.null(hit)) return(hit)
  meta <- inla_rs_model_metadata(model, as.integer(order), gm, isTRUE(cyclic))
  assign(key, meta, envir = .inla_rs_meta_cache)
  meta
}

.inla_rs_effect_theta_len <- function(model, order = 0L, group_model = NULL) {
  as.integer(.inla_rs_model_meta(model, order, group_model)$theta_len)
}

.inla_rs_default_theta <- function(model, order = 0L, group_model = NULL) {
  as.numeric(.inla_rs_model_meta(model, order, group_model)$default_theta)
}

.inla_rs_match_hyper_slot <- function(key, internals) {
  norm <- tolower(gsub("[-.]", "_", as.character(key)[1]))
  exact <- which(tolower(internals) == norm)
  if (length(exact) == 1L) return(exact)
  aliases <- switch(
    norm,
    prec = grep("precision", tolower(internals), fixed = TRUE),
    precision = grep("precision", tolower(internals), fixed = TRUE),
    rho = grep("rho", tolower(internals), fixed = TRUE),
    phi = grep("phi", tolower(internals), fixed = TRUE),
    range = grep("range", tolower(internals), fixed = TRUE),
    integer(0)
  )
  if (length(aliases) == 1L) aliases else integer(0)
}

.inla_rs_effect_priors <- function(fs, model, order = 0L, group_model = NULL) {
  meta <- .inla_rs_model_meta(model, order, group_model)
  prior_names <- as.character(meta$prior_names)
  params <- lapply(meta$prior_params, as.numeric)
  args <- fs$args

  if (!is.null(args$prior)) {
    prior_names[1] <- as.character(args$prior)[1]
    params[[1]] <- if (is.null(args$param)) numeric(0) else as.numeric(args$param)
    return(list(names = prior_names, params = params))
  }

  hyper <- args$hyper
  if (is.null(hyper)) return(list(names = prior_names, params = params))
  if (!is.list(hyper) || is.null(base::names(hyper))) {
    stop("f(..., hyper=) must be a named list", call. = FALSE)
  }
  internals <- as.character(meta$hyper_internal)
  if (length(prior_names) != length(internals)) {
    stop("f(..., hyper=) cannot override slots for joint-prior model '", model,
         "'; pass prior= and param= for the joint prior", call. = FALSE)
  }
  for (key in base::names(hyper)) {
    value <- hyper[[key]]
    if (!is.list(value)) {
      stop("hyper slot '", key, "' must be a list", call. = FALSE)
    }
    if (is.null(value$prior) && is.null(value$param)) next
    slot <- .inla_rs_match_hyper_slot(key, internals)
    if (length(slot) != 1L) {
      stop("unknown or ambiguous hyper slot '", key, "' for model '", model, "'", call. = FALSE)
    }
    if (!is.null(value$prior)) prior_names[slot] <- as.character(value$prior)[1]
    if (!is.null(value$param)) params[[slot]] <- as.numeric(value$param)
  }
  list(names = prior_names, params = params)
}

#' Define an R-callback generic latent model (Python ``inla.define`` analogue).
#'
#' @param n Latent dimension.
#' @param Q Function `function(theta)` returning a `dgCMatrix` precision.
#' @param n_theta Number of hyperparameters.
#' @param initial Starting θ.
#' @param log.prior Optional `function(theta)` log-prior density.
inla_rs_rgeneric_define <- function(n, Q, n_theta = 1L, initial = NULL, log.prior = NULL,
                                    name = "rgeneric") {
  n <- as.integer(n)[1]
  n_theta <- as.integer(n_theta)[1]
  if (is.null(initial)) initial <- rep(0.0, n_theta)
  initial <- as.numeric(initial)
  if (length(initial) != n_theta) {
    stop("initial length must equal n_theta", call. = FALSE)
  }
  if (!is.function(Q)) stop("Q must be a function(theta)", call. = FALSE)
  list(
    n = n,
    Q = Q,
    n_theta = n_theta,
    initial = initial,
    log.prior = log.prior,
    name = as.character(name)[1]
  )
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
#' Preprocessing helpers: [inla_rs_group], [inla_rs_scale_model].
inla_rs <- function(
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
    deterministic = FALSE,
    control.compute = NULL,
    ...) {
  # Rust owns control names, defaults and validation (shared with Python).
  .controls <- .inla_rs_resolve_controls(
    control.compute,
    strategy = strategy,
    step_or_f0 = step_or_f0,
    fixed_prec = fixed_prec,
    deterministic = deterministic
  )
  strategy <- .controls$strategy
  step_or_f0 <- .controls$step_or_f0
  fixed_prec <- .controls$fixed_prec
  deterministic <- .controls$deterministic
  dic <- isTRUE(.controls$dic)
  waic <- isTRUE(.controls$waic)
  cpo <- isTRUE(.controls$cpo)

  supported <- c(
    "gaussian", "poisson", "binomial", "nbinomial", "negative_binomial", "negbin",
    "cbinomial",
    "zeroinflatedpoisson0", "zeroinflatedpoisson1", "zero_inflated_poisson", "zip",
    "zeroinflatedbinomial0", "zeroinflatedbinomial1", "zero_inflated_binomial", "zib",
    "laplace", "exponential", "exponential_survival", "exponential.surv", "exponential_surv",
    "weibull", "weibull_survival", "weibull.surv", "weibull_surv"
  )
  fam <- tolower(as.character(family)[1])
  if (!(fam %in% supported)) {
    stop("Unsupported family '", family, "'. Supported: ", paste(supported, collapse = ", "),
         call. = FALSE)
  }
  # Canonical aliases (also accepted in Rust canonicalize_family)
  fam <- switch(
    fam,
    "exponential.surv" = "exponential_survival",
    "exponential_surv" = "exponential_survival",
    "weibull.surv" = "weibull_survival",
    "weibull_surv" = "weibull_survival",
    "negbin" = "negative_binomial",
    "nbinomial" = "negative_binomial",
    "zip" = "zero_inflated_poisson",
    "zeroinflatedpoisson0" = "zero_inflated_poisson",
    "zib" = "zero_inflated_binomial",
    "zeroinflatedbinomial0" = "zero_inflated_binomial",
    "cbinomial" = "binomial",
    fam
  )

  data <- as.data.frame(data)
  if (is.null(event) && fam %in% c("exponential", "exponential_survival", "weibull", "weibull_survival")) {
    if (!is.null(data[["event"]])) {
      event <- as.numeric(data[["event"]])
    }
  }
  resp_var <- all.vars(formula)[1]
  y <- as.numeric(data[[resp_var]])
  n_obs <- length(y)

  f_env <- new.env(parent = parent.frame())
  f_env$f <- function(x, w = NULL, model = "iid", order = 0L, graph = NULL,
                      scale.model = NULL, values = NULL, initial = NULL,
                      group = NULL, control.group = NULL, copy = NULL, n = NULL,
                      ...) {
    model_chr <- as.character(model)[1]
    # The shared Rust registry owns the scale.model default (intrinsic models).
    if (is.null(scale.model)) {
      scale.model <- isTRUE(.inla_rs_model_meta(model_chr)$default_scale_model)
    }
    extra <- list(...)
    if (!is.null(copy)) extra$copy <- copy
    if (!is.null(n)) extra$n <- n
    if (!missing(w)) {
      w_sub <- substitute(w)
      extra$weights <- if (is.symbol(w_sub)) as.character(w_sub) else w
    }
    list(
      name = deparse(substitute(x)),
      model = model_chr,
      order = as.integer(order)[1],
      graph = graph,
      scale.model = isTRUE(scale.model),
      values = values,
      initial = initial,
      group = if (is.null(group)) NULL else deparse(substitute(group)),
      control.group = control.group,
      args = extra
    )
  }
  f_env$inla.group <- inla_rs_group

  f_calls <- .inla_rs_find_all_f_calls(formula[[3]])
  f_structs <- lapply(f_calls, function(fc) eval(fc, envir = f_env))

  # Fixed-effects design matrix from stripped formula.
  # When only f() terms remain, build X ourselves so typos in covariates still
  # surface as model.matrix errors instead of being swallowed.
  rhs_fixed <- .inla_rs_strip_f(formula[[3]])
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

  family_free_prec <- FALSE
  initial_log_prec <- 0.0
  prec_fixed <- FALSE
  family_prior_name <- "loggamma"
  family_prior_param <- c(1.0, 5e-5)
  if (!is.null(control.family)) {
    prec <- tryCatch(control.family$hyper$prec, error = function(e) NULL)
    if (!is.null(prec)) {
      if (isTRUE(prec$fixed)) prec_fixed <- TRUE
      if (!is.null(prec$initial)) initial_log_prec <- as.numeric(prec$initial)[1]
      if (!is.null(prec$prior)) family_prior_name <- as.character(prec$prior)[1]
      if (!is.null(prec$param)) family_prior_param <- as.numeric(prec$param)
    }
  }
  has_f_obs_prec <- FALSE
  for (fs in f_structs) {
    if (!is.null(fs$args$obs_precision)) {
      obs_precision <- as.numeric(fs$args$obs_precision)[1]
      initial_log_prec <- log(obs_precision)
      has_f_obs_prec <- TRUE
    }
  }
  if (fam %in% c("gaussian", "normal")) {
    if (!prec_fixed && !has_f_obs_prec) {
      family_free_prec <- TRUE
    }
  }
  obs_precision <- exp(initial_log_prec)

  # ModelPlan fast path: gaussian + single AR1 + no fixed effects + identity index
  if (identical(fam, "gaussian") &&
      !family_free_prec &&
      length(f_structs) == 1L &&
      ncol(X) == 0L &&
      identical(tolower(f_structs[[1]]$model), "ar1") &&
      is.null(f_structs[[1]]$group) &&
      is.null(f_structs[[1]]$args$hyper) &&
      is.null(f_structs[[1]]$args$prior) &&
      is.null(f_structs[[1]]$args$param)) {
    fs <- f_structs[[1]]
    idx_name <- fs$name
    if (!is.null(data[[idx_name]])) {
      idx_val <- data[[idx_name]]
      n_raw <- length(idx_val)
      # Check for 1..n contiguous
      if (is.numeric(idx_val) && min(idx_val) == 1 && max(idx_val) == n_raw && length(unique(idx_val)) == n_raw) {
        theta_init <- if (!is.null(initial_theta)) {
          as.numeric(initial_theta)
        } else if (!is.null(fs$initial)) {
          as.numeric(fs$initial)
        } else {
          c(0.0, 0.0) # default log_prec=0, logit_rho=0
        }
        raw <- inla_rs_run_gaussian_ar1_plan(
          y_obs = as.numeric(y),
          name = idx_name,
          obs_precision = obs_precision,
          strategy = strategy,
          step_or_f0 = step_or_f0,
          initial_theta = theta_init
        )
        return(.inla_rs_attach_summaries(
          raw,
          effect_types = "ar1",
          effect_ns = as.integer(n_raw),
          effect_orders = 0L,
          effect_names = idx_name,
          fixed_names = character(0)
        ))
      }
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
  effect_copy_of <- integer(0)
  effect_n_main <- integer(0)
  effect_group_models <- character(0)
  effect_group_ns <- integer(0)
  effect_group_scales <- integer(0)
  effect_names_acc <- character(0)
  adj_lists <- list()
  effect_ids <- list()
  effect_positions <- list()
  effect_nrow <- integer(0)
  effect_ncol <- integer(0)
  effect_cyclic <- integer(0)
  effect_season <- integer(0)
  effect_layouts <- character(0)
  effect_meshes <- list()
  prior_names <- character(0)
  prior_params <- list()
  theta <- numeric(0)

  add_triplets <- function(rows0, cols0, vals) {
    a_i <<- c(a_i, as.integer(rows0))
    a_j <<- c(a_j, as.integer(cols0))
    a_x <<- c(a_x, as.numeric(vals))
  }

  iidkd_dim <- function(model) {
    switch(model, iid2d = 2L, iid3d = 3L, iid4d = 4L, iid5d = 5L, NA_integer_)
  }

  weight_vec <- function(spec) {
    if (is.null(spec)) return(rep(1.0, n_obs))
    if (is.character(spec) && length(spec) == 1L) {
      if (spec %in% c("1", "Intercept")) return(rep(1.0, n_obs))
      if (is.null(data[[spec]])) {
        stop("weights column '", spec, "' not found in data", call. = FALSE)
      }
      return(as.numeric(data[[spec]]))
    }
    if (is.numeric(spec) && length(spec) == 1L) return(rep(as.numeric(spec), n_obs))
    if (is.numeric(spec) && length(spec) == n_obs) return(as.numeric(spec))
    stop("weights must be 1, a column name, or a length-n numeric vector", call. = FALSE)
  }

  resolve_matrix <- function(spec, what) {
    if (is.null(spec)) return(NULL)
    if (is.character(spec) && length(spec) == 1L) {
      if (is.null(data[[spec]])) {
        stop(what, " '", spec, "' not found in data", call. = FALSE)
      }
      spec <- data[[spec]]
    }
    if (is.list(spec) && !is.null(spec[[what]])) spec <- spec[[what]]
    as.matrix(spec)
  }

  p <- ncol(X)
  if (p > 0L) {
    for (j in seq_len(p)) {
      rows <- which(X[, j] != 0)
      if (length(rows)) {
        add_triplets(rows - 1L, rep(col_off + j - 1L, length(rows)), X[rows, j])
      }
    }
    effect_types <- "fixed"
    effect_ns <- as.integer(p)
    effect_scales <- 0L
    effect_theta_lens <- 0L
    effect_orders <- 0L
    effect_copy_of <- -1L
    effect_n_main <- 0L
    effect_group_models <- ""
    effect_group_ns <- 0L
    effect_group_scales <- 0L
    effect_nrow <- 0L
    effect_ncol <- 0L
    effect_cyclic <- 0L
    effect_season <- 0L
    effect_layouts <- "simple"
    adj_lists[[1L]] <- list()
    effect_meshes[[1L]] <- list()
    effect_ids[[1L]] <- colnames(X)
    effect_positions[[1L]] <- numeric(0)
    effect_names_acc <- "fixed"
    col_off <- as.integer(p)
  }

  for (fs in f_structs) {
    model <- if (!is.null(fs$args$copy)) "copy" else tolower(fs$model)
    supported <- .inla_rs_supported_f_models()
    if (!(model %in% supported)) {
      stop("Unsupported f() model '", fs$model, "'. Supported: ",
           paste(supported, collapse = ", "), call. = FALSE)
    }
    idx_name <- fs$name
    # Support inla.group(...) captured as name string — evaluate index from data
    if (grepl("^inla\\.group\\(", idx_name) || grepl("^inla_rs_group\\(", idx_name)) {
      idx <- eval(parse(text = idx_name), envir = data)
    } else if (!is.null(fs$values)) {
      idx <- fs$values
    } else {
      if (is.null(data[[idx_name]])) {
        stop("Index variable '", idx_name, "' not found in data", call. = FALSE)
      }
      idx <- data[[idx_name]]
    }
    if (any(is.na(idx))) {
      stop("NA in index for f(", idx_name, ")", call. = FALSE)
    }

    order <- fs$order
    if (is.na(order) || is.null(order)) order <- 0L
    nrow_i <- 0L
    ncol_i <- 0L
    cyclic_i <- 0L
    season_i <- 0L
    layout <- "simple"
    mesh_store <- NULL
    if (identical(model, "crw2") && !is.null(fs$args$layout)) {
      layout <- tolower(as.character(fs$args$layout)[1])
      if (!(layout %in% c("simple", "pairs", "block"))) {
        stop("crw2 layout must be 'simple', 'pairs', or 'block'", call. = FALSE)
      }
    }
    group_model <- NULL
    group_scale <- FALSE
    if (!is.null(fs$group)) {
      cg <- fs$control.group
      group_model <- if (is.list(cg) && !is.null(cg$model)) {
        tolower(as.character(cg$model)[1])
      } else if (is.character(cg) && length(cg) > 0L) {
        tolower(cg[1])
      } else {
        "ar1"
      }
      group_scale <- is.list(cg) && isTRUE(cg$scale.model)
    }

    if (!is.null(group_model)) {
      groupable <- c("iid", "rw1", "rw2", "ar1", "ar", "arp", "seasonal", "crw1", "crw2")
      if (!(model %in% groupable)) {
        stop("grouped effect model '", model, "' is not supported", call. = FALSE)
      }
      if (!(group_model %in% c("iid", "rw1", "rw2", "ar1"))) {
        stop("unsupported control.group model '", group_model, "'", call. = FALSE)
      }
      if (identical(model, "crw2") && layout != "simple") {
        stop("grouped crw2 currently supports layout='simple' only", call. = FALSE)
      }
      group_name <- as.character(fs$group)[1]
      if (is.null(data[[group_name]])) {
        stop("Group variable '", group_name, "' not found in data", call. = FALSE)
      }
      group_idx <- data[[group_name]]
      if (length(group_idx) != n_obs || any(is.na(group_idx))) {
        stop("Group variable '", group_name, "' must be complete and length n", call. = FALSE)
      }
      main_levels <- sort(unique(idx))
      group_levels <- sort(unique(group_idx))
      n_main <- length(main_levels)
      group_n <- length(group_levels)
      n_e <- as.integer(n_main * group_n)
      main_col <- match(idx, main_levels) - 1L
      group_col <- match(group_idx, group_levels) - 1L
      zcol <- group_col * n_main + main_col
      add_triplets(seq_len(n_obs) - 1L, col_off + zcol, weight_vec(fs$args$weights))
      graph <- NULL
      order_enc <- as.integer(order)
      if (identical(model, "seasonal")) {
        s <- fs$args$season
        if (is.null(s)) s <- fs$args$s
        if (is.null(s)) s <- if (order > 0L) order else 4L
        season_i <- as.integer(s)[1]
      }
      effect_ids[[length(effect_ids) + 1L]] <- main_levels
    } else if (identical(model, "copy")) {
      src_name <- as.character(fs$args$copy)[1]
      src_i <- match(src_name, effect_names_acc)
      if (is.na(src_i)) {
        stop("f(", idx_name, ", copy='", src_name,
             "'): source not found (must appear first)", call. = FALSE)
      }
      n_e <- effect_ns[src_i]
      zcol <- as.integer(idx)
      if (min(zcol) >= 1L) zcol <- zcol - 1L
      if (any(zcol < 0L | zcol >= n_e)) {
        lev <- sort(unique(idx))
        if (length(lev) != n_e) {
          stop("copy index for '", idx_name, "' incompatible with source n=",
               n_e, call. = FALSE)
        }
        zcol <- match(idx, lev) - 1L
      }
      add_triplets(seq_len(n_obs) - 1L, col_off + zcol, weight_vec(fs$args$weights))
      graph <- NULL
      order_enc <- 0L
      effect_copy_of <- c(effect_copy_of, as.integer(src_i - 1L))
      effect_ids[[length(effect_ids) + 1L]] <- effect_ids[[src_i]]
    } else if (model == "fgn" && order %in% c(3L, 4L)) {
      # Approx FGN: observations map to z-block only; latent length (order+1)*n_time
      lev <- sort(unique(idx))
      n_time <- length(lev)
      n_e <- as.integer((order + 1L) * n_time)
      zcol <- match(idx, lev) - 1L
      add_triplets(seq_len(n_obs) - 1L, col_off + zcol, rep(1.0, n_obs))
      graph <- NULL
      order_enc <- order
      effect_ids[[length(effect_ids) + 1L]] <- lev
    } else if (model == "rw2d" || model == "matern2d") {
      nrow <- as.integer(fs$args$nrow)[1]
      ncol <- as.integer(fs$args$ncol)[1]
      if (is.na(nrow) || is.na(ncol) || nrow < 1L || ncol < 1L) {
        stop(model, " requires nrow= and ncol=", call. = FALSE)
      }
      if (model == "rw2d" && (nrow < 3L || ncol < 3L)) {
        stop("rw2d requires nrow= and ncol= (>=3)", call. = FALSE)
      }
      n_e <- as.integer(nrow * ncol)
      nrow_i <- as.integer(nrow)
      ncol_i <- as.integer(ncol)
      cyclic_i <- if (isTRUE(fs$args$cyclic)) 1L else 0L
      zcol <- as.integer(idx)
      if (min(zcol) >= 1L) zcol <- zcol - 1L
      if (any(zcol < 0L | zcol >= n_e)) {
        stop(model, " index out of range for nrow*ncol=", n_e, call. = FALSE)
      }
      add_triplets(seq_len(n_obs) - 1L, col_off + zcol, rep(1.0, n_obs))
      graph <- NULL
      order_enc <- as.integer(order)
      effect_ids[[length(effect_ids) + 1L]] <- seq_len(n_e)
    } else if (model %in% c("besag", "bym", "bym2")) {
      graph <- fs$graph
      if (is.null(graph)) graph <- adj_list
      graph <- .inla_rs_normalize_adj(graph)
      if (is.null(graph)) {
        stop(model, " requires graph= or adj_list=", call. = FALSE)
      }
      n_graph <- length(graph)
      umin <- 1L
      zcol <- as.integer(idx - umin)
      if (any(zcol < 0L | zcol >= n_graph)) {
        stop(model, " index out of range for graph size ", n_graph, call. = FALSE)
      }
      if (identical(model, "bym")) {
        # Observe u + v
        add_triplets(seq_len(n_obs) - 1L, col_off + zcol, rep(1.0, n_obs))
        add_triplets(seq_len(n_obs) - 1L, col_off + n_graph + zcol, rep(1.0, n_obs))
        n_e <- as.integer(2L * n_graph)
      } else {
        add_triplets(seq_len(n_obs) - 1L, col_off + zcol, rep(1.0, n_obs))
        n_e <- as.integer(n_graph)
      }
      order_enc <- as.integer(order)
      effect_ids[[length(effect_ids) + 1L]] <- seq_len(n_graph)
    } else if (!is.na(iidkd_dim(model))) {
      d <- iidkd_dim(model)
      wspec <- fs$args$weights
      n_req <- if (!is.null(fs$args$n)) as.integer(fs$args$n)[1] else NA_integer_
      lev <- sort(unique(idx))
      n_units <- length(lev)
      zcol <- match(idx, lev) - 1L
      if (is.list(wspec) && length(wspec) == d) {
        n_e <- as.integer(d * n_units)
        for (k in seq_len(d)) {
          wv <- weight_vec(wspec[[k]])
          add_triplets(seq_len(n_obs) - 1L, col_off + (k - 1L) * n_units + zcol, wv)
        }
        effect_ids[[length(effect_ids) + 1L]] <- lev
      } else {
        if (is.na(n_req)) {
          stop(model, " requires n= (latent length ", d, "*n_units) or weights=list(...) of length ",
               d, call. = FALSE)
        }
        if (n_req <= 0L || (n_req %% d) != 0L) {
          stop(model, ": n=", n_req, " must be positive and divisible by ", d, call. = FALSE)
        }
        n_e <- n_req
        n_u <- as.integer(n_e / d)
        z_try <- as.integer(idx)
        if (min(z_try) >= 1L) z_try <- z_try - 1L
        if (min(z_try) >= 0L && max(z_try) < n_e) {
          z_map <- z_try
        } else if (n_units == n_u) {
          z_map <- zcol
        } else {
          stop(model, " index does not map into n=", n_e, call. = FALSE)
        }
        add_triplets(seq_len(n_obs) - 1L, col_off + z_map, weight_vec(wspec))
        effect_ids[[length(effect_ids) + 1L]] <- seq_len(n_e)
      }
      graph <- NULL
      order_enc <- as.integer(order)
    } else if (identical(model, "spde")) {
      if (!is.null(group_model)) {
        stop("grouped SPDE effects are not supported", call. = FALSE)
      }
      spde_mod <- fs$args$spde_model
      if (is.null(spde_mod)) spde_mod <- fs$args$mesh
      verts <- fs$args$vertices
      tris <- fs$args$triangles
      if (!is.null(spde_mod)) {
        if (is.null(verts) && !is.null(spde_mod$vertices)) verts <- spde_mod$vertices
        if (is.null(tris) && !is.null(spde_mod$triangles)) tris <- spde_mod$triangles
      }
      verts <- resolve_matrix(verts, "vertices")
      tris <- resolve_matrix(tris, "triangles")
      if (is.null(verts) || is.null(tris)) {
        stop("f(..., model=\"spde\") requires vertices= and triangles= (or mesh=/spde_model=)",
             call. = FALSE)
      }
      if (ncol(verts) != 2L) stop("spde vertices must be N x 2", call. = FALSE)
      if (ncol(tris) != 3L) stop("spde triangles must be M x 3", call. = FALSE)
      loc <- fs$args$loc
      if (!is.null(loc)) {
        if (is.character(loc) && length(loc) == 1L) {
          if (is.null(data[[loc]])) stop("loc column '", loc, "' not found in data", call. = FALSE)
          loc <- data[[loc]]
        }
        loc <- as.matrix(loc)
        if (ncol(loc) != 2L) stop("spde loc must have 2 columns", call. = FALSE)
        loc_x <- as.numeric(loc[, 1])
        loc_y <- as.numeric(loc[, 2])
      } else {
        lx <- fs$args$loc_x
        ly <- fs$args$loc_y
        if (is.null(lx)) lx <- "loc_x"
        if (is.null(ly)) ly <- "loc_y"
        if (is.character(lx) && length(lx) == 1L && !is.null(data[[lx]])) {
          loc_x <- as.numeric(data[[lx]])
        } else if (is.numeric(lx)) {
          loc_x <- as.numeric(lx)
        } else {
          stop("f(..., model=\"spde\") needs loc= or loc_x=/loc_y= columns", call. = FALSE)
        }
        if (is.character(ly) && length(ly) == 1L && !is.null(data[[ly]])) {
          loc_y <- as.numeric(data[[ly]])
        } else if (is.numeric(ly)) {
          loc_y <- as.numeric(ly)
        } else {
          stop("f(..., model=\"spde\") needs loc= or loc_x=/loc_y= columns", call. = FALSE)
        }
      }
      if (length(loc_x) != n_obs || length(loc_y) != n_obs) {
        stop("spde loc_x/loc_y must have length n", call. = FALSE)
      }
      a_spde <- inla_rs_spde_projector_csc(verts, tris, loc_x, loc_y)
      sm <- Matrix::summary(a_spde)
      add_triplets(as.integer(sm$i) - 1L, col_off + as.integer(sm$j) - 1L, sm$x)
      n_e <- as.integer(nrow(verts))
      graph <- NULL
      order_enc <- as.integer(order)
      effect_ids[[length(effect_ids) + 1L]] <- seq_len(n_e)
      mesh_store <- list(vertices = verts, triangles = tris)
    } else if (identical(model, "crw2")) {
      lev <- sort(unique(idx))
      n_main <- length(lev)
      zcol <- match(idx, lev) - 1L
      wv <- weight_vec(fs$args$weights)
      if (layout %in% c("pairs", "block")) {
        n_e <- as.integer(2L * n_main)
        cols <- if (identical(layout, "pairs")) 2L * zcol else zcol
        add_triplets(seq_len(n_obs) - 1L, col_off + cols, wv)
      } else {
        n_e <- as.integer(n_main)
        add_triplets(seq_len(n_obs) - 1L, col_off + zcol, wv)
      }
      graph <- NULL
      order_enc <- as.integer(order)
      effect_ids[[length(effect_ids) + 1L]] <- lev
    } else {
      # Generic: unique sorted levels → latent size
      lev <- sort(unique(idx))
      n_e <- length(lev)
      zcol <- match(idx, lev) - 1L
      add_triplets(seq_len(n_obs) - 1L, col_off + zcol, weight_vec(fs$args$weights))
      graph <- NULL
      order_enc <- as.integer(order)
      if (identical(model, "seasonal")) {
        s <- fs$args$season
        if (is.null(s)) s <- fs$args$s
        if (is.null(s)) s <- if (order > 0L) order else 4L
        season_i <- as.integer(s)[1]
      }
      effect_ids[[length(effect_ids) + 1L]] <- lev
    }

    if (!identical(model, "copy")) {
      effect_copy_of <- c(effect_copy_of, -1L)
    }
    effect_names_acc <- c(effect_names_acc, idx_name)
    effect_types <- c(effect_types, model)
    effect_ns <- c(effect_ns, n_e)
    effect_scales <- c(effect_scales, if (isTRUE(fs$scale.model)) 1L else 0L)
    effect_n_main <- c(effect_n_main, if (is.null(group_model)) 0L else as.integer(n_main))
    effect_group_models <- c(effect_group_models, if (is.null(group_model)) "" else group_model)
    effect_group_ns <- c(effect_group_ns, if (is.null(group_model)) 0L else as.integer(group_n))
    effect_group_scales <- c(effect_group_scales, if (isTRUE(group_scale)) 1L else 0L)
    tlen <- .inla_rs_effect_theta_len(model, order, group_model)
    effect_theta_lens <- c(effect_theta_lens, tlen)
    effect_orders <- c(effect_orders, as.integer(order_enc))
    effect_nrow <- c(effect_nrow, as.integer(nrow_i))
    effect_ncol <- c(effect_ncol, as.integer(ncol_i))
    effect_cyclic <- c(effect_cyclic, as.integer(cyclic_i))
    effect_season <- c(effect_season, as.integer(season_i))
    effect_layouts <- c(effect_layouts, layout)
    prior_spec <- .inla_rs_effect_priors(fs, model, order, group_model)
    prior_names <- c(prior_names, prior_spec$names)
    prior_params <- c(prior_params, prior_spec$params)
    if (model %in% c("besag", "bym", "bym2")) {
      adj_lists[[length(adj_lists) + 1L]] <- graph
    } else {
      adj_lists[[length(adj_lists) + 1L]] <- list()
    }
    if (is.null(mesh_store)) {
      effect_meshes[[length(effect_meshes) + 1L]] <- list()
    } else {
      effect_meshes[[length(effect_meshes) + 1L]] <- mesh_store
    }
    ids <- effect_ids[[length(effect_ids)]]
    n_knots <- if (identical(model, "crw2") && layout %in% c("pairs", "block")) {
      as.integer(n_e / 2L)
    } else {
      n_e
    }
    pos <- numeric(0)
    raw_pos <- fs$args$positions
    if (!is.null(raw_pos)) {
      if (is.character(raw_pos) && length(raw_pos) == 1L) {
        if (is.null(data[[raw_pos]])) {
          stop("positions column '", raw_pos, "' not found in data", call. = FALSE)
        }
        pos <- as.numeric(data[[raw_pos]])
      } else {
        pos <- as.numeric(raw_pos)
      }
    } else {
      cand <- suppressWarnings(as.numeric(ids))
      if (length(cand) == n_knots && all(is.finite(cand))) {
        pos <- cand
      }
    }
    if (length(pos) > 0L && all(is.finite(pos))) {
      effect_positions[[length(effect_positions) + 1L]] <- pos
    } else {
      effect_positions[[length(effect_positions) + 1L]] <- numeric(0)
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
        theta <- c(theta, .inla_rs_default_theta(model, order, group_model))
      }
    }
    col_off <- col_off + n_e
  }

  if (length(effect_types) == 0L) {
    stop("Formula has no f() terms and no fixed effects", call. = FALSE)
  }

  if (!is.null(initial_theta)) {
    theta <- as.numeric(initial_theta)
  } else if (family_free_prec) {
    theta <- c(initial_log_prec, theta)
  }

  if (is.null(E)) E <- numeric(0)
  if (is.null(Ntrials)) Ntrials <- numeric(0)
  if (is.null(event)) event <- numeric(0)

  raw <- inla_rs_run_inla_structured(
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
    effect_copy_of = effect_copy_of,
    effect_n_main = effect_n_main,
    effect_group_models = effect_group_models,
    effect_group_ns = effect_group_ns,
    effect_group_scales = effect_group_scales,
    adj_lists = adj_lists,
    effect_positions = effect_positions,
    prior_names = prior_names,
    prior_params = prior_params,
    fixed_prec = fixed_prec,
    E = E,
    Ntrials = Ntrials,
    event = event,
    size = size,
    zero_prob = zero_prob,
    inflation = inflation,
    alpha = alpha,
    gamma = gamma,
    shape = shape,
    deterministic = deterministic,
    gaussian_free_prec = family_free_prec,
    family_prior_name = family_prior_name,
    family_prior_param = family_prior_param,
    effect_nrow = effect_nrow,
    effect_ncol = effect_ncol,
    effect_cyclic = effect_cyclic,
    effect_season = effect_season,
    effect_layouts = effect_layouts,
    effect_meshes = effect_meshes,
    dic = dic,
    waic = waic,
    cpo = cpo
  )

  .inla_rs_attach_summaries(
    raw,
    effect_types = effect_types,
    effect_ns = effect_ns,
    effect_orders = effect_orders,
    effect_names = {
      nm <- character(0)
      if (p > 0L) {
        nm <- c(nm, "fixed")
      }
      for (i in seq_along(f_structs)) {
        nm <- c(nm, f_structs[[i]]$name)
      }
      nm
    },
    fixed_names = if (p > 0L) colnames(X) else character(0),
    effect_ids = effect_ids,
    effect_group_models = effect_group_models,
    family_free_prec = family_free_prec
  )
}

#' Labels / transform kinds for each internal θ component, in optim order.
#'
#' Delegates to the shared Rust registry so R and Python label θ identically.
.inla_rs_hyper_labels <- function(effect_types, effect_names, effect_orders = NULL,
                                  effect_group_models = NULL, family_free_prec = FALSE) {
  kinds <- character(0)
  labels <- character(0)
  if (family_free_prec) {
    kinds <- c(kinds, "exp")
    labels <- c(labels, "Precision for the Gaussian observations")
  }
  for (i in seq_along(effect_types)) {
    typ <- tolower(effect_types[i])
    if (typ == "fixed") next
    nm <- if (i <= length(effect_names) && nzchar(effect_names[i])) {
      effect_names[i]
    } else {
      typ
    }
    ord <- if (!is.null(effect_orders) && length(effect_orders) >= i) {
      as.integer(effect_orders[i])
    } else {
      0L
    }
    gm <- if (!is.null(effect_group_models) && length(effect_group_models) >= i) {
      effect_group_models[[i]]
    } else {
      NULL
    }
    meta <- .inla_rs_model_meta(typ, ord, gm)
    kinds <- c(kinds, as.character(meta$hyper_transforms))
    labels <- c(labels, paste(as.character(meta$hyper_labels), "for", nm))
  }
  list(kind = kinds, label = labels)
}

#' Transform a scalar hyper summary from internal θ to natural scale.
.inla_rs_transform_hyper <- function(kind, stats) {
  # stats: named mean, sd, q025, q50, q975, mode (internal)
  tr_vals <- function(x) {
    switch(
      kind,
      "exp" = exp(x),
      "rho" = 2 / (1 + exp(-x)) - 1,
      "phi" = 1 / (1 + exp(-x)),
      "hurst" = 0.5 + 0.5 / (1 + exp(-x)),
      x
    )
  }
  out <- stats
  out["mode"] <- tr_vals(stats[["mode"]])
  out["mean"] <- tr_vals(stats[["mean"]])
  out["q025"] <- tr_vals(stats[["q025"]])
  out["q50"] <- tr_vals(stats[["q50"]])
  out["q975"] <- tr_vals(stats[["q975"]])
  # Delta-method sd on natural scale when transform is differentiable
  if (kind == "exp" && is.finite(stats[["sd"]])) {
    out["sd"] <- exp(stats[["mean"]]) * stats[["sd"]]
  } else if (kind == "rho" && is.finite(stats[["sd"]])) {
    r <- out["mean"]
    out["sd"] <- 0.5 * (1 - r * r) * stats[["sd"]]
  } else if (kind == "phi" && is.finite(stats[["sd"]])) {
    p <- out["mean"]
    out["sd"] <- p * (1 - p) * stats[["sd"]]
  } else if (kind == "hurst" && is.finite(stats[["sd"]])) {
    h <- out["mean"]
    out["sd"] <- 2 * (h - 0.5) * (1 - h) * stats[["sd"]]
  } else {
    out["sd"] <- stats[["sd"]]
  }
  # Keep quantile order after monotonic transforms
  if (kind %in% c("exp", "rho", "phi", "hurst")) {
    qs <- sort(c(out["q025"], out["q975"]))
    out["q025"] <- qs[1]
    out["q975"] <- qs[2]
  }
  out
}

#' Build Gaussian interim summary tables + class `"inla_rs"`.
.inla_rs_attach_summaries <- function(raw, effect_types, effect_ns, effect_orders = NULL,
                                    effect_names, fixed_names, effect_ids = NULL,
                                    effect_group_models = NULL,
                                    family_free_prec = FALSE) {
  means <- as.numeric(raw$latent_means)
  vars <- as.numeric(raw$latent_variances)
  off <- 0L
  summary.random <- list()
  summary.fixed <- NULL

  for (i in seq_along(effect_types)) {
    n_e <- as.integer(effect_ns[i])
    idx <- (off + 1L):(off + n_e)
    mu <- means[idx]
    sd <- sqrt(pmax(vars[idx], 0))
    id_vals <- if (!is.null(effect_ids) && length(effect_ids) >= i && length(effect_ids[[i]]) == n_e) {
      effect_ids[[i]]
    } else {
      seq_len(n_e)
    }
    tab <- data.frame(
      ID = id_vals,
      mean = mu,
      sd = sd,
      `0.025quant` = mu - 1.96 * sd,
      `0.5quant` = mu,
      `0.975quant` = mu + 1.96 * sd,
      mode = mu,
      check.names = FALSE
    )
    typ <- effect_types[i]
    nm <- if (i <= length(effect_names) && nzchar(effect_names[i])) {
      effect_names[i]
    } else {
      paste0(typ, i)
    }
    if (identical(typ, "fixed")) {
      if (length(fixed_names) == n_e) {
        rownames(tab) <- fixed_names
      }
      summary.fixed <- tab[, c("mean", "sd", "0.025quant", "0.5quant", "0.975quant", "mode")]
    } else {
      summary.random[[nm]] <- tab
    }
    off <- off + n_e
  }

  pmu <- as.numeric(raw$predictor_means)
  psd <- sqrt(pmax(as.numeric(raw$predictor_variances), 0))
  summary.linear.predictor <- data.frame(
    mean = pmu,
    sd = psd,
    `0.025quant` = pmu - 1.96 * psd,
    `0.5quant` = pmu,
    `0.975quant` = pmu + 1.96 * psd,
    mode = pmu,
    check.names = FALSE
  )

  # Hyperpar: mode from optim; sd/quantiles from internal 1D marginals when present
  mode <- as.numeric(raw$mode)
  m <- length(mode)
  hyp_mean <- mode
  hyp_sd <- rep(NA_real_, m)
  hyp_q025 <- rep(NA_real_, m)
  hyp_q50 <- mode
  hyp_q975 <- rep(NA_real_, m)
  im <- raw$internal_marginals_hyperpar
  if (!is.null(im) && length(im) == m) {
    for (j in seq_len(m)) {
      mat <- im[[j]]
      if (is.null(dim(mat)) || ncol(mat) < 2L) next
      # Gaussian interim from density moments
      x <- mat[, 1]
      y <- mat[, 2]
      dx <- diff(x)
      mass <- sum(0.5 * (y[-length(y)] + y[-1]) * dx)
      if (mass > 0) {
        y <- y / mass
        ex <- sum(0.5 * (x[-length(x)] * y[-length(y)] + x[-1] * y[-1]) * dx)
        ex2 <- sum(0.5 * (x[-length(x)]^2 * y[-length(y)] + x[-1]^2 * y[-1]) * dx)
        hyp_mean[j] <- ex
        hyp_sd[j] <- sqrt(max(ex2 - ex * ex, 0))
        # Approximate quantiles via cumulative trapz
        cdf <- c(0, cumsum(0.5 * (y[-length(y)] + y[-1]) * dx))
        cdf <- cdf / max(cdf[length(cdf)], .Machine$double.eps)
        hyp_q025[j] <- approx(cdf, x, xout = 0.025, rule = 2)$y
        hyp_q50[j] <- approx(cdf, x, xout = 0.5, rule = 2)$y
        hyp_q975[j] <- approx(cdf, x, xout = 0.975, rule = 2)$y
      }
    }
  }
  # Internal-scale table (θ as optimized)
  summary.hyperpar.internal <- if (m > 0L) {
    data.frame(
      mean = hyp_mean,
      sd = hyp_sd,
      `0.025quant` = hyp_q025,
      `0.5quant` = hyp_q50,
      `0.975quant` = hyp_q975,
      mode = mode,
      check.names = FALSE,
      row.names = paste0("theta", seq_len(m))
    )
  } else {
    NULL
  }

  # Natural-scale table (Precision / Range / Rho / …) matching classic INLA summaries
  hyp_meta <- .inla_rs_hyper_labels(
    effect_types, effect_names, effect_orders, effect_group_models, family_free_prec
  )
  summary.hyperpar <- if (m > 0L && length(hyp_meta$kind) == m) {
    nat_mean <- hyp_mean
    nat_sd <- hyp_sd
    nat_q025 <- hyp_q025
    nat_q50 <- hyp_q50
    nat_q975 <- hyp_q975
    nat_mode <- mode
    for (j in seq_len(m)) {
      tr <- .inla_rs_transform_hyper(hyp_meta$kind[j], c(
        mean = hyp_mean[j], sd = hyp_sd[j],
        q025 = hyp_q025[j], q50 = hyp_q50[j], q975 = hyp_q975[j],
        mode = mode[j]
      ))
      nat_mean[j] <- tr["mean"]
      nat_sd[j] <- tr["sd"]
      nat_q025[j] <- tr["q025"]
      nat_q50[j] <- tr["q50"]
      nat_q975[j] <- tr["q975"]
      nat_mode[j] <- tr["mode"]
    }
    data.frame(
      mean = nat_mean,
      sd = nat_sd,
      `0.025quant` = nat_q025,
      `0.5quant` = nat_q50,
      `0.975quant` = nat_q975,
      mode = nat_mode,
      check.names = FALSE,
      row.names = hyp_meta$label
    )
  } else {
    summary.hyperpar.internal
  }

  out <- raw
  out$summary.random <- summary.random
  out$summary.fixed <- summary.fixed
  out$summary.linear.predictor <- summary.linear.predictor
  out$summary.hyperpar <- summary.hyperpar
  out$summary.hyperpar.internal <- summary.hyperpar.internal
  out$internal.marginals.hyperpar <- im
  out$waic <- if (!is.null(raw$waic)) as.numeric(raw$waic) else NA_real_
  out$waic_lppd <- if (!is.null(raw$waic_lppd)) as.numeric(raw$waic_lppd) else NA_real_
  out$waic_effective_params <- if (!is.null(raw$waic_effective_params)) {
    as.numeric(raw$waic_effective_params)
  } else {
    NA_real_
  }
  out$effects <- list(
    types = effect_types,
    ns = as.integer(effect_ns),
    names = effect_names
  )
  # Convenience: map FGN internal θ → Hurst when a single FGN block is present.
  fgn_i <- which(effect_types == "fgn")
  if (length(fgn_i) == 1L) {
    off <- if (isTRUE(family_free_prec)) 1L else 0L
    hi <- as.numeric(mode[off + 2L])
    out$hurst <- if (length(hi) == 1L && is.finite(hi)) {
      0.5 + 0.5 / (1.0 + exp(-hi))
    } else {
      NA_real_
    }
  } else {
    out$hurst <- NA_real_
  }
  class(out) <- c("inla_rs", "list")
  out
}

#' Build a linear combination (1-based latent indices).
#' @export
inla_rs_make_lincomb <- function(name, idx, weights) {
  if (length(idx) != length(weights)) {
    stop("idx and weights must have the same length", call. = FALSE)
  }
  list(
    name = as.character(name)[1],
    idx = as.integer(idx) - 1L,
    weights = as.numeric(weights)
  )
}

inla_rs_lincomb <- function(q_i, q_p, q_x, q_n, means, comb_names, comb_idx, comb_weights) {
  .Call(
    "wrap__inla_rs_lincomb",
    as.integer(q_i),
    as.integer(q_p),
    as.numeric(q_x),
    as.integer(q_n)[1],
    as.numeric(means),
    as.character(comb_names),
    comb_idx,
    comb_weights
  )
}

#' Linear combinations from a fitted `"inla_rs"` object.
#' @export
inla_rs_lincomb_fit <- function(fit, lincombs) {
  if (is.null(fit$posterior_q_n) || as.integer(fit$posterior_q_n) < 1L) {
    stop("fit has no stored posterior precision", call. = FALSE)
  }
  if (!is.null(lincombs$name) && !is.null(lincombs$idx)) {
    lincombs <- list(lincombs)
  }
  names <- vapply(lincombs, function(lc) as.character(lc$name)[1], character(1))
  idxs <- lapply(lincombs, function(lc) as.integer(lc$idx))
  wts <- lapply(lincombs, function(lc) as.numeric(lc$weights))
  inla_rs_lincomb(
    fit$posterior_q_i,
    fit$posterior_q_p,
    fit$posterior_q_x,
    fit$posterior_q_n,
    fit$latent_means,
    names,
    idxs,
    wts
  )
}

inla_rs_posterior_sample <- function(q_i, q_p, q_x, q_n, means, n_samples = 1L, seed = 1) {
  .Call(
    "wrap__inla_rs_posterior_sample",
    as.integer(q_i),
    as.integer(q_p),
    as.numeric(q_x),
    as.integer(q_n)[1],
    as.numeric(means),
    as.integer(n_samples)[1],
    as.numeric(seed)[1]
  )
}

inla_rs_emarginal <- function(x, y, g_of_x) {
  .Call(
    "wrap__inla_rs_emarginal",
    as.numeric(x),
    as.numeric(y),
    as.numeric(g_of_x)
  )
}

#' Joint latent posterior draws from a fitted `"inla_rs"` object.
#' @export
inla_rs_posterior_sample_fit <- function(fit, n = 1L, seed = 1) {
  if (is.null(fit$posterior_q_n) || as.integer(fit$posterior_q_n) < 1L) {
    stop("fit has no stored posterior precision", call. = FALSE)
  }
  flat <- inla_rs_posterior_sample(
    fit$posterior_q_i,
    fit$posterior_q_p,
    fit$posterior_q_x,
    fit$posterior_q_n,
    fit$latent_means,
    n_samples = n,
    seed = seed
  )
  n_lat <- as.integer(fit$posterior_q_n)
  matrix(flat, nrow = as.integer(n), ncol = n_lat, byrow = TRUE)
}

