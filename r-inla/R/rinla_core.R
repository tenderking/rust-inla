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

rinla_core_run_inla_inference <- function(initial_theta, model_type, y_obs, obs_precision, strategy = "ccd", step_or_f0 = 1.0) {
  .Call(
    "wrap__rinla_run_inla_inference",
    as.numeric(initial_theta),
    as.character(model_type),
    as.numeric(y_obs),
    as.numeric(obs_precision),
    as.character(strategy),
    as.numeric(step_or_f0)
  )
}

rinla_core_inla <- function(formula, data, family = "gaussian", strategy = "ccd", step_or_f0 = 1.0, initial_theta = NULL) {
  # 1. Parse formula
  # Response variable
  resp_var <- all.vars(formula)[1]
  y <- data[[resp_var]]
  
  # Term analysis to find f(...) call
  rhs_terms <- formula[[3]]
  
  f_call <- NULL
  if (is.call(rhs_terms) && rhs_terms[[1]] == as.symbol("f")) {
    f_call <- rhs_terms
  } else {
    find_f_call <- function(expr) {
      if (is.call(expr)) {
        if (expr[[1]] == as.symbol("f")) {
          return(expr)
        } else {
          for (i in 2:length(expr)) {
            res <- find_f_call(expr[[i]])
            if (!is.null(res)) return(res)
          }
        }
      }
      return(NULL)
    }
    f_call <- find_f_call(rhs_terms)
  }
  
  if (is.null(f_call)) {
    stop("Could not find f(...) term in formula")
  }
  
  # Evaluate the f_call in a custom environment to capture its arguments
  f_env <- new.env(parent = parent.frame())
  f_env$f <- function(x, model = "fgn", ...) {
    return(list(name = deparse(substitute(x)), model = model, args = list(...)))
  }
  
  f_struct <- eval(f_call, envir = f_env)
  idx_var <- f_struct$name
  idx_val <- data[[idx_var]]
  model_type <- f_struct$model
  
  # 2. Determine initial theta
  if (is.null(initial_theta)) {
    if (model_type == "fgn" || model_type == "ar1") {
      # FGN and AR1 have 2 hyperparameters
      initial_theta <- c(0.0, 0.0)
    } else {
      # RW2 has 1 hyperparameter
      initial_theta <- c(0.0)
    }
  }
  
  obs_precision <- 1.0
  if (!is.null(f_struct$args$obs_precision)) {
    obs_precision <- f_struct$args$obs_precision
  }
  
  # 3. Call Rust solver
  res <- rinla_core_run_inla_inference(
    initial_theta = initial_theta,
    model_type = model_type,
    y_obs = y,
    obs_precision = obs_precision,
    strategy = strategy,
    step_or_f0 = step_or_f0
  )
  
  return(res)
}

