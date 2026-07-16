n = 1000
x <- rnorm(n)
eta = 2 + x
if (!exists("y"))
    y = rpois(n, exp(eta))


r = inla(y ~ 1 + x, 
         family = "poisson", 
         control.fixed = list(mean.intercept = log(mean(y)), prec.intercept = 0.001), 
         data = data.frame(y, x), 
         control.predictor = list(hyper = list(prec = list(initial = 25))), 
         control.compute = list(dic = TRUE, po = TRUE), 
         control.inla = list(strategy = "gaussian"), 
         inla.call = "inla.mkl.work", 
         inla.mode = "classic",
         verbose = T)

rr = inla(y ~ 1 + x, 
         family = "poisson", 
         control.fixed = list(mean.intercept = log(mean(y)), prec.intercept = 0.001), 
         data = data.frame(y, x), 
         control.compute = list(dic = TRUE, po = TRUE, waic = TRUE), 
         control.inla = list(strategy = "gaussian"), 
         inla.call = "inla.mkl.work", 
         inla.mode = "experimental",
         verbose = T)

head(cbind(classic = r$dic$local.dic - r$dic$local.p.eff,
           experimental = rr$dic$local.dic - rr$dic$local.p.eff, 
           plugin = -2*dpois(y, lambda = mean(y), log = TRUE)))

head(cbind(sat.classic = r$dic$local.dic.sat - r$dic$local.p.eff,
           sat.experimental = rr$dic$local.dic.sat - rr$dic$local.p.eff,
           sat.plugin = -2*(dpois(y, lambda = mean(y), log = TRUE) - dpois(y, lambda = y, log = TRUE))))


r$summary.fixed
rr$summary.fixed
