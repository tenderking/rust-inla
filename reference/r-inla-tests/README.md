# Reference: upstream R-INLA tests

Curated subset of scripts from
[hrue/r-inla-testing](https://github.com/hrue/r-inla-testing) (zip snapshot:
`r-inla-testing-main.zip` at the repo root — **gitignored**, do not commit).

## Purpose

These are **reference scenarios**, not CI tests for rust-inla.

- They call classic `INLA::inla(...)` and related R-INLA APIs.
- They will **not** run against `r-inla` / `inla_core` as-is.
- Use them as gold-standard problem setups when porting features or
  comparing hyperparameter recovery, mlik/DIC/CPO, and latent models.

Re-implement interesting cases as Rust unit/integration tests
(`crates/inla_core/tests/reference_ports.rs`) or `r-inla/smoke.sh`
checks. Progress checklist: [`plan.md`](../../plan.md).

## Included directories

| Directory | Maps to rust-inla |
|-----------|-------------------|
| `test-fgn` | FGN precision / Hurst recovery (R-INLA uses AR `order=3/4` approx; we use exact dense Q) |
| `test-ar1`, `test-ar` | AR(1) / AR(p) |
| `test-rw2`, `test-rw2d` | RW2 / RW2D |
| `test-seasonal`, `test-iid` | Seasonal / iid |
| `test-bym`, `test-besag2`, `test-graph` | Besag / BYM / graphs |
| `test-matern2d`, `test-spde`, `test-fmesher` | Spatial / FEM |
| `test-ccd-integration` | CCD / eigen design for θ integration |
| `test-gaussian`, `test-poisson`, `test-binomial`, `test-nbinom` | Likelihoods |
| `test-0inflated`, `test-zeroinflated-poisson` | Zero-inflated Poisson |
| `test-exponential`, `test-weibull` | Survival |
| `test-cpo`, `test-dic`, `test-mlik` | Model selection |

## Intentionally omitted

Large data / solver / plugin suites from the full dump, including:

- `test-speed`, `test-preopt`, `test-type-4` (shapefiles, RData, multi‑MB archives)
- `test-pardiso*`, `test-taucs`, `test-stiles`
- `test-rgeneric*`, `test-cgeneric*`
- VB, GCPO, lincomb, copy, group, and other APIs rust-inla does not expose yet

## License / attribution

Scripts originate from the R-INLA testing repository. See that project’s
`LICENSE` for terms. The full zip is kept locally only as an extraction
source.
