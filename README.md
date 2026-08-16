# rust-inla

**Status: experimental (0.1).** A Rust engine for
[Integrated Nested Laplace Approximation (INLA)](https://www.r-inla.org/) with
Python and R bindings. Useful today for sparse GMRF-style models and for
prototyping; **not** a drop-in replacement for classic R-INLA, and not yet a
stable public API.

Roadmap and known gaps: [GitHub Issues](https://github.com/tenderking/rust-inla/issues).
Design boundaries: [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Why this project

- **One engine, two front-ends** — shared Rust core with native sparse-matrix
  handoffs to R (`dgCMatrix`) and Python (SciPy), without file-based glue.
- **Layered workspace** — meshing, numerics, inference, and bindings are
  separate crates so you can extend the right layer.
- **Modern Rust stack** — faer sparse/dense LDLᵀ, Rayon-parallel CCD
  integration, and a registry-driven model surface shared by R and Python.

## Workspace

| Crate | Description |
|---|---|
| `crates/inla_fmesher` | Mesh generation, topology, point location, FEM block assembly |
| `crates/inla_math` | Sparse CSC, LDLT, design helpers, CCD/grid, Nelder–Mead |
| `crates/inla_stats` | Likelihoods, latent GMRFs, INLA inference, DIC/CPO/PIT/WAIC |
| `crates/inla_core` | Facade re-exporting the three crates (preferred by bindings) |
| `crates/inla_sys` | Optional legacy `gmrflib` FFI via bindgen (needs local gmrflib) |
| `py-inla` | Python front-end (`PyO3` / Maturin); import name `inla` |
| `r-inla` | R front-end (`extendr`); load via `source` + dynload for now |

Prefer `inla_core::…` from bindings, or depend on a leaf crate when iterating
on one layer.

## Features (today)

- Pure-Rust inference: faer LDLᵀ, Nelder–Mead hyperparameters, Laplace
  approximation
- Sparse CSC + Rayon CCD integration
- Model-selection diagnostics: DIC, CPO, PIT, WAIC, marginal likelihood
- Shared model registry / options bag / structured Q so R and Python do not
  drift apart

## Supported models

### Formula / inference (R `inla_rs`, Python `inla`)

**Latent `f()` models:** `iid`, `rw1`, `rw2`, `rw2d`, `ar1`, `ar` / `arp`,
`besag`, `bym`, `bym2`, `fgn`, `seasonal`, `crw1`, `crw2` (`simple` / `pairs` /
`block` in Python), `matern2d`, `spde` (Python formula; R dedicated API),
`copy` (`f(j, copy="i")` with free β)

**SPDE:** triangular mesh → FEM `Q(κ,τ)` + barycentric projector `A`; R
`inla_rs_spde(...)`, Python `f(model='spde', ...)` or matrix helpers.
θ = `[log τ, log κ]`.

**Families:** Gaussian, Poisson, Binomial, Negative Binomial, zero-inflated
Poisson/Binomial, Laplace, Exponential / Weibull survival (right-censoring via
`event`; R can auto-read `data$event`)

### Known gaps

Tracked as issues (seed with `./scripts/seed-roadmap-issues.sh` if empty):

- R `rgeneric` callbacks during hyperparameter optimisation
- R multi-effect `f(model="spde")` (Python formula works; R uses `inla_rs_spde`)
- R CRW2 layouts beyond `simple`
- Sparse/banded factor path for large FGN approx and other sparse GMRFs

## Building and testing

```bash
# Core workspace (matches CI; excludes R binding and optional gmrflib FFI)
cargo check --workspace --exclude r-inla
cargo test --workspace --exclude r-inla --exclude inla_sys
cargo clippy --workspace --exclude r-inla --exclude inla_sys --all-targets -- -D warnings
cargo fmt --all -- --check
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for smoke tests and PR expectations.

### Python (`py-inla`)

```bash
cd py-inla
pip install maturin
maturin develop --release
```

```python
import inla

result = inla(
    formula="successes <- covariate_x + f(spatial_idx, model='besag')",
    family="cbinomial",  # alias of binomial
    data={..., "adj_matrix": adj},
    Ntrials=np.column_stack([y, n]),
)
print(result.latent_means[0])
```

### R (`r-inla`)

Not a CRAN package yet—build the shared library and load it:

```bash
cargo build -p r-inla --release
# or: make smoke-r
```

```r
source("r-inla/R/inla_rs.R")
.inla_rs_dynload("target/release/libinla_rs.so")
inla_rs_ar1_precision_csc(n = 100L, rho = 0.7, tau = 1.0)
```

### Optional `inla_sys` (legacy gmrflib)

Only needed if you are regenerating C bindings against a local `gmrflib` tree:

```bash
cargo build -p inla_sys --features generate-bindings
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

Upstream reference scripts under [`reference/r-inla-tests/`](reference/r-inla-tests/)
are curated from [hrue/r-inla-testing](https://github.com/hrue/r-inla-testing)
for scenario ideas only; they are not part of rust-inla CI.
