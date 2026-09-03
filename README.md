# rust-inla

[![CI](https://github.com/tenderking/rust-inla/actions/workflows/ci.yml/badge.svg)](https://github.com/tenderking/rust-inla/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**Status: experimental (0.1).** A modern, high-performance Rust engine for
[Integrated Nested Laplace Approximation (INLA)](https://www.r-inla.org/) with
Python and R bindings. Designed for sparse Gaussian Markov Random Field (GMRF)
models, spatial statistics, and fast Bayesian inference.

Roadmap and issue tracking: [GitHub Issues](https://github.com/tenderking/rust-inla/issues).
Architecture and design boundaries: [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Why this project

- **One engine, two front-ends**: Shared Rust core with native sparse-matrix handoffs to Python (`scipy.sparse.csc_matrix`) and R (`Matrix::dgCMatrix`), without file-based IPC or CLI subprocess glue.
- **Modern numerical stack**: Built on [faer](https://github.com/sarah-ek/faer-rs) sparse/dense LDLᵀ factorizations, Takahashi selected inversion for latent marginal variances, and Rayon-parallelized CCD / grid numerical integration.
- **Layered workspace**: Clean separation between meshing, numerical algebra, statistical inference, and language skins.
- **Single source of truth**: Model metadata, parameterizations, hyperparameter transformations, and default priors live in Rust (`inla_stats::registry`), preventing cross-language drift.

## Workspace layout

| Crate / Directory | Description |
|---|---|
| `crates/inla_fmesher` | Triangular mesh generation, topology, point location, and 2D/barrier FEM assembly |
| `crates/inla_math` | Sparse CSC algebra, LDLᵀ solver, design matrices, CCD / grid integration, Nelder–Mead |
| `crates/inla_stats` | Likelihoods, latent GMRFs, INLA inference engine, DIC / WAIC / CPO / PIT / marginal likelihood |
| `crates/inla_core` | Public facade re-exporting core crates for language bindings |
| `crates/inla_sys` | Optional legacy GMRFLib FFI via bindgen (requires local GMRFLib installation) |
| `py-inla` | Python front-end (`PyO3` / Maturin); importable as `inla` |
| `r-inla` | R front-end (`extendr`); loaded via `inla_rs` |

## Capabilities & Models

### Latent effects (`f()` models)

- **Random Walks & 1D**:
  - `rw1`, `rw2`: Discrete second differences on regular grids, or continuous irregular knot locations via Lindgren & Rue (2008) Galerkin precision (`crw2` simple).
  - `crw1`, `crw2`: Continuous-time random walks (`simple`, `pairs`, and state-space augmented `block` layouts with derivative tracking).
  - `seasonal`: Cyclic seasonal effects with arbitrary period lengths.
- **Autoregressive**:
  - `ar1`: Stationary first-order autoregressive process.
  - `ar` / `arp`: Higher-order autoregressive processes with PACF parametrization.
  - `fgn`: Fractional Gaussian noise (exact via Trench algorithm, or fast sparse circulant approximation).
- **Spatial & 2D**:
  - `besag`: Intrinsic Conditional Autoregressive (ICAR) on adjacency graphs with connected component analysis and sum-to-zero constraints.
  - `bym`, `bym2`: Combined spatial + unstructured effects (Simpson et al. scaled PC-prior parameterization).
  - `rw2d`: Second-order random walk on regular 2D lattices.
  - `matern2d`: 2D Matérn covariance on regular grids.
  - `spde`: SPDE-based spatial models on arbitrary 2D triangular meshes with FEM $Q(\tau, \kappa)$ and barycentric projector matrix $A$.
- **Unstructured & Grouped**:
  - `iid`: Unstructured Gaussian effects ($1\text{D} \dots 5\text{D}$).
  - `copy`: Shared latent effects (`f(j, copy="i")`) with free scaling parameter $\beta$.
  - `group`: Space $\times$ time interaction models via Kronecker products (e.g., Besag $\otimes$ AR(1)).

### Likelihood families

- **Continuous**: Gaussian, Laplace.
- **Discrete / Count**: Poisson, Binomial, Negative Binomial, Zero-inflated Poisson (`zip`), Zero-inflated Binomial (`zib`).
- **Survival**: Exponential and Weibull survival with right-censoring (`event` / `status`).

### Priors

- **PC Priors**: Penalized Complexity priors for precision (`pc.prec`), correlation (`pc.cor0`, `pc.cor1`), autoregressive persistence (`pc.rho0`, `pc.rho1`), BYM2 mixing (`pc.bym2`), SPDE range and variance (`pc.spde`, `pc.range`, `pc.matern`).
- **Standard**: `loggamma`, `gaussian`, `flat`, `uniform`, `logitbeta`, `wishart2d`.

### Model Diagnostics

- **Marginal log-likelihood**: Integrated evidence approximation.
- **Criteria**: DIC, WAIC.
- **Leave-one-out cross-validation**: Conditional Predictive Ordinates (CPO), Probability Integral Transform (PIT), with failure/instability diagnostics.
- **Selected Inversion**: Takahashi-computed marginal standard deviations for all latent nodes and linear predictors.

## Getting started

### Python (`py-inla`)

Requires Python 3.13+.

```bash
# Build and install locally with Maturin
cd py-inla
pip install maturin
maturin develop --release
```

#### Formula interface

```python
import numpy as np
import inla

# Simulated Poisson regression with spatial ICAR effect
adj = {1: [2, 3], 2: [1, 4], 3: [1, 4], 4: [2, 3]}
data = {
    "y": np.array([2, 5, 3, 8]),
    "x": np.array([0.1, -0.4, 0.5, 1.2]),
    "region": np.array([1, 2, 3, 4]),
    "graph": adj,
}

res = inla(
    "y ~ x + f(region, model='besag', graph='graph', scale_model=True)",
    data=data,
    family="poisson",
    dic=True,
    waic=True,
)

print(f"Marginal log-likelihood: {res.marginal_log_lik:.2f}")
print("Fixed effects:", res.summary_fixed)
print("Spatial effect means:", res.summary_random["region"]["mean"])
```

#### Declarative `ModelSpec` interface

```python
from inla import ModelSpec, Poisson, Besag, fit

class SpatialModel(ModelSpec):
    response = "y"
    family = Poisson()
    fixed = ["x"]
    spatial = Besag("region", graph=adj, scale_model=True)

res = fit(SpatialModel, data=data)
```

### R (`r-inla`)

Build the native library and load the R package:

```bash
cargo build -p r-inla --release
```

```r
source("r-inla/R/inla_rs.R")
source("r-inla/R/summary.R")
.inla_rs_dynload("target/release/libinla_rs.so")  # or .dll on Windows / .dylib on macOS

# Fit a model with formula interface
fit <- inla_rs(
  y ~ x + f(region_id, model = "besag", graph = adj_matrix, scale.model = TRUE),
  data = df,
  family = "gaussian"
)

summary(fit)
```

## Development & testing

```bash
# Run workspace unit and integration tests
cargo test --workspace --exclude r-inla --exclude inla_sys

# Clippy check
cargo clippy --workspace --exclude r-inla --exclude inla_sys --all-targets -- -D warnings

# Format check
cargo fmt --all -- --check

# Python test suite
cd py-inla
pytest tests/
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution guidelines, smoke test commands, and coding standards.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE)).
