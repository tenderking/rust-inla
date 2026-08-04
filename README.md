# rust-inla

A fast, memory-safe Rust implementation of the [Integrated Nested Laplace Approximation (INLA)](https://www.r-inla.org/) methodology. This project replaces the legacy C/Fortran backend of R-INLA with a pure-Rust engine that can be shared across both Python and R front-ends.

## Workspace Structure

The repository is a Cargo workspace mirroring the classic R-INLA split (fmesher / gmrflib / inference):

| Crate | Description |
|---|---|
| `crates/inla_fmesher` | Mesh generation, topology, point location, FEM block assembly |
| `crates/inla_math` | Sparse CSC, LDLT, design helpers, CCD/grid, Nelder–Mead (gmrflib-like) |
| `crates/inla_stats` | Likelihoods, latent GMRFs, INLA inference, DIC/CPO/PIT |
| `crates/inla_core` | Thin facade re-exporting the three crates (stable API for bindings) |
| `crates/inla_sys` | Optional legacy C FFI bindings to `gmrflib` via `bindgen` |
| `py-inla` | Python front-end via `PyO3` / `Maturin` |
| `r-inla` | R front-end via `extendr` |

Downstream crates should prefer `inla_core::…` for a stable surface, or depend on `inla_math` / `inla_fmesher` / `inla_stats` directly when iterating on one layer.

## Features

- **Pure-Rust inference engine** — faer sparse/dense $LDL^T$, Nelder–Mead hyperparameter optimisation, and Laplace approximation with analytic gradients/Hessians
- **Rayon-parallelised CCD integration** — the hyperparameter grid loop runs across CPU cores; large sparse factors can use faer Rayon as well
- **Sparse matrix support** — `sprs` CSC + faer factorize; exports to `dgCMatrix` for R and SciPy for Python
- **R bridge** — `extendr` exports precision matrices directly as native `dgCMatrix` S4 objects, bypassing file I/O
- **Python bridge** — `PyO3` / Maturin exposes the same engine with zero-copy sparse matrix handoffs
- **Model selection** — DIC, CPO, PIT, and marginal likelihoods with outlier detection heuristics

## Supported Models

Status is tracked in [`plan.md`](plan.md).

### Formula / inference (R `inla_rs`, Python `inla`)

**Latent `f()` models:** `iid`, `rw1`, `rw2`, `rw2d`, `ar1`, `ar` / `arp`, `besag`, `bym`, `bym2`, `fgn`, `seasonal`, `crw1`, `crw2` (`simple`/`pairs`/`block` in Python), `matern2d`, `spde` (Python formula; R dedicated API)

**SPDE (dedicated API):** triangular mesh → FEM `Q(κ,τ)` + barycentric projector `A`; R `inla_rs_spde(...)`, Python `f(model='spde', ...)` or `spde_precision_matrix` / `spde_projector_matrix`. θ = `[log τ, log κ]`.

**Families:** Gaussian, Poisson, Binomial, Negative Binomial, Zero-inflated Poisson/Binomial, Laplace, Exponential / Weibull survival (right-censoring via `event`; R auto-reads `data$event` when omitted)

### Still partial / deferred

**SPDE in multi-effect R formulas** — Python `f(model='spde')` works; R still uses dedicated `inla_rs_spde(...)`  
**CRW2 `layout="block"`** — Q + Python formula; R structured still defaults to `"simple"`  
**copy** — shared latent with β scaling not started

## Building

```bash
# Check all crates compile
cargo check --workspace

# Run all tests
cargo test --workspace
```

### Python (`py-inla`)

```bash
cd py-inla
pip install maturin
maturin develop --release
```

High-level R-parity API — one frontend:

```python
import inla

# `~` or `<-` both work as the response separator
result = inla(
    formula="successes <- covariate_x + f(spatial_idx, model='besag')",
    family="cbinomial",  # alias of binomial
    data={..., "adj_matrix": adj},  # DataFrame.to_dict(orient="series") works
    Ntrials=np.column_stack([y, n]),  # cbind(k, n) parity
)
print(result.latent_means[0])  # intercept (fixed effects first)

# Custom latent model (R inla.rgeneric.define)
model = inla.generic.define(
    n=20,
    Q=lambda theta: ...,  # precision matrix
    n_theta=1,
)
result = inla("y ~ -1 + f(idx, model='rgeneric')", data=..., rgeneric=model)
```

Matrix constructors (``ar1_precision_matrix_csc``, …) remain for simulation / custom ``Q``.

### R (`r-inla`)

```bash
cargo build -p r-inla --release
```

Then in R:
```r
source("r-inla/R/inla_rs.R")
.inla_rs_dynload("target/release/libinla_rs.so")
inla_rs_ar1_precision_csc(n = 100L, rho = 0.7, tau = 1.0)
```

### C FFI bindings (`inla_sys`)

Pre-generated bindings are used by default. To regenerate from the `gmrflib` headers:

```bash
cargo build -p inla_sys --features generate-bindings
```

## License

See [LICENSE](LICENSE).
