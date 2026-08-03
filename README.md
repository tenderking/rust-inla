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

Status is tracked in [`plan.md`](plan.md). Many GMRF builders and likelihood kernels exist in Rust; only a subset is wired through the R/Python formula APIs and e2e ports.

### Formula / inference (R `inla_rs`, Python `inla`)

**Latent `f()` models:** `iid`, `rw1` (Python), `rw2`, `ar1`, `besag`, `fgn` (exact dense or AR-mixture `order=3/4`)

**SPDE (dedicated API):** triangular mesh → FEM `Q(κ,τ)` + barycentric projector `A`; R `inla_rs_spde(...)`, Python `spde_precision_matrix` / `spde_projector_matrix` + `run_inla_inference(..., a=A)`. θ = `[log τ, log κ]`.

**Families with e2e coverage:** Gaussian, Poisson, Binomial (plus Laplace smoke on R)

### Precision / likelihood kernels only (not full formula e2e)

**Latent Q builders:** AR(p), RW1, CRW1/CRW2, seasonal, BYM (2n block; needs A for full model), Matérn 2D lattice (e2e port with A = I)

**Likelihood eval units:** Negative binomial, zero-inflated Poisson/Binomial, Exponential / Weibull survival
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
