# Reference test port plan

Port scenarios from [`reference/r-inla-tests/`](reference/r-inla-tests/) onto
**rust-inla**. Upstream scripts call classic `INLA::inla` and are reference
only — do not run them as CI.

Engine layout (bindings consume [`inla_core`](crates/inla_core/) as a facade):

| Crate | Role |
|-------|------|
| `inla_math` | CSC, faer sparse/dense LDLᵀ, CCD/grid, design, constraints |
| `inla_stats` | Likelihoods, GMRFs, INLA inference, DIC/CPO/PIT |
| `inla_fmesher` | Mesh / FEM |
| `r-inla` / `py-inla` | R (`extendr`) and Python (`PyO3`) front-ends |

## Legend

- `[x]` done (Rust e2e and/or R/Python smoke against this port)
- `[~]` partial (unit / precision / likelihood covered; no full e2e yet)
- `[ ]` blocked or not started (feature gap or deferred)

## Checklist

### Latent models

- [x] **test-ar1** — `reference_ports::port_ar1_gaussian` (τ free, ρ fixed) + R smoke AR1 (τ, ρ)
- [x] **test-ar** — `reference_ports::port_arp_gaussian` (AR(2) via PACF)
- [x] **test-fgn** — exact dense (`port_fgn_gaussian` + smoke Hurst validation) **and** R-INLA AR-mixture `order=3/4` (`port_fgn_approx_order4_gaussian` + classic formula smoke). Tables from `hrue/r-inla` `fgn-tables-{3,4}.h`.
- [x] **test-rw1** — `reference_ports::port_rw1_gaussian`
- [x] **test-rw2** — `reference_ports::port_rw2_gaussian` + R smoke RW2
- [x] **test-rw2d** — unit `rw2d::test_rw2d_cyclic`; e2e not required for grid Q alone
- [x] **test-seasonal** — `reference_ports::port_seasonal_gaussian`
- [x] **test-iid** — `reference_ports::port_iid_gaussian` (+ R smoke iid + non-Gaussian families)
- [x] **test-besag2** / **test-graph** — `reference_ports::port_besag_gaussian` (cycle graph); R/Python formula `f(..., model="besag")`
- [~] **test-bym** — precision unit `besag::test_besag_and_bym`; e2e blocked (Q is 2n without A-matrix)
- [x] **test-matern2d** — unit + `reference_ports::port_matern2d_gaussian` (grid obs, A = I)
- [x] **test-spde** — FEM Q + barycentric projector A + `port_spde_gaussian`; R `inla_rs_spde` / Python native helpers
- [~] **test-fmesher** — units in `fmesher::tests` (koala boundary load); R smoke loads example mesh

### Integration / inference machinery

- [x] **test-ccd-integration** — units in `inla_math::integration` + used by all e2e CCD fits
- [x] **sparse LDLᵀ** — faer backend in `inla_math` (`sparse-ldlt`, default on): simplicial/supernodal AUTO, Rayon for large `n`, blocked multi-RHS `diag(Q⁻¹)`
- [x] **dense faer** — SIMD LDLᵀ / LLᵀ invert / self-adjoint EVD (`dense_faer`); used by exact FGN and `scale.model`
- [x] **hard constraints** — sum-to-zero / `ConstraintSpec` + precision augmentation

### Likelihoods / families

- [x] **test-gaussian** — e2e under iid/ar1/… ports
- [x] **test-poisson** — `reference_ports::port_iid_poisson` + R smoke
- [x] **test-binomial** — `reference_ports::port_iid_binomial` + R smoke
- [~] **test-nbinom** — likelihood unit `evaluates_negative_binomial_likelihood`
- [~] **test-0inflated** / **test-zeroinflated-poisson** — ZIP likelihood units
- [~] **test-exponential** / **test-weibull** — survival likelihood units (PC-prior scripts deferred)
- [~] **test-laplace** — likelihood unit + R smoke `family="laplace"` (no reference_ports e2e)

### Model selection

- [x] **test-cpo** / **test-dic** / **test-mlik** — asserted finite on `port_iid_gaussian_model_selection`

## Implementation locations

| Artifact | Role |
|----------|------|
| [`crates/inla_stats/tests/reference_ports.rs`](crates/inla_stats/tests/reference_ports.rs) | Rust e2e ports |
| [`r-inla/smoke.sh`](r-inla/smoke.sh) | R bridge: mesh, AR1, FGN (exact + order=4), RW2, iid+poisson/binomial/laplace, FGN Hurst validation |
| [`py-inla/`](py-inla/) | Python formula API + pytest |
| Existing `#[cfg(test)]` modules | Precision / likelihood / CCD / CPO / DIC units |

**R/Python `f()` models today:** `iid`, `rw2`, `ar1`, `besag`, `fgn` (exact `order=0` or approx `3`/`4`).

## Verification

```bash
cargo test -p inla_stats --test reference_ports
cargo test -p inla_math --lib
cargo test -p inla_stats --lib
make smoke-r          # or: cd r-inla && ./smoke.sh
make smoke-py         # optional Python smoke
cargo bench -p inla_math --bench ar1_ldlt
```

## Out of scope / deferred

- Installing or calling upstream R-INLA in CI
- Committing `r-inla-testing-main.zip`
- Full BYM with projection matrix A / BYM2 parameterization
- Formula `f(..., model="spde")` multi-effect path (dedicated `inla_rs_spde` / Python `run_inla_inference(..., a=)` first)
- Exact dense FGN at large `n` (inherently Θ(n³); prefer `order=3/4` sparse approx)
- Polars-style `unsafe` micro-optimizations (not the bottleneck vs faer factorize/solve)
- Takahashi selective sparse inversion (blocked multi-RHS is the current `diag(Q⁻¹)` path)
