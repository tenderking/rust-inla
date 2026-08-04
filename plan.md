# Reference test port plan

Port scenarios from [`reference/r-inla-tests/`](reference/r-inla-tests/) onto
**rust-inla**. Upstream scripts call classic `INLA::inla` and are reference
only — do not run them as CI.

Engine layout (bindings consume [`inla_core`](crates/inla_core/) as a facade):

| Crate | Role |
|-------|------|
| `inla_math` | CSC, faer sparse/dense LDLᵀ, CCD/grid, design, constraints |
| `inla_stats` | Likelihoods, GMRFs, INLA inference, DIC/CPO/PIT |
| `inla_fmesher` | Mesh / FEM / barycentric projector |
| `r-inla` / `py-inla` | R (`extendr`) and Python (`PyO3`) front-ends |

## Legend

- `[x]` done (Rust e2e and/or R/Python smoke against this port)
- `[~]` partial (unit / precision / likelihood covered; no full e2e yet)
- `[ ]` blocked or not started (feature gap or deferred)

## Checklist

### Latent models

- [x] **test-ar1** — `reference_ports::port_ar1_gaussian` + R/Python formula `ar1`
- [x] **test-ar** / **arp** — `port_arp_gaussian` with free AR(2) PACF (`θ` length `1+p`) + formula `ar`/`arp`
- [x] **test-fgn** — exact dense + AR-mixture `order=3/4` ports and smoke
- [x] **test-rw1** — `port_rw1_gaussian` + R/Python formula `rw1`
- [x] **test-rw2** — `port_rw2_gaussian` + R/Python formula `rw2`
- [x] **test-rw2d** — `port_rw2d_gaussian` + R/Python formula `rw2d` (`nrow`/`ncol`; cyclic rankdef)
- [x] **test-seasonal** — `port_seasonal_gaussian` + formula `seasonal` (`season=` / cyclic)
- [x] **test-crw1** / **test-crw2** — `port_crw1_gaussian`, `port_crw2_gaussian` (`layout="simple"`) + formula with `positions=`
- [x] **test-iid** — `port_iid_gaussian` (+ non-Gaussian family ports)
- [x] **test-besag2** / **test-graph** — `port_besag_gaussian`; formula `besag`
- [x] **group** — `kronecker_csc(Q_g, Q_main)`; Python `group=` + `control_group=`; `port_group_besag_ar1_gaussian`
- [x] **test-bym** / **bym2** — classic BYM (`2n` + A=`u+v`) + BYM2 formula; ports + R/Python
- [~] **copy** — shared latent with β scaling (not started)
- [x] **rgeneric** — Python `inla.define` / formula; R `inla_rs_rgeneric_define()` (formula R-callback fit still thin)
- [x] **test-matern2d** — formula `f(model='matern2d', nrow=, ncol=)` + port
- [x] **test-spde** — FEM Q + projector A + formula `f(model='spde')` (Python); R dedicated `inla_rs_spde`
- [x] **crw2 layouts** — `simple` / `pairs` / `block` productized in Python; `port_crw2_pairs_gaussian`

### Integration / inference machinery

- [x] **test-ccd-integration** — `inla_math::integration` + e2e CCD fits
- [x] **sparse LDLᵀ** — faer (`sparse-ldlt`): AUTO supernodal, Rayon, blocked `diag(Q⁻¹)`
- [x] **dense faer** — LDLᵀ / LLᵀ invert / self-adjoint EVD
- [x] **hard constraints** — sum-to-zero / `ConstraintSpec`
- [x] **hyperpriors** — `HyperPriorStack::default_for_effect` for `iid`/`rw*`/`seasonal`/`ar*`/`crw*`/`besag`/`fgn`/`spde`

### Likelihoods / families

- [x] **test-gaussian** — e2e under iid/ar1/… ports
- [x] **test-poisson** — `port_iid_poisson` + R/Python
- [x] **test-binomial** — `port_iid_binomial` + R/Python
- [x] **test-nbinom** — `port_iid_nbinom` + formula/smoke
- [x] **test-0inflated** — `port_iid_zip` / `port_iid_zib` + formula/smoke
- [x] **test-exponential** / **test-weibull** — survival ports with `event` 0/1 censoring + auto `data$event` on R
- [x] **test-laplace** — `port_iid_laplace` + R/Python smoke

### Model selection

- [x] **test-cpo** / **test-dic** / **test-mlik** — `port_iid_gaussian_model_selection`

## Implementation locations

| Artifact | Role |
|----------|------|
| [`crates/inla_stats/tests/reference_ports.rs`](crates/inla_stats/tests/reference_ports.rs) | Rust e2e ports (22+) |
| [`r-inla/smoke.sh`](r-inla/smoke.sh) | R bridge smoke: latents, families, SPDE, FGN Hurst |
| [`py-inla/tests/`](py-inla/tests/) | Formula + SPDE + latent/family pytest |
| [`crates/inla_stats/src/priors.rs`](crates/inla_stats/src/priors.rs) | Default hyperprior stacks |

**R/Python `f()` models:** `iid`, `rw1`, `rw2`, `ar1`, `ar`/`arp`, `besag`, `fgn`, `seasonal`, `crw1`, `crw2`  
**SPDE:** dedicated API (`inla_rs_spde` / Python `a=`), not multi-effect formula yet.

## Verification

```bash
cargo test -p inla_stats --test reference_ports
cargo test -p inla_math --lib
cargo test -p inla_stats --lib
make smoke-r
# Python:
#   maturin develop --manifest-path py-inla/Cargo.toml && pytest py-inla
cargo bench -p inla_math --bench ar1_ldlt
```

## Out of scope / deferred

- Installing or calling upstream R-INLA in CI
- Committing `r-inla-testing-main.zip`
- Formula `copy=` / shared latent with free β
- R `rgeneric` optimization callbacks during Nelder–Mead (define helper only; use Python for e2e custom Q)
- R multi-effect `f(model="spde")` (Python formula done; R still uses `inla_rs_spde`)
- R structured CRW2 non-`simple` layouts (Python `pairs`/`block` done)
- Exact dense FGN at large `n` (prefer `order=3/4` sparse approx)
- Polars-style `unsafe` micro-optimizations
- Takahashi selective sparse inversion
