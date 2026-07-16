# Reference test port plan

Port scenarios from [`reference/r-inla-tests/`](reference/r-inla-tests/) onto
**rust-inla** (`inla_core` / `r-inla`). Upstream scripts call classic
`INLA::inla` and are reference only — do not run them as CI.

## Legend

- `[x]` done (Rust e2e and/or R smoke against this port)
- `[~]` partial (unit / precision / likelihood covered; no full e2e yet)
- `[ ]` blocked or not started (feature gap or deferred)

## Checklist

### Latent models

- [x] **test-ar1** — `reference_ports::port_ar1_gaussian` (τ free, ρ fixed) + R smoke AR1 (τ, ρ)
- [x] **test-ar** — `reference_ports::port_arp_gaussian` (AR(2) via PACF)
- [x] **test-fgn** — exact dense (`port_fgn_gaussian` + smoke) **and** R-INLA AR-mixture `order=3/4` (`port_fgn_approx_order4_gaussian` + classic formula smoke). Tables from `hrue/r-inla` `fgn-tables-{3,4}.h`.
- [x] **test-rw2** — `reference_ports::port_rw2_gaussian` + R smoke RW2
- [x] **test-rw2d** — unit `rw2d::test_rw2d_cyclic`; e2e not required for grid Q alone
- [x] **test-seasonal** — `reference_ports::port_seasonal_gaussian`
- [x] **test-iid** — `reference_ports::port_iid_gaussian` (+ existing unit e2e)
- [x] **test-besag2** / **test-graph** — `reference_ports::port_besag_gaussian` (cycle graph)
- [~] **test-bym** — precision unit `besag::test_besag_and_bym`; e2e blocked (Q is 2n without A-matrix)
- [~] **test-matern2d** — unit `matern2d::test_matern2d_nu1`
- [~] **test-spde** — unit `spde::test_spde_precision`
- [~] **test-fmesher** — units in `fmesher::tests` (koala boundary load)

### Integration / inference machinery

- [x] **test-ccd-integration** — units in `integration` + used by all e2e CCD fits

### Likelihoods / families

- [x] **test-gaussian** — e2e under iid/ar1/… ports
- [x] **test-poisson** — `reference_ports::port_iid_poisson`
- [x] **test-binomial** — `reference_ports::port_iid_binomial`
- [~] **test-nbinom** — likelihood unit `evaluates_negative_binomial_likelihood`
- [~] **test-0inflated** / **test-zeroinflated-poisson** — ZIP likelihood units
- [~] **test-exponential** / **test-weibull** — survival likelihood units (PC-prior scripts deferred)

### Model selection

- [x] **test-cpo** / **test-dic** / **test-mlik** — asserted finite on `port_iid_gaussian_model_selection`

## Implementation locations

| Artifact | Role |
|----------|------|
| [`crates/inla_core/tests/reference_ports.rs`](crates/inla_core/tests/reference_ports.rs) | Rust e2e ports |
| [`r-inla/smoke.sh`](r-inla/smoke.sh) | R bridge: AR1, FGN, RW2 |
| Existing `#[cfg(test)]` modules | Precision / likelihood / CCD / CPO / DIC units |

## Verification

```bash
cargo test -p inla_core --test reference_ports
cargo test -p inla_core --lib
cd r-inla && ./smoke.sh
```

## Out of scope (this pass)

- Extending R formula API beyond `ar1` / `fgn` / `rw2`
- Installing or calling upstream R-INLA
- Committing `r-inla-testing-main.zip`
- Full SPDE observation models / BYM with projection matrix A
