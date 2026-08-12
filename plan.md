# Reference test port plan

Port scenarios from [`reference/r-inla-tests/`](reference/r-inla-tests/) onto
**rust-inla**. Upstream scripts call classic `INLA::inla` and are reference
only — do not run them as CI.

Engine layout (bindings consume [`inla_core`](crates/inla_core/) as a facade):

| Crate | Role |
|-------|------|
| `inla_math` | CSC, faer sparse/dense LDLᵀ, CCD/grid, design, constraints |
| `inla_stats` | Likelihoods, GMRFs, INLA inference, DIC/CPO/PIT/WAIC |
| `inla_fmesher` | Mesh / FEM / barycentric projector |
| `r-inla` / `py-inla` | R (`extendr`) and Python (`PyO3`) front-ends |

## Boundary responsibilities (Rust IR vs language skins)

Shared model semantics and a flat **execution plan** (effects, priors, constraints,
observation family, optional projector `A`) live in Rust. R and Python only
parse host formulas / data frames into that plan and reshape results for local
conventions. Heavy numeric buffers (CSC triples, `y`, design columns) cross FFI
as borrowed arrays—not JSON.

| Concern | Rust (`inla_*`) | Python (`py-inla`) | R (`r-inla`) |
| ---------------------------------- | -------- | -------------- | ----------------- |
| Likelihood / GMRF / constraint validation | **Rust** | delegate | delegate |
| Hyperparameter internal ↔ natural maps (`from.theta`) | **Rust** | delegate | delegate |
| Model / prior / `scale.model` defaults | **Rust** | delegate | delegate |
| LDLᵀ, CCD/grid, Laplace, DIC/CPO/PIT/WAIC | **Rust** | delegate | delegate |
| Latent stack layout (block offsets / names) | **Rust** | delegate | delegate |
| Built-in `Q(θ)` + rank deficiency | **Rust** | delegate | delegate |
| `rgeneric` / `inla.define` `Q` callbacks | invoke via FFI | **Python callback** | **R callback** |
| `None` / missing → omit optional fields | — | **PyO3** | — |
| `NULL` / `NA` handling | — | — | **extendr** |
| NumPy / SciPy CSC conversion | — | **PyO3 layer** | — |
| R vector / `data.frame` / `dgCMatrix` conversion | — | — | **extendr layer** |
| Python exceptions | — | **PyO3** | — |
| R `stop` / conditions | — | — | **extendr** |
| `f()` / formula / method conventions | — | **Python** (`formula.py`, `api.py`) | **R** (`inla_rs`, S3 summary/plot) |
| pandas integration | — | **Python** | — |
| tidyverse / formula-data ergonomics | — | — | **R** |

Target: adapters stay thin; if R and Python disagree numerically for the same
plan, the bug belongs in Rust (or the plan each adapter emitted).

### Option A: Spec / Plan IR in `inla_stats`

Plan types live in [`crates/inla_stats/src/plan.rs`](crates/inla_stats/src/plan.rs)
and are re-exported by `inla_core` (facade). This avoids a `core ↔ stats` cycle.

```text
R / Python skins
    → ModelSpec          (language-neutral request; Option = use default)
    → resolve()          (validate + statistical/engine defaults)
    → ModelPlan          (executable IR + LatentLayout + hyper transforms)
    → inla_stats engine  (e.g. run_gaussian_ar1_plan for the v1 slice)
         → inla_math / inla_fmesher
```

- **v1 slice:** one `Ar1` effect + fixed-precision Gaussian, η = x.
- Bindings fast-path when formula is `y ~ -1 + f(idx, model='ar1')` with
  identity index (`0..n-1` or `1..n`): `py-inla` → `run_gaussian_ar1_plan`,
  `r-inla` → `inla_rs_run_gaussian_ar1_plan`.
- **Shared structured θ→Q:** [`crates/inla_stats/src/structured.rs`](crates/inla_stats/src/structured.rs)
  (`build_structured_precision` / `structured_constraints`). R structured runner and
  Python `build_prior` (non-rgeneric / non-group / non-spde) both call this so model
  `match` arms are not duplicated in each binding.
- No formula / `PyObject` / `SEXP` in `ModelSpec` or `ModelPlan`.
- Binding/API defaults stay in R/Python; statistical + engine defaults in `resolve`.

### The three anti-drift mechanisms

1. **Model registry** —
   [`crates/inla_stats/src/registry.rs`](crates/inla_stats/src/registry.rs).
   `model_metadata(model, order, group_model, cyclic)` returns θ-length, default
   θ, rank deficiency, `scale.model` default, per-θ labels + natural-scale
   transforms, and default priors. R (`.inla_rs_model_meta`) and Python
   (`_model_meta`) both cache-wrap it; neither keeps a local table, so a new
   model needs no binding edits.
2. **Named option bag** —
   [`crates/inla_stats/src/options.rs`](crates/inla_stats/src/options.rs).
   Skins pass a `list`/`dict` to `resolve_compute_options`, which canonicalizes
   R dots and Python underscores, fills defaults and *rejects unknown keys*. A
   new control is one Rust field plus an optional alias, not a new positional
   FFI argument.
3. **Cross-language conformance test** —
   [`py-inla/tests/test_cross_language_conformance.py`](py-inla/tests/test_cross_language_conformance.py)
   with the R driver in `py-inla/tests/conformance/fit_models.R`. The same five
   models are fitted in both languages and θ, mlik, DIC, WAIC, hyperparameter
   labels/mean/sd and random-effect means are compared elementwise. Skips when
   `Rscript` or `target/release/libinla_rs.so` is missing.

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
