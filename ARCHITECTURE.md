# Architecture

How rust-inla splits work between the Rust engine and the R / Python skins.
Open feature work lives in [GitHub Issues](https://github.com/tenderking/rust-inla/issues), not in this file.

## Crates

| Crate | Role |
|-------|------|
| `inla_math` | CSC, faer sparse/dense LDLᵀ, CCD/grid, design, constraints |
| `inla_stats` | Likelihoods, GMRFs, INLA inference, DIC/CPO/PIT/WAIC |
| `inla_fmesher` | Mesh / FEM / barycentric projector |
| `inla_core` | Facade re-exporting the three crates for bindings |
| `r-inla` / `py-inla` | R (`extendr`) and Python (`PyO3`) front-ends |
| `inla_sys` | Optional legacy `gmrflib` FFI (needs a local gmrflib tree) |

## Boundary responsibilities

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

Adapters stay thin: if R and Python disagree numerically for the same plan, the
bug belongs in Rust (or the plan each adapter emitted).

## Spec / Plan IR

Plan types live in [`crates/inla_stats/src/plan.rs`](crates/inla_stats/src/plan.rs)
and are re-exported by `inla_core`.

```text
R / Python skins
    → ModelSpec          (language-neutral request; Option = use default)
    → resolve()          (validate + statistical/engine defaults)
    → ModelPlan          (executable IR + LatentLayout + hyper transforms)
    → inla_stats engine
         → inla_math / inla_fmesher
```

- Bindings fast-path when formula is `y ~ -1 + f(idx, model='ar1')` with
  identity index: `run_gaussian_ar1_plan` / `inla_rs_run_gaussian_ar1_plan`.
- Shared structured θ→Q:
  [`crates/inla_stats/src/structured.rs`](crates/inla_stats/src/structured.rs)
  (`build_structured_precision` / `structured_constraints`).
- No formula / `PyObject` / `SEXP` in `ModelSpec` or `ModelPlan`.

## Anti-drift mechanisms

1. **Model registry** —
   [`crates/inla_stats/src/registry.rs`](crates/inla_stats/src/registry.rs).
   R and Python both call `model_metadata(...)`; neither keeps a local θ table.
2. **Named option bag** —
   [`crates/inla_stats/src/options.rs`](crates/inla_stats/src/options.rs).
   Skins pass a list/dict; Rust rejects unknown keys.
3. **Cross-language conformance** —
   [`py-inla/tests/test_cross_language_conformance.py`](py-inla/tests/test_cross_language_conformance.py)
   with [`py-inla/tests/conformance/fit_models.R`](py-inla/tests/conformance/fit_models.R).

## Verification

```bash
cargo test --workspace --exclude r-inla --exclude inla_sys
cargo clippy --workspace --exclude r-inla --exclude inla_sys --all-targets -- -D warnings
make smoke-r
# Python: maturin develop --manifest-path py-inla/Cargo.toml && pytest py-inla
```

Reference scenarios from upstream R-INLA live under
[`reference/r-inla-tests/`](reference/r-inla-tests/) (not CI).
