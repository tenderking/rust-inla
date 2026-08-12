---
name: rust-inla-architecture
description: >-
  rust-inla crate layout, Rust-vs-binding split, and anti-drift rules.
  Use when adding models, priors, controls, FFI, R/Python API, or when
  deciding where a change belongs (inla_stats vs r-inla vs py-inla).
---

# rust-inla architecture

Shared statistical semantics live in Rust. R and Python are thin skins: formula
sugar, host types, and result reshaping. If the two languages disagree
numerically for the same plan, the bug is in Rust (or the plan each skin emitted).

## Crate map

| Crate | Owns |
|-------|------|
| `inla_math` | CSC, faer LDLᵀ, CCD/grid, design, `ConstraintSpec` |
| `inla_stats` | Likelihoods, GMRFs, inference, DIC/CPO/PIT/WAIC, **registry / options / plan / structured** |
| `inla_fmesher` | Mesh / FEM / projector |
| `inla_core` | Facade re-export only — do not put new logic here |
| `r-inla` / `py-inla` | `extendr` / `PyO3` conversion + formula UX |

`inla_core` must not grow a `stats ↔ core` cycle. Plan/registry/options live in
`inla_stats` and are re-exported.

## Put it in Rust

- θ-length, default θ, rank deficiency, hyper labels, internal↔natural maps
- `scale.model` defaults, default hyperpriors
- `Q(θ)` dispatch, constraints, validation
- Engine controls (`strategy`, `dic`/`waic`/`cpo`, marginal index selection)
- Latent block layout (offsets / names)

Single tables:

- Models: `crates/inla_stats/src/registry.rs` (`model_metadata`)
- Controls: `crates/inla_stats/src/options.rs` (`resolve_compute_options`)
- Multi-effect Q: `crates/inla_stats/src/structured.rs`
- Spec → plan: `crates/inla_stats/src/plan.rs`

Bindings **cache-wrap** these. They must not keep parallel match tables.

## Keep in the language skin

- Formula / `f()` parsing, S3 vs Python method conventions
- `None`/`NULL`/`NA` → omit optional FFI fields
- NumPy / SciPy CSC or R `dgCMatrix` conversion
- Host exceptions / `stop()`
- pandas / tidyverse ergonomics

Heavy numeric buffers cross FFI as borrowed arrays, not JSON.

## Do not

- Add a positional FFI argument for a new control — add a field on
  `ComputeOptions` plus an optional alias in `canonical_key`.
- Hardcode θ indices or hyper labels in R/Python summaries.
- Implement `Q(θ)` again in `r-inla/src/inference.rs` or
  `py-inla/python/inla/api.py` for a built-in model (use
  `build_structured_precision`).
- Put formula strings or `PyObject`/`SEXP` into `ModelSpec` / `ModelPlan`.

## Custom models

`rgeneric` / `inla.define`: Rust invokes a host callback for `Q(θ)`. Natural-scale
maps for custom θ stay on the host; built-in models use the registry transforms.

## More

- Adding a latent: skill `add-latent-model`
- How to test: skill `verify-inla`
- Architecture: `ARCHITECTURE.md`
- Roadmap / deferred: GitHub Issues (`./scripts/seed-roadmap-issues.sh`)
