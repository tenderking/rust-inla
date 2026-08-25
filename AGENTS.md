# rust-inla agent instructions

These instructions are the shared architecture contract for coding agents.
Detailed rationale lives in `ARCHITECTURE.md`.

## Core rule

Shared statistical meaning belongs in Rust. R and Python parse host inputs into
the same language-neutral contract and reshape results. Never fix a semantic gap
in only one binding.

## Ownership

| Area | Owner |
|------|-------|
| Generic algebra, solvers, integration, constraints | `inla_math` |
| Likelihoods, GMRFs, inference, diagnostics | `inla_stats` |
| Model registry, priors, options, plans, structured Q | `inla_stats` |
| Mesh, FEM, projector | `inla_fmesher` |
| Compatibility re-exports only | `inla_core` |
| Formula/data UX, host conversion, callbacks, presentation | `r-inla`, `py-inla` |
| Optional legacy GMRFLib FFI | `inla_sys` |

Dependencies flow from bindings through `inla_core` to leaf crates. Do not add
statistical logic to `inla_core` or create `inla_stats ↔ inla_core` cycles.

## Placement

Put a change in `inla_stats` when it defines:

- θ meaning, length, defaults, priors, labels, or natural transforms;
- `Q(θ)`, latent dimensions, rank deficiency, scaling, or constraints;
- likelihood, diagnostic, or engine-control semantics;
- effect ordering or executable-plan behavior.

Put model-independent numerical machinery in `inla_math`.

Bindings may own:

- formula and typed API parsing;
- mapping host data, indices, and weights into a plan/projector;
- `None`/`NULL`/`NA`, NumPy/SciPy, R/Matrix, and exception conversion;
- host callbacks and host-specific result presentation.

Equivalent R and Python inputs must emit equivalent dimensions, θ order, `A`,
controls, priors, and model metadata.

## Single sources of truth

- Models and hyper metadata: `crates/inla_stats/src/registry.rs`
- Default and named priors: `crates/inla_stats/src/priors.rs`
- Multi-effect Q: `crates/inla_stats/src/structured.rs`
- Constraints: `structured.rs::structured_constraints`
- Controls and aliases: `crates/inla_stats/src/options.rs`
- Request → executable IR: `crates/inla_stats/src/plan.rs`

Do not add parallel semantic model allowlists, θ tables, labels, prior tables,
or built-in Q implementations to bindings. Parser-only aliases must resolve to
a registered model.

## Required invariants

For each built-in effect before optional group augmentation:

```text
metadata.theta_len
  == metadata.default_theta.len()
  == metadata.hyper.len()
  == default_prior_stack.theta_dim()
  == θ consumed by Q(θ)
```

Grouped effects append group metadata and priors in the same order.
Host-callback models use their declared callback θ dimension.

Also verify:

- `Q` is square and matches the declared latent size.
- Prior, Q, and summary θ ordering agree.
- Joint priors consume all of their θ coordinates; never replace them with
  per-slot or generic fallback priors.
- Constraints cover the intended null space at the correct block offset.
- Group/copy/replicate layouts preserve deterministic θ and latent ordering.
- Built-in labels and transforms come from registry metadata.

## Prohibited shortcuts

- No positional FFI arguments for new controls; extend `ComputeOptions`.
- No heavy numeric arrays over JSON; use typed borrowed buffers/CSC arrays.
- No formula strings, `PyObject`, or `SEXP` in `ModelSpec`/`ModelPlan`.
- No binding-only fallback when Rust rejects a built-in model or prior.
- Do not silently substitute a generic prior for missing registry support.

## Debugging cross-language drift

1. Compare emitted θ order/defaults, block sizes, `A`, controls, and priors.
2. If plans differ, fix the adapter emitting the wrong contract.
3. If plans agree, fix shared Rust or nondeterminism.
4. Add one R/Python conformance case; do not add compensating skin behavior.

## Verification

After semantic changes:

1. Add Rust invariant/unit tests and a behavioral test where appropriate.
2. Rebuild both R and Python native bindings.
3. Run Rust tests and clippy.
4. Run Python lint/tests and cross-language conformance.
5. Run the R smoke test when R or shared FFI behavior changed.

See `.cursor/skills/verify-inla/SKILL.md` for the repository command matrix.
