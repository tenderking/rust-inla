---
name: add-latent-model
description: >-
  Add a built-in latent f() model while preserving one Rust-owned statistical
  contract across the engine, R, and Python. Use for a new GMRF,
  hyperparameterization, rank deficiency, or model-specific formula input.
---

# Add a latent model

Read and follow `AGENTS.md`. Use this workflow to extend the built-in latent
model system without duplicating statistical behavior in a binding.

## Decide the shape of the change

Before editing, determine:

- Is this a new model, a parameterization of an existing model, or host syntax?
- Which values define the model instance, and do they belong in the shared plan?
- What is θ order, its default, its natural transform, and its prior structure?
- What latent dimension and sparsity pattern does `Q(θ)` have?
- Is Q singular? If so, what is the complete null space?
- Does the model require a custom projector or only ordinary indices?

If the change only alters formula spelling or host conversion, keep it in the
binding. Otherwise, define its semantics in `inla_stats`.

## Implement the shared contract

1. Register metadata and defaults.
2. Define the exact prior, including joint-prior dimensionality.
3. Implement Q and validate model-specific dimensions.
4. Add structured dispatch and complete null-space constraints when needed.
5. Extend the shared plan if existing fields cannot represent the model cleanly.

Keep this invariant:

```text
metadata θ == default θ == prior θ == Q θ == summary θ
```

Fail explicitly when any part of that contract is missing. Avoid generic
fallback priors and overloaded fields whose meaning depends on undocumented
conventions.

## Adapt and verify

- Let bindings parse host data into the shared contract; do not reproduce Q,
  priors, transforms, labels, or rank rules.
- Update model discovery and typed APIs only where the binding requires it.
- Add focused tests for the mathematical property being introduced: Q values,
  sparsity, null space, parameter transform, projector, or prior.
- Add a behavioral test for the model path and a cross-language case when both
  bindings expose it.
- Follow `verify-inla`, rebuilding native bindings after shared changes.
