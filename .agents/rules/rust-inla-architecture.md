# rust-inla architecture rule

Apply this rule when changing models, priors, controls, plans, inference,
constraints, FFI, formulas, summaries, or R/Python bindings.

Read and follow @AGENTS.md.

In particular:

- Keep shared statistical meaning in Rust.
- Keep `inla_core` as a re-export facade.
- Keep R/Python limited to host parsing, conversion, callbacks, and presentation.
- Preserve the θ/prior/Q invariants and cross-language plan equivalence.
- Rebuild both bindings and run conformance after semantic changes.
