# rust-inla repository instructions

Read and follow `/AGENTS.md` before making architectural, statistical, FFI, R,
or Python changes.

Key rules:

- Shared statistical semantics belong in `inla_stats`; bindings are thin host adapters.
- `inla_core` is a re-export facade, not a home for new logic.
- Registry metadata, prior dimensions, Q consumption, and summary θ ordering must agree.
- Do not duplicate built-in Q, θ tables, priors, labels, or transforms in R/Python.
- Equivalent R and Python inputs must emit equivalent plans and projectors.
- Rebuild both native bindings and run cross-language conformance after semantic changes.

Use `ARCHITECTURE.md` for design rationale and the repository tests for executable
examples.
