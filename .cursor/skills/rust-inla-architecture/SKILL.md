---
name: rust-inla-architecture
description: >-
  Decide where rust-inla changes belong and preserve one statistical contract
  across Rust, R, and Python. Use when adding or debugging models, priors,
  controls, plans, FFI, formula behavior, summaries, or cross-language drift.
---

# rust-inla architecture

Read and follow the repository-wide architecture contract in `AGENTS.md`.

For the current task:

1. Identify the contract owner listed in `AGENTS.md`.
2. Keep shared statistical semantics in Rust and bindings host-specific.
3. Check the θ/prior/Q dimensional invariants before editing.
4. If R and Python differ, compare emitted plans before changing the engine.
5. Use `add-latent-model` for a new built-in effect.
6. Use `verify-inla` after semantic or binding changes.

Detailed rationale is in `ARCHITECTURE.md`.
