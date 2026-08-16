---
name: verify-inla
description: >-
  Run rust-inla tests, R/Python smoke, and cross-language conformance.
  Use after engine, registry, binding, or summary changes, or when the
  user asks to test, smoke, or check R vs Python agreement.
---

# Verify rust-inla

Prefer the smallest command that covers the change, then widen.

## Rust

```bash
cargo test -p inla_stats --lib
cargo test -p inla_math --lib
cargo test -p inla_stats --test reference_ports
cargo test --workspace
```

`reference_ports` is the behavioral stand-in for `reference/r-inla-tests/`
(do not run upstream R-INLA scripts as CI).

## Python

From repo root, with `.venv` if present:

```bash
cd py-inla && ruff check python tests && ruff format --check python tests
cd py-inla && maturin develop --release
../.venv/bin/python -m pytest tests -q
```

Conformance (R vs Python, same fit):

```bash
cargo build -p r-inla --release   # needs target/release/libinla_rs.so
cd py-inla && ../.venv/bin/python -m pytest tests/test_cross_language_conformance.py -q
```

Skips if `Rscript` or the release `.so` is missing, or if
`INLA_SKIP_R_CONFORMANCE` is set. Driver:
`py-inla/tests/conformance/fit_models.R`.

Compared fields: θ mode, mlik, DIC, WAIC, hyper labels/mean/sd, random-effect
means. A gap on one language only (e.g. missing WAIC) must fail here.

## R

```bash
make smoke-r
# or: cd r-inla && ./smoke.sh
```

Loads `target/release/libinla_rs.so`. Rebuild first after Rust/FFI changes.

## After a semantic change

If you touched registry, structured Q, constraints, options, or summaries:

1. `cargo test -p inla_stats --lib`
2. Rebuild **both** bindings (release).
3. `pytest py-inla/tests` including conformance.
4. `make smoke-r` if R FFI or `inla_rs.R` changed.

Do not treat a Python-only or R-only green as sufficient for shared semantics.
