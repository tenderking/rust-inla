---
name: add-latent-model
description: >-
  Add a built-in latent f() model (iid, rw, ar1, besag, …) to rust-inla.
  Use when introducing a new GMRF, hyperparameter, rank deficiency, or
  when a model currently needs R/Python table edits.
---

# Add a latent model

One Rust table, then Q + constraints + a port test. Bindings should not grow
new `match` arms for θ-length, defaults, or labels.

## Checklist

1. **Registry** — `crates/inla_stats/src/registry.rs`
   - Add the name to `SUPPORTED_MODELS`.
   - In `model_metadata`: θ slots (`HyperSlotMeta` + `default_theta`),
     rank deficiency (or special-case like seasonal/`rw2d` cyclic),
     `default_scale_model` if intrinsic.
   - Default priors via `HyperPriorStack::default_for_effect` in `priors.rs`.
   - Unit test: `theta_len == default_theta.len() == hyper.len()`.

2. **Precision** — `Q(θ)` in the existing module (`ar1.rs`, `besag.rs`, …) or a
   new `inla_stats` file. Export CSC from `inla_stats` / `inla_core`.

3. **Structured dispatch** — `crates/inla_stats/src/structured.rs`
   - `build_structured_precision` match arm.
   - `structured_constraints` if rankdef > 0 (use `sum_to_zero_constraint`,
     `plane_constraint_2d`, or `seasonal_constraint` — do not invent a 1-row
     constraint for a larger null space).
   - Encode extra integers in `StructuredEffect.order` the same way existing
     models do (`rw2d` ±nrow, seasonal season length).

4. **Do not** add θ-length / default-θ / label tables in
   `r-inla/R/inla_rs.R` or `py-inla/python/inla/api.py`. Those wrap
   `model_metadata`. Formula parsers only need to accept the new `model=`
   string and any extra kwargs (`nrow`, `graph`, `season`, …).

5. **Tests**
   - `inla_stats` unit test for Q / constraints (null space annihilated).
   - E2E: `crates/inla_stats/tests/reference_ports.rs`.
   - Cross-language: add a case to
     `py-inla/tests/conformance/fit_models.R` **and**
     `py-inla/tests/test_cross_language_conformance.py` (same formula, same data).
   - Smoke: `r-inla/smoke.sh` and a pytest in `py-inla/tests/`.

6. **Verify** — follow skill `verify-inla`. Rebuild both bindings after
   registry/structured changes (`cargo build -p r-inla --release`,
   `maturin develop --release` in `py-inla`).

## Constraints

Intrinsic models must constrain the **full** null space:

| Model | Rank |
|-------|------|
| `rw1`, `besag`, `bym` (spatial block) | 1 |
| `rw2` | 2 (const + linear) |
| `rw2d` non-cyclic | 3 (`plane_constraint_2d`) |
| `rw2d` cyclic | 1 |
| `seasonal` | `season - 1` |

A single sum-to-zero on seasonal/`rw2`/`rw2d` leaves Q singular.

## Controls

A new engine flag is a field on `ComputeOptions` in
`crates/inla_stats/src/options.rs` plus an alias in `canonical_key`.
Unknown keys must stay rejected.
