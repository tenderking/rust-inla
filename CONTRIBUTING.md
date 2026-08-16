# Contributing

Thanks for interest in rust-inla. This project is an **experimental** Rust INLA
engine with R and Python bindings—not a drop-in replacement for classic R-INLA.

## Development setup

```bash
cargo check --workspace --exclude r-inla
cargo test --workspace --exclude r-inla --exclude inla_sys
cargo clippy --workspace --exclude r-inla --exclude inla_sys --all-targets -- -D warnings
cargo fmt --all -- --check
cd py-inla && ruff check python tests && ruff format --check python tests
```

`r-inla` needs a local R install (extendr). `inla_sys` bindgen needs a local
`gmrflib` tree and is excluded from CI.

```bash
make smoke-r    # release r-inla + R smoke
make smoke-py   # maturin + Python smoke (Python 3.13+)
```

Architecture notes: [`ARCHITECTURE.md`](ARCHITECTURE.md). Cursor agent skills
under [`.cursor/skills/`](.cursor/skills/) document the same boundaries for
automated edits.

## Issues and roadmap

Feature and performance work is tracked as **GitHub Issues**, not markdown
checklists. To (re)seed the initial roadmap issues after cloning:

```bash
gh auth login
./scripts/seed-roadmap-issues.sh
```

Please open an issue before large design changes so we can keep the shared
Rust IR (`ModelSpec` / `ModelPlan` / registry) as the source of truth.

## Pull requests

- Prefer small, reviewable PRs.
- Match `cargo fmt` / clippy (`-D warnings`) and `ruff check` / `ruff format` used in CI.
- Add or extend a `reference_ports` / smoke case when changing model behaviour.
- Keep statistical defaults and θ→Q logic in Rust; keep formula parsing in the
  language skin.

## License

Contributions are under the Apache License 2.0 — see [LICENSE](LICENSE).
