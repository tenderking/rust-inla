SHELL := /bin/bash

.PHONY: test bench build-r smoke-r smoke-py

test:
	cargo test --workspace

bench:
	# Only inla_math has Criterion benches; --workspace also re-runs every
	# crate's unit-test harness (all #[test] show as "ignored") which is noise.
	cargo bench -p inla_math --bench ar1_ldlt

build-r:
	cargo build -p r-inla --release

smoke-r:
	cd r-inla && ./smoke.sh

smoke-py:
	cd py-inla && ./smoke.sh
