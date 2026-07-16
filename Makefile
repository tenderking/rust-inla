SHELL := /bin/bash

.PHONY: test bench build-r smoke-r

test:
	cargo test --workspace

bench:
	cargo bench --workspace

build-r:
	cargo build -p r-inla --release

smoke-r:
	cd r-inla && ./smoke.sh
