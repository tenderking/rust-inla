#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v Rscript >/dev/null 2>&1; then
  echo "Rscript not found in PATH" >&2
  exit 1
fi

echo "[1/2] Building r-inla (release)..."
cargo build -p r-inla --release

echo "[2/2] Running SPDE validation (rust-inla only)..."
Rscript validate_spde.R
