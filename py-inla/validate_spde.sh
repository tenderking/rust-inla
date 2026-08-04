#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(cd "$DIR/.." >/dev/null 2>&1 && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

echo "Using virtual environment: $VENV_DIR"
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating Python virtual environment using uv..."
  uv venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Installing dependencies..."
uv pip install numpy scipy maturin matplotlib

echo "Building py-inla (maturin develop)..."
cd "$DIR"
maturin develop --release

echo "Running SPDE validation (rust-inla only)..."
python validate_spde.py
