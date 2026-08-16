#!/usr/bin/env bash
set -euo pipefail

# Find workspace root and py-inla directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$( cd "$DIR/.." >/dev/null 2>&1 && pwd )"
VENV_DIR="$ROOT_DIR/.venv"

echo "Using virtual environment: $VENV_DIR"

# Create venv if not exists using uv
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment using uv..."
    uv venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Install required dependencies using uv pip
echo "Installing dependencies using uv..."
uv pip install "maturin>=1.14" "numpy>=2.5" "scipy>=1.16" "pytest>=9.1" "pandas>=3.0" "ruff>=0.12"

# Build and install the Python bindings in develop mode
echo "Running maturin develop in py-inla..."
cd "$DIR"
maturin develop

echo "Running ruff..."
ruff check python tests
ruff format --check python tests

# Run pytest on the tests directory
echo "Running test suite..."
pytest -v
