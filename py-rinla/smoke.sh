#!/usr/bin/env bash
set -euo pipefail

# Find workspace root and py-rinla directory
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
uv pip install numpy scipy pytest maturin

# Build and install the Python bindings in develop mode
echo "Running maturin develop in py-rinla..."
cd "$DIR"
maturin develop

# Run pytest on the tests directory
echo "Running test suite..."
pytest -v
