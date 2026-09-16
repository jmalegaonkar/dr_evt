#!/usr/bin/env bash
# Run the dr_evt_market adapter and prediction test suite.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX:-$REPO_ROOT/install}"
PYTHON_BIN="${PYTHON_EXECUTABLE:-python3}"

if [[ "$INSTALL_PREFIX" != /* ]]; then
    INSTALL_PREFIX="$REPO_ROOT/${INSTALL_PREFIX#./}"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: Python interpreter not found: $PYTHON_BIN" >&2
    exit 1
fi

MARKET_PATH="$INSTALL_PREFIX/lib/python:$REPO_ROOT/python"
export PYTHONPATH="$MARKET_PATH${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"
"$PYTHON_BIN" -m unittest discover -s python/dr_evt_market/tests -v
