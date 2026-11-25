#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
  echo "Created .env from .env.example (fill in GEMINI_API_KEY for live tests)."
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

echo "Using Python: $PYTHON_BIN ($("$PYTHON_BIN" --version))"
"$PYTHON_BIN" -m py_compile __init__.py gemini_decisions/*.py
echo "Syntax check OK."

if command -v hermes >/dev/null 2>&1; then
  echo "Running Hermes plugin doctor..."
  hermes plugins doctor "$ROOT_DIR" --ci
  echo "Running Hermes plugin catalog validator..."
  hermes plugins validate "$ROOT_DIR"
fi

echo "Setup complete."
