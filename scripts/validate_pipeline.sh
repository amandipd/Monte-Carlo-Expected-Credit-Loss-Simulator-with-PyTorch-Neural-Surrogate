#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Running unit tests..."
poetry run python -m pytest -q -m "not integration"

echo
echo "Running end-to-end pipeline validation..."
poetry run python scripts/validate_pipeline.py
