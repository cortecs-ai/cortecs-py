#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

echo "Starting local CI checks..."

# Ruff
echo "Checking code formatting and linting with ruff"
# python3 -m pip install --upgrade pip
# python3 -m pip install ruff
ruff check --config 'pyproject.toml' --fix .

echo "Local CI checks completed."