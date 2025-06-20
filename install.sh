#!/bin/bash

set -e  # Exit on any error

# Extract project name from pyproject.toml (first "name" match)
PACKAGE_NAME=$(grep -m1 '^name *= *' pyproject.toml | cut -d '"' -f2)
VENV_DIR=".venv_${PACKAGE_NAME}"

echo
echo "🔧 Creating virtual environment in $VENV_DIR ..."
python3 -m venv "$VENV_DIR"

echo
echo "🔌 Activating virtual environment ..."
source "$VENV_DIR/bin/activate"

echo
echo "📦 Installing local project with pip install ."
pip install .

echo
echo "✅ Done! Environment is ready in: $VENV_DIR"
echo "💡 To activate it later, run:"
echo "    source $VENV_DIR/bin/activate"
echo "💡 To deactivate the environment, just run:"
echo "    deactivate"
