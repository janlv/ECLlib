#!/bin/bash

set -e  # Exit on any error

# Extract project name from pyproject.toml (first "name" match)
PACKAGE_NAME=$(grep -m1 '^name *= *' pyproject.toml | cut -d '"' -f2)
VENV_DIR=".venv_${PACKAGE_NAME}"

echo "🔧 Creating virtual environment in $VENV_DIR ..."
python3 -m venv "$VENV_DIR"

echo "🔌 Activating virtual environment ..."
source "$VENV_DIR/bin/activate"

echo "📦 Installing local project with pip install ."
pip install .

if [ -f requirements.txt ]; then
    echo "➕ Installing additional dependencies from requirements.txt ..."
    pip install -r requirements.txt
else
    echo "ℹ️ No requirements.txt found – skipping extra dependencies."
fi

echo "✅ Done! Environment is ready in: $VENV_DIR"
echo "💡 To activate it later, run:"
echo "    source $VENV_DIR/bin/activate"
echo "💡 To deactivate the environment, just run:"
echo "    deactivate"
