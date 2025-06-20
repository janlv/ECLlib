#!/bin/bash

# List of local libraries to install in editable mode (relative to project root)
LOCAL_LIBS=(
    "../proclib"
)

set -e  # Exit on any error

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Extract project name from pyproject.toml (first "name" match)
PACKAGE_NAME=$(grep -m1 '^name *= *' pyproject.toml | cut -d '"' -f2)
VENV_DIR=".venv_${PACKAGE_NAME}"

# Create the virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo
    echo "🔧 Creating virtual environment in $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
else
    echo
    echo "ℹ️ Virtual environment already exists in $VENV_DIR"
fi

echo
echo "🔌 Activating virtual environment from $VENV_DIR ..."
source "$VENV_DIR/bin/activate"

# Install main project in editable mode
echo
echo "📦 Installing main project in editable mode ..."
pip install -e .

# Install local libraries in editable mode
for LIB in "${LOCAL_LIBS[@]}"; do
    echo
    echo "📦 Installing $LIB in editable mode ..."
    pip install -e "$LIB"
done

echo
echo "✅ Development environment is ready!"
echo "🔄 To activate, run: source $VENV_DIR/bin/activate"
echo "💡 To deactivate, run: deactivate"
