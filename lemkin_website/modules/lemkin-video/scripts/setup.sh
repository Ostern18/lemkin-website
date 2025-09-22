#!/bin/bash
# Setup script for lemkin-video

set -e

echo "Setting up Video..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )(.+)')
required_version="3.10"
if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"; then
    echo "Error: Python 3.10+ is required. Found: $python_version"
    exit 1
fi

echo "✓ Python version check passed"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install package in development mode
echo "Installing lemkin-video in development mode..."
pip install -e ".[dev]"

# Verify installation
echo "Verifying installation..."
lemkin-video --version

# Run basic tests if they exist
if [ -f "tests/test_core.py" ]; then
    echo "Running basic tests..."
    pytest tests/test_core.py -v
    echo "✓ Tests passed"
fi

echo ""
echo "🎉 Video setup complete!"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the CLI:"
echo "  lemkin-video --help"
echo ""
echo "To run tests:"
echo "  make test"
echo ""
echo "To see all available commands:"
echo "  make help"
