#!/bin/bash
# Script to set up the development environment for the Enron Email Analysis Pipeline
# This script automates the initial setup process for new team members

set -e  # Exit immediately if a command exits with a non-zero status

echo "Setting up development environment for Enron Email Analysis Pipeline..."

# Check if Python 3.9+ is installed
python_version=$(python3 --version 2>&1 | awk '{print $2}')
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 9 ]); then
    echo "Error: Python 3.9 or higher is required. Found Python $python_version"
    exit 1
fi

echo "✅ Python $python_version detected"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo "✅ Virtual environment activated"

# Install dependencies
echo "Installing development dependencies..."
pip install -e ".[dev]"
echo "✅ Dependencies installed"

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install
echo "✅ Pre-commit hooks installed"

# Create data and output directories if they don't exist
if [ ! -d "data" ]; then
    echo "Creating data directory..."
    mkdir -p data
    echo "✅ Data directory created"
else
    echo "✅ Data directory already exists"
fi

if [ ! -d "output" ]; then
    echo "Creating output directory..."
    mkdir -p output
    echo "✅ Output directory created"
else
    echo "✅ Output directory already exists"
fi

echo ""
echo "🎉 Development environment setup complete! 🎉"
echo ""
echo "Next steps:"
echo "1. Make sure to keep your virtual environment activated:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Read the GETTING_STARTED.md file for more information on how to use the project"
echo ""
echo "3. Run the tests to make sure everything is working:"
echo "   make test"
echo ""
echo "Happy coding! 🚀"
