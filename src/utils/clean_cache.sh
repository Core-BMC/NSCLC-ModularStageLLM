#!/bin/bash
# Clean Python cache files from the project
# This script should be run from the project root directory

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get project root (2 levels up from src/utils)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT" || exit 1

echo "Cleaning Python cache files from: $PROJECT_ROOT"
echo ""

# Remove __pycache__ directories (excluding venv)
find . -type d -name "__pycache__" -not -path "./venv/*" -exec rm -r {} + 2>/dev/null
echo "✓ Removed __pycache__ directories"

# Remove .pyc files (excluding venv)
find . -name "*.pyc" -not -path "./venv/*" -delete 2>/dev/null
echo "✓ Removed .pyc files"

# Remove .pyo files (optimized bytecode, excluding venv)
find . -name "*.pyo" -not -path "./venv/*" -delete 2>/dev/null
echo "✓ Removed .pyo files"

# Remove .pytest_cache if exists
if [ -d ".pytest_cache" ]; then
    rm -r .pytest_cache
    echo "✓ Removed .pytest_cache"
fi

# Remove .mypy_cache if exists
if [ -d ".mypy_cache" ]; then
    rm -r .mypy_cache
    echo "✓ Removed .mypy_cache"
fi

# Remove .coverage if exists
if [ -f ".coverage" ]; then
    rm .coverage
    echo "✓ Removed .coverage"
fi

# Remove htmlcov if exists
if [ -d "htmlcov" ]; then
    rm -r htmlcov
    echo "✓ Removed htmlcov"
fi

echo ""
echo "Cache cleanup completed!"

