#!/usr/bin/env python3
"""Clean Python cache files from the project."""

import os
import shutil
from pathlib import Path


def find_project_root(start_path: Path = None) -> Path:
    """Find the project root directory.
    
    Looks for common project root markers like .git, pyproject.toml, setup.py, etc.
    
    Args:
        start_path: Starting path for search (default: current file's directory)
        
    Returns:
        Path to project root
    """
    if start_path is None:
        start_path = Path(__file__).parent
    
    current = Path(start_path).resolve()
    
    # Look for project root markers
    markers = ['.git', 'pyproject.toml', 'setup.py', 'requirements.txt', 'venv']
    
    # Go up the directory tree
    for parent in [current] + list(current.parents):
        # Check if any marker exists
        if any((parent / marker).exists() for marker in markers):
            return parent
    
    # If no marker found, assume we're in src/utils, so go up 2 levels
    return current.parent.parent


def clean_cache():
    """Remove Python cache files and directories."""
    project_root = find_project_root()
    
    removed_items = {
        '__pycache__': 0,
        '.pyc': 0,
        '.pyo': 0,
        '.pytest_cache': 0,
        '.mypy_cache': 0,
        '.coverage': 0,
        'htmlcov': 0,
    }
    
    # Remove __pycache__ directories (excluding venv)
    for pycache_dir in project_root.rglob('__pycache__'):
        if 'venv' not in str(pycache_dir):
            try:
                shutil.rmtree(pycache_dir)
                removed_items['__pycache__'] += 1
            except Exception as e:
                print(f"Warning: Could not remove {pycache_dir}: {e}")
    
    # Remove .pyc files (excluding venv)
    for pyc_file in project_root.rglob('*.pyc'):
        if 'venv' not in str(pyc_file):
            try:
                pyc_file.unlink()
                removed_items['.pyc'] += 1
            except Exception as e:
                print(f"Warning: Could not remove {pyc_file}: {e}")
    
    # Remove .pyo files (excluding venv)
    for pyo_file in project_root.rglob('*.pyo'):
        if 'venv' not in str(pyo_file):
            try:
                pyo_file.unlink()
                removed_items['.pyo'] += 1
            except Exception as e:
                print(f"Warning: Could not remove {pyo_file}: {e}")
    
    # Remove .pytest_cache
    pytest_cache = project_root / '.pytest_cache'
    if pytest_cache.exists():
        try:
            shutil.rmtree(pytest_cache)
            removed_items['.pytest_cache'] = 1
        except Exception as e:
            print(f"Warning: Could not remove .pytest_cache: {e}")
    
    # Remove .mypy_cache
    mypy_cache = project_root / '.mypy_cache'
    if mypy_cache.exists():
        try:
            shutil.rmtree(mypy_cache)
            removed_items['.mypy_cache'] = 1
        except Exception as e:
            print(f"Warning: Could not remove .mypy_cache: {e}")
    
    # Remove .coverage
    coverage_file = project_root / '.coverage'
    if coverage_file.exists():
        try:
            coverage_file.unlink()
            removed_items['.coverage'] = 1
        except Exception as e:
            print(f"Warning: Could not remove .coverage: {e}")
    
    # Remove htmlcov
    htmlcov_dir = project_root / 'htmlcov'
    if htmlcov_dir.exists():
        try:
            shutil.rmtree(htmlcov_dir)
            removed_items['htmlcov'] = 1
        except Exception as e:
            print(f"Warning: Could not remove htmlcov: {e}")
    
    # Print summary
    print("=" * 50)
    print("Cache Cleanup Summary")
    print("=" * 50)
    total_removed = 0
    for item_type, count in removed_items.items():
        if count > 0:
            print(f"✓ Removed {count} {item_type} item(s)")
            total_removed += count
    
    if total_removed == 0:
        print("No cache files found to remove.")
    else:
        print(f"\nTotal: {total_removed} cache item(s) removed")
    print("=" * 50)


if __name__ == "__main__":
    clean_cache()

