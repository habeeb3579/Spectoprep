#!/usr/bin/env bash

# Version extraction script for Python projects
# Sets environment variables for GitHub Actions and updates conda-recipe/meta.yaml

set -e  # Exit immediately if a command exits with a non-zero status

echo "Extracting package version..."
PKG_VERSION=""

# Helper function to check if file exists and is readable
check_file() {
  if [ -f "$1" ] && [ -r "$1" ]; then
    return 0
  else
    return 1
  fi
}

# First attempt: Try using Python to extract from pyproject.toml (most reliable)
if check_file "pyproject.toml"; then
  echo "Found pyproject.toml, extracting version with Python..."
  
  # Use Python to parse TOML properly - handles all common project structures
  PKG_VERSION=$(python -c '
import re
import sys

try:
    # First try with tomli (Python 3.11+ has tomllib built-in)
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            # Fall back to regex if no TOML parser is available
            raise ImportError("No TOML parser available")
    
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
        # Check common locations for version
        if "project" in data and "version" in data["project"]:
            print(data["project"]["version"])
        elif "tool" in data:
            if "poetry" in data["tool"] and "version" in data["tool"]["poetry"]:
                print(data["tool"]["poetry"]["version"])
            elif "hatch" in data["tool"] and "version" in data["tool"]["hatch"]:
                print(data["tool"]["hatch"]["version"])
            else:
                # Fall back to regex as last resort
                raise KeyError("Version not found in expected locations")
        else:
            raise KeyError("No project metadata found")
except Exception:
    # Fall back to regex method
    try:
        with open("pyproject.toml", "r") as f:
            content = f.read()
        match = re.search(r"version\s*=\s*[\'\"](.*?)[\'\"]", content)
        if match:
            print(match.group(1))
        else:
            print("")
    except Exception:
        print("")
' 2>/dev/null || echo "")
fi

# Second attempt: Try looking for __version__ in Python files
if [ -z "$PKG_VERSION" ]; then
  echo "Looking for __version__ in Python files..."
  
  # Find likely module directories (src or package name directories)
  PACKAGE_DIRS=$(find . -maxdepth 2 -type d -not -path '*/\.*' -not -path '*/venv*' -not -path '*/tests*' -not -path './docs*' 2>/dev/null || echo "")
  
  # Check each directory for __init__.py with version
  for DIR in $PACKAGE_DIRS; do
    if [ -f "$DIR/__init__.py" ]; then
      echo "Checking $DIR/__init__.py..."
      VERSION_LINE=$(grep -E '__version__\s*=\s*' "$DIR/__init__.py" 2>/dev/null || echo "")
      if [ -n "$VERSION_LINE" ]; then
        PKG_VERSION=$(echo "$VERSION_LINE" | grep -oP '__version__\s*=\s*[\'"]?\K[^\'"]*')
        if [ -n "$PKG_VERSION" ]; then
          echo "Found version $PKG_VERSION in $DIR/__init__.py"
          break
        fi
      fi
    fi
  done
fi

# Third attempt: Look in setup.py or setup.cfg
if [ -z "$PKG_VERSION" ]; then
  if check_file "setup.py"; then
    echo "Checking setup.py..."
    PKG_VERSION=$(grep -oP 'version\s*=\s*[\'"]?\K[^\'"]*' setup.py 2>/dev/null || echo "")
  fi
  
  if [ -z "$PKG_VERSION" ] && check_file "setup.cfg"; then
    echo "Checking setup.cfg..."
    PKG_VERSION=$(grep -oP 'version\s*=\s*\K.*' setup.cfg 2>/dev/null || echo "")
  fi
fi

# Final fallback to a default version
if [ -z "$PKG_VERSION" ]; then
  echo "WARNING: Could not determine package version automatically."
  echo "Using fallback version 0.1.0"
  PKG_VERSION="0.1.0"
fi

# Clean up version string (remove whitespace, quotes)
PKG_VERSION=$(echo "$PKG_VERSION" | tr -d "[:space:]\"'")

# Set the version in the GitHub environment
echo "PKG_VERSION=$PKG_VERSION" >> $GITHUB_ENV
echo "Building with version $PKG_VERSION"

# Update the meta.yaml file
if [ -f "conda-recipe/meta.yaml" ]; then
  echo "Updating conda-recipe/meta.yaml with version $PKG_VERSION..."
  
  # Different sed syntax required for macOS vs Linux
  if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s/{% set version = .*/{% set version = \"$PKG_VERSION\" %}/" conda-recipe/meta.yaml
  else
    sed -i "s/{% set version = .*/{% set version = \"$PKG_VERSION\" %}/" conda-recipe/meta.yaml
  fi
  
  echo "Updated meta.yaml version:"
  grep "{% set version" conda-recipe/meta.yaml
else
  echo "ERROR: conda-recipe/meta.yaml file not found"
  exit 1
fi