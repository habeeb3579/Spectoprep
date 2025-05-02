#!/usr/bin/env bash

# Version extraction script for Python projects
# Sets environment variables for GitHub Actions and updates conda-recipe/meta.yaml

set -eo pipefail  # Exit on error and pipe errors

echo "Extracting package version..."
PKG_VERSION=""

# Helper function to check if file exists and is readable
check_file() {
    if [ -f "$1" ] && [ -r "$1" ]; then
        return 0
    else
        echo "File $1 not found or not readable" >&2
        return 1
    fi
}

# First attempt: Extract from pyproject.toml
if check_file "pyproject.toml"; then
    echo "Found pyproject.toml, extracting version with Python..."

    PKG_VERSION=$(python3 -c '
import re
import sys

try:
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            raise ImportError("No TOML parser available")

    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
        version = (
            data.get("project", {}).get("version") or
            data.get("tool", {}).get("poetry", {}).get("version") or
            data.get("tool", {}).get("hatch", {}).get("version")
        )
        if version:
            print(version)
            sys.exit(0)
        else:
            raise ValueError("Version not found")

except Exception:
    try:
        with open("pyproject.toml", "r") as f:
            content = f.read()
        match = re.search(
            r"^version\s*=\s*[\'\\\"]([^\'\\\"]+)[\'\\\"]",
            content,
            flags=re.IGNORECASE | re.MULTILINE
        )
        if match:
            print(match.group(1).strip())
        else:
            print("", file=sys.stderr)
            sys.exit(1)
    except Exception as parse_error:
        print(f"Error parsing pyproject.toml: {str(parse_error)}", file=sys.stderr)
        sys.exit(1)
' 2>/dev/null || echo "")
fi

# Second attempt: Look for __version__ in Python package __init__.py
if [ -z "$PKG_VERSION" ]; then
    echo "Looking for __version__ in Python files..."

    PACKAGE_DIRS=$(find . -maxdepth 2 -type d \
        -not -path '*/\.*' \
        -not -path '*/venv*' \
        -not -path '*/tests*' \
        -not -path './docs*' \
        2>/dev/null || echo "")

    for DIR in $PACKAGE_DIRS; do
        if [ -f "$DIR/__init__.py" ]; then
            echo "Checking $DIR/__init__.py..."
            VERSION_LINE=$(grep -E '__version__\s*=' "$DIR/__init__.py" 2>/dev/null || echo "")
            if [ -n "$VERSION_LINE" ]; then
                PKG_VERSION=$(echo "$VERSION_LINE" | awk -F"['\"]" '{print $2}')
                if [ -n "$PKG_VERSION" ]; then
                    echo "Found version $PKG_VERSION in $DIR/__init__.py"
                    break
                fi
            fi
        fi
    done
fi

# Third attempt: Try setup.py or setup.cfg
if [ -z "$PKG_VERSION" ]; then
    if check_file "setup.py"; then
        echo "Checking setup.py..."
        PKG_VERSION=$(grep -E "version\s*=" setup.py | awk -F"['\"]" '{print $2}' | head -n1)
    fi

    if [ -z "$PKG_VERSION" ] && check_file "setup.cfg"; then
        echo "Checking setup.cfg..."
        PKG_VERSION=$(grep -E "^version\s*=" setup.cfg | awk -F= '{print $2}' | tr -d ' ')
    fi
fi

# Final fallback to a default version
if [ -z "$PKG_VERSION" ]; then
    echo "WARNING: Could not determine package version automatically." >&2
    echo "Using fallback version 0.1.0" >&2
    PKG_VERSION="0.1.0"
fi

# Clean up version string (remove whitespace, quotes)
PKG_VERSION=$(echo "$PKG_VERSION" | tr -d "[:space:]\"'")

# Validate version format (semver-like)
if ! [[ "$PKG_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([a-zA-Z0-9\.\-\+]*)?$ ]]; then
    echo "ERROR: Invalid version format: $PKG_VERSION" >&2
    exit 1
fi

# Set the version in GitHub Actions environment if available
if [ -n "${GITHUB_ENV:-}" ]; then
    echo "PKG_VERSION=$PKG_VERSION" >> "$GITHUB_ENV"
fi

echo "Building with version $PKG_VERSION"

# Update the meta.yaml file if present
if [ -f "conda-recipe/meta.yaml" ]; then
    echo "Updating conda-recipe/meta.yaml with version $PKG_VERSION..."

    # Different sed syntax for macOS vs Linux
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/{% set version = .*/{% set version = \"$PKG_VERSION\" %}/" conda-recipe/meta.yaml
    else
        sed -i "s/{% set version = .*/{% set version = \"$PKG_VERSION\" %}/" conda-recipe/meta.yaml
    fi

    echo "Updated meta.yaml version:"
    grep "{% set version" conda-recipe/meta.yaml
else
    echo "ERROR: conda-recipe/meta.yaml file not found" >&2
    exit 1
fi