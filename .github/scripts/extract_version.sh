#!/usr/bin/env bash

# Version extraction script optimized for pyproject.toml-based projects
# Sets environment variables for GitHub Actions

set -e  # Exit immediately if a command exits with a non-zero status

echo "Extracting package version from pyproject.toml..."
PKG_VERSION=""

# Function to check if file exists and is readable
check_file() {
  if [ -f "$1" ] && [ -r "$1" ]; then
    return 0
  else
    return 1
  fi
}

# Check for pyproject.toml
if check_file "pyproject.toml"; then
  echo "Found pyproject.toml, attempting to extract version..."
  
  # Method 1: Try with grep
  PKG_VERSION=$(grep -oP "version\s*=\s*['\"]\\K[^'\"]*" pyproject.toml 2>/dev/null || true)
  
  # Method 2: Try with Python and regex if grep failed
  if [ -z "$PKG_VERSION" ]; then
    echo "Trying Python regex to extract version..."
    PKG_VERSION=$(python -c "import re; f = open('pyproject.toml', 'r').read(); print(re.search(r'version\s*=\s*[\'\"](.*?)[\'\"]', f).group(1))" 2>/dev/null || true)
  fi
  
  # Method 3: Try with Python and proper TOML parsing
  if [ -z "$PKG_VERSION" ]; then
    echo "Trying TOML parsing..."
    
    # Try with tomlkit first (more common with Poetry)
    PKG_VERSION=$(python -c "
try:
    import tomlkit
    data = tomlkit.parse(open('pyproject.toml').read())
    # Check for Poetry structure
    if 'tool' in data and 'poetry' in data['tool']:
        print(data['tool']['poetry']['version'])
    # Check for standard PEP 621 project metadata
    elif 'project' in data:
        print(data['project']['version'])
    else:
        print('')
except Exception as e:
    print('')
" 2>/dev/null || true)
    
    # If tomlkit failed, try with tomli (PEP 621 compliant parser)
    if [ -z "$PKG_VERSION" ]; then
      PKG_VERSION=$(python -c "
try:
    import tomli
    with open('pyproject.toml', 'rb') as f:
        data = tomli.load(f)
        # Check for standard PEP 621 project metadata
        if 'project' in data:
            print(data['project']['version'])
        # Check for Poetry structure
        elif 'tool' in data and 'poetry' in data['tool']:
            print(data['tool']['poetry']['version'])
        # Check for other common tools
        elif 'tool' in data:
            for tool in ['flit', 'hatch', 'setuptools']:
                if tool in data['tool'] and 'version' in data['tool'][tool]:
                    print(data['tool'][tool]['version'])
                    break
        else:
            print('')
except Exception as e:
    print('')
" 2>/dev/null || true)
    fi
    
    # Last resort: try the built-in configparser module with a workaround
    if [ -z "$PKG_VERSION" ]; then
      PKG_VERSION=$(python -c "
try:
    # Create a temporary file with section headers for configparser
    import tempfile, os
    with open('pyproject.toml', 'r') as src:
        content = '[dummy_section]\n' + src.read()
    
    temp = tempfile.NamedTemporaryFile(delete=False)
    with open(temp.name, 'w') as dst:
        dst.write(content)
    
    import configparser
    config = configparser.ConfigParser()
    config.read(temp.name)
    
    # Look for version in common locations
    if 'version' in config['dummy_section']:
        print(config['dummy_section']['version'].strip('\"').strip(\"'\"))
    else:
        print('')
        
    # Clean up
    os.unlink(temp.name)
except Exception as e:
    print('')
" 2>/dev/null || true)
    fi
  fi
  
  echo "Extracted version: $PKG_VERSION"
else
  echo "ERROR: pyproject.toml file not found in current directory"
fi

# If still no version found, try looking in __init__.py
if [ -z "$PKG_VERSION" ]; then
  echo "No version found in pyproject.toml, checking __init__.py files..."
  
  # Find all __init__.py files
  INIT_FILES=$(find . -type f -name "__init__.py" -not -path "*/\.*" -not -path "*/venv/*" -not -path "*/tests/*" 2>/dev/null || true)
  
  # Loop through each file
  for INIT_FILE in $INIT_FILES; do
    echo "Checking $INIT_FILE..."
    PKG_VERSION=$(grep -oP "__version__\s*=\s*['\"]\\K[^'\"]*" "$INIT_FILE" 2>/dev/null || true)
    if [ -n "$PKG_VERSION" ]; then
      echo "Found version $PKG_VERSION in $INIT_FILE"
      break
    fi
  done
fi

# Final fallback to a default version if nothing else worked
if [ -z "$PKG_VERSION" ]; then
  echo "WARNING: Could not determine package version automatically."
  echo "Using fallback version 0.0.0"
  PKG_VERSION="0.0.0"
fi

# Set the version in the GitHub environment
echo "PKG_VERSION=$PKG_VERSION" >> $GITHUB_ENV
echo "Building Conda package with version $PKG_VERSION"

# Update the meta.yaml file
if [ -f "conda-recipe/meta.yaml" ]; then
  echo "Updating conda-recipe/meta.yaml with version $PKG_VERSION..."
  # Different sed syntax required for macOS
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