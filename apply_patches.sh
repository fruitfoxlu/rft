#!/usr/bin/env bash
# Apply patches to openrlhf site-packages.
# Usage: bash apply_patches.sh
#
# This script applies all patches in patches/ to the installed openrlhf package.
# Run after `pip install openrlhf==0.9.3` to apply our fixes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_PKG="$(python -c 'import openrlhf; import os; print(os.path.dirname(os.path.dirname(openrlhf.__file__)))')"

echo "Applying patches to: $SITE_PKG"

for patch in "$SCRIPT_DIR/patches"/*.patch; do
    name="$(basename "$patch")"
    echo "  Applying $name..."
    # Use --directory to set the base path, strip 1 level from diff paths
    patch -p0 --forward --directory="$SITE_PKG" < "$patch" 2>&1 || {
        echo "  WARNING: $name may already be applied (or failed)"
    }
done

echo "Done."
