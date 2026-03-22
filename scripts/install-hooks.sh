#!/bin/bash
# Install pre-commit hook

set -e

# Create .git/hooks if it doesn't exist
mkdir -p .git/hooks

# Copy the hook
cp scripts/pre-commit .git/hooks/pre-commit

# Make it executable
chmod +x .git/hooks/pre-commit

echo "Pre-commit hook installed!"
echo "It will run ruff, pyright, and pytest before each commit."
