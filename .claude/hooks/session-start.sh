#!/bin/bash
set -euo pipefail

# Only run in remote (Claude Code on the web) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

echo "Installing dependencies..."
uv sync --all-extras --dev
uv pip install -e .

echo "Setting PYTHONPATH..."
echo 'export PYTHONPATH="."' >> "$CLAUDE_ENV_FILE"

echo "Session start hook complete."
