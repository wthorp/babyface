#!/usr/bin/env bash
set -euo pipefail

# Ensure PATH includes uv regardless of login shell state
export PATH="$HOME/.local/bin:$PATH"

# Install uv if the binary is missing
if ! [ -x "$HOME/.local/bin/uv" ]; then
  echo "[run.sh] uv not found — installing …"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Sync entry point and dependencies
uv sync --quiet

# Run babypics with all forwarded args
uv run babypics "$@"
