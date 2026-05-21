#!/usr/bin/env bash
set -euo pipefail

# Ensure PATH includes uv regardless of login shell state
export PATH="$HOME/.local/bin:$PATH"

# Install uv if the binary is missing
if ! [ -x "$HOME/.local/bin/uv" ]; then
  echo "[run.sh] uv not found — installing …"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Sync dependencies (project itself is not installed; src/ is exposed via .pth below)
uv sync --quiet --no-install-project

# Ensure src/ is importable in the venv via a .pth file (self-healing if .venv is recreated).
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
SITE_PACKAGES="$(uv run python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$REPO_ROOT/src" > "$SITE_PACKAGES/babyface.pth"

# Run babyface CLI with all forwarded args
uv run python -m babyface.cli cluster \
  --db /Users/bill/Desktop/babyface/digikam4.db \
  --recognition-db /Users/bill/Desktop/babyface/recognition.db \
  --photo-root /Volumes/Data/photos \
  --thumbnails-db /Users/bill/Desktop/babyface/thumbnails-digikam.db \
  --baby-names Quin --baby-names Felix \
  --load-embeddings /Users/bill/Desktop/babyface/embeddings_full.pt \
  --semi-supervised --nearest-centroid --max-seeds 500 \
  --export-html clusters.html