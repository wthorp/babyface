#!/usr/bin/env bash
# Embed SigLIP using thumbnails only (no --photo-root) to avoid OOM.
# Full-res photos can be 36MB+ decompressed; thumbnails are ~256px JPEG.
# SigLIP operates at 224px so thumbnails lose nothing meaningful.
#
# After embedding, runs the full label stage WITH --photo-root so the
# HTML gallery can render face crops from the actual library.
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

if ! [ -x "$HOME/.local/bin/uv" ]; then
  echo "[run_siglip_embed.sh] installing uv …"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
DB="$REPO_ROOT/digikam4.db"
THUMBS="$REPO_ROOT/thumbnails-digikam.db"
CACHE_DIR="$REPO_ROOT/embeddings"
PHOTO_ROOT="${PHOTO_ROOT:-/data/photos/photos}"
DEVICE="${DEVICE:-cpu}"

echo "[run_siglip_embed.sh] installing fusion deps …"
uv pip install --quiet lightgbm transformers

# Auto-detect torch's bundled libgomp (needed by LightGBM on this system)
TORCH_LIB="$(uv run python -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="${TORCH_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[run_siglip_embed.sh] embedding SigLIP via thumbnails only (no photo-root) …"
uv run python -m babyface.cli embed \
  --db "$DB" \
  --thumbnails-db "$THUMBS" \
  --cache-dir "$CACHE_DIR" \
  --device "$DEVICE" \
  --backbones siglip
# No --photo-root: forces thumbnail fallback, avoids loading full-res files

echo "[run_siglip_embed.sh] running label stage …"
uv run python -m babyface.cli label \
  --db "$DB" \
  --thumbnails-db "$THUMBS" \
  --photo-root "$PHOTO_ROOT" \
  --cache-dir "$CACHE_DIR" \
  --reject-threshold 0.1 \
  --export-html "$REPO_ROOT/review.html" \
  --writeback "$REPO_ROOT/predictions.json"

echo "[run_siglip_embed.sh] done."
