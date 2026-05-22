#!/usr/bin/env bash
set -euo pipefail

# Multi-model fusion labeling pipeline (embed → label).
# Embeds once per backbone (cached), then fuses + evaluates + labels.
#
# Tunables (override via env):
#   DEVICE=cuda            GPU for embedding (default: cpu)
#   BACKBONES="dinov2"     space-separated backbone keys (default: all three)
#   PHOTO_ROOT=/mnt/photos photo library mount (default: /Volumes/Data/photos)
#   FORCE_EMBED=1          re-embed even if a cache already exists
# Extra args after the script name are forwarded to `label`, e.g.:
#   ./run2.sh --reject-threshold 0.6 --coherence-lambda 1.0 --audit-tagged

# Ensure PATH includes uv regardless of login shell state
export PATH="$HOME/.local/bin:$PATH"

# Install uv if the binary is missing
if ! [ -x "$HOME/.local/bin/uv" ]; then
  echo "[run2.sh] uv not found — installing …"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Fusion pipeline needs the extra deps (lightgbm, transformers; shap is optional).
# Sync core fusion deps only — shap pulls in llvmlite which can conflict on some platforms.
uv pip install --quiet lightgbm transformers

# LightGBM needs libgomp; on systems without system OpenMP, use torch's bundled copy.
TORCH_LIB="$(uv run python -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).parent / "lib")' 2>/dev/null || true)"
if [ -d "$TORCH_LIB" ]; then
  export LD_LIBRARY_PATH="${TORCH_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Ensure src/ is importable in the venv via a .pth file (self-healing).
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
SITE_PACKAGES="$(uv run python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$REPO_ROOT/src" > "$SITE_PACKAGES/babyface.pth"

# Keep the container alive: NanoClaw kills the container if /workspace/.heartbeat
# isn't touched for 30 minutes. Long embeds leave Claude Code idle long enough
# to trigger this. Touch every 5 min for the lifetime of this script.
(while true; do touch /workspace/.heartbeat; sleep 300; done) &
HEARTBEAT_PID=$!
trap "kill $HEARTBEAT_PID 2>/dev/null" EXIT

DEVICE="${DEVICE:-cpu}"
BACKBONES="${BACKBONES:-dinov2 arcface siglip}"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/embeddings}"
PHOTO_ROOT="${PHOTO_ROOT:-/Volumes/Data/photos}"
DB="$REPO_ROOT/digikam4.db"
THUMBS="$REPO_ROOT/thumbnails-digikam.db"

# Assemble repeated --backbones flags from the space-separated list.
BB_ARGS=()
for b in $BACKBONES; do BB_ARGS+=(--backbones "$b"); done

# 1. Embed once — skip if a cache already exists (FORCE_EMBED=1 to redo).
#    `embed` and `label` run as separate processes, so the macOS torch/LightGBM
#    OpenMP clash never arises here (each process loads only what it needs).
if [ -n "${FORCE_EMBED:-}" ] || ! ls "$CACHE_DIR"/emb_*.pt >/dev/null 2>&1; then
  echo "[run2.sh] Embedding backbones: $BACKBONES (device=$DEVICE) …"
  uv run python -m babyface.cli embed \
    --db "$DB" --thumbnails-db "$THUMBS" --photo-root "$PHOTO_ROOT" \
    --cache-dir "$CACHE_DIR" --device "$DEVICE" "${BB_ARGS[@]}"
else
  echo "[run2.sh] Cache present in $CACHE_DIR — skipping embed (set FORCE_EMBED=1 to redo)."
fi

# 2. Fuse + evaluate + label. Extra CLI args are forwarded to `label`.
uv run python -m babyface.cli label \
  --db "$DB" --thumbnails-db "$THUMBS" --photo-root "$PHOTO_ROOT" \
  --cache-dir "$CACHE_DIR" \
  --export-html "$REPO_ROOT/review.html" \
  --writeback "$REPO_ROOT/predictions.json" \
  "$@"
