#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/node/.local/bin:$PATH"
REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

VENV_PY="$REPO/.venv/bin/python"
TORCH_LIB="$("$VENV_PY" -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).parent / "lib")' 2>/dev/null || true)"
if [ -d "$TORCH_LIB" ]; then
  export LD_LIBRARY_PATH="${TORCH_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

(while true; do touch /workspace/.heartbeat; sleep 300; done) &
HB=$!
trap "kill $HB 2>/dev/null" EXIT

echo "=== RUN 7: tagRegion ground truth (fixed centroids) + pseudo-labels from run6 ==="
"$VENV_PY" -m babyface.cli label \
  --db digikam4.db \
  --photo-root /data/photos/photos \
  --cache-dir embeddings \
  --min-face-px 48 \
  --ambiguous-margin 0.10 \
  --pseudo-labels predictions6.json \
  --quality-cache quality_cache.json \
  --export-html review7.html \
  --writeback predictions7.json
echo "=== run7 complete ==="
