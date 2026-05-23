#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/node/.local/bin:$PATH"
REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

# Auto-recover uv + Python 3.14 after container restarts (they live in non-persistent /home/node/.local)
if ! command -v uv &>/dev/null; then
  echo "uv not found, reinstalling..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
if ! "$REPO/.venv/bin/python" --version &>/dev/null 2>&1; then
  echo "venv Python broken, reinstalling Python 3.14..."
  uv python install 3.14
fi

VENV_PY="$REPO/.venv/bin/python"
TORCH_LIB="$("$VENV_PY" -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).parent / "lib")' 2>/dev/null || true)"
if [ -d "$TORCH_LIB" ]; then
  export LD_LIBRARY_PATH="${TORCH_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec "$VENV_PY" -m babyface.cli web \
  --predictions predictions9.json \
  --photo-root /data/photos/photos \
  --host 0.0.0.0 \
  --port 8080 \
  "$@"
