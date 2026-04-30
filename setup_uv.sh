#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
REQS="$ROOT_DIR/requirements.txt"

if command -v uv >/dev/null 2>&1; then
  echo "Attempting to install with uv (system mode): uv pip install -r $REQS --system"
  if uv pip install -r "$REQS" --system; then
    echo "uv install succeeded"
  else
    echo "uv install failed; falling back to pip"
    if python3 -m pip install -r "$REQS"; then
      echo "pip install succeeded"
    else
      echo "pip install failed in current environment. Creating a virtualenv at .venv and installing there."
      python3 -m venv "$ROOT_DIR/.venv"
      "$ROOT_DIR/.venv/bin/python" -m pip install --upgrade pip
      "$ROOT_DIR/.venv/bin/python" -m pip install -r "$REQS"
      echo "Installed into .venv. To use it: source .venv/bin/activate" 
    fi
  fi
else
  echo "uv not found; attempting pip install. To use uv install it or ensure it's on PATH."
  if python3 -m pip install -r "$REQS"; then
    echo "pip install succeeded"
  else
    echo "pip install failed in current environment. Creating a virtualenv at .venv and installing there."
    python3 -m venv "$ROOT_DIR/.venv"
    "$ROOT_DIR/.venv/bin/python" -m pip install --upgrade pip
    "$ROOT_DIR/.venv/bin/python" -m pip install -r "$REQS"
    echo "Installed into .venv. To use it: source .venv/bin/activate"
  fi
fi

echo "Done. Run: python train.py"
