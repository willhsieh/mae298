#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PATH="${BATTLESHIP_VENV:-$ROOT/.env}"
PYTHON="$VENV_PATH/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python not found at $PYTHON" >&2
  echo "Create a venv at $ROOT/.env or set BATTLESHIP_VENV to its path." >&2
  exit 1
fi

exec "$PYTHON" "$ROOT/scripts/benchmark.py"
