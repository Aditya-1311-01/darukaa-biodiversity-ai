#!/usr/bin/env bash
# One-command setup for macOS / Linux.
#   bash setup.sh
set -euo pipefail

PY="${PYTHON:-python3}"
VENV=".venv"

echo "==> Checking Python"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Python not found. Install Python 3.10 or newer, then run this again." >&2
  exit 1
fi
"$PY" - <<'PYCHK'
import sys
if sys.version_info < (3, 10):
    sys.exit(f"Python 3.10+ required, found {sys.version.split()[0]}")
print(f"    using Python {sys.version.split()[0]}")
PYCHK

echo "==> Creating virtual environment in $VENV"
if [ -d "$VENV" ]; then
  echo "    $VENV already exists, reusing it"
else
  "$PY" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "==> Installing dependencies (first run downloads the embedding model, give it a few minutes)"
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "==> Created .env — open it and add your Groq key from https://console.groq.com"
else
  echo "==> .env already present, leaving it alone"
fi

echo "==> Building the vector index"
python -m knowledge.ingest || echo "    index build skipped; the API will use the keyword fallback"

echo "==> Running tests"
python -m pytest -q || echo "    tests reported failures, see above"

cat <<'DONE'

Setup complete.

  source .venv/bin/activate
  uvicorn app.main:app --reload

Then open http://localhost:8000
DONE
