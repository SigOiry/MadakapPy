#!/usr/bin/env bash

# Launch Madakappy using the Conda env on Linux.
# Make executable with: chmod +x launch/run_madakappy.sh

set -euo pipefail

ENV_NAME="Madakappy"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_ROOT="${SCRIPT_DIR}/.."
cd "$APP_ROOT"

# Try finding conda.sh in common locations
CONDASH=""
for p in \
  "$HOME/miniconda3/etc/profile.d/conda.sh" \
  "$HOME/opt/miniconda3/etc/profile.d/conda.sh" \
  "/opt/miniconda3/etc/profile.d/conda.sh" \
  "$HOME/anaconda3/etc/profile.d/conda.sh" \
  "/usr/local/anaconda3/etc/profile.d/conda.sh"; do
  if [ -f "$p" ]; then CONDASH="$p"; break; fi
done

if [ -z "$CONDASH" ]; then
  if command -v conda >/dev/null 2>&1; then
    CONDASH="$(conda info --base)/etc/profile.d/conda.sh"
  fi
fi

if [ -z "$CONDASH" ] || [ ! -f "$CONDASH" ]; then
  echo "[ERROR] Could not find conda.sh. Please install Miniconda/Anaconda." 1>&2
  exit 1
fi

source "$CONDASH"
conda activate "$ENV_NAME"

export MADAKAPPY_UI=flet
python -m app.main

