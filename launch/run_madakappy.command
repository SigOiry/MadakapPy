#!/usr/bin/env bash

# Launch Madakappy using the Conda env on macOS.
# Double-click this file in Finder. If it does not run, do:
#   chmod +x launch/run_madakappy.command

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
  # Try conda in PATH
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1090
    CONDASH="$(conda info --base)/etc/profile.d/conda.sh"
  fi
fi

if [ -z "$CONDASH" ] || [ ! -f "$CONDASH" ]; then
  osascript -e 'display alert "Madakappy" message "Could not find conda.sh. Please install Miniconda/Anaconda."'
  exit 1
fi

# shellcheck disable=SC1090
source "$CONDASH"
conda activate "$ENV_NAME"

export MADAKAPPY_UI=flet
python -m app.main

