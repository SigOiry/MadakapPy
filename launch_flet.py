"""
Entry-point wrapper for packing the Flet UI.

Usage (from repo root):
    flet pack launch_flet.py --name Madakappy --icon icon/madakappy.ico --assets assets

This avoids relative-import issues when running the bundled executable.
"""
from pathlib import Path
import sys

# Ensure the repo root is on sys.path so `app.*` imports work when bundled.
ROOT = Path(__file__).resolve().parent
# If packing from the app directory, allow a relative "app" folder one level up.
APP_DIR = ROOT / "app"
if APP_DIR.exists():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(APP_DIR.parent))

from app.flet_app import run_flet_app  # noqa: E402


if __name__ == "__main__":
    run_flet_app()
