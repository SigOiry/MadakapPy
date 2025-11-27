"""
Entry-point wrapper for packing the Flet UI.

Usage (from repo root):
    flet pack launch_flet.py --name Madakappy --icon icon/madakappy.ico --assets assets

This avoids relative-import issues when running the bundled executable.
"""
from pathlib import Path
import sys


def _setup_gdal_proj_env():
    """
    When running as a frozen .exe, point GDAL/PROJ to the bundled data dirs.
    """
    # In a PyInstaller/Flet-packed app, sys._MEIPASS is the temp folder
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)
    else:
        # normal Python run - use source directory
        base = Path(__file__).resolve().parent

    gdal_data = base / "gdal_data"
    proj_data = base / "proj_data"

    if gdal_data.exists():
        os.environ.setdefault("GDAL_DATA", str(gdal_data))
    if proj_data.exists():
        os.environ.setdefault("PROJ_LIB", str(proj_data))

_setup_gdal_proj_env()

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
