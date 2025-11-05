try:
    # When run as a module: `python -m app.main`
    from .ui import run_app
except ImportError:
    # When run directly: `python app/main.py`
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from app.ui import run_app


if __name__ == "__main__":
    run_app()
