try:
    # When run as a module: `python -m app.main`
    from .ui import run_app as run_tk
except ImportError:
    # When run directly: `python app/main.py`
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from app.ui import run_app as run_tk

def _should_use_flet() -> bool:
    import os
    return os.environ.get("MADAKAPPY_UI", "").lower() in {"flet", "web"}

def run_app():
    if _should_use_flet():
        try:
            from .flet_app import run_flet_app
            run_flet_app()
            return
        except Exception as e:
            print(f"Flet UI failed: {e}. Falling back to Tk UI…")
    run_tk()


if __name__ == "__main__":
    run_app()
