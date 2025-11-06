# Madakappy

Running the app
---------------

Option A — Double‑click launchers (recommended)

- Windows: double‑click `launch/run_madakappy.bat`
  - It locates Conda, activates the `Madakappy` env, and runs the app.

- macOS: double‑click `launch/run_madakappy.command`
  - If it doesn’t run, open Terminal and run: `chmod +x launch/run_madakappy.command`

- Linux: make executable and run
  - `chmod +x launch/run_madakappy.sh`
  - Double‑click (depending on file manager) or run: `./launch/run_madakappy.sh`

Option B — From terminal

- Activate the Conda env: `conda activate Madakappy`
- Launch Tk UI: `python -m app.main`
- Launch Flet UI (modern):
  - Windows PowerShell: `$env:MADAKAPPY_UI = "flet"; python -m app.main`
  - macOS/Linux: `MADAKAPPY_UI=flet python -m app.main`

Notes
- Ensure the `Madakappy` Conda environment is created/updated: `conda env update -f environment.yml`
- The Flet UI is defaulted by the launchers. If Flet isn’t installed, the app falls back to the Tk UI.
