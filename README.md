# Madakappy

Seaweed mapping and biomass estimation suite with two UIs (modern Flet and fallback Tk). This guide covers installation options and what each major feature does.

## Install / Launch

### Option 1: Conda environment (all platforms)
1. Create or update the env: `conda env update -f environment.yml`
2. Activate: `conda activate Madakappy`
3. Run:
   - Modern Flet UI (recommended): `MADAKAPPY_UI=flet python -m app.main` (PowerShell: `$env:MADAKAPPY_UI="flet"; python -m app.main`)
   - Tk UI fallback: `python -m app.main`

### Option 2: Packaged installer (Windows, via Inno)
- Build the .exe with `flet pack` + your Inno setup script, then install.
- After installation, launch the “Madakappy” shortcut; it ships the Conda-based runtime and opens the Flet UI.

### Option 3: Simple launchers (source checkout)
- Windows: double-click `launch/run_madakappy.bat`
- macOS: double-click `launch/run_madakappy.command` (make executable if needed)
- Linux: `chmod +x launch/run_madakappy.sh` then run it

## Features and how to use them

### Project Paths
- **Input raster**: GeoTIFF to process (projected CRS required).
- **Output directory**: Root where results, models, and history are written.
- **Run name (optional)**: Custom label used in output folder/file names and history.
- **Model directory (Flet UI)**: Folder containing `.joblib` RF models; refreshes the dropdown.
- **Custom AOI (.shp)**: Skip preselection and use your own polygons; they open in the editor for review.

### Preselection (AOI detection)
- Detects cultivation plots from the raster using expected width/length ranges, small-plot buffering, and a blue-band quantile.
- Produces AOI polygons you can review/confirm before classification; exports an AOI preview map.

### Training (Random Forest)
- Inputs: training raster, labeled polygons, class column, optional per-class pixel cap, optional model name.
- Trains an RF, saves the model (`Model` directory or your chosen folder) and a confusion matrix image.
- Model name is used for the `.joblib` filename; blank keeps timestamped naming.

### Classification (RF, pixel-wise vectorization)
- Select a trained model and set pixels-per-polygon sampling limit.
- Biomass options:
  - **Preset models**: Madagascar linear/quadratic, Indonesia curve.
  - **Custom equation**: Provide `biomass (g) = f(x)` using `x` = area in cm².
  - **Computation mode**: Per-pixel sum (default) or polygon-area based.
  - Growth rate and SD stored/applied for 7-day projections.
- Outputs per-run shapefile in `Output/PixelRF/Run_<name or timestamp>`, plus an interactive HTML map preview.
- Catalogue remembers runs; selecting one reopens its map (or rebuilds it if missing).

### Classification (Statistics / “dark rows”)
- Lightweight rule-based classifier for dark linear features.
- Respects the same biomass presets/custom equations and computation modes.
- Outputs to `Output/2-Stats/Run_<name or timestamp>` with a preview map.

### Biomass estimation
- Driven by chosen model and computation mode:
  - **Per-pixel**: compute biomass per pixel then sum per polygon.
  - **Polygon area**: apply the biomass curve directly to polygon area (cm²).
- Custom equations are validated for safety and use NumPy helpers (`np.log`, `np.sqrt`, etc.).

### History / Catalogue (Flet UI)
- Shows past runs stored under the current Output directory.
- Clicking a run opens its map (rebuilds if needed) and displays saved settings/metrics.

## Notes / tips
- Keep rasters in projected CRS (e.g., UTM) so pixel sizes are in meters.
- The AOI editor opens after preselection; confirm before classification continues.
- If Flet is unavailable, the app falls back to the Tk UI automatically.***
