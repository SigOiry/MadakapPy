# Madakappy

🌊 Seaweed mapping & biomass estimation with a modern Flet UI (and a Tk fallback). This guide covers installation and what each feature does.

## 🚀 Install / Launch

Pick one:

1) **Installer (Windows)**  
   - Download and run: `XXX`  
   - Launch the “Madakappy” shortcut; everything is bundled.

2) **Manual (Conda, all platforms)**  
   - Create/update env: `conda env update -f environment.yml`  
   - Activate: `conda activate Madakappy`  
   - Run Flet UI (recommended):  
     - PowerShell: `$env:MADAKAPPY_UI="flet"; python -m app.main`  
     - macOS/Linux: `MADAKAPPY_UI=flet python -m app.main`  
   - Tk fallback: `python -m app.main`  
   - Quick launchers (after env):  
     - Windows: double-click `launch/run_madakappy.bat`  
     - macOS: `chmod +x launch/run_madakappy.command` then double-click  
     - Linux: `chmod +x launch/run_madakappy.sh` then run/double-click

## 🧭 Key panels

- **Project Paths**: input raster (projected CRS), output directory, optional run name.  
- **Model directory (Flet)**: folder with `.joblib` RF models; refresh to rescan.  
- **Custom AOI (.shp)**: skip preselection and use your own polygons.

## 🔎 Preselection (AOI detection)
- Uses expected width/length, small-plot buffer, and blue quantile to find plots.  
- Outputs AOI polygons and an HTML preview; you confirm before classification.

## 🧠 Training (Random Forest)
- Inputs: training raster, labeled polygons, class column, optional per-class cap, optional model name.  
- Saves the `.joblib` model (and confusion matrix) to your model directory; custom name if provided.

## 🗺️ Classification (RF, pixel-wise)
- Choose a model and pixels-per-polygon sampling cap.  
- Biomass presets: Madagascar linear/quadratic, Indonesia, or custom formula (`x` = area in cm²).  
- Biomass computation: per-pixel sum (default) or polygon-area based; growth rate & SD recorded.  
- Outputs: `Output/PixelRF/Run_<name|timestamp>` shapefile + HTML map; Catalogue remembers runs.

## 🌑 Classification (Statistics / “dark rows”)
- Lightweight rule-based detector for dark linear plots.  
- Same biomass options/modes; outputs to `Output/2-Stats/Run_<name|timestamp>` with a preview map.

## 🌱 Biomass estimation
- Per-pixel: compute biomass per pixel, sum per polygon.  
- Polygon-area: apply biomass curve to polygon area (cm²).  
- Custom formulas validated (NumPy helpers like `np.log`, `np.sqrt` allowed).

## 📜 History / Catalogue (Flet)
- Lists previous runs under the current Output directory; clicking opens/rebuilds the map and shows saved settings/metrics.

## ✅ Tips
- Use projected CRS (e.g., UTM) so pixel size is in meters.  
- Confirm AOIs after preselection before running classification.  
- If Flet isn’t available, the Tk UI launches automatically.
