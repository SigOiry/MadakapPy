from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
from typing import Tuple

import geopandas as gpd
import numpy as np
from PIL import Image
import rasterio
from rasterio import mask as rio_mask
from rasterio.warp import transform_bounds
from shapely.geometry import mapping
from shapely.ops import unary_union


def _band_stats(src: rasterio.io.DatasetReader, geom) -> Tuple[np.ndarray, np.ndarray]:
    indexes = list(range(1, min(3, src.count) + 1))
    data, _ = rio_mask(src, [mapping(geom)], indexes=indexes, crop=True, filled=False)
    data = data.astype("float32")
    data[(data == 0) | (data == 65535)] = np.nan
    means = np.array([np.nanmean(band) for band in data], dtype=np.float32)
    stds = np.array([np.nanstd(band) for band in data], dtype=np.float32)
    with np.errstate(invalid="ignore"):
        means /= 65535.0
        stds /= 65535.0
    return means, stds


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr)
    min_v = float(np.nanmin(arr[finite]))
    max_v = float(np.nanmax(arr[finite]))
    if max_v - min_v < 1e-8:
        return np.zeros_like(arr)
    return (arr - min_v) / (max_v - min_v)


def _ensure_brightness_score(gdf: gpd.GeoDataFrame, raster_path: str | Path) -> gpd.GeoDataFrame:
    def find_col(name: str) -> str | None:
        for col in gdf.columns:
            if col.lower() == name.lower():
                return col
        return None

    bright_col = find_col("brightness")
    score_col = find_col("score")

    if bright_col is None or gdf[bright_col].isna().all():
        brightness = np.zeros(len(gdf), dtype=np.float32)
        with rasterio.open(raster_path) as src:
            sample = gdf.to_crs(src.crs) if gdf.crs and str(gdf.crs) != str(src.crs) else gdf
            for idx, geom in enumerate(sample.geometry):
                if geom is None or geom.is_empty:
                    brightness[idx] = np.nan
                    continue
                try:
                    means, _ = _band_stats(src, geom)
                    brightness[idx] = float(np.nanmean(means)) if np.isfinite(means).any() else np.nan
                except Exception:
                    brightness[idx] = np.nan
        default_bright = (
            float(np.nanmean(brightness[np.isfinite(brightness)])) if np.isfinite(brightness).any() else 0.0
        )
        brightness = np.nan_to_num(brightness, nan=default_bright)
        gdf["brightness"] = brightness
    else:
        gdf["brightness"] = gdf[bright_col].astype(float)

    if score_col is None or gdf[score_col].isna().all():
        gdf["score"] = 1.0 - _normalize(gdf["brightness"].to_numpy())
    else:
        gdf["score"] = gdf[score_col].astype(float)
    return gdf


def _prepare_base_overlay(raster_path: str | Path) -> tuple[str, str]:
    with rasterio.open(raster_path) as src:
        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        max_w = 1600
        scale = min(1.0, max_w / max(1, src.width))
        out_h = max(1, int(src.height * scale))
        out_w = max(1, int(src.width * scale))
        if src.count >= 3:
            bands = src.read((1, 2, 3), out_shape=(3, out_h, out_w)).astype("float32")
            img_bands = []
            for band in bands:
                vmin, vmax = float(np.nanpercentile(band, 2)), float(np.nanpercentile(band, 98))
                vmax = vmin + 1.0 if vmax <= vmin else vmax
                img_bands.append(np.clip((band - vmin) / (vmax - vmin), 0.0, 1.0))
            rgb = np.transpose((np.stack(img_bands, axis=0) * 255).astype("uint8"), (1, 2, 0))
            pil = Image.fromarray(rgb, mode="RGB")
        else:
            gray = src.read(1, out_shape=(out_h, out_w)).astype("float32")
            vmin, vmax = float(np.nanpercentile(gray, 2)), float(np.nanpercentile(gray, 98))
            vmax = vmin + 1.0 if vmax <= vmin else vmax
            norm = np.clip((gray - vmin) / (vmax - vmin), 0.0, 1.0)
            pil = Image.fromarray((norm * 255).astype("uint8"), mode="L").convert("RGB")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64_png = base64.b64encode(buf.getvalue()).decode("ascii")
    bounds_json = json.dumps([[south, west], [north, east]])
    return bounds_json, b64_png


def build_classification_map(shp_path: str | Path, raster_path: str | Path, mode: str = "stats") -> Path:
    shp_path = Path(shp_path)
    raster_path = Path(raster_path)
    gdf = gpd.read_file(shp_path)
    gdf = gdf.loc[~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise RuntimeError("Classification output contains no geometries.")

    mode = (mode or "stats").lower()
    if mode not in {"stats", "rf"}:
        mode = "stats"

    plot_col = next((c for c in gdf.columns if c.lower() == "plot_id"), None)
    if plot_col is None:
        gdf["plot_id"] = np.arange(1, len(gdf) + 1)
    else:
        gdf["plot_id"] = gdf[plot_col]

    area_col = next(
        (
            c
            for c in gdf.columns
            if c.lower() in {"seg_area", "plot_area", "plot_area_", "plot_area_m2", "plotarea"}
        ),
        None,
    )
    areas: np.ndarray | None = None
    if area_col:
        try:
            areas = np.nan_to_num(gdf[area_col].astype(float).to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            areas = None
    if areas is None:
        area_source = gdf
        if gdf.crs:
            try:
                area_source = gdf.to_crs("EPSG:3857")
            except Exception:
                area_source = gdf
        areas = np.nan_to_num(area_source.geometry.area.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    if gdf.crs and str(gdf.crs) != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    gdf = _ensure_brightness_score(gdf, raster_path)

    class_col = None
    if mode == "rf":
        for candidate in ("majority", "majority_", "majclass", "rf_class", "class"):
            class_col = next((c for c in gdf.columns if c.lower() == candidate), None)
            if class_col:
                break

    features = []
    score_vals: list[float] = []
    bright_vals: list[float] = []
    rf_classes: list[str] = []
    class_values = gdf[class_col].tolist() if class_col else [None] * len(gdf)

    plot_geoms: dict[str, list] = {}
    for idx, (geom, score, bright, plot_id, class_value) in enumerate(
        zip(gdf.geometry, gdf["score"], gdf["brightness"], gdf["plot_id"], class_values)
    ):
        if geom is None or geom.is_empty:
            continue
        s = float(np.clip(score if np.isfinite(score) else 0.0, 0.0, 1.0))
        b = float(np.clip(bright if np.isfinite(bright) else 0.0, 0.0, 1.0))
        score_vals.append(s)
        bright_vals.append(b)
        area = float(areas[idx]) if idx < len(areas) else 0.0
        pid = str(plot_id)
        plot_geoms.setdefault(pid, []).append(geom)
        props = {
            "score": s,
            "brightness": b,
            "plot_id": pid,
            "seg_area": area,
        }
        if mode == "rf":
            cls_text = ""
            if class_value is not None:
                cls_text = str(class_value).strip()
            if not cls_text or cls_text.lower() == "nan":
                cls_text = "Unknown"
            props["rf_class"] = cls_text
            rf_classes.append(cls_text)
        features.append({"geometry": geom.__geo_interface__, "properties": props})

    score_min = float(min(score_vals))
    score_max = float(max(score_vals))
    bright_min = float(min(bright_vals))
    bright_max = float(max(bright_vals))

    plot_features = []
    for pid, geoms in plot_geoms.items():
        try:
            merged = unary_union(geoms)
        except Exception:
            merged = geoms[0]
            for geom in geoms[1:]:
                try:
                    merged = merged.union(geom)
                except Exception:
                    continue
        if merged is None or merged.is_empty:
            continue
        plot_features.append({"geometry": merged.__geo_interface__, "properties": {"plot_id": pid}})

    bounds_json, b64_png = _prepare_base_overlay(raster_path)

    rf_checkbox_html = ""
    class_colors: dict[str, str] = {}
    if mode == "rf":
        ordered_classes = list(dict.fromkeys(rf_classes))
        if not ordered_classes:
            ordered_classes = ["Unknown"]
        palette = [
            "#0d3b66",
            "#f95738",
            "#43aa8b",
            "#f4d35e",
            "#577590",
            "#ff6f59",
            "#8ac926",
            "#ffbf69",
            "#277da1",
            "#c44536",
        ]
        class_colors = {
            cls: palette[idx % len(palette)] for idx, cls in enumerate(ordered_classes)
        }
        checkbox_items = []
        for cls in ordered_classes:
            safe = html.escape(cls, quote=True)
            checkbox_items.append(
                f'<label><input type="checkbox" class="class-filter" data-class="{safe}" checked /> {safe}</label>'
            )
        rf_checkbox_html = "\n      ".join(checkbox_items)

    stats_template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Classification Preview</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    html, body, #map { width: 100%; height: 100%; margin: 0; padding: 0; }
    .controls {
      position: absolute;
      top: 12px;
      right: 12px;
      background: rgba(255,255,255,0.95);
      border-radius: 10px;
      padding: 12px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.2);
      font-family: "Segoe UI", sans-serif;
      min-width: 280px;
      z-index: 1000;
    }
    .controls label { display: block; margin-bottom: 6px; font-weight: 600; }
    .controls input[type="range"] { width: 100%; }
    .controls select, .controls input[type="checkbox"] { width: 100%; padding: 4px; }
    .stat { font-size: 0.85rem; color: #444; margin-top: 4px; }
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="controls">
    <label><input type="checkbox" id="togglePolygons" checked /> Show polygons</label>
    <label>Score threshold: <span id="scoreVal"></span></label>
    <input type="range" id="scoreSlider" min="0" max="1" step="0.01" />
    <label>Brightness min: <span id="brightMinVal"></span></label>
    <input type="range" id="brightSliderMin" min="0" max="1" step="0.01" />
    <label>Brightness max: <span id="brightMaxVal"></span></label>
    <input type="range" id="brightSliderMax" min="0" max="1" step="0.01" />
    <label>Color by:</label>
    <select id="colorMode">
      <option value="none">None</option>
      <option value="brightness">Brightness</option>
      <option value="score">Score</option>
    </select>
    <div class="stat" id="countStat"></div>
  </div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const imageBounds = __IMAGE_BOUNDS__;
    const imageData = "data:image/png;base64,__IMAGE_DATA__";
    const features = __FEATURES__;
    const plotFeatures = __PLOT_FEATURES__;
    const scoreMin = __SCORE_MIN__;
    const scoreMax = __SCORE_MAX__;
    const brightMin = __BRIGHT_MIN__;
    const brightMax = __BRIGHT_MAX__;

    const map = L.map('map', { maxZoom: 22 });
    L.imageOverlay(imageData, imageBounds).addTo(map);
    map.fitBounds(imageBounds);

    const segmentLayer = L.layerGroup().addTo(map);
    const plotLayer = L.layerGroup().addTo(map);

    const toggle = document.getElementById("togglePolygons");
    const scoreSlider = document.getElementById("scoreSlider");
    const brightSliderMin = document.getElementById("brightSliderMin");
    const brightSliderMax = document.getElementById("brightSliderMax");
    const scoreLabel = document.getElementById("scoreVal");
    const brightMinLabel = document.getElementById("brightMinVal");
    const brightMaxLabel = document.getElementById("brightMaxVal");
    const colorMode = document.getElementById("colorMode");
    const countStat = document.getElementById("countStat");

    scoreSlider.min = scoreMin.toFixed(3);
    scoreSlider.max = scoreMax.toFixed(3);
    const defaultScore = Math.max(scoreMin, Math.min(scoreMax, 0.85));
    scoreSlider.value = defaultScore.toFixed(3);
    brightSliderMin.min = brightMin.toFixed(3);
    brightSliderMin.max = brightMax.toFixed(3);
    const defaultBrightMin = Math.max(brightMin, Math.min(brightMax, 0.0));
    brightSliderMin.value = defaultBrightMin.toFixed(3);
    brightSliderMax.min = brightMin.toFixed(3);
    brightSliderMax.max = brightMax.toFixed(3);
    const defaultBrightMax = Math.max(defaultBrightMin, Math.min(brightMax, 0.2));
    brightSliderMax.value = defaultBrightMax.toFixed(3);

    function formatVal(v) { return Number(v).toFixed(2); }

    function ramp(val, colors) {
      const v = Math.max(0, Math.min(1, val));
      if (colors.length === 1) return colors[0];
      const idx = v * (colors.length - 1);
      const low = Math.floor(idx);
      const high = Math.min(colors.length - 1, low + 1);
      const t = idx - low;
      function interp(c1, c2) {
        const a = parseInt(c1.slice(1,3), 16);
        const b = parseInt(c2.slice(1,3), 16);
        const c = parseInt(c1.slice(3,5), 16);
        const d = parseInt(c2.slice(3,5), 16);
        const e = parseInt(c1.slice(5,7), 16);
        const f = parseInt(c2.slice(5,7), 16);
        const r = Math.round(a + (b - a) * t).toString(16).padStart(2,"0");
        const g = Math.round(c + (d - c) * t).toString(16).padStart(2,"0");
        const b2 = Math.round(e + (f - e) * t).toString(16).padStart(2,"0");
        return "#" + r + g + b2;
      }
      return interp(colors[low], colors[high]);
    }

    function styleFor(props) {
      const mode = colorMode.value;
      if (mode === "brightness") {
        const c = ramp(props.brightness, ["#0d3b66", "#f4d35e", "#ee964b"]);
        return { color: c, weight: 0, fillColor: c, fillOpacity: 0.55 };
      }
      if (mode === "score") {
        const c = ramp(1 - props.score, ["#022873", "#37b24d", "#fff3bf"]);
        return { color: c, weight: 0, fillColor: c, fillOpacity: 0.55 };
      }
      return { color: "rgba(255,204,0,0.5)", fillColor: "rgba(255,204,0,0.5)", weight: 0, fillOpacity: 0.5 };
    }

    function render() {
      segmentLayer.clearLayers();
      plotLayer.clearLayers();
      if (!toggle.checked) {
        countStat.textContent = "Polygons hidden";
        return;
      }
      const scoreThreshold = Number(scoreSlider.value);
      const brightMinThreshold = Number(brightSliderMin.value);
      const brightMaxThreshold = Number(brightSliderMax.value);
      const filtered = [];
      features.forEach(f => {
        const props = f.properties;
        if (props.score < scoreThreshold) return;
        if (props.brightness < brightMinThreshold) return;
        if (props.brightness > brightMaxThreshold) return;
        filtered.push(f);
      });
      const plotAreas = {};
      filtered.forEach(f => {
        const pid = f.properties.plot_id || "plot";
        plotAreas[pid] = (plotAreas[pid] || 0) + (f.properties.seg_area || 0);
      });
      filtered.forEach(f => {
        const props = f.properties;
        const pid = props.plot_id || "plot";
        const plotArea = plotAreas[pid] || 0;
        L.geoJSON(f.geometry, {
          style: () => styleFor(props),
          onEachFeature: (_feature, lyr) => {
            lyr.bindTooltip(`Plot ${pid} selected area: ${plotArea.toFixed(2)} m^2`, { sticky: true });
          },
        }).addTo(segmentLayer);
      });
      const totalArea = Object.values(plotAreas).reduce((sum, val) => sum + val, 0);
      plotFeatures.forEach(f => {
        const pid = f.properties.plot_id || "plot";
        const plotArea = plotAreas[pid] || 0;
        L.geoJSON(f.geometry, {
          style: () => ({ color: "#000000", opacity: 0, fillOpacity: 0, weight: 0 }),
          onEachFeature: (_feature, lyr) => {
            lyr.bindTooltip(`Plot ${pid} total selected area: ${plotArea.toFixed(2)} m^2`, { sticky: true });
          },
        }).addTo(plotLayer);
      });
      countStat.textContent = `${filtered.length} polygon(s) | Total area: ${totalArea.toFixed(2)} m^2 (score >= ${formatVal(scoreSlider.value)}, brightness ${formatVal(brightSliderMin.value)}-${formatVal(brightSliderMax.value)})`;
    }

    function refreshLabels() {
      scoreLabel.textContent = formatVal(scoreSlider.value);
      brightMinLabel.textContent = formatVal(brightSliderMin.value);
      brightMaxLabel.textContent = formatVal(brightSliderMax.value);
    }

    function handleBrightnessInput(changed) {
      const minVal = Number(brightSliderMin.value);
      const maxVal = Number(brightSliderMax.value);
      if (minVal > maxVal) {
        if (changed === "min") {
          brightSliderMax.value = brightSliderMin.value;
        } else {
          brightSliderMin.value = brightSliderMax.value;
        }
      }
      refreshLabels();
      render();
    }

    toggle.addEventListener("change", render);
    scoreSlider.addEventListener("input", () => { refreshLabels(); render(); });
    brightSliderMin.addEventListener("input", () => handleBrightnessInput("min"));
    brightSliderMax.addEventListener("input", () => handleBrightnessInput("max"));
    colorMode.addEventListener("change", render);

    refreshLabels();
    render();
  </script>
</body>
</html>"""

    rf_template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Classification Preview</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    html, body, #map { width: 100%; height: 100%; margin: 0; padding: 0; }
    .controls {
      position: absolute;
      top: 12px;
      right: 12px;
      background: rgba(255,255,255,0.95);
      border-radius: 10px;
      padding: 12px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.2);
      font-family: "Segoe UI", sans-serif;
      min-width: 280px;
      z-index: 1000;
      max-height: 80vh;
      overflow-y: auto;
    }
    .controls label { display: block; margin-bottom: 6px; font-weight: 600; }
    .class-list {
      margin-top: 8px;
      border: 1px solid #e5e5e5;
      border-radius: 6px;
      padding: 8px;
      background: #fafafa;
    }
    .class-list strong {
      display: block;
      margin-bottom: 6px;
      font-size: 0.9rem;
    }
    .class-list label {
      font-weight: 400;
      margin-bottom: 4px;
    }
    .stat { font-size: 0.85rem; color: #444; margin-top: 8px; }
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="controls">
    <label><input type="checkbox" id="togglePolygons" checked /> Show polygons</label>
    <div class="class-list">
      <strong>Classes</strong>
      __CLASS_CHECKBOXES__
    </div>
    <div class="stat" id="countStat"></div>
  </div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const imageBounds = __IMAGE_BOUNDS__;
    const imageData = "data:image/png;base64,__IMAGE_DATA__";
    const features = __FEATURES__;
    const classColors = __CLASS_COLORS__;
    const plotFeatures = __PLOT_FEATURES__;

    const map = L.map('map', { maxZoom: 22 });
    L.imageOverlay(imageData, imageBounds).addTo(map);
    map.fitBounds(imageBounds);

    const segmentLayer = L.layerGroup().addTo(map);
    const plotLayer = L.layerGroup().addTo(map);
    const toggle = document.getElementById("togglePolygons");
    const countStat = document.getElementById("countStat");
    const classFilters = Array.from(document.querySelectorAll(".class-filter"));

    function styleFor(cls) {
      const color = classColors[cls] || "#ffcc00";
      return { color, fillColor: color, fillOpacity: 0.55, weight: 0 };
    }

    function activeClasses() {
      const allowed = new Set();
      classFilters.forEach(cb => {
        if (cb.checked) allowed.add(cb.dataset.class);
      });
      return allowed;
    }

    function render() {
      segmentLayer.clearLayers();
      plotLayer.clearLayers();
      if (!toggle.checked) {
        countStat.textContent = "Polygons hidden";
        return;
      }
      const allowed = activeClasses();
      if (!allowed.size) {
        countStat.textContent = "Select at least one class";
        return;
      }
      const filtered = [];
      features.forEach(f => {
        const cls = f.properties.rf_class || "Unknown";
        if (!allowed.has(cls)) return;
        filtered.push(f);
      });
      const plotAreas = {};
      filtered.forEach(f => {
        const pid = f.properties.plot_id || "plot";
        plotAreas[pid] = (plotAreas[pid] || 0) + (f.properties.seg_area || 0);
      });
      filtered.forEach(f => {
        const props = f.properties;
        const pid = props.plot_id || "plot";
        const plotArea = plotAreas[pid] || 0;
        const cls = props.rf_class || "Unknown";
        L.geoJSON(f.geometry, {
          style: () => styleFor(cls),
          onEachFeature: (_feature, lyr) => {
            lyr.bindTooltip(`Plot ${pid} selected area: ${plotArea.toFixed(2)} m^2 (class: ${cls})`, { sticky: true });
          },
        }).addTo(segmentLayer);
      });
      const totalArea = Object.values(plotAreas).reduce((sum, val) => sum + val, 0);
      plotFeatures.forEach(f => {
        const pid = f.properties.plot_id || "plot";
        const plotArea = plotAreas[pid] || 0;
        L.geoJSON(f.geometry, {
          style: () => ({ color: "#000000", opacity: 0, fillOpacity: 0, weight: 0 }),
          onEachFeature: (_feature, lyr) => {
            lyr.bindTooltip(`Plot ${pid} total selected area: ${plotArea.toFixed(2)} m^2`, { sticky: true });
          },
        }).addTo(plotLayer);
      });
      countStat.textContent = `${filtered.length} polygon(s) | Total area: ${totalArea.toFixed(2)} m^2 | Classes: ${Array.from(allowed).join(", ")}`;
    }

    toggle.addEventListener("change", render);
    classFilters.forEach(cb => cb.addEventListener("change", render));

    render();
  </script>
</body>
</html>"""

    template = rf_template if mode == "rf" else stats_template

    html_content = template
    html_content = html_content.replace("__IMAGE_BOUNDS__", bounds_json)
    html_content = html_content.replace("__IMAGE_DATA__", b64_png)
    html_content = html_content.replace("__FEATURES__", json.dumps(features))
    html_content = html_content.replace("__PLOT_FEATURES__", json.dumps(plot_features))
    html_content = html_content.replace("__SCORE_MIN__", f"{score_min:.6f}")
    html_content = html_content.replace("__SCORE_MAX__", f"{score_max:.6f}")
    html_content = html_content.replace("__BRIGHT_MIN__", f"{bright_min:.6f}")
    html_content = html_content.replace("__BRIGHT_MAX__", f"{bright_max:.6f}")
    if mode == "rf":
        html_content = html_content.replace("__CLASS_CHECKBOXES__", rf_checkbox_html)
        html_content = html_content.replace("__CLASS_COLORS__", json.dumps(class_colors))

    out_path = shp_path.with_name(f"{shp_path.stem}_map.html")
    out_path.write_text(html_content, encoding="utf-8")
    return out_path
