from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import mapping
try:
    from .biomass import biomass_from_area_m2
except Exception:  # noqa: BLE001
    from biomass import biomass_from_area_m2  # type: ignore


def _load_raster_overlay(raster_path: Path) -> tuple[str, str]:
    """Return (bounds_json, base64_png). Always returns something even if raster fails."""
    fallback_bounds = json.dumps([[-90, -180], [90, 180]])
    fallback_img = base64.b64encode(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT\x08\xd7c``\x00\x00\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82").decode(
        "ascii"
    )
    try:
        with rasterio.open(raster_path) as src:
            src_crs = src.crs if src.crs else "EPSG:4326"
            west, south, east, north = transform_bounds(src_crs, "EPSG:4326", *src.bounds, densify_pts=21)
            max_w = 1600
            scale = min(1.0, max_w / max(1, src.width))
            out_h = max(1, int(src.height * scale))
            out_w = max(1, int(src.width * scale))
            bands = src.read(out_shape=(src.count, out_h, out_w)).astype("float32")
            if bands.shape[0] >= 3:
                rgb = bands[:3]
            else:
                rgb = np.repeat(bands[:1], 3, axis=0)
            img_bands = []
            for band in rgb:
                vmin, vmax = float(np.nanpercentile(band, 2)), float(np.nanpercentile(band, 98))
                vmax = vmin + 1.0 if vmax <= vmin else vmax
                img_bands.append(np.clip((band - vmin) / (vmax - vmin), 0.0, 1.0))
            rgb_img = np.transpose((np.stack(img_bands, axis=0) * 255).astype("uint8"), (1, 2, 0))
            buf = io.BytesIO()
            from PIL import Image

            Image.fromarray(rgb_img, mode="RGB").save(buf, format="PNG")
            b64_png = base64.b64encode(buf.getvalue()).decode("ascii")
        bounds_json = json.dumps([[south, west], [north, east]])
        return bounds_json, b64_png
    except Exception:
        return fallback_bounds, fallback_img


def _choose_projected_crs(primary: object | None, secondary: object | None) -> str | object:
    def _is_projected(c):
        try:
            return getattr(c, "is_projected", False)
        except Exception:
            return False

    if primary and _is_projected(primary):
        return primary
    if secondary and _is_projected(secondary):
        return secondary
    return "EPSG:3857"


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr)
    mn, mx = float(np.nanmin(arr[finite])), float(np.nanmax(arr[finite]))
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _ensure_brightness_score(gdf: gpd.GeoDataFrame, raster_path: Path) -> gpd.GeoDataFrame:
    bright_col = next((c for c in gdf.columns if c.lower() == "brightness"), None)
    score_col = next((c for c in gdf.columns if c.lower() == "score"), None)
    if bright_col is None:
        gdf["brightness"] = 0.0
    else:
        gdf["brightness"] = gdf[bright_col].astype(float)
    if score_col is None:
        gdf["score"] = 1.0 - _normalize(gdf["brightness"].to_numpy())
    else:
        gdf["score"] = gdf[score_col].astype(float)
    return gdf


def _ensure_plot_id(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    plot_col = next((c for c in gdf.columns if c.lower() == "plot_id"), None)
    if plot_col is None:
        gdf["plot_id"] = np.arange(1, len(gdf) + 1, dtype=np.int64)
    else:
        gdf["plot_id"] = gdf[plot_col]
    gdf["plot_id"] = gdf["plot_id"].apply(lambda v: str(v))
    return gdf


def _safe_read_vector(path: Path) -> gpd.GeoDataFrame:
    try:
        return gpd.read_file(path)
    except Exception:
        if path.suffix.lower() in {".geojson", ".json"}:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                feats = data.get("features", []) if isinstance(data, dict) else []
                return gpd.GeoDataFrame.from_features(feats)
            except Exception:
                pass
        raise


def _areas_by_feature(
    gdf_main: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame | None, project_crs: object | None
) -> np.ndarray:
    if len(gdf_main) == 0:
        return np.zeros(0, dtype=np.float64)
    proj_crs = _choose_projected_crs(getattr(gdf_main, "crs", None), getattr(aoi, "crs", None))
    if project_crs is not None and hasattr(project_crs, "is_projected") and project_crs.is_projected:
        proj_crs = project_crs

    main = gdf_main
    try:
        main = main.to_crs(proj_crs)
    except Exception:
        try:
            main = main.set_crs(proj_crs)
        except Exception:
            pass

    aoi_lookup = {}
    if aoi is not None and len(aoi) > 0 and "plot_id" in aoi.columns:
        try:
            aoi_proj = aoi
            try:
                aoi_proj = aoi_proj.to_crs(proj_crs)
            except Exception:
                if aoi_proj.crs is None:
                    aoi_proj = aoi_proj.set_crs(proj_crs)
            for pid, geom in zip(aoi_proj["plot_id"], aoi_proj.geometry):
                if geom is not None and not geom.is_empty:
                    aoi_lookup[str(pid)] = geom
        except Exception:
            aoi_lookup = {}
    if not aoi_lookup:
        proj_crs = proj_crs

    areas = np.zeros(len(main), dtype=np.float64)
    for idx, (geom, pid) in enumerate(zip(main.geometry, main["plot_id"])):
        area_val = 0.0
        if geom is None or geom.is_empty:
            areas[idx] = 0.0
            continue
        target = aoi_lookup.get(str(pid))
        try:
            if target is not None:
                area_val = float(geom.intersection(target).area)
            else:
                area_val = float(geom.area)
        except Exception:
            try:
                area_val = float(geom.area)
            except Exception:
                area_val = 0.0
        areas[idx] = max(0.0, area_val)
    return areas


def _class_column(gdf: gpd.GeoDataFrame) -> str | None:
    for name in ("majority", "majority_", "majclass", "rf_class", "class"):
        col = next((c for c in gdf.columns if c.lower() == name), None)
        if col:
            return col
    return None


def _biomass_from_area(area_m2: float, model: str = "madagascar", biomass_formula: str | None = None) -> float:
    area_m2 = max(0.0, float(area_m2))
    return float(np.asarray(biomass_from_area_m2(area_m2, model=model, custom_formula=biomass_formula)).item())


def _build_features(
    gdf: gpd.GeoDataFrame,
    areas_m2: Iterable[float],
    mode: str,
    biomass_model: str,
    biomass_formula: str | None,
) -> tuple[list[dict], dict[str, str]]:
    features: list[dict] = []
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
    class_colors: dict[str, str] = {}

    cls_col = _class_column(gdf) if mode == "rf" else None
    biomass_col = next((c for c in gdf.columns if c.lower() == "biomass_g"), None)
    class_values = gdf[cls_col].tolist() if cls_col else [None] * len(gdf)
    biomass_vals = gdf[biomass_col].astype(float).tolist() if biomass_col else [None] * len(gdf)
    classes_seen: list[str] = []

    for geom, pid, score, bright, area_val, cls_val, biomass_val in zip(
        gdf.geometry, gdf["plot_id"], gdf["score"], gdf["brightness"], areas_m2, class_values, biomass_vals
    ):
        if geom is None or geom.is_empty:
            continue
        area_clean = float(area_val) if np.isfinite(area_val) else 0.0
        if biomass_val is not None and np.isfinite(biomass_val):
            biomass_clean = float(biomass_val)
        else:
            biomass_clean = _biomass_from_area(area_clean, biomass_model, biomass_formula)
        props = {
            "plot_id": str(pid),
            "score": float(np.clip(score if np.isfinite(score) else 0.0, 0.0, 1.0)),
            "brightness": float(np.clip(bright if np.isfinite(bright) else 0.0, 0.0, 1.0)),
            "area_m2": area_clean,
            "biomass_g": biomass_clean,
        }
        if cls_col:
            cls_text = "Unknown" if cls_val is None else str(cls_val).strip() or "Unknown"
            props["rf_class"] = cls_text
            classes_seen.append(cls_text)
        features.append({"geometry": geom.__geo_interface__, "properties": props})

    if mode == "rf" and cls_col:
        ordered = list(dict.fromkeys(classes_seen))
        if not ordered:
            ordered = ["Unknown"]
        class_colors = {cls: palette[idx % len(palette)] for idx, cls in enumerate(ordered)}
        for feat in features:
            cls = feat["properties"].get("rf_class") or "Unknown"
            feat["properties"]["rf_class"] = cls

    return features, class_colors


def build_classification_map(
    shp_path: str | Path,
    raster_path: str | Path,
    mode: str = "stats",
    aoi_path: str | Path | None = None,
    biomass_model: str = "madagascar",
    biomass_formula: str | None = None,
    growth_rate_pct: float = 5.8,
    growth_rate_sd: float = 0.7,
) -> Path:
    shp_path = Path(shp_path)
    raster_path = Path(raster_path)
    aoi_path = Path(aoi_path) if aoi_path else None
    out_path = shp_path.with_name(f"{shp_path.stem}_map.html")

    try:
        gdf = _safe_read_vector(shp_path)
        gdf = gdf.loc[~gdf.geometry.is_empty].copy()
        if gdf.empty:
            raise RuntimeError("Classification output contains no geometries.")

        raster_crs = None
        try:
            with rasterio.open(raster_path) as _src:
                raster_crs = _src.crs
        except Exception:
            raster_crs = None

        try:
            if gdf.crs is None and raster_crs:
                gdf = gdf.set_crs(raster_crs)
        except Exception:
            pass
        try:
            if gdf.crs and str(gdf.crs) != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
        except Exception:
            pass

        gdf = _ensure_plot_id(gdf)
        gdf = _ensure_brightness_score(gdf, raster_path)

        aoi_gdf = None
        if aoi_path and aoi_path.exists():
            try:
                aoi_gdf = _safe_read_vector(aoi_path)
                aoi_gdf = aoi_gdf.loc[~aoi_gdf.geometry.is_empty].copy()
                if not aoi_gdf.empty:
                    aoi_gdf = _ensure_plot_id(aoi_gdf)
                    if aoi_gdf.crs is None and raster_crs is not None:
                        try:
                            aoi_gdf = aoi_gdf.set_crs(raster_crs)
                        except Exception:
                            pass
            except Exception:
                aoi_gdf = None

        areas_m2 = _areas_by_feature(gdf, aoi_gdf, raster_crs)

        bounds_json, b64_png = _load_raster_overlay(raster_path)
        features, class_colors = _build_features(gdf, areas_m2, mode.lower(), biomass_model, biomass_formula)

        aoi_features: list[dict] = []
        if aoi_gdf is not None and not aoi_gdf.empty:
            try:
                aoi_display = aoi_gdf
                try:
                    if aoi_display.crs and str(aoi_display.crs) != "EPSG:4326":
                        aoi_display = aoi_display.to_crs("EPSG:4326")
                except Exception:
                    pass
                for geom, pid in zip(aoi_display.geometry, aoi_display["plot_id"]):
                    if geom is None or geom.is_empty:
                        continue
                    aoi_features.append({"geometry": geom.__geo_interface__, "properties": {"plot_id": str(pid)}})
            except Exception:
                aoi_features = []

        score_vals = [f["properties"]["score"] for f in features]
        bright_vals = [f["properties"]["brightness"] for f in features]
        score_min = float(min(score_vals)) if score_vals else 0.0
        score_max = float(max(score_vals)) if score_vals else 1.0
        bright_min = float(min(bright_vals)) if bright_vals else 0.0
        bright_max = float(max(bright_vals)) if bright_vals else 1.0

        rf_controls = ""
        if mode.lower() == "rf" and class_colors:
            checkboxes = []
            for cls in class_colors:
                checkboxes.append(
                    f'<label><input type="checkbox" class="class-filter" data-class="{cls}" checked /> {cls}</label>'
                )
            rf_controls = "\n        ".join(checkboxes)

        show_class_controls = bool(class_colors)
        show_score_controls = not show_class_controls

        template = """<!DOCTYPE html>
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
      min-width: 260px;
      z-index: 1000;
    }
    .controls label { display: block; margin-bottom: 6px; font-weight: 600; }
    .controls select { width: 100%; padding: 4px; }
    .stat { font-size: 0.85rem; color: #444; margin-top: 6px; }
    .class-list label { font-weight: 400; }
    .chart-panel {
      position: absolute;
      bottom: 12px;
      left: 12px;
      background: rgba(255,255,255,0.96);
      border-radius: 10px;
      padding: 12px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.25);
      font-family: "Segoe UI", sans-serif;
      width: 560px;
      z-index: 1100;
      display: none;
    }
    .chart-panel header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }
    .chart-panel h3 { margin: 0; font-size: 1rem; }
    .chart-panel button {
      border: none;
      background: #eee;
      padding: 4px 8px;
      border-radius: 6px;
      cursor: pointer;
    }
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="controls">
    <label><input type="checkbox" id="toggleSegments" /> Show segments</label>
    __SCORE_FILTERS__
    __COLOR_SECTION__
    __CLASS_CHECKBOXES__
    <div class="stat" id="countStat"></div>
  </div>
  <div class="chart-panel" id="chartPanel">
    <header>
      <h3 id="chartTitle">Biomass projection</h3>
      <button id="closeChart">Close</button>
    </header>
    <canvas id="bioChart" width="520" height="340"></canvas>
  </div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <script>
    const imageBounds = __BOUNDS__;
    const imageData = "data:image/png;base64,__IMAGE__";
    const features = __FEATURES__;
    const aoiFeatures = __Aoi__;
    const classColors = __CLASS_COLORS__;
    const scoreMin = __SCORE_MIN__;
    const scoreMax = __SCORE_MAX__;
    const brightMin = __BRIGHT_MIN__;
    const brightMax = __BRIGHT_MAX__;
    const growthRatePct = __GROWTH_RATE__;
    const growthRateSd = __GROWTH_SD__;
    const chartPanel = document.getElementById("chartPanel");
    const chartTitle = document.getElementById("chartTitle");
    const closeChart = document.getElementById("closeChart");
    const chartCanvas = document.getElementById("bioChart");
    let chartInstance = null;
    let currentChartPid = null;

    const map = L.map('map', { maxZoom: 22 });
    L.imageOverlay(imageData, imageBounds).addTo(map);
    map.fitBounds(imageBounds);

    const segPane = map.createPane("segments-pane");
    segPane.style.zIndex = "450";
    const aoiPane = map.createPane("aoi-pane");
    aoiPane.style.zIndex = "650";
    aoiPane.style.pointerEvents = "auto";

    const segmentLayer = L.layerGroup().addTo(map);
    const aoiLayer = L.layerGroup().addTo(map);

    const toggleSegments = document.getElementById("toggleSegments");
    const colorMode = document.getElementById("colorMode");
    const scoreSlider = document.getElementById("scoreSlider");
    const brightMinSlider = document.getElementById("brightMin");
    const brightMaxSlider = document.getElementById("brightMax");
    const scoreLabel = document.getElementById("scoreVal");
    const brightMinLabel = document.getElementById("brightMinVal");
    const brightMaxLabel = document.getElementById("brightMaxVal");
    const classFilters = Array.from(document.querySelectorAll(".class-filter"));
    const countStat = document.getElementById("countStat");

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

    function colorFor(props) {
      const mode = colorMode ? colorMode.value : (props.rf_class ? "class" : "none");
      if (mode === "brightness") return ramp(props.brightness, ["#0d3b66", "#f4d35e", "#ee964b"]);
      if (mode === "score") return ramp(1 - props.score, ["#022873", "#37b24d", "#fff3bf"]);
      if (mode === "class" && props.rf_class) return classColors[props.rf_class] || "#ffcc00";
      return "rgba(255,204,0,0.6)";
    }

    function activeClasses() {
      const allowed = new Set();
      classFilters.forEach(cb => { if (cb.checked) allowed.add(cb.dataset.class); });
      return allowed;
    }

    function computeAggregates(items) {
      const totals = {};
      items.forEach(f => {
        const props = f.properties || {};
        const pid = props.plot_id || "plot";
        const a = Number(props.area_m2 || 0);
        const bio = Number(props.biomass_g || 0);
        if (!totals[pid]) totals[pid] = { area: 0, biomass: 0, biomass_sd: 0 };
        if (isFinite(a) && a > 0) totals[pid].area += a;
        if (isFinite(bio) && bio > 0) {
          totals[pid].biomass += bio;
          const sdField = Number(props.biomass_sd || 0);
          if (isFinite(sdField) && sdField > 0) {
            totals[pid].biomass_sd += sdField * sdField; // sum variances
          }
        }
      });
      Object.values(totals).forEach(t => {
        t.biomass_sd = Math.sqrt(t.biomass_sd);
      });
      return totals;
    }
    function projectBiomass(baseGram, baseSd) {
      const rate = Math.max(0, Number(growthRatePct) || 0) / 100.0;
      const sdRate = Math.max(0, Number(growthRateSd) || 0) / 100.0;
      const mean = [];
      const sd = [];
      const base = Math.max(0, Number(baseGram) || 0);
      const baseSdVal = Math.max(0, Number(baseSd) || 0);
      mean.push(base);
      sd.push(baseSdVal > 0 ? baseSdVal : base * sdRate);
      for (let day = 1; day <= 7; day++) {
        const prevMean = mean[day - 1];
        const prevSd = sd[day - 1];
        const nextMean = prevMean + (prevMean * rate);
        const nextSd = Math.sqrt(Math.pow(prevSd * (1 + rate), 2) + Math.pow(prevMean * sdRate, 2));
        mean.push(nextMean);
        sd.push(nextSd);
      }
      return { mean, sd };
    }

    let currentPlotStats = computeAggregates(features);
    function refreshLabels() {
      if (scoreLabel) scoreLabel.textContent = Number(scoreSlider.value).toFixed(2);
      if (brightMinLabel) brightMinLabel.textContent = Number(brightMinSlider.value).toFixed(2);
      if (brightMaxLabel) brightMaxLabel.textContent = Number(brightMaxSlider.value).toFixed(2);
    }

    function syncDefaults() {
      if (scoreSlider) scoreSlider.value = scoreMin;
      if (brightMinSlider) brightMinSlider.value = brightMin;
      if (brightMaxSlider) brightMaxSlider.value = brightMax;
      toggleSegments.checked = false;
      refreshLabels();
    }
    syncDefaults();

    function renderSegments() {
      segmentLayer.clearLayers();
      if (!toggleSegments.checked) {
        currentPlotStats = computeAggregates(features);
        countStat.textContent = "Segments hidden";
        aoiLayer.bringToFront();
        return;
      }
      let itemsFiltered = features.slice();
      if (scoreSlider && brightMinSlider && brightMaxSlider) {
        const scoreCut = Number(scoreSlider.value);
        const bMin = Number(brightMinSlider.value);
        let bMax = Number(brightMaxSlider.value);
        if (bMin > bMax) {
          bMax = bMin;
          brightMaxSlider.value = bMax;
        }
        itemsFiltered = [];
        features.forEach(f => {
          const p = f.properties || {};
          if (p.score < scoreCut) return;
          if (p.brightness < bMin) return;
          if (p.brightness > bMax) return;
          itemsFiltered.push(f);
        });
      }
      let items = itemsFiltered;
      if (classFilters.length > 0) {
        const allowed = activeClasses();
        items = items.filter(f => allowed.has((f.properties && f.properties.rf_class) || "Unknown"));
      }
      currentPlotStats = computeAggregates(items);
      refreshChartIfOpen();
      items.forEach(f => {
        const props = f.properties;
        L.geoJSON(f.geometry, {
          pane: "segments-pane",
          style: () => {
            const color = colorFor(props);
            return { color, fillColor: color, fillOpacity: 0.55, weight: 0 };
          },
        }).addTo(segmentLayer);
      });
      countStat.textContent = `${items.length} segment(s)`;
      aoiLayer.bringToFront();
    }

    function buildAoiLayer() {
      aoiLayer.clearLayers();
      if (!aoiFeatures || !aoiFeatures.length) return;
      aoiFeatures.forEach(f => {
        const pid = (f.properties && f.properties.plot_id) || "plot";
        L.geoJSON(f.geometry, {
          pane: "aoi-pane",
          interactive: true,
          bubblingMouseEvents: false,
          style: () => ({
            color: "#ff6600",
            weight: 2.5,
            opacity: 0.35,
            fillColor: "#ff6600",
            fillOpacity: 0.0,
          }),
          onEachFeature: (_feature, lyr) => {
            lyr.on("click", () => {
              showChartForPlot(pid);
            });
          },
        }).addTo(aoiLayer);
      });
      aoiLayer.bringToFront();
      refreshChartIfOpen();
    }

    toggleSegments.addEventListener("change", renderSegments);
    if (colorMode) colorMode.addEventListener("change", renderSegments);
    classFilters.forEach(cb => cb.addEventListener("change", renderSegments));
    if (scoreSlider) scoreSlider.addEventListener("input", () => { refreshLabels(); renderSegments(); });
    if (brightMinSlider) brightMinSlider.addEventListener("input", () => { refreshLabels(); renderSegments(); });
    if (brightMaxSlider) brightMaxSlider.addEventListener("input", () => { refreshLabels(); renderSegments(); });
    closeChart.addEventListener("click", () => {
      chartPanel.style.display = "none";
    });

    function ensureChart(data) {
      const labels = Array.from({length: data.mean.length}, (_, idx) => `Day ${idx}`);
      const targetLine = 2.5; // tons
      const eps = 1e-6;
      const meanTons = data.mean.map(v => Math.max(eps, v / 1_000_000.0));
      const lower = [];
      const upper = [];
      data.sd.forEach((sdVal, i) => {
        const baseTons = meanTons[i];
        const rel = Math.max(0, sdVal / Math.max(eps, data.mean[i]));
        const factor = rel * i; // cumulative: day index multiplies sd
        lower.push(Math.max(eps, baseTons * (1 - factor)));
        upper.push(Math.max(eps, baseTons * (1 + factor)));
      });
      if (chartInstance) {
        chartInstance.data.labels = labels;
        chartInstance.data.datasets[0].data = meanTons;
        chartInstance.options.plugins.errorBars.lower = lower;
        chartInstance.options.plugins.errorBars.upper = upper;
        chartInstance.options.plugins.targetLine.y = Math.max(eps, targetLine);
        const yMax = Math.max(...upper, targetLine);
        chartInstance.options.scales.y.max = yMax * 1.05;
        chartInstance.update();
        return;
      }
      const errorBarPlugin = {
        id: "errorBars",
        lower,
        upper,
        afterDatasetsDraw(chart, args, opts) {
          const {ctx, scales: {x, y}} = chart;
          ctx.save();
          ctx.strokeStyle = "#555";
          ctx.lineWidth = 1.5;
          meanTons.forEach((val, i) => {
            const xPos = x.getPixelForValue(i);
            const yVal = y.getPixelForValue(opts.upper?.[i] || val);
            const yMin = y.getPixelForValue(opts.lower?.[i] || val);
            ctx.beginPath();
            ctx.moveTo(xPos, yVal);
            ctx.lineTo(xPos, yMin);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(xPos - 6, yVal);
            ctx.lineTo(xPos + 6, yVal);
            ctx.moveTo(xPos - 6, yMin);
            ctx.lineTo(xPos + 6, yMin);
            ctx.stroke();
          });
          ctx.restore();
        }
      };
      const targetLinePlugin = {
        id: "targetLine",
        y: targetLine,
        afterDraw(chart, args, opts) {
          const {ctx, chartArea: {left, right}, scales: {y}} = chart;
          const yPos = y.getPixelForValue(opts.y);
          ctx.save();
          ctx.strokeStyle = "red";
          ctx.setLineDash([6, 6]);
          ctx.beginPath();
          ctx.moveTo(left, yPos);
          ctx.lineTo(right, yPos);
          ctx.stroke();
          ctx.restore();
        }
      };
      chartInstance = new Chart(chartCanvas.getContext("2d"), {
        type: "bar",
        data: {
          labels,
          datasets: [
            {
              label: "Biomass (tons)",
              data: meanTons,
              backgroundColor: "#88c0d0",
            },
          ],
        },
        options: {
          responsive: false,
          scales: {
            y: {
              beginAtZero: true,
              min: 0,
              title: {display: true, text: "Biomass (tons)"},
              max: Math.max(...upper, targetLine) * 1.05,
            },
          },
          plugins: {
            legend: {display: false},
            errorBars: {lower, upper},
            targetLine: {y: Math.max(eps, targetLine)},
          },
        },
        plugins: [errorBarPlugin, targetLinePlugin],
      });
    }

    function showChartForPlot(pid) {
      const stats = currentPlotStats[pid];
      if (!stats || !isFinite(stats.biomass) || stats.biomass <= 0) {
        chartTitle.textContent = `Plot ${pid}: no biomass available`;
        chartPanel.style.display = "block";
        if (chartInstance) chartInstance.destroy();
        chartInstance = null;
        currentChartPid = pid;
        return;
      }
      const proj = projectBiomass(stats.biomass, stats.biomass_sd);
      chartTitle.textContent = `Plot ${pid}: biomass projection`;
      ensureChart(proj);
      chartPanel.style.display = "block";
      currentChartPid = pid;
    }

    function refreshChartIfOpen() {
      if (chartPanel.style.display === "block" && currentChartPid) {
        showChartForPlot(currentChartPid);
      }
    }

    buildAoiLayer();
    renderSegments();
  </script>
</body>
</html>"""

        html_body = (
            template.replace("__BOUNDS__", bounds_json)
            .replace("__IMAGE__", b64_png)
            .replace("__FEATURES__", json.dumps(features))
            .replace("__Aoi__", json.dumps(aoi_features))
            .replace("__CLASS_COLORS__", json.dumps(class_colors))
            .replace("__GROWTH_RATE__", f"{float(growth_rate_pct):.6f}")
            .replace("__GROWTH_SD__", f"{float(growth_rate_sd):.6f}")
            .replace(
                "__CLASS_CHECKBOXES__",
                ("<div class='class-list'>" + rf_controls + "</div>") if show_class_controls else "",
            )
            .replace(
                "__SCORE_FILTERS__",
                (
                    """
    <label>Score >= <span id="scoreVal"></span></label>
    <input type="range" id="scoreSlider" min="__SCORE_MIN__" max="__SCORE_MAX__" step="0.01" />
    <label>Brightness <span id="brightMinVal"></span> - <span id="brightMaxVal"></span></label>
    <input type="range" id="brightMin" min="__BRIGHT_MIN__" max="__BRIGHT_MAX__" step="0.01" />
    <input type="range" id="brightMax" min="__BRIGHT_MIN__" max="__BRIGHT_MAX__" step="0.01" />
    """
                )
                if show_score_controls
                else "",
            )
            .replace(
                "__COLOR_SECTION__",
                """
    <label>Color by:</label>
    <select id="colorMode">
      <option value="none">None</option>
      <option value="brightness">Brightness</option>
      <option value="score">Score</option>
    </select>
    """
                if show_score_controls
                else "",
            )
            .replace("__SCORE_MIN__", f"{score_min:.4f}")
            .replace("__SCORE_MAX__", f"{score_max:.4f}")
            .replace("__BRIGHT_MIN__", f"{bright_min:.4f}")
            .replace("__BRIGHT_MAX__", f"{bright_max:.4f}")
        )

        out_path.write_text(html_body, encoding="utf-8")
        return out_path
    except Exception as exc:  # write minimal error page
        msg = f"Failed to render preview: {exc}"
        fallback = f"<html><body><pre>{msg}</pre></body></html>"
        out_path.write_text(fallback, encoding="utf-8")
        return out_path
