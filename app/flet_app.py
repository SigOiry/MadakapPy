from __future__ import annotations

import os
import threading
import json
import shutil
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, Any


def run_flet_app() -> None:
    try:
        import flet as ft
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Flet is not installed. Install with 'pip install flet'.") from e

    # Local imports (avoid heavy deps until UI launches)
    from .predetection import detect_cultivation_plots, save_preselection_to_output
    from .segmentation import run_segmentation
    from .classification import (
        train_model_from_training_polys,
        apply_model_with_pixel_sampling,
    )
    from .simple_classifier import classify_dark_linear_polygons
    from .map_preview import build_classification_map
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import rasterio
    from rasterio.warp import transform_bounds
    from PIL import Image
    import io, base64

    def _palette():
        return {
            "primary": "#7BC8C6",
            "accent":  "#A7C7E7",
            "warn":    "#F7B7A3",
            "surface": "#FAFAF7",
            "card":    "#FFFFFF",
            "alt":     "#F4F7FB",
            "border":  "#E6E6E0",
            "muted":   "#6E7A8A",
        }

    def _resolve_window_icon() -> str | None:
        """Return an absolute path to the best icon for the current platform."""
        project_root = Path(__file__).resolve().parent.parent
        icon_dir = project_root / "icon"
        png_path = icon_dir / "madakappy.png"
        ico_path = icon_dir / "madakappy.ico"

        target: Path | None = None
        # Windows expects a .ico to update Explorer/taskbar thumbnails.
        if os.name == "nt":
            png_exists = png_path.exists()
            ico_exists = ico_path.exists()
            needs_refresh = False
            if png_exists and ico_exists:
                try:
                    needs_refresh = png_path.stat().st_mtime > ico_path.stat().st_mtime
                except OSError:
                    needs_refresh = True
            elif png_exists:
                needs_refresh = True

            if png_exists and needs_refresh:
                try:
                    sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
                    with Image.open(png_path) as img:
                        img.convert("RGBA").save(ico_path, format="ICO", sizes=sizes)
                    ico_exists = True
                except Exception:
                    target = png_path

            if ico_exists:
                target = ico_path
            elif png_exists and target is None:
                target = png_path
        else:
            if png_path.exists():
                target = png_path
            elif ico_path.exists():
                target = ico_path

        if target is not None:
            try:
                return str(target.resolve())
            except OSError:
                return str(target)
        return None

    if not hasattr(ft, "ExpansionTile"):
        class _CompatExpansionTile(ft.UserControl):
            def __init__(
                self,
                *,
                title: ft.Control,
                controls: Optional[list[ft.Control]] = None,
                subtitle: Optional[ft.Control] = None,
                initially_expanded: bool = False,
            ) -> None:
                super().__init__()
                self._title = title
                self._subtitle = subtitle
                self._controls = list(controls or [])
                self._expanded = initially_expanded
                self._chevron: ft.Icon | None = None
                self._body: ft.Column | None = None

            def build(self) -> ft.Control:
                pal = _palette()
                self._chevron = ft.Icon(
                    name=ft.icons.KEYBOARD_ARROW_DOWN if self._expanded else ft.icons.KEYBOARD_ARROW_RIGHT,
                    color=pal["muted"],
                )
                header_controls = [self._title]
                if self._subtitle is not None:
                    header_controls.append(self._subtitle)
                header_column = ft.Column(
                    controls=header_controls,
                    spacing=2,
                    expand=True,
                )
                header_row = ft.Row(
                    controls=[
                        ft.Container(self._chevron, width=28),
                        header_column,
                    ],
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=8,
                    expand=True,
                )
                header = ft.Container(
                    content=header_row,
                    on_click=self._toggle,
                )
                self._body = ft.Column(
                    controls=list(self._controls),
                    spacing=8,
                    visible=self._expanded,
                )
                return ft.Column(
                    controls=[
                        header,
                        self._body,
                    ],
                    spacing=6,
                    expand=True,
                )

            def _toggle(self, _=None) -> None:
                self._expanded = not self._expanded
                if self._body is not None:
                    self._body.visible = self._expanded
                if self._chevron is not None:
                    self._chevron.name = (
                        ft.icons.KEYBOARD_ARROW_DOWN if self._expanded else ft.icons.KEYBOARD_ARROW_RIGHT
                    )
                self.update()

        ft.ExpansionTile = _CompatExpansionTile

    def section(
        title: str,
        *children: ft.Control,
        icon: Optional[str] = None,
        subtitle: Optional[str] = None,
        bgcolor: Optional[str] = None,
        expanded: bool = False,
    ) -> ft.Card:
        pal = _palette()
        tile = ft.ExpansionTile(
            title=ft.Row([
                ft.Icon(icon, color=pal["primary"]) if icon else ft.Container(),
                ft.Text(title, size=16, weight=ft.FontWeight.W_600),
            ], spacing=8),
            subtitle=ft.Text(subtitle or "", size=12, color=pal["muted"]) if subtitle else None,
            initially_expanded=expanded,
            controls=list(children),
        )
        return ft.Card(
            content=ft.Container(tile, padding=12, bgcolor=(bgcolor or pal["card"]), border_radius=12, border=ft.border.all(1, pal["border"]))
        )

    def labeled_row(label: str, control: ft.Control, icon: Optional[str] = None, tip: Optional[str] = None) -> ft.Row:
        pal = _palette()
        lbl_core = ft.Row([
            ft.Icon(icon, size=16, color=pal["muted"]) if icon else ft.Container(width=0),
            ft.Text(label, tooltip=(tip or None)),
        ], spacing=6)
        return ft.Row([ft.Container(lbl_core, width=230), ft.Container(control, expand=True)])

    _EDIT_SERVERS: dict[str, tuple[ThreadingHTTPServer, threading.Thread, int]] = {}

    def _purge_shapefile(base_path: Path) -> None:
        suffixes = [".shp", ".shx", ".dbf", ".prj", ".cpg", ".shp.xml"]
        stem = base_path.with_suffix("")
        for suf in suffixes:
            try:
                target = stem.with_suffix(suf)
                if target.exists():
                    target.unlink()
            except Exception:
                pass

    def _cleanup_root_tiffs() -> None:
        try:
            root = Path.cwd()
            for tif in root.glob("*.tif"):
                try:
                    tif.unlink()
                except Exception:
                    pass
        except Exception:
            pass

    def _normalized_path(path_str: str | None) -> Optional[Path]:
        txt = (path_str or "").strip()
        if not txt:
            return None
        try:
            path = Path(txt).expanduser()
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            else:
                path = path.resolve()
        except Exception:
            try:
                path = Path(txt).expanduser().absolute()
            except Exception:
                return None
        return path

    def _history_root_for_output(path_str: str | None) -> Optional[Path]:
        base = _normalized_path(path_str)
        if base is None:
            return None
        return base if base.name.lower() == "output" else base / "Output"

    def _history_file_for_output(path_str: str | None) -> Optional[Path]:
        root = _history_root_for_output(path_str)
        if root is None:
            return None
        return root / "run_history.json"

    def _load_run_history(path_str: str | None) -> list[dict[str, Any]]:
        fp = _history_file_for_output(path_str)
        if fp is None or not fp.exists():
            return []
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
        except Exception:
            pass
        return []

    def _append_run_history(path_str: str | None, record: dict[str, Any]) -> Optional[dict[str, Any]]:
        root = _history_root_for_output(path_str)
        if root is None:
            return None
        entry = dict(record)
        entry["output_root"] = str(root)
        fp = root / "run_history.json"
        try:
            root.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            existing = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else []
        except Exception:
            existing = []
        if not isinstance(existing, list):
            existing = []
        existing.append(entry)
        if len(existing) > 200:
            existing = existing[-200:]
        try:
            fp.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        except Exception:
            pass
        return entry

    def _update_history_record(history_root: str | None, run_id: str, updates: dict[str, Any]) -> None:
        if not history_root or not run_id or not updates:
            return
        fp = Path(history_root) / "run_history.json"
        if not fp.exists():
            return
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, list):
            return
        updated = False
        for item in data:
            if not isinstance(item, dict):
                continue
            if item.get("id") == run_id:
                item.update(updates)
                updated = True
                break
        if updated:
            try:
                fp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except Exception:
                pass

    def _ensure_edit_server(shp_path: Path, target_crs: str | None) -> int:
        key = str(shp_path.resolve())
        if key in _EDIT_SERVERS:
            return _EDIT_SERVERS[key][2]

        class _EditHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args) -> None:  # noqa: A003
                return

            def _set_headers(self) -> None:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")

            def do_OPTIONS(self) -> None:  # noqa: N802
                self.send_response(204)
                self._set_headers()
                self.end_headers()

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    payload = json.loads(body.decode("utf-8"))
                    feats = payload.get("features", [])
                    tgt = target_crs or "EPSG:4326"
                    if feats:
                        gdf_new = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
                        if tgt and str(gdf_new.crs) != str(tgt):
                            gdf_new = gdf_new.to_crs(tgt)
                        _purge_shapefile(shp_path)
                        gdf_new.to_file(shp_path)
                        resp = {"status": "ok", "saved": len(gdf_new)}
                    else:
                        _purge_shapefile(shp_path)
                        resp = {"status": "ok", "saved": 0}
                    self.send_response(200)
                except Exception as exc:  # noqa: BLE001
                    resp = {"status": "error", "message": str(exc)}
                    self.send_response(500)
                self._set_headers()
                self.end_headers()
                self.wfile.write(json.dumps(resp).encode("utf-8"))

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), _EditHandler)
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        _EDIT_SERVERS[key] = (httpd, thread, port)
        return port


    def build_preview_html(aoi_path: str, raster_path: str) -> Optional[str]:
        try:
            gdf = gpd.read_file(aoi_path)
            if gdf is None or len(gdf) == 0:
                return None
            with rasterio.open(raster_path) as src:
                west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
                max_w = 1600
                scale = min(1.0, max_w / max(1, src.width))
                out_h = max(1, int(src.height * scale))
                out_w = max(1, int(src.width * scale))
                if src.count >= 3:
                    bands = src.read(indexes=(1, 2, 3), out_shape=(3, out_h, out_w)).astype("float32")
                    for i in range(3):
                        b = bands[i]
                        vmin, vmax = float(np.nanpercentile(b, 2)), float(np.nanpercentile(b, 98))
                        if vmax <= vmin:
                            vmax = vmin + 1.0
                        bands[i] = np.clip((b - vmin) / (vmax - vmin), 0.0, 1.0)
                    img = np.transpose((bands * 255.0).astype("uint8"), (1, 2, 0))
                    pil = Image.fromarray(img, mode="RGB")
                else:
                    gray = src.read(1, out_shape=(out_h, out_w)).astype("float32")
                    vmin, vmax = float(np.nanpercentile(gray, 2)), float(np.nanpercentile(gray, 98))
                    if vmax <= vmin:
                        vmax = vmin + 1.0
                    norm = np.clip((gray - vmin) / (vmax - vmin), 0.0, 1.0)
                    pil = Image.fromarray((norm * 255.0).astype("uint8"), mode="L")
                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                b64_png = base64.b64encode(buf.getvalue()).decode("ascii")

            try:
                gdf_wgs = gdf.to_crs("EPSG:4326") if gdf.crs and str(gdf.crs) != "EPSG:4326" else gdf
            except Exception:
                gdf_wgs = gdf

            geojson = json.dumps(gdf_wgs.__geo_interface__)
            bounds_json = json.dumps([[south, west], [north, east]])
            port = _ensure_edit_server(Path(aoi_path), str(gdf.crs) if gdf.crs else "EPSG:4326")

            template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>AOI Editor</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <link rel="stylesheet" href="https://unpkg.com/@geoman-io/leaflet-geoman-free@2.13.0/dist/leaflet-geoman.css" />
  <style>
    html, body { width: 100%; height: 100%; margin: 0; padding: 0; }
    body { display: flex; flex-direction: column; font-family: sans-serif; }
    #map { flex: 1; width: 100%; }
    .toolbar {
      position: sticky;
      top: 0;
      display: flex;
      gap: 8px;
      align-items: center;
      justify-content: flex-end;
      padding: 10px;
      background: rgba(255,255,255,0.95);
      box-shadow: 0 2px 6px rgba(0,0,0,0.2);
      z-index: 1000;
    }
    #saveBtn { padding: 6px 12px; border: none; border-radius: 6px; background: #0b8457; color: #fff; cursor: pointer; font-weight: 600; }
    #saveBtn:hover { background: #0a704a; }
    .status { padding: 6px 10px; border-radius: 6px; color: #fff; background: rgba(0,0,0,0.6); }
  </style>
</head>
<body>
  <div class="toolbar">
    <button id="saveBtn" title="Write current polygons to the shapefile.">Save edits</button>
    <div class="status" id="status">Ready</div>
  </div>
  <div id="map"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/@geoman-io/leaflet-geoman-free@2.13.0/dist/leaflet-geoman.min.js"></script>
  <script>
    const imageBounds = __IMAGE_BOUNDS__;
    const imageData = "data:image/png;base64,__IMAGE_DATA__";
    const initialGeoJson = __GEOJSON__;
    const SAVE_URL = "__SAVE_URL__";

    const map = L.map('map', { maxZoom: 22 });
    L.imageOverlay(imageData, imageBounds).addTo(map);
    const drawnItems = L.featureGroup().addTo(map);

    L.geoJSON(initialGeoJson, {
      style: () => ({ color: "yellow", weight: 1, fillColor: "yellow", fillOpacity: 0.25 })
    }).eachLayer(layer => {
      drawnItems.addLayer(layer);
    });

    const statusEl = document.getElementById("status");
    const saveBtn = document.getElementById("saveBtn");
    let dirty = false;

    function setStatus(text, ok=true) {
      statusEl.textContent = text;
      statusEl.style.background = ok ? "rgba(0, 123, 67, 0.8)" : "rgba(176, 0, 32, 0.8)";
    }

    function markDirty() {
      dirty = true;
      setStatus("Edits pending - click Save", false);
    }

    const bounds = drawnItems.getBounds();
    if (bounds.isValid()) {
      map.fitBounds(bounds);
    } else {
      map.fitBounds(imageBounds);
    }

    map.pm.addControls({
      position: "topleft",
      drawMarker: false,
      drawCircle: false,
      drawCircleMarker: false,
      drawPolyline: false,
      drawRectangle: true,
      cutPolygon: true,
      editMode: true,
      dragMode: true,
      removalMode: true
    });

    map.on("pm:create", e => {
      drawnItems.addLayer(e.layer);
      markDirty();
    });
    map.on("pm:remove", e => {
      if (e.layer && drawnItems.hasLayer(e.layer)) {
        drawnItems.removeLayer(e.layer);
      }
      markDirty();
    });
    const dirtyEvents = ["pm:update", "pm:cut", "pm:edit", "pm:dragend", "pm:rotateend"];
    dirtyEvents.forEach(evt => map.on(evt, markDirty));

    function saveGeoJSON() {
      setStatus("Saving...", true);
      const features = [];
      drawnItems.eachLayer(layer => {
        try {
          features.push(layer.toGeoJSON());
        } catch (err) {
          console.error(err);
        }
      });
      const fc = { type: "FeatureCollection", features };
      fetch(SAVE_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(fc)
      })
        .then(resp => resp.json())
        .then(data => {
          if (data.status === "ok") {
            setStatus("Saved polygons: " + data.saved, true);
            dirty = false;
          } else {
            setStatus("Save failed: " + data.message, false);
          }
        })
        .catch(err => {
          console.error(err);
          setStatus("Save failed", false);
        });
    }
    saveBtn.addEventListener("click", saveGeoJSON);
  </script>
</body>
</html>"""
            html = (
                template.replace("__IMAGE_BOUNDS__", bounds_json)
                .replace("__IMAGE_DATA__", b64_png)
                .replace("__GEOJSON__", geojson)
                .replace("__SAVE_URL__", f"http://127.0.0.1:{port}/")
            )
            out_html = Path(aoi_path).parent / "preselection_preview.html"
            out_html.write_text(html, encoding="utf-8")
            return str(out_html)
        except Exception:
            return None

    def main(page: ft.Page):
        pal = _palette()
        page.title = "Madakappy â€“ Seaweed Mapping Suite"
        page.theme_mode = ft.ThemeMode.LIGHT
        # Normalize title to plain ASCII to avoid encoding artifacts
        page.title = "Madakappy - Seaweed Mapping Suite"
        try:
            page.theme = ft.Theme(color_scheme_seed=pal["primary"])  # type: ignore
        except Exception:
            pass
        try:
            icon_path = _resolve_window_icon()
            if icon_path:
                page.window_icon_path = icon_path
        except Exception:
            pass
        page.window_maximized = True
        page.padding = 16
        page.bgcolor = pal["surface"]

        # Inputs
        default_in = str((Path("Data") / "All_cropped.tif").resolve())
        default_out = str((Path.cwd() / "Output").resolve())
        default_otb = str(Path("OTB") / "bin" / "otbcli_LargeScaleMeanShift.bat")

        in_raster = ft.TextField(value=default_in, expand=True)
        output_dir = ft.TextField(value=default_out, expand=True)
        otb_bin = ft.TextField(value=default_otb, expand=True)
        custom_aoi = ft.TextField(value="", expand=True, hint_text="Optional AOI shapefile (.shp)")
        train_image = ft.TextField(value=default_in, expand=True, hint_text="Training image (GeoTIFF)")

        # Preselection controls (size-only workflow)
        pre_wmin = ft.TextField(value="5", width=120, text_align=ft.TextAlign.RIGHT)
        pre_wmax = ft.TextField(value="20", width=120, text_align=ft.TextAlign.RIGHT, hint_text="Optional")
        pre_lmin = ft.TextField(value="20", width=120, text_align=ft.TextAlign.RIGHT)
        pre_lmax = ft.TextField(value="60", width=120, text_align=ft.TextAlign.RIGHT, hint_text="Optional")
        pre_small_buffer = ft.TextField(value="0.3", width=140, text_align=ft.TextAlign.RIGHT, hint_text="Buffer (m)")
        pre_quantile = ft.Slider(min=0.05, max=0.9, value=0.15, divisions=17, expand=True)
        pre_quantile_label = ft.Text("", color=pal["muted"])
        pre_disabled_hint = ft.Text("", size=12, color=pal["muted"])
        pre_inputs: list[ft.Control] = [pre_wmin, pre_wmax, pre_lmin, pre_lmax, pre_small_buffer, pre_quantile]
        pre_button_ref: dict[str, Optional[ft.ElevatedButton]] = {"btn": None}

        def _update_quant_label(val: float | None) -> None:
            pre_quantile_label.value = f"Blue quantile: {val:.2f}" if val is not None else "Blue quantile: --"
            try:
                if pre_quantile_label.page:
                    pre_quantile_label.update()
            except Exception:
                pass

        _update_quant_label(pre_quantile.value)

        def _on_quantile_change(e) -> None:
            try:
                v = float(e.control.value)
            except Exception:
                v = pre_quantile.value or 0.2
            _update_quant_label(v)

        pre_quantile.on_change = _on_quantile_change

        def _safe_update(ctrl: ft.Control) -> None:
            try:
                if ctrl.page:
                    ctrl.update()
            except Exception:
                pass

        def sync_preselection_state() -> None:
            disabled = bool((custom_aoi.value or "").strip())
            for ctrl in pre_inputs:
                ctrl.disabled = disabled
                _safe_update(ctrl)
            pre_disabled_hint.value = (
                "Preselection parameters are disabled while a custom AOI is set." if disabled else ""
            )
            _safe_update(pre_disabled_hint)
            btn = pre_button_ref["btn"]
            if btn is not None:
                btn.disabled = disabled
                _safe_update(btn)

        custom_aoi.on_change = lambda e: sync_preselection_state()

        # Status
        step_text = ft.Text("Ready", size=12, color=pal["muted"]) 
        progress = ft.ProgressBar(value=0.0, color=pal["primary"], bgcolor=pal["border"]) 
        result_text = ft.Text("", selectable=True, color=pal["muted"]) 
        workflow_gate: dict[str, threading.Event | None] = {"event": None}
        confirm_btn = ft.ElevatedButton(
            "Confirm polygons and continue",
            visible=False,
            disabled=True,
        )

        history_state: dict[str, Any] = {"records": [], "selected_id": None, "current_root": None}
        history_hint = ft.Text("", size=12, color=pal["muted"])
        history_list = ft.ListView(height=240, spacing=4, auto_scroll=False, controls=[])
        history_summary = ft.Text("Select a run to see details.", selectable=True, color=pal["muted"])
        history_settings = ft.Text("", selectable=True, color=pal["muted"], size=12, font_family="Consolas")

        def _current_history_root_str() -> Optional[str]:
            root = _history_root_for_output((output_dir.value or "").strip())
            return str(root) if root else None

        def _set_history_records(records: list[dict[str, Any]]) -> None:
            history_state["records"] = records
            display = list(reversed(records))
            tiles: list[ft.Control] = []
            if not display:
                tiles.append(
                    ft.ListTile(
                        title=ft.Text("No runs saved yet.", color=pal["muted"]),
                        leading=ft.Icon(ft.icons.INBOX, color=pal["muted"]),
                    )
                )
            else:
                for rec in display:
                    rid = rec.get("id")
                    title = rec.get("label") or rid or "Workflow run"
                    subtitle = rec.get("summary") or rec.get("result_path") or ""
                    subtitle = (subtitle.splitlines()[0][:120]) if subtitle else ""
                    tiles.append(
                        ft.ListTile(
                            title=ft.Text(title),
                            subtitle=ft.Text(subtitle) if subtitle else None,
                            on_click=lambda e, rid=rid: select_history_record(rid),
                            trailing=ft.Icon(ft.icons.MAP, color=pal["muted"]),
                        )
                    )
            history_list.controls = tiles
            history_list.update()

        def refresh_history_panel(_=None) -> None:
            root = _current_history_root_str()
            history_state["current_root"] = root
            history_hint.value = f"Stored under: {root}" if root else "Set a valid output directory to see saved runs."
            records = _load_run_history((output_dir.value or "").strip()) if root else []
            _set_history_records(records)
            history_summary.value = "Select a run to see details."
            history_settings.value = ""
            history_hint.update()
            history_summary.update()
            history_settings.update()

        def select_history_record(run_id: str | None) -> None:
            if not run_id:
                return
            record = next((rec for rec in history_state.get("records", []) if rec.get("id") == run_id), None)
            if not record:
                return
            history_state["selected_id"] = run_id
            metrics = record.get("metrics") or {}
            summary_lines = [
                f"Run: {record.get('label', run_id)}",
                f"Result: {record.get('result_path', 'Unknown')}",
            ]
            if metrics:
                summary_lines.append(
                    f"AOIs: {metrics.get('aois', 'n/a')} | Segments: {metrics.get('segments', 'n/a')}"
                )
            summary_lines.append(record.get("summary", ""))
            history_summary.value = "\n".join(line for line in summary_lines if line)
            settings_payload = record.get("settings") or {}
            history_settings.value = json.dumps(settings_payload, indent=2, ensure_ascii=False) if settings_payload else ""
            history_summary.update()
            history_settings.update()
            _launch_history_preview(record)

        def _launch_history_preview(record: dict[str, Any]) -> None:
            preview = record.get("preview_map")
            if preview and os.path.exists(preview):
                page.pubsub.send_all({"kind": "classification_preview", "path": preview})
                return
            shp = record.get("result_path")
            raster = (record.get("settings") or {}).get("input_raster")
            if not shp or not raster or not os.path.exists(shp) or not os.path.exists(raster):
                page.pubsub.send_all(
                    {"kind": "result", "text": "Preview unavailable because the saved files are missing."}
                )
                return

            def _worker():
                try:
                    mode_val = record.get("mode") or "rf"
                    bm = (
                        ((record.get("settings") or {}).get("classification") or {}).get("biomass_model")
                        or "madagascar"
                    )
                    bf = ((record.get("settings") or {}).get("classification") or {}).get("biomass_formula")
                    gr = float(((record.get("settings") or {}).get("classification") or {}).get("growth_rate_pct", 5.8))
                    gr_sd = float(((record.get("settings") or {}).get("classification") or {}).get("growth_rate_sd", 0.7))
                    new_preview = build_classification_map(
                        shp,
                        raster,
                        mode="stats" if mode_val == "stats" else "rf",
                        aoi_path=record.get("aoi_path"),
                        biomass_model=bm,
                        biomass_formula=bf,
                        growth_rate_pct=gr,
                        growth_rate_sd=gr_sd,
                    )
                except Exception as exc:  # noqa: BLE001
                    page.pubsub.send_all({"kind": "result", "text": f"Failed to rebuild preview: {exc}"})
                    return
                if new_preview:
                    record["preview_map"] = str(new_preview)
                    _update_history_record(record.get("output_root"), record.get("id"), {"preview_map": str(new_preview)})
                    page.pubsub.send_all({"kind": "classification_preview", "path": str(new_preview)})

            threading.Thread(target=_worker, daemon=True).start()

        def _field_float(field: ft.TextField, default: float | None = None) -> float | None:
            txt = (field.value or "").strip()
            if not txt:
                return default
            return float(txt)

        def _field_int(field: ft.TextField, default: int) -> int:
            try:
                txt = (field.value or "").strip()
                return int(txt) if txt else default
            except Exception:
                return default

        def collect_run_settings() -> dict[str, Any]:
            def _clean(field: ft.TextField) -> str:
                return (field.value or "").strip()

            pre_vals = {
                "min_width_m": _field_float(pre_wmin),
                "max_width_m": _field_float(pre_wmax),
                "min_length_m": _field_float(pre_lmin),
                "max_length_m": _field_float(pre_lmax),
                "small_polygon_buffer_m": _field_float(pre_small_buffer, 0.3),
                "blue_quantile": float(pre_quantile.value or 0.0),
                "enabled": not bool(_clean(custom_aoi)),
            }
            seg_vals = {
                "tile_size_m": _field_int(tile_size, 40),
                "spatialr": _field_int(spatialr, 5),
                "minsize": _field_int(minsize, 5),
            }
            bio_mode = (biomass_mode.value or "preset")
            bio_formula_val = (biomass_formula.value or "").strip()
            bio_model_val = (biomass_model.value or "madagascar")
            try:
                growth_rate_val = float(growth_rate.value or "5.8")
            except Exception:
                growth_rate_val = 5.8
            try:
                growth_sd_val = float(growth_sd.value or "0.7")
            except Exception:
                growth_sd_val = 0.7
            clf_vals = {
                "mode": (classifier_mode.value or "rf"),
                "model_path": _clean(model_path),
                "max_pixels_per_polygon": _field_int(max_pixels_per_polygon, 200),
                "biomass_mode": bio_mode,
                "biomass_model": "custom" if bio_mode == "custom" else bio_model_val,
                "biomass_formula": bio_formula_val if bio_mode == "custom" else None,
                "growth_rate_pct": growth_rate_val,
                "growth_rate_sd": growth_sd_val,
            }
            train_vals = {
                "training_image": _clean(train_image),
                "training_polygons": _clean(train_polys),
                "class_column": (class_column.value or "").strip(),
                "max_pixels_per_class": _field_int(max_pixels_per_class, 0),
            }
            return {
                "input_raster": _clean(in_raster),
                "output_dir": _clean(output_dir),
                "otb_path": _clean(otb_bin),
                "custom_aoi": _clean(custom_aoi),
                "aoi_source": "custom" if _clean(custom_aoi) else "preselection",
                "preselection": pre_vals,
                "segmentation": seg_vals,
                "classification": clf_vals,
                "training": train_vals,
            }

        def on_msg(msg: dict):
            try:
                kind = msg.get("kind")
                if kind == "progress":
                    step_text.value = msg.get("text", "")
                    r = msg.get("ratio")
                    if r is not None:
                        progress.value = max(0.0, min(1.0, float(r)))
                elif kind == "preview":
                    pth = msg.get("path")
                    if pth:
                        # Try to open in the default browser
                        try:
                            page.launch_url("file:///" + str(Path(pth).resolve()).replace("\\", "/"))
                        except Exception:
                            pass
                        # Also show a lightweight dialog with an Open Again action
                        def _open_again(_: object | None = None):
                            try:
                                page.launch_url("file:///" + str(Path(pth).resolve()).replace("\\", "/"))
                            except Exception:
                                pass
                        dlg = ft.AlertDialog(
                            modal=True,
                            title=ft.Text("Preselection preview"),
                            content=ft.Text(f"Opened map in your browser.\n{pth}"),
                            actions=[
                                ft.TextButton("Open Again", on_click=_open_again),
                                ft.TextButton("Close", on_click=lambda e: (setattr(dlg, "open", False), page.update())),
                            ],
                        )
                        page.dialog = dlg
                        dlg.open = True
                elif kind == "classification_preview":
                    pth = msg.get("path")
                    if pth:
                        def _open_preview(_=None):
                            try:
                                page.launch_url("file:///" + str(Path(pth).resolve()).replace("\\", "/"))
                            except Exception:
                                pass
                        _open_preview()
                        dlg = ft.AlertDialog(
                            modal=True,
                            title=ft.Text("Classification preview"),
                            content=ft.Text(f"Opened map in your browser.\n{pth}"),
                            actions=[
                                ft.TextButton("Open Again", on_click=lambda e: _open_preview()),
                                ft.TextButton("Close", on_click=lambda e: (setattr(dlg, "open", False), page.update())),
                            ],
                        )
                        page.dialog = dlg
                        dlg.open = True
                elif kind == "result":
                    result_text.value = msg.get("text", "")
                elif kind == "workflow_done":
                    step_text.value = msg.get("text", "Workflow complete")
                    progress.value = 0.0
                    result_text.value = msg.get("text", "Workflow complete")
                    confirm_btn.visible = False
                    confirm_btn.disabled = True
                elif kind == "history_update":
                    record = msg.get("record")
                    if isinstance(record, dict):
                        target_root = record.get("output_root")
                        if target_root and target_root == _current_history_root_str():
                            refresh_history_panel()
                elif kind == "await_polygons":
                    step_text.value = msg.get("text", "Review polygons")
                    detail = msg.get("detail")
                    if detail:
                        result_text.value = detail
                    confirm_btn.visible = True
                    confirm_btn.disabled = False
                page.update()
            except Exception:
                pass

        def _on_confirm_polygons(_):
            event = workflow_gate.get("event")
            if event:
                workflow_gate["event"] = None
                confirm_btn.disabled = True
                confirm_btn.visible = False
                result_text.value = "Polygons confirmed. Continuing workflow..."
                page.update()
                event.set()
        confirm_btn.on_click = _on_confirm_polygons

        page.pubsub.subscribe(on_msg)
        tabs_ref: ft.Ref[ft.Tabs] = ft.Ref()

        # Pickers
        def on_pick_raster(e: ft.FilePickerResultEvent):
            try:
                if e.files:
                    selected = e.files[0].path
                    prev_in = in_raster.value
                    in_raster.value = selected
                    in_raster.update()
                    train_val = (train_image.value or "").strip()
                    if not train_val or (prev_in and train_val == (prev_in or "").strip()):
                        train_image.value = selected
                        train_image.update()
            except Exception:
                pass
        def on_pick_output(e: ft.FilePickerResultEvent):
            try:
                p = getattr(e, "path", None)
                if p:
                    output_dir.value = p; output_dir.update()
                    refresh_history_panel()
            except Exception:
                pass
        def on_pick_otb(e: ft.FilePickerResultEvent):
            try:
                if e.files:
                    otb_bin.value = e.files[0].path; otb_bin.update()
            except Exception:
                pass
        def on_pick_custom_aoi(e: ft.FilePickerResultEvent):
            try:
                if e.files:
                    custom_aoi.value = e.files[0].path
                    custom_aoi.update()
                    sync_preselection_state()
            except Exception:
                pass
        def on_pick_train_image(e: ft.FilePickerResultEvent):
            try:
                if e.files:
                    train_image.value = e.files[0].path
                    train_image.update()
            except Exception:
                pass
        fp_raster = ft.FilePicker(on_result=on_pick_raster)
        dp_output = ft.FilePicker(on_result=on_pick_output)
        fp_otb = ft.FilePicker(on_result=on_pick_otb)
        fp_custom_aoi = ft.FilePicker(on_result=on_pick_custom_aoi)
        fp_train_image = ft.FilePicker(on_result=on_pick_train_image)
        # Classification pickers (defined later)
        fp_train = ft.FilePicker()
        page.overlay.extend([fp_raster, dp_output, fp_otb, fp_custom_aoi, fp_train, fp_train_image])

        # Actions
        def run_preselection_only():
            if not in_raster.value.strip() or not output_dir.value.strip():
                step_text.value = "Please set input raster and output directory"; page.update(); return
            custom_path = (custom_aoi.value or "").strip()
            if custom_path and not os.path.exists(custom_path):
                step_text.value = "Custom AOI shapefile not found"; page.update(); return
            if custom_path and not custom_path.lower().endswith(".shp"):
                step_text.value = "Custom AOI must be a .shp file"; page.update(); return
            bio_mode_val = (biomass_mode.value or "preset")
            if bio_mode_val == "custom" and not (biomass_formula.value or "").strip():
                step_text.value = "Enter biomass (g) = f(x) with x as plot area (cm^2)"
                page.update()
                return
            settings_snapshot = collect_run_settings()
            def worker():
                try:
                    if custom_path:
                        page.pubsub.send_all({"kind": "progress", "text": "Using custom AOI shapefile", "ratio": 0.0})
                        aoi_final = save_preselection_to_output(custom_path, output_dir.value.strip())
                        try:
                            n_polys = len(gpd.read_file(aoi_final))
                        except Exception:
                            n_polys = None
                        lines = []
                        if n_polys is not None:
                            lines.append(f"Custom AOI: {n_polys} polygons")
                        lines.append(f"Saved: {aoi_final}")
                        result_msg = "\n".join(lines)
                    else:
                        min_w = _field_float(pre_wmin, 0.0) or 0.0
                        max_w = _field_float(pre_wmax)
                        min_l = _field_float(pre_lmin, 0.0) or 0.0
                        max_l = _field_float(pre_lmax)
                        small_buf = _field_float(pre_small_buffer, 0.3)
                        quant = float(pre_quantile.value or 0.15)
                        aoi_temp, n_polys = detect_cultivation_plots(
                            raster_path=in_raster.value.strip(),
                            output_root=output_dir.value.strip(),
                            min_width_m=min_w,
                            max_width_m=max_w,
                            min_length_m=min_l,
                            max_length_m=max_l,
                            small_polygon_buffer_m=small_buf if small_buf is not None else 0.0,
                            blue_quantile=quant,
                            progress=lambda d,t,n: page.pubsub.send_all({"kind":"progress","text": f"Preselection: {n or ''}", "ratio": (0 if t==0 else d/max(1,t))}),
                        )
                        aoi_final = save_preselection_to_output(aoi_temp, output_dir.value.strip())
                        result_msg = f"Preselection: {n_polys} polygons\nSaved: {aoi_final}"
                    path = build_preview_html(aoi_final, in_raster.value.strip())
                    if path:
                        page.pubsub.send_all({"kind": "preview", "path": path})
                    page.pubsub.send_all({"kind":"result","text": result_msg})
                except Exception as ex:  # noqa: BLE001
                    page.pubsub.send_all({"kind":"result","text": f"Preselection failed: {ex}"})
            threading.Thread(target=worker, daemon=True).start()

        def run_workflow(_):
            confirm_btn.visible = False
            confirm_btn.disabled = True
            workflow_gate["event"] = None
            page.update()
            if not in_raster.value.strip() or not output_dir.value.strip() or not os.path.exists(otb_bin.value.strip()):
                step_text.value = "Set input raster, output directory and a valid OTB path"; page.update(); return
            mode_sel = (classifier_mode.value or "rf")
            if mode_sel == "rf" and not (model_path.value or "").strip():
                step_text.value = "Select a model before running the workflow"
                page.update()
                return
            custom_path = (custom_aoi.value or "").strip()
            if custom_path and not os.path.exists(custom_path):
                step_text.value = "Custom AOI shapefile not found"; page.update(); return
            if custom_path and not custom_path.lower().endswith(".shp"):
                step_text.value = "Custom AOI must be a .shp file"; page.update(); return
            settings_snapshot = collect_run_settings()
            def worker():
                try:
                    bio_model = (
                        (settings_snapshot.get("classification") or {}).get("biomass_model") or "madagascar"
                    )
                    bio_formula = (settings_snapshot.get("classification") or {}).get("biomass_formula")
                    growth_rate_val = float((settings_snapshot.get("classification") or {}).get("growth_rate_pct", 5.8))
                    growth_sd_val = float((settings_snapshot.get("classification") or {}).get("growth_rate_sd", 0.7))
                    mode_local = mode_sel
                    # Preselection first
                    custom_path_local = custom_path
                    if custom_path_local:
                        page.pubsub.send_all({"kind": "progress", "text": "Using custom AOI shapefile", "ratio": 0.0})
                        aoi_final = save_preselection_to_output(custom_path_local, output_dir.value.strip())
                        try:
                            n_polys = len(gpd.read_file(aoi_final))
                        except Exception:
                            n_polys = None
                        lines = []
                        if n_polys is not None:
                            lines.append(f"Custom AOI: {n_polys} polygons")
                        lines.append(f"Saved: {aoi_final}")
                        result_msg = "\n".join(lines)
                    else:
                        min_w = _field_float(pre_wmin, 0.0) or 0.0
                        max_w = _field_float(pre_wmax)
                        min_l = _field_float(pre_lmin, 0.0) or 0.0
                        max_l = _field_float(pre_lmax)
                        small_buf = _field_float(pre_small_buffer, 0.3)
                        quant = float(pre_quantile.value or 0.15)
                        aoi_temp, n_polys = detect_cultivation_plots(
                            raster_path=in_raster.value.strip(),
                            output_root=output_dir.value.strip(),
                            min_width_m=min_w,
                            max_width_m=max_w,
                            min_length_m=min_l,
                            max_length_m=max_l,
                            small_polygon_buffer_m=small_buf if small_buf is not None else 0.0,
                            blue_quantile=quant,
                            progress=lambda d,t,n: page.pubsub.send_all({"kind":"progress","text": "Preselection", "ratio": (0 if t==0 else d/max(1,t))}),
                        )
                        aoi_final = save_preselection_to_output(aoi_temp, output_dir.value.strip())
                        result_msg = f"Preselection: {n_polys} polygons\nSaved: {aoi_final}"
                    path = build_preview_html(aoi_final, in_raster.value.strip())
                    if path:
                        page.pubsub.send_all({"kind": "preview", "path": path})
                    page.pubsub.send_all({"kind": "result", "text": result_msg})
                    confirm_event = threading.Event()
                    workflow_gate["event"] = confirm_event
                    page.pubsub.send_all({
                        "kind": "await_polygons",
                        "text": "Review polygons and click Confirm to continue.",
                        "detail": result_msg,
                    })
                    confirm_event.wait()
                    workflow_gate["event"] = None

                    # Parse segmentation parameters
                    tile_sz = _field_int(tile_size, 40)
                    spatial = _field_int(spatialr, 5)
                    mins = _field_int(minsize, 5)

                    # Classification settings
                    cap_pol = 200
                    model_file = None
                    if mode_local == "rf":
                        try:
                            cap_pol = int(max_pixels_per_polygon.value or "0")
                        except Exception:
                            cap_pol = 0
                        cap_pol = max(1, cap_pol) if cap_pol else 200
                        model_file = (model_path.value or "").strip()
                        if not model_file or not os.path.exists(model_file):
                            raise RuntimeError("Selected model file does not exist")

                    # Prepare AOI list
                    if not aoi_final or not os.path.exists(aoi_final):
                        raise RuntimeError("AOI file missing after preselection.")
                    aoi_gdf = gpd.read_file(aoi_final)
                    aoi_gdf = aoi_gdf.loc[~aoi_gdf.geometry.is_empty].reset_index(drop=True)
                    if aoi_gdf is None or len(aoi_gdf) == 0:
                        raise RuntimeError("No polygons to process after preselection.")

                    run_ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
                    base_out = Path(output_dir.value.strip())
                    temp_root = base_out.parent if base_out.name.lower() == "output" else base_out
                    work_dir = temp_root / "Temp" / f"workflow_{run_ts}"
                    shutil.rmtree(work_dir, ignore_errors=True)
                    work_dir.mkdir(parents=True, exist_ok=True)

                    seg_frames: list[gpd.GeoDataFrame] = []
                    seg_counter = 1
                    n_aoi = len(aoi_gdf)
                    for idx, geom in enumerate(aoi_gdf.geometry, start=1):
                        if geom is None or geom.is_empty:
                            continue
                        single = gpd.GeoDataFrame({"geometry": [geom]}, crs=aoi_gdf.crs)
                        single_path = work_dir / f"aoi_{idx}.shp"
                        single.to_file(single_path)

                        page.pubsub.send_all(
                            {"kind": "result", "text": f"AOI {idx}/{n_aoi}: running segmentation"}
                        )
                        seg_res = run_segmentation(
                            in_raster=in_raster.value.strip(),
                            output_root=str(work_dir / f"seg_run_{idx}"),
                            otb_bin=otb_bin.value.strip(),
                            tile_size_m=tile_sz,
                            spatialr=spatial,
                            minsize=mins,
                            aoi_path=str(single_path),
                            progress=lambda d, t, n, i=idx: page.pubsub.send_all(
                                {
                                    "kind": "progress",
                                    "text": f"Segmentation (AOI {i}/{n_aoi})",
                                    "ratio": (0 if t == 0 else d / max(1, t)),
                                }
                            ),
                        )
                        final_seg_path = seg_res.out_shp
                        try:
                            seg_gdf = gpd.read_file(final_seg_path)
                            seg_gdf = seg_gdf.loc[~seg_gdf.geometry.is_empty].reset_index(drop=True)
                            seg_len = len(seg_gdf)
                            if seg_len > 0:
                                drop_cols = [c for c in seg_gdf.columns if c.lower().startswith(("var", "mean"))]
                                if drop_cols:
                                    seg_gdf = seg_gdf.drop(columns=drop_cols, errors="ignore")
                                seg_gdf["seg_id"] = np.arange(seg_counter, seg_counter + seg_len, dtype=np.int64)
                                seg_gdf["plot_id"] = int(idx)
                                try:
                                    target_geom = (
                                        single.to_crs(seg_gdf.crs).geometry.iloc[0]
                                        if seg_gdf.crs and single.crs and str(seg_gdf.crs) != str(single.crs)
                                        else single.geometry.iloc[0]
                                    )
                                except Exception:
                                    target_geom = None
                                if target_geom is not None:
                                    try:
                                        seg_gdf["plot_area"] = seg_gdf.geometry.intersection(target_geom).area
                                    except Exception:
                                        seg_gdf["plot_area"] = seg_gdf.geometry.area
                                else:
                                    seg_gdf["plot_area"] = seg_gdf.geometry.area
                                seg_counter += seg_len
                                seg_frames.append(seg_gdf)
                        except Exception as seg_err:
                            page.pubsub.send_all(
                                {"kind": "result", "text": f"Warning: could not prepare segments for AOI {idx}: {seg_err}"}
                            )
                        page.pubsub.send_all(
                            {"kind": "result", "text": f"AOI {idx}/{n_aoi}: segmentation complete"}
                        )

                    if not seg_frames:
                        raise RuntimeError("No segments produced for classification.")

                    combined = gpd.GeoDataFrame(
                        pd.concat(seg_frames, ignore_index=True),
                        crs=seg_frames[0].crs if hasattr(seg_frames[0], "crs") else None,
                    )
                    combined_path = work_dir / "combined_segments.shp"
                    combined.to_file(combined_path)

                    page.pubsub.send_all({"kind": "result", "text": "Segments merged. Running classification..."})

                    try:
                        preview_path = None
                        summary_text = ""
                        result_output_path: Optional[str] = None
                        if mode_local == "rf":
                            cls_res = apply_model_with_pixel_sampling(
                                in_raster.value.strip(),
                                str(combined_path),
                                model_file,
                                output_dir.value.strip(),
                                max_pixels_per_polygon=cap_pol,
                                progress=lambda d, t, n: page.pubsub.send_all(
                                    {
                                        "kind": "progress",
                                        "text": "Classifying",
                                        "ratio": (0 if t == 0 else d / max(1, t)),
                                    }
                                ),
                                generate_preview=True,
                                aoi_path=aoi_final,
                                biomass_model=bio_model,
                                biomass_formula=bio_formula,
                                growth_rate_pct=growth_rate_val,
                                growth_rate_sd=growth_sd_val,
                            )
                            summary_text = f"Workflow complete. Classification saved to:\n{cls_res.output_path}"
                            result_output_path = str(cls_res.output_path)
                            page.pubsub.send_all({"kind": "result", "text": summary_text})
                            preview_path = cls_res.preview_map
                        else:
                            stats_res = classify_dark_linear_polygons(
                                in_raster.value.strip(),
                                str(combined_path),
                                output_dir.value.strip(),
                                biomass_model=bio_model,
                                biomass_formula=bio_formula,
                                growth_rate_pct=growth_rate_val,
                                growth_rate_sd=growth_sd_val,
                                progress=lambda d, t, n: page.pubsub.send_all(
                                    {
                                        "kind": "progress",
                                        "text": n or "Statistics classifier",
                                        "ratio": (0 if t == 0 else d / max(1, t)),
                                    }
                                ),
                            )
                            summary_text = (
                                "Workflow complete (statistics classifier).\n"
                                f"Selected {stats_res.selected_count} polygons via spectral rules.\n"
                                f"Saved to:\n{stats_res.output_path}"
                            )
                            result_output_path = str(stats_res.output_path)
                            page.pubsub.send_all({"kind": "result", "text": summary_text})
                            try:
                                preview_path = build_classification_map(
                                    stats_res.output_path,
                                    in_raster.value.strip(),
                                    mode="stats",
                                    aoi_path=aoi_final,
                                    biomass_model=bio_model,
                                    biomass_formula=bio_formula,
                                    growth_rate_pct=growth_rate_val,
                                    growth_rate_sd=growth_sd_val,
                                )
                            except Exception:
                                preview_path = None
                        if preview_path:
                            page.pubsub.send_all({"kind": "classification_preview", "path": str(preview_path)})
                        if result_output_path:
                            completed = datetime.now()
                            label = completed.strftime("%Y-%m-%d %H:%M:%S")
                            metrics = {"aois": int(n_aoi), "segments": int(len(combined))}
                            try:
                                safe_settings = json.loads(json.dumps(settings_snapshot))
                            except Exception:
                                safe_settings = settings_snapshot
                            record = {
                                "id": run_ts,
                                "label": f"{label} ({(mode_local or '').upper()})",
                                "mode": mode_local,
                                "completed_at": completed.isoformat(),
                                "summary": summary_text,
                                "result_path": result_output_path,
                                "preview_map": str(preview_path) if preview_path else None,
                                "aoi_path": str(aoi_final),
                                "run_folder": str(Path(result_output_path).parent),
                                "settings": safe_settings,
                                "metrics": metrics,
                            }
                            stored = _append_run_history(settings_snapshot.get("output_dir"), record)
                            if stored:
                                page.pubsub.send_all({"kind": "history_update", "record": stored})
                        _cleanup_root_tiffs()
                        page.pubsub.send_all({"kind": "workflow_done", "text": "Workflow complete."})
                    finally:
                        shutil.rmtree(work_dir, ignore_errors=True)
                except Exception as ex:  # noqa: BLE001
                    page.pubsub.send_all({"kind":"result","text": f"Workflow failed: {ex}"})
            threading.Thread(target=worker, daemon=True).start()

        # Layout â€“ Project Paths (neutral)
        left = section(
            "Project Paths",
            labeled_row("Input raster", ft.Row([in_raster, ft.OutlinedButton("Browse", icon=ft.icons.FOLDER_OPEN, on_click=lambda _: fp_raster.pick_files(allow_multiple=False))], expand=True), icon=ft.icons.IMAGE_OUTLINED, tip="Path to input imagery (GeoTIFF)."),
            labeled_row("Output directory", ft.Row([output_dir, ft.OutlinedButton("Browse", icon=ft.icons.FOLDER_OPEN, on_click=lambda _: dp_output.get_directory_path())], expand=True), icon=ft.icons.FOLDER, tip="Folder where results will be written."),
            labeled_row("OTB path", ft.Row([otb_bin, ft.OutlinedButton("Browse", icon=ft.icons.BUILD_OUTLINED, on_click=lambda _: fp_otb.pick_files(allow_multiple=False))], expand=True), icon=ft.icons.BUILD_OUTLINED, tip="Path to OTB LargeScaleMeanShift executable."),
            labeled_row(
                "Custom AOI (.shp)",
                ft.Row(
                    [
                        custom_aoi,
                        ft.OutlinedButton(
                            "Browse",
                            icon=ft.icons.MAP,
                            on_click=lambda _: fp_custom_aoi.pick_files(allow_multiple=False),
                        ),
                    ],
                    spacing=6,
                    expand=True,
                ),
                icon=ft.icons.MAP,
                tip="Optional shapefile used as the AOI. When set, preselection is skipped and this file opens in the editor.",
            ),
            subtitle="Set inputs and outputs.",
            bgcolor=pal["alt"],
            expanded=True,
        )
        # Preselection (emerald)
        pre = section(
            "Preselection",
            ft.Text("Set the expected plot sizes (meters). Leave max fields empty if unknown.", color=pal["muted"]),
            ft.Row(
                [
                    labeled_row("Min width (m)", pre_wmin, icon=ft.icons.UNFOLD_MORE, tip="Plots narrower than this are ignored when estimating orientation or merging."),
                    labeled_row("Max width (m)", pre_wmax, icon=ft.icons.UNFOLD_MORE, tip="Optional ceiling for plot width."),
                ],
                wrap=True,
            ),
            ft.Row(
                [
                    labeled_row("Min length (m)", pre_lmin, icon=ft.icons.SPACE_BAR, tip="Minimum expected plot length."),
                    labeled_row("Max length (m)", pre_lmax, icon=ft.icons.SPACE_BAR, tip="Optional ceiling for plot length."),
                ],
                wrap=True,
            ),
            ft.Row(
                [
                    labeled_row(
                        "Small-plot buffer (m)",
                        pre_small_buffer,
                        icon=ft.icons.CROP_FREE,
                        tip="Undersized polygons are buffered by this amount before merging so they can fuse with neighbours.",
                    ),
                ],
                wrap=True,
            ),
            ft.Container(
                ft.Column(
                    [
                        pre_quantile_label,
                        pre_quantile,
                    ],
                    spacing=4,
                ),
                padding=ft.padding.symmetric(horizontal=4),
            ),
            pre_disabled_hint,
            subtitle="Tune AOI detection and run.",
            bgcolor="#EAF7F5",
        )
        # Segmentation (blue)
        tile_size = ft.TextField(value="40", width=120)
        spatialr = ft.TextField(value="5", width=120)
        minsize = ft.TextField(value="5", width=120)
        seg = section(
            "Segmentation",
            ft.Row([
                labeled_row("Tile size (m)", tile_size, icon=ft.icons.STRAIGHTEN, tip="Tile size for mean-shift segmentation."),
                labeled_row("Spatialr", spatialr, icon=ft.icons.GRAIN, tip="Spatial radius parameter of mean-shift."),
                labeled_row("Minsize", minsize, icon=ft.icons.DETAILS, tip="Minimum region size for merging."),
            ], wrap=True),
            subtitle="OTB LargeScaleMeanShift parameters.",
            bgcolor="#ECF4FB",
        )

        # Classification (amber)
        model_path = ft.Dropdown(
            options=[],
            expand=True,
            hint_text="Select a model from the 'Model' directory",
        )
        model_hint = ft.Text("", size=12, color=pal["muted"])
        max_pixels_per_polygon = ft.TextField(value="200", width=120, hint_text="Pixels/polygon", text_align=ft.TextAlign.RIGHT)
        biomass_mode = ft.RadioGroup(
            value="preset",
            content=ft.Row(
                [
                    ft.Radio(value="preset", label="Preset model"),
                    ft.Radio(value="custom", label="Custom equation"),
                ],
                wrap=True,
            ),
        )
        biomass_model = ft.Dropdown(
            value="madagascar",
            options=[
                ft.dropdown.Option(key="madagascar", text="Madagascar (default curve)"),
                # ft.dropdown.Option(key="indonesia", text="Indonesia (Nurdin et al. 2023)"),
            ],
            width=220,
        )
        biomass_formula = ft.TextField(
            value="",
            width=260,
            hint_text="Example: 0.8 * x ** 1.25",
        )
        biomass_hint = ft.Text("Use x for the plot area in square centimeters.", size=12, color=pal["muted"])
        def sync_biomass_mode(_=None) -> None:
            is_custom = (biomass_mode.value or "preset") == "custom"
            biomass_model.disabled = is_custom
            biomass_formula.disabled = not is_custom
            if is_custom:
                biomass_model.value = "custom"
            _safe_update(biomass_model)
            _safe_update(biomass_formula)

        biomass_mode.on_change = sync_biomass_mode
        growth_rate = ft.TextField(
            value="5.8",
            width=120,
            suffix_text="%/day",
            text_align=ft.TextAlign.RIGHT,
            tooltip="Daily growth rate applied to biomass to project the next 7 days.",
        )
        growth_sd = ft.TextField(
            value="0.7",
            width=120,
            suffix_text="%/day",
            text_align=ft.TextAlign.RIGHT,
            tooltip="Standard deviation for growth rate (informational).",
        )
        train_polys = ft.TextField(value="", expand=True, hint_text="Training polygons (.shp/.gpkg/.geojson)")
        class_column = ft.Dropdown(options=[], expand=True)
        max_pixels_per_class = ft.TextField(value="0", width=120, hint_text="0 = no cap")
        train_info = ft.Text("", size=12, color=pal["muted"])
        classifier_mode = ft.RadioGroup(
            value="rf",
            content=ft.Row(
                [
                    ft.Radio(value="rf", label="Random Forest model"),
                    ft.Radio(value="stats", label="Statistics (dark rows)"),
                ],
                wrap=True,
            ),
        )
        rf_controls: list[ft.Control] = []
        model_hint_state = {"text": "Select a trained model from the 'Model' directory."}
        model_hint.value = model_hint_state["text"]

        def update_model_hint() -> None:
            if (classifier_mode.value or "rf") == "stats":
                model_hint.value = "Statistics mode enabled – Random Forest controls are disabled."
            else:
                model_hint.value = model_hint_state["text"]
            _safe_update(model_hint)

        def sync_classifier_mode() -> None:
            use_rf = (classifier_mode.value or "rf") == "rf"
            for ctrl in rf_controls:
                ctrl.disabled = not use_rf
                _safe_update(ctrl)
            update_model_hint()

        classifier_mode.on_change = lambda e: sync_classifier_mode()

        def switch_to_training_tab(_=None):
            tabs = tabs_ref.current
            if tabs is not None:
                tabs.selected_index = 1
                tabs.update()

        def on_pick_train(e: ft.FilePickerResultEvent):
            try:
                if e.files:
                    train_polys.value = e.files[0].path
                    train_polys.update()
                    load_fields(None)
            except Exception:
                pass

        fp_train.on_result = on_pick_train

        def refresh_model_list(_=None):
            root = Path("Model")
            options: list[ft.dropdown.Option] = []
            note = ""
            try:
                files = sorted(p for p in root.glob("*.joblib")) if root.exists() else []
                if files:
                    options = [ft.dropdown.Option(text=fp.name, key=str(fp.resolve())) for fp in files]
                    current_keys = {opt.key or opt.text for opt in options}
                    if model_path.value not in current_keys:
                        first = options[0]
                        model_path.value = first.key or first.text
                    note = f"Found {len(files)} model(s) in '{root}'."
                else:
                    model_path.value = None
                    note = f"No .joblib models found in '{root}'."
            except Exception as exc:  # noqa: BLE001
                model_path.value = None
                note = f"Failed to read '{root}': {exc}"
            model_path.options = options
            model_path.update()
            model_hint_state["text"] = note
            update_model_hint()

        def load_fields(_):
            try:
                path = (train_polys.value or "").strip()
                if path and os.path.exists(path):
                    gdf = gpd.read_file(path)
                    cols = [c for c in gdf.columns if c.lower() != "geometry"]
                    class_column.options = [ft.dropdown.Option(c) for c in cols]
                    if cols:
                        class_column.value = cols[0]
                    train_info.value = f"Polygons: {len(gdf)} | Fields: {len(cols)}"
                    page.update()
            except Exception:
                train_info.value = "Could not read fields."
                page.update()

        refresh_btn = ft.IconButton(icon=ft.icons.REFRESH, tooltip="Rescan Model directory", on_click=refresh_model_list)
        rf_controls.extend([model_path, refresh_btn, max_pixels_per_polygon])
        train_tab_btn = ft.TextButton("Need to train a model? Open the Train Model tab →", on_click=switch_to_training_tab)
        rf_controls.append(train_tab_btn)
        clf = section(
            "Classification",
            labeled_row(
                "Method",
                classifier_mode,
                icon=ft.icons.TUNE,
                tip="Pick Random Forest (requires a trained model) or the statistics-based classifier.",
            ),
            labeled_row(
                "Model (.joblib)",
                ft.Row(
                    [
                        model_path,
                        refresh_btn,
                    ],
                    spacing=8,
                    expand=True,
                ),
                icon=ft.icons.SAVE_ALT,
                tip="Select a trained classifier stored under the Model directory.",
            ),
            model_hint,
            labeled_row(
                "Pixels/polygon (prediction)",
                max_pixels_per_polygon,
                icon=ft.icons.SPEED,
                tip="Number of pixels sampled in each polygon when applying the model.",
            ),
            train_tab_btn,
            subtitle="Choose between a trained Random Forest and the simple statistics classifier.",
            bgcolor="#FFF4EA",
        )
        bio_section = section(
            "Biomass Estimation",
            labeled_row(
                "Mode",
                biomass_mode,
                icon=ft.icons.SCIENCE,
                tip="Switch between presets and your own biomass equation.",
            ),
            labeled_row(
                "Preset model",
                biomass_model,
                icon=ft.icons.BIOTECH,
                tip="Built-in biomass-area relationships.",
            ),
            labeled_row(
                "Biomass (g) =",
                biomass_formula,
                icon=ft.icons.FORMAT_COLOR_TEXT,
                tip="Enter biomass (g) = f(x) with x as plot area (cm^2).",
            ),
            ft.Row(
                [
                    labeled_row(
                        "Growth rate",
                        growth_rate,
                        icon=ft.icons.TRENDING_UP,
                        tip="Daily growth rate (%) used to project biomass for the next 7 days.",
                    ),
                    labeled_row(
                        "Std. dev.",
                        growth_sd,
                        icon=ft.icons.SHOW_CHART,
                        tip="Standard deviation for the growth rate (stored for reference).",
                    ),
                ],
                wrap=True,
            ),
            ft.Container(biomass_hint, padding=ft.padding.only(left=12)),
            subtitle="Choose \"Custom equation\" and use x for plot area (cm^2).",
            bgcolor="#EEF4ED",
        )
        def on_train_model(_):
            switch_to_training_tab()
            img = (train_image.value or "").strip() or (in_raster.value or "").strip()
            polys = (train_polys.value or "").strip()
            cls_col = (class_column.value or "").strip()
            out_root = (output_dir.value or "").strip()
            if not img:
                step_text.value = "Select a training image"
                page.update()
                return
            if not os.path.exists(img):
                step_text.value = "Training image path is invalid"
                page.update()
                return
            if not polys:
                step_text.value = "Select training polygons"
                page.update()
                return
            if not os.path.exists(polys):
                step_text.value = "Training polygons path is invalid"
                page.update()
                return
            if not cls_col:
                step_text.value = "Pick the class column"
                page.update()
                return
            if not out_root:
                step_text.value = "Set an output directory first"
                page.update()
                return
            step_text.value = "Training model..."
            page.update()

            def progress_cb(done: int, total: int, note: str = "") -> None:
                label = "Training"
                if note:
                    label = f"{label} – {note}"
                ratio = (done / max(1, total)) if total else 0.0
                page.pubsub.send_all({"kind": "progress", "text": label, "ratio": ratio})

            def worker():
                try:
                    cap = 0
                    try:
                        cap = int(max_pixels_per_class.value or "0")
                    except Exception:
                        cap = 0
                    res = train_model_from_training_polys(
                        img,
                        polys,
                        cls_col,
                        out_root,
                        progress=progress_cb,
                        max_pixels_per_class=(cap if cap > 0 else None),
                    )
                    page.pubsub.send_all({"kind": "result", "text": f"Training complete. Model saved to {res.model_path}"})

                    def finalize():
                        refresh_model_list()
                        model_path.value = str(res.model_path)
                        model_path.update()
                        step_text.value = "Training complete"
                        page.update()

                    page.call_from_thread(finalize)
                except Exception as exc:  # noqa: BLE001
                    page.pubsub.send_all({"kind": "result", "text": f"Training failed: {exc}"})

                    def fail():
                        step_text.value = "Training failed"
                        page.update()

                    page.call_from_thread(fail)

            threading.Thread(target=worker, daemon=True).start()

        training_section = section(
            "Training Inputs",
            labeled_row(
                "Training image",
                ft.Row(
                    [
                        train_image,
                        ft.OutlinedButton(
                            "Browse",
                            icon=ft.icons.IMAGE_OUTLINED,
                            on_click=lambda _: fp_train_image.pick_files(allow_multiple=False),
                        ),
                    ],
                    spacing=6,
                    expand=True,
                ),
                icon=ft.icons.IMAGE_OUTLINED,
                tip="Raster used to sample per-pixel spectra for training.",
            ),
            labeled_row(
                "Training polygons",
                ft.Row(
                    [
                        train_polys,
                        ft.OutlinedButton(
                            "Browse",
                            icon=ft.icons.FOLDER_OPEN,
                            on_click=lambda _: fp_train.pick_files(allow_multiple=False),
                        ),
                        ft.OutlinedButton("Load Fields", icon=ft.icons.TABLE_VIEW, on_click=load_fields),
                    ],
                    spacing=6,
                    expand=True,
                ),
                icon=ft.icons.MAP_OUTLINED,
                tip="Vector dataset containing class labels.",
            ),
            ft.Row(
                [
                    labeled_row(
                        "Class column",
                        class_column,
                        icon=ft.icons.LIST_ALT,
                        tip="Attribute containing the class name.",
                    ),
                    labeled_row(
                        "Max pixels/class",
                        max_pixels_per_class,
                        icon=ft.icons.SPEED,
                        tip="Classes exceeding this limit are randomly down-sampled (0 = no cap).",
                    ),
                ],
                wrap=True,
            ),
            ft.Container(train_info, padding=ft.padding.only(left=8)),
            ft.Text(
                "If a class contains more pixels than the limit, samples are randomly drawn to match the requested count.",
                color=pal["muted"],
                size=12,
            ),
            subtitle="Train a new model from labeled polygons.",
            bgcolor=pal["alt"],
        )
        train_btn = ft.ElevatedButton("Train Model", icon=ft.icons.SCIENCE, on_click=on_train_model)

        # Footer / actions
        pre_btn = ft.ElevatedButton("Run Preselection", icon=ft.icons.SEARCH, on_click=lambda _: run_preselection_only())
        pre_button_ref["btn"] = pre_btn
        run_btn = ft.ElevatedButton("Run Workflow", icon=ft.icons.PLAY_ARROW, on_click=run_workflow)
        sync_preselection_state()

        # Two-column responsive layout: each section ~half width
        left_col = ft.Column([left, pre], expand=1, spacing=12)
        right_col = ft.Column([seg, clf, bio_section], expand=1, spacing=12)

        main_row = ft.Row(
            [left_col, right_col],
            expand=True,
            spacing=12,
            vertical_alignment=ft.CrossAxisAlignment.START,
        )
        action_row = ft.Row(
            [
                ft.Container(expand=True),
                pre_btn,
                run_btn,
            ],
            alignment=ft.MainAxisAlignment.END,
            spacing=12,
        )
        workflow_body = ft.ListView(
            controls=[
                ft.Container(content=main_row, expand=True),
                ft.Container(content=action_row, expand=True),
            ],
            expand=True,
            spacing=12,
            auto_scroll=False,
            padding=0,
        )

        training_body = ft.Column(
            [
                training_section,
                ft.Row(
                    [
                        ft.Container(expand=True),
                        train_btn,
                    ],
                    alignment=ft.MainAxisAlignment.END,
                    spacing=12,
                ),
            ],
            expand=True,
            spacing=12,
        )

        result_panel = ft.Column(
            [result_text, confirm_btn],
            spacing=6,
            horizontal_alignment=ft.CrossAxisAlignment.START,
        )

        history_refresh_btn = ft.IconButton(
            icon=ft.icons.REFRESH,
            tooltip="Reload saved runs",
            on_click=refresh_history_panel,
        )
        history_panel = section(
            "Processed Runs",
            ft.Column(
                [
                    ft.Row(
                        [
                            ft.Container(history_hint, expand=True),
                            history_refresh_btn,
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    ft.Row(
                        [
                            ft.Container(history_list, expand=1),
                            ft.Column(
                                [
                                    ft.Text("Summary", weight=ft.FontWeight.W_600, color=pal["muted"]),
                                    history_summary,
                                    ft.Text("Saved inputs", weight=ft.FontWeight.W_600, color=pal["muted"]),
                                    ft.Container(
                                        history_settings,
                                        border=ft.border.all(1, pal["border"]),
                                        padding=10,
                                        bgcolor=pal["alt"],
                                        height=240,
                                    ),
                                ],
                                spacing=6,
                                expand=1,
                            ),
                        ],
                        spacing=12,
                        vertical_alignment=ft.CrossAxisAlignment.START,
                    ),
                ],
                spacing=10,
            ),
            icon=ft.icons.HISTORY,
            subtitle="Select a processed run to review its settings and reopen the map.",
            bgcolor=pal["card"],
            expanded=True,
        )
        catalogue_body = ft.Column(
            [
                history_panel,
            ],
            expand=True,
            spacing=12,
        )

        tabs = ft.Tabs(
            ref=tabs_ref,
            selected_index=0,
            expand=1,
            tabs=[
                ft.Tab(text="Workflow", content=workflow_body),
                ft.Tab(text="Train Model", content=training_body),
                ft.Tab(text="Catalogue", content=catalogue_body),
            ],
        )

        page.add(
            ft.Column(
                [
                    tabs,
                    ft.Container(step_text, padding=10),
                    ft.Container(progress, padding=8),
                    result_panel,
                ],
                expand=True,
                spacing=8,
            )
        )

        sync_biomass_mode()
        sync_classifier_mode()
        refresh_model_list()
        refresh_history_panel()

    import flet as ft  # type: ignore
    ft.app(target=main)


if __name__ == "__main__":
    run_flet_app()





