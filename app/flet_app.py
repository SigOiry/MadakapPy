from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional


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
    import numpy as np
    import geopandas as gpd
    import folium
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

    def section(title: str, *children: ft.Control, icon: Optional[str] = None, subtitle: Optional[str] = None, bgcolor: Optional[str] = None) -> ft.Card:
        pal = _palette()
        tile = ft.ExpansionTile(
            title=ft.Row([
                ft.Icon(icon, color=pal["primary"]) if icon else ft.Container(),
                ft.Text(title, size=16, weight=ft.FontWeight.W_600),
            ], spacing=8),
            subtitle=ft.Text(subtitle or "", size=12, color=pal["muted"]) if subtitle else None,
            initially_expanded=True,
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

            m = folium.Map()
            folium.raster_layers.ImageOverlay(
                image=f"data:image/png;base64,{b64_png}",
                bounds=[[south, west], [north, east]],
                opacity=1.0,
            ).add_to(m)
            folium.GeoJson(
                data=gdf_wgs.__geo_interface__,
                style_function=lambda _: {"color": "yellow", "weight": 1, "fillColor": "yellow", "fillOpacity": 0.25},
            ).add_to(m)
            m.fit_bounds([[south, west], [north, east]])
            html = m.get_root().render()
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

        # Preselection controls (with requested defaults)
        pre_disable_filters = ft.Checkbox(label="Disable filters (raw)", value=False)
        pre_disable_filters.tooltip = "Bypass all polygon filters; export raw rectangles."
        pre_mask_mode = ft.Dropdown(options=[ft.dropdown.Option("and"), ft.dropdown.Option("or")], value="and", width=140)
        pre_min_region = ft.TextField(value="300", width=120)
        pre_ar_min = ft.TextField(value="1.2", width=120)
        pre_ar_max = ft.TextField(value="12.0", width=120)
        pre_orient_tol = ft.TextField(value="12", width=120)
        pre_wmin = ft.TextField(value="5", width=120)
        pre_wmax = ft.TextField(value="20", width=120)
        pre_lmin = ft.TextField(value="20", width=120)
        pre_lmax = ft.TextField(value="60", width=120)
        # Image transform & detection tunables
        pre_downscale = ft.TextField(value="1200", width=120)
        pre_blue_band = ft.TextField(value="1", width=100)
        pre_clahe_enable = ft.Checkbox(label="CLAHE", value=True)
        pre_clahe_enable.tooltip = "Enable local-contrast enhancement (CLAHE)."
        pre_clahe_clip = ft.TextField(value="2.0", width=100)
        pre_clahe_tile = ft.TextField(value="8", width=100)
        pre_tex_win = ft.TextField(value="10", width=100)
        pre_thr_b_relax = ft.TextField(value="0.95", width=100)
        pre_thr_t_relax = ft.TextField(value="0.90", width=100)
        pre_morph_radius = ft.TextField(value="1", width=100)
        pre_hole_frac = ft.TextField(value="0.4", width=100)
        pre_approx_eps = ft.TextField(value="0.02", width=100)
        pre_fill_min = ft.TextField(value="0.35", width=100)
        pre_nms_iou = ft.TextField(value="0.25", width=100)
        pre_buf_shrink = ft.TextField(value="0.05", width=100)

        # Status
        step_text = ft.Text("Ready", size=12, color=pal["muted"]) 
        progress = ft.ProgressBar(value=0.0, color=pal["primary"], bgcolor=pal["border"]) 
        result_text = ft.Text("", selectable=True, color=pal["muted"]) 

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
                elif kind == "result":
                    result_text.value = msg.get("text", "")
                page.update()
            except Exception:
                pass

        page.pubsub.subscribe(on_msg)

        # Pickers
        def on_pick_raster(e: ft.FilePickerResultEvent):
            try:
                if e.files:
                    in_raster.value = e.files[0].path; in_raster.update()
            except Exception:
                pass
        def on_pick_output(e: ft.FilePickerResultEvent):
            try:
                p = getattr(e, "path", None)
                if p:
                    output_dir.value = p; output_dir.update()
            except Exception:
                pass
        def on_pick_otb(e: ft.FilePickerResultEvent):
            try:
                if e.files:
                    otb_bin.value = e.files[0].path; otb_bin.update()
            except Exception:
                pass
        fp_raster = ft.FilePicker(on_result=on_pick_raster)
        dp_output = ft.FilePicker(on_result=on_pick_output)
        fp_otb = ft.FilePicker(on_result=on_pick_otb)
        # Classification pickers (defined later)
        fp_model = ft.FilePicker()
        fp_train = ft.FilePicker()
        page.overlay.extend([fp_raster, dp_output, fp_otb, fp_model, fp_train])

        # Actions
        def run_preselection_only():
            if not in_raster.value.strip() or not output_dir.value.strip():
                step_text.value = "Please set input raster and output directory"; page.update(); return
            def worker():
                try:
                    url = None
                    aoi_temp, n_polys = detect_cultivation_plots(
                        raster_path=in_raster.value.strip(),
                        output_root=output_dir.value.strip(),
                        downscale_max=int(pre_downscale.value or 1200),
                        disable_filters=bool(pre_disable_filters.value),
                        min_region_px=int(pre_min_region.value or 300),
                        mask_mode=(pre_mask_mode.value or "and"),
                        orient_tolerance_deg=float(pre_orient_tol.value or 12),
                        ar_min=float(pre_ar_min.value or 1.2),
                        ar_max=float(pre_ar_max.value or 12.0),
                        blue_band_index=int(pre_blue_band.value or 1),
                        clahe_enable=bool(pre_clahe_enable.value),
                        clahe_clip=float(pre_clahe_clip.value or 2.0),
                        clahe_tile=int(pre_clahe_tile.value or 8),
                        texture_window=int(pre_tex_win.value or 10),
                        thr_relax_blue=float(pre_thr_b_relax.value or 0.95),
                        thr_relax_tex=float(pre_thr_t_relax.value or 0.90),
                        morph_close_radius=int(pre_morph_radius.value or 1),
                        hole_area_frac=float(pre_hole_frac.value or 0.4),
                        approx_epsilon_frac=float(pre_approx_eps.value or 0.02),
                        fill_ratio_min=float(pre_fill_min.value or 0.35),
                        nms_iou_thresh=float(pre_nms_iou.value or 0.25),
                        buffer_shrink=float(pre_buf_shrink.value or 0.05),
                        width_m_range=((float(pre_wmin.value), float(pre_wmax.value)) if float(pre_wmax.value or 0) >= float(pre_wmin.value or 0) and float(pre_wmax.value or 0) > 0 else None),
                        length_m_range=((float(pre_lmin.value), float(pre_lmax.value)) if float(pre_lmax.value or 0) >= float(pre_lmin.value or 0) and float(pre_lmax.value or 0) > 0 else None),
                        progress=lambda d,t,n: page.pubsub.send_all({"kind":"progress","text": f"Preselection: {n or ''}", "ratio": (0 if t==0 else d/max(1,t))}),
                    )
                    aoi_final = save_preselection_to_output(aoi_temp, output_dir.value.strip())
                    path = build_preview_html(aoi_final, in_raster.value.strip())
                    if path:
                        page.pubsub.send_all({"kind": "preview", "path": path})
                    page.pubsub.send_all({"kind":"result","text": f"Preselection: {n_polys} polygons\nSaved: {aoi_final}"})
                except Exception as ex:  # noqa: BLE001
                    page.pubsub.send_all({"kind":"result","text": f"Preselection failed: {ex}"})
            threading.Thread(target=worker, daemon=True).start()

        def run_workflow(_):
            if not in_raster.value.strip() or not output_dir.value.strip() or not os.path.exists(otb_bin.value.strip()):
                step_text.value = "Set input raster, output directory and a valid OTB path"; page.update(); return
            def worker():
                try:
                    # Preselection first
                    aoi_temp, _ = detect_cultivation_plots(
                        raster_path=in_raster.value.strip(),
                        output_root=output_dir.value.strip(),
                        downscale_max=int(pre_downscale.value or 1200),
                        disable_filters=bool(pre_disable_filters.value),
                        min_region_px=int(pre_min_region.value or 300),
                        mask_mode=(pre_mask_mode.value or "and"),
                        orient_tolerance_deg=float(pre_orient_tol.value or 12),
                        ar_min=float(pre_ar_min.value or 1.2),
                        ar_max=float(pre_ar_max.value or 12.0),
                        blue_band_index=int(pre_blue_band.value or 1),
                        clahe_enable=bool(pre_clahe_enable.value),
                        clahe_clip=float(pre_clahe_clip.value or 2.0),
                        clahe_tile=int(pre_clahe_tile.value or 8),
                        texture_window=int(pre_tex_win.value or 10),
                        thr_relax_blue=float(pre_thr_b_relax.value or 0.95),
                        thr_relax_tex=float(pre_thr_t_relax.value or 0.90),
                        morph_close_radius=int(pre_morph_radius.value or 1),
                        hole_area_frac=float(pre_hole_frac.value or 0.4),
                        approx_epsilon_frac=float(pre_approx_eps.value or 0.02),
                        fill_ratio_min=float(pre_fill_min.value or 0.35),
                        nms_iou_thresh=float(pre_nms_iou.value or 0.25),
                        buffer_shrink=float(pre_buf_shrink.value or 0.05),
                        width_m_range=((float(pre_wmin.value), float(pre_wmax.value)) if float(pre_wmax.value or 0) >= float(pre_wmin.value or 0) and float(pre_wmax.value or 0) > 0 else None),
                        length_m_range=((float(pre_lmin.value), float(pre_lmax.value)) if float(pre_lmax.value or 0) >= float(pre_lmin.value or 0) and float(pre_lmax.value or 0) > 0 else None),
                        progress=lambda d,t,n: page.pubsub.send_all({"kind":"progress","text": "Preselection", "ratio": (0 if t==0 else d/max(1,t))}),
                    )
                    aoi_final = save_preselection_to_output(aoi_temp, output_dir.value.strip())
                    path = build_preview_html(aoi_final, in_raster.value.strip())
                    if path:
                        page.pubsub.send_all({"kind": "preview", "path": path})

                    # Segmentation
                    seg_res = run_segmentation(
                        in_raster=in_raster.value.strip(),
                        output_root=output_dir.value.strip(),
                        otb_bin=otb_bin.value.strip(),
                        progress=lambda d,t,n: page.pubsub.send_all({"kind":"progress","text": "Segmentation", "ratio": (0 if t==0 else d/max(1,t))}),
                    )
                    page.pubsub.send_all({"kind":"result","text": f"Segmentation done. Outputs in: {seg_res.out_dir}"})
                except Exception as ex:  # noqa: BLE001
                    page.pubsub.send_all({"kind":"result","text": f"Workflow failed: {ex}"})
            threading.Thread(target=worker, daemon=True).start()

        # Layout â€“ Project Paths (neutral)
        left = section(
            "Project Paths",
            labeled_row("Input raster", ft.Row([in_raster, ft.OutlinedButton("Browse", icon=ft.icons.FOLDER_OPEN, on_click=lambda _: fp_raster.pick_files(allow_multiple=False))], expand=True), icon=ft.icons.IMAGE_OUTLINED, tip="Path to input imagery (GeoTIFF)."),
            labeled_row("Output directory", ft.Row([output_dir, ft.OutlinedButton("Browse", icon=ft.icons.FOLDER_OPEN, on_click=lambda _: dp_output.get_directory_path())], expand=True), icon=ft.icons.FOLDER, tip="Folder where results will be written."),
            labeled_row("OTB path", ft.Row([otb_bin, ft.OutlinedButton("Browse", icon=ft.icons.BUILD_OUTLINED, on_click=lambda _: fp_otb.pick_files(allow_multiple=False))], expand=True), icon=ft.icons.BUILD_OUTLINED, tip="Path to OTB LargeScaleMeanShift executable."),
            subtitle="Set inputs and outputs.",
            bgcolor=pal["alt"],
        )
        # Preselection (emerald)
        pre = section(
            "Preselection",
            # Collapsible subsections
            ft.ExpansionTile(
                title=ft.Row([ft.Icon(ft.icons.FILTER_ALT), ft.Text("Masks & gating", weight=ft.FontWeight.W_600)], spacing=8),
                initially_expanded=True,
                controls=[
                    ft.Row([pre_disable_filters, labeled_row("Mask mode", pre_mask_mode, icon=ft.icons.FILTER_ALT, tip="Combine masks: AND conservative, OR permissive."), labeled_row("Min region (px)", pre_min_region, icon=ft.icons.GRID_ON, tip="Minimum connected-component size at downscaled resolution.")], wrap=True),
                    ft.Row([labeled_row("AR min", pre_ar_min, icon=ft.icons.STRAIGHTEN, tip="Minimum aspect ratio (long/short)."), labeled_row("AR max", pre_ar_max, icon=ft.icons.STRAIGHTEN, tip="Maximum aspect ratio (long/short)."), labeled_row("Orient tol (deg)", pre_orient_tol, icon=ft.icons.EXPLORE, tip="Tolerance around dominant plot orientation.")], wrap=True),
                    ft.Row([labeled_row("Width m [min]", pre_wmin, icon=ft.icons.UNFOLD_MORE, tip="Minimum expected plot width (m)."), labeled_row("Width m [max]", pre_wmax, icon=ft.icons.UNFOLD_MORE, tip="Maximum expected plot width (m)."), labeled_row("Length m [min]", pre_lmin, icon=ft.icons.SPACE_BAR, tip="Minimum expected plot length (m)."), labeled_row("Length m [max]", pre_lmax, icon=ft.icons.SPACE_BAR, tip="Maximum expected plot length (m).")], wrap=True),
                ],
            ),
            ft.ExpansionTile(
                title=ft.Row([ft.Icon(ft.icons.IMAGE), ft.Text("Image transforms", weight=ft.FontWeight.W_600)], spacing=8),
                initially_expanded=False,
                controls=[
                    ft.Row([labeled_row("Downscale max (px)", pre_downscale, icon=ft.icons.PHOTO_SIZE_SELECT_LARGE, tip="Max width/height of working raster for detection."), labeled_row("Blue band index", pre_blue_band, icon=ft.icons.PALETTE, tip="1-based band index used as Blue." )], wrap=True),
                    ft.Row([pre_clahe_enable, labeled_row("CLAHE clip", pre_clahe_clip, icon=ft.icons.TONALITY, tip="CLAHE clip limit."), labeled_row("CLAHE tile", pre_clahe_tile, icon=ft.icons.GRID_ON, tip="CLAHE tile size (pixels).")], wrap=True),
                    ft.Row([labeled_row("Texture window", pre_tex_win, icon=ft.icons.TEXTURE, tip="Window size for local standard-deviation texture.")], wrap=True),
                ],
            ),
            ft.ExpansionTile(
                title=ft.Row([ft.Icon(ft.icons.TUNE), ft.Text("Thresholds", weight=ft.FontWeight.W_600)], spacing=8),
                initially_expanded=False,
                controls=[
                    ft.Row([labeled_row("Blue thr relax", pre_thr_b_relax, icon=ft.icons.TUNE, tip="Multiplier < 1.0 lowers Otsu threshold for blue mask."), labeled_row("Tex thr relax", pre_thr_t_relax, icon=ft.icons.TUNE, tip="Multiplier < 1.0 lowers Otsu threshold for texture mask.")], wrap=True),
                ],
            ),
            ft.ExpansionTile(
                title=ft.Row([ft.Icon(ft.icons.CATEGORY), ft.Text("Morphology & detection", weight=ft.FontWeight.W_600)], spacing=8),
                initially_expanded=False,
                controls=[
                    ft.Row([labeled_row("Closing radius", pre_morph_radius, icon=ft.icons.CATEGORY, tip="Binary closing radius (pixels)."), labeled_row("Hole area frac", pre_hole_frac, icon=ft.icons.DONUT_LARGE, tip="Fraction of min region to fill holes."), labeled_row("Approx eps frac", pre_approx_eps, icon=ft.icons.STRAIGHTEN, tip="Perimeter fraction used by polygon simplification.")], wrap=True),
                ],
            ),
            ft.ExpansionTile(
                title=ft.Row([ft.Icon(ft.icons.LAYERS), ft.Text("Post-filtering", weight=ft.FontWeight.W_600)], spacing=8),
                initially_expanded=False,
                controls=[
                    ft.Row([labeled_row("Fill ratio min", pre_fill_min, icon=ft.icons.SPACE_BAR, tip="Contour area / rectangle area must be >= this value."), labeled_row("NMS IoU", pre_nms_iou, icon=ft.icons.LAYERS_CLEAR, tip="Max overlap allowed during non-max suppression."), labeled_row("Shrink buffer (m)", pre_buf_shrink, icon=ft.icons.BORDER_INNER, tip="Inset polygons by this distance (meters).")], wrap=True),
                ],
            ),
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
        clf_mode = ft.RadioGroup(value="existing", content=ft.Row([ft.Radio(value="existing", label="Use existing model"), ft.Radio(value="train", label="Train new model")]))
        model_path = ft.TextField(value="", expand=True, hint_text="Path to .joblib model")
        train_polys = ft.TextField(value="", expand=True, hint_text="Path to training polygons (.shp/.gpkg/.geojson)")
        class_column = ft.Dropdown(options=[], width=240)
        max_pixels_per_class = ft.TextField(value="0", width=100, hint_text="0 = no cap")
        max_pixels_per_polygon = ft.TextField(value="200", width=100)

        def load_fields(_):
            try:
                if train_polys.value.strip() and os.path.exists(train_polys.value.strip()):
                    gdf = gpd.read_file(train_polys.value.strip())
                    cols = [c for c in gdf.columns if c.lower() != "geometry"]
                    class_column.options = [ft.dropdown.Option(c) for c in cols]
                    if cols:
                        class_column.value = cols[0]
                    page.update()
            except Exception:
                pass

        clf = section(
            "Classification",
            ft.Row([ft.Text("Mode", width=210, tooltip="Use an existing model or train a new one."), clf_mode], wrap=False),
            ft.Divider(height=8),
            labeled_row("Model (.joblib)", ft.Row([model_path, ft.OutlinedButton("Browse", icon=ft.icons.UPLOAD_FILE, on_click=lambda _: fp_model.pick_files(allow_multiple=False))], expand=True), icon=ft.icons.SAVE_ALT, tip="Path to saved classifier model (.joblib)."),
            labeled_row("Training polygons", ft.Row([train_polys, ft.OutlinedButton("Browse", icon=ft.icons.FOLDER_OPEN, on_click=lambda _: fp_train.pick_files(allow_multiple=False)), ft.OutlinedButton("Load Fields", icon=ft.icons.TABLE_VIEW, on_click=load_fields)], expand=True), icon=ft.icons.MAP_OUTLINED, tip="Vector dataset of labeled training polygons."),
            ft.Row([labeled_row("Class column", class_column, icon=ft.icons.LIST_ALT, tip="Column name containing class labels."), labeled_row("Max pixels/class", max_pixels_per_class, icon=ft.icons.SPEED, tip="Maximum sampled pixels per class (0 = no cap).")], wrap=True),
            ft.Row([labeled_row("Max pixels/polygon", max_pixels_per_polygon, icon=ft.icons.SPEED, tip="Maximum sampled pixels per polygon.")], wrap=True),
            subtitle="Use an existing model or train a new one, then apply.",
            bgcolor="#FFF4EA",
        )

        # Footer / actions
        pre_btn = ft.ElevatedButton("Run Preselection", icon=ft.icons.SEARCH, on_click=lambda _: run_preselection_only())
        run_btn = ft.ElevatedButton("Run Workflow", icon=ft.icons.PLAY_ARROW, on_click=run_workflow)

        # Two-column responsive layout: each section ~half width
        left_col = ft.Column([left, pre], expand=1, spacing=12)
        right_col = ft.Column([seg, clf], expand=1, spacing=12)

        page.add(
            ft.Column([
                ft.Row([left_col, right_col], expand=True, spacing=12),
                ft.Container(ft.Row([step_text, ft.Container(expand=True), pre_btn, run_btn], alignment=ft.MainAxisAlignment.SPACE_BETWEEN), padding=10),
                ft.Container(progress, padding=8),
                result_text,
            ], expand=True, spacing=12)
        )

    import flet as ft  # type: ignore
    ft.app(target=main)


if __name__ == "__main__":
    run_flet_app()



