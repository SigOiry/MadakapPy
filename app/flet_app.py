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

        # Preselection controls (size-only workflow)
        pre_wmin = ft.TextField(value="5", width=120, text_align=ft.TextAlign.RIGHT)
        pre_wmax = ft.TextField(value="20", width=120, text_align=ft.TextAlign.RIGHT, hint_text="Optional")
        pre_lmin = ft.TextField(value="20", width=120, text_align=ft.TextAlign.RIGHT)
        pre_lmax = ft.TextField(value="60", width=120, text_align=ft.TextAlign.RIGHT, hint_text="Optional")
        pre_small_buffer = ft.TextField(value="0.3", width=140, text_align=ft.TextAlign.RIGHT, hint_text="Buffer (m)")
        pre_quantile = ft.Slider(min=0.05, max=0.9, value=0.15, divisions=17, expand=True)
        pre_quantile_label = ft.Text("", color=pal["muted"])

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

        # Status
        step_text = ft.Text("Ready", size=12, color=pal["muted"]) 
        progress = ft.ProgressBar(value=0.0, color=pal["primary"], bgcolor=pal["border"]) 
        result_text = ft.Text("", selectable=True, color=pal["muted"]) 

        def _field_float(field: ft.TextField, default: float | None = None) -> float | None:
            txt = (field.value or "").strip()
            if not txt:
                return default
            return float(txt)

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
        fp_train = ft.FilePicker()
        page.overlay.extend([fp_raster, dp_output, fp_otb, fp_train])

        # Actions
        def run_preselection_only():
            if not in_raster.value.strip() or not output_dir.value.strip():
                step_text.value = "Please set input raster and output directory"; page.update(); return
            def worker():
                try:
                    url = None
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
                    min_w = _field_float(pre_wmin, 0.0) or 0.0
                    max_w = _field_float(pre_wmax)
                    min_l = _field_float(pre_lmin, 0.0) or 0.0
                    max_l = _field_float(pre_lmax)
                    small_buf = _field_float(pre_small_buffer, 0.3)
                    quant = float(pre_quantile.value or 0.15)
                    aoi_temp, _ = detect_cultivation_plots(
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
                    path = build_preview_html(aoi_final, in_raster.value.strip())
                    if path:
                        page.pubsub.send_all({"kind": "preview", "path": path})

                    # Segmentation
                    seg_res = run_segmentation(
                        in_raster=in_raster.value.strip(),
                        output_root=output_dir.value.strip(),
                        otb_bin=otb_bin.value.strip(),
                        aoi_path=aoi_final,
                        progress=lambda d,t,n: page.pubsub.send_all({"kind":"progress","text": "Segmentation", "ratio": (0 if t==0 else d/max(1,t))}),
                    )
                    final_seg_path = seg_res.out_shp
                    aoi_note = ""
                    if aoi_final and os.path.exists(aoi_final):
                        try:
                            seg_gdf = gpd.read_file(seg_res.out_shp)
                            aoi_gdf = gpd.read_file(aoi_final)
                            if len(seg_gdf) > 0 and len(aoi_gdf) > 0:
                                if seg_gdf.crs and aoi_gdf.crs and str(seg_gdf.crs) != str(aoi_gdf.crs):
                                    aoi_gdf = aoi_gdf.to_crs(seg_gdf.crs)
                                clipped = gpd.sjoin(seg_gdf, aoi_gdf, predicate="intersects", how="inner")
                                if "index_right" in clipped.columns:
                                    clipped = clipped.drop(columns=["index_right"])
                                clipped = clipped.loc[~clipped.geometry.is_empty].reset_index(drop=True)
                                if len(clipped) > 0:
                                    filtered = seg_res.out_dir / f"{seg_res.out_shp.stem}_AOI.shp"
                                    clipped.to_file(filtered)
                                    final_seg_path = filtered
                                    aoi_note = f"\nAOI-filtered segments: {len(clipped)}"
                                else:
                                    aoi_note = "\nAOI filter returned zero segments."
                        except Exception as aoi_err:
                            aoi_note = f"\nAOI filter failed: {aoi_err}"
                    page.pubsub.send_all({"kind":"result","text": f"Segmentation done. Outputs in: {seg_res.out_dir}\nFinal segments: {final_seg_path}{aoi_note}"})
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
        clf_mode = ft.RadioGroup(
            value="existing",
            content=ft.Row(
                [
                    ft.Radio(value="existing", label="Use existing model"),
                    ft.Radio(value="train", label="Train new model"),
                ]
            ),
        )
        model_path = ft.Dropdown(
            options=[],
            expand=True,
            hint_text="Select a model from the 'Model' directory",
        )
        model_hint = ft.Text("", size=12, color=pal["muted"])
        max_pixels_per_polygon = ft.TextField(value="200", width=120, hint_text="Pixels/polygon", text_align=ft.TextAlign.RIGHT)
        train_polys = ft.TextField(value="", expand=True, hint_text="Path to training polygons (.shp/.gpkg/.geojson)")
        class_column = ft.Dropdown(options=[], width=240)
        max_pixels_per_class = ft.TextField(value="0", width=100, hint_text="0 = no cap")

        def on_pick_train(e: ft.FilePickerResultEvent):
            try:
                if e.files:
                    train_polys.value = e.files[0].path
                    train_polys.update()
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
                    options = [ft.dropdown.Option(fp.name, key=str(fp.resolve())) for fp in files]
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
            model_hint.value = note
            model_hint.update()

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

        existing_controls = ft.Column(
            [
                labeled_row(
                    "Model (.joblib)",
                    ft.Row(
                        [
                            model_path,
                            ft.IconButton(icon=ft.icons.REFRESH, tooltip="Rescan Model directory", on_click=refresh_model_list),
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
            ],
            spacing=8,
            visible=True,
        )

        train_controls = ft.Column(
            [
                labeled_row(
                    "Training polygons",
                    ft.Row(
                        [
                            train_polys,
                            ft.OutlinedButton("Browse", icon=ft.icons.FOLDER_OPEN, on_click=lambda _: fp_train.pick_files(allow_multiple=False)),
                            ft.OutlinedButton("Load Fields", icon=ft.icons.TABLE_VIEW, on_click=load_fields),
                        ],
                        spacing=6,
                        expand=True,
                    ),
                    icon=ft.icons.MAP_OUTLINED,
                    tip="Vector dataset of labeled training polygons.",
                ),
                ft.Row(
                    [
                        labeled_row(
                            "Class column",
                            class_column,
                            icon=ft.icons.LIST_ALT,
                            tip="Column containing class labels.",
                        ),
                        labeled_row(
                            "Max pixels/class",
                            max_pixels_per_class,
                            icon=ft.icons.SPEED,
                            tip="Maximum sampled pixels per class (0 = no cap).",
                        ),
                    ],
                    wrap=True,
                ),
            ],
            spacing=10,
            visible=False,
        )

        def on_clf_mode_change(_=None):
            use_existing = (clf_mode.value or "existing") == "existing"
            existing_controls.visible = use_existing
            train_controls.visible = not use_existing
            page.update()

        clf_mode.on_change = on_clf_mode_change
        clf = section(
            "Classification",
            ft.Row([ft.Text("Mode", width=210, tooltip="Use an existing model or train a new one."), clf_mode], wrap=False),
            ft.Divider(height=8),
            existing_controls,
            train_controls,
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

        refresh_model_list()
        on_clf_mode_change()

    import flet as ft  # type: ignore
    ft.app(target=main)


if __name__ == "__main__":
    run_flet_app()



