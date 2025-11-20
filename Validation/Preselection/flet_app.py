from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional


def run_flet_app() -> None:
    """Entry to launch the Flet UI. Import flet lazily to avoid hard dependency."""
    try:
        import flet as ft
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Flet is not installed. Install with 'pip install flet' and rerun."
        ) from e

    from .segmentation import run_segmentation
    from .predetection import run_predetection
    from .classification import (
        train_model_from_training_polys,
        apply_model_with_pixel_sampling,
    )
    import geopandas as gpd  # used to read columns for training polygons

    # ───────────────────────── UI helpers ─────────────────────────
    def _palette():
        return {
            "primary": "#2563EB",
            "primary_hover": "#3B82F6",
            "surface": "#FFFFFF",
            "card": "#F7F9FC",
            "alt": "#F8FAFF",
            "border": "#E5E7EB",
            "muted": "#6B7280",
        }

    def section(title: str, *children: ft.Control, icon: str | None = None, subtitle: str | None = None, bgcolor: str | None = None) -> ft.Card:
        pal = _palette()
        header = ft.Row([
            ft.Icon(icon, color=pal["primary"]) if icon else ft.Container(),
            ft.Text(title, size=16, weight=ft.FontWeight.W_600),
        ], spacing=8, alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        desc = ft.Text(subtitle or "", size=12, color=pal["muted"]) if subtitle else ft.Container(height=0)
        return ft.Card(
            content=ft.Container(
                content=ft.Column([
                    header,
                    desc,
                    *children,
                ], spacing=12),
                padding=16,
                bgcolor=bgcolor or pal["card"],
                border_radius=12,
                border=ft.border.all(1, pal["border"]),
            ),
            elevation=2,
        )

    def labeled_row(label: str, field: ft.Control, icon: str | None = None) -> ft.Row:
        pal = _palette()
        lbl = ft.Row([
            ft.Icon(icon, size=16, color=pal["muted"]) if icon else ft.Container(width=0),
            ft.Text(label),
        ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        return ft.Row(
            controls=[ft.Container(lbl, width=210), ft.Container(field, expand=True)],
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

    # ───────────────────────── App main ─────────────────────────
    def main(page: ft.Page):
        page.title = "Madakappy – Segmentation & Classification"
        page.theme_mode = ft.ThemeMode.LIGHT
        try:
            page.theme = ft.Theme(color_scheme_seed=_palette()["primary"])
        except Exception:
            pass
        page.window_maximized = True
        page.padding = 16

        # Shared state with defaults
        default_in = str((Path("Data") / "All_cropped.tif").resolve())
        default_out = str((Path.cwd() / "Output").resolve())
        default_otb = str(Path("OTB") / "bin" / "otbcli_LargeScaleMeanShift.bat")

        in_raster = ft.TextField(value=default_in, hint_text="Path to input GeoTIFF", expand=True)
        output_dir = ft.TextField(value=default_out, hint_text="Path to output directory", expand=True)
        otb_bin = ft.TextField(value=default_otb, hint_text="Path to OTB LargeScaleMeanShift (.bat/.exe)", expand=True)
        pre_detect = ft.Checkbox(value=True, label="Pre‑detect cultivation plots (recommended)")
        btn_predetect = ft.FilledButton(
            "Run pre-detection",
            icon=ft.icons.PLAY_CIRCLE,
            on_click=lambda e: run_predetect_action(e),
        )

        tile_size = ft.TextField(value="40", width=120)
        spatialr = ft.TextField(value="5", width=120)
        minsize = ft.TextField(value="5", width=120)

        last_pred_path: Optional[str] = None
        # Mode selector: prefer SegmentedButton if available, else RadioGroup
        if hasattr(ft, "SegmentedButton"):
            mode = ft.SegmentedButton(
                segments=[
                    ft.Segment(value="existing", label=ft.Text("Use existing model")),
                    ft.Segment(value="train", label=ft.Text("Train new model")),
                ],
                selected={"existing"},
                on_change=lambda e: update_mode_visibility(),
            )
            def get_mode() -> str:
                return next(iter(mode.selected)) if getattr(mode, "selected", None) else "existing"
        else:
            mode = ft.RadioGroup(
                value="existing",
                content=ft.Row([
                    ft.Radio(value="existing", label="Use existing model"),
                    ft.Radio(value="train", label="Train new model"),
                ]),
                on_change=lambda e: update_mode_visibility(),
            )
            def get_mode() -> str:
                return getattr(mode, "value", "existing") or "existing"

        model_path = ft.TextField(value="", hint_text="Path to .joblib model", expand=True)
        # Training controls
        train_polys = ft.TextField(value="", hint_text="Path to training polygons (SHP/GPKG/GeoJSON)", expand=True)
        class_column = ft.Dropdown(options=[], width=240)
        max_pix_per_class = ft.TextField(value="0", width=140, tooltip="0 = use all")

        # Apply controls
        max_pix_per_polygon = ft.TextField(value="200", width=140)

        # Progress and output
        step_text = ft.Text("Ready", size=12, color="#374151")
        progress = ft.ProgressBar(width=None, value=0.0)
        cm_image = ft.Image(src=None, width=540, height=400, fit=ft.ImageFit.CONTAIN, visible=False)
        result_text = ft.Text("", selectable=True)

        # File pickers with reactive updates
        def on_pick_raster(e: ft.FilePickerResultEvent):
            try:
                if e.files and len(e.files) > 0:
                    in_raster.value = e.files[0].path
                    in_raster.update()
                    step_text.value = "Selected input raster"
                    step_text.update()
            except Exception:
                pass

        def on_pick_model(e: ft.FilePickerResultEvent):
            try:
                if e.files and len(e.files) > 0:
                    model_path.value = e.files[0].path
                    model_path.update()
                    step_text.value = "Selected model"
                    step_text.update()
            except Exception:
                pass

        def on_pick_train(e: ft.FilePickerResultEvent):
            try:
                if e.files and len(e.files) > 0:
                    train_polys.value = e.files[0].path
                    train_polys.update()
                    step_text.value = "Selected training polygons"
                    step_text.update()
            except Exception:
                pass

        def on_pick_output(e: ft.FilePickerResultEvent):
            try:
                p = getattr(e, "path", None)
                if p:
                    output_dir.value = p
                    output_dir.update()
                    step_text.value = "Selected output directory"
                    step_text.update()
            except Exception:
                pass

        def on_pick_otb(e: ft.FilePickerResultEvent):
            try:
                if e.files and len(e.files) > 0:
                    otb_bin.value = e.files[0].path
                    otb_bin.update()
                    step_text.value = "Selected OTB path"
                    step_text.update()
            except Exception:
                pass

        fp_raster = ft.FilePicker(on_result=on_pick_raster)
        fp_model = ft.FilePicker(on_result=on_pick_model)
        fp_train = ft.FilePicker(on_result=on_pick_train)
        dp_output = ft.FilePicker(on_result=on_pick_output)
        fp_otb = ft.FilePicker(on_result=on_pick_otb)
        page.overlay.extend([fp_raster, fp_model, fp_train, dp_output, fp_otb])

        # UI-thread message handler via pubsub
        def on_msg(msg: dict):
            try:
                kind = msg.get("kind")
                if kind == "progress":
                    step_text.value = msg.get("text", "")
                    ratio = msg.get("ratio")
                    if ratio is not None:
                        progress.value = max(0.0, min(1.0, float(ratio)))
                elif kind == "status":
                    step_text.value = msg.get("text", "")
                elif kind == "cm":
                    p = msg.get("path")
                    if p:
                        step_text.value = f"Confusion matrix saved: {p}"
                elif kind == "clear_cm":
                    cm_image.visible = False
                    cm_image.src = None
                    cm_image.src_base64 = None
                elif kind == "result":
                    result_text.value = msg.get("text", "")
                elif kind == "error":
                    step_text.value = f"Error: {msg.get('text','')}"
                elif kind == "preview_map":
                    p = msg.get("path")
                    if p:
                        try:
                            page.launch_url("file:///" + str(Path(p).resolve()).replace("\\", "/"))
                            step_text.value = "Opened classification preview map"
                        except Exception:
                            step_text.value = f"Preview saved to: {p}"
                page.update()
            except Exception:
                pass

        page.pubsub.subscribe(on_msg)

        # Load training columns
        def on_load_fields(_):
            p = train_polys.value.strip()
            if not p or not os.path.exists(p):
                step_text.value = "Training file not found"
                page.update();
                return
            try:
                gdf = gpd.read_file(p)
                cols = [c for c in gdf.columns if c.lower() != "geometry"]
                class_column.options = [ft.dropdown.Option(c) for c in cols]
                if cols:
                    class_column.value = cols[0]
                step_text.value = f"Loaded fields ({len(cols)})"
            except Exception as e:  # noqa: BLE001
                step_text.value = f"Failed to read fields: {e}"
            page.update()

        def run_predetect_action(_):
            nonlocal last_pred_path
            if not in_raster.value.strip():
                step_text.value = "Please select an input raster"
                page.update()
                return
            if not output_dir.value.strip():
                step_text.value = "Please select an output directory"
                page.update()
                return

            page.pubsub.send_all({"kind": "progress", "text": "Pre-detecting plots", "ratio": 0.0})

            def worker_pd():
                nonlocal last_pred_path
                try:
                    def cb_pd(done: int, total: int, note: str = ""):
                        ratio = 0 if total == 0 else done / max(1, total)
                        page.pubsub.send_all({
                            "kind": "progress",
                            "text": note or "Pre-detecting plots",
                            "ratio": ratio,
                        })

                    res = run_predetection(
                        in_raster=in_raster.value.strip(),
                        output_root=output_dir.value.strip(),
                        progress=cb_pd,
                    )
                    last_pred_path = str(res.vector_path)
                    page.pubsub.send_all({
                        "kind": "status",
                        "text": f"Pre-detection complete ({res.plot_count} plots)",
                    })
                    page.pubsub.send_all({
                        "kind": "result",
                        "text": f"Pre-detection output: {res.vector_path}\nSummary: {res.summary_path}",
                    })
                except Exception as exc:  # noqa: BLE001
                    page.pubsub.send_all({
                        "kind": "error",
                        "text": f"Pre-detection failed: {exc}",
                    })

            threading.Thread(target=worker_pd, daemon=True).start()

        def run_workflow(_):
            # Validate inputs quickly
            if not in_raster.value.strip():
                step_text.value = "Please select an input raster"; page.update(); return
            if not output_dir.value.strip():
                step_text.value = "Please select an output directory"; page.update(); return
            if not otb_bin.value.strip() or not os.path.exists(otb_bin.value.strip()):
                step_text.value = "Set a valid OTB LargeScaleMeanShift path"; page.update(); return

            # Reset output UI (from UI thread)
            page.pubsub.send_all({"kind": "clear_cm"})
            page.pubsub.send_all({"kind": "result", "text": ""})

            def worker():
                nonlocal last_pred_path
                try:
                    # Segmentation step
                    def cb(done: int, total: int, note: str = ""):
                        ratio = 0 if total == 0 else done / max(1, total)
                        page.pubsub.send_all({"kind": "progress", "text": f"Segmentation • {note}", "ratio": ratio})

                    # Optional pre‑detection of plots
                    aoi_for_run = None
                    if pre_detect.value:
                        try:
                            def cb_pd(done: int, total: int, note: str = ""):
                                ratio = 0 if total == 0 else done / max(1, total)
                                page.pubsub.send_all({"kind": "progress", "text": note or "Pre‑detecting plots", "ratio": ratio})

                            res_pd = run_predetection(
                                in_raster=in_raster.value.strip(),
                                output_root=output_dir.value.strip(),
                                progress=cb_pd,
                            )
                            last_pred_path = str(res_pd.vector_path)
                            page.pubsub.send_all({
                                "kind": "result",
                                "text": f"Pre-detection output: {res_pd.vector_path}\nSummary: {res_pd.summary_path}",
                            })
                            if res_pd.plot_count > 0:
                                aoi_for_run = last_pred_path
                                page.pubsub.send_all({"kind": "status", "text": f"Using {res_pd.plot_count} pre‑detected plots"})
                            else:
                                page.pubsub.send_all({"kind": "status", "text": "Pre‑detection produced no plots; using full extent"})
                        except Exception as e:
                            page.pubsub.send_all({"kind": "status", "text": f"Pre‑detection skipped: {e}"})

                    page.pubsub.send_all({"kind": "progress", "text": "Segmentation • starting", "ratio": 0.0})
                    seg_res = run_segmentation(
                        in_raster=in_raster.value.strip(),
                        output_root=output_dir.value.strip(),
                        otb_bin=otb_bin.value.strip(),
                        tile_size_m=int(tile_size.value or 40),
                        spatialr=int(spatialr.value or 5),
                        minsize=int(minsize.value or 5),
                        aoi_path=aoi_for_run,
                        aoi_min_coverage=0.05,
                        clip_to_aoi=True,
                        progress=cb,
                    )
                    page.pubsub.send_all({"kind": "progress", "text": "Segmentation • done", "ratio": 1.0})

                    # Model step
                    selected_mode = get_mode()
                    mdl_path: Optional[str] = None
                    if selected_mode == "existing":
                        mp = model_path.value.strip()
                        if not mp or not os.path.exists(mp):
                            page.pubsub.send_all({"kind": "error", "text": "No model selected"})
                            return
                        mdl_path = mp
                        # Try to display CM image
                        cm_path = Path(mp).with_name(Path(mp).stem + "_cm.png")
                        if cm_path.exists():
                            page.pubsub.send_all({"kind": "cm", "path": str(cm_path)})
                    else:
                        # Train model
                        tp = train_polys.value.strip()
                        if not tp or not os.path.exists(tp):
                            page.pubsub.send_all({"kind": "error", "text": "Training polygons missing"})
                            return
                        cc = (class_column.value or "").strip()
                        if not cc:
                            page.pubsub.send_all({"kind": "error", "text": "Select class column"})
                            return
                        mppc = int(max_pix_per_class.value or 0) or None

                        def cb_train(done: int, total: int, note: str = ""):
                            ratio = 0 if total == 0 else done / max(1, total)
                            page.pubsub.send_all({"kind": "progress", "text": f"Training • {note}", "ratio": ratio})

                        page.pubsub.send_all({"kind": "progress", "text": "Training • starting", "ratio": 0.0})
                        tres = train_model_from_training_polys(
                            raster_path=in_raster.value.strip(),
                            training_polys_path=tp,
                            class_column=cc,
                            output_root=output_dir.value.strip(),
                            progress=cb_train,
                            max_pixels_per_class=mppc,
                        )
                        mdl_path = str(tres.model_path)
                        # Show CM image
                        if tres.cm_path and Path(tres.cm_path).exists():
                            page.pubsub.send_all({"kind": "cm", "path": str(tres.cm_path)})
                        page.pubsub.send_all({"kind": "progress", "text": "Training • done", "ratio": 1.0})

                    # Apply step
                    mpp = max(1, int(max_pix_per_polygon.value or 200))

                    def cb_apply(done: int, total: int, note: str = ""):
                        ratio = 0 if total == 0 else done / max(1, total)
                        page.pubsub.send_all({"kind": "progress", "text": f"Classifying • {note}", "ratio": ratio})

                    page.pubsub.send_all({"kind": "progress", "text": "Classifying • starting", "ratio": 0.0})
                    ares = apply_model_with_pixel_sampling(
                        raster_path=in_raster.value.strip(),
                        segments_path=str(seg_res.out_shp),
                        model_path=mdl_path,
                        output_root=output_dir.value.strip(),
                        max_pixels_per_polygon=mpp,
                        progress=cb_apply,
                        generate_preview=True,
                        biomass_model="madagascar",
                    )
                    page.pubsub.send_all({"kind": "progress", "text": "Workflow complete", "ratio": 1.0})
                    page.pubsub.send_all({"kind": "result", "text": f"Output: {ares.output_path}"})
                    if ares.preview_map:
                        page.pubsub.send_all({"kind": "preview_map", "path": str(ares.preview_map)})
                except Exception as ex:  # noqa: BLE001
                    page.pubsub.send_all({"kind": "error", "text": str(ex)})

            threading.Thread(target=worker, daemon=True).start()
        left = section(
            "Project Paths",
            labeled_row("Input raster", ft.Row([in_raster, ft.OutlinedButton("Browse", icon=ft.icons.FOLDER_OPEN, on_click=lambda _: fp_raster.pick_files(allow_multiple=False))], expand=True), icon=ft.icons.IMAGE_OUTLINED),
            labeled_row("Output directory", ft.Row([output_dir, ft.OutlinedButton("Browse", icon=ft.icons.FOLDER, on_click=lambda _: dp_output.get_directory_path())], expand=True), icon=ft.icons.FOLDER),
            labeled_row("OTB path", ft.Row([otb_bin, ft.OutlinedButton("Browse", icon=ft.icons.BUILD_OUTLINED, on_click=lambda _: fp_otb.pick_files(allow_multiple=False))], expand=True), icon=ft.icons.BUILD_OUTLINED),
            labeled_row(
                "Plot pre‑detection",
                ft.Row([pre_detect, btn_predetect], spacing=12),
                icon=ft.icons.DETAILS,
            ),
            icon=ft.icons.FOLDER_OPEN,
            subtitle="Select input data and where to write outputs.",
            bgcolor=_palette()["card"],
        )
        seg = section(
            "Segmentation",
            ft.Row([
                labeled_row("Tile size (m)", tile_size, icon=ft.icons.STRAIGHTEN),
                labeled_row("Spatialr", spatialr, icon=ft.icons.GRAIN),
                labeled_row("Minsize", minsize, icon=ft.icons.DETAILS),
            ], wrap=True, spacing=10),
            icon=ft.icons.TUNE,
            subtitle="Tune LargeScaleMeanShift parameters.",
            bgcolor=_palette()["card"],
        )
        left_col = ft.Column([left, seg], spacing=12, expand=True)

        # Existing model and training sections with visibility toggling
        existing_container = ft.Container(
            content=labeled_row(
                "Model (.joblib)",
                ft.Row([model_path, ft.OutlinedButton("Browse", icon=ft.icons.UPLOAD_FILE, on_click=lambda _: fp_model.pick_files(allow_multiple=False))], expand=True),
                icon=ft.icons.SAVE_ALT,
            ),
            visible=True,
        )
        load_fields_btn = ft.OutlinedButton("Load Fields from Training", icon=ft.icons.TABLE_VIEW, on_click=on_load_fields)
        training_container = ft.Container(
            content=ft.Column([
                labeled_row("Training polygons", ft.Row([train_polys, ft.OutlinedButton("Browse", icon=ft.icons.FOLDER_OPEN, on_click=lambda _: fp_train.pick_files(allow_multiple=False))], expand=True), icon=ft.icons.MAP_OUTLINED),
                labeled_row("Class column", class_column, icon=ft.icons.LIST_ALT),
                labeled_row("Max pixels per class", max_pix_per_class, icon=ft.icons.SPEED),
                load_fields_btn,
            ], spacing=8),
            visible=False,
        )
        clf = section(
            "Classification",
            ft.Row([ft.Text("Mode", width=210), mode]),
            ft.Container(height=6),
            existing_container,
            ft.Divider(height=10, color=_palette()["border"]),
            training_container,
            icon=ft.icons.ASSIGNMENT,
            subtitle="Use an existing model or train a new one.",
            bgcolor=_palette()["alt"],
        )
        apply = section(
            "Apply",
            labeled_row("Max pixels per polygon", max_pix_per_polygon, icon=ft.icons.SPEED),
            icon=ft.icons.PLAYLIST_ADD_CHECK,
            subtitle="Subsample pixels per polygon and compute proportions.",
            bgcolor=_palette()["card"],
        )
        right_col = ft.Column([clf, apply], spacing=12, expand=True)

        # Footer with run button and progress
        run_btn = ft.ElevatedButton("Run Workflow", icon=ft.icons.PLAY_ARROW, style=ft.ButtonStyle(
            color={ft.MaterialState.DEFAULT: ft.colors.WHITE},
            bgcolor={ft.MaterialState.DEFAULT: "#2563EB", ft.MaterialState.HOVERED: "#3B82F6"},
            padding=20,
        ), on_click=run_workflow)

        def update_mode_visibility():
            is_train = (get_mode() == "train")
            training_container.visible = is_train
            existing_container.visible = not is_train
            page.update()

        # Initialize visibility
        update_mode_visibility()

        footer = ft.Container(
            ft.Row([step_text, ft.Container(expand=True), run_btn], vertical_alignment=ft.CrossAxisAlignment.CENTER),
            padding=12,
            bgcolor=_palette()["surface"],
            border=ft.border.all(1, _palette()["border"]),
            border_radius=12,
        )

        page.add(
            ft.Row([
                ft.Column([left_col], expand=1),
                ft.Column([right_col], expand=1),
            ], expand=True),
            footer,
            ft.Container(progress, padding=8),
        )

    # Launch app
    import flet as ft  # type: ignore  # reimport for ft.app

    ft.app(target=main)


if __name__ == "__main__":
    run_flet_app()
