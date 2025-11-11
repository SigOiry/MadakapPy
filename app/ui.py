from __future__ import annotations

import os
from pathlib import Path
import time
import threading
import subprocess
from typing import List

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
try:
    import ttkbootstrap as tb  # optional modern theme
except Exception:  # noqa: BLE001
    tb = None

from PIL import Image, ImageTk
import rasterio

from .config import Session, save_session
from .classification import (
    train_model_from_training_polys,
    apply_model_with_pixel_sampling,
)
from .predetection import detect_cultivation_plots, save_preselection_to_output
import geopandas as gpd


 


class ImageSelectorApp(ttk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master, padding=12)
        self.master.title("Madakappy – Segmentation & Classification")
        self.master.minsize(980, 640)

        # Core state
        self.images: List[str] = []
        self.output_dir: str | None = None
        self._seg_gdf: gpd.GeoDataFrame | None = None
        self._last_seg_path: str | None = None
        self._model_path: str | None = None
        self._last_aoi_path: str | None = None
        self._cm_photo = None

        # Segmentation parameters
        self.var_in_raster = tk.StringVar(value=str(Path("Data") / "All_cropped.tif"))
        self.var_tile_size = tk.IntVar(value=40)
        self.var_spatialr = tk.IntVar(value=5)
        self.var_minsize = tk.IntVar(value=5)
        self.var_otb_bin = tk.StringVar(value="OTB/bin/otbcli_LargeScaleMeanShift.bat")

        # Classification parameters
        self.var_model_mode = tk.StringVar(value="existing")  # 'existing' or 'train'
        self.var_model_path = tk.StringVar()
        self.var_train_path = tk.StringVar()
        self.var_class_col = tk.StringVar()
        self.var_max_pixels_per_class = tk.IntVar(value=0)  # 0 = no cap
        self.var_max_pixels_per_polygon = tk.IntVar(value=200)

        # Preselection parameters
        self.var_pre_mask_mode = tk.StringVar(value="and")  # 'and' or 'or'
        self.var_pre_min_region_px = tk.IntVar(value=300)
        self.var_pre_ar_min = tk.DoubleVar(value=1.2)
        self.var_pre_ar_max = tk.DoubleVar(value=12.0)
        self.var_pre_orient_tol = tk.DoubleVar(value=12.0)
        self.var_pre_wmin = tk.DoubleVar(value=5.0)
        self.var_pre_wmax = tk.DoubleVar(value=20.0)
        self.var_pre_lmin = tk.DoubleVar(value=20.0)
        self.var_pre_lmax = tk.DoubleVar(value=60.0)
        self.var_pre_disable_filters = tk.BooleanVar(value=False)

        self._build()

    def _build(self) -> None:
        self.grid(row=0, column=0, sticky="nsew")
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)

        self.columnconfigure(0, weight=1)

        # Main content (two columns, no scrolling)
        content = ttk.Frame(self, padding=(16, 12), style="Content.TFrame")
        content.grid(row=0, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1, uniform="cols")
        content.columnconfigure(1, weight=1, uniform="cols")

        # Header / App bar
        header = ttk.Frame(content, padding=(0, 0, 0, 6), style="Header.TFrame")
        header.grid(row=0, column=0, columnspan=2, sticky="we")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Madakappy", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w")
        btn_bar = ttk.Frame(header, style="Header.TFrame")
        btn_bar.grid(row=0, column=1, sticky="e")
        ttk.Button(btn_bar, text="Settings", command=self._on_settings, width=10).grid(row=0, column=0, padx=(0,6))
        ttk.Button(btn_bar, text="Help", command=self._on_help, width=8).grid(row=0, column=1)
        ttk.Label(header, text="Segmentation and Classification", font=("Segoe UI", 10)).grid(row=1, column=0, sticky="w", pady=(2,0))

        # Stepper
        self.stepper = ttk.Frame(content, padding=(0, 0, 0, 10), style="Header.TFrame")
        self.stepper.grid(row=1, column=0, columnspan=2, sticky="we")
        self._step_labels = [
            ttk.Label(self.stepper, text="1. Segmentation", style="Step.Pending.TLabel"),
            ttk.Label(self.stepper, text="2. Model", style="Step.Pending.TLabel"),
            ttk.Label(self.stepper, text="3. Classify", style="Step.Pending.TLabel"),
        ]
        for i, lab in enumerate(self._step_labels):
            lab.grid(row=0, column=2*i, padx=(0 if i == 0 else 10, 0))
            if i < 2:
                ttk.Label(self.stepper, text="→", style="Step.Separator.TLabel").grid(row=0, column=2*i+1, padx=6)
        self._update_stepper(state=0)

        # Left column
        left = ttk.Frame(content, style="Content.TFrame")
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        left.columnconfigure(0, weight=1)

        io = ttk.LabelFrame(left, text="Project Paths", style="Section.TLabelframe")
        io.grid(row=0, column=0, sticky="we", pady=(0, 10))
        for i in range(3):
            io.columnconfigure(i, weight=1)
        ttk.Label(io, text="Input raster").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(io, textvariable=self.var_in_raster, width=56).grid(row=0, column=1, sticky="we", padx=6, pady=6)
        ttk.Button(io, text="📁 Browse", command=self._on_pick_in_raster).grid(row=0, column=2, padx=6, pady=6)

        ttk.Label(io, text="Output directory").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        self.out_entry = ttk.Entry(io)
        self.out_entry.grid(row=1, column=1, sticky="we", padx=6, pady=6)
        ttk.Button(io, text="📂 Select", command=self._on_pick_output).grid(row=1, column=2, padx=6, pady=6)

        try:
            default_out = str((Path.cwd() / "Output").resolve())
            self.output_dir = default_out
            self.out_entry.delete(0, tk.END)
            self.out_entry.insert(0, default_out)
        except Exception:
            pass

        # Preselection params
        pre = ttk.LabelFrame(left, text="Preselection (AOI detection)", style="Section.TLabelframe")
        pre.grid(row=1, column=0, sticky="we", pady=(0,8))
        for i in range(8):
            pre.columnconfigure(i, weight=1)
        # Disable-all-filters checkbox
        chk = ttk.Checkbutton(pre, text="Disable all filters (raw)", variable=self.var_pre_disable_filters, command=self._update_pre_controls_state)
        chk.grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Label(pre, text="Mask mode").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        rb_and = ttk.Radiobutton(pre, text="AND", variable=self.var_pre_mask_mode, value="and")
        rb_or = ttk.Radiobutton(pre, text="OR", variable=self.var_pre_mask_mode, value="or")
        rb_and.grid(row=0, column=1, sticky="w", padx=6)
        rb_or.grid(row=0, column=2, sticky="w", padx=6)
        ttk.Label(pre, text="Min region (px)").grid(row=0, column=3, sticky="w", padx=8)
        ent_minreg = ttk.Entry(pre, textvariable=self.var_pre_min_region_px, width=10)
        ent_minreg.grid(row=0, column=4, sticky="w", padx=6)
        ttk.Label(pre, text="AR min").grid(row=1, column=0, sticky="w", padx=8)
        ent_armin = ttk.Entry(pre, textvariable=self.var_pre_ar_min, width=8)
        ent_armin.grid(row=1, column=1, sticky="w", padx=6)
        ttk.Label(pre, text="AR max").grid(row=1, column=2, sticky="w", padx=8)
        ent_armax = ttk.Entry(pre, textvariable=self.var_pre_ar_max, width=8)
        ent_armax.grid(row=1, column=3, sticky="w", padx=6)
        ttk.Label(pre, text="Orient tol (deg)").grid(row=1, column=4, sticky="w", padx=8)
        ent_orient = ttk.Entry(pre, textvariable=self.var_pre_orient_tol, width=8)
        ent_orient.grid(row=1, column=5, sticky="w", padx=6)
        ttk.Label(pre, text="Width m [min,max]").grid(row=2, column=0, sticky="w", padx=8)
        ent_wmin = ttk.Entry(pre, textvariable=self.var_pre_wmin, width=8)
        ent_wmax = ttk.Entry(pre, textvariable=self.var_pre_wmax, width=8)
        ent_wmin.grid(row=2, column=1, sticky="w", padx=6)
        ent_wmax.grid(row=2, column=2, sticky="w", padx=6)
        ttk.Label(pre, text="Length m [min,max]").grid(row=2, column=3, sticky="w", padx=8)
        ent_lmin = ttk.Entry(pre, textvariable=self.var_pre_lmin, width=8)
        ent_lmax = ttk.Entry(pre, textvariable=self.var_pre_lmax, width=8)
        ent_lmin.grid(row=2, column=4, sticky="w", padx=6)
        ent_lmax.grid(row=2, column=5, sticky="w", padx=6)

        # Store refs for enabling/disabling
        self._pre_controls = [rb_and, rb_or, ent_minreg, ent_armin, ent_armax, ent_orient, ent_wmin, ent_wmax, ent_lmin, ent_lmax]

        # Segmentation params
        seg = ttk.LabelFrame(left, text="Segmentation (LargeScaleMeanShift)", style="Section.TLabelframe")
        seg.grid(row=2, column=0, sticky="we")
        for i in range(6):
            seg.columnconfigure(i, weight=1)
        ttk.Label(seg, text="Tile size (m)").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(seg, textvariable=self.var_tile_size, width=10).grid(row=0, column=1, sticky="w", padx=6, pady=6)
        ttk.Label(seg, text="Spatialr").grid(row=0, column=2, sticky="w", padx=8, pady=6)
        ttk.Entry(seg, textvariable=self.var_spatialr, width=10).grid(row=0, column=3, sticky="w", padx=6, pady=6)
        ttk.Label(seg, text="Minsize").grid(row=0, column=4, sticky="w", padx=8, pady=6)
        ttk.Entry(seg, textvariable=self.var_minsize, width=10).grid(row=0, column=5, sticky="w", padx=6, pady=6)

        ttk.Label(seg, text="OTB LargeScaleMeanShift path").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(seg, textvariable=self.var_otb_bin, width=56).grid(row=1, column=1, columnspan=4, sticky="we", padx=6, pady=6)
        ttk.Button(seg, text="🔧 Browse", command=self._on_pick_otb).grid(row=1, column=5, padx=6, pady=6)

        # No per-step run button; a single workflow button is provided in footer

        # Right column
        right = ttk.Frame(content, style="ContentAlt.TFrame")
        right.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        right.columnconfigure(0, weight=1)

        clf = ttk.LabelFrame(right, text="Classification", style="SectionAlt.TLabelframe")
        clf.grid(row=0, column=0, sticky="we")
        for i in range(6):
            clf.columnconfigure(i, weight=1)
        ttk.Label(clf, text="Mode").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Radiobutton(clf, text="Use existing model", variable=self.var_model_mode, value="existing", command=self._toggle_model_mode).grid(row=0, column=1, sticky="w", padx=6, pady=4)
        ttk.Radiobutton(clf, text="Train new model", variable=self.var_model_mode, value="train", command=self._toggle_model_mode).grid(row=0, column=2, sticky="w", padx=6, pady=4)

        # Existing model frame
        self.frm_existing = ttk.Frame(clf)
        self.frm_existing.grid(row=1, column=0, columnspan=6, sticky="we")
        self.frm_existing.columnconfigure(1, weight=1)
        ttk.Label(self.frm_existing, text="Model (.joblib)").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(self.frm_existing, textvariable=self.var_model_path).grid(row=0, column=1, sticky="we", padx=6, pady=4)
        ttk.Button(self.frm_existing, text="Browse…", command=self._on_browse_model).grid(row=0, column=2, padx=6, pady=4)

        # Training frame
        self.frm_train = ttk.Frame(clf)
        self.frm_train.grid(row=2, column=0, columnspan=6, sticky="we")
        for i in range(3):
            self.frm_train.columnconfigure(i, weight=1)
        ttk.Label(self.frm_train, text="Training polygons").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(self.frm_train, textvariable=self.var_train_path).grid(row=0, column=1, sticky="we", padx=6, pady=4)
        ttk.Button(self.frm_train, text="📁 Browse", command=self._on_browse_training).grid(row=0, column=2, padx=6, pady=4)
        ttk.Label(self.frm_train, text="Class column").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.cmb_class_col = ttk.Combobox(self.frm_train, textvariable=self.var_class_col, state="readonly")
        self.cmb_class_col.grid(row=1, column=1, sticky="w", padx=6, pady=4)
        ttk.Button(self.frm_train, text="Load Fields", command=self._load_training_columns).grid(row=1, column=2, padx=6, pady=4)
        ttk.Label(self.frm_train, text="Max pixels per class (train)").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(self.frm_train, textvariable=self.var_max_pixels_per_class, width=12).grid(row=2, column=1, sticky="w", padx=6, pady=4)
        ttk.Button(self.frm_train, text="Train Model", command=self._on_train_model).grid(row=2, column=2, padx=6, pady=4)
        self.lbl_training_info = ttk.Label(self.frm_train, text="", foreground="#666")
        self.lbl_training_info.grid(row=3, column=0, columnspan=3, sticky="w", padx=6, pady=4)

        # Confusion matrix preview
        self.cm_label = ttk.Label(right, style="ContentAlt.TLabel")
        self.cm_label.grid(row=1, column=0, sticky="w", padx=6, pady=(8, 0))

        # Apply controls
        applyf = ttk.Frame(right, style="Card.TFrame")
        applyf.grid(row=2, column=0, sticky="we", pady=(8, 0))
        applyf.columnconfigure(1, weight=1)
        ttk.Label(applyf, text="Max pixels per polygon (apply)").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(applyf, textvariable=self.var_max_pixels_per_polygon, width=12).grid(row=0, column=1, sticky="w", padx=6, pady=4)
        # Apply happens as part of full workflow

        # Footer / status
        footer = ttk.Frame(content, padding=(0, 8, 0, 0), style="Footer.TFrame")
        footer.grid(row=2, column=0, columnspan=2, sticky="we")
        footer.columnconfigure(0, weight=1)
        self.status = ttk.Label(footer, text="Ready", foreground="#444")
        self.status.grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(footer, orient=tk.HORIZONTAL, mode="determinate", length=280)
        self.progress.grid(row=0, column=1, sticky="e", padx=(0, 8))
        self.eta = ttk.Label(footer, text="", foreground="#666")
        self.eta.grid(row=0, column=2, sticky="e", padx=(0, 8))
        self.pre_btn = ttk.Button(footer, text="Run Preselection", command=self._on_run_preselection)
        self.pre_btn.grid(row=0, column=3, sticky="e", padx=(0,8))
        self.run_btn = ttk.Button(footer, text="▶ Run Workflow", command=self._on_run_workflow, style="Primary.TButton")
        self.run_btn.grid(row=0, column=4, sticky="e")

        self._toggle_model_mode()
        # Shortcuts
        try:
            self.master.bind('<F5>', lambda e: self._on_run_workflow())
            self.master.bind('<Control-o>', lambda e: self._on_pick_in_raster())
        except Exception:
            pass

    # ─────────── UI helpers ───────────
    def _set_controls_state(self, state: str) -> None:
        for child in self.winfo_children():
            try:
                child.configure(state=state)
            except Exception:
                pass
        try:
            self.run_btn.configure(state=state)
        except Exception:
            pass

    def _update_progress(self, done: int, total: int, start_time: float, note: str = "") -> None:
        self.progress.configure(maximum=max(1, total), value=done)
        elapsed = max(0.001, time.time() - start_time)
        rate = done / elapsed
        remaining = max(0, total - done)
        eta_sec = int(remaining / rate) if rate > 0 else 0
        h, rem = divmod(eta_sec, 3600)
        m, s = divmod(rem, 60)
        self.eta.configure(text=f"{done}/{total} • ETA {h:02d}:{m:02d}:{s:02d}")
        if note:
            self._set_status(note)

    def _set_status(self, text: str) -> None:
        self.status.configure(text=text)
        self._toast(text, kind="info", duration_ms=1800)

    def _toggle_model_mode(self) -> None:
        mode = self.var_model_mode.get()
        if mode == "train":
            self.frm_train.grid()
            self.frm_existing.grid_remove()
            self._update_stepper(state=1)
        else:
            self.frm_existing.grid()
            self.frm_train.grid_remove()
            self._update_stepper(state=1)

    # ─────────── IO pickers ───────────
    def _on_pick_in_raster(self) -> None:
        path = filedialog.askopenfilename(title="Select input raster", filetypes=[
            ("GeoTIFF", "*.tif *.tiff"),
            ("All files", "*.*"),
        ])
        if path:
            self.var_in_raster.set(path)
            self._set_status("Input raster selected")

    def _on_pick_output(self) -> None:
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.output_dir = path
            self.out_entry.delete(0, tk.END)
            self.out_entry.insert(0, path)
            self._set_status("Output directory set")

    def _on_pick_otb(self) -> None:
        path = filedialog.askopenfilename(title="Select OTB LargeScaleMeanShift", filetypes=[
            ("Batch/Executable", "*.bat *.exe"),
            ("All files", "*.*"),
        ])
        if path:
            self.var_otb_bin.set(path)
            self._set_status("OTB path set")

    # ─────────── Segmentation ───────────
    def _on_run_workflow(self) -> None:
        in_raster = self.var_in_raster.get()
        if not in_raster:
            self._invalidate_field(self.out_entry, msg="Select an input raster")
            self._toast("Please select an input raster", kind="error")
            return
        self.output_dir = self.output_dir or self.out_entry.get()
        if not self.output_dir:
            self._invalidate_field(self.out_entry, msg="Select output directory")
            self._toast("Please select an output directory", kind="error")
            return
        if not self.var_otb_bin.get() or not os.path.exists(self.var_otb_bin.get()):
            self._invalidate_field(None, msg="Invalid OTB path")
            self._toast("Set a valid OTB LargeScaleMeanShift path", kind="error")
            return

        session = Session(
            images=self.images,
            output_dir=self.output_dir,
            seg_in_raster=in_raster,
            seg_tile_size_m=int(self.var_tile_size.get()),
            seg_spatialr=int(self.var_spatialr.get()),
            seg_minsize=int(self.var_minsize.get()),
            seg_otb_bin=self.var_otb_bin.get() or None,
        )
        save_session(session, Path(self.output_dir))

        self._set_controls_state("disabled")
        self.progress.configure(value=0, maximum=100)
        self.eta.configure(text="")
        self._set_status("Running workflow…")

        def worker():
            from .segmentation import run_segmentation
            start0 = time.time()

            def make_cb(step: str):
                def cb(done: int, total: int, note: str = ""):
                    msg = f"{step} • {note}" if note else step
                    self.after(0, self._update_progress, done, max(1, total), start0, msg)
                return cb

            # Step 0: preselection (AOI)
            try:
                # Collect optional ranges
                def rng(vmin, vmax):
                    try:
                        a = float(vmin)
                        b = float(vmax)
                        return (a, b) if a > 0 and b > 0 and b >= a else None
                    except Exception:
                        return None
                width_rng = rng(self.var_pre_wmin.get(), self.var_pre_wmax.get())
                length_rng = rng(self.var_pre_lmin.get(), self.var_pre_lmax.get())
                aoi_temp, n_polys = detect_cultivation_plots(
                    raster_path=session.seg_in_raster,
                    output_root=session.output_dir,
                    min_region_px=int(self.var_pre_min_region_px.get()),
                    mask_mode=self.var_pre_mask_mode.get(),
                    disable_filters=bool(self.var_pre_disable_filters.get()),
                    orient_tolerance_deg=float(self.var_pre_orient_tol.get()),
                    ar_min=float(self.var_pre_ar_min.get()),
                    ar_max=float(self.var_pre_ar_max.get()),
                    width_m_range=width_rng,
                    length_m_range=length_rng,
                    progress=make_cb("Preselection"),
                )
                aoi_final = save_preselection_to_output(aoi_temp, session.output_dir)
                self._last_aoi_path = str(aoi_final)
            except Exception as e:
                self.after(0, lambda: self._on_workflow_failed(f"Preselection failed: {e}"))
                return

            # Step 1: segmentation (full image)
            try:
                seg_res = run_segmentation(
                    in_raster=session.seg_in_raster,
                    output_root=session.output_dir,
                    otb_bin=session.seg_otb_bin,
                    tile_size_m=int(session.seg_tile_size_m),
                    spatialr=int(session.seg_spatialr),
                    minsize=int(session.seg_minsize),
                    aoi_path=self._last_aoi_path,
                    progress=make_cb("Segmentation"),
                )
                # Filter segments to AOI
                try:
                    seg_gdf = gpd.read_file(seg_res.out_shp)
                    aoi_gdf = gpd.read_file(self._last_aoi_path) if self._last_aoi_path else None
                    if aoi_gdf is not None and len(aoi_gdf) > 0:
                        if seg_gdf.crs and aoi_gdf.crs and str(seg_gdf.crs) != str(aoi_gdf.crs):
                            aoi_gdf = aoi_gdf.to_crs(seg_gdf.crs)
                        filt = gpd.sjoin(seg_gdf, aoi_gdf, predicate="intersects", how="inner")
                        if "index_right" in filt.columns:
                            filt = filt.drop(columns=["index_right"])
                        # Drop duplicates that may arise from overlapping AOIs
                        filt = filt.loc[~filt.geometry.is_empty].copy()
                        filt = filt.reset_index(drop=True)
                        out_aoi_seg = seg_res.out_dir / f"{seg_res.out_shp.stem}_AOI.shp"
                        filt.to_file(out_aoi_seg)
                        self._last_seg_path = str(out_aoi_seg)
                    else:
                        self._last_seg_path = str(seg_res.out_shp)
                except Exception:
                    # If filtering fails, fall back to full segmentation
                    self._last_seg_path = str(seg_res.out_shp)
            except Exception as e:
                self.after(0, lambda: self._on_workflow_failed(f"Segmentation failed: {e}"))
                return

            # Step 2: get model (existing or train)
            mode = self.var_model_mode.get()
            model_path: str | None = None
            if mode == "existing":
                model_path = self.var_model_path.get() or self._model_path
                if not model_path or not os.path.exists(model_path):
                    # Prompt once
                    def pick_model():
                        path = filedialog.askopenfilename(title="Select model file", filetypes=[("Joblib model", "*.joblib"), ("All files", "*.*")])
                        return path
                    model_path = self._blocking_dialog(pick_model)
                if not model_path:
                    self.after(0, lambda: self._on_workflow_failed("No model selected."))
                    return
                self._model_path = model_path
                self.after(0, lambda: self._load_cm_preview_for_model(model_path))
            else:
                # Train new model
                train_path = self.var_train_path.get()
                if not train_path or not os.path.exists(train_path):
                    def pick_train():
                        return filedialog.askopenfilename(title="Select training polygons", filetypes=[("Vector files", "*.shp *.gpkg *.geojson"), ("All files", "*.*")])
                    train_path = self._blocking_dialog(pick_train)
                    if not train_path:
                        self.after(0, lambda: self._on_workflow_failed("No training polygons provided."))
                        return
                    self.var_train_path.set(train_path)
                # Choose class column if missing
                class_col = self.var_class_col.get()
                if not class_col:
                    try:
                        gdf = gpd.read_file(train_path)
                        cols = [c for c in gdf.columns if c.lower() != "geometry"]
                    except Exception:
                        cols = []
                    if not cols:
                        self.after(0, lambda: self._on_workflow_failed("No non-geometry fields in training data."))
                        return
                    class_col = self._blocking_pick_class_col(cols)
                    if not class_col:
                        self.after(0, lambda: self._on_workflow_failed("No class column selected."))
                        return
                    self.var_class_col.set(class_col)

                cap = 0
                try:
                    cap = int(self.var_max_pixels_per_class.get() or 0)
                except Exception:
                    cap = 0
                try:
                    res = train_model_from_training_polys(
                        session.seg_in_raster,
                        train_path,
                        class_col,
                        session.output_dir,
                        progress=make_cb("Training"),
                        max_pixels_per_class=(cap if cap > 0 else None),
                    )
                    model_path = str(res.model_path)
                    self._model_path = model_path
                    self.after(0, lambda: self._post_training_preview(res))
                except Exception as e:
                    self.after(0, lambda: self._on_workflow_failed(f"Training failed: {e}"))
                    return

            # Step 3: apply
            try:
                cap_pol = 0
                try:
                    cap_pol = int(self.var_max_pixels_per_polygon.get() or 0)
                except Exception:
                    cap_pol = 0
                cap_pol = max(1, cap_pol) if cap_pol else 200
                apply_res = apply_model_with_pixel_sampling(
                    session.seg_in_raster,
                    self._last_seg_path,
                    model_path,
                    session.output_dir,
                    max_pixels_per_polygon=cap_pol,
                    progress=make_cb("Classifying"),
                )
                self.after(0, lambda: self._on_workflow_done(apply_res.output_path))
            except Exception as e:
                self.after(0, lambda: self._on_workflow_failed(f"Classification failed: {e}"))

        threading.Thread(target=worker, daemon=True).start()

    def _on_run_preselection(self) -> None:
        in_raster = self.var_in_raster.get()
        if not in_raster:
            self._invalidate_field(self.out_entry, msg="Select an input raster")
            self._toast("Please select an input raster", kind="error")
            return
        self.output_dir = self.output_dir or self.out_entry.get()
        if not self.output_dir:
            self._invalidate_field(self.out_entry, msg="Select output directory")
            self._toast("Please select an output directory", kind="error")
            return

        self._set_controls_state("disabled")
        self.progress.configure(value=0, maximum=100)
        self.eta.configure(text="")
        self._set_status("Running preselection…")

        def worker():
            start0 = time.time()
            def cb(done: int, total: int, note: str = ""):
                self.after(0, self._update_progress, done, max(1, total), start0, note or "Preselection")
            try:
                aoi_temp, n_polys = detect_cultivation_plots(
                    raster_path=in_raster,
                    output_root=self.output_dir,
                    min_region_px=int(self.var_pre_min_region_px.get()),
                    mask_mode=self.var_pre_mask_mode.get(),
                    disable_filters=bool(self.var_pre_disable_filters.get()),
                    orient_tolerance_deg=float(self.var_pre_orient_tol.get()),
                    ar_min=float(self.var_pre_ar_min.get()),
                    ar_max=float(self.var_pre_ar_max.get()),
                    width_m_range=(
                        (float(self.var_pre_wmin.get()), float(self.var_pre_wmax.get()))
                        if self.var_pre_wmin.get() and self.var_pre_wmax.get() and float(self.var_pre_wmax.get() or 0) >= float(self.var_pre_wmin.get() or 0)
                        else None
                    ),
                    length_m_range=(
                        (float(self.var_pre_lmin.get()), float(self.var_pre_lmax.get()))
                        if self.var_pre_lmin.get() and self.var_pre_lmax.get() and float(self.var_pre_lmax.get() or 0) >= float(self.var_pre_lmin.get() or 0)
                        else None
                    ),
                    progress=cb,
                )
                aoi_final = save_preselection_to_output(aoi_temp, self.output_dir)
                self._last_aoi_path = str(aoi_final)
                def done_ui():
                    self._set_controls_state("normal")
                    self._set_status("Preselection complete")
                    messagebox.showinfo("Preselection", f"Detected {n_polys} plots.\nSaved to:\n{aoi_final}")
                self.after(0, done_ui)
            except Exception as e:
                self.after(0, lambda: self._on_workflow_failed(f"Preselection failed: {e}"))

        threading.Thread(target=worker, daemon=True).start()

    def _update_pre_controls_state(self) -> None:
        disabled = ("disabled" if self.var_pre_disable_filters.get() else "normal")
        for w in getattr(self, "_pre_controls", []):
            try:
                w.configure(state=disabled)
            except Exception:
                pass

    def _blocking_dialog(self, func):
        # Run a filedialog-like function in the main thread and wait for its result
        result_box = {"val": None}
        evt = threading.Event()
        def ask():
            try:
                result_box["val"] = func()
            finally:
                evt.set()
        self.after(0, ask)
        evt.wait()
        return result_box["val"]

    def _blocking_pick_class_col(self, columns: list[str]) -> str | None:
        # Simple modal dialog to pick a class column
        out = {"val": None}
        evt = threading.Event()
        def build():
            dlg = tk.Toplevel(self)
            dlg.title("Select class column")
            dlg.transient(self.master)
            dlg.resizable(False, False)
            ttk.Label(dlg, text="Choose the class column:").grid(row=0, column=0, padx=10, pady=8, sticky="w")
            var = tk.StringVar(value=(columns[0] if columns else ""))
            cmb = ttk.Combobox(dlg, textvariable=var, values=columns, state="readonly", width=36)
            cmb.grid(row=1, column=0, padx=10, pady=6)
            btns = ttk.Frame(dlg)
            btns.grid(row=2, column=0, padx=10, pady=8, sticky="e")
            def ok():
                out["val"] = var.get()
                dlg.destroy(); evt.set()
            def cancel():
                out["val"] = None
                dlg.destroy(); evt.set()
            ttk.Button(btns, text="Cancel", command=cancel).grid(row=0, column=0, padx=(0,6))
            ttk.Button(btns, text="OK", command=ok).grid(row=0, column=1)
            dlg.grab_set(); dlg.focus_force()
        self.after(0, build)
        evt.wait()
        return out["val"]

    def _post_training_preview(self, res) -> None:
        try:
            if res.cm_path and Path(res.cm_path).exists():
                im = Image.open(res.cm_path)
                im.thumbnail((800, 600))
                self._cm_photo = ImageTk.PhotoImage(im)
                self.cm_label.configure(image=self._cm_photo)
        except Exception:
            pass

    def _on_workflow_done(self, out_path: str) -> None:
        self._set_status("Workflow complete")
        self._set_controls_state("normal")
        messagebox.showinfo("Done", f"Classification written to:\n{out_path}")

    def _on_workflow_failed(self, msg: str) -> None:
        self._set_controls_state("normal")
        self._set_status(msg)
        messagebox.showerror("Workflow failed", msg)

    # ─────────── App bar + UX helpers ───────────
    def _on_settings(self) -> None:
        messagebox.showinfo("Settings", "Settings will go here.")

    def _on_help(self) -> None:
        messagebox.showinfo("Help", "1) Set paths 2) Choose mode 3) Run Workflow.")

    def _update_stepper(self, state: int) -> None:
        for i, lab in enumerate(self._step_labels):
            if i == state:
                lab.configure(style="Step.Active.TLabel")
            elif i < state:
                lab.configure(style="Step.Done.TLabel")
            else:
                lab.configure(style="Step.Pending.TLabel")

    def _invalidate_field(self, widget: ttk.Entry | None, msg: str = "") -> None:
        # basic inline validation message via toast; could extend to per-field labels
        if widget is not None:
            try:
                widget.focus_set()
            except Exception:
                pass
        if msg:
            self._toast(msg, kind="error")

    def _toast(self, message: str, kind: str = "info", duration_ms: int = 1500) -> None:
        try:
            toast = tk.Toplevel(self)
            toast.overrideredirect(True)
            toast.attributes("-topmost", True)
            bg = {"info": "#111827", "error": "#7F1D1D"}.get(kind, "#111827")
            fg = "#FFFFFF"
            frame = ttk.Frame(toast)
            frame.pack(fill="both", expand=True)
            lbl = tk.Label(frame, text=message, bg=bg, fg=fg, padx=12, pady=8, font=("Segoe UI", 9))
            lbl.pack()
            # position bottom-right
            self.update_idletasks()
            x = self.winfo_rootx() + self.winfo_width() - 320
            y = self.winfo_rooty() + self.winfo_height() - 80
            toast.geometry(f"300x40+{x}+{y}")
            toast.after(duration_ms, toast.destroy)
        except Exception:
            pass

    def _run_segmentation_worker(self, session: Session) -> None:
        from .segmentation import run_segmentation
        try:
            in_raster = session.seg_in_raster
            if not in_raster or not os.path.exists(in_raster):
                raise RuntimeError("Input raster does not exist")
            if not session.seg_otb_bin or not os.path.exists(session.seg_otb_bin):
                raise RuntimeError("OTB LargeScaleMeanShift not found. Set it above.")

            start = time.time()
            def cb(done: int, total: int, note: str = ""):
                self.after(0, self._update_progress, done, total, start, note)

            res = run_segmentation(
                in_raster=in_raster,
                output_root=session.output_dir,
                otb_bin=session.seg_otb_bin,
                tile_size_m=int(session.seg_tile_size_m),
                spatialr=int(session.seg_spatialr),
                minsize=int(session.seg_minsize),
                progress=cb,
            )

            def done_ui():
                self._update_progress(res.tiles_total, res.tiles_total, start, "Segmentation complete")
                self._set_controls_state("normal")
                messagebox.showinfo("Done", f"Segmentation finished. Outputs in:\n{res.out_dir}")
                self._last_seg_path = str(res.out_shp)
                self._after_segmentation(in_raster, self._last_seg_path)
            self.after(0, done_ui)
        except Exception as e:
            msg = str(e)
            def err_ui():
                self._set_controls_state("normal")
                self._set_status(f"Error: {msg}")
                messagebox.showerror("Segmentation failed", msg)
            self.after(0, err_ui)

    def _after_segmentation(self, raster_path: str, seg_shp: str) -> None:
        try:
            self._seg_gdf = gpd.read_file(seg_shp)
        except Exception as e:
            messagebox.showerror("Load failed", f"Could not read segments: {e}")
            return
        self._last_seg_path = seg_shp
        self._set_status("Ready for training/apply")

    # ─────────── Training / Existing model ───────────
    def _on_browse_training(self) -> None:
        path = filedialog.askopenfilename(title="Select training polygons", filetypes=[
            ("Vector files", "*.shp *.gpkg *.geojson"),
            ("All files", "*.*"),
        ])
        if path:
            self.var_train_path.set(path)
            self._load_training_columns()

    def _load_training_columns(self) -> None:
        path = self.var_train_path.get()
        if not path:
            return
        try:
            gdf = gpd.read_file(path)
            cols = [c for c in gdf.columns if c.lower() != "geometry"]
            self.cmb_class_col.configure(values=cols)
            if cols:
                self.var_class_col.set(cols[0])
            self.lbl_training_info.configure(text=f"Polygons: {len(gdf)} | Fields: {len(cols)}")
        except Exception as e:
            messagebox.showerror("Load failed", f"Could not read training polygons: {e}")

    def _on_train_model(self) -> None:
        if self.var_model_mode.get() != "train":
            return
        train_path = self.var_train_path.get()
        class_col = self.var_class_col.get()
        if not train_path or not class_col:
            messagebox.showwarning("Missing inputs", "Pick training polygons and select class column")
            return
        in_raster = self.var_in_raster.get()
        out_root = self.output_dir or self.out_entry.get()
        self._set_status("Training model…")

        def cb(done: int, total: int, note: str = ""):
            start = getattr(self, "_train_start", None)
            if start is None:
                self._train_start = time.time(); start = self._train_start
            self.after(0, self._update_progress, done, max(1,total), start, note)

        def work():
            try:
                cap = 0
                try:
                    cap = int(self.var_max_pixels_per_class.get() or 0)
                except Exception:
                    cap = 0
                res = train_model_from_training_polys(
                    in_raster,
                    train_path,
                    class_col,
                    out_root,
                    progress=cb,
                    max_pixels_per_class=(cap if cap > 0 else None),
                )
                self._model_path = str(res.model_path)
                def ok():
                    self._set_status("Training complete")
                    # Show confusion matrix
                    try:
                        if res.cm_path and Path(res.cm_path).exists():
                            im = Image.open(res.cm_path)
                            im.thumbnail((800, 600))
                            self._cm_photo = ImageTk.PhotoImage(im)
                            self.cm_label.configure(image=self._cm_photo)
                    except Exception:
                        pass
                    messagebox.showinfo("Model saved", f"Saved → {self._model_path}")
            
                self.after(0, ok)
            except Exception as e:
                msg = str(e)
                self.after(0, lambda: messagebox.showerror("Training failed", msg))
        threading.Thread(target=work, daemon=True).start()

    def _on_browse_model(self) -> None:
        path = filedialog.askopenfilename(title="Select model file", filetypes=[
            ("Joblib model", "*.joblib"),
            ("All files", "*.*"),
        ])
        if path:
            self.var_model_path.set(path)
            self._model_path = path
            self._load_cm_preview_for_model(path)

    def _load_cm_preview_for_model(self, model_path: str) -> None:
        try:
            p = Path(model_path)
            cand = p.with_name(p.stem + "_cm.png")
            if cand.exists():
                im = Image.open(cand)
                im.thumbnail((800, 600))
                self._cm_photo = ImageTk.PhotoImage(im)
                self.cm_label.configure(image=self._cm_photo)
                self._set_status("Loaded confusion matrix")
        except Exception:
            pass

    # ─────────── Apply model ───────────
    def _on_apply_model(self) -> None:
        # Resolve model path: either trained or existing
        if self.var_model_mode.get() == "existing":
            self._model_path = self.var_model_path.get() or self._model_path
        if not self._model_path:
            messagebox.showwarning("No model", "Provide a model (train or select existing)")
            return
        in_raster = self.var_in_raster.get()
        seg_path = self._last_seg_path
        if not seg_path:
            messagebox.showwarning("No segments", "Run segmentation first")
            return
        out_root = self.output_dir or self.out_entry.get()
        self._set_status("Classifying segments…")

        def cb(done: int, total: int, note: str = ""):
            start = getattr(self, "_apply_start", None)
            if start is None:
                self._apply_start = time.time(); start = self._apply_start
            self.after(0, self._update_progress, done, max(1,total), start, note)

        def work():
            try:
                cap = 0
                try:
                    cap = int(self.var_max_pixels_per_polygon.get() or 0)
                except Exception:
                    cap = 0
                cap = max(1, cap) if cap else 200
                res = apply_model_with_pixel_sampling(
                    in_raster,
                    seg_path,
                    self._model_path,
                    out_root,
                    max_pixels_per_polygon=cap,
                    progress=cb,
                )
                def ok():
                    self._set_status("Classification complete")
                    messagebox.showinfo("Done", f"Classification written to:\n{res.output_path}")
                self.after(0, ok)
            except Exception as e:
                msg = str(e)
                self.after(0, lambda: messagebox.showerror("Classification failed", msg))
        threading.Thread(target=work, daemon=True).start()


def run_app() -> None:
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        # Use 'clam' to allow custom backgrounds consistently across platforms
        try:
            style.theme_use("clam")
        except Exception:
            pass
        # Global spacing + fonts
        root.option_add("*Font", ("Segoe UI", 10))
        # Base component spacing
        style.configure("TLabel", padding=(2, 2), background="#FFFFFF")
        style.configure("TButton", padding=(8, 4))
        style.configure("TEntry", padding=(2, 2))
        style.configure("TLabelframe", padding=(10, 8))
        style.configure("TLabelframe.Label", font=("Segoe UI", 11, "bold"))

        # Section backgrounds
        style.configure("Content.TFrame", background="#FFFFFF")
        style.configure("ContentAlt.TFrame", background="#FAFAFA")
        style.configure("Header.TFrame", background="#FFFFFF")
        style.configure("Footer.TFrame", background="#FFFFFF")
        style.configure("Section.TLabelframe", background="#F7F9FC")
        style.configure("Section.TLabelframe.Label", background="#F7F9FC")
        style.configure("SectionAlt.TLabelframe", background="#F8FAFF")
        style.configure("SectionAlt.TLabelframe.Label", background="#F8FAFF")
        style.configure("Card.TFrame", background="#F9FAFB")
        style.configure("ContentAlt.TLabel", background="#FAFAFA")

        # Primary action button
        style.configure(
            "Primary.TButton",
            font=("Segoe UI Semibold", 11),
            padding=(16, 8),
            foreground="#ffffff",
            background="#2563EB",
            bordercolor="#1E40AF",
            focusthickness=3,
            focuscolor="#93C5FD",
        )
        style.map(
            "Primary.TButton",
            background=[("pressed", "#1D4ED8"), ("active", "#3B82F6"), ("!active", "#2563EB")],
            foreground=[("disabled", "#dddddd"), ("!disabled", "#ffffff")],
        )
    except Exception:
        pass
    # Launch full screen (maximized) but allow resizing
    try:
        # On Windows, 'zoomed' opens maximized with window controls
        root.state('zoomed')
    except Exception:
        # Fallback: set to screen size
        try:
            sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
            root.geometry(f"{sw}x{sh}+0+0")
        except Exception:
            pass
    # Ensure true fullscreen is off so user can resize and use system controls
    try:
        root.attributes('-fullscreen', False)
    except Exception:
        pass
    root.resizable(True, True)

    _ = ImageSelectorApp(root)
    root.mainloop()
