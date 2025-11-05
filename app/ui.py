from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk
import numpy as np
import rasterio
from rasterio.enums import Resampling

from .config import Session, save_session


IMAGE_EXTS = (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp")


class ImageSelectorApp(ttk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master, padding=10)
        self.master.title("Madakappy – Image Selection")
        self.master.minsize(900, 520)

        self.images: List[str] = []
        self.output_dir: str | None = None

        self._thumb_cache: dict[str, ImageTk.PhotoImage] = {}

        self._build()

    def _build(self) -> None:
        self.grid(row=0, column=0, sticky="nsew")

        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(2, weight=1)

        # Header
        title = ttk.Label(self, text="Select input images and output directory", font=("Segoe UI", 12, "bold"))
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        # Controls left side
        controls = ttk.Frame(self)
        controls.grid(row=1, column=0, sticky="we", padx=(0, 10))
        controls.columnconfigure(0, weight=1)

        add_btn = ttk.Button(controls, text="Add Image…", command=self._on_add)
        add_btn.grid(row=0, column=0, sticky="w")
        rm_btn = ttk.Button(controls, text="Remove Selected", command=self._on_remove)
        rm_btn.grid(row=0, column=1, sticky="w", padx=(8, 0))
        clear_btn = ttk.Button(controls, text="Clear", command=self._on_clear)
        clear_btn.grid(row=0, column=2, sticky="w", padx=(8, 0))

        out_btn = ttk.Button(controls, text="Select Output Directory…", command=self._on_pick_output)
        out_btn.grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))

        self.out_label = ttk.Label(controls, text="No output directory selected", foreground="#666")
        self.out_label.grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))

        # List of images
        list_frame = ttk.LabelFrame(self, text="Selected image")
        list_frame.grid(row=2, column=0, sticky="nsew", padx=(0, 10))
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        self.listbox = tk.Listbox(list_frame, selectmode=tk.SINGLE, activestyle="none")
        self.listbox.grid(row=0, column=0, sticky="nsew")

        sb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        # Preview pane
        preview = ttk.LabelFrame(self, text="Preview")
        preview.grid(row=2, column=1, sticky="nsew")
        preview.rowconfigure(0, weight=1)
        preview.columnconfigure(0, weight=1)

        self.preview_lbl = ttk.Label(preview, anchor="center")
        self.preview_lbl.grid(row=0, column=0, sticky="nsew")

        # Footer
        footer = ttk.Frame(self)
        footer.grid(row=3, column=0, columnspan=2, sticky="we", pady=(8, 0))
        footer.columnconfigure(0, weight=1)

        self.status = ttk.Label(footer, text="Ready", foreground="#666")
        self.status.grid(row=0, column=0, sticky="w")

        proceed = ttk.Button(footer, text="Save Selection", command=self._on_save)
        proceed.grid(row=0, column=1, sticky="e")

    def _on_add(self) -> None:
        filetypes = [
            ("Image files", "*.tif *.tiff *.jpg *.jpeg *.png *.bmp"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(title="Select image", filetypes=filetypes)
        if not path:
            return
        if Path(path).suffix.lower() not in IMAGE_EXTS:
            self._set_status("Unsupported file type")
            return
        # Enforce single selection: replace current list
        self.images = [path]
        self.listbox.delete(0, tk.END)
        self.listbox.insert(tk.END, os.path.basename(path))
        self.listbox.selection_set(0)
        self._set_status("Image selected")
        # Auto-preview
        self._show_preview(path)

    def _on_remove(self) -> None:
        sel = list(self.listbox.curselection())
        if not sel:
            return
        sel.sort(reverse=True)
        for idx in sel:
            self.listbox.delete(idx)
            del self.images[idx]
        self._clear_preview()
        self._set_status("Removed selected item(s)")

    def _on_clear(self) -> None:
        self.images.clear()
        self.listbox.delete(0, tk.END)
        self._clear_preview()
        self._set_status("Cleared list")

    def _on_pick_output(self) -> None:
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.output_dir = path
            self.out_label.configure(text=path, foreground="#000")
            self._set_status("Output directory set")

    def _on_select(self, _evt=None) -> None:
        sel = self.listbox.curselection()
        if not sel:
            self._clear_preview()
            return
        idx = sel[0]
        img_path = self.images[idx]
        self._show_preview(img_path)

    def _clear_preview(self) -> None:
        self.preview_lbl.configure(image="", text="No preview")

    def _show_preview(self, path: str) -> None:
        try:
            if path in self._thumb_cache:
                self.preview_lbl.configure(image=self._thumb_cache[path], text="")
                return

            suffix = Path(path).suffix.lower()
            if suffix in (".tif", ".tiff"):
                im = render_geotiff_thumbnail(path, size=600)
            else:
                im = Image.open(path)
                im.thumbnail((600, 600))

            photo = ImageTk.PhotoImage(im)
            self._thumb_cache[path] = photo
            self.preview_lbl.configure(image=photo, text="")
        except Exception as e:
            self.preview_lbl.configure(text=f"Preview unavailable\n{e}")

    def _on_save(self) -> None:
        if not self.images:
            messagebox.showwarning("Nothing to save", "Please add at least one image.")
            return
        if not self.output_dir:
            messagebox.showwarning("No output directory", "Please select an output directory.")
            return
        session = Session(images=self.images, output_dir=self.output_dir)
        out = save_session(session, Path(self.output_dir))
        self._set_status(f"Saved selection → {out}")
        messagebox.showinfo("Saved", f"Selection saved to:\n{out}")

    def _set_status(self, text: str) -> None:
        self.status.configure(text=text)


def run_app() -> None:
    root = tk.Tk()
    try:
        root.iconbitmap(default='')  # safe no-op on most platforms
    except Exception:
        pass
    style = ttk.Style()
    # Use a modern theme if available
    for theme in ("vista", "clam", "default"):
        try:
            style.theme_use(theme)
            break
        except Exception:
            continue
    app = ImageSelectorApp(root)
    root.mainloop()


def _percentile_stretch(arr: np.ndarray, lower: float = 2, upper: float = 98) -> np.ndarray:
    out = np.empty_like(arr, dtype=np.float32)
    for i in range(arr.shape[0]):
        band = arr[i]
        # Compute robust min/max per band ignoring NaNs
        flat = band[np.isfinite(band)]
        if flat.size == 0:
            out[i] = 0
            continue
        p_low = np.percentile(flat, lower)
        p_high = np.percentile(flat, upper)
        if p_high == p_low:
            p_high = p_low + 1e-6
        scaled = (band - p_low) / (p_high - p_low)
        out[i] = np.clip(scaled, 0, 1)
    return (out * 255).astype(np.uint8)


def _choose_rgb_indexes(count: int) -> Tuple[int, int, int]:
    # Default mapping for your 5-band: 1:Blue, 2:Green, 3:Red, 4:RedEdge, 5:NIR
    if count >= 3:
        return (3, 2, 1)  # R, G, B
    elif count == 2:
        return (2, 1, 1)
    else:
        return (1, 1, 1)


def render_geotiff_thumbnail(path: str, size: int = 600) -> Image.Image:
    with rasterio.open(path) as ds:
        idx = _choose_rgb_indexes(ds.count)

        # Compute output size keeping aspect ratio
        scale = min(size / ds.width, size / ds.height)
        if scale > 1:
            scale = 1  # avoid upscaling
        out_h = max(1, int(ds.height * scale))
        out_w = max(1, int(ds.width * scale))

        arr = ds.read(
            indexes=list(idx),
            out_shape=(3, out_h, out_w),
            resampling=Resampling.bilinear,
            masked=True,
        ).astype("float32")

        # Replace masked with NaN for percentile calc
        if np.ma.isMaskedArray(arr):
            mask = np.ma.getmaskarray(arr)
            arr = np.ma.filled(arr, np.nan)
        else:
            mask = None

        arr8 = _percentile_stretch(arr)

        if mask is not None:
            # Set masked pixels to 0 (black)
            arr8 = np.where(mask, 0, arr8)

        rgb = np.transpose(arr8, (1, 2, 0))  # HWC
        im = Image.fromarray(rgb, mode="RGB")
        return im
