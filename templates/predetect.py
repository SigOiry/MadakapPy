from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon
import cv2
from skimage.filters import gaussian
from skimage.morphology import remove_small_objects, remove_small_holes, closing, disk

import geopandas as gpd

Progress = Callable[[int, int, str], None]


def _ensure_temp(output_root: str | os.PathLike) -> Path:
    base = Path(output_root)
    temp_dir = base.parent / "Temp" if base.name.lower() == "output" else base / "Temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def detect_cultivation_plots(
    raster_path: str | os.PathLike,
    output_root: str | os.PathLike,
    downscale_max: int = 1200,
    min_region_px: int = 300,
    merge_size_px: int = 3,
    width_m_range: tuple[float, float] | None = None,
    length_m_range: tuple[float, float] | None = None,
    area_m_min: float = 150.0,
    area_m_max: float | None = 900.0,
    blue_band_index: int | None = 1,
    progress: Optional[Progress] = None,
) -> Tuple[Path, int]:
    """
    Fast heuristic plot detector to build an AOI polygon file from the raster.

    - Downscales the image for speed
    - Converts to grayscale and applies local thresholding
    - Morphological cleanup and small region removal
    - Vectorizes mask into polygons and writes a temporary AOI shapefile

    Returns (aoi_path, polygon_count).
    """
    raster_path = str(raster_path)
    # Default to Temp; caller can move results, or we can additionally save to Output on request
    temp_dir = _ensure_temp(output_root)
    aoi_path = temp_dir / "plots_aoi.shp"

    if progress:
        progress(0, 3, "Pre‑detecting plots: loading raster")

    with rasterio.open(raster_path) as src:
        # Compute output size while keeping aspect ratio
        scale = min(downscale_max / src.width, downscale_max / src.height, 1.0)
        out_h = max(1, int(src.height * scale))
        out_w = max(1, int(src.width * scale))

        # Always load bands 1,2,3 to build features (blue + texture from RGB)
        idx = [1, 2, 3] if src.count >= 3 else [1]
        arr = src.read(indexes=idx, out_shape=(len(idx), out_h, out_w), resampling=Resampling.bilinear).astype("float32")

        # Normalize per-band to 0..1 robustly
        for i in range(arr.shape[0]):
            band = arr[i]
            vmin = np.nanpercentile(band, 2)
            vmax = np.nanpercentile(band, 98)
            if vmax <= vmin:
                vmax = vmin + 1.0
            arr[i] = np.clip((band - vmin) / (vmax - vmin), 0, 1)

        # --- Two-band discrimination: (1) blue with mild contrast enhancement, (2) RGB texture (10x10 window) ---
        # 1) Blue band with mild CLAHE
        blue = arr[0]
        blue8 = np.clip(blue * 255.0, 0, 255).astype(np.uint8)
        try:
            blue_enh8 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blue8)
        except Exception:
            blue_enh8 = blue8

        # 2) Texture band as mean local std over bands 1,2,3 with 10x10 window
        def local_std(img: np.ndarray, k: int = 10) -> np.ndarray:
            img32 = img.astype(np.float32)
            ksz = (int(k), int(k))
            m = cv2.boxFilter(img32, ddepth=-1, ksize=ksz, normalize=True)
            m2 = cv2.boxFilter(img32 * img32, ddepth=-1, ksize=ksz, normalize=True)
            var = np.maximum(m2 - m * m, 0.0)
            return np.sqrt(var)

        if arr.shape[0] >= 3:
            stds = [local_std(arr[i], 10) for i in range(3)]
            tex = np.mean(stds, axis=0)
        else:
            tex = local_std(arr[0], 10)
        # Normalize texture to 0..1 robustly
        tmin = float(np.nanpercentile(tex, 2))
        tmax = float(np.nanpercentile(tex, 98))
        if tmax <= tmin:
            tmax = tmin + 1.0
        tex_norm = np.clip((tex - tmin) / (tmax - tmin), 0.0, 1.0)
        tex8 = np.clip(tex_norm * 255.0, 0, 255).astype(np.uint8)

        # Combine both: dark in blue and sufficiently textured
        inv_blue8 = (255 - blue_enh8)
        # Otsu thresholds with slight relaxation to include faded plots
        thr_b, _ = cv2.threshold(inv_blue8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr_b = int(max(0, min(255, thr_b * 0.95)))
        _, bin_b = cv2.threshold(inv_blue8, thr_b, 255, cv2.THRESH_BINARY)

        thr_t, _ = cv2.threshold(tex8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr_t = int(max(0, min(255, thr_t * 0.90)))
        _, bin_t = cv2.threshold(tex8, thr_t, 255, cv2.THRESH_BINARY)

        bin_ = (bin_b > 0) & (bin_t > 0)
        # Morphological cleanup
        bin_ = closing(bin_.astype(bool), footprint=disk(1))
        bin_ = remove_small_objects(bin_, min_size=min_region_px)
        bin_ = remove_small_holes(bin_, area_threshold=int(min_region_px * 0.4))
        mask = bin_.astype(np.uint8)

        if progress:
            progress(2, 3, "Pre‑detecting plots: rectangle fitting")

        # Build transform for downscaled image
        scale_x = src.width / out_w
        scale_y = src.height / out_h
        transform = src.transform * rasterio.Affine.scale(scale_x, scale_y)

        # Save QA image (enhanced blue) next to temp AOI
        try:
            blur_path = temp_dir / "plots_blur.tif"
            with rasterio.open(
                blur_path,
                "w",
                driver="GTiff",
                height=out_h,
                width=out_w,
                count=1,
                dtype="uint8",
                crs=src.crs,
                transform=transform,
            ) as dst:
                dst.write(blue_enh8, 1)
        except Exception:
            pass

        # Find contours and fit rotated rectangles
        bin8 = (mask.astype(np.uint8) * 255)
        cnts, _ = cv2.findContours(bin8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rect_polys: list[Polygon] = []
        angles: list[float] = []
        RAW_MODE = False  # we will export raw separately but final AOI uses orientation-only filtering
        # First pass: gather rectangles with basic filters
        # Compute meters per pixel at the downscaled resolution
        px_m = abs(src.transform.a) * scale_x
        py_m = abs(src.transform.e) * scale_y if hasattr(src.transform, 'e') else abs(src.transform.a) * scale_y
        avg_mpp = float((px_m + py_m) / 2.0)

        for c in cnts:
            if len(c) < 5:
                continue
            # Prefer 4‑vertex polygons via approxPolyDP; fallback to minAreaRect
            per = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * per, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                pts = approx.reshape(-1, 2).astype(np.float32)
                # Compute rotated rectangle from polygon
                rect = cv2.minAreaRect(pts)
            else:
                rect = cv2.minAreaRect(c)
            (cx, cy), (w, h), ang = rect
            if w <= 1 or h <= 1:
                continue
            rect_area = float(w * h)
            if not RAW_MODE and rect_area < float(min_region_px):
                continue
            long_side = max(w, h)
            short_side = max(1.0, min(w, h))
            ar = float(long_side / short_side)
            # Cultivation plots elongated; relax aspect ratio tolerance further
            if not RAW_MODE and (ar < 1.2 or ar > 12.0):
                continue
            ca = float(cv2.contourArea(c))
            fill = ca / rect_area
            if not RAW_MODE and fill < 0.35:  # discard wispy regions
                continue
            # Normalize angle so 0..90 is the orientation of the long side
            if w < h:
                ang = ang + 90.0
            ang = abs(ang) % 90.0

            # Physical dimension filtering (meters)
            long_m = float(long_side) * avg_mpp
            short_m = float(short_side) * avg_mpp
            # Ensure long_m >= short_m
            if short_m > long_m:
                long_m, short_m = short_m, long_m
            # Area/size filters (disabled in RAW_MODE)
            if not RAW_MODE:
                area_m = (float(rect_area) * (avg_mpp ** 2))
                if area_m < float(area_m_min):
                    continue
                if area_m_max is not None and area_m > float(area_m_max):
                    continue
                if width_m_range is not None:
                    wmin, wmax = width_m_range
                    if not (wmin <= short_m <= wmax):
                        continue
                if length_m_range is not None:
                    lmin, lmax = length_m_range
                    if not (lmin <= long_m <= lmax):
                        continue
            angles.append(ang)
            # Convert rect to polygon in pixel space then to map coordinates
            box_pts = cv2.boxPoints(rect)  # (x,y)
            # Re-orient to represent the long side consistently
            pts_xy = [transform * (float(px), float(py)) for px, py in box_pts]
            rect_polys.append(Polygon(pts_xy))

        # Export raw rectangles before any orientation filtering
        try:
            gpd.GeoDataFrame(geometry=rect_polys, crs=src.crs).to_file(temp_dir / "plots_raw.shp")
        except Exception:
            pass

        if not rect_polys:
            # Fallback to mask vectorization
            geoms = []
            for geom, val in shapes(mask.astype(np.uint8), mask=None, transform=transform):
                if int(val) == 1:
                    poly = shape(geom)
                    if isinstance(poly, (Polygon, MultiPolygon)) and poly.area > 0:
                        geoms.append(poly)
            merged = unary_union(geoms) if geoms else []
            if not geoms:
                gpd.GeoDataFrame(geometry=[], crs=src.crs).to_file(aoi_path)
                return aoi_path, 0
            if isinstance(merged, Polygon):
                out_geoms = [merged]
            elif isinstance(merged, MultiPolygon):
                out_geoms = list(merged.geoms)
            else:
                out_geoms = list(getattr(merged, "geoms", [merged]))
            gdf = gpd.GeoDataFrame(geometry=out_geoms, crs=src.crs)
            gdf.to_file(aoi_path)
            if progress:
                progress(3, 3, f"Pre‑detected plots: {len(gdf)}")
            return aoi_path, len(gdf)

        # Filter rectangles by dominant orientation (only orientation gate)
        if angles:
            angs = np.array(angles)
            hist, edges = np.histogram(angs, bins=12, range=(0, 90))
            dom_bin = int(np.argmax(hist))
            a0 = 0.5 * (edges[dom_bin] + edges[dom_bin + 1])
            rect_polys = [p for p, ang in zip(rect_polys, angles) if abs(ang - a0) <= 12.0] or rect_polys

        # Non‑max suppression to avoid duplicates while keeping individual plots
        def iou(a: Polygon, b: Polygon) -> float:
            inter = a.intersection(b).area
            if inter <= 0:
                return 0.0
            return inter / (a.union(b).area)

        # NMS and export filtered set
        rect_polys_sorted = sorted(rect_polys, key=lambda p: p.area, reverse=True)
        kept: list[Polygon] = []
        for p in rect_polys_sorted:
            if all(iou(p, q) < 0.25 for q in kept):
                kept.append(p)
        try:
            kept = [p.buffer(-0.05, join_style=2) if p.buffer(-0.05).area > 0 else p for p in kept]
        except Exception:
            pass
        try:
            gpd.GeoDataFrame(geometry=kept, crs=src.crs).to_file(temp_dir / "plots_filtered.shp")
        except Exception:
            pass
        gdf = gpd.GeoDataFrame(geometry=kept, crs=src.crs)
        gdf.to_file(aoi_path)

        if progress:
            progress(3, 3, f"Pre‑detecting plots: {len(gdf)} polygons")

        return aoi_path, len(gdf)


def save_predetection_to_output(
    aoi_temp_path: str | os.PathLike,
    output_root: str | os.PathLike,
) -> Path:
    """Copy/convert the AOI to Output/0-Predetection/Run_YYYY-mm-dd_HHMMSS folder and return path."""
    import shutil
    from datetime import datetime

    base = Path(output_root)
    if base.name.lower() == "output":
        out_base = base
    else:
        out_base = base / "Output"
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = out_base / "0-Predetection" / f"Run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Shapefile is multi-file. Copy all sidecars with same stem.
    src = Path(aoi_temp_path)
    stem = src.stem
    for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):  # some may not exist
        p = src.with_suffix(ext)
        if p.exists():
            shutil.copy2(p, run_dir / p.name)
    # Also copy QA blurred TIFF if present in temp
    blur = src.parent / "plots_blur.tif"
    if blur.exists():
        shutil.copy2(blur, run_dir / "plots_blur.tif")
    return run_dir / (stem + ".shp")
