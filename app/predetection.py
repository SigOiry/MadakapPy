from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon
import cv2
from skimage.morphology import remove_small_objects, remove_small_holes, closing, disk

import geopandas as gpd

Progress = Callable[[int, int, str], None]

_MIN_REGION_PX = 300
_MORPH_CLOSE_RADIUS = 1
_HOLE_AREA_FRAC = 0.4
_APPROX_EPSILON_FRAC = 0.02
_FILL_RATIO_MIN = 0.35
_NMS_IOU_THRESH = 0.25


def _ensure_temp(output_root: str | os.PathLike) -> Path:
    base = Path(output_root)
    temp_dir = base.parent / "Temp" if base.name.lower() == "output" else base / "Temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def detect_cultivation_plots(
    raster_path: str | os.PathLike,
    output_root: str | os.PathLike,
    min_width_m: float = 5.0,
    max_width_m: float | None = 20.0,
    min_length_m: float = 20.0,
    max_length_m: float | None = 60.0,
    small_polygon_buffer_m: float = 0.3,
    blue_quantile: float = 0.15,
    orient_tolerance_deg: float = 12.0,
    progress: Optional[Progress] = None,
) -> Tuple[Path, int]:
    """
    Fast heuristic plot detector that prepares AOI polygons from the raster.

    - Uses the raw-resolution blue band (band 1) with percentile thresholding
    - Applies a 5x5 focal (max) filter to build a candidate mask
    - Morphological cleanup, contour fitting, and orientation gating
    - Buffers/merges undersized polygons, filters by min area, and writes an AOI

    Returns (aoi_path, polygon_count).
    """
    raster_path = str(raster_path)
    # Default to Temp; caller can move results, or we can additionally save to Output on request
    temp_dir = _ensure_temp(output_root)
    aoi_path = temp_dir / "plots_aoi.shp"

    min_width = max(0.0, float(min_width_m))
    min_length = max(0.0, float(min_length_m))
    max_width = float(max_width_m) if (max_width_m is not None and float(max_width_m) > 0) else None
    max_length = float(max_length_m) if (max_length_m is not None and float(max_length_m) > 0) else None
    blue_quantile = float(np.clip(blue_quantile, 0.0, 1.0))
    min_area_m = (min_width * min_length) if (min_width > 0 and min_length > 0) else 0.0
    small_buffer = max(0.0, float(small_polygon_buffer_m))

    if progress:
        progress(0, 3, "Pre-detecting plots: loading raster")
    with rasterio.open(raster_path) as src:
        # Load blue band (band 1) at full resolution
        b_idx = 1 if src.count >= 1 else src.count
        arr = src.read(b_idx, masked=True)
        if np.ma.isMaskedArray(arr):
            blue = np.asarray(arr.filled(np.nan), dtype=np.float32)
        else:
            blue = arr.astype(np.float32)

        # Treat zero values as no-data before computing the percentile
        zero_mask = blue == 0
        if np.any(zero_mask):
            blue[zero_mask] = np.nan
        if not np.isfinite(blue).any():
            raise ValueError("Blue band contains only zero or no-data values.")

        percentile = float(blue_quantile * 100.0)
        p80_val = float(np.nanpercentile(blue, percentile))
        valid_mask = np.isfinite(blue)
        filtered = np.zeros_like(blue, dtype=np.uint8)
        filtered[valid_mask] = (blue[valid_mask] < p80_val).astype(np.uint8)

        # 5x5 focal max filter (equivalent to terra::focal(..., fun='max'))
        kernel = np.ones((5, 5), dtype=np.uint8)
        foc = cv2.dilate(filtered, kernel)
        foc = (foc > 0).astype(np.uint8)
        foc_raster = foc.copy()

        # Continue workflow from the focal output
        bin_ = foc.astype(bool)
        if _MORPH_CLOSE_RADIUS > 0:
            bin_ = closing(bin_, footprint=disk(_MORPH_CLOSE_RADIUS))
        bin_ = remove_small_objects(bin_, min_size=int(_MIN_REGION_PX))
        hole_area_thresh = max(1, int(_MIN_REGION_PX * _HOLE_AREA_FRAC))
        bin_ = remove_small_holes(bin_, area_threshold=hole_area_thresh)
        mask = bin_.astype(np.uint8)

        if progress:
            progress(2, 3, "Pre-detecting plots: rectangle fitting")
        transform = src.transform

        # Export the focal mask for QA
        try:
            foc_path = temp_dir / "plots_foc.tif"
            with rasterio.open(
                foc_path,
                "w",
                driver="GTiff",
                height=src.height,
                width=src.width,
                count=1,
                dtype="uint8",
                crs=src.crs,
                transform=transform,
            ) as dst:
                dst.write(foc_raster, 1)
        except Exception:
            pass

        # Find contours and fit rotated rectangles
        bin8 = (mask.astype(np.uint8) * 255)
        cnts, _ = cv2.findContours(bin8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rect_records: list[dict] = []
        px_m = abs(src.transform.a)
        py_m = abs(getattr(src.transform, "e", src.transform.a))
        avg_mpp = float((px_m + py_m) / 2.0)

        for c in cnts:
            if len(c) < 5:
                continue
            per = cv2.arcLength(c, True)
            eps = float(_APPROX_EPSILON_FRAC) * per
            approx = cv2.approxPolyDP(c, eps, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                pts = approx.reshape(-1, 2).astype(np.float32)
                rect = cv2.minAreaRect(pts)
            else:
                rect = cv2.minAreaRect(c)
            (_cx, _cy), (w, h), ang = rect
            if w <= 1 or h <= 1:
                continue
            rect_area = float(w * h)
            ca = float(cv2.contourArea(c))
            fill = ca / rect_area if rect_area > 0 else 0.0
            if fill < float(_FILL_RATIO_MIN):
                continue
            if w < h:
                ang = ang + 90.0
            ang = abs(ang) % 90.0

            long_side = max(w, h)
            short_side = max(1.0, min(w, h))
            long_m = float(long_side) * avg_mpp
            short_m = float(short_side) * avg_mpp
            if short_m > long_m:
                long_m, short_m = short_m, long_m

            size_ok = True
            if min_width > 0 and short_m < min_width:
                size_ok = False
            if min_length > 0 and long_m < min_length:
                size_ok = False
            if max_width is not None and short_m > max_width:
                size_ok = False
            if max_length is not None and long_m > max_length:
                size_ok = False

            box_pts = cv2.boxPoints(rect)
            pts_xy = [transform * (float(px), float(py)) for px, py in box_pts]
            rect_records.append(
                {
                    "geometry": Polygon(pts_xy),
                    "angle": ang,
                    "size_ok": size_ok,
                }
            )

        raw_polys = [rec["geometry"] for rec in rect_records]
        try:
            gpd.GeoDataFrame(geometry=raw_polys, crs=src.crs).to_file(temp_dir / "plots_raw.shp")
        except Exception:
            pass

        if not rect_records:
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
            if min_area_m > 0:
                out_geoms = [g for g in out_geoms if g.area >= min_area_m]
            gdf = gpd.GeoDataFrame(geometry=out_geoms, crs=src.crs)
            gdf.to_file(aoi_path)
            if progress:
                progress(3, 3, f"Pre-detected plots: {len(gdf)}")
            return aoi_path, len(gdf)

        orientation_pool = [rec for rec in rect_records if rec["size_ok"]] or rect_records
        dom_angle = None
        tol = float(max(0.0, orient_tolerance_deg))
        if orientation_pool:
            angs = np.array([rec["angle"] for rec in orientation_pool])
            hist, edges = np.histogram(angs, bins=12, range=(0, 90))
            dom_bin = int(np.argmax(hist))
            dom_angle = 0.5 * (edges[dom_bin] + edges[dom_bin + 1])

        oriented_polys = raw_polys
        if dom_angle is not None:
            oriented_polys = [rec["geometry"] for rec in rect_records if abs(rec["angle"] - dom_angle) <= tol] or raw_polys

        def iou(a: Polygon, b: Polygon) -> float:
            inter = a.intersection(b).area
            if inter <= 0:
                return 0.0
            return inter / (a.union(b).area)

        rect_polys_sorted = sorted(oriented_polys, key=lambda p: p.area, reverse=True)
        kept: list[Polygon] = []
        thr = float(max(0.0, min(1.0, _NMS_IOU_THRESH)))
        for p in rect_polys_sorted:
            if all(iou(p, q) < thr for q in kept):
                kept.append(p)

        adjusted: list[Polygon] = []
        for poly in kept:
            geom = poly
            if min_area_m > 0 and geom.area < min_area_m and small_buffer > 0:
                buffered = geom.buffer(small_buffer, join_style=2)
                if buffered.area > 0:
                    geom = buffered
            adjusted.append(geom)

        merged_polys: list[Polygon] = []
        if adjusted:
            merged_geom = unary_union(adjusted)
            if isinstance(merged_geom, Polygon):
                merged_polys = [merged_geom]
            elif isinstance(merged_geom, MultiPolygon):
                merged_polys = list(merged_geom.geoms)
            else:
                merged_polys = list(getattr(merged_geom, "geoms", []))

        final_polys: list[Polygon] = merged_polys or []
        if min_area_m > 0:
            final_polys = [poly for poly in final_polys if poly.area >= min_area_m]

        try:
            gpd.GeoDataFrame(geometry=final_polys, crs=src.crs).to_file(temp_dir / "plots_filtered.shp")
        except Exception:
            pass
        gdf = gpd.GeoDataFrame(geometry=final_polys, crs=src.crs)
        gdf.to_file(aoi_path)

        if progress:
            progress(3, 3, f"Pre-detecting plots: {len(gdf)} polygons")
        return aoi_path, len(gdf)


def save_preselection_to_output(
    aoi_temp_path: str | os.PathLike,
    output_root: str | os.PathLike,
) -> Path:
    """Copy/convert the AOI to Output/0-Preselection/Run_YYYY-mm-dd_HHMMSS folder and return path."""
    import shutil
    from datetime import datetime

    base = Path(output_root)
    if base.name.lower() == "output":
        out_base = base
    else:
        out_base = base / "Output"
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = out_base / "0-Preselection" / f"Run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Shapefile is multi-file. Copy all sidecars with same stem.
    src = Path(aoi_temp_path)
    stem = src.stem
    for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):  # some may not exist
        p = src.with_suffix(ext)
        if p.exists():
            shutil.copy2(p, run_dir / p.name)
    # Also copy QA rasters (focal mask + legacy blur) if present in temp
    for extra in ("plots_foc.tif", "plots_blur.tif"):
        extra_src = src.parent / extra
        if extra_src.exists():
            shutil.copy2(extra_src, run_dir / extra_src.name)
    return run_dir / (stem + ".shp")
