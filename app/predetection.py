from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from shapely.geometry import Polygon
from skimage.morphology import closing, disk, remove_small_holes, remove_small_objects


Progress = Callable[[int, int, str], None]


@dataclass
class PreDetectionResult:
    out_dir: Path
    vector_path: Path
    plot_count: int
    duration_sec: float
    summary_path: Path
    qa_path: Optional[Path] = None


@dataclass
class PlotCandidate:
    geometry: Polygon
    area_m2: float
    length_m: float
    width_m: float
    angle_deg: float
    source: str


def _ensure_predetect_dir(base_out: Path) -> tuple[Path, str]:
    """
    Mirror the segmentation folder structure but rooted at 0-Predetection.
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder = f"0-Predetection/Run_{ts}"
    if base_out.name.lower() == "output":
        out_dir = base_out / folder
    else:
        out_dir = base_out / "Output" / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, ts


def _pixel_metrics(transform: Affine) -> tuple[float, float, float]:
    """
    Compute pixel width, height and area in map units even if transform has rotation.
    """
    px_width = math.hypot(transform.a, transform.b)
    px_height = math.hypot(transform.d, transform.e)
    pixel_area = abs(transform.a * transform.e - transform.b * transform.d)
    return px_width, px_height, pixel_area


def _polygon_metrics(poly: Polygon) -> Optional[Tuple[float, float, float]]:
    """
    Return (length_m, width_m, angle_deg) for the minimum rotated rectangle of poly.
    """
    try:
        rect = poly.minimum_rotated_rectangle
        coords = list(rect.exterior.coords)
        if len(coords) < 4:
            return None
        edges: List[Tuple[float, float]] = []
        for i in range(4):
            x0, y0 = coords[i]
            x1, y1 = coords[(i + 1) % len(coords)]
            dx = x1 - x0
            dy = y1 - y0
            length = math.hypot(dx, dy)
            if length <= 0:
                continue
            angle = (math.degrees(math.atan2(dy, dx)) + 180.0) % 180.0
            edges.append((length, angle))
        if len(edges) < 2:
            return None
        edges.sort(key=lambda item: item[0], reverse=True)
        length_m = edges[0][0]
        width_m = edges[1][0]
        angle_deg = edges[0][1]
        return length_m, width_m, angle_deg
    except Exception:
        return None


def _angle_diff(a: float, b: float) -> float:
    """
    Minimal absolute difference between two 0-180 orientations.
    """
    return abs(((a - b + 90.0) % 180.0) - 90.0)


def _circular_mean(angles: Sequence[float]) -> Optional[float]:
    """
    Compute circular mean of orientations (0-180 symmetrical).
    """
    if not angles:
        return None
    doubled = np.deg2rad(np.array(angles, dtype=np.float64) * 2.0)
    sin_sum = np.sin(doubled).mean()
    cos_sum = np.cos(doubled).mean()
    if abs(sin_sum) < 1e-9 and abs(cos_sum) < 1e-9:
        return None
    mean = 0.5 * math.degrees(math.atan2(sin_sum, cos_sum))
    return mean % 180.0


def _prepare_intensity(stack: np.ndarray, nodata_vals: Optional[Sequence[float]]) -> np.ndarray:
    """
    Convert the raster stack into an 8-bit image emphasising darker plots.
    """
    if stack.ndim != 3:
        raise ValueError("Raster read must produce a (bands, rows, cols) array.")
    data = stack.astype("float32")
    raw = data.copy()
    mask_invalid = np.zeros(data.shape[1:], dtype=bool)
    nodata_vals = nodata_vals or []
    for idx, band in enumerate(data):
        nodata = None
        if idx < len(nodata_vals):
            nodata = nodata_vals[idx]
        if nodata is not None and math.isfinite(nodata):
            mask_invalid |= np.isclose(band, nodata, equal_nan=False)
        mask_invalid |= ~np.isfinite(band)
    if mask_invalid.any():
        data[:, mask_invalid] = np.nan
    gray = np.nanmean(data, axis=0)
    if not np.isfinite(gray).any():
        gray = np.nanmean(np.nan_to_num(raw, copy=False, nan=0.0), axis=0)
    if not np.isfinite(gray).any():
        raise RuntimeError("Raster contains no usable pixels for pre-detection.")
    max_val = np.nanpercentile(gray, 99) if np.isfinite(gray).any() else 1.0
    gray = np.where(np.isfinite(gray), gray, max_val)
    lo, hi = np.percentile(gray, [2, 98])
    if hi - lo < 1e-6:
        lo, hi = float(np.min(gray)), float(np.max(gray))
    if hi - lo < 1e-6:
        hi = lo + 1.0
    norm = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
    img8 = (norm * 255.0).astype("uint8")
    return img8


def _detect_rectangles_highres(
    img8: np.ndarray,
    transform: Affine,
    min_area_px: int,
    max_area_px: Optional[int],
    min_aspect: float,
) -> List[PlotCandidate]:
    """
    Detect rectangular plots directly on the full-resolution grayscale image.
    """
    candidates: List[PlotCandidate] = []
    blurred = cv2.GaussianBlur(img8, (5, 5), 0)
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if len(cnt) < 5:
            continue
        rect = cv2.minAreaRect(cnt)
        (width_px, height_px) = rect[1]
        if width_px <= 1 or height_px <= 1:
            continue
        area_px = width_px * height_px
        if area_px < max(4, min_area_px):
            continue
        if max_area_px is not None and area_px > max_area_px:
            continue
        major = max(width_px, height_px)
        minor = min(width_px, height_px)
        if minor <= 0:
            continue
        aspect = major / minor
        if aspect < min_aspect:
            continue
        box = cv2.boxPoints(rect)
        pts = [transform * (float(col), float(row)) for col, row in box]
        poly = Polygon(pts)
        if not poly.is_valid or poly.area <= 0:
            continue
        metrics = _polygon_metrics(poly)
        if metrics is None:
            continue
        length_m, width_m, angle_deg = metrics
        candidates.append(
            PlotCandidate(
                geometry=poly,
                area_m2=float(poly.area),
                length_m=float(length_m),
                width_m=float(width_m),
                angle_deg=float(angle_deg),
                source="highres",
            )
        )
    return candidates


def _texture_mask_candidates(
    raster_path: str | Path,
    downscale_max: int = 1200,
    min_region_px: int = 300,
) -> tuple[List[PlotCandidate], Optional[dict]]:
    """
    Detect plots on a downscaled raster using dual-threshold + texture heuristic.
    Returns candidates and optional QA image payload.
    """
    raster_path = str(raster_path)
    candidates: List[PlotCandidate] = []
    qa_payload: Optional[dict] = None

    with rasterio.open(raster_path) as src:
        scale = min(downscale_max / src.width, downscale_max / src.height, 1.0)
        out_h = max(1, int(src.height * scale))
        out_w = max(1, int(src.width * scale))

        idx = [1, 2, 3] if src.count >= 3 else [1]
        arr = src.read(indexes=idx, out_shape=(len(idx), out_h, out_w), resampling=rasterio.enums.Resampling.bilinear).astype("float32")

        for i in range(arr.shape[0]):
            band = arr[i]
            vmin = np.nanpercentile(band, 2)
            vmax = np.nanpercentile(band, 98)
            if vmax <= vmin:
                vmax = vmin + 1.0
            arr[i] = np.clip((band - vmin) / (vmax - vmin), 0, 1)

        blue = arr[0]
        blue8 = np.clip(blue * 255.0, 0, 255).astype(np.uint8)
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            blue_enh8 = clahe.apply(blue8)
        except Exception:
            blue_enh8 = blue8

        def local_std(img: np.ndarray, k: int = 10) -> np.ndarray:
            img32 = img.astype(np.float32)
            ksz = (int(k), int(k))
            mean = cv2.boxFilter(img32, ddepth=-1, ksize=ksz, normalize=True)
            mean_sq = cv2.boxFilter(img32 * img32, ddepth=-1, ksize=ksz, normalize=True)
            var = np.maximum(mean_sq - mean * mean, 0.0)
            return np.sqrt(var)

        if arr.shape[0] >= 3:
            tex = np.mean([local_std(arr[i], 10) for i in range(3)], axis=0)
        else:
            tex = local_std(arr[0], 10)
        tmin = float(np.nanpercentile(tex, 2))
        tmax = float(np.nanpercentile(tex, 98))
        if tmax <= tmin:
            tmax = tmin + 1.0
        tex_norm = np.clip((tex - tmin) / (tmax - tmin), 0.0, 1.0)
        tex8 = np.clip(tex_norm * 255.0, 0, 255).astype(np.uint8)

        inv_blue8 = (255 - blue_enh8)
        thr_b, _ = cv2.threshold(inv_blue8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr_b = int(max(0, min(255, thr_b * 0.95)))
        _, bin_b = cv2.threshold(inv_blue8, thr_b, 255, cv2.THRESH_BINARY)

        thr_t, _ = cv2.threshold(tex8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr_t = int(max(0, min(255, thr_t * 0.90)))
        _, bin_t = cv2.threshold(tex8, thr_t, 255, cv2.THRESH_BINARY)

        mask = (bin_b > 0) & (bin_t > 0)
        mask = closing(mask.astype(bool), footprint=disk(1))
        mask = remove_small_objects(mask, min_size=min_region_px)
        mask = remove_small_holes(mask, area_threshold=int(min_region_px * 0.4))
        mask8 = (mask.astype(np.uint8) * 255)

        scale_x = src.width / out_w
        scale_y = src.height / out_h
        transform = src.transform * Affine.scale(scale_x, scale_y)

        contours, _ = cv2.findContours(mask8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) < 5:
                continue
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            if w <= 1 or h <= 1:
                continue
            area_px = w * h
            if area_px < float(min_region_px):
                continue
            long_side = max(w, h)
            short_side = max(1.0, min(w, h))
            aspect = float(long_side / short_side)
            if aspect < 1.2 or aspect > 15.0:
                continue
            box = cv2.boxPoints(rect)
            pts = [transform * (float(px), float(py)) for px, py in box]
            poly = Polygon(pts)
            if not poly.is_valid or poly.area <= 0:
                continue
            metrics = _polygon_metrics(poly)
            if metrics is None:
                continue
            length_m, width_m, angle_deg = metrics
            candidates.append(
                PlotCandidate(
                    geometry=poly,
                    area_m2=float(poly.area),
                    length_m=float(length_m),
                    width_m=float(width_m),
                    angle_deg=float(angle_deg),
                    source="texture",
                )
            )

        qa_payload = {
            "data": blue_enh8,
            "transform": transform,
            "crs": src.crs,
        }

    return candidates, qa_payload


def _dedupe_candidates(cands: Sequence[PlotCandidate], iou_threshold: float = 0.25) -> List[PlotCandidate]:
    """
    Non-maximum suppression based on IoU to avoid duplicates.
    """
    kept: List[PlotCandidate] = []
    for cand in sorted(cands, key=lambda c: c.area_m2, reverse=True):
        geom = cand.geometry
        if geom.is_empty:
            continue
        keep = True
        for other in kept:
            inter = geom.intersection(other.geometry).area
            if inter <= 0:
                continue
            union = geom.union(other.geometry).area
            if union <= 0:
                continue
            iou = inter / union
            if iou >= iou_threshold:
                keep = False
                break
        if keep:
            kept.append(cand)
    return kept


def run_predetection(
    in_raster: str | Path,
    output_root: str | Path,
    *,
    min_plot_area_m2: float = 300.0,
    max_plot_area_m2: Optional[float] = 700.0,
    min_aspect_ratio: float = 1.8,
    orientation_tolerance_deg: float = 15.0,
    progress: Optional[Progress] = None,
) -> PreDetectionResult:
    """
    Detect elongated cultivation plots and write rectangular polygons to disk.
    The workflow fuses a high-resolution rectangle detector with a
    downscaled texture-based heuristic inspired by templates/predetect.py.
    """
    start = time.time()
    in_raster = str(in_raster)
    output_root = str(output_root)

    stage_total = 6
    stage = 0
    if progress:
        progress(stage, stage_total, "Loading raster…")

    with rasterio.open(in_raster) as src:
        stack = src.read()
        transform = src.transform
        crs = src.crs
        nodata_vals = src.nodatavals
        px_width, px_height, px_area = _pixel_metrics(transform)

    img8 = _prepare_intensity(stack, nodata_vals)

    if px_area <= 0 or not math.isfinite(px_area):
        min_area_px = int(max(10, min_plot_area_m2))
        max_area_px = int(max_plot_area_m2) if (max_plot_area_m2 is not None and math.isfinite(max_plot_area_m2) and max_plot_area_m2 > 0) else None
    else:
        min_area_px = int(max(10, min_plot_area_m2 / px_area))
        max_area_px = None
        if max_plot_area_m2 is not None and math.isfinite(max_plot_area_m2) and max_plot_area_m2 > 0:
            max_area_px = int(max_plot_area_m2 / px_area)
    if max_area_px is not None and max_area_px <= min_area_px:
        max_area_px = min_area_px + 1

    hr_candidates = _detect_rectangles_highres(
        img8=img8,
        transform=transform,
        min_area_px=min_area_px,
        max_area_px=max_area_px,
        min_aspect=min_aspect_ratio,
    )
    stage += 1
    if progress:
        progress(stage, stage_total, f"High-res candidates: {len(hr_candidates)}")

    texture_candidates, qa_payload = _texture_mask_candidates(
        raster_path=in_raster,
        downscale_max=1200,
        min_region_px=200,
    )
    stage += 1
    if progress:
        progress(stage, stage_total, f"Texture candidates: {len(texture_candidates)}")

    candidates = hr_candidates + texture_candidates

    max_area_m2_val = float("inf")
    if max_plot_area_m2 is not None and math.isfinite(max_plot_area_m2) and max_plot_area_m2 > 0:
        max_area_m2_val = float(max_plot_area_m2)

    filtered: List[PlotCandidate] = []
    for cand in candidates:
        if cand.area_m2 <= 0:
            continue
        if cand.area_m2 < min_plot_area_m2:
            continue
        if cand.area_m2 > max_area_m2_val:
            continue
        aspect = cand.length_m / max(1e-6, cand.width_m)
        if aspect < min_aspect_ratio:
            continue
        filtered.append(cand)

    stage += 1
    if progress:
        progress(stage, stage_total, f"Filtered candidates: {len(filtered)}")

    orientation_ref = _circular_mean([c.angle_deg for c in filtered]) if filtered else None
    orient_filtered = filtered
    if orientation_ref is not None and orientation_tolerance_deg is not None and orientation_tolerance_deg >= 0:
        orient_filtered = [
            cand for cand in filtered
            if _angle_diff(cand.angle_deg, orientation_ref) <= orientation_tolerance_deg
        ] or filtered

    deduped = _dedupe_candidates(orient_filtered, iou_threshold=0.3)
    stage += 1
    if progress:
        progress(stage, stage_total, f"Selected plots: {len(deduped)}")

    base_out = Path(output_root)
    out_dir, ts = _ensure_predetect_dir(base_out)
    out_path = out_dir / f"Predetection_{ts}.gpkg"
    qa_path: Optional[Path] = None

    records = []
    geometries = []
    for idx, cand in enumerate(deduped, start=1):
        records.append(
            {
                "plot_id": idx,
                "area_m2": float(cand.area_m2),
                "len_m": float(cand.length_m),
                "wid_m": float(cand.width_m),
                "ang_deg": float(cand.angle_deg),
                "source": cand.source,
            }
        )
        geometries.append(cand.geometry)

    gdf = gpd.GeoDataFrame(records, geometry=geometries, crs=crs)
    if len(gdf) > 0:
        gdf.to_file(out_path, driver="GPKG")
    else:
        out_path = out_path.with_suffix(".geojson")
        out_path.write_text('{"type": "FeatureCollection", "features": []}', encoding="utf-8")

    if qa_payload and qa_payload.get("data") is not None:
        qa_path = out_dir / f"Predetection_{ts}_qa.tif"
        data = qa_payload["data"].astype("uint8")
        with rasterio.open(
            qa_path,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype="uint8",
            crs=qa_payload.get("crs"),
            transform=qa_payload.get("transform"),
        ) as dst:
            dst.write(data, 1)

    dur = time.time() - start
    stage += 1
    if progress:
        progress(stage, stage_total, f"Pre-detection complete ({len(gdf)} plots).")

    summary = {
        "input_raster": in_raster,
        "output_path": str(out_path),
        "plot_count": int(len(gdf)),
        "min_plot_area_m2": float(min_plot_area_m2),
        "max_plot_area_m2": None if not math.isfinite(max_area_m2_val) else float(max_area_m2_val),
        "min_aspect_ratio": float(min_aspect_ratio),
        "orientation_reference_deg": float(orientation_ref) if orientation_ref is not None else None,
        "orientation_tolerance_deg": float(orientation_tolerance_deg),
        "source_counts": {
            "highres": len(hr_candidates),
            "texture": len(texture_candidates),
            "after_filter": len(deduped),
        },
        "duration_sec": float(dur),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return PreDetectionResult(
        out_dir=out_dir,
        vector_path=out_path,
        plot_count=len(gdf),
        duration_sec=dur,
        summary_path=summary_path,
        qa_path=qa_path,
    )
