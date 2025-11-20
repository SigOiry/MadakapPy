from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping


Progress = Callable[[int, int, str], None]


@dataclass
class SimpleClassifyResult:
    output_path: Path
    duration_sec: float
    selected_count: int
    line_count: int
    darkness_threshold: float


def _band_stats(
    src: rasterio.io.DatasetReader,
    geom,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    indexes = list(range(1, min(3, count) + 1))
    band_data, _ = rio_mask(src, [mapping(geom)], indexes=indexes, crop=True, filled=False)
    band_data = band_data.astype("float32")
    band_data[(band_data == 0) | (band_data == 65535)] = np.nan
    means_list: list[float] = []
    stds_list: list[float] = []
    for band in band_data:
        finite = np.isfinite(band)
        if not finite.any():
            means_list.append(np.nan)
            stds_list.append(np.nan)
            continue
        means_list.append(float(np.nanmean(band[finite])))
        stds_list.append(float(np.nanstd(band[finite])))
    means = np.array(means_list, dtype=np.float32)
    stds = np.array(stds_list, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        means /= 65535.0
        stds /= 65535.0
    return means, stds


def _biomass_from_area_cm2(area_cm2: np.ndarray | float, model: str) -> np.ndarray:
    arr = np.asarray(area_cm2, dtype=np.float64)
    arr = np.maximum(arr, 0.0)
    model_key = (model or "madagascar").strip().lower()
    if model_key == "indonesia":
        with np.errstate(invalid="ignore"):
            return 0.014 * np.power(arr, 1.65)
    return (-0.0022 * arr * arr) + (2.3389 * arr) + 29.771


def classify_dark_linear_polygons(
    raster_path: str | Path,
    segments_path: str | Path,
    output_root: str | Path,
    biomass_model: str = "madagascar",
    progress: Optional[Progress] = None,
) -> SimpleClassifyResult:
    """
    Select floating algae ropes using spectral features only (means/stds/indices).
    """
    start = time.time()
    raster_path = str(raster_path)
    segments_path = str(segments_path)
    base_out = Path(output_root)

    gdf = gpd.read_file(segments_path)
    gdf = gdf.reset_index(drop=True)
    seg_crs = gdf.crs

    total = len(gdf)
    if total == 0:
        raise RuntimeError("No segments available for classification.")

    with rasterio.open(raster_path) as src:
        if seg_crs and src.crs and str(seg_crs) != str(src.crs):
            gdf = gdf.to_crs(src.crs)

        band_means = np.full((total, 3), np.nan, dtype=np.float32)
        band_stds = np.full((total, 3), np.nan, dtype=np.float32)
        spectral_idx = np.full((total, 2), np.nan, dtype=np.float32)
        brightness = np.full(total, np.nan, dtype=np.float32)

        for idx, geom in enumerate(gdf.geometry):
            if progress and (idx % 20 == 0 or idx + 1 == total):
                progress(idx + 1, total, "Evaluating polygons")
            if geom is None or geom.is_empty:
                continue
            try:
                means, stds = _band_stats(src, geom, src.count)
            except Exception:
                means = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
                stds = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
            if means.size < 3:
                means = np.pad(means, (0, 3 - means.size), constant_values=np.nan)
                stds = np.pad(stds, (0, 3 - stds.size), constant_values=np.nan)
            band_means[idx] = means
            band_stds[idx] = stds
            if np.isfinite(means).any():
                brightness[idx] = float(np.nanmean(means))
            g, r = means[1], means[2]
            b = means[0]
            denom_gr = g + r if np.isfinite(g) and np.isfinite(r) else np.nan
            denom_bg = b + g if np.isfinite(b) and np.isfinite(g) else np.nan
            nd_gr = (g - r) / denom_gr if denom_gr and denom_gr != 0 else np.nan
            nd_bg = (b - g) / denom_bg if denom_bg and denom_bg != 0 else np.nan
            spectral_idx[idx] = np.array([nd_gr, nd_bg], dtype=np.float32)

    valid_brightness = np.isfinite(brightness)
    if not valid_brightness.any():
        raise RuntimeError("Could not compute spectral statistics for polygons.")

    norm_brightness = brightness.copy()
    finite_vals = norm_brightness[valid_brightness]
    min_b, max_b = float(np.nanmin(finite_vals)), float(np.nanmax(finite_vals))
    if max_b - min_b > 1e-8:
        norm_brightness[valid_brightness] = (finite_vals - min_b) / (max_b - min_b)
    else:
        norm_brightness[valid_brightness] = 0.0

    std_total = np.nanmean(band_stds, axis=1)
    valid_std = np.isfinite(std_total)
    if valid_std.any():
        std_min = float(np.nanmin(std_total[valid_std]))
        std_max = float(np.nanmax(std_total[valid_std]))
        if std_max - std_min > 1e-8:
            std_norm = (std_total - std_min) / (std_max - std_min)
        else:
            std_norm = np.zeros_like(std_total)
    else:
        std_norm = np.zeros_like(std_total)

    nd_gr = spectral_idx[:, 0]
    nd_bg = spectral_idx[:, 1]

    brightness_col = np.nan_to_num(norm_brightness, nan=0.0).reshape(-1, 1) * 2.0
    stdnorm_col = np.nan_to_num(std_norm, nan=0.0).reshape(-1, 1)
    feature_matrix = np.column_stack(
        [
            band_means,
            band_stds,
            np.nan_to_num(nd_gr, nan=0.0).reshape(-1, 1),
            np.nan_to_num(nd_bg, nan=0.0).reshape(-1, 1),
            brightness_col,
            stdnorm_col,
        ]
    )
    col_medians = np.nanmedian(feature_matrix, axis=0)
    feature_matrix = np.where(np.isfinite(feature_matrix), feature_matrix, col_medians)

    valid_rows = np.all(np.isfinite(feature_matrix), axis=1)
    if valid_rows.sum() < 2:
        selected = np.zeros(total, dtype=bool)
        topn = max(1, int(0.02 * total))
        ranked = np.argsort(np.nan_to_num(norm_brightness, nan=1.0))
        selected[ranked[:topn]] = True
    else:
        val_idx = np.where(valid_rows)[0]
        vals = feature_matrix[val_idx]
        gram = vals @ vals.T
        norms = np.sum(vals * vals, axis=1, keepdims=True)
        dist_sq = norms + norms.T - 2.0 * gram
        np.fill_diagonal(dist_sq, -np.inf)
        i0, i1 = np.unravel_index(np.argmax(dist_sq), dist_sq.shape)
        centroids = np.stack([vals[i0], vals[i1]], axis=0)

        def kmeans_step(data: np.ndarray, cents: np.ndarray, iters: int = 12) -> tuple[np.ndarray, np.ndarray]:
            labels = np.zeros(data.shape[0], dtype=np.int32)
            for _ in range(iters):
                d0 = np.sum((data - cents[0]) ** 2, axis=1)
                d1 = np.sum((data - cents[1]) ** 2, axis=1)
                new_labels = (d1 < d0).astype(np.int32)
                if np.array_equal(new_labels, labels):
                    break
                labels = new_labels
                for k in range(2):
                    members = data[labels == k]
                    if len(members) > 0:
                        cents[k] = np.mean(members, axis=0)
            return labels, cents

        lbls, centroids = kmeans_step(vals, centroids)
        target_cluster = None
        brightness_by_cluster = []
        for k in range(2):
            members = val_idx[lbls == k]
            if members.size == 0:
                brightness_by_cluster.append(np.inf)
            else:
                brightness_by_cluster.append(float(np.nanmean(norm_brightness[members])))
        target_cluster = int(np.argmin(brightness_by_cluster))
        selected = np.zeros(total, dtype=bool)
        selected[val_idx[lbls == target_cluster]] = True
        if selected.sum() == 0:
            fallback_take = max(1, int(0.02 * total))
            ranked = np.argsort(norm_brightness[val_idx])
            selected[val_idx[ranked[:fallback_take]]] = True

    gdf["mean_blue"] = band_means[:, 0]
    gdf["mean_green"] = band_means[:, 1]
    gdf["mean_red"] = band_means[:, 2]
    gdf["std_blue"] = band_stds[:, 0]
    gdf["std_green"] = band_stds[:, 1]
    gdf["std_red"] = band_stds[:, 2]
    gdf["nd_gr"] = spectral_idx[:, 0]
    gdf["nd_bg"] = spectral_idx[:, 1]
    gdf["brightness"] = norm_brightness
    gdf["score"] = np.nan_to_num(1.0 - norm_brightness, nan=0.0)
    gdf["cluster_id"] = np.where(selected, 1, 0)
    gdf["selected"] = selected

    area_cm2 = np.asarray(gdf.geometry.area, dtype=np.float64) * 10000.0
    gdf["plot_area_cm2"] = area_cm2
    gdf["biomass_g"] = _biomass_from_area_cm2(area_cm2, biomass_model)

    if seg_crs:
        gdf = gdf.to_crs(seg_crs)

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_base = base_out if base_out.name.lower() == "output" else base_out / "Output"
    folder = out_base / "2-Stats" / f"Run_{ts}"
    folder.mkdir(parents=True, exist_ok=True)
    out_path = folder / f"Stats_{ts}.shp"
    gdf.to_file(out_path)

    dur = time.time() - start
    if progress:
        progress(1, 1, f"Stats classifier selected {int(selected.sum())} polygons")
    return SimpleClassifyResult(
        output_path=out_path,
        duration_sec=dur,
        selected_count=int(selected.sum()),
        line_count=0,
        darkness_threshold=float(np.nanmean(norm_brightness[selected])) if selected.any() else float("nan"),
    )
