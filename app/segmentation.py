from __future__ import annotations

import os
import math
import time
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio import features as rio_features


ProgressCallback = Callable[[int, int, str], None]


@dataclass
class SegmentationResult:
    out_dir: Path
    out_shp: Optional[Path]
    tiles_total: int
    duration_sec: float


def _ensure_dirs(base_out: Path) -> Tuple[Path, Path, Path, str]:
    """Create output and temp directories.

    If base_out points to an "Output" folder, write runs under it and create
    a sibling "Temp" folder at the project root. Otherwise, create an
    "Output" subfolder under base_out and a "Temp" under base_out.
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    foldername = f"1-Segmentation/Run_{ts}"
    if base_out.name.lower() == "output":
        out_dir = base_out / foldername
        temp_dir = base_out.parent / "Temp"
    else:
        out_dir = base_out / "Output" / foldername
        temp_dir = base_out / "Temp"
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    out_shp = out_dir / f"Segmentation_{ts}.shp"
    return out_dir, temp_dir, out_shp, ts


def _ranger_from_band1(src: rasterio.io.DatasetReader, window: rasterio.windows.Window) -> float:
    band1 = src.read(1, window=window, masked=True)
    arr = band1.astype("float32")
    arr[(arr == 0) | (arr == 65535)] = np.nan
    ranger = float(np.nanstd(arr)) * 0.5
    if not np.isfinite(ranger) or ranger == 0:
        ranger = 1.0
    return ranger


def _maybe_merge_shapefiles(temp_dir: Path, out_shp: Path) -> Optional[Path]:
    ogr = shutil.which("ogr2ogr")
    if not ogr:
        return None
    segs = sorted(str(p) for p in temp_dir.glob("seg_*.shp"))
    if not segs:
        return None
    subprocess.run([ogr, "-skipfailures", str(out_shp), segs[0]], check=True)
    for shp in segs[1:]:
        subprocess.run([ogr, "-skipfailures", "-update", "-append", str(out_shp), shp, "-nln", out_shp.stem], check=True)
    return out_shp


def run_segmentation(
    in_raster: str | os.PathLike,
    output_root: str | os.PathLike,
    otb_bin: str | os.PathLike,
    tile_size_m: int = 40,
    spatialr: int = 5,
    minsize: int = 5,
    aoi_path: str | os.PathLike | None = None,
    progress: Optional[ProgressCallback] = None,
) -> SegmentationResult:
    """
    Mimic the R 'Segmentation' chunk using OTB LargeScaleMeanShift.

    - Tiles the raster by physical size (m) converted to pixels via geotransform
    - Computes ranger per tile: sd(band1) (ignoring 0 and 65535)
    - Optionally skips tiles that do not intersect `aoi_path` polygons
    - Calls OTB LargeScaleMeanShift per tile to produce BOTH raster and vector outputs
      (runs raster mode, then vector mode)
    - Writes per-tile raster and shapefiles to Temp (temporary workspace)
    - Merges all tile results into a single shapefile in the Run folder
    - Cleans the Temp folder at the end of the process

    progress callback receives (done, total, note)
    """

    in_raster = str(in_raster)
    output_root = str(output_root)
    otb_bin = str(otb_bin)
    aoi_path_str = str(aoi_path) if aoi_path else None

    if not os.path.exists(in_raster):
        raise FileNotFoundError(f"Input raster not found: {in_raster}")
    if not os.path.exists(otb_bin):
        raise FileNotFoundError("OTB LargeScaleMeanShift not found; set the proper path")
    if aoi_path_str and not os.path.exists(aoi_path_str):
        raise FileNotFoundError(f"AOI polygons not found: {aoi_path_str}")

    base_out = Path(output_root)
    out_dir, temp_dir, out_shp, _ts = _ensure_dirs(base_out)

    start = time.time()
    done = 0
    with rasterio.open(in_raster) as src:
        px_size_x = abs(src.transform.a)
        tile_px = max(1, int(round(tile_size_m / px_size_x)))
        W, H = src.width, src.height
        cols = math.ceil(W / tile_px)
        rows = math.ceil(H / tile_px)
        total = cols * rows
        aoi_mask = None
        if aoi_path_str:
            try:
                import geopandas as gpd
                from shapely.ops import unary_union
            except Exception as e:  # pragma: no cover - optional dependency guard
                raise RuntimeError("AOI filtering requires GeoPandas/Shapely.") from e
            aoi_gdf = gpd.read_file(aoi_path_str)
            if len(aoi_gdf) > 0:
                if src.crs and aoi_gdf.crs and str(src.crs) != str(aoi_gdf.crs):
                    aoi_gdf = aoi_gdf.to_crs(src.crs)
                geoms = [
                    geom.buffer(0)
                    for geom in aoi_gdf.geometry
                    if geom is not None and not geom.is_empty
                ]
                if geoms:
                    # Rasterized AOI mask in full image space (1 inside AOI, 0 outside)
                    union_geom = unary_union(geoms)
                    aoi_mask = rio_features.rasterize(
                        [(union_geom, 1)],
                        out_shape=(H, W),
                        transform=src.transform,
                        fill=0,
                        all_touched=False,
                        dtype="uint8",
                    )

        if progress:
            progress(0, total, "Initializing tiles...")

        for r in range(rows):
            for c in range(cols):
                x0 = c * tile_px
                y0 = r * tile_px
                w = min(tile_px, W - x0)
                h = min(tile_px, H - y0)
                if w <= 0 or h <= 0:
                    done += 1
                    if progress:
                        progress(done, total, None or "")
                    continue

                window = rasterio.windows.Window(x0, y0, w, h)
                transform = src.window_transform(window)

                # Skip tiles that have no AOI pixels at all
                tile_aoi = None
                if aoi_mask is not None:
                    tile_aoi = aoi_mask[int(y0) : int(y0 + h), int(x0) : int(x0 + w)]
                    if tile_aoi.size == 0 or not np.any(tile_aoi):
                        done += 1
                        if progress:
                            progress(done, total, "Skipping tile outside AOI.")
                        continue

                ranger = _ranger_from_band1(src, window)

                tile_path = temp_dir / f"tile_{r:03d}_{c:03d}.tif"
                profile = src.profile.copy()
                profile.update({
                    "height": int(h),
                    "width": int(w),
                    "transform": transform,
                    "driver": "GTiff",
                })
                with rasterio.open(tile_path, "w", **profile) as dst:
                    for b in range(1, src.count + 1):
                        data = src.read(b, window=window)
                        # If AOI is provided, zero-out pixels outside AOI in this tile
                        if tile_aoi is not None:
                            mask = tile_aoi == 0
                            if mask.any():
                                data = data.copy()
                                data[mask] = 0
                        dst.write(data, b)

                # Write per-tile segments into Temp
                seg_path = temp_dir / f"seg_{r:03d}_{c:03d}.shp"
                ras_path = temp_dir / f"ras_{r:03d}_{c:03d}.tif"

                in_arg = tile_path.as_posix()
                vec_out_arg = seg_path.as_posix()
                ras_out_arg = ras_path.as_posix()
                raster_args = [
                    "-in", in_arg,
                    "-spatialr", str(spatialr),
                    "-ranger", f"{ranger}",
                    "-minsize", str(minsize),
                    "-mode", "raster",
                    "-mode.raster.out", ras_out_arg,
                ]
                vector_args = [
                    "-in", in_arg,
                    "-spatialr", str(spatialr),
                    "-ranger", f"{ranger}",
                    "-minsize", str(minsize),
                    "-mode", "vector",
                    "-mode.vector.out", vec_out_arg,
                ]

                note = f"Tile {done+1}/{total} | ranger={ranger:.3f}"
                if progress:
                    progress(done, total, note)

                if otb_bin.lower().endswith(".bat"):
                    # Use 'call' so batch files can invoke other scripts and return
                    raster_cmd = ["cmd", "/c", "call", otb_bin] + raster_args
                    vector_cmd = ["cmd", "/c", "call", otb_bin] + vector_args
                else:
                    raster_cmd = [otb_bin] + raster_args
                    vector_cmd = [otb_bin] + vector_args

                # Log output per tile to aid debugging
                logs_dir = temp_dir / "logs"
                logs_dir.mkdir(exist_ok=True)
                log_fp = logs_dir / f"seg_{r:03d}_{c:03d}.log"
                try:
                    # RASTER RUN
                    proc_r = subprocess.run(raster_cmd, capture_output=True, text=True)
                    with log_fp.open("w", encoding="utf-8", errors="ignore") as lf:
                        lf.write("RASTER COMMAND:\n" + " ".join(raster_cmd) + "\n\n")
                        lf.write("STDOUT:\n" + (proc_r.stdout or "") + "\n\n")
                        lf.write("STDERR:\n" + (proc_r.stderr or "") + "\n\n")
                    # VECTOR RUN
                    proc_v = subprocess.run(vector_cmd, capture_output=True, text=True)
                    with log_fp.open("a", encoding="utf-8", errors="ignore") as lf:
                        lf.write("VECTOR COMMAND:\n" + " ".join(vector_cmd) + "\n\n")
                        lf.write("STDOUT:\n" + (proc_v.stdout or "") + "\n\n")
                        lf.write("STDERR:\n" + (proc_v.stderr or "") + "\n")

                    if proc_r.returncode != 0 or proc_v.returncode != 0:
                        if progress:
                            comb = (proc_v.stderr or proc_v.stdout or proc_r.stderr or proc_r.stdout or "").strip().splitlines()[-1:] or ["error"]
                            progress(done, total, f"Failed tile {r},{c}: {comb[0][:160]}")
                    else:
                        # Some installs may exit 0 but not write output on parameter mismatch
                        if not seg_path.exists() or not ras_path.exists():
                            if progress:
                                progress(done, total, f"No output for tile {r},{c} (see logs)")
                except Exception as e:
                    with log_fp.open("a", encoding="utf-8", errors="ignore") as lf:
                        lf.write(f"EXCEPTION: {e}\n")
                    if progress:
                        progress(done, total, f"Exception tile {r},{c}: {e}")

                done += 1
                if progress:
                    progress(done, total, note)

    merged = None
    merged = None
    # Merge all per-tile shapefiles from Temp into the Run folder's out_shp
    segs_paths = sorted(temp_dir.glob("seg_*.shp"))
    if segs_paths:
        try:
            # Prefer Python merge via GeoPandas (available in env)
            import geopandas as gpd

            gdfs = []
            for p in segs_paths:
                try:
                    gdfs.append(gpd.read_file(p))
                except Exception:
                    continue

            if gdfs:
                merged_gdf = gpd.GeoDataFrame(
                    pd.concat(gdfs, ignore_index=True),
                    crs=gdfs[0].crs if hasattr(gdfs[0], "crs") else None,
                )

                # Drop polygons whose meanBx fields are zero
                mean_cols = [c for c in merged_gdf.columns if c.lower().startswith("mean")]
                if mean_cols:
                    keep_idx: list[int] = []
                    for idx, row in merged_gdf[mean_cols].iterrows():
                        drop = False
                        for col in mean_cols:
                            val = row[col]
                            try:
                                val_f = float(val)
                            except Exception:
                                continue
                            if not np.isfinite(val_f):
                                continue
                            if val_f == 0.0:
                                drop = True
                                break
                        if not drop:
                            keep_idx.append(idx)
                    if keep_idx:
                        merged_gdf = merged_gdf.iloc[keep_idx].reset_index(drop=True)
                    else:
                        merged_gdf = merged_gdf.iloc[0:0]

                merged_gdf = merged_gdf.reset_index(drop=True)
                merged_gdf["seg_id"] = np.arange(1, len(merged_gdf) + 1, dtype="int64")
                drop_cols = [c for c in merged_gdf.columns if c.lower().startswith(("var", "mean"))]
                if drop_cols:
                    merged_gdf = merged_gdf.drop(columns=drop_cols, errors="ignore")
                merged_gdf.to_file(out_shp)
                merged = out_shp
        except Exception:
            # Fallback to ogr2ogr if available
            try:
                ogr = shutil.which("ogr2ogr")
                if ogr:
                    segs = [str(p) for p in segs_paths]
                    subprocess.run([ogr, "-skipfailures", str(out_shp), segs[0]], check=True)
                    for shp in segs[1:]:
                        subprocess.run(
                            [ogr, "-skipfailures", "-update", "-append", str(out_shp), shp, "-nln", out_shp.stem],
                            check=True,
                        )
                    merged = out_shp
            except Exception:
                merged = None

    # Cleanup: remove entire Temp directory contents
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass

    # Cleanup: remove any TIFFs created at the chosen root (only our patterns)
    try:
        root_dir = base_out.parent if base_out.name.lower() == "output" else base_out
        for pattern in ("tile_*.tif", "ras_*.tif", "seg_*.tif"):
            for fp in root_dir.glob(pattern):
                try:
                    fp.unlink()
                except Exception:
                    pass
    except Exception:
        pass

    if merged is None:
        raise RuntimeError("Failed to produce merged segmentation shapefile. Check Temp/logs and ensure GeoPandas or GDAL (ogr2ogr) is available.")

    duration = time.time() - start
    return SegmentationResult(out_dir=out_dir, out_shp=merged, tiles_total=done, duration_sec=duration)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Run OTB-based segmentation on a raster")
    p.add_argument("--in", dest="in_raster", required=True, help="Input raster (GeoTIFF)")
    p.add_argument("--out-dir", dest="out_dir", required=True, help="Output root directory")
    p.add_argument("--otb", dest="otb", required=True, help="Path to OTB LargeScaleMeanShift (.bat/.exe)")
    p.add_argument("--tile-size-m", type=int, default=40)
    p.add_argument("--spatialr", type=int, default=5)
    p.add_argument("--minsize", type=int, default=5)
    args = p.parse_args()

    def cli_progress(done: int, total: int, note: str = ""):
        pct = 0 if total == 0 else int(100 * done / total)
        msg = f"[{pct:3d}%] {done}/{total}"
        if note:
            msg += f" | {note}"
        print(msg, flush=True)

    res = run_segmentation(
        in_raster=args.in_raster,
        output_root=args.out_dir,
        otb_bin=args.otb,
        tile_size_m=args.tile_size_m,
        spatialr=args.spatialr,
        minsize=args.minsize,
        progress=cli_progress,
    )
    print(f"Done. Output: {res.out_dir}")
