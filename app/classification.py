from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from collections import Counter, defaultdict

import joblib
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features as rio_features
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


Progress = Callable[[int, int, str], None]


@dataclass
class TrainResult:
    model_path: Path
    classes: List[str]
    duration_sec: float
    cm: Optional[np.ndarray] = None
    cm_labels: Optional[List[str]] = None
    cm_path: Optional[Path] = None


@dataclass
class ApplyResult:
    output_path: Path
    duration_sec: float


def _model_dir(base_out: Path) -> Path:
    if base_out.name.lower() == "output":
        return base_out.parent / "Model"
    return base_out / "Model"


def _rf_features_for_polygon(src: rasterio.io.DatasetReader, geom) -> np.ndarray:
    data, _ = rio_mask(src, [mapping(geom)], crop=True, filled=False)
    data = data.astype("float32")
    data[(data == 0) | (data == 65535)] = np.nan
    finite = np.isfinite(data)
    if finite.any():
        max_val = float(np.nanmax(data))
        if max_val > 1.0:
            data /= 65535.0
    means = [np.nanmean(band) for band in data]
    # Handle empty (all-NaN) polygons
    means = [0.0 if not np.isfinite(v) else float(v) for v in means]
    return np.array(means, dtype=np.float32)


def extract_features(
    raster_path: str | os.PathLike,
    segments_path: str | os.PathLike,
    poly_indexes: Optional[Sequence[int]] = None,
    progress: Optional[Progress] = None,
) -> Tuple[gpd.GeoDataFrame, np.ndarray]:
    gdf = gpd.read_file(segments_path).copy()
    gdf = gdf.reset_index(drop=True)
    if "seg_id" not in gdf.columns:
        gdf["seg_id"] = np.arange(1, len(gdf) + 1, dtype=np.int32)
    if poly_indexes is not None:
        gdf = gdf.iloc[list(poly_indexes)].copy().reset_index(drop=True)

    feats: List[np.ndarray] = []
    start = time.time()
    with rasterio.open(raster_path) as src:
        total = len(gdf)
        for i, geom in enumerate(gdf.geometry):
            v = _rf_features_for_polygon(src, geom)
            feats.append(v)
            if progress and ((i % 10) == 0 or i + 1 == total):
                progress(i + 1, total, f"Extracting features {i+1}/{total}")
    X = np.vstack(feats) if feats else np.empty((0, 0), dtype=np.float32)
    return gdf, X


def train_model(
    raster_path: str | os.PathLike,
    segments_path: str | os.PathLike,
    labels: Dict[int, str],
    output_root: str | os.PathLike,
    progress: Optional[Progress] = None,
) -> TrainResult:
    # labels map: polygon index -> class string
    idxs = sorted(labels.keys())
    if not idxs:
        raise ValueError("No labeled polygons provided")

    gdf, X = extract_features(raster_path, segments_path, poly_indexes=idxs, progress=progress)
    y_str = [labels[i] for i in idxs]

    le = LabelEncoder()
    y = le.fit_transform(y_str)

    if X.size == 0:
        raise RuntimeError("No features extracted; check raster/segments overlap")

    if progress:
        progress(0, 1, "Training Random Forest…")

    start = time.time()
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    dur = time.time() - start

    base_out = Path(output_root)
    mdir = _model_dir(base_out)
    mdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    mpath = mdir / f"RF_{ts}.joblib"

    payload = {
        "model": rf,
        "label_encoder": le,
        "bands": X.shape[1],
    }
    joblib.dump(payload, mpath)
    return TrainResult(model_path=mpath, classes=list(le.classes_), duration_sec=dur)


def train_model_from_training_polys(
    raster_path: str | os.PathLike,
    training_polys_path: str | os.PathLike,
    class_column: str,
    output_root: str | os.PathLike,
    progress: Optional[Progress] = None,
    max_pixels_per_class: Optional[int] = None,
    random_state: int = 42,
) -> TrainResult:
    gdf = gpd.read_file(training_polys_path)
    if class_column not in gdf.columns:
        raise ValueError(f"Column '{class_column}' not found in training polygons")
    # Drop rows with missing labels
    gdf = gdf.dropna(subset=[class_column]).reset_index(drop=True)
    if len(gdf) == 0:
        raise ValueError("No labeled polygons after dropping missing labels")

    # Per-pixel extraction: for each polygon, extract every pixel's band values
    X_chunks: list[np.ndarray] = []
    y_chunks: list[str] = []
    with rasterio.open(raster_path) as src:
        # Reproject training polygons to raster CRS if needed
        try:
            if gdf.crs and src.crs and str(gdf.crs) != str(src.crs):
                gdf = gdf.to_crs(src.crs)
        except Exception:
            pass
        total = len(gdf)
        for i, row in enumerate(gdf.itertuples(index=False)):
            geom = getattr(row, 'geometry')
            label = str(getattr(row, class_column))
            try:
                data, _ = rio_mask(src, [mapping(geom)], crop=True, filled=False)
                # data: (bands, h, w) masked array
                if np.ma.isMaskedArray(data):
                    arr = data
                else:
                    arr = np.ma.array(data, mask=False)
                # Replace sentinel values with mask
                mask_bad = np.zeros(arr.shape[1:], dtype=bool)
                for b in range(arr.shape[0]):
                    band = np.array(arr[b], dtype=np.float32)
                    bad = (band == 0) | (band == 65535) | ~np.isfinite(band)
                    mask_bad |= np.array(bad)
                mask_any = np.array(np.ma.getmaskarray(arr)).any(axis=0) | mask_bad
                if mask_any.size == 0:
                    continue
                valid_idx = np.where(~mask_any)
                if valid_idx[0].size == 0:
                    continue
                # Gather per-pixel vectors
                pixels = np.stack([np.array(arr[b], dtype=np.float32)[valid_idx] for b in range(arr.shape[0])], axis=1)
                X_chunks.append(pixels)
                y_chunks.extend([label] * pixels.shape[0])
            except Exception:
                # Skip polygons that fail masking
                continue
            if progress and (i % 5 == 0 or i + 1 == total):
                progress(i + 1, total, f"Extracting pixels {i+1}/{total}")
    if not X_chunks:
        raise RuntimeError("No training pixels extracted; check training polygons and raster overlap")
    X = np.vstack(X_chunks)
    y_str = y_chunks

    # Optionally cap number of pixels per class by random sampling
    if max_pixels_per_class is not None and max_pixels_per_class > 0:
        if progress:
            progress(0, 1, f"Capping to {max_pixels_per_class} pixels per class…")
        # Build label encoder on full set first (preserves class order)
        le_tmp = LabelEncoder()
        y_full = le_tmp.fit_transform(y_str)
        idx_all = np.arange(len(y_full))
        rng = np.random.default_rng(random_state)
        keep_indices = []
        for cls_val in np.unique(y_full):
            cls_idx = idx_all[y_full == cls_val]
            if cls_idx.size > max_pixels_per_class:
                sel = rng.choice(cls_idx, size=max_pixels_per_class, replace=False)
                keep_indices.append(sel)
            else:
                keep_indices.append(cls_idx)
        if keep_indices:
            keep_idx = np.concatenate(keep_indices)
            # Shuffle to avoid class blocks
            rng.shuffle(keep_idx)
            X = X[keep_idx]
            y_str = [y_str[i] for i in keep_idx]

    le = LabelEncoder()
    y = le.fit_transform(y_str)

    # Ensure CV split count is valid after capping
    # Each class must have at least n_splits samples
    unique, counts = np.unique(y, return_counts=True)
    min_per_class = int(counts.min()) if counts.size else 0
    if min_per_class < 2:
        raise RuntimeError("Not enough samples per class after capping; increase the per-class limit or add data.")
    n_splits = min(5, min_per_class)

    # Hyperparameter tuning via randomized search
    if progress:
        progress(0, 1, "Tuning hyperparameters…")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    param_dist = {
        "n_estimators": [200, 300, 400, 500, 600],
        "max_depth": [None, 10, 20, 30],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }
    base_rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_dist,
        n_iter=20,
        cv=skf,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
        verbose=0,
        random_state=random_state,
    )
    start = time.time()
    search.fit(X, y)
    best_rf = search.best_estimator_
    # Cross-validated predictions for confusion matrix
    y_pred_cv = cross_val_predict(best_rf, X, y, cv=skf, n_jobs=-1)
    cm = confusion_matrix(y, y_pred_cv, labels=np.unique(y))
    dur = time.time() - start

    base_out = Path(output_root)
    mdir = _model_dir(base_out)
    mdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    mpath = mdir / f"RF_{ts}.joblib"

    payload = {
        "model": best_rf,
        "label_encoder": le,
        "bands": X.shape[1],
        "class_column": class_column,
    }
    joblib.dump(payload, mpath)

    # Save confusion matrix plot
    cm_path = mdir / f"RF_{ts}_cm.png"
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=list(le.classes_), yticklabels=list(le.classes_),
           ylabel="True label", xlabel="Predicted label",
           title="Confusion Matrix (CV)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Annotate cells
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(cm_path)
    plt.close(fig)

    return TrainResult(
        model_path=mpath,
        classes=list(le.classes_),
        duration_sec=dur,
        cm=cm,
        cm_labels=list(le.classes_),
        cm_path=cm_path,
    )


def apply_model_to_segments(
    raster_path: str | os.PathLike,
    segments_path: str | os.PathLike,
    model_path: str | os.PathLike,
    output_root: str | os.PathLike,
    progress: Optional[Progress] = None,
) -> ApplyResult:
    payload = joblib.load(model_path)
    rf: RandomForestClassifier = payload["model"]
    le: LabelEncoder = payload["label_encoder"]
    bands: int = payload["bands"]

    gdf_all, X = extract_features(raster_path, segments_path, poly_indexes=None, progress=progress)
    if X.shape[1] != bands:
        raise RuntimeError("Model expects a different number of bands than provided by raster")

    if progress:
        progress(0, 1, "Predicting classes for segments…")
    y_pred = rf.predict(X)
    classes = le.inverse_transform(y_pred)
    out_gdf = gdf_all.copy()
    out_gdf["Class"] = classes

    base_out = Path(output_root)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder = base_out / "Output" / f"2-RF/Run_{ts}" if base_out.name.lower() != "output" else base_out / f"2-RF/Run_{ts}"
    folder.mkdir(parents=True, exist_ok=True)
    out_path = folder / f"Classification_{ts}.shp"
    out_gdf.to_file(out_path)
    return ApplyResult(output_path=out_path, duration_sec=0.0)


def _sanitize_field_name(name: str, used: set[str]) -> str:
    # Keep alnum and underscores, lower-case; prefix with 'prop_'
    import re
    base = re.sub(r"[^0-9a-zA-Z_]+", "_", name).lower()
    if not base:
        base = "cls"
    # Shapefile field name limit ~10 chars; keep short
    prefix = "prop_"
    max_total = 10
    avail = max(1, max_total - len(prefix))
    trimmed = base[:avail]
    candidate = prefix + trimmed
    i = 1
    while candidate in used or len(candidate) > max_total:
        suffix = str(i)
        avail2 = max(1, max_total - len(prefix) - len(suffix))
        candidate = prefix + base[:avail2] + suffix
        i += 1
    used.add(candidate)
    return candidate


def apply_model_with_pixel_sampling(
    raster_path: str | os.PathLike,
    segments_path: str | os.PathLike,
    model_path: str | os.PathLike,
    output_root: str | os.PathLike,
    max_pixels_per_polygon: int = 200,
    progress: Optional[Progress] = None,
) -> ApplyResult:
    payload = joblib.load(model_path)
    rf: RandomForestClassifier = payload["model"]
    le: LabelEncoder = payload["label_encoder"]
    bands: int = payload["bands"]

    gdf = gpd.read_file(segments_path)

    # Pre-create columns for proportions
    used_fields: set[str] = set()
    cls_to_field: dict[str, str] = {}
    for cls in le.classes_:
        fld = _sanitize_field_name(str(cls), used_fields)
        cls_to_field[str(cls)] = fld
        gdf[fld] = 0.0
    # Also add count and majority fields
    if "n_samp" not in gdf.columns:
        gdf["n_samp"] = 0
    if "majority" not in gdf.columns:
        gdf["majority"] = None

    try:
        cap = int(max_pixels_per_polygon)
    except Exception:
        cap = 200
    if cap <= 0:
        cap = 1

    start = time.time()
    with rasterio.open(raster_path) as src:
        # Reproject segments to raster CRS if needed
        try:
            if hasattr(gdf, "crs") and gdf.crs and src.crs and str(gdf.crs) != str(src.crs):
                gdf = gdf.to_crs(src.crs)
        except Exception:
            pass

        height, width = src.height, src.width
        if progress:
            progress(0, 4, "Rasterizing segments")
        seg_id_grid = rio_features.rasterize(
            (
                (geom, int(seg_id))
                for geom, seg_id in zip(gdf.geometry, gdf["seg_id"])
                if geom is not None and not geom.is_empty
            ),
            out_shape=(height, width),
            transform=src.transform,
            fill=-1,
            dtype="int32",
        )

        if src.count < bands:
            raise RuntimeError(
                f"Raster has {src.count} band(s) but model expects {bands}."
            )

        sample_store: dict[int, np.ndarray] = {}
        sample_counts: dict[int, int] = defaultdict(int)
        total_seen: dict[int, int] = defaultdict(int)
        rng = np.random.default_rng(42)

        if progress:
            progress(1, 4, "Sampling pixels")

        indexes = list(range(1, bands + 1))
        for _, window in src.block_windows(1):
            data = src.read(indexes, window=window).astype(np.float32)
            if np.isfinite(data).any() and float(np.nanmax(data)) > 1.0:
                data /= 65535.0
            mask_bad = np.zeros((window.height, window.width), dtype=bool)
            for b in range(data.shape[0]):
                band = data[b]
                bad = (band == 0) | (band == 65535) | ~np.isfinite(band)
                mask_bad |= bad
            seg_block = seg_id_grid[
                window.row_off : window.row_off + window.height,
                window.col_off : window.col_off + window.width,
            ]
            valid = (~mask_bad) & (seg_block >= 0)
            if not valid.any():
                continue
            seg_vals = np.unique(seg_block[valid])
            for seg_val in seg_vals:
                mask_seg = valid & (seg_block == seg_val)
                pix = data[:, mask_seg].reshape(data.shape[0], -1).T
                if pix.size == 0:
                    continue
                store = sample_store.get(int(seg_val))
                if store is None:
                    store = np.zeros((cap, bands), dtype=np.float32)
                    sample_store[int(seg_val)] = store
                for row in pix:
                    total_seen[int(seg_val)] += 1
                    ts = total_seen[int(seg_val)]
                    if sample_counts[int(seg_val)] < cap:
                        store[sample_counts[int(seg_val)]] = row
                        sample_counts[int(seg_val)] += 1
                    else:
                        j = rng.integers(0, ts)
                        if j < cap:
                            store[j] = row

        sample_pixels: List[np.ndarray] = []
        sample_seg_ids: List[np.ndarray] = []
        for seg_id, store in sample_store.items():
            count = sample_counts.get(seg_id, 0)
            if count <= 0:
                continue
            sample_pixels.append(store[:count])
            sample_seg_ids.append(np.full(count, int(seg_id), dtype=np.int32))

        if sample_pixels:
            if progress:
                progress(2, 4, "Classifying samples")
            X = np.vstack(sample_pixels)
            seg_sample_ids = np.concatenate(sample_seg_ids)
            y_pred = rf.predict(X)
            classes = le.inverse_transform(y_pred)

            counts_by_seg: dict[int, Counter] = defaultdict(Counter)
            for seg_id, cls in zip(seg_sample_ids, classes):
                counts_by_seg[int(seg_id)][str(cls)] += 1

            if progress:
                progress(3, 4, "Aggregating results")

            seg_id_to_idx = {int(seg_id): idx for idx, seg_id in enumerate(gdf["seg_id"])}
            for seg_id, counter in counts_by_seg.items():
                total = sum(counter.values())
                if total <= 0:
                    continue
                row_idx = seg_id_to_idx.get(int(seg_id))
                if row_idx is None:
                    continue
                gdf.at[row_idx, "n_samp"] = int(total)
                for cls in le.classes_:
                    fld = cls_to_field[str(cls)]
                    gdf.at[row_idx, fld] = float(counter.get(str(cls), 0)) / float(total)
                maj = max(counter.items(), key=lambda kv: kv[1])[0]
                gdf.at[row_idx, "majority"] = str(maj)

    base_out = Path(output_root)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder = base_out / "Output" / f"2-RF/Run_{ts}" if base_out.name.lower() != "output" else base_out / f"2-RF/Run_{ts}"
    folder.mkdir(parents=True, exist_ok=True)
    keep_cols = []
    if "seg_id" in gdf.columns:
        keep_cols.append("seg_id")
    keep_cols.extend([c for c in cls_to_field.values()])
    keep_cols.append("majority")
    keep_cols = [c for c in keep_cols if c in gdf.columns]
    keep_cols.append("geometry")
    gdf = gdf[keep_cols]
    out_path = folder / f"Classification_{ts}.shp"
    gdf.to_file(out_path)
    return ApplyResult(output_path=out_path, duration_sec=time.time() - start)
