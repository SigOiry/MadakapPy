from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional


SESSION_FILENAME = "session.json"


@dataclass
class Session:
    images: List[str]
    output_dir: Optional[str] = None
    # Segmentation parameters
    seg_in_raster: Optional[str] = None
    seg_tile_size_m: int = 40
    seg_spatialr: int = 5
    seg_minsize: int = 5
    seg_otb_bin: Optional[str] = None
    biomass_model: str = "madagascar"


def save_session(session: Session, location: Path | None = None) -> Path:
    target = Path(location) if location else Path.cwd()
    target.mkdir(parents=True, exist_ok=True)
    out = target / SESSION_FILENAME
    with out.open("w", encoding="utf-8") as f:
        json.dump(asdict(session), f, indent=2)
    return out


def load_session(location: Path | None = None) -> Optional[Session]:
    base = Path(location) if location else Path.cwd()
    fp = base / SESSION_FILENAME
    if not fp.exists():
        return None
    with fp.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Session(
        images=data.get("images", []),
        output_dir=data.get("output_dir"),
        seg_in_raster=data.get("seg_in_raster"),
        seg_tile_size_m=int(data.get("seg_tile_size_m", 40)),
        seg_spatialr=int(data.get("seg_spatialr", 5)),
        seg_minsize=int(data.get("seg_minsize", 5)),
        seg_otb_bin=data.get("seg_otb_bin"),
        biomass_model=(data.get("biomass_model") or "madagascar"),
    )
