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
    return Session(images=data.get("images", []), output_dir=data.get("output_dir"))

