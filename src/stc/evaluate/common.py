from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..workloads import ROOT, load


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_dir(profile: str) -> Path:
    d = ROOT / "runs" / profile
    d.mkdir(parents=True, exist_ok=True)
    return d


def prof(profile: str) -> dict[str, Any]:
    return load("experiment")["profiles"][profile]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))
