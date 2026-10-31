"""The release gate over the generated results."""

from __future__ import annotations

import json
import operator
from typing import Any

from ..workloads import ROOT, load
from .common import log

OPS = {">=": operator.ge, "<=": operator.le, "<": operator.lt, ">": operator.gt, "==": operator.eq}


def lookup(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def check(results: dict[str, Any], items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for it in items:
        v = lookup(results, it["path"])
        out.append({**it, "actual": v, "ok": v is not None and bool(OPS[it["op"]](v, it["value"]))})
    return out


def gate(profile: str = "full") -> bool:
    path = ROOT / "reports" / "results.json" if profile == "full" else ROOT / "runs" / profile / "results.json"
    r = json.loads(path.read_text())
    rel = load("release")
    floors, targets = check(r, rel["floors"]), check(r, rel["targets"])
    for f in floors:
        log(f"{'PASS' if f['ok'] else 'FAIL'}  {f['name']}: {f['actual']} {f['op']} {f['value']}")
    for t in targets:
        log(f"{'met ' if t['ok'] else 'miss'}  target {t['name']}: {t['actual']} {t['op']} {t['value']}")
    applicable = floors if profile == "full" else [f for f in floors if not f["path"].startswith("heldout.")]
    ok = all(f["ok"] for f in applicable)
    log("release gate: " + ("PASS" if ok else "FAIL"))
    return ok
