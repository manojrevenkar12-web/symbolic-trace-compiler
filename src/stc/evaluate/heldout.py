"""The blind held-out programs: written from heldout/WRITING_PROGRAMS.md by someone who had not
seen the compiler. Each program's calls are replayed under every policy and checked against eager;
the compiled program is then fuzzed and its size checks mutation-tested. The first full run is
kept in reports/heldout/first_run.json and never rewritten."""

from __future__ import annotations

import hashlib
import importlib.util
import time
from pathlib import Path
from typing import Any

import torch

from .. import jit
from ..trace import Unsupported
from ..verify import fuzz, necessity
from ..workloads import ROOT, load
from .common import log, prof, run_dir, write_json

SETS = ROOT / "heldout"


def status(name: str) -> str:
    f = SETS / name / "STATUS"
    return f.read_text().split()[0] if f.exists() else "heldout"


def first_run_path(name: str) -> Path:
    return ROOT / "reports" / "heldout" / f"{name}_first_run.json"


def programs_sha(name: str) -> str:
    h = hashlib.sha256()
    for p in sorted((SETS / name / "programs").glob("p*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def load_program(path: Path):
    spec = importlib.util.spec_from_file_location(f"heldout_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build()


def _same(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return (
            isinstance(b, torch.Tensor)
            and a.shape == b.shape
            and a.dtype == b.dtype
            and (torch.allclose(a, b, rtol=1e-4, atol=1e-5, equal_nan=True) if a.is_floating_point() else torch.equal(a, b))
        )
    if isinstance(a, (tuple, list)):
        return isinstance(b, (tuple, list)) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def run_program(path: Path, draws: int) -> dict[str, Any]:
    program, calls, about = load_program(path)
    out: dict[str, Any] = {"program": path.stem, "about": about, "calls": len(calls), "policies": {}}
    for policy in load("experiment")["policies"]:
        c = jit(program, policy=policy)
        wrong, unsupported, crashed = 0, None, None
        try:
            with torch.no_grad():
                for args in calls:
                    if not _same(c(*args), program(*args)):
                        wrong += 1
        except Unsupported as e:
            unsupported = str(e)
        except Exception as e:  # a tracer bug: recorded as a result, not hidden
            crashed = f"{type(e).__name__}: {e}"[:300]
        out["policies"][policy] = {
            "compiles": c.stats.compiles,
            "wrong_calls": wrong,
            "unsupported": unsupported or crashed,
            "crashed": crashed,
            "guards": c.programs()[0].guards if c.entries else [],
        }
        if policy == "symbolic" and unsupported is None and crashed is None and c.entries:
            rep = fuzz(c, calls[0], n=draws, seed=5, cap=128)
            out["fuzz"] = rep.as_dict()
            out["necessity"] = necessity(c, calls[0], n=draws, seed=6, cap=128)
    return out


def run_set(name: str, profile: str) -> dict[str, Any]:
    files = sorted((SETS / name / "programs").glob("p*.py"))
    draws = prof(profile)["heldout_draws"]
    rows = [run_program(p, draws) for p in files]
    for r in rows:
        s = r["policies"]["symbolic"]
        log(
            f"{name} {r['program']}: symbolic {s['compiles']} compiles, {s['wrong_calls']} wrong calls"
            f"{', unsupported: ' + s['unsupported'] if s['unsupported'] else ''}; static {r['policies']['static']['compiles']}"
            f"; fuzz violations {r.get('fuzz', {}).get('violations')}"
        )
    return {"set": name, "status": status(name), "programs_sha256": programs_sha(name), "rows": rows}


def run_heldout(profile: str) -> dict[str, Any]:
    """Every set under heldout/: regression sets are rerun each time; a held-out set's first
    full run is recorded once and its programs may not change afterwards."""
    import json

    out = {}
    t0 = time.time()
    for d in sorted(p for p in SETS.iterdir() if (p / "programs").is_dir()):
        if not list((d / "programs").glob("p*.py")):
            log(f"{d.name}: no programs yet")
            continue
        first = first_run_path(d.name)
        if (
            profile == "full"
            and status(d.name) == "heldout"
            and first.exists()
            and json.loads(first.read_text())["programs_sha256"] != programs_sha(d.name)
        ):
            raise SystemExit(f"heldout/{d.name}/programs changed after its first run; a changed set is a new set (AGENTS.md)")
        res = run_set(d.name, profile)
        out[d.name] = res
        if profile == "full" and res["status"] == "heldout" and not first.exists():
            write_json(first, {"recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **res})
            log(f"{d.name}: first run recorded")
    write_json(run_dir(profile) / "heldout.json", out)
    log(f"held-out done in {time.time() - t0:.0f}s")
    return out
