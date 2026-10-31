"""Thunder on the same calls (run in CI: needs torch and lightning-thunder versions that agree).

For each inference workload, replay the same calls through:
- `thunder.jit(model)` (its default cache, "constant values": sizes are constants),
- `thunder.jit(model, cache="symbolic values")`,
- `stc.jit(model, policy="symbolic")`,
and record compiles (cache misses), wall time and agreement with eager. Same process, same
machine, same calls. Prints one `::notice` line per workload so the numbers reach the CI annotations,
and writes runs/thunder_bridge.json.

    python scripts/thunder_bridge.py --scale 0.25
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time

import torch

import stc
from stc.workloads import is_train, make_inputs, make_model, names, shapes


def maxdiff(a, b) -> float:
    return float((a.float() - b.float()).abs().max()) if isinstance(a, torch.Tensor) else max(maxdiff(x, y) for x, y in zip(a, b))


def replay(kind: str, name: str, scale: float) -> dict:
    import thunder

    model = make_model(name)
    if kind == "thunder-constant":
        f = thunder.jit(model)
    elif kind == "thunder-symbolic":
        f = thunder.jit(model, cache="symbolic values")
    else:
        f = stc.jit(model, policy="symbolic")
    calls = shapes(name, scale)
    worst, wall, err = 0.0, 0.0, None
    try:
        with torch.no_grad():
            for i, (b, t) in enumerate(calls):
                a = make_inputs(name, b, t, i)
                ref = model(*a)
                t0 = time.perf_counter()
                out = f(*a)
                wall += time.perf_counter() - t0
                worst = max(worst, maxdiff(out, ref))
    except Exception as e:  # report, do not hide
        err = f"{type(e).__name__}: {str(e)[:600]}"
    if kind.startswith("thunder"):
        compiles = thunder.cache_misses(f)
    else:
        compiles = f.stats.compiles
    return {
        "workload": name,
        "system": kind,
        "calls": len(calls),
        "distinct_shapes": len(set(calls)),
        "compiles": compiles,
        "wall_s": wall,
        "max_abs_diff": worst,
        "error": err,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--out", default="runs/thunder_bridge.json")
    a = ap.parse_args()
    import thunder

    meta = {
        "torch": torch.__version__,
        "thunder": getattr(thunder, "__version__", "unknown"),
        "python": sys.version.split()[0],
        "machine": platform.platform(),
        "scale": a.scale,
    }
    rows = []
    for name in names():
        if is_train(name):
            continue
        group = [replay(kind, name, a.scale) for kind in ("thunder-constant", "thunder-symbolic", "stc-symbolic")]
        rows += group
        # GitHub keeps at most 10 annotations per step: one per workload, each carrying the metadata
        print(f"::notice title=thunder-bridge::{json.dumps({'meta': meta, 'rows': group})}", flush=True)
    from pathlib import Path

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
