"""reports/results.json and docs/REPORT.md, generated from a run. Do not edit them by hand."""

from __future__ import annotations

import json
import time
from typing import Any

from ..workloads import ROOT, load
from .common import run_dir, write_json
from .gate import check

POLICY_LABEL = {
    "static": "static (sizes are constants)",
    "auto": "auto (sizes that changed become symbols)",
    "symbolic": "symbolic (sizes are symbols from the first call)",
}


def _load(p) -> Any:
    return json.loads(p.read_text()) if p.exists() else None


def collect(profile: str) -> dict[str, Any]:
    rd = run_dir(profile)
    r: dict[str, Any] = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "profile": profile}
    st = _load(rd / "study.json")
    if st:
        wl = load("workloads")["workloads"]
        r["workloads"] = {n: {"about": w["about"], "model": w["model"]} for n, w in wl.items()}
        r["rows"] = st["rows"]
        r["fuzz"] = st["fuzz"]
        r["necessity"] = st["necessity"]
        by = {(x["workload"], x["policy"]): x for x in st["rows"]}
        sym = [x for x in st["rows"] if x["policy"] == "symbolic"]
        r["summary"] = {
            "max_abs_diff": max(x["max_abs_diff"] for x in st["rows"]),
            "fuzz_violations": sum(x["violations"] for x in st["fuzz"]),
            "fuzz_draws": sum(x["draws"] for x in st["fuzz"]),
            "symbolic_minus_static_max": max(
                by[(x["workload"], "symbolic")]["compiles"] - by[(x["workload"], "static")]["compiles"] for x in sym
            ),
            "decode_symbolic_compiles": by.get(("greedy_decode", "symbolic"), {}).get("compiles"),
            "symbolic_max_compiles": max(x["compiles"] for x in sym),
            "symbolic_max_prologue_us": max(x["prologue_us_per_call"] for x in sym),
            "calls": sum(x["calls"] for x in st["rows"] if x["policy"] == "static"),
            "static_compiles": sum(x["compiles"] for x in st["rows"] if x["policy"] == "static"),
            "symbolic_compiles": sum(x["compiles"] for x in sym),
            "static_compile_ms": sum(x["compile_ms"] for x in st["rows"] if x["policy"] == "static"),
            "symbolic_compile_ms": sum(x["compile_ms"] for x in sym),
        }
    held = _load(rd / "heldout.json") or {}
    r["sets"] = {}
    for name, res in held.items():
        first = _load(ROOT / "reports" / "heldout" / f"{name}_first_run.json") if profile == "full" else None
        src = first or res  # a set's blind result is its first run, even after it became a regression set
        rows = src["rows"]
        supported = [x for x in rows if x["policies"]["symbolic"]["unsupported"] is None]
        r["sets"][name] = {
            "status": res["status"],
            "recorded_at": (first or {}).get("recorded_at"),
            "programs": len(rows),
            "supported": len(supported),
            "supported_share": len(supported) / max(1, len(rows)),
            "wrong_or_violations": sum(x["policies"][p]["wrong_calls"] for x in rows for p in x["policies"])
            + sum(x.get("fuzz", {}).get("violations", 0) for x in rows),
            "static_compiles": sum(x["policies"]["static"]["compiles"] for x in supported),
            "auto_compiles": sum(x["policies"]["auto"]["compiles"] for x in supported),
            "symbolic_compiles": sum(x["policies"]["symbolic"]["compiles"] for x in supported),
            "calls": sum(x["calls"] for x in supported),
            "rows": rows,
            "blind_first_run": first is not None,
            "rerun": {
                "supported": sum(1 for x in res["rows"] if x["policies"]["symbolic"]["unsupported"] is None),
                "wrong_or_violations": sum(x["policies"][p]["wrong_calls"] for x in res["rows"] for p in x["policies"])
                + sum(x.get("fuzz", {}).get("violations", 0) for x in res["rows"]),
            },
        }
    current = load("experiment").get("heldout_set", "v2")
    if current in r["sets"]:
        r["heldout"] = r["sets"][current]
    bridge = _load(ROOT / "reports" / "thunder_bridge.json")
    if bridge:
        r["thunder"] = bridge
    return r


def program_examples() -> dict[str, Any]:
    """The symbolic program each workload compiles on its first call (for reading)."""
    import torch

    from .. import jit
    from ..workloads import make_inputs, make_model, names, shapes

    out = {}
    for name in names():
        m = make_model(name)
        c = jit(m, policy="symbolic")
        b, t = shapes(name)[0]
        with torch.no_grad():
            c(*make_inputs(name, b, t, 0))
        p = c.last_program()
        lines = p.prologue_src.splitlines()
        cut = next((i for i, ln in enumerate(lines) if ln.strip() == "return None"), 0) + 1
        n_inputs = lines[0].count(",") + 1
        out[name] = {
            "prologue_checks": "\n".join(lines[cut:]),
            "computation": p.computation_src,
            "guards": p.guards,
            "symbols": p.symbols,
            "inputs": n_inputs,
            "nodes": p.n_nodes,
            "nodes_traced": p.n_nodes_traced,
        }
    return out


def write_reports(profile: str) -> dict[str, Any]:
    r = collect(profile)
    r["examples"] = program_examples()
    rel = load("release")
    r["gate"] = {"floors": check(r, rel["floors"]), "targets": check(r, rel["targets"])}
    out = ROOT / "reports" if profile == "full" else run_dir(profile)
    write_json(out / "results.json", r)
    target = ROOT / "docs" / "REPORT.md" if profile == "full" else out / "REPORT.md"
    target.write_text(markdown(r))
    return r


def _f(v: Any, d: int = 1) -> str:
    return "n/a" if v is None else f"{v:.{d}f}"


def markdown(r: dict[str, Any]) -> str:
    parts = [f"# Report\n\nGenerated by `stc report` ({r['generated_at']}, profile `{r['profile']}`). Do not edit.\n"]
    if "rows" in r:
        s = r["summary"]
        parts.append(
            f"## Recompilations\n\n{s['calls']} calls over {len(r['workloads'])} workloads, each replayed under every policy "
            f"and compared with eager PyTorch call by call (largest difference: {s['max_abs_diff']:.1e}).\n"
        )
        lines = [
            "| Workload | Policy | Calls | Distinct shapes | Programs compiled | Compile time (ms) | Prologue (µs/call) | Eager (ms) | Compiled path (ms) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for x in r["rows"]:
            lines.append(
                f"| {x['workload']} | {x['policy']} | {x['calls']} | {x['distinct_shapes']} | {x['compiles']} | {x['compile_ms']:.0f} | "
                f"{x['prologue_us_per_call']:.1f} | {x['eager_ms']:.0f} | {x['compiled_ms']:.0f} |"
            )
        parts.append(
            "\n".join(lines) + "\n\nThe compiled path runs the same PyTorch calls as eager (no fusion), so its time over eager "
            "is the cost of compiling and checking: this study measures the cache, not kernels.\n"
        )
        lines = [
            "| Workload | Policy | Draws | Eager accepted | Program accepted | Violations | Coverage |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for x in r["fuzz"]:
            lines.append(
                f"| {x['workload']} | {x['policy']} | {x['draws']} | {x['eager_ok']} | {x['accepted']} | {x['violations']} | {100 * x['coverage']:.0f}% |"
            )
        parts.append(
            "## Are the guards sound?\n\nAfter each workload, its cached programs are fuzzed with new sizes (some keep the relations "
            "the trace found, some break them; sizes from 1 to well past any limit). A violation is an accepted draw on which the "
            "program disagrees with eager. Coverage is the share of draws eager accepts that some cached program accepts.\n\n"
            + "\n".join(lines)
            + "\n"
        )
        lines = []
        for w, checks in r["necessity"].items():
            for c in checks:
                lines.append(f"| {w} | `{c['check']}` | {c['without_it']} |")
        parts.append(
            "## What each size check protects\n\nEach check of the symbolic program was removed in turn and the same draws "
            'fuzzed again: "wrong results" means the weakened program then returned wrong values; "errors" means it accepted '
            'sizes that fail inside the computation instead of missing the cache; "none seen" means neither happened in these draws.\n\n'
            "| Workload | Check | Without it |\n|---|---|---|\n" + "\n".join(lines) + "\n"
        )
    for name, h in r.get("sets", {}).items():
        lines = [
            "| Program | What it is | Calls | Static | Auto | Symbolic | Wrong calls | Fuzz violations | Unsupported |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for x in h["rows"]:
            p = x["policies"]
            wrong = sum(p[k]["wrong_calls"] for k in p)
            lines.append(
                f"| {x['program']} | {x['about']} | {x['calls']} | {p['static']['compiles']} | {p['auto']['compiles']} | "
                f"{p['symbolic']['compiles']} | {wrong} | {x.get('fuzz', {}).get('violations', 'n/a')} | {p['symbolic']['unsupported'] or ''} |"
            )
        rr = h["rerun"]
        head = (
            f"## Held-out programs, set {name}: first run (blind)\n\nWritten from `heldout/{name}/WRITING_PROGRAMS.md` by someone who had not "
            f"seen the compiler; first run {h.get('recorded_at')}, kept as recorded. Status now: {h['status']} (`heldout/{name}/STATUS`); "
            f"rerun today: {rr['supported']} of {h['programs']} supported, {rr['wrong_or_violations']} wrong calls or violations.\n\n"
            if h["blind_first_run"]
            else f"## Programs, set {name} (regression)\n\nWritten blind, then studied before a first run was recorded (see "
            f"`heldout/{name}/STATUS`); rerun every time.\n\n"
        )
        parts.append(head + "\n".join(lines) + "\n")
        nec = [f"| {x['program']} | `{c['check']}` | {c['without_it']} |" for x in h["rows"] for c in x.get("necessity", [])]
        if nec:
            parts.append(
                f"Size checks of set {name}, removed one at a time:\n\n| Program | Check | Without it |\n|---|---|---|\n"
                + "\n".join(nec)
                + "\n"
            )
    t = r.get("thunder")
    if t:
        lines = [
            "| Workload | System | Calls | Distinct shapes | Programs compiled | Wall time (s) | Largest difference | Error |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for x in t["rows"]:
            lines.append(
                f"| {x['workload']} | {x['system']} | {x['calls']} | {x['distinct_shapes']} | {x['compiles']} | {x['wall_s']:.1f} | "
                f"{x['max_abs_diff']:.1e} | {x['error'] or ''} |"
            )
        m = t["meta"]
        parts.append(
            f"## Thunder on the same calls\n\nMeasured in CI (run {t.get('run', 'n/a')}, commit {t.get('commit', 'n/a')[:8]}): "
            f"torch {m['torch']}, lightning-thunder {m['thunder']}, {m['machine']}, workload scale {m['scale']}, CPU.\n\n"
            + "\n".join(lines)
            + "\n"
        )
    g = r.get("gate")
    if g:
        fmt = lambda v: "n/a" if v is None else (f"{v:.3g}" if isinstance(v, float) else str(v))  # noqa: E731
        parts.append(
            "## Release gate\n\n| | Check | Actual | Required |\n|---|---|---:|---:|\n"
            + "\n".join(
                f"| {'pass' if x['ok'] else '**fail**'} | {x['name']} | {fmt(x['actual'])} | {x['op']} {x['value']} |" for x in g["floors"]
            )
            + "\n\nTargets (reported, not gated):\n\n| | Target | Actual | Aim |\n|---|---|---:|---:|\n"
            + "\n".join(
                f"| {'met' if x['ok'] else 'missed'} | {x['name']} | {fmt(x['actual'])} | {x['op']} {x['value']} |" for x in g["targets"]
            )
            + "\n"
        )
    return "\n".join(parts)


__all__ = ["write_reports", "collect", "POLICY_LABEL"]
