"""The whole run: study, held-out, report, gate. Full-profile outputs are committed."""

from __future__ import annotations

import time
from typing import Any

from .common import log, run_dir, write_json


def step_study(profile: str) -> dict[str, Any]:
    from .study import run_study

    res = run_study(profile)
    write_json(run_dir(profile) / "study.json", res)
    return res


def step_heldout(profile: str) -> dict[str, Any]:
    from .heldout import run_heldout

    return run_heldout(profile)


def step_report(profile: str) -> dict[str, Any]:
    from .report import write_reports

    return write_reports(profile)


def step_gate(profile: str) -> bool:
    from .gate import gate

    return gate(profile)


STEPS = {"study": step_study, "heldout": step_heldout, "report": step_report, "gate": step_gate}
ORDER = ("study", "heldout", "report")


def run_all(profile: str) -> bool:
    for name in ORDER:
        t = time.time()
        STEPS[name](profile)
        log(f"step {name} done in {time.time() - t:.0f}s")
    return step_gate(profile)
