"""Import the Thunder comparison measured in CI into reports/thunder_bridge.json.

The CI job prints one `::notice title=thunder-bridge::<json>` line per result; GitHub keeps them
as check-run annotations. This script reads them for a given run and writes the file with the run
id and commit, so the committed numbers name where they were measured.

    python scripts/import_bridge.py <run_id>
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = "manojrevenkar12-web/symbolic-trace-compiler"


def gh(path: str):
    return json.loads(subprocess.run(["gh", "api", path], check=True, capture_output=True, text=True).stdout)


def main(run_id: str) -> int:
    run = gh(f"repos/{REPO}/actions/runs/{run_id}")
    jobs = gh(f"repos/{REPO}/actions/runs/{run_id}/jobs")["jobs"]
    job = next(j for j in jobs if j["name"].startswith("thunder"))
    if job["conclusion"] != "success":
        raise SystemExit(f"the thunder job concluded {job['conclusion']}")
    anns = gh(f"repos/{REPO}/check-runs/{job['id']}/annotations?per_page=100")
    rows, meta = [], None
    for a in anns:
        if a.get("title") == "thunder-bridge":
            rows.append(json.loads(a["message"]))
        elif a.get("title") == "thunder-bridge-meta":
            meta = json.loads(a["message"])
    if not rows or meta is None:
        raise SystemExit("no thunder-bridge annotations on that run")
    out = {"run": run_id, "commit": run["head_sha"], "meta": meta, "rows": rows}
    Path("reports/thunder_bridge.json").write_text(json.dumps(out, indent=1))
    print(f"imported {len(rows)} results from run {run_id} ({run['head_sha'][:8]})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
