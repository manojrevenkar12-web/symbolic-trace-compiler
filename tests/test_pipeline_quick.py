"""The whole pipeline on the quick profile (CI): every step runs and the gate's floors hold."""

import json

import pytest

from stc.workloads import ROOT


@pytest.mark.slow
def test_quick_pipeline_end_to_end():
    from stc.evaluate.pipeline import run_all

    assert run_all("quick")
    r = json.loads((ROOT / "runs" / "quick" / "results.json").read_text())
    assert r["summary"]["fuzz_violations"] == 0 and r["summary"]["max_abs_diff"] <= 1e-5
    assert (ROOT / "runs" / "quick" / "REPORT.md").read_text().startswith("# Report")
