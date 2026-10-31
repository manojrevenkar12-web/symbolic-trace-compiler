"""stc <step> [--profile quick|full]"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    from .evaluate.pipeline import STEPS, run_all

    ap = argparse.ArgumentParser(prog="stc", description=__doc__)
    ap.add_argument("step", choices=[*STEPS, "all"])
    ap.add_argument("--profile", default="full", choices=["quick", "full"])
    a = ap.parse_args(argv)
    if a.step == "all":
        return 0 if run_all(a.profile) else 1
    out = STEPS[a.step](a.profile)
    return 0 if (a.step != "gate" or out) else 1


if __name__ == "__main__":
    sys.exit(main())
