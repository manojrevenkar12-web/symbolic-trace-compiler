# Working on symbolic-trace-compiler

## Fast feedback

- `pytest -q -m "not slow"`: symbols and guards, tracing and shape rules against eager, codegen,
  the workload models, the fuzzer (including that it catches a removed guard), isolation of the
  lookasides (about 30 s).
- `pytest -q -m slow`: the whole pipeline on the quick profile.
- `ruff check src tests scripts && ruff format --check src tests scripts`.
- `cd web && npm test && npm run lint && npm run build && npm run e2e` for the dashboard.
- `stc all --profile full` rebuilds the committed results and ends with the release gate;
  `stc <step> --profile full` runs one step (study, heldout, report, gate).

## Invariants (each has a test or a gate floor)

1. **A compiled program equals eager PyTorch.** Every workload call is compared with eager on the
   same weights; training compares losses per step and weights at the end. A shape rule must
   agree with PyTorch's meta kernels at trace time or tracing stops (`ops.emit`).
2. **A guard is trusted only after it is fuzzed.** Changes to `symbolic.py`, `ops.py`,
   `codegen.py` or `jit.py` rerun the study; fuzz violations must stay at 0. The fuzzer must keep
   catching a removed guard (`tests/test_models.py::test_fuzzer_catches_a_missing_guard`).
3. **Unsupported is loud.** A call the tracer does not handle raises `Unsupported` naming it (or
   falls back to eager when asked, counted in the stats). Nothing is run eagerly behind the
   program's back.
4. **Lookasides exist only during a trace** and only in the tracing thread
   (`tests/test_isolation.py`).
5. **Settings live in `config/`** (workloads, profiles, gate), not in code.
6. **Reports are generated** by `stc report`: `reports/results.json`, `reports/heldout/*`,
   `docs/REPORT.md`, `docs/SUPPORTED_OPS.md`. `reports/thunder_bridge.json` is imported from a CI
   run by `scripts/import_bridge.py` and names that run.

## The held-out rule

Program sets under `heldout/<set>/programs` are written from `WRITING_PROGRAMS.md` and
`SUPPORTED_OPS.md` by someone who has not seen the compiler. A held-out set's first full run is
`reports/heldout/<set>_first_run.json` and is never rewritten; its programs may not change
afterwards (`stc heldout` refuses). Never change the compiler to do better on a held-out set. If
you study a set's failures, write `regression` and why to its `STATUS`, keep it as a regression
set (rerun every time), and have someone who has not seen the code write a new set. `v1` went
through exactly this.
