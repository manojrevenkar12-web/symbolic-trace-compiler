# symbolic-trace-compiler

A source-to-source trace compiler for PyTorch in which tensor sizes are **symbols**. It traces a
program once, writes the conditions under which the traced program stays valid as a Python
*prologue*, and reuses the program for every later call whose sizes satisfy them. Each guard is
checked by fuzzing against eager PyTorch.

It targets one of the hard open problems of trace-based compilers such as
[Thunder](https://github.com/Lightning-AI/lightning-thunder). Thunder's default cache treats every
tensor dimension as a constant, and its experimental `"symbolic values"` cache still checks exact
tensor shapes (`TODO: establish guarding logic to allow non-broadcast shape change` in
`thunder/core/jit_ext.py`; issue "Thunder needs to support dynamic shape"). A model fed changing
sequence lengths therefore recompiles once per length.

```text
def prologue(p0, …, p28, a0):                 # the module's 29 parameters, then the tokens
    if (… types, dtypes, devices and parameter shapes …):
        return None
    s0 = a0.shape[1]
    if not (s0 >= 2):                          # sizes 0 and 1 get their own program
        return None
    if not (s0 <= 256):                        # from `assert t <= block_size` in the model
        return None
    return (s0,)

def computation(p0, …, p28, a0, s0):
    t0 = torch.arange(0, s0, dtype=torch.int64, device=torch.device('cpu'))  # int64[s0]
    t1 = torch.nn.functional.embedding(a0, p0, padding_idx=None)  # float32[1, s0, 64]
    …
    t9 = torch.Tensor.view(t7, 1, s0, 4, 16)  # float32[1, s0, 4, 16]
```

This is the program that served all 120 steps of greedy decoding: one compile instead of 120.

## Results (committed full run, [`docs/REPORT.md`](docs/REPORT.md))

**Recompilations.** Five workloads with realistic size variation: chat prompts on GPT- and
Llama-style models, decoding, padded encoder batches, and training with variable lengths. Each
was replayed under three policies, and every call was compared with eager PyTorch on the same
weights. Training compared the loss at every step and the weights at the end.

| Workload | Calls | Distinct shapes | Static | Auto | Symbolic |
|---|---:|---:|---:|---:|---:|
| Chat prefill, GPT | 200 | 183 | 183 | 4 | **2** |
| Chat prefill, Llama (RoPE table sliced to length, grouped-query attention) | 200 | 179 | 179 | 3 | **2** |
| Greedy decoding, no KV cache | 120 | 120 | 120 | 2 | **1** |
| Padded encoder batches with a padding mask | 150 | 148 | 148 | 2 | **1** |
| Training, variable length (forward, loss, backward, AdamW) | 60 | 48 | 48 | 2 | **1** |

- In total: **678 programs with sizes as constants, 7 with sizes as symbols**.
- Every call agreed with eager (largest difference 0.0).
- The second symbolic program in chat prefill is for batch size 1, which broadcasting treats
  differently.
- *Static* is how Thunder's default cache treats sizes. *Auto* is PyTorch's
  automatic-dynamic idea: dimensions seen changing become symbols.

**Guards found from the code.** The guards come from what the model does with its sizes, not from
hand-written annotations:
- Llama's prologue checks `s1 <= 256`, because the RoPE table is sliced to the sequence length
  and broadcasting would fail beyond its 256 rows.
- The training step checks `(s0 * s1) == (s2 * s3)`. The targets need only the same number of
  elements as the logits, because both are flattened for the loss.

**Soundness.** After each workload, its cached programs were fuzzed with 300 new size draws:
- some draws keep the relations the trace found and some break them;
- sizes range from 1 to well past any limit in the model.

Result: **0 wrong results in 3,600 draws**. Each guard of the symbolic programs was then removed in
turn and the same draws fuzzed again. The report says whether that guard protects against wrong
results, against errors inside the program, or showed no effect in these draws. A test checks
that the fuzzer catches a removed branch guard.

**Blind programs.** Ten programs per set, written by someone who never saw the compiler, from a
one-page brief and the list of supported operations. They contain size traps: branches on sizes,
slices past the end, sizes of 1, int arguments that set sizes, sizes computed from other sizes.

- **Set v2, first run (blind):**
  - 9 of 10 programs compiled;
  - no wrong call under any policy, and no wrong result in 1,800 fuzz draws;
  - over their 127 calls, 111 programs with constant sizes, 73 under auto and 52 with symbolic
    sizes;
  - removing guards produced wrong results for three programs (a branch on `h * w > 1024`, a
    slice bound, a channel count), which shows those guards are needed.
  - The tenth crashed the tracer: `t % window` needed `SymInt.__rmod__`. That is fixed, and the
    set is now a regression set.
- **Set v1** was seen failing before its first run was recorded, so it is a development and
  regression set. The details are in each set's `STATUS`.

**Thunder on the same calls.** A CI job installs torch 2.8 and lightning-thunder 0.2.6 (versions
that agree) and replays the inference workloads through:
- `thunder.jit` with its default cache;
- `thunder.jit` with `cache="symbolic values"`;
- this compiler.

All three run in one process, on the same calls. The numbers are imported from that CI run into
[`reports/thunder_bridge.json`](reports/thunder_bridge.json), which names the run.

## How it works

1. **Trace.** The user's Python runs with tensor proxies whose sizes are symbols carrying the
   sizes seen at trace time:
   - Calls to `torch`, `torch.nn.functional` and `torch.Tensor` are recorded through lookasides,
     installed for one trace in the tracing thread.
   - Methods on tensors the code closed over are recorded too, with the tensor captured, so
     `pos[:, : t + 1]` and `cls.expand(b, -1, -1)` stay symbolic.
   - Scales such as `head_dim ** -0.5` or `1 / math.sqrt(d)` become symbolic floats printed into
     the program.
   - Unsupported calls raise `Unsupported` naming the call, or fall back to eager when asked.
2. **Shape rules.** Each operation has a symbolic shape rule that records the guards its decisions
   need: broadcasting, slicing past the end, `reshape(-1, …)` divisibility, matmul and attention
   size agreement. The same call is then run on meta tensors at the traced sizes, and tracing stops
   if PyTorch disagrees with the rule.
3. **Guards.** Equalities become replacements, so one symbol names all equal dimensions. The other
   guards are simplified with sympy, de-duplicated and printed as Python.
4. **Passes and code generation.** Dead-code removal and common-subexpression merging run on the
   trace. A prologue and a computation are generated as readable Python, with each value's
   symbolic shape as a comment.
5. **Cache.** Programs are kept most-recently-used first. Policies are `static`, `auto` and
   `symbolic`. The stats count compiles, prologue time and fallbacks.

```python
import stc
fast = stc.jit(model)           # policy="symbolic" by default
y = fast(tokens)                # traces on the first call; later lengths reuse the program
print(fast.last_program().source)
```

## Limits

- **CPU only, no fusion.** The computation calls the same PyTorch operations as eager, so this
  measures the cache (compiles, checks, correctness), not kernels. A real deployment would put an
  executor such as nvFuser or Inductor behind the trace.
- **Not inside Thunder.** This is a separate compiler with Thunder's overall design, not a patch
  to Thunder's interpreter. Thunder's source does not run here: this machine's torch is newer
  than Thunder 0.2.6 supports. The comparison runs in CI.
- **Fixed-size Python loops are unrolled,** so a loop over `range(window)` specialises the window
  (set v1, p03). Data-dependent control flow is unsupported.
- **Lookasides work at the module level,** so code that binds `from torch import cat` before
  tracing escapes them (`docs/adr/0002`).
- **Prologue cost grows with parameters.** About 40–60 µs per call for 29 parameter tensors is
  spent checking their dtype, device and shape. Under the static policy it also grows with the
  number of cached programs, which are tried in turn: chat prefill tried 90 prologues per call on
  average and spent 3.0 ms per call checking them, against 58 µs with symbolic sizes.
- **The workload models are small** (2 layers, 64 wide). Shape logic does not depend on size;
  absolute times do.
- **No functional transforms** (grad, vmap) on traces: training relies on PyTorch's autograd
  through the generated calls.

## Run it

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest -m "not slow"
stc all --profile quick          # the pipeline in a few minutes
stc all --profile full           # the committed run (about 30 minutes on two cores)
cd web && npm ci && npm run dev  # the dashboard over reports/results.json
```

The repository is organised as follows:
- `src/stc/`: the compiler. `symbolic.py`, `trace.py` and `ops.py` cover symbols, proxies,
  lookasides and shape rules. `frontend.py`, `codegen.py` and `jit.py` cover tracing, passes,
  code generation and the cache. `verify.py` holds the fuzzer and the guard mutation tests.
- `src/stc/models/` and `src/stc/workloads.py`: the workload models and their size streams.
- `src/stc/evaluate/`: the study, the held-out runs, the report and the gate.
- `heldout/`: blind program sets and their writing guide.
- `scripts/thunder_bridge.py`: the Thunder comparison, run in CI.
- `web/`: a React and Motion dashboard with loading, error and empty states, tested with Vitest,
  Playwright and axe.

Built by Manoj Kumar Prakash with Claude Code (Anthropic). Claude wrote most of the code under my
direction. The blind program sets were written by separate Claude subagents that saw only the
writing guide and the list of supported operations.
