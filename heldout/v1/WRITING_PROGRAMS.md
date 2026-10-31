# Writing held-out programs for a shape-polymorphic PyTorch compiler

A compiler traces a PyTorch program once, with the input sizes treated as symbols, and reuses the
compiled program for later calls whose sizes differ. Before reusing it, it checks conditions on
the sizes ("guards"). If a guard is missing, a later call can silently get a wrong result; if
guards are too strict, the compiler recompiles more than it needs to. Your programs test this.
Write them from this page and `SUPPORTED_OPS.md` only. Do not read anything else in the
repository (no code, tests, configs, reports or other programs).

## What to write

Ten files `heldout/v1/programs/p01.py` ... `p10.py`. Each defines:

```python
import torch
import torch.nn.functional as F

def build():
    """Return (program, calls, about)."""
    torch.manual_seed(0)
    program = ...          # a function of tensors/ints, or an nn.Module in eval mode
    calls = [...]          # 12-30 tuples of positional arguments, in the order they are called
    about = "one sentence: what the program does and what about its sizes is tricky"
    return program, calls, about
```

Use only operations in `SUPPORTED_OPS.md`. Build the call arguments with `torch.randn`,
`torch.randint`, `torch.ones`, Python ints and so on, inside `build()`, seeded. Every call must
run in plain eager PyTorch without error.

## What makes a good program

Make them look like real model or data code (attention variants, pooling, padding and masks,
patching images into tokens, sliding windows, loss functions, reshapes between layouts), and make
their sizes vary across the calls the way real traffic would: batch sizes, sequence lengths,
image sizes, sometimes a dimension of size 1, sometimes two inputs whose sizes must agree.

At least half should contain a trap where the result depends on the sizes in a way a careless
compiler could get wrong without an error, for example:

- Python control flow on a size (`if x.shape[1] > 64: ... else: ...`);
- slices whose end may run past the end of a dimension (`x[:, :n]` with `n` sometimes larger);
- negative indices or `-1` in reshapes; `x.shape[-1] // 2` splits;
- a dimension that is sometimes 1 and broadcasts;
- an int argument that sets a size (a window, a number of heads, a crop length);
- sizes computed from other sizes (`h * w`, `t - k + 1`).

## Rules

No randomness inside the program itself (only in `build()`), no in-place operations, no file,
network or process access. Keep each program under ~60 lines. When done, write
`heldout/v1/programs/NOTES.md` with one line per program saying what it does and where its trap is.
