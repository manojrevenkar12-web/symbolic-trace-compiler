# 2. Lookasides at module level, not a bytecode interpreter

Thunder acquires programs by interpreting Python bytecode, with lookasides for torch functions.
This compiler runs the user's code as ordinary Python and installs the lookasides on the
`torch`, `torch.nn.functional` and `torch.Tensor` namespaces for the duration of one trace, in
the tracing thread only (`math.sqrt` too, so a scale computed from a size stays symbolic). It is
far smaller, and good enough for model code that calls torch through those names. Code that
binds a torch function to a local name before tracing (`from torch import cat`) escapes it; a
TorchFunctionMode could not replace it because PyTorch's argument parser calls `__index__` on
symbolic ints before the mode sees the call, which would specialise them.
