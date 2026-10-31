# 1. Sizes are symbols with hints; decisions become guards

Tracing runs the user's Python once, with tensor proxies whose sizes are symbols. Each symbol
carries the size seen at trace time (its hint). Any decision that depends on a size (a broadcast,
a slice end, a Python branch) is taken on the hints and recorded as a guard on the symbols.
Equalities are applied as replacements so one symbol names all equal dimensions. The guards are
emitted as a Python prologue; a cached program is reused only when its prologue accepts.

Sizes 0 and 1 are specialised (broadcasting depends on them), as in PyTorch's symbolic shapes;
this is why the symbolic policy compiles a second program for batch size 1.
