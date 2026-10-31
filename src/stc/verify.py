"""Are the guards sound, and how much do they accept?

For a compiled callable with programs in its cache, draw input shapes around the traced ones:
some keep the relations the trace found (equal dimensions stay equal), some break them, sizes
go below, between and above the hints, including 1. For each draw:

- run eager PyTorch (the reference);
- run every cached program's prologue;
- if a prologue accepts, run that program and compare with eager.

A *soundness violation* is a draw where a prologue accepted and the program's result differs from
eager (or one of them raised and the other did not). That must never happen.

*Coverage* is the share of the eager-valid draws some cached program accepts: how often a call
like this would run without compiling. Static programs accept only their exact shapes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import torch

from .jit import Compiled


@dataclass
class Report:
    draws: int = 0
    eager_ok: int = 0
    accepted: int = 0
    accepted_eager_ok: int = 0
    accepted_eager_fail: int = 0  # accepted inputs eager rejects: the program then raises inside the computation
    violations: int = 0
    examples: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return self.accepted_eager_ok / max(1, self.eager_ok)

    def as_dict(self) -> dict[str, Any]:
        return {
            "draws": self.draws,
            "eager_ok": self.eager_ok,
            "accepted": self.accepted,
            "violations": self.violations,
            "accepted_eager_fail": self.accepted_eager_fail,
            "coverage": self.coverage,
            "examples": self.examples[:5],
        }


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if a.is_floating_point():
            return bool(torch.allclose(a, b, rtol=1e-4, atol=1e-5, equal_nan=True))
        return bool(torch.equal(a, b))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def draw_sizes(rng: random.Random, hints: list[int], linked: bool, cap: int) -> list[int]:
    """New sizes for one call's dimensions; `linked` keeps equal hints equal."""
    out: list[int] = []
    memo: dict[int, int] = {}
    for h in hints:
        if linked and h in memo:
            out.append(memo[h])
            continue
        r = rng.random()
        if r < 0.15:
            v = h
        elif r < 0.25:
            v = rng.choice([1, 2])
        elif r < 0.4:
            v = rng.randint(max(2, h), max(3, cap))  # far above the hint, past any limit in the program
        else:
            v = rng.randint(2, max(3, min(cap, 2 * h + 8)))
        memo[h] = v
        out.append(v)
    return out


def make_inputs(examples: tuple, sizes: list[list[int]], rng: random.Random) -> tuple:
    out = []
    for ex, sz in zip(examples, sizes):
        if isinstance(ex, torch.Tensor):
            g = torch.Generator().manual_seed(rng.randrange(1 << 30))
            if ex.dtype in (torch.long, torch.int32, torch.int64):
                hi = int(ex.max().item()) + 1 if ex.numel() else 2
                out.append(torch.randint(0, max(hi, 2), sz, generator=g, dtype=ex.dtype))
            elif ex.dtype == torch.bool:
                out.append(torch.rand(sz, generator=g) > 0.5)
            else:
                out.append(torch.randn(sz, generator=g, dtype=ex.dtype))
        elif isinstance(ex, int) and not isinstance(ex, bool):
            out.append(sz[0])
        else:
            out.append(ex)
    return tuple(out)


def fuzz(compiled: Compiled, examples: tuple, n: int = 200, seed: int = 0, cap: int = 64, input_fn=None) -> Report:
    """`input_fn(rng, sizes)`, if given, builds valid-looking inputs (e.g. a padding mask)."""
    rng = random.Random(seed)
    rep = Report()
    fn = compiled.fn
    flat_state = compiled.state()
    hints = []
    for ex in examples:
        if isinstance(ex, torch.Tensor):
            hints.append(list(ex.shape))
        elif isinstance(ex, int) and not isinstance(ex, bool):
            hints.append([ex])
        else:
            hints.append([])
    all_hints = [h for hs in hints for h in hs]
    for _ in range(n):
        linked = rng.random() < 0.6
        flat = draw_sizes(rng, all_hints, linked, cap)
        sizes, k = [], 0
        for hs in hints:
            sizes.append(flat[k : k + len(hs)])
            k += len(hs)
        args = input_fn(rng, sizes) if input_fn else make_inputs(examples, sizes, rng)
        rep.draws += 1
        try:
            with torch.no_grad():
                ref = fn(*args)
            ok = True
        except Exception:
            ref, ok = None, False
        rep.eager_ok += ok
        full = flat_state + list(args)
        for e in compiled.entries:
            syms = e.program.prologue(*full)
            if syms is None:
                continue
            rep.accepted += 1
            rep.accepted_eager_ok += ok
            rep.accepted_eager_fail += not ok
            try:
                with torch.no_grad():
                    got = e.program.computation(*full, *syms, *e.program.constants)
                good = ok and _same(got, ref)
            except Exception as ex:
                good = not ok
                if ok:
                    rep.examples.append(f"compiled raised {type(ex).__name__} for sizes {sizes}")
            if not good:
                rep.violations += 1
                if ok:
                    rep.examples.append(f"result differs from eager for sizes {sizes}")
            break
    return rep


# ---------------------------------------------------------------------- guard necessity
def guard_blocks(prologue_src: str) -> list[tuple[int, str]]:
    """The size checks of a prologue (equalities between sources and symbolic guards), as
    (line index, condition) pairs. Type, rank, dtype and device checks are not included."""
    lines = prologue_src.splitlines()
    out = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("if not (") or (s.startswith("if ") and "shape[" in s and "!=" in s and "_shape[" not in s):
            out.append((i, s[3:-1]))
    return out


def without_line(prologue_src: str, i: int) -> str:
    lines = prologue_src.splitlines()
    return "\n".join(lines[:i] + lines[i + 2 :])


def necessity(compiled: Compiled, examples: tuple, n: int, seed: int = 1, cap: int = 64, input_fn=None) -> list[dict[str, Any]]:
    """For each size check of the most recent program: remove it and fuzz the same draws.

    `without_it` is "wrong results" when the weakened prologue accepts a draw on which the program
    disagrees with eager; "errors" when it accepts draws eager rejects too (the call then fails
    inside the computation instead of missing the cache); "none seen" when neither happened."""
    from .codegen import GLOBALS

    prog = compiled.entries[0].program
    saved0 = compiled.entries
    compiled.entries = saved0[:1]
    base = fuzz(compiled, examples, n=n, seed=seed, cap=cap, input_fn=input_fn)
    compiled.entries = saved0
    results = []
    for idx, cond in guard_blocks(prog.prologue_src):
        src = without_line(prog.prologue_src, idx)
        g = dict(GLOBALS, Tensor=torch.Tensor)
        exec(compile(src, "<stc-mutant>", "exec"), g)  # noqa: S102 - generated here
        from .codegen import safe_prologue

        mutant = type(compiled.entries[0].program)(**{**prog.__dict__, "prologue": safe_prologue(g["prologue"]), "prologue_src": src})
        saved = compiled.entries
        compiled.entries = [type(saved[0])(mutant, saved[0].shapes)]
        try:
            rep = fuzz(compiled, examples, n=n, seed=seed, cap=cap, input_fn=input_fn)
        finally:
            compiled.entries = saved
        kind = "wrong results" if rep.violations else "errors" if rep.accepted_eager_fail > base.accepted_eager_fail else "none seen"
        results.append(
            {
                "check": cond,
                "without_it": kind,
                "violations": rep.violations,
                "accepted_failing": rep.accepted_eager_fail - base.accepted_eager_fail,
                "draws": rep.draws,
            }
        )
    return results
