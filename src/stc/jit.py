"""`stc.jit`: a callable that compiles on a cache miss and reuses programs whose prologue accepts.

Policies decide which sizes become symbols:

- `static`: every size is a constant, so any new shape compiles a new program. This is how
  Thunder's default cache (`constant values`) treats tensor sizes.
- `auto`: start static; when a call misses, the dimensions that differ from the most recent
  program's sizes become symbols from then on (PyTorch's automatic dynamic shapes).
- `symbolic`: every tensor dimension of a call argument, and every int argument, is a symbol
  from the first call, except sizes 0 and 1 (broadcasting depends on them).

Parameters and buffers of a module are always static.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from .codegen import Program, generate
from .frontend import StateReader, trace
from .trace import Unsupported

POLICIES = ("static", "auto", "symbolic")


@dataclass
class Stats:
    calls: int = 0
    hits: int = 0
    compiles: int = 0
    fallbacks: int = 0
    compile_ms: float = 0.0
    prologue_ns: int = 0  # time in prologues, accepted or not
    run_ns: int = 0
    prologues_run: int = 0
    unsupported: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["unsupported"] = list(self.unsupported)
        d["prologue_us_per_call"] = self.prologue_ns / 1e3 / max(1, self.calls)
        return d


@dataclass
class Entry:
    program: Program
    shapes: list[tuple]  # the call's tensor shapes it was traced with (for `auto`)
    hits: int = 0


class Compiled:
    def __init__(
        self, fn, *, policy: str = "symbolic", fallback: bool = False, specialize_zero_one: bool = True, static_args: set[int] | None = None
    ):
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        self.fn = fn
        self.module = fn if isinstance(fn, torch.nn.Module) else None
        self.policy = policy
        self.fallback = fallback
        self.specialize_zero_one = specialize_zero_one
        self.static_args = static_args or set()
        self.entries: list[Entry] = []
        self.stats = Stats()
        self.dynamic_dims: set[tuple[int, int | None]] = set()  # `auto`: dims seen changing
        self.state = StateReader(self.module)

    # ------------------------------------------------------------------ calling
    def __call__(self, *args):
        st = self.stats
        st.calls += 1
        flat = self.state() + list(args)
        t0 = time.perf_counter_ns()
        for i, e in enumerate(self.entries):
            st.prologues_run += 1
            syms = e.program.prologue(*flat)
            if syms is not None:
                t1 = time.perf_counter_ns()
                st.prologue_ns += t1 - t0
                st.hits += 1
                e.hits += 1
                if i:  # most recently used first
                    self.entries.insert(0, self.entries.pop(i))
                out = e.program.computation(*flat, *syms, *e.program.constants)
                st.run_ns += time.perf_counter_ns() - t1
                return out
        st.prologue_ns += time.perf_counter_ns() - t0
        try:
            entry = self._compile(args)
        except Unsupported as ex:
            if not self.fallback:
                raise
            st.fallbacks += 1
            if str(ex) not in st.unsupported:
                st.unsupported.append(str(ex))
            return self.fn(*args)
        self.entries.insert(0, entry)
        syms = entry.program.prologue(*flat)
        if syms is None:
            raise AssertionError("a fresh program's prologue rejected the inputs it was traced with:\n" + entry.program.prologue_src)
        t1 = time.perf_counter_ns()
        out = entry.program.computation(*flat, *syms, *entry.program.constants)
        st.run_ns += time.perf_counter_ns() - t1
        return out

    # ------------------------------------------------------------------ compiling
    def _dynamic_map(self, args) -> tuple[dict, bool]:
        dyn: dict[tuple[int, int | None], bool] = {}
        for i, a in enumerate(args):
            dims = range(a.ndim) if isinstance(a, torch.Tensor) else [None] if isinstance(a, int) and not isinstance(a, bool) else []
            for d in dims:
                if i in self.static_args:
                    dyn[(i, d)] = False
                elif self.policy == "static":
                    dyn[(i, d)] = False
                elif self.policy == "symbolic":
                    dyn[(i, d)] = True
                else:
                    dyn[(i, d)] = (i, d) in self.dynamic_dims
        return dyn, False

    def _compile(self, args) -> Entry:
        shapes = [tuple(a.shape) if isinstance(a, torch.Tensor) else a for a in args]
        if self.policy == "auto" and self.entries:
            last = self.entries[0].shapes
            for i, (now, then) in enumerate(zip(shapes, last)):
                if isinstance(now, tuple) and isinstance(then, tuple) and len(now) == len(then):
                    for d, (x, y) in enumerate(zip(now, then)):
                        if x != y:
                            self.dynamic_dims.add((i, d))
                elif isinstance(now, int) and isinstance(then, int) and now != then:
                    self.dynamic_dims.add((i, None))
        dyn, _ = self._dynamic_map(args)
        t0 = time.perf_counter()
        tr = trace(
            self.fn,
            tuple(args),
            module=self.module,
            dynamic=dyn,
            default_dynamic=self.policy == "symbolic",
            specialize_zero_one=self.specialize_zero_one,
        )
        prog = generate(tr)
        self.stats.compiles += 1
        self.stats.compile_ms += (time.perf_counter() - t0) * 1e3
        return Entry(prog, shapes)

    # ------------------------------------------------------------------ inspection
    def last_program(self) -> Program | None:
        return self.entries[0].program if self.entries else None

    def programs(self) -> list[Program]:
        return [e.program for e in self.entries]


def jit(fn, **kw) -> Compiled:
    """Compile `fn` (a function of tensors and Python scalars, or an `nn.Module`)."""
    return Compiled(fn, **kw)
