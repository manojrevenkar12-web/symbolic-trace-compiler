"""Passes over the trace and generation of the cached program: a prologue and a computation.

The prologue receives the flat inputs and returns the values of the live symbols if every guard
holds, else None. It checks, in order of cost: types, ranks, dtypes and devices; static sizes;
binds each symbol from its first source; checks every other source of the symbol (dimensions
unified by a broadcast or a matmul); then the remaining guards.

The computation receives the flat inputs and the symbols and runs the recorded PyTorch calls.
Symbolic sizes used as arguments are printed as Python expressions over the symbols.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field
from typing import Any

import sympy
import torch

from .frontend import Trace
from .symbolic import SymFloat, SymInt, pycode
from .trace import Node, TensorProxy


# ---------------------------------------------------------------------- passes
def output_proxies(out: Any) -> list[TensorProxy]:
    if isinstance(out, TensorProxy):
        return [out]
    if isinstance(out, (list, tuple)):
        return [p for o in out for p in output_proxies(o)]
    if isinstance(out, dict):
        return [p for o in out.values() for p in output_proxies(o)]
    return []


def node_inputs(n: Node) -> list[TensorProxy]:
    found: list[TensorProxy] = []

    def walk(v):
        if isinstance(v, TensorProxy):
            found.append(v)
        elif isinstance(v, (list, tuple)):
            for i in v:
                walk(i)
        elif isinstance(v, dict):
            for i in v.values():
                walk(i)
        elif isinstance(v, slice):
            walk(v.start)
            walk(v.stop)
            walk(v.step)

    walk(n.args)
    walk(n.kwargs)
    return found


def dce(nodes: list[Node], out: Any) -> list[Node]:
    """Drop pure nodes whose outputs nobody uses."""
    live = {id(p) for p in output_proxies(out)}
    keep = []
    for n in reversed(nodes):
        if not n.pure or any(id(o) in live for o in n.outputs):
            keep.append(n)
            live |= {id(p) for p in node_inputs(n)}
    return list(reversed(keep))


@dataclass
class Renamer:
    alias: dict[int, TensorProxy] = field(default_factory=dict)

    def __call__(self, p: TensorProxy) -> TensorProxy:
        while id(p) in self.alias:
            p = self.alias[id(p)]
        return p


def cse(nodes: list[Node], env) -> tuple[list[Node], Renamer]:
    """Merge pure nodes that make the same call on the same values."""
    ren = Renamer()
    seen: dict[str, Node] = {}
    keep = []
    for n in nodes:
        if n.pure:
            key = n.target + "|" + render(n.args, env, ren) + "|" + render(n.kwargs, env, ren)
            if key in seen:
                for a, b in zip(n.outputs, seen[key].outputs):
                    ren.alias[id(a)] = b
                continue
            seen[key] = n
        keep.append(n)
    return keep, ren


# ---------------------------------------------------------------------- rendering values as Python
def render(v: Any, env, ren: Renamer | None = None) -> str:
    if isinstance(v, TensorProxy):
        return (ren(v) if ren else v).name
    if isinstance(v, SymInt):
        return pycode(env.simplify(v.expr))
    if isinstance(v, SymFloat):
        return f"float({pycode(env.simplify(v.expr))})"
    if isinstance(v, bool) or v is None:
        return repr(v)
    if isinstance(v, int):
        return repr(v)
    if isinstance(v, float):
        if math.isinf(v):
            return "float('inf')" if v > 0 else "float('-inf')"
        if math.isnan(v):
            return "float('nan')"
        return repr(v)
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, torch.dtype):
        return str(v)
    if isinstance(v, torch.device):
        return f"torch.device({str(v)!r})"
    if isinstance(v, slice):
        return f"slice({render(v.start, env, ren)}, {render(v.stop, env, ren)}, {render(v.step, env, ren)})"
    if v is Ellipsis:
        return "..."
    if isinstance(v, torch.Size):
        return render(tuple(v), env, ren)
    if isinstance(v, tuple):
        inner = ", ".join(render(i, env, ren) for i in v)
        return f"({inner},)" if len(v) == 1 else f"({inner})"
    if isinstance(v, list):
        return "[" + ", ".join(render(i, env, ren) for i in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k!r}: {render(i, env, ren)}" for k, i in v.items()) + "}"
    raise TypeError(f"cannot render {type(v).__name__}")


def render_call(n: Node, env, ren: Renamer) -> str:
    if n.target == "operator.getitem":
        return f"{render(n.args[0], env, ren)}[{render(n.args[1], env, ren)}]"
    args = [render(a, env, ren) for a in n.args]
    args += [f"{k}={render(v, env, ren)}" for k, v in n.kwargs.items()]
    return f"{n.target}({', '.join(args)})"


# ---------------------------------------------------------------------- the program
@dataclass
class Program:
    prologue_src: str
    computation_src: str
    symbols: list[str]
    guards: list[str]
    n_nodes_traced: int
    n_nodes: int
    raw_guards: int
    specializations: list[str]
    hints: dict[str, int]
    prologue: Any = None
    computation: Any = None
    constants: tuple = ()

    @property
    def source(self) -> str:
        return self.prologue_src + "\n\n" + self.computation_src


GLOBALS = {"torch": torch, "operator": operator, "math": math, "float": float, "min": min, "max": max}
GLOBALS.update(
    {
        "D_" + str(d).split(".")[-1]: d
        for d in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float64,
            torch.int64,
            torch.int32,
            torch.int16,
            torch.int8,
            torch.uint8,
            torch.bool,
            torch.complex64,
            torch.complex128,
        ]
    }
)


def generate(t: Trace) -> Program:
    env = t.env
    names = [s.name for s in t.inputs]
    nodes = dce(t.tracer.nodes, t.output)
    nodes, ren = cse(nodes, env)
    live = env.live_symbols()
    sym_names = [s.name for s in live]

    # ---------------------------------------------------------------- prologue
    # one short-circuiting condition for types, ranks, dtypes, devices and static sizes; then the
    # symbol bindings; then one condition for equalities between sources and the guards
    P = [f"def prologue({', '.join(names)}):"]
    meta: list[str] = []
    for spec in t.inputs:
        n = spec.name
        if spec.kind == "tensor":
            dt = "D_" + str(spec.dtype).split(".")[-1]
            meta.append(f"not isinstance({n}, Tensor)")
            meta.append(f"{n}.dtype is not {dt}")
            if spec.device is not None and spec.device.type != "cpu":
                meta.append(f"{n}.device != torch.device({str(spec.device)!r})")
            else:
                meta.append(f"{n}.device.type != 'cpu'")
            if all(isinstance(x, int) for x in spec.shape):
                meta.append(f"{n}.shape != {tuple(spec.shape)!r}")
            else:
                meta.append(f"{n}.dim() != {len(spec.shape)}")
                meta += [f"{n}.shape[{d}] != {x}" for d, x in enumerate(spec.shape) if isinstance(x, int)]
        elif spec.kind == "int":
            meta.append(f"type({n}) is not int")
            if spec.sym is None:
                meta.append(f"{n} != {spec.value!r}")
        else:
            v = spec.value
            if v is None or isinstance(v, bool):
                meta.append(f"{n} is not {v!r}")
            else:
                meta.append(f"type({n}) is not {type(v).__name__} or {n} != {render(v, env)}")
    if meta:
        P.append("    if (" + "\n            or ".join(meta) + "):")
        P.append("        return None")
    checks: list[str] = []
    for s_ in env.symbols:
        srcs = env.sources.get(s_, [])
        if s_ in env.replacements:
            expr = pycode(env.simplify(s_))
            checks += [f"{src.code(names)} != {expr}" for src in srcs]
        else:
            P.append(f"    {s_.name} = {srcs[0].code(names)}")
            checks += [f"{src.code(names)} != {s_.name}" for src in srcs[1:]]
    guards = env.final_guards()
    guard_src = [pycode(g) for g in guards]
    for c in checks:
        P.append(f"    if {c}:")
        P.append("        return None")
    for g in guard_src:
        P.append(f"    if not ({g}):")
        P.append("        return None")
    P.append(f"    return ({', '.join(sym_names)}{',' if len(sym_names) == 1 else ''})")

    # ---------------------------------------------------------------- computation
    const_names = [p.name for p, _ in t.tracer.captured]
    C = [f"def computation({', '.join(names + sym_names + const_names)}):"]
    for n in nodes:
        outs = [o.name for o in n.outputs]
        lhs = outs[0] if len(outs) == 1 and not _multi(n) else "(" + ", ".join(outs) + ("," if len(outs) == 1 else "") + ")"
        ann = "; ".join(_annot(o, env) for o in n.outputs)
        C.append(f"    {lhs} = {render_call(n, env, ren)}  # {ann}")
    C.append(f"    return {render(t.output, env, ren)}")
    prog = Program(
        prologue_src="\n".join(P),
        computation_src="\n".join(C),
        symbols=sym_names,
        guards=guard_src,
        n_nodes_traced=len(t.tracer.nodes),
        n_nodes=len(nodes),
        raw_guards=env.raw_guards,
        specializations=list(env.specializations),
        hints={s.name: env.hints[s] for s in live},
    )
    g = dict(GLOBALS, Tensor=torch.Tensor)
    exec(compile(prog.prologue_src, "<stc-prologue>", "exec"), g)  # noqa: S102 - generated by this module
    exec(compile(prog.computation_src, "<stc-computation>", "exec"), g)  # noqa: S102
    prog.prologue, prog.computation = safe_prologue(g["prologue"]), g["computation"]
    prog.constants = tuple(tv for _, tv in t.tracer.captured)
    return prog


def safe_prologue(fn):
    """A guard that cannot be evaluated for these inputs (a size used as a divisor is 0) rejects."""

    def prologue(*args):
        try:
            return fn(*args)
        except ArithmeticError:
            return None

    prologue.__wrapped__ = fn
    return prologue


def _annot(p: TensorProxy, env) -> str:
    dims = ", ".join(pycode(env.simplify(d.expr)) if isinstance(d, SymInt) else str(d) for d in p.shape)
    return f"{str(p.dtype).split('.')[-1]}[{dims}]"


def _multi(n: Node) -> bool:
    return n.target in ("torch.Tensor.split", "torch.split", "torch.Tensor.chunk")


def symbol_values_from(prog: Program, flat: list) -> tuple | None:
    return prog.prologue(*flat)


__all__ = ["generate", "Program", "dce", "cse", "render", "sympy"]
