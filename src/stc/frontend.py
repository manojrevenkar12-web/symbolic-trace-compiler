"""From a Python callable and example inputs to a trace.

Inputs are flattened into one positional list: a module's parameters and buffers first (always
static shapes), then the call's positional arguments. A policy decides, per tensor dimension and
per integer input, whether it becomes a symbol or a constant of the program.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from .ops import lookasides
from .symbolic import ShapeEnv, Source, SymInt
from .trace import _ACTIVE, _TRACE_LOCK, TensorProxy, Tracer, Unsupported


@dataclass
class InputSpec:
    kind: str  # "tensor", "int", "const"
    name: str
    dtype: torch.dtype | None = None
    device: torch.device | None = None
    shape: tuple = ()  # ints and SymInts (as traced, before replacements)
    value: Any = None  # constant value, or the hint of an int
    sym: Any = None  # SymInt for a symbolic int input
    is_param: bool = False


@dataclass
class Trace:
    env: ShapeEnv
    tracer: Tracer
    inputs: list[InputSpec]
    output: Any  # nested structure of proxies / constants / SymInts
    n_params: int
    trace_ms: float = 0.0
    notes: list[str] = field(default_factory=list)


def _flatten_module(module: torch.nn.Module) -> list[tuple[str, torch.nn.Module, str, bool, torch.Tensor]]:
    out = []
    seen: set[int] = set()
    for mname, m in module.named_modules():
        for pname, p in list(m._parameters.items()):
            if p is not None and id(p) not in seen:
                seen.add(id(p))
                out.append((f"{mname}.{pname}".strip("."), m, pname, True, p))
        for bname, b in list(m._buffers.items()):
            if b is not None and id(b) not in seen:
                seen.add(id(b))
                out.append((f"{mname}.{bname}".strip("."), m, bname, False, b))
    return out


class StateReader:
    """Reads a module's parameters and buffers in trace order, cheaply, on every call (a
    parameter replaced by a new tensor is picked up)."""

    def __init__(self, module: torch.nn.Module | None):
        self.slots = [(m._parameters if p else m._buffers, a) for _, m, a, p, _ in _flatten_module(module)] if module is not None else []

    def __call__(self) -> list[torch.Tensor]:
        return [d[a] for d, a in self.slots]


def module_state(module: torch.nn.Module | None) -> list[torch.Tensor]:
    if module is None:
        return []
    return [t for _, _, _, _, t in _flatten_module(module)]


def trace(
    fn,
    args: tuple,
    *,
    module: torch.nn.Module | None = None,
    dynamic: dict[tuple[int, int | None], bool] | None = None,
    default_dynamic: bool = True,
    specialize_zero_one: bool = True,
) -> Trace:
    """Trace `fn(*args)` (or `module(*args)`).

    `dynamic[(i, d)]` overrides whether dimension `d` of call argument `i` is a symbol
    (`d=None` for an integer argument); otherwise `default_dynamic` decides.
    """
    dynamic = dynamic or {}
    env = ShapeEnv(specialize_zero_one=specialize_zero_one)
    tr = Tracer(env)
    specs: list[InputSpec] = []
    proxies: list[Any] = []
    state = _flatten_module(module) if module is not None else []
    flat_index = 0
    for _qual, _, _, _is_param, t in state:
        name = f"p{flat_index}"
        shape = tuple(int(s) for s in t.shape)
        p = tr.proxy(shape, t.dtype, t.device, name=name)
        specs.append(InputSpec("tensor", name, t.dtype, t.device, shape, is_param=True))
        proxies.append(p)
        flat_index += 1
    n_params = flat_index
    call_proxies = []
    for i, a in enumerate(args):
        name = f"a{i}"
        if isinstance(a, torch.Tensor):
            shape = tuple(
                env.create(int(s), Source(flat_index, d), dynamic=dynamic.get((i, d), default_dynamic)) for d, s in enumerate(a.shape)
            )
            p = tr.proxy(shape, a.dtype, a.device, name=name)
            specs.append(InputSpec("tensor", name, a.dtype, a.device, shape))
            call_proxies.append(p)
        elif isinstance(a, int) and not isinstance(a, bool):
            v = env.create(a, Source(flat_index, None), dynamic=dynamic.get((i, None), default_dynamic))
            specs.append(InputSpec("int", name, value=a, sym=v if isinstance(v, SymInt) else None))
            call_proxies.append(v)
        elif a is None or isinstance(a, (bool, float, str)):
            specs.append(InputSpec("const", name, value=a))
            call_proxies.append(a)
        else:
            raise Unsupported(f"argument {i} of type {type(a).__name__} (tensors, ints, floats, bools, strings and None only)")
        flat_index += 1
    tr.input_names = [s.name for s in specs]
    t0 = time.perf_counter()
    with _TRACE_LOCK:
        _ACTIVE.append(tr)
        saved = []
        try:
            for (_, m, attr, is_param, _), p in zip(state, proxies):
                d = m._parameters if is_param else m._buffers
                saved.append((d, attr, d[attr]))
                d[attr] = p
            with lookasides():
                out = (module if module is not None else fn)(*call_proxies)
        finally:
            for d, attr, v in saved:
                d[attr] = v
            _ACTIVE.pop()
    _check_output(out)
    return Trace(env, tr, specs, out, n_params, trace_ms=(time.perf_counter() - t0) * 1e3)


def _check_output(out: Any) -> None:
    if isinstance(out, (TensorProxy, SymInt)) or out is None or isinstance(out, (int, float, bool, str)):
        return
    if isinstance(out, (list, tuple)):
        for o in out:
            _check_output(o)
        return
    if isinstance(out, dict):
        for o in out.values():
            _check_output(o)
        return
    if isinstance(out, torch.Tensor):
        raise Unsupported("the function returned a real tensor that was not computed from its inputs")
    raise Unsupported(f"output of type {type(out).__name__}")
