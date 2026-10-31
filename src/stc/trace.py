"""The trace: a straight-line SSA program of PyTorch calls over tensor proxies.

Tracing runs the user's Python with `TensorProxy` objects in place of tensors. Calls to supported
PyTorch functions and tensor methods are recorded as nodes; Python control flow runs normally,
and any decision it takes on a size becomes a guard (see `symbolic.py`). Calls the tracer does not
support raise `Unsupported` naming the call, rather than being silently run eagerly.

Torch functions are intercepted by *lookasides*: while a trace runs, the supported entries of
`torch`, `torch.nn.functional` and `torch.Tensor` are temporarily replaced by recording shims.
Thunder uses the same idea inside its bytecode interpreter; here it is done at the module level,
so the user's code runs as ordinary Python.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field
from typing import Any

import sympy
import torch

from .symbolic import ShapeEnv, SymFloat, SymInt, unwrap


class Unsupported(RuntimeError):
    """The traced program used something the compiler does not handle."""


_TRACE_LOCK = threading.RLock()
_ACTIVE: list[Tracer] = []


def active() -> Tracer:
    if not _ACTIVE:
        raise RuntimeError("no active trace")
    return _ACTIVE[-1]


@dataclass(eq=False)
class TensorProxy:
    name: str
    shape: tuple[Any, ...]
    dtype: torch.dtype
    device: torch.device
    tracer: Tracer = field(repr=False)

    # ------------------------------------------------------------------ metadata
    @property
    def ndim(self) -> int:
        return len(self.shape)

    def dim(self) -> int:
        return len(self.shape)

    def size(self, d: int | None = None):
        if d is None:
            return torch.Size([x for x in self.shape]) if all(isinstance(x, int) for x in self.shape) else tuple(self.shape)
        return self.shape[d]

    def numel(self):
        out: Any = 1
        for s in self.shape:
            out = out * s
        return out

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    @property
    def is_floating_point_dtype(self) -> bool:
        return self.dtype.is_floating_point

    def is_floating_point(self) -> bool:
        return self.dtype.is_floating_point

    def __len__(self) -> int:
        return int(self.shape[0])

    def __bool__(self) -> bool:
        raise Unsupported(f"the truth value of tensor {self.name} depends on its data")

    def __iter__(self):
        raise Unsupported(f"iterating over tensor {self.name} depends on its size and data")

    def item(self):
        raise Unsupported(f"{self.name}.item() depends on tensor data")

    def tolist(self):
        raise Unsupported(f"{self.name}.tolist() depends on tensor data")

    __hash__ = object.__hash__

    # every torch call that reaches a proxy without a lookaside lands here
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # a real tensor's method or operator called with a proxy (`bias * x`, `w.matmul(x)`)
        from .ops import capture_args, tensor_method_rule

        rule = tensor_method_rule(getattr(func, "__name__", ""))
        if rule is not None and getattr(torch.Tensor, func.__name__, None) is not None:
            return rule(*capture_args(tuple(args)), **capture_args(kwargs or {}))
        name = getattr(func, "__qualname__", None) or getattr(func, "__name__", repr(func))
        raise Unsupported(f"torch call `{name}` is not supported by the tracer")

    def __getattr__(self, name: str):
        from .ops import METHODS

        if name in METHODS:
            from .ops import capture_args

            rule = METHODS[name]
            return lambda *a, **k: rule(self, *capture_args(a), **capture_args(k))
        raise Unsupported(f"tensor method or attribute `{name}` is not supported by the tracer")

    def __repr__(self) -> str:
        return f"TensorProxy({self.name}, shape={self.shape}, dtype={self.dtype})"


def _binop(name, reverse=False):
    def f(self, other):
        from .ops import METHODS, capture_args

        other = capture_args(other)
        if reverse and not isinstance(other, TensorProxy):
            return METHODS["__r" + name + "__"](self, other)  # e.g. 2.0 / x -> torch.Tensor.__rtruediv__(x, 2.0)
        return METHODS[name](other, self) if reverse else METHODS[name](self, other)

    return f


for _dunder, _m, _rev in [
    ("__add__", "add", False),
    ("__radd__", "add", True),
    ("__sub__", "sub", False),
    ("__rsub__", "sub", True),
    ("__mul__", "mul", False),
    ("__rmul__", "mul", True),
    ("__truediv__", "div", False),
    ("__rtruediv__", "truediv", True),
    ("__matmul__", "matmul", False),
    ("__rmatmul__", "matmul", True),
    ("__pow__", "pow", False),
    ("__rpow__", "pow", True),
    ("__lt__", "lt", False),
    ("__le__", "le", False),
    ("__gt__", "gt", False),
    ("__ge__", "ge", False),
    ("__eq__", "eq", False),
    ("__ne__", "ne", False),
    ("__and__", "logical_and", False),
    ("__or__", "logical_or", False),
]:
    setattr(TensorProxy, _dunder, _binop(_m, _rev))
TensorProxy.__neg__ = lambda self: __import__("stc.ops", fromlist=["METHODS"]).METHODS["neg"](self)
TensorProxy.__invert__ = lambda self: __import__("stc.ops", fromlist=["METHODS"]).METHODS["logical_not"](self)
TensorProxy.__getitem__ = lambda self, idx: __import__("stc.ops", fromlist=["METHODS"]).METHODS["__getitem__"](
    self, __import__("stc.ops", fromlist=["capture_args"]).capture_args(idx)
)
TensorProxy.T = property(lambda self: __import__("stc.ops", fromlist=["METHODS"]).METHODS["t_property"](self))
TensorProxy.mT = property(lambda self: __import__("stc.ops", fromlist=["METHODS"]).METHODS["transpose"](self, -2, -1))


@dataclass
class Node:
    target: str  # how the call is printed, e.g. "torch.matmul", "operator.getitem"
    args: tuple
    kwargs: dict
    outputs: list[TensorProxy]
    pure: bool = True  # no randomness or side effects: may be CSE'd


@dataclass
class Tracer:
    env: ShapeEnv
    nodes: list[Node] = field(default_factory=list)
    counter: itertools.count = field(default_factory=lambda: itertools.count())
    inputs: list[Any] = field(default_factory=list)  # flat inputs: TensorProxy, SymInt or constants
    input_names: list[str] = field(default_factory=list)
    captured: list[tuple[TensorProxy, torch.Tensor]] = field(default_factory=list)  # real tensors the code closed over
    _captured_ids: dict[int, TensorProxy] = field(default_factory=dict)

    def capture(self, t: torch.Tensor) -> TensorProxy:
        """A real tensor reached a traced call (a constant the code closed over): the program
        keeps a reference to it and receives it as an extra argument."""
        p = self._captured_ids.get(id(t))
        if p is None:
            p = TensorProxy(f"c{len(self.captured)}", tuple(int(x) for x in t.shape), t.dtype, t.device, self)
            self.captured.append((p, t))
            self._captured_ids[id(t)] = p
        return p

    def new_name(self) -> str:
        return f"t{next(self.counter)}"

    def proxy(self, shape, dtype, device, name: str | None = None) -> TensorProxy:
        return TensorProxy(name or self.new_name(), tuple(shape), dtype, device, self)

    def record(self, target: str, args: tuple, kwargs: dict, outputs: list[TensorProxy], pure: bool = True) -> None:
        self.nodes.append(Node(target, args, kwargs, outputs, pure))


# ---------------------------------------------------------------------- meta validation
_EMPTY = torch.empty  # captured before any lookaside replaces torch.empty


def to_meta(x: Any) -> Any:
    """The same call on meta tensors with the hint sizes: checks a rule and gives the dtype."""
    if isinstance(x, TensorProxy):
        return _EMPTY([int(unwrap_hint(s)) for s in x.shape], dtype=x.dtype, device="meta")
    if isinstance(x, (SymInt, SymFloat)):
        return x.hint
    if isinstance(x, torch.Tensor):
        return _EMPTY(tuple(x.shape), dtype=x.dtype, device="meta")
    if isinstance(x, slice):
        return slice(to_meta(x.start), to_meta(x.stop), to_meta(x.step))
    if isinstance(x, (list, tuple)):
        return type(x)(to_meta(i) for i in x) if not isinstance(x, torch.Size) else x
    if isinstance(x, dict):
        return {k: to_meta(v) for k, v in x.items()}
    if isinstance(x, torch.device):
        return torch.device("meta")
    return x


def unwrap_hint(s: Any) -> int:
    return s.hint if isinstance(s, SymInt) else int(s)


def shape_hint(shape) -> list[int]:
    return [unwrap_hint(s) for s in shape]


def check_shape(rule: str, symbolic: tuple, meta_shape: tuple) -> None:
    if shape_hint(symbolic) != list(meta_shape):
        raise RuntimeError(f"shape rule for {rule} disagrees with PyTorch: {shape_hint(symbolic)} vs {list(meta_shape)}")


def sym_size(x: Any) -> Any:
    return sympy.sympify(unwrap(x))
