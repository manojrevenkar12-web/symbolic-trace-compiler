"""Shape rules for the supported PyTorch operations, and the lookasides that record them.

Every rule computes the output shape symbolically, records the guards its decisions need, and
records one node. The same call is then run on meta tensors with the hint sizes: if PyTorch's
answer differs from the rule's, tracing stops with an error naming the rule. The meta call also
gives the output dtype, so dtype promotion is PyTorch's, not a copy of it.
"""

from __future__ import annotations

import builtins
import math
import operator
import threading
from collections.abc import Callable
from typing import Any

import sympy
import torch
import torch.nn.functional as F

from .symbolic import SymFloat, SymInt, prod, sym_eq_static, sym_sqrt, unwrap
from .trace import _ACTIVE, TensorProxy, Unsupported, active, check_shape, shape_hint, to_meta

_LOCAL = threading.local()

# originals, captured before any lookaside is installed
ORIG: dict[str, Callable] = {}
METHODS: dict[str, Callable] = {}
LOOKASIDES: dict[tuple[Any, str], Callable] = {}


def resolve(target: str) -> Callable:
    if target in ORIG:
        return ORIG[target]
    if target.startswith("torch.Tensor."):
        return getattr(torch.Tensor, target.split(".", 2)[2])
    if target == "operator.getitem":
        return operator.getitem
    raise KeyError(target)


# ---------------------------------------------------------------------- recording
def _device_of(args: tuple, kwargs: dict) -> torch.device:
    for a in list(args) + list(kwargs.values()):
        if isinstance(a, TensorProxy):
            return a.device
        if isinstance(a, (list, tuple)):
            for b in a:
                if isinstance(b, TensorProxy):
                    return b.device
    d = kwargs.get("device")
    return torch.device(d) if d is not None else torch.device("cpu")


def emit(target: str, args: tuple, kwargs: dict, shapes: list[tuple], *, pure: bool = True, meta_kwargs: dict | None = None):
    """Record a call whose outputs have the given symbolic shapes (one tensor or several)."""
    tr = active()
    fn = resolve(target)
    mk = to_meta(kwargs)
    if "device" in kwargs:
        mk["device"] = "meta"
    if meta_kwargs:
        mk.update(meta_kwargs)
    was = getattr(_LOCAL, "tracing", False)
    _LOCAL.tracing = False  # torch's own Python (meta kernels) must see the real functions
    try:
        out = fn(*to_meta(args), **mk)
    except Exception as e:  # the eager call would fail too
        raise RuntimeError(f"{target} fails for the traced sizes: {e}") from e
    finally:
        _LOCAL.tracing = was
    metas = list(out) if isinstance(out, (tuple, list)) else [out]
    if len(metas) != len(shapes):
        raise RuntimeError(f"{target}: rule gives {len(shapes)} outputs, PyTorch {len(metas)}")
    dev = _device_of(args, kwargs)
    outs = []
    for m, s in zip(metas, shapes):
        check_shape(target, tuple(s), tuple(m.shape))
        outs.append(tr.proxy(s, m.dtype, dev))
    tr.record(target, args, kwargs, outs, pure)
    return outs if isinstance(out, (tuple, list)) else outs[0]


def _env():
    return active().env


def _is_tensor(x: Any) -> bool:
    return isinstance(x, TensorProxy)


def _shape(x: Any) -> tuple:
    if isinstance(x, TensorProxy):
        return x.shape
    if isinstance(x, torch.Tensor):
        raise Unsupported("a real tensor reached a traced operation; pass it as an input or register it as a buffer")
    return ()  # Python scalars broadcast as 0-d


# ---------------------------------------------------------------------- shape helpers
def bdim(a: Any, b: Any) -> Any:
    if isinstance(a, int) and a == 1:
        return b
    if isinstance(b, int) and b == 1:
        return a
    k = sym_eq_static(a, b)
    if k is True:
        return a
    env = _env()
    ha = a.hint if isinstance(a, SymInt) else a
    hb = b.hint if isinstance(b, SymInt) else b
    if ha == hb:
        env.guard(sympy.Eq(sympy.sympify(unwrap(a)), sympy.sympify(unwrap(b))))
        return SymInt.make(unwrap(a), env) if isinstance(a, SymInt) else a
    if ha == 1:
        env.guard(sympy.Eq(sympy.sympify(unwrap(a)), 1))
        return b
    if hb == 1:
        env.guard(sympy.Eq(sympy.sympify(unwrap(b)), 1))
        return a
    raise RuntimeError(f"sizes {ha} and {hb} do not broadcast")


def broadcast(*shapes: tuple) -> tuple:
    nd = max((len(s) for s in shapes), default=0)
    out = []
    for i in range(nd):
        cur: Any = 1
        for s in shapes:
            j = len(s) - nd + i
            if j >= 0:
                cur = bdim(cur, s[j])
        out.append(cur)
    return tuple(out)


def same(a: Any, b: Any, what: str) -> Any:
    """Two sizes that must be equal (no broadcasting)."""
    k = sym_eq_static(a, b)
    if k is True:
        return a
    if k is False:
        raise RuntimeError(f"{what}: {a} != {b}")
    _env().guard(sympy.Eq(sympy.sympify(unwrap(a)), sympy.sympify(unwrap(b))))
    return SymInt.make(unwrap(a), _env()) if isinstance(a, SymInt) else a


def norm_dim(d: int, nd: int) -> int:
    if not isinstance(d, int):
        raise Unsupported("a dimension index must be a Python int")
    if d < -nd or d >= max(nd, 1):
        raise RuntimeError(f"dimension {d} out of range for rank {nd}")
    return d % nd if nd else 0


def le(a: Any, b: Any) -> bool:
    """Decide a <= b, guarding the branch if it depends on the symbols."""
    if isinstance(a, int) and isinstance(b, int):
        return a <= b
    return _env().decide(sympy.Le(sympy.sympify(unwrap(a)), sympy.sympify(unwrap(b))))


def smin(a: Any, b: Any) -> Any:
    return a if le(a, b) else b


def smax(a: Any, b: Any) -> Any:
    return b if le(a, b) else a


def mk(e: Any) -> Any:
    return SymInt.make(unwrap(e), _env()) if not isinstance(e, int) else e


# ---------------------------------------------------------------------- rule families
def unary(target: str) -> Callable:
    def rule(x, *a, **k):
        return emit(target, (x, *a), k, [_shape(x)])

    return rule


def binary(target: str) -> Callable:
    def rule(x, y, *a, **k):
        if not (_is_tensor(x) or _is_tensor(y)):
            raise Unsupported(f"{target} on two non-tensors")
        return emit(target, (x, y, *a), k, [broadcast(_shape(x), _shape(y))])

    return rule


def reduction(target: str) -> Callable:
    def rule(x, dim=None, keepdim=False, *a, **k):
        if "dim" in k:
            dim = k.pop("dim")
        if "keepdim" in k:
            keepdim = k.pop("keepdim")
        nd = x.ndim
        if dim is None:
            dims = set(range(nd))
        else:
            dims = {norm_dim(d, nd) for d in (dim if isinstance(dim, (tuple, list)) else [dim])}
        shape = [1 if i in dims else s for i, s in enumerate(x.shape)] if keepdim else [s for i, s in enumerate(x.shape) if i not in dims]
        args = (x,) if dim is None and not keepdim else (x, dim, keepdim)
        return emit(target, (*args, *a), k, [tuple(shape)])

    return rule


def same_shape_dim(target: str) -> Callable:
    """softmax-like: (x, dim, ...) -> same shape."""

    def rule(x, *a, **k):
        return emit(target, (x, *a), k, [x.shape])

    return rule


# ---------------------------------------------------------------------- views
def _shape_args(args: tuple) -> list:
    if len(args) == 1 and isinstance(args[0], (tuple, list, torch.Size)):
        return list(args[0])
    return list(args)


def reshape(x, *shape, _target="torch.Tensor.reshape"):
    shape = _shape_args(shape)
    numel = prod(x.shape)
    neg = [i for i, s in enumerate(shape) if isinstance(s, int) and s == -1]
    if len(neg) > 1:
        raise RuntimeError("only one -1 allowed in a shape")
    known = prod(s for s in shape if not (isinstance(s, int) and s == -1))
    out = list(shape)
    if neg:
        e_num, e_den = sympy.sympify(unwrap(numel)), sympy.sympify(unwrap(known))
        if isinstance(numel, int) and isinstance(known, int):
            if numel % known:
                raise RuntimeError("reshape: size not divisible")
            out[neg[0]] = numel // known
        else:
            q = _env().simplify(e_num / e_den)
            if not (q.is_integer or (q.is_number and float(q).is_integer())):
                _env().guard(sympy.Eq(sympy.Mod(e_num, e_den), 0))
                q = _env().simplify(sympy.floor(e_num / e_den))
            out[neg[0]] = mk(q)
    else:
        same(numel, known, "reshape: number of elements")
    out = [mk(s) for s in out]
    return emit(_target, (x, *shape), {}, [tuple(out)])


def view(x, *shape):
    if len(shape) == 1 and isinstance(shape[0], torch.dtype):
        raise Unsupported("view(dtype)")
    return reshape(x, *shape, _target="torch.Tensor.view")


def transpose(x, d0, d1, _target="torch.Tensor.transpose"):
    nd = x.ndim
    a, b = norm_dim(d0, nd), norm_dim(d1, nd)
    s = list(x.shape)
    s[a], s[b] = s[b], s[a]
    return emit(_target, (x, d0, d1), {}, [tuple(s)])


def permute(x, *dims, _target="torch.Tensor.permute"):
    dims = _shape_args(dims)
    nd = x.ndim
    order = [norm_dim(d, nd) for d in dims]
    return emit(_target, (x, *dims), {}, [tuple(x.shape[i] for i in order)])


def unsqueeze(x, d, _target="torch.Tensor.unsqueeze"):
    nd = x.ndim + 1
    i = norm_dim(d, nd)
    s = list(x.shape)
    s.insert(i, 1)
    return emit(_target, (x, d), {}, [tuple(s)])


def squeeze(x, d=None, _target="torch.Tensor.squeeze"):
    if d is None:
        if any(not isinstance(s, int) for s in x.shape):
            pass  # symbolic sizes are >= 2, so never squeezed
        s = [v for v in x.shape if not (isinstance(v, int) and v == 1)]
        return emit(_target, (x,), {}, [tuple(s)])
    i = norm_dim(d, x.ndim)
    s = list(x.shape)
    if isinstance(s[i], int) and s[i] == 1:
        s.pop(i)
    return emit(_target, (x, d), {}, [tuple(s)])


def flatten(x, start_dim=0, end_dim=-1, _target="torch.Tensor.flatten"):
    nd = x.ndim
    if nd == 0:
        return emit(_target, (x, start_dim, end_dim), {}, [(1,)])
    a, b = norm_dim(start_dim, nd), norm_dim(end_dim, nd)
    s = list(x.shape)
    merged = mk(prod(s[a : b + 1]))
    return emit(_target, (x, start_dim, end_dim), {}, [tuple(s[:a] + [merged] + s[b + 1 :])])


def expand(x, *sizes):
    sizes = _shape_args(sizes)
    nd = len(sizes)
    pad = nd - x.ndim
    out = []
    for i, n in enumerate(sizes):
        if i < pad:
            out.append(n)
            continue
        old = x.shape[i - pad]
        if isinstance(n, int) and n == -1:
            out.append(old)
        elif isinstance(old, int) and old == 1:
            out.append(n)
        else:
            out.append(same(old, n, "expand"))
    return emit("torch.Tensor.expand", (x, *sizes), {}, [tuple(mk(v) for v in out)])


def expand_as(x, other):
    return expand(x, *other.shape)


def contiguous(x, *a, **k):
    return emit("torch.Tensor.contiguous", (x, *a), k, [x.shape])


def t_property(x):
    if x.ndim > 2:
        raise Unsupported(".T on more than two dimensions")
    return emit("torch.Tensor.t", (x,), {}, [tuple(reversed(x.shape))])


# ---------------------------------------------------------------------- indexing
def _slice_len(D: Any, sl: slice) -> Any:
    step = 1 if sl.step is None else sl.step
    if isinstance(step, SymInt):
        step = _env().specialize(unwrap(step), "slice step")
    if not isinstance(step, int) or step <= 0:
        raise Unsupported("slice steps must be positive Python ints")

    def norm(v, default):
        if v is None:
            return default
        if (isinstance(v, int) and v < 0) or (isinstance(v, SymInt) and not le(0, v)):
            v = mk(D + v)  # counts from the end; may still fall before the start
            return v if le(0, v) else 0
        return v

    start = norm(sl.start, 0)
    stop = norm(sl.stop, D)
    start = smin(start, D)
    stop = smin(stop, D)
    length = mk(stop - start) if le(start, stop) else 0
    if step != 1 and not isinstance(length, int):
        length = mk(sympy.ceiling(sympy.sympify(unwrap(length)) / step))
    elif step != 1:
        length = -(-length // step)
    return length


def getitem(x, idx):
    if not isinstance(idx, tuple):
        idx = (idx,)
    if any(isinstance(i, (TensorProxy, list, torch.Tensor)) for i in idx):
        raise Unsupported("indexing with tensors or lists (advanced indexing)")
    n_real = builtins.sum(1 for i in idx if i is not None and i is not Ellipsis)
    if n_real > x.ndim:
        raise RuntimeError("too many indices")
    expanded: list = []
    for i in idx:
        if i is Ellipsis:
            expanded.extend([slice(None)] * (x.ndim - n_real))
        else:
            expanded.append(i)
    if Ellipsis not in idx:
        expanded.extend([slice(None)] * (x.ndim - n_real))
    out, d = [], 0
    for i in expanded:
        if i is None:
            out.append(1)
        elif isinstance(i, slice):
            out.append(_slice_len(x.shape[d], i))
            d += 1
        elif isinstance(i, (int, SymInt)):
            d += 1
        else:
            raise Unsupported(f"index of type {type(i).__name__}")
    return emit("operator.getitem", (x, idx), {}, [tuple(out)])


# ---------------------------------------------------------------------- linear algebra
def matmul(a, b, _target="torch.matmul"):
    if not (_is_tensor(a) and _is_tensor(b)):
        raise Unsupported("matmul with a non-tensor")
    sa, sb = list(a.shape), list(b.shape)
    if len(sa) == 1 and len(sb) == 1:
        same(sa[0], sb[0], "dot")
        return emit(_target, (a, b), {}, [()])
    if len(sb) == 1:
        same(sa[-1], sb[0], "matmul")
        return emit(_target, (a, b), {}, [tuple(sa[:-1])])
    if len(sa) == 1:
        same(sa[0], sb[-2], "matmul")
        return emit(_target, (a, b), {}, [tuple(sb[:-2] + sb[-1:])])
    same(sa[-1], sb[-2], "matmul inner dimension")
    batch = broadcast(tuple(sa[:-2]), tuple(sb[:-2]))
    return emit(_target, (a, b), {}, [tuple(batch) + (sa[-2], sb[-1])])


def linear(x, w, b=None):
    same(x.shape[-1], w.shape[-1], "linear in_features")
    out = tuple(x.shape[:-1]) + (w.shape[0],) if w.ndim == 2 else tuple(x.shape[:-1])
    if b is not None:
        broadcast(out, _shape(b))
    return emit("torch.nn.functional.linear", (x, w, b), {}, [out])


def outer(a, b):
    return emit("torch.outer", (a, b), {}, [(a.shape[0], b.shape[0])])


def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    same(q.shape[-1], k.shape[-1], "attention head size (q, k)")
    same(k.shape[-2], v.shape[-2], "attention key length (k, v)")
    batch = broadcast(tuple(q.shape[:-2]), tuple(k.shape[:-2]), tuple(v.shape[:-2])) if not enable_gqa else tuple(q.shape[:-2])
    out = tuple(batch) + (q.shape[-2], v.shape[-1])
    if attn_mask is not None:
        broadcast(tuple(batch) + (q.shape[-2], k.shape[-2]), _shape(attn_mask))
    kw = {"attn_mask": attn_mask, "dropout_p": dropout_p, "is_causal": is_causal}
    if scale is not None:
        kw["scale"] = scale
    if enable_gqa:
        kw["enable_gqa"] = True
    return emit("torch.nn.functional.scaled_dot_product_attention", (q, k, v), kw, [out], pure=dropout_p == 0.0)


# ---------------------------------------------------------------------- normalisation, embeddings, losses
def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    for a, b in zip(reversed(list(normalized_shape)), reversed(x.shape)):
        same(a, b, "layer_norm normalized_shape")
    return emit("torch.nn.functional.layer_norm", (x, tuple(normalized_shape), weight, bias, eps), {}, [x.shape])


def rms_norm(x, normalized_shape, weight=None, eps=None):
    for a, b in zip(reversed(list(normalized_shape)), reversed(x.shape)):
        same(a, b, "rms_norm normalized_shape")
    return emit("torch.nn.functional.rms_norm", (x, tuple(normalized_shape), weight, eps), {}, [x.shape])


def embedding(idx, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    if max_norm is not None or sparse:
        raise Unsupported("embedding with max_norm or sparse gradients")
    return emit("torch.nn.functional.embedding", (idx, weight), {"padding_idx": padding_idx}, [tuple(idx.shape) + (weight.shape[1],)])


def cross_entropy(inp, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean", label_smoothing=0.0):
    same(inp.shape[0], target.shape[0], "cross_entropy batch")
    shape = () if reduction != "none" else tuple(target.shape)
    kw = {"ignore_index": ignore_index, "reduction": reduction, "label_smoothing": label_smoothing}
    return emit("torch.nn.functional.cross_entropy", (inp, target), kw, [shape])


def dropout(x, p=0.5, training=True, inplace=False):
    if inplace:
        raise Unsupported("in-place dropout")
    return emit("torch.nn.functional.dropout", (x,), {"p": p, "training": training}, [x.shape], pure=not (training and p > 0))


def act(target: str, *, allow: tuple = ()) -> Callable:
    def rule(x, *a, **k):
        if k.pop("inplace", False):
            raise Unsupported(f"in-place {target}")
        k = {kk: vv for kk, vv in k.items() if kk in allow}
        return emit(target, (x, *a), k, [x.shape])

    return rule


# ---------------------------------------------------------------------- joins and splits
def cat(tensors, dim=0):
    tensors = list(tensors)
    nd = tensors[0].ndim
    d = norm_dim(dim, nd)
    total: Any = 0
    base = list(tensors[0].shape)
    for t in tensors:
        for i in range(nd):
            if i != d:
                base[i] = same(base[i], t.shape[i], "cat")
        total = total + t.shape[d]
    base[d] = mk(total)
    return emit("torch.cat", (tensors, dim), {}, [tuple(base)])


def stack(tensors, dim=0):
    tensors = list(tensors)
    base = list(tensors[0].shape)
    for t in tensors[1:]:
        for i in range(len(base)):
            base[i] = same(base[i], t.shape[i], "stack")
    d = norm_dim(dim, len(base) + 1)
    base.insert(d, len(tensors))
    return emit("torch.stack", (tensors, dim), {}, [tuple(base)])


def split(x, split_size, dim=0, _target="torch.Tensor.split"):
    d = norm_dim(dim, x.ndim)
    D = x.shape[d]
    if isinstance(split_size, (list, tuple)):
        sizes = list(split_size)
        same(mk(sum(sizes[1:], sizes[0])), D, "split sizes")
    else:
        if not isinstance(D, int):
            D = _env().specialize(unwrap(D), "number of split outputs")
        if isinstance(split_size, SymInt):
            split_size = int(split_size)
        n = -(-D // split_size)
        sizes = [split_size] * (n - 1) + [D - split_size * (n - 1)]
    shapes = []
    for s in sizes:
        sh = list(x.shape)
        sh[d] = mk(s)
        shapes.append(tuple(sh))
    return tuple(emit(_target, (x, split_size, dim), {}, shapes))


def chunk(x, chunks, dim=0):
    d = norm_dim(dim, x.ndim)
    D = x.shape[d]
    if isinstance(D, int):
        size = -(-D // chunks)
        return _chunk_static(x, chunks, dim, D, size, d)
    env = _env()
    env.guard(sympy.Eq(sympy.Mod(sympy.sympify(unwrap(D)), chunks), 0))
    each = mk(sympy.sympify(unwrap(D)) / chunks)
    shapes = []
    for _ in range(chunks):
        sh = list(x.shape)
        sh[d] = each
        shapes.append(tuple(sh))
    return tuple(emit("torch.Tensor.chunk", (x, chunks, dim), {}, shapes))


def _chunk_static(x, chunks, dim, D, size, d):
    sizes = []
    left = D
    while left > 0:
        sizes.append(min(size, left))
        left -= size
    shapes = []
    for s in sizes:
        sh = list(x.shape)
        sh[d] = s
        shapes.append(tuple(sh))
    return tuple(emit("torch.Tensor.chunk", (x, chunks, dim), {}, shapes))


# ---------------------------------------------------------------------- creation
def _factory_shape(args) -> tuple:
    return tuple(mk(s) for s in _shape_args(args))


def factory(target: str) -> Callable:
    def rule(*size, **k):
        if "out" in k or "generator" in k:
            raise Unsupported(f"{target} with out= or generator=")
        if "size" in k:
            size = (k.pop("size"),)
        shape = _factory_shape(size)
        return emit(target, (tuple(shape),), k, [shape])

    return rule


def full(size, fill_value, **k):
    shape = _factory_shape((size,))
    return emit("torch.full", (tuple(shape), fill_value), k, [shape])


def like(target: str) -> Callable:
    def rule(x, *a, **k):
        return emit(target, (x, *a), k, [x.shape])

    return rule


def arange(*a, **k):
    if len(a) == 1:
        start, end, step = 0, a[0], 1
    elif len(a) == 2:
        start, end, step = a[0], a[1], 1
    else:
        start, end, step = a
    if isinstance(start, float) or isinstance(end, float) or isinstance(step, float):
        if any(isinstance(v, SymInt) for v in (start, end, step)):
            raise Unsupported("arange with float bounds and symbolic sizes")
        n = max(0, math.ceil((end - start) / step))
    else:
        if not isinstance(step, int) or step <= 0:
            raise Unsupported("arange step must be a positive Python int")
        length = mk(end - start)
        length = length if le(0, length) else 0
        n = (
            length
            if step == 1
            else mk(sympy.ceiling(sympy.sympify(unwrap(length)) / step))
            if not isinstance(length, int)
            else -(-length // step)
        )
    return emit("torch.arange", tuple(a), k, [(n,)])


def tensor_const(data, **k):
    if isinstance(data, (TensorProxy, SymInt)) or (
        isinstance(data, (list, tuple)) and any(isinstance(v, (TensorProxy, SymInt)) for v in data)
    ):
        raise Unsupported("torch.tensor of traced values")
    return emit("torch.tensor", (data,), k, [tuple(ORIG["torch.tensor"](data).shape)])


def where(cond, a, b):
    return emit("torch.where", (cond, a, b), {}, [broadcast(_shape(cond), _shape(a), _shape(b))])


def masked_fill(x, mask, value):
    broadcast(x.shape, _shape(mask))
    out = broadcast(x.shape, _shape(mask))
    for o, s in zip(out[-x.ndim :] if x.ndim else (), x.shape):
        same(o, s, "masked_fill mask must broadcast to the tensor")
    return emit("torch.Tensor.masked_fill", (x, mask, value), {}, [x.shape])


def to(x, *a, **k):
    for v in list(a) + list(k.values()):
        if isinstance(v, TensorProxy):
            raise Unsupported("to(other_tensor)")
    return emit("torch.Tensor.to", (x, *a), k, [x.shape])


def type_as(x, other):
    return emit("torch.Tensor.type_as", (x, other), {}, [x.shape])


def clamp(x, min=None, max=None):
    return emit("torch.Tensor.clamp", (x,), {"min": min, "max": max}, [x.shape])


def cumsum(x, dim, **k):
    return emit("torch.Tensor.cumsum", (x, dim), k, [x.shape])


def argmax(x, dim=None, keepdim=False):
    if dim is None:
        return emit("torch.Tensor.argmax", (x,), {}, [()])
    d = norm_dim(dim, x.ndim)
    shape = [1 if i == d else s for i, s in enumerate(x.shape)] if keepdim else [s for i, s in enumerate(x.shape) if i != d]
    return emit("torch.Tensor.argmax", (x, dim, keepdim), {}, [tuple(shape)])


def tri(target):
    def rule(x, diagonal=0):
        return emit(target, (x, diagonal), {}, [x.shape])

    return rule


def size_method(x, d=None):
    return x.size(d)


# ---------------------------------------------------------------------- tables
UNARY = [
    "exp",
    "log",
    "sqrt",
    "rsqrt",
    "tanh",
    "sigmoid",
    "relu",
    "abs",
    "neg",
    "sin",
    "cos",
    "reciprocal",
    "square",
    "logical_not",
    "detach",
    "clone",
    "float",
    "half",
    "bfloat16",
    "long",
    "int",
    "bool",
    "double",
    "isnan",
]
BINARY = [
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "maximum",
    "minimum",
    "lt",
    "le",
    "gt",
    "ge",
    "eq",
    "ne",
    "logical_and",
    "logical_or",
    "remainder",
    "floor_divide",
]
REDUCTIONS = ["sum", "mean", "amax", "amin", "var", "std", "logsumexp", "all", "any"]

for n in UNARY:
    METHODS[n] = unary(f"torch.Tensor.{n}")
for n in BINARY:
    METHODS[n] = binary(f"torch.Tensor.{n}")
for n in REDUCTIONS:
    METHODS[n] = reduction(f"torch.Tensor.{n}")
for n in ["add", "sub", "mul", "truediv", "matmul", "pow"]:
    METHODS[f"__r{n}__"] = binary(f"torch.Tensor.__r{n}__")
METHODS.update(
    {
        "softmax": same_shape_dim("torch.Tensor.softmax"),
        "log_softmax": same_shape_dim("torch.Tensor.log_softmax"),
        "view": view,
        "reshape": reshape,
        "transpose": transpose,
        "permute": permute,
        "unsqueeze": unsqueeze,
        "squeeze": squeeze,
        "flatten": flatten,
        "expand": expand,
        "expand_as": expand_as,
        "contiguous": contiguous,
        "t_property": t_property,
        "t": t_property,
        "__getitem__": getitem,
        "matmul": lambda a, b: matmul(a, b, _target="torch.matmul"),
        "masked_fill": masked_fill,
        "to": to,
        "type_as": type_as,
        "clamp": clamp,
        "cumsum": cumsum,
        "argmax": argmax,
        "tril": tri("torch.Tensor.tril"),
        "triu": tri("torch.Tensor.triu"),
        "split": split,
        "chunk": chunk,
        "size": size_method,
    }
)

FUNCS: dict[str, Callable] = {
    "torch.matmul": matmul,
    "torch.bmm": lambda a, b: matmul(a, b, _target="torch.bmm"),
    "torch.mm": lambda a, b: matmul(a, b, _target="torch.mm"),
    "torch.outer": outer,
    "torch.cat": cat,
    "torch.concat": cat,
    "torch.stack": stack,
    "torch.where": where,
    "torch.arange": arange,
    "torch.ones": factory("torch.ones"),
    "torch.zeros": factory("torch.zeros"),
    "torch.empty": factory("torch.empty"),
    "torch.full": full,
    "torch.ones_like": like("torch.ones_like"),
    "torch.zeros_like": like("torch.zeros_like"),
    "torch.full_like": like("torch.full_like"),
    "torch.tensor": tensor_const,
    "torch.tril": tri("torch.tril"),
    "torch.triu": tri("torch.triu"),
    "torch.transpose": lambda x, a, b: transpose(x, a, b, _target="torch.transpose"),
    "torch.permute": lambda x, dims: permute(x, *dims, _target="torch.permute"),
    "torch.reshape": lambda x, shape: reshape(x, *shape, _target="torch.reshape"),
    "torch.flatten": lambda x, s=0, e=-1: flatten(x, s, e, _target="torch.flatten"),
    "torch.unsqueeze": lambda x, d: unsqueeze(x, d, _target="torch.unsqueeze"),
    "torch.squeeze": lambda x, d=None: squeeze(x, d, _target="torch.squeeze"),
    "torch.split": lambda x, s, dim=0: split(x, s, dim, _target="torch.split"),
    "torch.softmax": same_shape_dim("torch.softmax"),
    "torch.nn.functional.linear": linear,
    "torch.nn.functional.scaled_dot_product_attention": sdpa,
    "torch.nn.functional.layer_norm": layer_norm,
    "torch.nn.functional.rms_norm": rms_norm,
    "torch.nn.functional.embedding": embedding,
    "torch.nn.functional.cross_entropy": cross_entropy,
    "torch.nn.functional.dropout": dropout,
    "torch.nn.functional.gelu": act("torch.nn.functional.gelu", allow=("approximate",)),
    "torch.nn.functional.silu": act("torch.nn.functional.silu"),
    "torch.nn.functional.relu": act("torch.nn.functional.relu"),
    "torch.nn.functional.softmax": same_shape_dim("torch.nn.functional.softmax"),
    "torch.nn.functional.log_softmax": same_shape_dim("torch.nn.functional.log_softmax"),
    "torch.nn.functional.sigmoid": act("torch.sigmoid"),
    "torch.nn.functional.tanh": act("torch.tanh"),
}
for n in ["exp", "log", "sqrt", "rsqrt", "tanh", "sigmoid", "relu", "abs", "neg", "sin", "cos", "logical_not"]:
    FUNCS[f"torch.{n}"] = unary(f"torch.{n}")
for n in ["add", "sub", "mul", "div", "pow", "maximum", "minimum", "logical_and", "logical_or", "eq", "ne", "lt", "gt"]:
    FUNCS[f"torch.{n}"] = binary(f"torch.{n}")
for n in ["sum", "mean", "amax", "amin", "var", "logsumexp"]:
    FUNCS[f"torch.{n}"] = reduction(f"torch.{n}")


def _module_and_attr(target: str):
    if target.startswith("torch.nn.functional."):
        return F, target.rsplit(".", 1)[1]
    return torch, target.split(".", 1)[1]


for _t in list(FUNCS):
    _m, _a = _module_and_attr(_t)
    if hasattr(_m, _a):
        ORIG[_t] = getattr(_m, _a)
        LOOKASIDES[(_m, _a)] = FUNCS[_t]


def _needs_trace(args, kwargs) -> bool:
    def walk(v):
        if isinstance(v, (TensorProxy, SymInt, SymFloat)):
            return True
        if isinstance(v, (list, tuple)):
            return any(walk(i) for i in v)
        if isinstance(v, dict):
            return any(walk(i) for i in v.values())
        return False

    return walk(args) or walk(kwargs)


def capture_args(v: Any) -> Any:
    """Real tensors among a traced call's arguments become captured constants of the program."""
    if isinstance(v, torch.Tensor):  # including parameters of modules the function closes over
        return active().capture(v)
    if isinstance(v, list):
        return [capture_args(i) for i in v]
    if isinstance(v, tuple) and not isinstance(v, torch.Size):
        return tuple(capture_args(i) for i in v)
    if isinstance(v, dict):
        return {k: capture_args(i) for k, i in v.items()}
    return v


def _shim(orig: Callable, rule: Callable, name: str) -> Callable:
    def f(*args, **kwargs):
        if getattr(_LOCAL, "tracing", False) and _ACTIVE:
            if name.split(".")[-1] in ("arange", "ones", "zeros", "empty", "full", "tensor") or _needs_trace(args, kwargs):
                if name.endswith(".tensor"):
                    return rule(*args, **kwargs)
                return rule(*capture_args(args), **capture_args(kwargs))
            raise Unsupported(f"`{name}` called only on real tensors inside a traced function; pass them as inputs")
        return orig(*args, **kwargs)

    f.__name__ = getattr(orig, "__name__", name)
    f.__wrapped__ = orig
    return f


DUNDER = {
    "__add__": "add",
    "__sub__": "sub",
    "__mul__": "mul",
    "__truediv__": "div",
    "__matmul__": "matmul",
    "__pow__": "pow",
    "__lt__": "lt",
    "__le__": "le",
    "__gt__": "gt",
    "__ge__": "ge",
    "__eq__": "eq",
    "__ne__": "ne",
    "__and__": "logical_and",
    "__or__": "logical_or",
    "__neg__": "neg",
    "__invert__": "logical_not",
    "__getitem__": "__getitem__",
    "__radd__": "__radd__",
    "__rsub__": "__rsub__",
    "__rmul__": "__rmul__",
    "__rtruediv__": "__rtruediv__",
    "__rmatmul__": "__rmatmul__",
    "__rpow__": "__rpow__",
}


def tensor_method_rule(name: str) -> Callable | None:
    """The rule for a `torch.Tensor` method or operator, by its Python name."""
    return METHODS.get(DUNDER.get(name, name))


def _method_shim(orig: Callable, name: str) -> Callable:
    """`torch.Tensor.<name>` while tracing: a method called on a real tensor the code closed over
    (`pos[:, :n]`, `cls.expand(b, -1, -1)`) is recorded with the tensor captured, so symbolic
    sizes in its arguments stay symbolic."""
    rule = tensor_method_rule(name)

    def f(self, *a, **k):
        if getattr(_LOCAL, "tracing", False) and _ACTIVE and isinstance(self, torch.Tensor):
            return rule(capture_args(self), *capture_args(a), **capture_args(k))
        return orig(self, *a, **k)

    f.__name__ = getattr(orig, "__name__", name)
    return f


def _math_shim(orig: Callable, sym: Callable) -> Callable:
    def f(x):
        if getattr(_LOCAL, "tracing", False) and isinstance(x, (SymInt, SymFloat)):
            return sym(x)
        return orig(x)

    return f


TENSOR_PATCHED = sorted(
    {n for n in list(METHODS) + list(DUNDER) if hasattr(torch.Tensor, n) and tensor_method_rule(n) is not None}
    - {"size", "t", "__eq__", "__ne__"}
)


class lookasides:
    """Context manager: install the recording shims for the duration of one trace."""

    def __enter__(self):
        self.saved = []
        for (m, a), rule in LOOKASIDES.items():
            target = ("torch.nn.functional." if m is F else "torch.") + a
            self.saved.append((m, a, getattr(m, a)))
            setattr(m, a, _shim(ORIG[target], rule, target))
        for n in TENSOR_PATCHED:
            orig = torch.Tensor.__dict__.get(n) or getattr(torch.Tensor, n)
            self.saved.append((torch.Tensor, n, torch.Tensor.__dict__.get(n, _ABSENT)))
            setattr(torch.Tensor, n, _method_shim(orig, n))
        self.saved.append((math, "sqrt", math.sqrt))
        math.sqrt = _math_shim(math.sqrt, sym_sqrt)
        _LOCAL.tracing = True
        return self

    def __exit__(self, *exc):
        _LOCAL.tracing = False
        for m, a, v in reversed(self.saved):
            if v is _ABSENT:
                delattr(m, a)
            else:
                setattr(m, a, v)
        return False


_ABSENT = object()


def supported_ops() -> dict[str, list[str]]:
    return {"functions": sorted(FUNCS), "tensor methods": sorted(k for k in METHODS if not k.startswith("_") and k != "t_property")}


__all__ = ["METHODS", "FUNCS", "lookasides", "resolve", "shape_hint"]
