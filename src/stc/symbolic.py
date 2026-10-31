"""Symbolic integers and the shape environment.

A traced program refers to tensor sizes through symbols (`s0`, `s1`, ...). Every decision the
tracer takes that depends on a size (two dimensions broadcast because they are equal, a slice
stops at the end of a dimension, Python code branches on `x.shape[1] > 64`) is recorded as a
*guard*: a condition on the symbols under which the traced program is valid. The guards become
the program's prologue, which decides at call time whether a cached program can be reused.

Each symbol carries a *hint*, the concrete size seen while tracing. A decision is taken by
evaluating it on the hints and recording the branch taken as a guard. This is the same contract
PyTorch's own symbolic shapes use; the difference here is that the whole program, guards and
computation, is emitted as readable Python.

Equalities between symbols (`Eq(s3, s1)`) are not kept as guards: they are applied as
replacements, so the program names one symbol for both dimensions and the prologue checks the
equality once while binding.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import sympy

INT = (int,)


class GuardFailedAtTrace(RuntimeError):
    """A condition that is false for the hints: the eager program would fail too."""


@dataclass
class Source:
    """Where a symbol's value is read at call time: input `arg` (flat index), dimension `dim`
    (None for an integer input)."""

    arg: int
    dim: int | None

    def code(self, names: list[str]) -> str:
        return names[self.arg] if self.dim is None else f"{names[self.arg]}.shape[{self.dim}]"


@dataclass
class ShapeEnv:
    specialize_zero_one: bool = True
    symbols: list[sympy.Symbol] = field(default_factory=list)
    hints: dict[sympy.Symbol, int] = field(default_factory=dict)
    sources: dict[sympy.Symbol, list[Source]] = field(default_factory=dict)
    replacements: dict[sympy.Symbol, sympy.Expr] = field(default_factory=dict)
    guards: list[sympy.Basic] = field(default_factory=list)
    raw_guards: int = 0
    specializations: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ symbols
    def create(self, hint: int, source: Source, *, dynamic: bool = True) -> int | SymInt:
        """A size from an input. Static, or 0/1 under 0/1 specialisation, gives a plain int."""
        if not dynamic or (self.specialize_zero_one and hint in (0, 1)):
            return hint
        s = sympy.Symbol(f"s{len(self.symbols)}", integer=True, positive=True)
        self.symbols.append(s)
        self.hints[s] = int(hint)
        self.sources[s] = [source]
        if self.specialize_zero_one:
            self.guards.append(sympy.Ge(s, 2))  # the symbol stands for sizes >= 2 only
        return SymInt(s, self)

    def simplify(self, e: Any) -> Any:
        e = sympy.sympify(e)
        for _ in range(8):
            new = e.xreplace(self.replacements)
            if new == e:
                break
            e = new
        if isinstance(e, sympy.Basic) and not e.is_Atom:
            e = sympy.simplify(e)
        return e

    def hint_of(self, e: Any) -> Any:
        e = self.simplify(e)
        v = e.xreplace({s: sympy.Integer(h) for s, h in self.hints.items()})
        if isinstance(v, sympy.logic.boolalg.BooleanAtom):
            return bool(v)
        if isinstance(v, sympy.Basic) and v.free_symbols:
            raise RuntimeError(f"cannot evaluate {e} on hints")
        if v.is_Integer:
            return int(v)
        if v.is_Rational or v.is_Float:
            return float(v)
        return bool(v)

    # ------------------------------------------------------------------ guards
    def guard(self, cond: Any) -> None:
        """Record that `cond` holds (it must hold for the hints)."""
        self.raw_guards += 1
        cond = self.simplify(cond)
        if cond is sympy.true or cond == True:  # noqa: E712
            return
        if cond is sympy.false or cond == False:  # noqa: E712
            raise GuardFailedAtTrace("a size condition is false for the traced sizes")
        if isinstance(cond, sympy.Eq):
            lhs, rhs = cond.lhs, cond.rhs
            for a, b in ((lhs, rhs), (rhs, lhs)):
                if isinstance(a, sympy.Symbol) and a in self.hints and a not in b.free_symbols:
                    # replace the newer symbol by the older expression
                    if isinstance(b, sympy.Symbol) and b in self.hints and self.symbols.index(b) > self.symbols.index(a):
                        a, b = b, a
                    self.replacements[a] = b
                    self.sources.setdefault(a, [])
                    self._resimplify()
                    return
        if not self.hint_of(cond):
            raise GuardFailedAtTrace(f"guard {cond} is false for the traced sizes")
        if cond not in self.guards:
            self.guards.append(cond)

    def _resimplify(self) -> None:
        out = []
        for g in self.guards:
            g2 = self.simplify(g)
            if g2 is sympy.true or g2 == True:  # noqa: E712
                continue
            if g2 not in out:
                out.append(g2)
        self.guards = out

    def decide(self, cond: Any) -> bool:
        """Evaluate a condition on the hints and guard the branch taken."""
        c = self.simplify(cond)
        if c is sympy.true or c == True:  # noqa: E712
            return True
        if c is sympy.false or c == False:  # noqa: E712
            return False
        v = bool(self.hint_of(c))
        self.guard(c if v else sympy.Not(c))
        return v

    def specialize(self, e: Any, why: str) -> int:
        v = int(self.hint_of(e))
        self.specializations.append(f"{why}: {self.simplify(e)} = {v}")
        self.guard(sympy.Eq(self.simplify(e), v))
        return v

    def final_guards(self) -> list[sympy.Basic]:
        """Guards after replacements, without duplicates or ones implied by symbol positivity."""
        self._resimplify()
        keep = []
        for g in self.guards:
            if isinstance(g, sympy.Ge) and isinstance(g.lhs, sympy.Symbol) and g.rhs == 2 and g.lhs in self.replacements:
                continue
            keep.append(g)
        return keep

    def live_symbols(self) -> list[sympy.Symbol]:
        return [s for s in self.symbols if s not in self.replacements]


def unwrap(x: Any) -> Any:
    return x.expr if isinstance(x, SymInt) else x


class SymInt:
    """An integer known only as an expression over size symbols, with a hint."""

    __slots__ = ("expr", "env")

    def __init__(self, expr: Any, env: ShapeEnv):
        self.expr = sympy.sympify(expr)
        self.env = env

    # an expression that simplifies to a constant stops being symbolic
    @staticmethod
    def make(expr: Any, env: ShapeEnv) -> int | SymInt:
        e = env.simplify(expr)
        if isinstance(e, sympy.Integer) or (isinstance(e, sympy.Basic) and e.is_Integer):
            return int(e)
        return SymInt(e, env)

    @property
    def hint(self) -> int:
        return int(self.env.hint_of(self.expr))

    def _bin(self, other: Any, fn, reverse: bool = False) -> int | SymInt | float:
        if isinstance(other, float):
            return fn(other, float(self)) if reverse else fn(float(self), other)
        if not isinstance(other, (int, SymInt)):
            return NotImplemented
        a, b = self.expr, sympy.sympify(unwrap(other))
        return SymInt.make(fn(b, a) if reverse else fn(a, b), self.env)

    def __add__(self, o):
        return self._bin(o, operator.add)

    def __radd__(self, o):
        return self._bin(o, operator.add, True)

    def __sub__(self, o):
        return self._bin(o, operator.sub)

    def __rsub__(self, o):
        return self._bin(o, operator.sub, True)

    def __mul__(self, o):
        return self._bin(o, operator.mul)

    def __rmul__(self, o):
        return self._bin(o, operator.mul, True)

    def __floordiv__(self, o):
        return self._bin(o, lambda a, b: sympy.floor(a / b) if not isinstance(a, float) else a // b)

    def __rfloordiv__(self, o):
        return self._bin(o, lambda a, b: sympy.floor(a / b) if not isinstance(a, float) else a // b, True)

    def __mod__(self, o):
        return self._bin(o, lambda a, b: sympy.Mod(a, b) if not isinstance(a, float) else a % b)

    def __rmod__(self, o):
        return self._bin(o, lambda a, b: sympy.Mod(a, b) if not isinstance(a, float) else a % b, True)

    def __truediv__(self, o):
        return SymFloat.make(self.expr, self.env) / o

    def __rtruediv__(self, o):
        return o / SymFloat.make(self.expr, self.env)

    def __pow__(self, o):
        if isinstance(o, int) and not isinstance(o, bool) and o >= 0:
            return SymInt.make(self.expr**o, self.env)
        return SymFloat.make(self.expr, self.env) ** o

    def __rpow__(self, o):
        if isinstance(o, int) and not isinstance(o, bool) and o > 0:
            return SymInt.make(sympy.Integer(o) ** self.expr, self.env)
        return o ** SymFloat.make(self.expr, self.env)

    def __abs__(self):
        return self  # sizes are positive

    def __neg__(self):
        return SymInt.make(-self.expr, self.env)

    def __pos__(self):
        return self

    def _cmp(self, o, fn) -> SymBool | bool:
        if isinstance(o, float):
            return fn(float(self), o)
        if not isinstance(o, (int, SymInt)):
            return NotImplemented
        return SymBool(fn(self.expr, sympy.sympify(unwrap(o))), self.env)

    def __eq__(self, o):  # type: ignore[override]
        return self._cmp(o, sympy.Eq)

    def __ne__(self, o):  # type: ignore[override]
        return self._cmp(o, sympy.Ne)

    def __lt__(self, o):
        return self._cmp(o, sympy.Lt)

    def __le__(self, o):
        return self._cmp(o, sympy.Le)

    def __gt__(self, o):
        return self._cmp(o, sympy.Gt)

    def __ge__(self, o):
        return self._cmp(o, sympy.Ge)

    def __hash__(self):
        return hash(self.expr)

    # leaving the symbolic world specialises: the value becomes a constant of the program
    def __index__(self) -> int:
        return self.env.specialize(self.expr, "used as a Python int")

    __int__ = __index__

    def __float__(self) -> float:
        return float(self.env.specialize(self.expr, "used as a Python float"))

    def __bool__(self) -> bool:
        return self.env.decide(sympy.Ne(self.expr, 0))

    def __repr__(self) -> str:
        return f"SymInt({self.env.simplify(self.expr)}, hint={self.hint})"


class SymFloat:
    """A float computed from sizes (an attention scale `head_dim ** -0.5`, `1 / math.sqrt(d)`).

    Arithmetic stays symbolic and is printed into the program, so the scale follows the size
    instead of fixing it. Leaving the symbolic world (`float(x)`) specialises the sizes it uses."""

    __slots__ = ("expr", "env")

    def __init__(self, expr: Any, env: ShapeEnv):
        self.expr = sympy.sympify(expr)
        self.env = env

    @staticmethod
    def make(expr: Any, env: ShapeEnv) -> SymFloat | float:
        e = env.simplify(sympy.sympify(expr))
        if e.is_number:
            return float(e)
        return SymFloat(e, env)

    def _v(self, o: Any) -> Any:
        if isinstance(o, (SymInt, SymFloat)):
            return o.expr
        if isinstance(o, bool):
            return None
        if isinstance(o, int):
            return sympy.Integer(o)
        if isinstance(o, float):
            return sympy.Float(o)
        return None

    def _bin(self, o, fn, rev=False):
        v = self._v(o)
        if v is None:
            return NotImplemented
        return SymFloat.make(fn(v, self.expr) if rev else fn(self.expr, v), self.env)

    def __add__(self, o):
        return self._bin(o, operator.add)

    def __radd__(self, o):
        return self._bin(o, operator.add, True)

    def __sub__(self, o):
        return self._bin(o, operator.sub)

    def __rsub__(self, o):
        return self._bin(o, operator.sub, True)

    def __mul__(self, o):
        return self._bin(o, operator.mul)

    def __rmul__(self, o):
        return self._bin(o, operator.mul, True)

    def __truediv__(self, o):
        return self._bin(o, operator.truediv)

    def __rtruediv__(self, o):
        return self._bin(o, operator.truediv, True)

    def __pow__(self, o):
        return self._bin(o, operator.pow)

    def __rpow__(self, o):
        return self._bin(o, operator.pow, True)

    def __neg__(self):
        return SymFloat.make(-self.expr, self.env)

    def _cmp(self, o, fn):
        v = self._v(o)
        if v is None:
            return NotImplemented
        return SymBool(fn(self.expr, v), self.env)

    def __lt__(self, o):
        return self._cmp(o, sympy.Lt)

    def __le__(self, o):
        return self._cmp(o, sympy.Le)

    def __gt__(self, o):
        return self._cmp(o, sympy.Gt)

    def __ge__(self, o):
        return self._cmp(o, sympy.Ge)

    def __float__(self) -> float:
        for sym in sorted(self.env.simplify(self.expr).free_symbols, key=str):
            self.env.specialize(sym, "used in a Python float")
        return float(self.env.hint_of(self.expr))

    @property
    def hint(self) -> float:
        return float(self.env.hint_of(self.expr))

    def __repr__(self) -> str:
        return f"SymFloat({self.expr}, hint={self.hint:.4g})"


def sym_sqrt(x: Any) -> Any:
    if isinstance(x, (SymInt, SymFloat)):
        return SymFloat.make(sympy.sqrt(x.expr), x.env)
    return __import__("math").sqrt(x)


class SymBool:
    __slots__ = ("expr", "env")

    def __init__(self, expr: Any, env: ShapeEnv):
        self.expr = expr
        self.env = env

    def __bool__(self) -> bool:
        return self.env.decide(self.expr)

    def __repr__(self) -> str:
        return f"SymBool({self.expr})"


# ---------------------------------------------------------------------- helpers for rules
def is_sym(x: Any) -> bool:
    return isinstance(x, SymInt)


def env_of(*xs: Any) -> ShapeEnv | None:
    for x in xs:
        if isinstance(x, SymInt):
            return x.env
        if isinstance(x, (list, tuple)):
            e = env_of(*x)
            if e is not None:
                return e
    return None


def sym_eq_static(a: Any, b: Any) -> bool | None:
    """True/False when the equality of two sizes is known without hints, else None."""
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    env = env_of(a, b)
    d = env.simplify(sympy.sympify(unwrap(a)) - sympy.sympify(unwrap(b)))
    if d == 0:
        return True
    if d.is_number:
        return False
    return None


def prod(xs: Iterable[Any]) -> Any:
    out: Any = 1
    for x in xs:
        out = out * x
    return out


# ---------------------------------------------------------------------- printing as Python
def pycode(e: Any) -> str:
    """Print a sympy integer/boolean expression as Python over symbol names."""
    e = sympy.sympify(e)
    if e is sympy.true:
        return "True"
    if e is sympy.false:
        return "False"
    if isinstance(e, sympy.Symbol):
        return e.name
    if e.is_Integer:
        return str(int(e))
    if e.is_Rational or e.is_Float:
        return repr(float(e))
    if isinstance(e, sympy.Add):
        terms = sorted(e.args, key=lambda t: (t.is_number, str(t)))
        out = pycode(terms[0])
        for t in terms[1:]:
            c, rest = t.as_coeff_Mul()
            if c.is_negative:
                out += f" - {pycode(-t)}"
            else:
                out += f" + {pycode(t)}"
        return f"({out})"
    if isinstance(e, sympy.Mul):
        c, rest = e.as_coeff_Mul()
        if c == -1:
            return f"(-{pycode(rest)})"
        return "(" + " * ".join(pycode(a) for a in e.args) + ")"
    if isinstance(e, sympy.Pow):
        if e.exp.is_Integer and e.exp > 0:
            return f"({pycode(e.base)} ** {int(e.exp)})"
        return f"({pycode(e.base)} ** {pycode(e.exp)})"  # a float expression (scale factors)
    if isinstance(e, sympy.floor):
        num, den = e.args[0].as_numer_denom()
        return f"({pycode(num)} // {pycode(den)})"
    if isinstance(e, sympy.ceiling):
        num, den = e.args[0].as_numer_denom()
        return f"(-((-{pycode(num)}) // {pycode(den)}))"
    if isinstance(e, sympy.Mod):
        return f"({pycode(e.args[0])} % {pycode(e.args[1])})"
    if isinstance(e, sympy.Min):
        return "min(" + ", ".join(pycode(a) for a in e.args) + ")"
    if isinstance(e, sympy.Max):
        return "max(" + ", ".join(pycode(a) for a in e.args) + ")"
    if isinstance(e, sympy.Rel):
        op = {"==": "==", "!=": "!=", "<=": "<=", "<": "<", ">=": ">=", ">": ">"}[e.rel_op]
        return f"{pycode(e.lhs)} {op} {pycode(e.rhs)}"
    if isinstance(e, sympy.And):
        return "(" + " and ".join(pycode(a) for a in e.args) + ")"
    if isinstance(e, sympy.Or):
        return "(" + " or ".join(pycode(a) for a in e.args) + ")"
    if isinstance(e, sympy.Not):
        return f"(not {pycode(e.args[0])})"
    raise ValueError(f"cannot print {type(e).__name__}: {e}")
