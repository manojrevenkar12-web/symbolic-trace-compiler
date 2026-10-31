"""Symbols, guards, replacements and printing."""

import sympy

from stc.symbolic import ShapeEnv, Source, SymInt, pycode


def env_with(*hints):
    e = ShapeEnv()
    return e, [e.create(h, Source(0, i)) for i, h in enumerate(hints)]


def test_zero_and_one_are_specialised_and_others_are_symbols():
    e, (a, b, c) = env_with(0, 1, 5)
    assert a == 0 and b == 1 and isinstance(c, SymInt) and c.hint == 5


def test_arithmetic_stays_symbolic_and_simplifies():
    e, (a, b) = env_with(4, 6)
    x = a * b * 12 // 12
    assert isinstance(x, SymInt) and x.hint == 24 and str(e.simplify(x.expr)) == "s0*s1"
    assert (a - a) == 0 and isinstance(a - a, int)


def test_branching_on_a_size_records_the_branch():
    e, (a,) = env_with(10)
    assert bool(a > 8)
    assert sympy.Gt(e.symbols[0], 8) in e.final_guards()


def test_equality_becomes_a_replacement_not_a_guard():
    e, (a, b) = env_with(7, 7)
    e.guard(sympy.Eq(b.expr, a.expr))
    assert e.symbols[1] in e.replacements and e.simplify(b.expr) == e.symbols[0]
    assert all(not isinstance(g, sympy.Eq) for g in e.final_guards())


def test_using_a_size_as_a_python_int_specialises_it():
    e, (a,) = env_with(9)
    assert int(a) == 9 and e.specializations and e.symbols[0] in e.replacements


def test_false_guard_is_rejected():
    e, (a,) = env_with(3)
    try:
        e.guard(sympy.Gt(a.expr, 100))
    except RuntimeError:
        return
    raise AssertionError("expected a failure")


def test_printing_as_python():
    s0, s1 = sympy.symbols("s0 s1", integer=True, positive=True)
    assert eval(pycode(sympy.floor((s0 * s1) / 4)), {"s0": 6, "s1": 3}) == 4
    assert eval(pycode(sympy.Mod(s0, 3)), {"s0": 7}) == 1
    assert eval(pycode(sympy.Min(s0, s1)), {"s0": 2, "s1": 5}) == 2
    assert eval(pycode(sympy.And(sympy.Le(s0, 9), sympy.Ne(s1, 4))), {"s0": 2, "s1": 5})
    assert eval(pycode(s0 - 2 * s1 + 3), {"s0": 10, "s1": 2}) == 9


def test_reflected_modulo_and_floor_division():
    e, (a,) = env_with(6)
    assert (20 % a).hint == 2 and (20 // a).hint == 3
