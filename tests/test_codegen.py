"""The generated program is readable Python, and the prologue is what decides reuse."""

import torch

import stc


def test_program_is_python_with_shape_comments():
    c = stc.jit(lambda x, w: (x @ w).relu())
    c(torch.randn(2, 5, 8), torch.randn(8, 3))
    p = c.last_program()
    compile(p.source, "x", "exec")
    assert "def prologue(a0, a1):" in p.prologue_src and "def computation(a0, a1, s0, s1, s2, s4)" in p.computation_src
    assert "# float32[s0, s1, s4]" in p.computation_src


def test_prologue_rejects_rank_dtype_device_and_relations():
    c = stc.jit(lambda x, y: x + y)
    c(torch.randn(3, 4), torch.randn(3, 4))
    pro = c.last_program().prologue
    assert pro(torch.randn(5, 6), torch.randn(5, 6)) == (5, 6)
    assert pro(torch.randn(5, 6, 1), torch.randn(5, 6)) is None
    assert pro(torch.randn(5, 6).double(), torch.randn(5, 6)) is None
    assert pro(torch.randn(5, 6), torch.randn(5, 7)) is None
    assert pro(torch.randn(1, 6), torch.randn(1, 6)) is None  # size 1 has its own program
    assert pro("x", torch.randn(5, 6)) is None


def test_most_recently_used_program_is_tried_first():
    c = stc.jit(lambda x: x * 2, policy="static")
    for n in (2, 3, 4, 2):
        c(torch.randn(n))
    assert c.stats.compiles == 3 and c.entries[0].shapes == [(2,)]
