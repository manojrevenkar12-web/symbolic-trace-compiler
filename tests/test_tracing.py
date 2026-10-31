"""Tracing, shape rules and generated programs against eager PyTorch."""

import pytest
import torch
import torch.nn.functional as F

import stc


def check(fn, calls, policy="symbolic", compiles=None):
    c = stc.jit(fn, policy=policy)
    with torch.no_grad():
        for args in calls:
            torch.testing.assert_close(c(*args), fn(*args))
    if compiles is not None:
        assert c.stats.compiles == compiles, [p.guards for p in c.programs()]
    return c


def r(*s):
    return torch.randn(*s)


def test_broadcasting_unifies_equal_sizes():
    c = check(lambda x, y: x + y, [(r(3, 4), r(3, 4)), (r(5, 6), r(5, 6))], compiles=1)
    assert "a1.shape[0] != s0" in c.last_program().prologue_src


def test_a_size_of_one_broadcasts_and_gets_its_own_program():
    c = check(lambda x, y: x * y, [(r(3, 4), r(3, 4)), (r(3, 4), r(1, 4)), (r(6, 4), r(6, 4))], compiles=2)
    assert c.stats.hits == 1


def test_reshape_with_minus_one_and_divisibility():
    c = check(lambda x: x.reshape(-1, 4).sum(0), [(r(2, 8),), (r(3, 12),), (r(5, 4),)], compiles=1)
    # s0 * s1 elements must split into rows of 4: the only condition, kept as a guard
    assert c.last_program().guards[-1] == "((s0 * s1) % 4) == 0"


def test_reshape_that_needs_a_divisibility_guard():
    fn = lambda x: x.view(-1, 3)  # noqa: E731
    c = check(fn, [(r(2, 6),), (r(4, 9),)], compiles=1)
    assert any("% 3" in g for g in c.last_program().guards)
    with pytest.raises(RuntimeError):
        c(r(2, 5))  # eager fails too: the guard rejects, the retrace raises the eager error


def test_slice_past_the_end_is_guarded():
    fn = lambda x, n: x[:, :n]  # noqa: E731
    c = check(fn, [(r(2, 10), 4), (r(3, 12), 6)], compiles=1)
    assert "s1 >= s2" in c.last_program().guards  # the slice end stays inside the dimension
    x = r(2, 5)
    torch.testing.assert_close(c(x, 9), fn(x, 9))  # n > size clamps in eager: a new program, not the old one
    assert c.stats.compiles == 2


def test_python_branch_on_a_size_gives_one_program_per_branch():
    def fn(x):
        return x.relu() if x.shape[1] > 8 else x.sigmoid()

    c = check(fn, [(r(2, 10),), (r(2, 12),), (r(2, 4),), (r(2, 5),)], compiles=2)
    assert any("> 8" in g or "<= 8" in g for p in c.programs() for g in p.guards)


def test_arange_and_causal_mask_stay_symbolic():
    def fn(x):
        t = x.shape[1]
        mask = torch.ones(t, t).tril() == 0
        return (x @ x.transpose(-2, -1)).masked_fill(mask, float("-inf")).softmax(-1) + torch.arange(t)

    check(fn, [(r(2, 5, 4),), (r(3, 9, 4),), (r(2, 17, 4),)], compiles=1)


def test_attention_cat_split_chunk_and_sdpa():
    def fn(x, w):
        q, k, v = (x @ w).split(8, dim=-1)
        a, b = torch.cat([q, k], dim=1).chunk(2, dim=1)
        y = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None], is_causal=True)
        return y.squeeze(1) + a - b

    w = r(8, 24)
    check(fn, [(r(2, 6, 8), w), (r(3, 10, 8), w), (r(4, 7, 8), w)], compiles=1)


def test_int_arguments_are_symbols():
    fn = lambda x, k: x[:, : x.shape[1] - k + 1] * 2  # noqa: E731
    check(fn, [(r(2, 10), 3), (r(2, 12), 5), (r(3, 9), 2)], compiles=1)


def test_static_policy_compiles_per_shape_and_auto_learns():
    calls = [(r(2, n),) for n in (3, 4, 5, 6, 7)]
    assert check(lambda x: x.exp(), calls, policy="static").stats.compiles == 5
    assert check(lambda x: x.exp(), calls, policy="auto").stats.compiles == 2


def test_dtype_change_misses_the_cache():
    c = stc.jit(lambda x: x * 2)
    c(r(3, 4))
    c(torch.ones(3, 4, dtype=torch.float64))
    assert c.stats.compiles == 2


def test_unsupported_calls_are_named_and_fallback_works():
    fn = lambda x: torch.nonzero(x)  # noqa: E731
    with pytest.raises(stc.Unsupported, match="nonzero"):
        stc.jit(fn)(r(3))
    c = stc.jit(fn, fallback=True)
    torch.testing.assert_close(c(r(3)).shape[1], 1)
    assert c.stats.fallbacks == 1 and c.stats.unsupported


def test_data_dependent_control_flow_is_unsupported():
    def fn(x):
        return x * 2 if x.sum() > 0 else x

    with pytest.raises(stc.Unsupported, match="truth value"):
        stc.jit(fn)(r(3))


def test_closed_over_tensors_and_modules_are_captured():
    lin = torch.nn.Linear(8, 4)
    bias = torch.randn(4)

    def fn(x):
        return lin(x) + bias

    c = check(fn, [(r(3, 8),), (r(5, 8),)], compiles=1)
    assert len(c.last_program().constants) == 3


def test_cse_and_dce():
    def fn(x):
        a = x.exp()
        b = x.exp()  # same call: merged
        x.sin()  # unused: dropped
        return a + b

    c = check(fn, [(r(4),)])
    p = c.last_program()
    assert p.n_nodes_traced == 4 and p.n_nodes == 2


def test_training_through_the_compiled_program():
    torch.manual_seed(0)
    m1 = torch.nn.Sequential(torch.nn.Linear(6, 12), torch.nn.GELU(), torch.nn.Linear(12, 3))
    m2 = __import__("copy").deepcopy(m1)
    c = stc.jit(m2)
    for n in (4, 7, 5):
        x = r(n, 6)
        m1(x).square().mean().backward()
        c(x).square().mean().backward()
    for a, b in zip(m1.parameters(), m2.parameters()):
        torch.testing.assert_close(a.grad, b.grad)
    assert c.stats.compiles == 1


def test_parameters_changing_dtype_are_caught():
    m = torch.nn.Linear(4, 4)
    c = stc.jit(m)
    c(r(2, 4))
    m.double()
    out = c(torch.randn(2, 4, dtype=torch.float64))
    assert out.dtype == torch.float64 and c.stats.compiles == 2


def test_float_scales_from_sizes_stay_symbolic():
    import math

    def fn(q, k):
        d = q.shape[-1]
        return (q @ k.transpose(-2, -1)) * d**-0.5 / math.sqrt(d)

    c = check(fn, [(r(2, 5, 8), r(2, 5, 8)), (r(2, 6, 16), r(2, 6, 16)), (r(3, 4, 32), r(3, 4, 32))], compiles=1)
    assert "float(" in c.last_program().computation_src


def test_methods_on_closed_over_tensors_keep_sizes_symbolic():
    cls, pos = torch.randn(1, 1, 4), torch.randn(1, 40, 4)

    def fn(x):
        b, t, _ = x.shape
        return torch.cat([cls.expand(b, -1, -1), x], dim=1) + pos[:, : t + 1]

    c = check(fn, [(r(2, 5, 4),), (r(3, 9, 4),), (r(4, 20, 4),)], compiles=1)
    assert not c.last_program().specializations


def test_real_tensor_on_the_left_of_an_operator():
    w = torch.randn(4)
    check(lambda x: w * x + 2.0 / (1.0 + x.abs()), [(r(3, 4),), (r(5, 4),)], compiles=1)
