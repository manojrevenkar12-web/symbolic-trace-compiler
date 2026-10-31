"""The workload models compile once for every shape and agree with eager; the fuzzer finds no
violation and catches a planted one."""

import random

import torch

import stc
from stc.models import GPT, Config, Encoder, GPTLoss, Llama
from stc.verify import fuzz, guard_blocks, necessity


def ids(b, t):
    return torch.randint(0, 512, (b, t))


def test_models_compile_once():
    for M, extra in [(GPT, None), (Llama, None), (Encoder, "mask"), (GPTLoss, "targets")]:
        m = M(Config()).eval()
        c = stc.jit(m)
        with torch.no_grad():
            for b, t in [(2, 7), (3, 19), (5, 40), (2, 8)]:
                a = (ids(b, t),) + ((torch.ones(b, t, dtype=torch.long),) if extra == "mask" else (ids(b, t),) if extra else ())
                torch.testing.assert_close(c(*a), m(*a), rtol=1e-4, atol=1e-5)
        assert c.stats.compiles == 1, (M.__name__, [p.guards for p in c.programs()])


def test_llama_guard_on_the_rope_table():
    c = stc.jit(Llama(Config()).eval())
    with torch.no_grad():
        c(ids(2, 10))
    assert "s1 <= 256" in c.last_program().guards


def test_training_loss_relates_targets_by_element_count():
    c = stc.jit(GPTLoss(Config()))
    c(ids(2, 10), ids(2, 10))
    assert "(s0 * s1) == (s2 * s3)" in c.last_program().guards


def test_fuzzer_finds_no_violation():
    m = GPT(Config()).eval()
    c = stc.jit(m)
    with torch.no_grad():
        c(ids(3, 11))

    def gen(rng: random.Random, sizes):
        return (torch.randint(0, 512, tuple(sizes[0]), generator=torch.Generator().manual_seed(rng.randrange(1 << 30))),)

    rep = fuzz(c, (ids(3, 11),), n=40, cap=300, input_fn=gen)
    assert rep.violations == 0 and rep.coverage > 0.6


def test_fuzzer_catches_a_missing_guard():
    def fn(x):
        return x.relu() if x.shape[1] > 8 else x.sigmoid()

    c = stc.jit(fn)
    c(torch.randn(2, 12))
    res = necessity(c, (torch.randn(2, 12),), n=60, cap=40)
    assert any(r["without_it"] == "wrong results" and "8" in r["check"] for r in res), res
    assert guard_blocks(c.last_program().prologue_src)
