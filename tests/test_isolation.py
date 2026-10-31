"""Lookasides are installed only while tracing, and compiled programs never see them."""

import threading

import torch
import torch.nn.functional as F

import stc
from stc import ops


def test_torch_is_restored_after_a_trace_even_on_error():
    before = (torch.matmul, F.linear, torch.arange)
    try:
        stc.jit(lambda x: torch.nonzero(x))(torch.randn(3))
    except stc.Unsupported:
        pass
    assert (torch.matmul, F.linear, torch.arange) == before
    assert not getattr(ops._LOCAL, "tracing", False)


def test_other_threads_run_real_torch_during_a_trace():
    seen = []

    def slow(x):
        t = threading.Thread(target=lambda: seen.append(torch.arange(3).sum().item()))
        t.start()
        t.join()
        return x + 1

    stc.jit(slow)(torch.randn(2))
    assert seen == [3]
