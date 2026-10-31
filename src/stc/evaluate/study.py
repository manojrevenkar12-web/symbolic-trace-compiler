"""Replay every workload under every policy; check each call against eager PyTorch.

Inference: the compiled module and the eager module are the same object, so every call is
compared with eager on identical weights. Training: two copies with the same initial weights,
one stepped through the compiled forward and one eager, each with its own AdamW; losses are
compared per step and the weights at the end.
"""

from __future__ import annotations

import copy
import random
import time
from typing import Any

import torch

from .. import jit
from ..verify import fuzz, necessity
from ..workloads import is_train, load, make_inputs, make_model, names, shapes
from .common import log, prof


def _maxdiff(a: Any, b: Any) -> float:
    if isinstance(a, torch.Tensor):
        return float((a.detach().float() - b.detach().float()).abs().max()) if a.numel() else 0.0
    return max(_maxdiff(x, y) for x, y in zip(a, b))


def run_workload(name: str, policy: str, scale: float) -> dict[str, Any]:
    calls = shapes(name, scale)
    torch.manual_seed(0)
    if is_train(name):
        return _run_train(name, policy, calls)
    model = make_model(name)
    c = jit(model, policy=policy)
    eager_ns = comp_ns = 0
    worst = 0.0
    with torch.no_grad():
        for i, (b, t) in enumerate(calls):
            a = make_inputs(name, b, t, i)
            t0 = time.perf_counter_ns()
            ref = model(*a)
            t1 = time.perf_counter_ns()
            out = c(*a)
            t2 = time.perf_counter_ns()
            eager_ns += t1 - t0
            comp_ns += t2 - t1
            worst = max(worst, _maxdiff(out, ref))
    return _summary(name, policy, calls, c, eager_ns, comp_ns, worst)


def _run_train(name: str, policy: str, calls) -> dict[str, Any]:
    base = make_model(name)
    m_eager, m_comp = copy.deepcopy(base), copy.deepcopy(base)
    opt_e = torch.optim.AdamW(m_eager.parameters(), lr=1e-3)
    opt_c = torch.optim.AdamW(m_comp.parameters(), lr=1e-3)
    c = jit(m_comp, policy=policy)
    eager_ns = comp_ns = 0
    worst = 0.0
    for i, (b, t) in enumerate(calls):
        a = make_inputs(name, b, t, i)
        t0 = time.perf_counter_ns()
        le = m_eager(*a)
        le.backward()
        opt_e.step()
        opt_e.zero_grad()
        t1 = time.perf_counter_ns()
        lc = c(*a)
        lc.backward()
        opt_c.step()
        opt_c.zero_grad()
        t2 = time.perf_counter_ns()
        eager_ns += t1 - t0
        comp_ns += t2 - t1
        worst = max(worst, abs(float(le) - float(lc)))
    for pe, pc in zip(m_eager.parameters(), m_comp.parameters()):
        worst = max(worst, float((pe - pc).abs().max()))
    return _summary(name, policy, calls, c, eager_ns, comp_ns, worst)


def _summary(name, policy, calls, c, eager_ns, comp_ns, worst) -> dict[str, Any]:
    st = c.stats
    progs = c.programs()
    return {
        "workload": name,
        "policy": policy,
        "calls": len(calls),
        "distinct_shapes": len(set(calls)),
        "compiles": st.compiles,
        "hits": st.hits,
        "compile_ms": st.compile_ms,
        "prologue_us_per_call": st.prologue_ns / 1e3 / max(1, st.calls),
        "prologues_per_call": st.prologues_run / max(1, st.calls),
        "eager_ms": eager_ns / 1e6,
        "compiled_ms": comp_ns / 1e6,
        "max_abs_diff": worst,
        "programs": [
            {
                "symbols": p.symbols,
                "guards": p.guards,
                "nodes": p.n_nodes,
                "nodes_traced": p.n_nodes_traced,
                "specializations": p.specializations,
            }
            for p in progs[:6]
        ],
        "_compiled": c,
    }


def _fuzz_inputs(name: str):
    w = load("workloads")["workloads"][name]
    vocab = load("workloads")["model"]["vocab"]

    def f(rng: random.Random, sizes):
        g = torch.Generator().manual_seed(rng.randrange(1 << 30))
        b, t = sizes[0]
        idx = torch.randint(0, vocab, (b, t), generator=g)
        if w["model"] == "encoder":
            return (idx, (torch.rand(*sizes[1], generator=g) > 0.2).long())
        if w["model"] == "gpt_train":
            return (idx, torch.randint(0, vocab, tuple(sizes[1]), generator=g))
        return (idx,)

    return f


def run_study(profile: str) -> dict[str, Any]:
    p = prof(profile)
    exp = load("experiment")
    rows, fuzzing, necessary = [], [], {}
    for name in names():
        for policy in exp["policies"]:
            t0 = time.time()
            r = run_workload(name, policy, p["calls_scale"])
            c = r.pop("_compiled")
            log(
                f"{name} / {policy}: {r['compiles']} compiles for {r['calls']} calls ({r['distinct_shapes']} distinct shapes), "
                f"compile {r['compile_ms']:.0f} ms, max diff {r['max_abs_diff']:.1e} ({time.time() - t0:.0f}s)"
            )
            rows.append(r)
            if is_train(name):
                continue
            ex = make_inputs(name, *shapes(name, p["calls_scale"])[0], 0)
            c.module.eval()
            rep = fuzz(c, ex, n=p["fuzz_draws"], seed=7, cap=exp["fuzz_cap"], input_fn=_fuzz_inputs(name))
            fuzzing.append({"workload": name, "policy": policy, **rep.as_dict()})
            log(f"  fuzz: {rep.violations} violations in {rep.draws} draws, coverage {rep.coverage:.0%}")
            if policy == "symbolic" and name in ("chat_prefill", "llama_prefill", "padded_encoder"):
                necessary[name] = necessity(c, ex, n=p["necessity_draws"], seed=3, cap=exp["fuzz_cap"], input_fn=_fuzz_inputs(name))
    return {"rows": rows, "fuzz": fuzzing, "necessity": necessary}
