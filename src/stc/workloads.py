"""Shape streams: the sequence of calls a workload makes, drawn once from its seed.

Kept free of `stc` internals so the Thunder bridge (CI) replays exactly the same calls.
"""

from __future__ import annotations

import functools
import math
import random
from pathlib import Path
from typing import Any

import torch
import yaml

from .models import MODELS, Config

ROOT = Path(__file__).resolve().parents[2]


@functools.lru_cache
def load(name: str) -> dict[str, Any]:
    return yaml.safe_load((ROOT / "config" / f"{name}.yaml").read_text())


def config() -> Config:
    return Config(**load("workloads")["model"])


def shapes(name: str, scale: float = 1.0) -> list[tuple[int, int]]:
    w = load("workloads")["workloads"][name]
    rng = random.Random(w["seed"])
    n = max(4, int(round(w["calls"] * scale)))
    b0, b1 = w["batch"]
    s = w["seq"]
    out = []
    for i in range(n):
        b = rng.randint(b0, b1)
        if "grow_from" in s:
            t = s["grow_from"] + i
        else:
            t = int(round(math.exp(math.log(s["median"]) + s["sigma"] * rng.gauss(0, 1))))
            t = max(s["min"], min(s["max"], t))
        out.append((b, t))
    return out


def make_model(name: str, seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    w = load("workloads")["workloads"][name]
    m = MODELS[w["model"]](config())
    return m.train() if w.get("train") else m.eval()


def make_inputs(name: str, b: int, t: int, i: int) -> tuple:
    w = load("workloads")["workloads"][name]
    g = torch.Generator().manual_seed(w["seed"] * 100_003 + i)
    vocab = load("workloads")["model"]["vocab"]
    idx = torch.randint(0, vocab, (b, t), generator=g)
    if w["model"] == "encoder":
        lengths = torch.randint(max(1, t // 3), t + 1, (b, 1), generator=g)
        lengths[0, 0] = t  # the batch is padded to its longest text
        mask = (torch.arange(t)[None, :] < lengths).long()
        return (idx, mask)
    if w["model"] == "gpt_train":
        return (idx, torch.randint(0, vocab, (b, t), generator=g))
    return (idx,)


def names() -> list[str]:
    return list(load("workloads")["workloads"])


def is_train(name: str) -> bool:
    return bool(load("workloads")["workloads"][name].get("train"))
