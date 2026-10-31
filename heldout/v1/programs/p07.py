import math

import torch
import torch.nn.functional as F


def build():
    """Return (program, calls, about)."""
    torch.manual_seed(0)
    neg_log_base = -math.log(10000.0)

    def program(x, offset):
        # partial rotary position embedding (GPT-NeoX style) for positions offset .. offset + t - 1
        b, h, t, dh = x.shape
        half = dh // 4
        rot = 2 * half
        inv_freq = torch.exp(torch.arange(half).float() * (neg_log_base * 2 / rot))
        pos = torch.arange(offset, offset + t).float()
        angles = torch.outer(pos, inv_freq)  # (t, half)
        cos, sin = angles.cos(), angles.sin()
        x1, x2, rest = x[..., :half], x[..., half:rot], x[..., rot:]
        rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return torch.cat([rotated, rest], dim=-1)

    specs = [  # (batch, heads, length, head_dim, offset)
        (2, 4, 16, 16, 0), (2, 4, 16, 16, 0), (2, 4, 1, 16, 16), (2, 4, 1, 16, 17),
        (2, 4, 1, 16, 18), (3, 4, 16, 16, 0), (2, 4, 9, 16, 0), (2, 4, 4, 16, 9),
        (1, 2, 16, 8, 0), (1, 2, 16, 12, 0), (1, 2, 16, 6, 0), (2, 4, 16, 18, 0),
        (2, 4, 1, 16, 500), (1, 1, 32, 16, 3), (2, 4, 16, 16, 0), (4, 8, 1, 16, 63),
    ]
    calls = [(torch.randn(b, h, t, dh), off) for b, h, t, dh, off in specs]
    about = ("Partial rotary embedding that rotates the first 2 * (head_dim // 4) channels and passes "
             "the rest through (head_dim 6, 12, 18 leave odd remainders), with an int offset that "
             "sets the positions for KV-cache decoding at length 1.")
    return program, calls, about
