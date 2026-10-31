import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalWindowAttention(nn.Module):
    """Block-local self-attention: pad the sequence to a multiple of the window, attend within each
    window with padded keys masked out, then drop the padding again."""

    def __init__(self, dim=24):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, window):
        b, t, d = x.shape
        pad = (window - t % window) % window
        xp = torch.cat([x, torch.zeros(b, pad, d)], dim=1)
        nw = (t + pad) // window
        xw = xp.reshape(b, nw, window, d)
        key_ok = (torch.arange(t + pad) < t).reshape(1, nw, 1, window)
        y = F.scaled_dot_product_attention(self.q(xw), self.k(xw), self.v(xw), attn_mask=key_ok)
        return self.out(y.reshape(b, nw * window, d)[:, :t])


def build():
    """Return (program, calls, about)."""
    torch.manual_seed(0)
    program = LocalWindowAttention().eval()
    plan = [
        (2, 32, 8), (2, 32, 8), (2, 33, 8), (2, 31, 8), (4, 40, 8), (1, 7, 8),
        (1, 8, 8), (1, 9, 8), (2, 32, 16), (2, 30, 16), (3, 1, 4), (2, 12, 1),
        (2, 50, 7), (1, 64, 64), (2, 32, 8),
    ]
    calls = [(torch.randn(b, t, 24), w) for b, t, w in plan]
    about = (
        "Windowed attention that pads T up to a multiple of the int window (pad = (w - T % w) % w, zero when T "
        "divides), masks the padded keys, and slices the padding off with [:, :T]."
    )
    return program, calls, about
