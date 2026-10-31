import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """GPT-style causal self-attention where the number of heads is chosen per call."""

    def __init__(self, dim=32):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, n_heads):
        b, t, d = x.shape
        hd = d // n_heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(b, t, n_heads, hd).transpose(1, 2)
        k = k.reshape(b, t, n_heads, hd).transpose(1, 2)
        v = v.reshape(b, t, n_heads, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf")).softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(b, t, d)
        return self.out(y)


def build():
    """Return (program, calls, about)."""
    torch.manual_seed(0)
    program = CausalSelfAttention().eval()
    plan = [
        (2, 10, 4), (2, 10, 4), (3, 7, 4), (1, 1, 4), (2, 10, 2), (2, 10, 8),
        (4, 16, 1), (1, 5, 8), (2, 3, 4), (2, 33, 4), (1, 1, 1), (5, 12, 2),
        (2, 10, 4), (2, 64, 8),
    ]
    calls = [(torch.randn(b, t, 32), h) for b, t, h in plan]
    about = (
        "Causal multi-head self-attention whose head count is an int argument: it sets the reshape "
        "and the 1/sqrt(head_dim) scale, and sequence length 1 makes the causal mask 1x1."
    )
    return program, calls, about
