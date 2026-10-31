import torch
import torch.nn.functional as F


def build():
    """Return (program, calls, about)."""
    torch.manual_seed(0)
    d_model = 32
    w_qkv = torch.randn(3 * d_model, d_model) * 0.1
    w_out = torch.randn(d_model, d_model) * 0.1

    def program(x, num_heads):
        # causal multi-head self-attention; the number of heads is a call argument
        b, t, d = x.shape
        head_dim = d // num_heads
        q, k, v = F.linear(x, w_qkv).chunk(3, dim=-1)
        q = q.reshape(b, t, num_heads, -1).transpose(1, 2)
        k = k.reshape(b, t, num_heads, -1).transpose(1, 2)
        v = v.reshape(b, -1, num_heads, head_dim).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / head_dim ** 0.5
        future = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, t, d)
        return F.linear(out, w_out)

    shapes = [
        (2, 10, 4), (2, 10, 4), (3, 10, 4), (2, 17, 4), (8, 17, 4),
        (1, 1, 4), (2, 10, 2), (2, 10, 8), (4, 33, 8), (1, 64, 1),
        (2, 10, 4), (5, 3, 16), (2, 12, 32), (1, 2, 4), (6, 24, 2),
        (2, 10, 8), (3, 7, 4),
    ]
    calls = [(torch.randn(b, t, d_model), h) for b, t, h in shapes]
    about = ("Causal multi-head self-attention where num_heads is an int argument that sets "
             "head_dim = d // num_heads, the -1 in the reshapes and the score scale, and the causal "
             "mask is built from the sequence length (including length 1).")
    return program, calls, about
