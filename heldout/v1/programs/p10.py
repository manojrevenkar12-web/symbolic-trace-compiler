import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyClassifier(nn.Module):
    """One pre-norm transformer block over token ids, then a pooled classification head."""

    def __init__(self, vocab=100, dim=32, heads=4, max_pos=64, classes=5):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_pos, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp_in = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.mlp_out = nn.Linear(4 * dim, dim)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(dim, classes)

    def forward(self, ids):
        b, t = ids.shape
        x = self.tok(ids) + self.pos(torch.arange(t))
        q, k, v = self.qkv(self.norm1(x)).split(self.dim, dim=-1)
        q, k, v = (z.reshape(b, t, self.heads, -1).transpose(1, 2) for z in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(b, t, -1)))
        x = x + self.drop(self.mlp_out(self.act(self.mlp_in(self.norm2(x)))))
        if t > 32:
            x = x[:, ::2]  # long inputs: pool every other token
        return self.head(x.mean(dim=1))


def build():
    """Return (program, calls, about)."""
    torch.manual_seed(0)
    program = TinyClassifier().eval()
    specs = [
        (2, 16), (2, 16), (4, 16), (2, 20), (1, 1), (2, 32), (2, 33), (2, 34),
        (3, 48), (1, 64), (2, 31), (5, 8), (2, 63), (1, 2), (2, 16), (8, 40),
    ]
    calls = [(torch.randint(0, 100, (b, t)),) for b, t in specs]
    about = ("Embedding plus causal transformer block in eval mode whose positions come from "
             "arange(t), with a Python branch on t > 32 that mean-pools only every other token "
             "(ceil(t / 2) of them, 33 vs 32 is the edge).")
    return program, calls, about
