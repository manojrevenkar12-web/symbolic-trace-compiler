"""Workload models, written in plain PyTorch the way public reference implementations write them.

- `GPT`: nanoGPT-style decoder (learned positions, fused qkv split, causal SDPA, GELU MLP).
- `Llama`: RMSNorm, rotary embeddings from a cached cos/sin table sliced to the sequence length,
  grouped-query attention by expand + reshape, SwiGLU.
- `Encoder`: BERT-style encoder with a padding mask broadcast to the attention scores and
  explicit softmax attention (no SDPA), mean pooling over unmasked tokens.

Small sizes so the study runs on a CPU; the shape logic is the same as at full size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Config:
    vocab: int = 512
    block: int = 256
    n_layer: int = 2
    n_head: int = 4
    n_kv_head: int = 2
    n_embd: int = 64
    dropout: float = 0.0


# ---------------------------------------------------------------------- GPT
class CausalSelfAttention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c_attn = nn.Linear(c.n_embd, 3 * c.n_embd)
        self.c_proj = nn.Linear(c.n_embd, c.n_embd)
        self.n_head, self.n_embd = c.n_head, c.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.fc = nn.Linear(c.n_embd, 4 * c.n_embd)
        self.proj = nn.Linear(4 * c.n_embd, c.n_embd)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.ln_1, self.attn = nn.LayerNorm(c.n_embd), CausalSelfAttention(c)
        self.ln_2, self.mlp = nn.LayerNorm(c.n_embd), MLP(c)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, c: Config | None = None):
        super().__init__()
        c = c or Config()
        self.c = c
        self.wte, self.wpe = nn.Embedding(c.vocab, c.n_embd), nn.Embedding(c.block, c.n_embd)
        self.h = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.lm_head = nn.Linear(c.n_embd, c.vocab, bias=False)

    def forward(self, idx):
        b, t = idx.size()
        assert t <= self.c.block, f"sequence of {t} tokens is longer than the block size {self.c.block}"
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        return self.lm_head(self.ln_f(x))


class GPTLoss(nn.Module):
    """A training step's forward: logits and the next-token loss."""

    def __init__(self, c: Config | None = None):
        super().__init__()
        self.gpt = GPT(c)

    def forward(self, idx, targets):
        logits = self.gpt(idx)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


# ---------------------------------------------------------------------- Llama
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Rotary(nn.Module):
    def __init__(self, dim: int, max_len: int, base: float = 10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, q, k):
        T = q.shape[-2]
        cos, sin = self.cos[:T], self.sin[:T]
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class LlamaAttention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.h, self.kv, self.d = c.n_head, c.n_kv_head, c.n_embd // c.n_head
        self.wq = nn.Linear(c.n_embd, self.h * self.d, bias=False)
        self.wk = nn.Linear(c.n_embd, self.kv * self.d, bias=False)
        self.wv = nn.Linear(c.n_embd, self.kv * self.d, bias=False)
        self.wo = nn.Linear(self.h * self.d, c.n_embd, bias=False)
        self.rope = Rotary(self.d, c.block)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h, self.d).transpose(1, 2)
        k = self.wk(x).view(B, T, self.kv, self.d).transpose(1, 2)
        v = self.wv(x).view(B, T, self.kv, self.d).transpose(1, 2)
        q, k = self.rope(q, k)
        rep = self.h // self.kv
        k = k[:, :, None, :, :].expand(B, self.kv, rep, T, self.d).reshape(B, self.h, T, self.d)
        v = v[:, :, None, :, :].expand(B, self.kv, rep, T, self.d).reshape(B, self.h, T, self.d)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(y.transpose(1, 2).reshape(B, T, self.h * self.d))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class LlamaBlock(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.n1, self.attn = RMSNorm(c.n_embd), LlamaAttention(c)
        self.n2 = RMSNorm(c.n_embd)
        hidden = int(8 * c.n_embd / 3)
        self.w1 = nn.Linear(c.n_embd, hidden, bias=False)
        self.w3 = nn.Linear(c.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, c.n_embd, bias=False)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        h = self.n2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class Llama(nn.Module):
    def __init__(self, c: Config | None = None):
        super().__init__()
        c = c or Config()
        self.tok = nn.Embedding(c.vocab, c.n_embd)
        self.layers = nn.ModuleList([LlamaBlock(c) for _ in range(c.n_layer)])
        self.norm = RMSNorm(c.n_embd)
        self.out = nn.Linear(c.n_embd, c.vocab, bias=False)

    def forward(self, idx):
        x = self.tok(idx)
        for layer in self.layers:
            x = layer(x)
        return self.out(self.norm(x))


# ---------------------------------------------------------------------- encoder with padding
class EncoderLayer(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.h, self.d = c.n_head, c.n_embd // c.n_head
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd)
        self.o = nn.Linear(c.n_embd, c.n_embd)
        self.ln1, self.ln2 = nn.LayerNorm(c.n_embd), nn.LayerNorm(c.n_embd)
        self.ff1, self.ff2 = nn.Linear(c.n_embd, 2 * c.n_embd), nn.Linear(2 * c.n_embd, c.n_embd)

    def forward(self, x, mask):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(B, T, self.h, self.d).permute(0, 2, 1, 3)
        k = k.reshape(B, T, self.h, self.d).permute(0, 2, 1, 3)
        v = v.reshape(B, T, self.h, self.d).permute(0, 2, 1, 3)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
        scores = scores.masked_fill(mask[:, None, None, :] == 0, float("-inf"))
        a = scores.softmax(dim=-1) @ v
        x = self.ln1(x + self.o(a.permute(0, 2, 1, 3).reshape(B, T, C)))
        return self.ln2(x + self.ff2(F.relu(self.ff1(x))))


class Encoder(nn.Module):
    def __init__(self, c: Config | None = None):
        super().__init__()
        c = c or Config()
        self.emb = nn.Embedding(c.vocab, c.n_embd)
        self.pos = nn.Embedding(c.block, c.n_embd)
        self.layers = nn.ModuleList([EncoderLayer(c) for _ in range(c.n_layer)])

    def forward(self, idx, mask):
        T = idx.shape[1]
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))[None]
        for layer in self.layers:
            x = layer(x, mask)
        m = mask.unsqueeze(-1).to(x.dtype)
        return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


MODELS = {"gpt": GPT, "gpt_train": GPTLoss, "llama": Llama, "encoder": Encoder}
