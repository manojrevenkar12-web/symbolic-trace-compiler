import torch
import torch.nn.functional as F


def rope_gqa_prepare(q, k, offset):
    """Rotary position embedding at positions offset..offset+T-1 (KV-cache decoding), then
    the pre-scaled queries and the keys repeated for grouped-query attention, as written to the cache.

    q (B, Hq, T, D), k (B, Hk, T, D) with Hq a multiple of Hk; returns (2, B, Hq, T, D).
    """
    hq, t, d = q.shape[1], q.shape[-2], q.shape[-1]
    hk = k.shape[1]
    half = d // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(half).float() / half))
    pos = torch.arange(offset, offset + t).float()
    ang = torch.outer(pos, inv_freq)
    cos, sin = ang.cos(), ang.sin()

    def rotate(x):
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    qr, kr = rotate(q), rotate(k)
    rep = hq // hk
    b = k.shape[0]
    kr = kr[:, :, None].expand(b, hk, rep, t, d).reshape(b, hk * rep, t, d)
    return torch.stack([qr * (d ** -0.5), kr], dim=0)


def build():
    """Return (program, calls, about)."""
    torch.manual_seed(0)
    program = rope_gqa_prepare
    plan = [
        (2, 8, 2, 16, 16, 0), (2, 8, 2, 16, 16, 0), (2, 8, 2, 1, 16, 16), (2, 8, 2, 1, 16, 17),
        (2, 8, 2, 1, 16, 100), (1, 8, 8, 12, 16, 0), (1, 8, 1, 12, 16, 0), (3, 4, 2, 7, 32, 5),
        (3, 4, 4, 7, 32, 5), (1, 2, 1, 1, 8, 0), (2, 8, 2, 16, 8, 0), (4, 6, 3, 9, 16, 3),
        (2, 8, 2, 16, 16, 2048), (1, 1, 1, 5, 64, 1),
    ]
    calls = []
    for b, hq, hk, t, d, off in plan:
        calls.append((torch.randn(b, hq, t, d), torch.randn(b, hk, t, d), off))
    about = (
        "RoPE for KV-cache decoding with grouped-query keys: the int offset changes only values (absolute positions, "
        "so it must not be baked in), the rotation splits the head dim at d//2, q is scaled by d**-0.5, and keys are "
        "repeated Hq//Hk times (Hk sometimes 1 or equal to Hq)."
    )
    return program, calls, about
