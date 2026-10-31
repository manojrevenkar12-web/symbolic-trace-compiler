import torch
import torch.nn.functional as F


def build():
    """Return (program, calls, about)."""
    torch.manual_seed(0)

    def program(q, k_cache, v_cache, k_new, v_new, window):
        # sliding-window attention over a KV cache (prefill or decode)
        tq = q.shape[2]
        k = torch.cat([k_cache, k_new], dim=2)
        v = torch.cat([v_cache, v_new], dim=2)
        keep = window + tq - 1  # keys any new query can still see
        k = k[:, :, -keep:]
        v = v[:, :, -keep:]
        length = k.shape[2]
        q_pos = torch.arange(tq) + (length - tq)
        k_pos = torch.arange(length)
        visible = (k_pos[None, :] <= q_pos[:, None]) & (k_pos[None, :] > q_pos[:, None] - window)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=visible)

    specs = [  # (batch, heads, cached, new, head_dim, window)
        (2, 4, 0, 6, 8, 4), (2, 4, 6, 1, 8, 4), (2, 4, 7, 1, 8, 4), (2, 4, 8, 1, 8, 4),
        (2, 4, 0, 6, 8, 4), (3, 4, 0, 6, 8, 4), (2, 4, 0, 3, 8, 8), (2, 4, 3, 1, 8, 8),
        (2, 4, 20, 1, 8, 8), (2, 4, 20, 4, 8, 8), (1, 2, 0, 1, 8, 1), (1, 2, 5, 1, 8, 1),
        (2, 4, 2, 2, 8, 16), (1, 4, 30, 1, 16, 32), (2, 4, 9, 1, 8, 4), (4, 4, 4, 4, 8, 2),
    ]
    calls = []
    for b, h, s, n, dh, win in specs:
        calls.append((torch.randn(b, h, n, dh), torch.randn(b, h, s, dh), torch.randn(b, h, s, dh),
                      torch.randn(b, h, n, dh), torch.randn(b, h, n, dh), win))
    about = ("Sliding-window KV-cache attention that keeps the last window + tq - 1 keys with a "
             "negative slice that may reach past the start (empty or short caches), and builds the "
             "mask from positions offset by length - tq.")
    return program, calls, about
