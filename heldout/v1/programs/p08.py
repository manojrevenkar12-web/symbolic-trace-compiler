import torch
import torch.nn.functional as F


def build():
    """Return (program, calls, about)."""
    torch.manual_seed(0)
    c = 16
    ln_w = torch.rand(c) + 0.5
    ln_b = torch.randn(c) * 0.1
    w_head = torch.randn(10, c) * 0.3

    def program(feat):
        # channels-first feature map -> tokens for LayerNorm -> back, then pool and classify
        b, ch, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # (b, h*w, c)
        tokens = F.layer_norm(tokens, (ch,), ln_w, ln_b)
        x = tokens.transpose(1, 2).reshape(b, ch, h, w)
        if h * w > 64:
            h2, w2 = h // 2, w // 2
            x = x[:, :, : 2 * h2, : 2 * w2]
            x = x.reshape(b, ch, h2, 2, w2, 2).amax(dim=(3, 5))  # 2x2 max-pool, odd edge dropped
        out = x.mean(dim=(2, 3))
        return F.linear(F.gelu(out), w_head)

    sizes = [
        (2, 8, 8), (2, 8, 8), (4, 8, 8), (2, 8, 9), (2, 9, 8), (2, 7, 7),
        (1, 1, 64), (2, 2, 40), (2, 16, 16), (1, 11, 7), (3, 4, 16), (3, 5, 13),
        (2, 32, 32), (1, 64, 1), (2, 3, 3), (2, 13, 5),
    ]
    calls = [(torch.randn(b, c, h, w),) for b, h, w in sizes]
    about = ("Layout round trip (NCHW -> tokens -> NCHW) followed by a Python branch on h * w > 64 "
             "that crops odd edges and max-pools 2x2 before the global mean, so the decision depends "
             "on the product of two sizes (8x8 = 64 does not pool, 8x9 does).")
    return program, calls, about
