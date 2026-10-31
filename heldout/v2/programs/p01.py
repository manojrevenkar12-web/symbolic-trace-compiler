import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """ViT-style patch embedding with 2D sin-cos positions; large images are 2x2 average-pooled first."""

    def __init__(self, patch=4, chans=3, dim=32):
        super().__init__()
        self.patch = patch
        self.proj = nn.Linear(chans * patch * patch, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, img):
        b, c, h, w = img.shape
        if h * w > 1024:
            h2, w2 = h // 2, w // 2
            img = img[:, :, : h2 * 2, : w2 * 2].reshape(b, c, h2, 2, w2, 2).mean(dim=(3, 5))
            h, w = h2, w2
        p = self.patch
        gh, gw = h // p, w // p
        x = img[:, :, : gh * p, : gw * p]
        x = x.reshape(b, c, gh, p, gw, p).permute(0, 2, 4, 1, 3, 5).reshape(b, gh * gw, c * p * p)
        tok = self.norm(self.proj(x))
        ys = torch.arange(gh).float()[:, None].expand(gh, gw).reshape(-1, 1)
        xs = torch.arange(gw).float()[None, :].expand(gh, gw).reshape(-1, 1)
        freq = torch.exp(torch.arange(8).float() * -0.5)[None, :]
        pos = torch.cat([torch.sin(ys * freq), torch.cos(ys * freq), torch.sin(xs * freq), torch.cos(xs * freq)], dim=-1)
        return tok + pos


def build():
    """Return (program, calls, about)."""
    torch.manual_seed(0)
    program = PatchEmbed().eval()
    sizes = [
        (2, 32, 32), (1, 32, 32), (4, 32, 33), (2, 17, 23), (2, 64, 48), (1, 8, 8),
        (3, 31, 33), (2, 4, 4), (2, 45, 30), (1, 7, 9), (2, 5, 128), (1, 100, 10),
        (1, 103, 10), (2, 32, 32), (4, 16, 64), (1, 33, 32),
    ]
    calls = [(torch.randn(b, 3, h, w),) for b, h, w in sizes]
    about = (
        "ViT patch embedding with 2D sin-cos positions; images over 1024 pixels are 2x2-pooled first "
        "(Python branch on h*w, boundary at 32x32 vs 33x32), and odd sizes are floor-cropped to the patch grid."
    )
    return program, calls, about
