import torch
import torch.nn.functional as F


def build():
    """Return (program, calls, about)."""
    torch.manual_seed(0)
    patch, channels, dim = 4, 3, 24
    weight = torch.randn(dim, channels * patch * patch) * 0.1
    bias = torch.randn(dim) * 0.1
    cls = torch.randn(1, 1, dim)
    pos = torch.randn(1, 65, dim) * 0.02  # cls + up to an 8x8 grid of patches

    def program(images):
        # ViT-style patch embedding: crop to whole patches, patchify, project, add positions
        b, c, h, w = images.shape
        gh, gw = h // patch, w // patch
        x = images[:, :, : gh * patch, : gw * patch]
        x = x.reshape(b, c, gh, patch, gw, patch)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(b, gh * gw, -1)
        tokens = F.linear(x, weight, bias)
        tokens = torch.cat([cls.expand(b, -1, -1), tokens], dim=1)
        tokens = tokens + pos[:, : gh * gw + 1]
        tokens = F.layer_norm(tokens, (dim,))
        return torch.cat([tokens[:, 0], tokens[:, 1:].mean(dim=1)], dim=-1)

    sizes = [
        (2, 32, 32), (2, 32, 32), (4, 32, 32), (2, 16, 16), (2, 17, 19),
        (1, 16, 16), (3, 8, 12), (2, 4, 4), (2, 7, 7), (1, 32, 30),
        (1, 30, 32), (2, 35, 33), (5, 12, 8), (2, 18, 18), (2, 19, 16),
        (1, 4, 31), (3, 31, 4),
    ]
    calls = [(torch.randn(b, channels, h, w),) for b, h, w in sizes]
    about = ("ViT patch embedding that crops images to h // 4 * 4 by w // 4 * 4, so odd sizes share "
             "a grid with smaller ones, and slices the position table to gh * gw + 1 rows computed "
             "from both image sizes.")
    return program, calls, about
