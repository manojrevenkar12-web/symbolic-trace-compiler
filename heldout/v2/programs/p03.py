import torch
import torch.nn.functional as F


def smooth(x, k):
    """Moving average of a batch of series over a window of k, plus the residual against it.

    Series shorter than the window fall back to their overall mean.
    """
    b, t = x.shape
    if t < k:
        return x.mean(dim=1, keepdim=True)
    c = torch.cat([torch.zeros(b, 1, dtype=x.dtype), x.cumsum(dim=1)], dim=1)
    avg = (c[:, k:] - c[:, :-k]) / k
    resid = x[:, k - 1 :] - avg
    return torch.stack([avg, resid], dim=-1)


def build():
    """Return (program, calls, about)."""
    torch.manual_seed(0)
    program = smooth
    plan = [
        (4, 50, 5), (4, 50, 5), (2, 50, 7), (4, 20, 5), (1, 5, 5), (3, 4, 5),
        (4, 50, 1), (2, 6, 5), (8, 100, 12), (2, 11, 12), (2, 12, 12), (1, 1, 1),
        (1, 1, 3), (4, 50, 5), (6, 30, 3),
    ]
    calls = [(torch.randn(b, t), k) for b, t, k in plan]
    about = (
        "Cumsum-based moving average with window k: output length t-k+1 comes from c[:, k:] - c[:, :-k], "
        "and a Python branch returns a (B, 1) mean instead when the series is shorter than the window (t == k vs t == k-1)."
    )
    return program, calls, about
