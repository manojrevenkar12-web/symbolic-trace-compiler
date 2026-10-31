import torch
import torch.nn.functional as F


def build():
    """Return (program, calls, about)."""
    torch.manual_seed(0)
    d = 8
    w_proj = torch.randn(6, d) * 0.2

    def program(x, window, stride):
        # sliding-window attention pooling over a 1D sequence (like an audio frame encoder)
        b, t, dim = x.shape
        n = t - window + 1
        frames = torch.stack([x[:, i : i + n] for i in range(window)], dim=2)  # (b, n, window, d)
        frames = frames[:, ::stride]
        weights = torch.softmax(frames.sum(dim=-1) / dim ** 0.5, dim=-1)
        pooled = (frames * weights.unsqueeze(-1)).sum(dim=2)
        return F.linear(pooled, w_proj).tanh()

    specs = [
        (2, 20, 3, 1), (2, 20, 3, 1), (4, 20, 3, 1), (2, 25, 3, 1), (2, 20, 5, 1),
        (1, 5, 5, 1), (3, 9, 1, 1), (2, 4, 4, 2), (2, 20, 3, 2), (2, 21, 3, 2),
        (2, 22, 3, 3), (1, 7, 2, 3), (2, 20, 3, 1), (2, 6, 2, 7), (1, 30, 8, 4),
        (3, 12, 3, 2), (2, 13, 3, 2),
    ]
    calls = [(torch.randn(b, t, d), k, s) for b, t, k, s in specs]
    about = ("Sliding-window pooling where the int window unrolls a Python loop and sets the frame "
             "count t - window + 1, and the int stride subsamples frames to ceil(n / stride) "
             "(including window == t and stride larger than n).")
    return program, calls, about
