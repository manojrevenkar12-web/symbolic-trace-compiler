import torch
import torch.nn.functional as F


def build():
    """Return (program, calls, about)."""
    torch.manual_seed(0)
    max_len, d = 16, 8
    w = torch.randn(4, d) * 0.3

    def program(x, lengths):
        # truncate or right-pad a padded batch to max_len, then masked mean pooling
        b, t, dim = x.shape
        x = x[:, :max_len]
        short = max_len - x.shape[1]
        if short > 0:
            x = torch.cat([x, torch.zeros(b, short, dim, dtype=x.dtype)], dim=1)
        positions = torch.arange(max_len)
        valid = positions[None, :] < lengths.clamp(max=t)[:, None]
        x = x.masked_fill(~valid.unsqueeze(-1), 0.0)
        count = valid.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = x.sum(dim=1) / count
        return F.linear(pooled, w)

    specs = [
        (2, 16), (2, 16), (3, 16), (2, 10), (4, 10), (2, 20), (1, 1), (2, 15),
        (2, 17), (5, 32), (1, 16), (3, 3), (2, 40), (1, 2), (2, 16), (6, 12),
    ]
    calls = []
    for b, t in specs:
        x = torch.randn(b, t, d)
        lengths = torch.randint(0, t + 1, (b,))
        calls.append((x, lengths))
    about = ("Masked mean pooling after x[:, :16] (an end that runs past short sequences) and a "
             "Python branch that pads only when the sequence is shorter than 16, with valid positions "
             "taken from per-row lengths.")
    return program, calls, about
