import torch
import torch.nn as nn
import torch.nn.functional as F


class BagOfEmbeddingsClassifier(nn.Module):
    """Text classifier: truncate to max_len, embed, masked mean + masked max pooling, linear head.

    Token id 0 is padding.
    """

    def __init__(self, vocab=100, dim=16, classes=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(2 * dim, classes)

    def forward(self, tokens, max_len):
        tokens = tokens[:, :max_len]
        keep = tokens.ne(0).unsqueeze(-1)
        e = self.emb(tokens) * keep.float()
        n = keep.sum(dim=1).clamp(min=1).float()
        mean_pool = e.sum(dim=1) / n
        max_pool = e.masked_fill(keep.logical_not(), -1e4).amax(dim=1)
        return self.head(torch.cat([mean_pool, max_pool], dim=-1))


def build():
    """Return (program, calls, about)."""
    torch.manual_seed(0)
    program = BagOfEmbeddingsClassifier().eval()
    plan = [
        (4, 20, 32), (4, 20, 32), (4, 40, 32), (2, 64, 32), (8, 32, 32), (8, 33, 32),
        (1, 5, 32), (4, 20, 8), (4, 20, 20), (4, 20, 21), (1, 1, 32), (3, 50, 1),
        (16, 12, 128), (4, 20, 32),
    ]
    calls = []
    for b, t, max_len in plan:
        ids = torch.randint(1, 100, (b, t))
        lengths = torch.randint(1, t + 1, (b,))
        tokens = torch.where(torch.arange(t)[None, :] < lengths[:, None], ids, torch.zeros_like(ids))
        calls.append((tokens, max_len))
    about = (
        "Embedding-bag text classifier that truncates with tokens[:, :max_len]: max_len is sometimes larger than "
        "the sequence, so the real length is min(T, max_len), and pooling divides by a clamped count of non-pad tokens."
    )
    return program, calls, about
