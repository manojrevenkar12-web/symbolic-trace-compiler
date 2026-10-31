import torch
import torch.nn.functional as F


def build():
    """Return (program, calls, about)."""
    torch.manual_seed(0)
    pad_id = 0

    def program(logits, targets):
        # next-token loss: shift by one, label smoothing, ignore padding, plus a z-loss
        v = logits.shape[-1]
        shift_logits = logits[:, :-1].reshape(-1, v)
        shift_targets = targets[:, 1:].reshape(-1)
        ce = F.cross_entropy(shift_logits, shift_targets, ignore_index=pad_id, label_smoothing=0.1)
        z = torch.logsumexp(shift_logits, dim=-1)
        keep = (shift_targets != pad_id).float()
        z_loss = (z.square() * keep).sum() / keep.sum().clamp(min=1.0)
        return ce + 1e-4 * z_loss

    specs = [  # (batch, seq, vocab, trailing pads)
        (2, 12, 50, 2), (2, 12, 50, 2), (4, 12, 50, 0), (2, 20, 50, 5), (1, 2, 50, 0),
        (2, 12, 51, 2), (3, 8, 7, 1), (1, 3, 50, 1), (8, 12, 50, 4), (2, 33, 50, 0),
        (2, 12, 2, 2), (1, 64, 50, 10), (2, 5, 50, 3), (5, 12, 50, 2),
    ]
    calls = []
    for b, t, v, p in specs:
        logits = torch.randn(b, t, v)
        targets = torch.cat([torch.randint(1, v, (b, t - p)), torch.zeros(b, p, dtype=torch.long)], dim=1)
        calls.append((logits, targets))
    about = ("Shifted language-model loss whose logits[:, :-1] and targets[:, 1:] give t - 1 rows "
             "(down to one), whose vocabulary size feeds reshape(-1, v), and whose two inputs must "
             "agree on batch and length.")
    return program, calls, about
