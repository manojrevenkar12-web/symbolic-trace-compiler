import torch
import torch.nn.functional as F


def masked_smoothed_ce(logits, targets, lengths, smoothing_pct):
    """Token-level label-smoothed cross entropy over padded sequences, averaged over real tokens.

    logits (B, T, V), targets (B, T) class ids, lengths (B,) real lengths (may exceed T),
    smoothing_pct an int percentage.
    """
    b, t, v = logits.shape
    mask = torch.arange(t)[None, :] < lengths[:, None]
    logp = F.log_softmax(logits.reshape(-1, v), dim=-1)
    onehot = (targets.reshape(-1, 1) == torch.arange(v)[None, :]).float()
    eps = smoothing_pct / 100.0
    target_dist = onehot * (1.0 - eps) + eps / v
    nll = -(target_dist * logp).sum(dim=-1).reshape(b, t)
    m = mask.float()
    return (nll * m).sum() / m.sum().clamp(min=1.0)


def build():
    """Return (program, calls, about)."""
    torch.manual_seed(0)
    program = masked_smoothed_ce
    plan = [
        (4, 12, 11, 10), (4, 12, 11, 10), (2, 9, 11, 10), (4, 12, 11, 0), (1, 1, 11, 10),
        (3, 20, 5, 10), (3, 20, 50, 10), (2, 7, 2, 20), (2, 7, 1, 10), (8, 16, 11, 5),
        (1, 30, 11, 10), (4, 12, 11, 10), (5, 3, 7, 0), (2, 40, 33, 15),
    ]
    calls = []
    for b, t, v, s in plan:
        logits = torch.randn(b, t, v)
        targets = torch.randint(0, v, (b, t))
        lengths = torch.randint(0, t + 3, (b,))
        calls.append((logits, targets, lengths, s))
    about = (
        "Masked label-smoothed cross entropy: the smoothing mass eps/V and the one-hot both depend on the "
        "vocabulary size V (including V=1), logits are flattened with reshape(-1, V), and the smoothing int sets eps."
    )
    return program, calls, about
