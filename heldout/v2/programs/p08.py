import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedGRU(nn.Module):
    """GRU unrolled over time in Python; the hidden state stops updating after each sequence's length.

    Returns concat(mean over time of hidden states, final hidden state).
    """

    def __init__(self, inp=8, hid=16):
        super().__init__()
        self.hid = hid
        self.x2h = nn.Linear(inp, 3 * hid)
        self.h2h = nn.Linear(hid, 3 * hid)

    def forward(self, x, lengths):
        b, t, _ = x.shape
        h = torch.zeros(b, self.hid)
        outs = []
        for i in range(t):
            rx, zx, nx = self.x2h(x[:, i]).chunk(3, dim=-1)
            rh, zh, nh = self.h2h(h).chunk(3, dim=-1)
            r = torch.sigmoid(rx + rh)
            z = torch.sigmoid(zx + zh)
            n = torch.tanh(nx + r * nh)
            h_new = (1 - z) * n + z * h
            alive = (lengths > i).float().unsqueeze(-1)
            h = alive * h_new + (1 - alive) * h
            outs.append(h)
        return torch.cat([torch.stack(outs, dim=1).mean(dim=1), h], dim=-1)


def build():
    """Return (program, calls, about)."""
    torch.manual_seed(0)
    program = MaskedGRU().eval()
    plan = [(4, 6), (4, 6), (2, 6), (4, 7), (4, 5), (1, 1), (3, 1), (8, 10), (2, 2), (4, 6), (1, 12), (5, 3)]
    calls = []
    for b, t in plan:
        calls.append((torch.randn(b, t, 8), torch.randint(1, t + 1, (b,))))
    about = (
        "GRU unrolled with a Python loop over x.shape[1], so the traced graph has one step per time step "
        "(T=6 vs 7 vs 5 vs 1 must not share a graph), with a length mask freezing finished sequences."
    )
    return program, calls, about
