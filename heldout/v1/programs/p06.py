import torch
import torch.nn.functional as F


def build():
    """Return (program, calls, about)."""
    torch.manual_seed(0)
    d = 16
    w_out = torch.randn(5, d) * 0.3

    def program(queries, memory, scale):
        # learned-query cross-attention pooling over a memory, with a broadcast per-sample scale
        dim = queries.shape[-1]
        mem = F.rms_norm(memory, (dim,)) * scale
        out = F.scaled_dot_product_attention(queries.unsqueeze(1), mem.unsqueeze(1), mem.unsqueeze(1))
        out = out.squeeze(1)  # drop the single head
        out = out.squeeze(1)  # meant for "one query"; is a no-op when there are several
        return F.linear(out, w_out)

    specs = [  # (batch, queries, memory length, scale batch)
        (2, 1, 10, 1), (2, 1, 10, 1), (3, 1, 10, 1), (2, 1, 25, 1), (2, 4, 10, 1),
        (2, 1, 10, 2), (4, 4, 10, 4), (1, 1, 1, 1), (1, 3, 1, 1), (2, 2, 10, 2),
        (5, 1, 7, 5), (1, 1, 40, 1), (2, 1, 10, 1), (3, 8, 12, 1), (2, 1, 3, 2),
    ]
    calls = []
    for b, q, s, sb in specs:
        calls.append((torch.randn(b, q, d), torch.randn(b, s, d), torch.rand(sb, 1, d) + 0.5))
    about = ("Cross-attention pooling whose second squeeze(1) drops the query axis only when there is "
             "exactly one query (so the output rank depends on that size), with a scale whose batch "
             "is sometimes 1 and broadcasts and a memory that is sometimes length 1.")
    return program, calls, about
