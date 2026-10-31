import torch
import torch.nn.functional as F


def clip_loss(img_emb, txt_emb, logit_scale):
    """Symmetric CLIP/InfoNCE loss with several captions per image.

    img_emb (N, D), txt_emb (N*K, D) where captions of image i are rows i*K .. i*K+K-1;
    the K captions are averaged. logit_scale is a log-temperature tensor of shape () or (1,).
    """
    n = img_emb.shape[0]
    k = txt_emb.shape[0] // n
    txt = txt_emb.reshape(n, k, -1).mean(dim=1)
    a = img_emb * img_emb.pow(2).sum(dim=-1, keepdim=True).clamp(min=1e-12).rsqrt()
    t = txt * txt.pow(2).sum(dim=-1, keepdim=True).clamp(min=1e-12).rsqrt()
    logits = logit_scale.exp() * (a @ t.t())
    labels = torch.arange(n)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


def build():
    """Return (program, calls, about)."""
    torch.manual_seed(0)
    program = clip_loss
    scale0 = torch.tensor(2.6593)
    scale1 = torch.tensor([2.6593])
    plan = [
        (8, 1, 32, 0), (8, 1, 32, 0), (8, 5, 32, 0), (16, 5, 32, 0), (8, 2, 32, 1),
        (1, 1, 32, 0), (1, 5, 32, 0), (4, 3, 64, 0), (4, 3, 16, 1), (32, 1, 32, 0),
        (2, 5, 8, 0), (8, 5, 32, 1), (3, 4, 32, 0), (8, 1, 32, 0),
    ]
    calls = []
    for n, k, d, s in plan:
        calls.append((torch.randn(n, d), torch.randn(n * k, d), scale1 if s else scale0))
    about = (
        "Symmetric CLIP loss with K captions per image: K is computed as txt rows // img rows (image and text "
        "batch sizes must agree up to K), batch size 1 gives a 1x1 logit matrix, and logit_scale is sometimes () and sometimes (1,)."
    )
    return program, calls, about
