import torch
import torch.nn.functional as F


def to_model_layout(img, mean, std, r):
    """Dataloader-side preprocessing: NHWC pixels in 0..255 -> normalised NCHW with space-to-depth.

    mean/std are per-channel (C,) or shared (1,); grayscale inputs are expanded to 3 channels
    after normalisation; height and width are cropped to multiples of the factor r.
    """
    b, h, w, c = img.shape
    x = (img / 255.0 - mean) / std
    if c == 1:
        x = x.expand(b, h, w, 3)
        c = 3
    hh, ww = h // r, w // r
    x = x[:, : hh * r, : ww * r, :]
    x = x.reshape(b, hh, r, ww, r, c).permute(0, 5, 2, 4, 1, 3).reshape(b, -1, hh, ww)
    return x


def build():
    """Return (program, calls, about)."""
    torch.manual_seed(0)
    program = to_model_layout
    rgb_mean = torch.tensor([0.485, 0.456, 0.406])
    rgb_std = torch.tensor([0.229, 0.224, 0.225])
    one_mean = torch.tensor([0.5])
    one_std = torch.tensor([0.25])
    plan = [
        (2, 32, 32, 3, "rgb", 2), (2, 32, 32, 3, "rgb", 2), (4, 33, 31, 3, "rgb", 2),
        (2, 32, 32, 3, "one", 2), (2, 32, 32, 1, "one", 2), (1, 28, 28, 1, "one", 4),
        (2, 32, 32, 3, "rgb", 4), (1, 7, 5, 3, "rgb", 3), (3, 16, 24, 3, "rgb", 1),
        (2, 3, 3, 3, "rgb", 3), (2, 17, 17, 1, "one", 2), (8, 64, 48, 3, "rgb", 8),
        (1, 9, 10, 3, "one", 4), (2, 32, 32, 3, "rgb", 2),
    ]
    calls = []
    for b, h, w, c, kind, r in plan:
        img = torch.randint(0, 256, (b, h, w, c)).float()
        mean, std = (rgb_mean, rgb_std) if kind == "rgb" else (one_mean, one_std)
        calls.append((img, mean, std, r))
    about = (
        "NHWC->NCHW space-to-depth preprocessing: mean/std are sometimes (1,) and broadcast, a Python branch "
        "expands 1-channel images to 3, h and w are floor-cropped to multiples of the int factor r, and the channel "
        "count comes from a -1 reshape."
    )
    return program, calls, about
