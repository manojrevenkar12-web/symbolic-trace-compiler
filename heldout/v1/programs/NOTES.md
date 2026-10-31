# Held-out programs

- p01: causal multi-head self-attention; trap: the int `num_heads` sets head_dim = d // num_heads, the `-1` in the reshapes and the score scale (1 to 32 heads), and the causal mask comes from t (including t = 1).
- p02: ViT patch embedding (patch 4); trap: images are cropped to h // 4 * 4 and w // 4 * 4, so 17x19 shares a grid with 16x16, and the position table is sliced to gh * gw + 1 rows computed from both sizes (from 1 patch up to 64).
- p03: sliding-window attention pooling over a sequence; trap: the int `window` unrolls a Python loop and gives t - window + 1 frames, and the int `stride` keeps ceil(n / stride) of them (window == t, window 1, stride > n).
- p04: truncate/pad to 16, then masked mean pooling by per-row lengths; trap: `x[:, :16]` runs past the end of short inputs and a Python branch pads only when t < 16 (t = 16, 15, 17, 1 and 40 all appear).
- p05: shifted next-token cross-entropy with label smoothing, ignore_index and z-loss; trap: `[:, :-1]`/`[:, 1:]` give t - 1 rows (down to one), vocab size drives `reshape(-1, v)` (2 to 51), and the two inputs must agree on batch and length.
- p06: learned-query cross-attention pooling over an RMS-normed memory; trap: the second `squeeze(1)` drops the query axis only when there is exactly one query, so the output rank changes with that size; the scale's batch is sometimes 1 and broadcasts; the memory is sometimes length 1.
- p07: partial rotary position embedding; trap: only the first 2 * (head_dim // 4) channels rotate (head_dim 6, 12, 18 leave odd remainders), and the int `offset` sets the positions (decode steps at t = 1 with offsets 16, 17, 18, 500).
- p08: NCHW -> tokens -> LayerNorm -> NCHW, then global pooling; trap: a Python branch on h * w > 64 crops odd edges and max-pools 2x2 first (8x8 = 64 does not pool, 8x9 and 9x8 do; 1x64 does not, 2x40 does).
- p09: sliding-window attention over a KV cache; trap: keeps the last window + tq - 1 keys with a negative slice that may reach past the start (empty or short caches) and builds the mask from positions offset by length - tq.
- p10: nn.Module (embedding plus one causal transformer block, eval mode) with a pooled classifier head; trap: positions come from arange(t), and a Python branch on t > 32 mean-pools only every other token (ceil(t / 2), 32 vs 33 vs 34 is the edge; also t = 1).
