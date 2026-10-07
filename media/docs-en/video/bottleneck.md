# The bottleneck in video inference: attention over a hundred thousand tokens

<p class="lead">An image model's optimisation concentrates on the linear layers and the small operations; a video model has only one thing: attention. Full 3D attention over a hundred thousand tokens is seventy or eighty percent of a step, and one 5-second 720p video takes over twenty minutes on a single card. This chapter breaks one video generation's time down by stage, then goes through the ways to attack attention one by one: a faster kernel (SageAttention), sparse attention (sliding tiles, radial, post-training sparsification), chunked computation, sequence parallelism and caching, with where each saves and how far they stack. It ends with the arithmetic of going from 25 minutes to 2.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Which stages take the time in one video generation? What is each one's share?
    2. Why is a video model's attention said to be naturally sparse? How is sparse attention's speedup computed?
    3. What locality do sliding-tile attention and radial attention each exploit?
    4. Can sparse attention stack with sequence parallelism and caching? Do they interfere?
    5. Why can the VAE decode not be ignored in video?

??? success "Answers for the self-test (answer first, then open this)"
    1. The text encoding (under a second), the denoising network's 50 steps (the overwhelming bulk, over twenty minutes on a single H100, with attention seventy or eighty percent of it), and the VAE decode (seconds to tens of seconds for 81 frames of 720p, depending on the tiling).
    2. In the attention score matrix, most of the weight concentrates between tokens that are adjacent in space and time (nearby positions in the same frame, the same region in neighbouring frames), and distant tokens contribute little. The speedup is about 1 / the attention density kept (times the sparse kernel's efficiency): keeping 20% is about 4 to 5 times faster attention.
    3. Sliding-tile attention organises the 3D tokens into spatiotemporal tiles and has each tile attend only to the tiles in a local window around it, which aligns naturally with FlashAttention's blocking and leaves no fragmented mask. Radial attention observes that attention decays as temporal distance grows and shrinks the spatial window exponentially with the temporal distance, bringing the complexity down to $O(N \log N)$.
    4. They stack, with caveats: the sparse attention has to use the same mask on every card; when sequence parallelism splits the tokens, the sparse pattern's locality is best aligned with the split (splitting by frame keeps sliding-tile attention's window within a card); and the caching's skip decisions have to be synchronised across cards.
    5. A video VAE's decode convolves in 3D in pixel space, where 81 frames of 720p has 81 times a single image's activations, so it will not fit without tiling and even tiled it often takes tens of seconds. Once the denoising has been parallelised and cached down to two or three minutes, the decode is a substantial share.

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/video-bottleneck.webp is in Chinese; put it back once the English version exists -->

## Where one generation's time goes {#一次生成的时间去哪了}

Breaking one Wan 2.1-14B generation of 81 frames of 720p on a single H100 down by stage (estimated with [the accounting chapter](../perf/accounting.md)'s formulas and typical utilizations):

```python
import math

T, H, W = 21, 90, 160                               # the latents
N = T * (H // 2) * (W // 2) + 512                    # 75,600 video tokens plus 512 text tokens
d, L, steps, cfg = 5120, 40, 50, 2
PEAK, MFU_LIN, MFU_ATT = 989e12, 0.5, 0.45

linear = L * 24 * N * d * d
attn = L * 4 * N * N * d
t_linear = linear / (MFU_LIN * PEAK)
t_attn = attn / (MFU_ATT * PEAK)
t_step = t_linear + t_attn
stages = [("文本编码（umT5-XXL）", 0.8), ("去噪 50 步 × CFG：线性层", t_linear * steps * cfg),
          ("去噪 50 步 × CFG：注意力", t_attn * steps * cfg), ("VAE 解码（时间分块）", 25.0)]
total = sum(t for _, t in stages)
print(f"{'阶段':<24} {'时间':>10} {'占比':>6}")
for name, t in stages:
    print(f"{name:<24} {t:>8.0f} s {t / total:>6.0%}")
print(f"{'合计':<24} {total:>8.0f} s = {total / 60:.1f} 分钟（单卡 H100，无任何加速）")
```

```text title="output"
阶段                               时间     占比
文本编码（umT5-XXL）                  1 s     0%
去噪 50 步 × CFG：线性层             387 s    26%
去噪 50 步 × CFG：注意力            1066 s    72%
VAE 解码（时间分块）                   25 s     2%
合计                           1479 s = 24.7 分钟（单卡 H100，无任何加速）
```

**Attention is seventy percent.** The linear layers are already the battlefield of FP8, compilation and fusion (the same as for images), and the VAE's tens of seconds need tiling and parallel decoding; what decides video inference is the attention.

## A faster kernel {#更快的-kernel}

The first layer of techniques does not change the computation, only computes the same attention faster:

| Technique | How | Typical gain (on the attention) | Cost |
| --- | --- | --- | --- |
| FlashAttention-2 / 3 | blocking with an online softmax; FA3 uses Hopper's asynchronous Tensor Cores and FP8 | the baseline / FA3 is 1.5 to 2x FA2 | FA3 is Hopper only |
| SageAttention 1 / 2 | $QK^\top$ in INT8 (smoothed by block), $PV$ in FP8 or FP16 | 2 to 3x FA2 | a few cases need tuning |
| Fused RoPE and attention | RoPE inside the kernel | a few percent | the implementation has to change |

SageAttention is all but the default for a video model: attention is seventy percent and it is 2.5 times faster, which is 1.6 times overall, without changing the model, retraining, or any visible difference in quality.

## Sparse attention: most of the scores are near zero anyway {#稀疏注意力大部分分数本来就接近零}

Full 3D attention lets every token see every token, but in a trained model the attention weight concentrates between tokens **adjacent in space and time**. Quantifying that property: assuming the attention weight decays with spatiotemporal distance, how much of the weight mass is covered by keeping neighbours within various ranges:

```python
import torch

torch.manual_seed(0)
T, Hp, Wp = 21, 45, 80                              # Wan's 720p token grid (after a patch of 2)
# a toy decaying distribution models attention mass falling with spatiotemporal distance: 0.5 per step of temporal distance and 0.85 per step of spatial
def mass_within(dt_max, ds_max):
    """一个位于中心的 query，它的注意力质量有多少落在 |Δt| ≤ dt_max 且 |Δh|,|Δw| ≤ ds_max 的邻居里"""
    t = torch.arange(-T // 2, T // 2 + 1).float()
    h = torch.arange(-Hp // 2, Hp // 2 + 1).float()
    w = torch.arange(-Wp // 2, Wp // 2 + 1).float()
    wt, wh, ww = 0.5 ** t.abs(), 0.85 ** h.abs(), 0.85 ** w.abs()
    full = wt.sum() * wh.sum() * ww.sum()
    kept = wt[t.abs() <= dt_max].sum() * wh[h.abs() <= ds_max].sum() * ww[w.abs() <= ds_max].sum()
    density = ((2 * dt_max + 1) * (2 * ds_max + 1) ** 2) / (T * Hp * Wp)
    return (kept / full).item(), min(1.0, density)

print(f"{'时间窗 ±Δt':>9} {'空间窗 ±Δs':>9} {'覆盖的注意力质量':>12} {'计算密度':>8} {'注意力加速（理想）':>14}")
for dt, ds in [(1, 4), (2, 8), (3, 12), (5, 16), (10, 40)]:
    m, dens = mass_within(dt, ds)
    print(f"{dt:>9} {ds:>9} {m:>12.1%} {dens:>8.1%} {1 / dens:>13.1f}×")
```

```text title="output"
  时间窗 ±Δt   空间窗 ±Δs     覆盖的注意力质量     计算密度      注意力加速（理想）
        1         4        18.5%     0.3%         311.1×
        2         8        48.1%     1.9%          52.3×
        3        12        71.1%     5.8%          17.3×
        5        16        87.2%    15.8%           6.3×
       10        40       100.0%   100.0%           1.0×
```

This is a toy decay model and a real model's attention map has to be measured on calibration data (which is every sparse method's first step), but the shape of the conclusion holds: **keeping ten or twenty percent of the computation covers eighty or ninety percent of the attention mass**, and covering that last tenth means opening the window to nearly global. Every sparse method bets that the last tenth does not affect the picture. Several ways to realise it:

| Method | Sparse pattern | How it stays efficient | Retraining needed |
| --- | --- | --- | --- |
| Sliding-tile attention (STA) | each spatiotemporal tile sees a neighbourhood window | the tiles align with FlashAttention's blocks, with no fragmented mask | a little fine-tuning helps |
| Radial attention | the spatial window shrinks exponentially with the temporal distance | a static mask, $O(N \log N)$ | fine-tuning |
| Sparse video generation (SVG) / dynamic sparsity | heads split into spatial and temporal ones, with each head's pattern decided online | a block-sparse kernel | none |
| Post-training top-k blocks | each query block computes only its k highest-scoring key blocks | estimate the scores in low precision first, then select | none |
| Native sparsity (NSA, DSA style) | a sparse structure from training onward | a dedicated kernel | yes (the model itself) |

The sparse speedup is 1 / the density x the kernel's efficiency. A block-sparse kernel (blocks of 64 to 128 tokens) reaches 70% to 90% efficiency while a fine-grained per-token mask is very inefficient, which is why every usable method is **block-level**. This is the same reasoning as sparse attention in an LLM (see [Long context, KV eviction and sparse attention](serving://topics/long-context/)), with the differences that video's locality is three-dimensional and there is no KV cache to save.

## Chunking, parallelism and caching: stacking them {#分块并行与缓存叠起来}

Beyond attention, three families of techniques work by splitting the computation up or doing it less often, and they stack with the kernels and the sparsity:

```python
PEAK, MFU = 989.0, 0.47                              # the H100's bf16 TFLOPS and a typical utilization
BASE = {"线性层": 2 * 50 * 1903.0, "注意力": 2 * 50 * 4682.0, "VAE": 25.0 * PEAK * MFU}   # counted in TFLOP (the VAE converted from 25 s)
# each technique's speedup on each part (typical values, estimated)
STEPS = [
    ("基线：单卡 H100，FA2，bf16",            {"线性层": 1.0, "注意力": 1.0, "VAE": 1.0}),
    ("+ SageAttention",                        {"注意力": 2.5}),
    ("+ FP8 线性层 + torch.compile",            {"线性层": 1.4}),
    ("+ 滑动块稀疏注意力（密度 20%，kernel 效率 80%）", {"注意力": 4.0}),
    ("+ TeaCache（阈值取中等）",                {"线性层": 1.7, "注意力": 1.7}),
    ("+ 8 卡：CFG 2 × Ulysses 4（效率 85%）",   {"线性层": 6.8, "注意力": 6.8, "VAE": 4.0}),
]
speed = {"线性层": 1.0, "注意力": 1.0, "VAE": 1.0}
print(f"{'配置':<44} {'线性层':>7} {'注意力':>7} {'VAE':>6} {'合计':>7}")
for name, gains in STEPS:
    for k, g in gains.items():
        speed[k] *= g
    t = {k: BASE[k] / (PEAK * MFU) / speed[k] for k in BASE}
    print(f"{name:<44} {t['线性层']:>5.0f} s {t['注意力']:>5.0f} s {t['VAE']:>4.0f} s {sum(t.values()):>5.0f} s")
```

```text title="output"
配置                                               线性层     注意力    VAE      合计
基线：单卡 H100，FA2，bf16                            409 s  1007 s   25 s  1442 s
+ SageAttention                                409 s   403 s   25 s   837 s
+ FP8 线性层 + torch.compile                      292 s   403 s   25 s   720 s
+ 滑动块稀疏注意力（密度 20%，kernel 效率 80%）               292 s   101 s   25 s   418 s
+ TeaCache（阈值取中等）                              172 s    59 s   25 s   256 s
+ 8 卡：CFG 2 × Ulysses 4（效率 85%）                 25 s     9 s    6 s    40 s
```

Stacked ideally, over twenty minutes becomes about a minute. A real system also has pipeline gaps, communication waits, the repeated computation of VAE tiling and a less than ideal utilization, so the published 8-card figures are 1.5 to 3 minutes, two or three times the ideal accounting. That gap is the systems engineering. Every row in the table has its price and its preconditions too:

- **SageAttention**: all but free, do it first.
- **FP8 plus compilation**: the standard treatment for the linear layers, the same as for images.
- **Sparse attention**: the largest gain, but it needs calibration (or fine-tuning) and a dedicated kernel, and content that does not match the sparse pattern (large motion, a global lighting change) loses quality.
- **TeaCache**: the threshold has to be swept; the skip decisions have to be synchronised across cards; useless for a few-step model.
- **Several cards**: 8 cards reach 80% to 90% real efficiency, and Ulysses's communication is a small share within NVLink; guidance parallelism requires a copy of the model on each of two cards.

A real system (xDiT, SGLang Diffusion, the various closed services) is built up layer by layer exactly this way.

Stack it yourself:

<div class="aig-widget" data-widget="video-stack"></div>

## The VAE decode: the overlooked tens of seconds {#vae-解码被忽视的那几十秒}

Once the denoising is down to two minutes, the VAE decode's tens of seconds can no longer be ignored. Its problem is not compute but **activation memory**: 81 frames of 720p has 81 times a single image's pixel-level features. The techniques:

- **Temporal tiling**: a causal VAE can decode in segments of latent frames, a few frames each, with only one segment's memory; Wan made this a streaming interface.
- **Spatial tiling**: as for images, with blended overlaps (see [The VAE and the latent space](../basics/vae-latent.md)).
- **Parallel decoding**: segments distributed across cards, each decoding its own, with overlapping frames blended at the boundaries.
- **Streaming output**: return each segment as it is decoded. A video can play while it generates, and the user's perceived latency goes from waiting for the whole thing to waiting for the first segment.

!!! interview "How to answer in an interview"
    Asked how video generation is optimised, lay out the accounting first: over twenty minutes for one 5-second 720p video on a single H100, with attention seventy percent, the linear layers twenty and the VAE decode tens of seconds. Then attention's three layers: at the kernel level SageAttention's 2.5 times is nearly free; at the sparse level, spatiotemporal locality means keeping ten or twenty percent of the blocks covers eighty or ninety percent of the attention mass, which a block-sparse kernel turns into 3 to 4 times; then stack caching (TeaCache) and several cards (guidance x Ulysses). Finish with the result: about a minute stacked ideally, with published 8-card figures of 1.5 to 3 minutes, the gap being pipeline gaps and communication waits. Add a word on the VAE: once the denoising is down, the decode's tens of seconds need temporal tiling, parallelism and streaming output.

## Exercises {#练习}

1. Change `mass_within`'s temporal decay from 0.5 to 0.9 (more vigorous motion, with distant frames more relevant). What mass does the same window cover? What does that suggest about choosing a sparse pattern?

??? success "Answer"
    The same window covers noticeably less and the temporal window has to grow to get back to 90%, so the density rises and the speedup falls. The lesson is that the sparse pattern should adapt to the content (dynamic sparsity, per-head patterns), since a static window loses quality on video with vigorous motion. That is the starting point for methods like SVG that decide online whether each head is spatial or temporal.

2. Use this chapter's stacking table: with only 1 card (dropping the last row), how many minutes can be reached? Which item is most worth doing first on one card?

??? success "Answer"
    Without the 8-card row, the preceding items stack to about 4 minutes (about 3 for the linear layers, about 1 for attention, tens of seconds for the VAE). Note that the linear layers become the bulk at that point, because sparsity and caching mostly act on attention. On one card, the first things worth doing are SageAttention (free, 2.5 times on attention) and sparse attention (the largest gain), with TeaCache next. Below that there is only parallelism, or a smaller or distilled model.

3. Using sparse attention together with Ulysses sequence parallelism, if the tokens are split evenly in order across 4 cards while the sparse pattern is "each frame sees two frames either side", what happens to the communication? What split would be better?

??? success "Answer"
    Ulysses does an all-to-all before attention to repartition by head, and each card computes complete attention for its own heads, so the sparse mask is fully visible within each card, the split does not affect correctness, but every card has to hold every token's K and V and the sparsity saves no communication. With Ring Attention, splitting by **frame** lets the "two frames either side" pattern exchange K and V only with the neighbouring cards on the ring, so the communication falls with the sparsity. So sparsity and parallelism have to be designed together: temporally local sparsity with a Ring split by frame, spatially local sparsity with a split by spatial block.

## Summary {#小结}

- [x] One video generation's time: denoising is over ninety percent, of which attention is seventy and the linear layers twenty; the VAE decode's tens of seconds cannot be ignored once the denoising is down.
- [x] Attention's three layers: the kernel (SageAttention's 2.5 times, nearly free), sparsity (spatiotemporal locality lets ten or twenty percent of the blocks cover eighty or ninety percent of the mass, which a block-level kernel turns into 3 to 4 times), and splitting or doing less (sequence parallelism, caching).
- [x] These stack: one card goes from 25 minutes to a few, and 8 cards reach about a minute ideally against published figures of 1.5 to 3; each item has preconditions (calibration, thresholds, synchronisation, NVLink).
- [x] The VAE decode needs temporal and spatial tiling, parallelism and streaming output; streaming turns the user's perceived latency into waiting for the first segment.
