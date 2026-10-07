# The inference arithmetic: FLOPs, bandwidth and how long one step takes

<p class="lead">Do the arithmetic before optimising. LLM inference's accounting (see [The mathematics of performance and serving](math://performance-math/)) has to be redone here: a diffusion model has no KV cache, every step is a complete forward pass, and the batch means something different. This chapter gives the FLOP formula for one DiT step, works out when attention overtakes the linear layers, whether a step is compute-bound or bandwidth-bound, and how long one step takes on a card, and lines those numbers up against the mainstream models. With this accounting in hand, where each later technique saves and by how much can be estimated before any work begins.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Which two terms make up a DiT layer's FLOPs? How does each relate to the token count $N$ and the hidden dimension $d$?
    2. At what token count does attention's computation exceed the linear layers'? Which side are FLUX at 1024² and a video model on?
    3. Is a diffusion step compute-bound or bandwidth-bound? Why does it differ from an LLM's decode?
    4. How do you estimate a step's time from its FLOPs? What is a normal model FLOPs utilization?
    5. Going from a batch of 1 to 4, how much does the throughput rise? And the latency?

??? success "Answers for the self-test (answer first, then open this)"
    1. The linear layers (QKV, the output projection, the MLP) are about $24 N d^2$, proportional to $N$; attention's $QK^\top$ and $PV$ are about $4 N^2 d$, proportional to $N^2$.
    2. Setting them equal gives $N = 6d$. FLUX's $d = 3072$, so the crossover is about 18k tokens: 1024²'s 4608 tokens are on the linear-layer-dominated side, 2048² is near the crossover, and video models (tens of thousands to over a hundred thousand tokens) have attention at seventy or eighty percent and above.
    3. Compute-bound. One step does a complete forward pass over every token, so each read of the weights does FLOPs on the order of $2N$ (with $N$ in the thousands to tens of thousands), an arithmetic intensity far past the roofline's knee; an LLM's decode computes one token per read of the weights and is bandwidth-bound.
    4. $t = \text{FLOP} / (\text{MFU} \times \text{the peak})$. A diffusion model's utilization is usually 30% to 50% (the attention and linear layers are a large share, but the normalisations, activation functions and modulation are a good many memory-bound operations); a UNet's is lower because of the convolutions and the many small operations.
    5. The throughput barely rises: the step is compute-bound to begin with and batching only computes more tokens at once, so the total FLOPs grow linearly; the latency grows nearly linearly too. This is the opposite of an LLM decode's "bigger batches always pay", and it is the root of why scheduling differs in a generation service.

## One step's FLOPs {#一步的-flop}

A DiT layer does exactly what an LLM's Transformer layer does: the QKV projection, attention, the output projection, the MLP. With a token count $N$, a hidden dimension $d$ and an MLP widened 4 times, one layer's FLOPs (counting a multiply-add as 2) are

$$
\text{FLOP}_\text{layer} \approx \underbrace{24\, N d^2}_{\text{linear layers}} + \underbrace{4\, N^2 d}_{\text{attention}}
$$

The two are equal at $N = 6d$. Writing that as a function and substituting the mainstream models:

```python
MODELS = {
    # name: (layers, hidden dimension, image tokens, text tokens, the guidance factor per step)
    "SD3-medium 1024²":        (24, 1536, 4096, 333, 2),
    "FLUX.1-dev 1024²":        (57, 3072, 4096, 512, 1),      # 19 dual-stream plus 38 single-stream, estimated at one width here
    "FLUX.1-dev 2048²":        (57, 3072, 16384, 512, 1),
    "CogVideoX-5B 480p 49 帧": (42, 3072, 17550, 226, 2),
    "Wan 2.1-14B 720p 81 帧":  (40, 5120, 75600, 512, 2),
    "HunyuanVideo 720p 129 帧":(60, 3072, 118800, 256, 1),
}

def step_flops(layers, d, n_img, n_txt):
    n = n_img + n_txt
    linear = 24 * n * d * d
    attn = 4 * n * n * d
    return layers * linear, layers * attn

print(f"{'模型':<26} {'token':>8} {'临界 N=6d':>9} {'线性 TFLOP':>10} {'注意力 TFLOP':>12} {'注意力占比':>9} {'一步合计':>9}")
for name, (L, d, ni, nt, cfg) in MODELS.items():
    lin, att = step_flops(L, d, ni, nt)
    print(f"{name:<26} {ni + nt:>8,} {6 * d:>9,} {lin / 1e12:>10.1f} {att / 1e12:>12.1f} {att / (lin + att):>9.0%} {(lin + att) * cfg / 1e12:>9.1f}")
```

```text title="output"
模型                            token   临界 N=6d   线性 TFLOP    注意力 TFLOP     注意力占比      一步合计
SD3-medium 1024²              4,429     9,216        6.0          2.9       32%      17.8
FLUX.1-dev 1024²              4,608    18,432       59.5         14.9       20%      74.4
FLUX.1-dev 2048²             16,896    18,432      218.1        200.0       48%     418.1
CogVideoX-5B 480p 49 帧       17,776    18,432      169.1        163.1       49%     664.4
Wan 2.1-14B 720p 81 帧        76,112    30,720     1915.4       4745.7       71%   13322.2
HunyuanVideo 720p 129 帧     119,056    18,432     1617.9      10450.5       87%   12068.4
```

Three conclusions:

- **Image models are on the linear-layer-dominated side**: attention is only a little over twenty percent for FLUX at 1024². There, quantizing the matrix multiplies and fusing the linear layers is more effective than optimising the attention kernel.
- **Video models are on the attention-dominated side**: attention is over seventy percent for Wan at 720p and over eighty for HunyuanVideo. There, everything turns on attention: FlashAttention and SageAttention, sparse attention, sequence parallelism.
- **Double the resolution and attention multiplies by 16**: going from 1024² to 2048², FLUX's linear layers grow 4 times and its attention 16, with the share jumping from twenty percent to fifty. The same model's bottleneck moves with the resolution.

These are estimates (FLUX's dual-stream blocks are wider than its single-stream ones, and Wan also has cross-attention), but the magnitudes and proportions are reliable enough to set the priorities.

The formula as a calculator: change the model, the token count and the step count, and watch the linear layers and attention swap places, and how long one generation takes on different cards:

<div class="aig-widget" data-widget="diffusion-flops"></div>

## Compute-bound or bandwidth-bound {#算力受限还是带宽受限}

The most important distinction in LLM inference is that prefill is compute-bound and decode is bandwidth-bound (see [The KV cache and the two phases of inference](llm://inference/kv-cache/)). A diffusion step amounts to a prefill over all of the tokens: each read of the weights does computation on $N$ tokens. The arithmetic intensity (FLOPs per byte of weights) is:

```python
H100 = dict(tflops=989, bw=3.35)          # dense bf16 TFLOPS, memory bandwidth TB/s
RTX4090 = dict(tflops=165, bw=1.0)
ridge = {k: v["tflops"] * 1e12 / (v["bw"] * 1e12) for k, v in [("H100", H100), ("RTX 4090", RTX4090)]}
print("屋顶线拐点（FLOP / 字节）：" + "，".join(f"{k} {v:.0f}" for k, v in ridge.items()))
print()
print(f"{'模型':<26} {'权重 GB(bf16)':>13} {'一步 TFLOP':>10} {'算术强度':>9}  结论")
PARAMS = {"SD3-medium 1024²": 2.0, "FLUX.1-dev 1024²": 11.9, "FLUX.1-dev 2048²": 11.9,
          "CogVideoX-5B 480p 49 帧": 5.0, "Wan 2.1-14B 720p 81 帧": 14.0, "HunyuanVideo 720p 129 帧": 12.7}
for name, (L, d, ni, nt, cfg) in MODELS.items():
    lin, att = step_flops(L, d, ni, nt)
    weight_bytes = PARAMS[name] * 1e9 * 2
    ai = (lin + att) / weight_bytes
    print(f"{name:<26} {weight_bytes / 1e9:>13.1f} {(lin + att) / 1e12:>10.1f} {ai:>9,.0f}  {'算力受限' if ai > ridge['H100'] else '带宽受限'}")
print()
print("对比 LLM decode：7B 模型 batch=1 每读 14 GB 权重只算 2×7G FLOP，算术强度 1——深陷带宽受限")
```

```text title="output"
屋顶线拐点（FLOP / 字节）：H100 295，RTX 4090 165

模型                           权重 GB(bf16)   一步 TFLOP      算术强度  结论
SD3-medium 1024²                     4.0        8.9     2,228  算力受限
FLUX.1-dev 1024²                    23.8       74.4     3,124  算力受限
FLUX.1-dev 2048²                    23.8      418.1    17,566  算力受限
CogVideoX-5B 480p 49 帧              10.0      332.2    33,218  算力受限
Wan 2.1-14B 720p 81 帧               28.0     6661.1   237,896  算力受限
HunyuanVideo 720p 129 帧             25.4    12068.4   475,133  算力受限

对比 LLM decode：7B 模型 batch=1 每读 14 GB 权重只算 2×7G FLOP，算术强度 1——深陷带宽受限
```

**Every diffusion step is far to the right of the knee**: it is compute-bound even at a batch of 1. That explains several things:

- Batching helps a diffusion model's throughput very little: an LLM's decode uses the batch to take the arithmetic intensity from 1 to the hundreds, while a diffusion step is already in the thousands.
- Weight quantization (W8, W4) mainly **saves memory** for a diffusion model and does not buy speed directly as it does for an LLM decode; speed needs the activations quantized too (W8A8, FP8 or even W4A4) so that the matrix multiplies themselves get faster (see the chapter on quantization).
- What is genuinely bandwidth-bound are the operations that **do no matrix multiply**: normalisation, AdaLN modulation, activation functions, residual additions, the VAE's convolutions. Their FLOPs are few but they read and write the whole activation. Their share of the timeline is often comparable to the matrix multiplies', and they are where torch.compile and operator fusion earn their keep (see the chapter on kernel acceleration).

## How long one step takes {#一步要多久}

With the FLOPs in hand, a step's time is $t = \text{FLOP} / (\text{MFU} \times \text{the peak})$. The model FLOPs utilization is the measured throughput over the peak, and a diffusion model's typical values:

| Case | Typical utilization | Why |
| --- | --- | --- |
| A DiT in bf16 with FlashAttention and torch.compile | 40% to 55% | almost entirely large matrix multiplies and attention |
| A DiT without compilation | 25% to 40% | the small kernels of modulation, normalisation and activation functions take the time |
| A UNet (SDXL) | 15% to 30% | convolutions, GroupNorms, layers of differing shapes |
| Any model at a very small batch and a very low resolution | lower still | the kernels are too small to fill the GPU |

```python
GPUS = {"H100 SXM": 989, "RTX 4090": 165, "RTX 5070 Ti（估）": 170, "A100": 312}   # dense bf16 TFLOPS
def one_step(name):                                   # the TFLOP of one forward pass from the formula above
    L, d, ni, nt, cfg = MODELS[name]
    lin, att = step_flops(L, d, ni, nt)
    return (lin + att) / 1e12
CASES = [("SDXL 1024² · 30 步 · CFG", 12.0, 60, 0.25),                               # the UNet: one forward pass worked back from measured latency
         ("FLUX.1-dev 1024² · 28 步", one_step("FLUX.1-dev 1024²"), 28, 0.45),
         ("Wan 2.1-14B 720p 81 帧 · 50 步 · CFG", one_step("Wan 2.1-14B 720p 81 帧"), 100, 0.45)]
print(f"{'配置':<36} " + " ".join(f"{g:>14}" for g in GPUS))
for name, tflop, nfe, mfu in CASES:
    row = []
    for g, peak in GPUS.items():
        secs = tflop * nfe / (mfu * peak)
        row.append(f"{secs:>12.1f} s" if secs < 600 else f"{secs / 60:>10.1f} min")
    print(f"{name:<36} " + " ".join(f"{r:>14}" for r in row))
print("（一次前向的 TFLOP 来自本章的公式，SDXL 按实测反推；MFU 按上表取值；没算文本编码和 VAE）")
```

```text title="output"
配置                                         H100 SXM       RTX 4090 RTX 5070 Ti（估）           A100
SDXL 1024² · 30 步 · CFG                       2.9 s         17.5 s         16.9 s          9.2 s
FLUX.1-dev 1024² · 28 步                       4.7 s         28.0 s         27.2 s         14.8 s
Wan 2.1-14B 720p 81 帧 · 50 步 · CFG         24.9 min      149.5 min      145.1 min       79.1 min
（一次前向的 TFLOP 来自本章的公式，SDXL 按实测反推；MFU 按上表取值；没算文本编码和 VAE）
```

This table is not for precise prediction but for **fixing the order of magnitude and spotting the unreasonable**: if FLUX takes 40 seconds for an image on a 4090 and the table says 8, the gap is either in the utilization (compilation off, the attention backend degraded), in memory (offloaded to the CPU), or in the data (the VAE decode not tiled and waiting on memory). Real latency also adds the text encoding (tens of milliseconds) and the VAE decode (about 0.1 to 0.3 seconds at 1024², seconds to tens of seconds for video).

The video row is the most worth looking at: **one 5-second 720p video takes over twenty minutes on a single H100** (published measurements are of the same order). That is why video inference has to use multi-GPU parallelism, caching and distillation: not as extras, but because without them it cannot be a service at all.

## What the batch means here {#batch-在这里意味着什么}

The heart of an LLM service is batching many requests to amortise the weight reads; a diffusion step is compute-bound already and a batch amortises nothing. Here is how the throughput and latency vary with it:

```python
def throughput(batch, tflop_per_image, peak, mfu_batch1=0.40, mfu_gain=0.08):
    """batch 变大时 MFU 略有提升（kernel 更满），但一步的 FLOP 线性增长。"""
    mfu = min(0.6, mfu_batch1 + mfu_gain * (batch - 1) ** 0.5)
    step = batch * tflop_per_image / (mfu * peak)          # this batch's time for one step, in seconds
    return mfu, step, batch / step                          # the utilization, the step time, and how many images' steps per second

print(f"{'batch':>5} {'MFU':>5} {'一步时间':>8} {'吞吐(相对 batch=1)':>18} {'单张延迟(相对)':>14}")
_, s1, t1 = throughput(1, 18.0, 989)
for b in (1, 2, 4, 8):
    mfu, s, t = throughput(b, 18.0, 989)
    print(f"{b:>5} {mfu:>5.0%} {s * 1000:>6.0f} ms {t / t1:>18.2f}× {s / s1:>13.2f}×")
```

```text title="output"
batch   MFU     一步时间     吞吐(相对 batch=1)       单张延迟(相对)
    1   40%     46 ms               1.00×          1.00×
    2   48%     76 ms               1.20×          1.67×
    4   54%    135 ms               1.35×          2.97×
    8   60%    243 ms               1.50×          5.33×
```

From a batch of 1 to 8, the throughput rises only thirty percent (from slightly better utilization) while the latency rises sixfold. So a generation service's scheduling logic is nearly the opposite of an LLM's: **do not chase large batches; fill each card and spread the requests out**. The batch genuinely helps when the requests are naturally the same shape ("four candidate images for one prompt"), or when a small model at a low resolution leaves kernels unable to fill the GPU (see [Scheduling a generation service](../serving/scheduling.md)).

!!! interview "How to answer in an interview"
    Asked where diffusion inference's bottleneck is, give the formula first: $24Nd^2 + 4N^2d$ per layer with a crossover at $N = 6d$; images are on the linear-layer side (where quantizing the matrix multiplies works) and video on the attention side (where everything turns on attention). Then the placement: every step's arithmetic intensity is in the thousands, far right of the roofline's knee, so it is compute-bound, the opposite of an LLM decode. Hence batching barely helps throughput and weight quantization saves memory rather than time; what is genuinely bandwidth-bound are the small operations like normalisation and modulation, which need compilation and fusion. Finish with the number: one 5-second 720p video takes over twenty minutes on a single H100, so video has to have parallelism, caching and distillation.

## Exercises {#练习}

1. Use this chapter's formula to compute how many times SD3-medium's step FLOPs grow going from 1024² to 2048², and what attention's share becomes. Then compare with FLUX: for the same resolution jump, which model's attention share changes more? Why?

??? success "Answer"
    The token count goes from 4429 to 16717, so the linear layers grow about 3.8 times and attention about 14, about 6 times in total; attention's share rises from about 25% to about 55%. SD3's $d = 1536$ is half FLUX's, so its crossover $6d$ is only 9216 and it enters attention domination earlier at the same token count. The smaller a model's hidden dimension, the more it depends on attention optimisation as the resolution rises.

2. On an RTX 4090 (165 bf16 TFLOPS, 1 TB/s) running Wan 2.1-1.3B ($d = 1536$, 30 layers) at 480p over 81 frames, estimate one step's FLOPs, its arithmetic intensity and its time (at 40% utilization). Is it compute-bound?

??? success "Answer"
    The latents are 21x60x104, so after a patch of 2 there are about 32,760 tokens plus 512 text ones; the linear layers are about $30 \times 24 \times 33k \times 1536^2 \approx 56$ TFLOP and attention about $30 \times 4 \times 33k^2 \times 1536 \approx 200$ TFLOP, so about 260 TFLOP per step; the weights are 2.6 GB, giving an arithmetic intensity of about 100,000, far compute-bound; a step is about $260 / (0.4 \times 165) \approx 4$ seconds, so 50 steps with guidance take 400. A "small" 1.3B model is not small on video, because the cost is in $N^2$ rather than in the parameter count.

3. A service batches FLUX requests to 4 before running them and claims the throughput quadrupled. Use this chapter's model to say what is wrong with that, and when it might be right.

??? success "Answer"
    When a step is compute-bound, a batch of 4 takes about 4 times the time of a batch of 1, so the throughput is unchanged and the latency is 4 times. Only when a batch of 1 leaves the GPU unfilled (a low resolution, a small model, kernels too small) does batching raise the utilization and with it the throughput, and then the gain is the utilization's improvement, never 4 times. The more likely real explanation is that the batch-of-1 implementation had another problem (no compilation, large CPU overhead).

## Summary {#小结}

- [x] A DiT step's FLOPs are $L\,(24Nd^2 + 4N^2d)$ with a crossover at $N = 6d$: image models are on the linear-layer side and video models on the attention side, and doubling the resolution multiplies attention by 16.
- [x] Every step's arithmetic intensity is in the thousands and it is compute-bound at a batch of 1; batching barely helps throughput, weight quantization mainly saves memory, and what is genuinely bandwidth-bound are the small operations like normalisation and modulation.
- [x] A step's time is about FLOP / (utilization x peak), with a DiT at 40% to 55% and a UNet at 15% to 30%; one 5-second 720p video takes over twenty minutes on a single H100.
- [x] This accounting places every later technique: few steps and caching reduce the forward passes, quantization and compilation lower the per-step cost, parallelism spreads a step out, and the batch is not the answer.
