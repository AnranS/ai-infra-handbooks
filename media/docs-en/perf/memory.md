# Memory and offload: fitting FLUX into 16 GB

<p class="lead">FLUX.1-dev's denoising network is 24 GB of bf16 weights, and T5 adds 9.5 GB, so even a 24 GB card cannot hold them, let alone 16 GB. Yet both run on such cards, through offload, quantization and tiling. This chapter first works out a diffusion model's memory budget (how much goes to the weights, the activations and the VAE peak), then quantifies the time each offload policy costs: how many bytes each of component offload, layer offload and group prefetching moves over PCIe, and how much they slow things down. By the end you can answer "this card, this model, this resolution: which policy".</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Which parts make up a diffusion model's memory at inference? What is the biggest difference from LLM inference?
    2. What do model offload and sequential (per-layer) offload each move, and when? What does each cost?
    3. Why does per-layer offload slow a step down several times over? How does prefetching hide it?
    4. How much memory do FP8 and NF4 weights take? What do they do to speed?
    5. How would you configure FLUX.1-dev on a 16 GB card?

??? success "Answers for the self-test (answer first, then open this)"
    1. The weights (denoising network, text encoder, VAE), one denoising step's activations, the VAE decode's activation peak, and the CUDA context and allocator fragmentation. The biggest difference from an LLM is that there is no KV cache: memory does not keep growing with the request count and sequence length, a step's activations are released as soon as they are used, and so the memory plan is static.
    2. Model offload: a whole component (T5, the denoising network, the VAE) is moved into the GPU when needed and back to the CPU afterwards, once each per generation, costing a few extra seconds per image. Per-layer offload: only one layer of the denoising network is resident and the next replaces it after each layer, so one step moves all of the weights, costing tens of gigabytes of transfer per step.
    3. A step moves 24 GB of weights, and PCIe 4.0 x16 is about 25 GB/s, so the transfer alone takes a second while the computation takes a fraction of one, leaving the GPU waiting for data most of the time. Prefetching: another CUDA stream moves layer $i+1$ in while layer $i$ computes, overlapping the transfer with the computation; as long as moving one layer is no slower than computing one, most of the slowdown is hidden. This requires the CPU memory to be pinned, so the transfer goes by DMA.
    4. FP8 takes 12B parameters to 12 GB and NF4 to about 6.5 GB. FP8's matrix multiplies are themselves faster on Hopper, Ada and later (which have FP8 Tensor Cores); NF4 has to dequantize to bf16 before computing, so the weight reads shrink but the computation does not, and since a diffusion model is compute-bound to begin with, NF4 is usually slower. What it buys is only fitting.
    5. The denoising network in FP8 or NF4 (12 GB or 6.5 GB), T5 in FP8 or NF4 or offloaded to the CPU after encoding, the VAE with tiled decoding on, and the rest for the activations. If that is still not enough, add group offload with prefetching for the denoising network.

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/media-memory.webp is in Chinese; put it back once the English version exists -->

## The memory budget {#显存账本}

One generation's memory splits into four parts:

| Part | Size | When it is held |
| --- | --- | --- |
| The denoising network's weights | parameters x bytes per parameter | the whole denoising process |
| The text encoder's weights | the same, with T5-XXL the bulk of it | only while encoding |
| The VAE's weights | small (84M to a few hundred million) | only while decoding |
| One denoising step's activations | proportional to tokens x hidden dimension, without the $N^2$ attention matrix under FlashAttention | every step, released after use |
| The VAE decode's activations | proportional to pixels x channels, often the peak | only while decoding |
| The CUDA context and allocator fragmentation | 1 to 2 GB | always |

Applying the formulas to the mainstream models at different precisions:

```python
GB = 1024 ** 3
MODELS = {
    # name: (denoising network parameters B, text encoder parameters B, VAE parameters B, hidden dimension, tokens including text, layers)
    "SDXL 1024²":          (2.6, 0.82, 0.08, 1280, 4096, 0),
    "FLUX.1-dev 1024²":    (11.9, 4.9, 0.08, 3072, 4608, 57),
    "Wan 2.1-14B 720p":    (14.0, 5.7, 0.13, 5120, 76112, 40),
    "HunyuanVideo 720p":   (12.7, 8.6, 0.25, 3072, 119056, 60),
}
BYTES = {"bf16": 2, "fp8": 1, "nf4": 0.5625}            # NF4 counted as 4 bits plus group scaling factors, about 4.5 bits

def activation_peak(d, n, layers, dtype_bytes=2):
    """一步里同时活着的激活：残差流 + 一层里的几份中间结果（QKV、MLP 的 4d 升维）；FlashAttention 不落地 N×N 矩阵"""
    per_layer = n * (d + 3 * d + 4 * d) * dtype_bytes   # the residual plus QKV plus the MLP's intermediate
    return per_layer * 2                                 # roughly two copies in flight (the current layer plus the residual's copy)

print(f"{'模型':<20} {'精度':<5} {'去噪权重':>8} {'文本编码器':>9} {'一步激活':>8} {'合计(含 1.5GB 开销)':>18}")
for name, (p_dn, p_te, p_vae, d, n, L) in MODELS.items():
    for dtype in ("bf16", "fp8", "nf4"):
        w_dn = p_dn * 1e9 * BYTES[dtype]
        w_te = p_te * 1e9 * BYTES[dtype]
        act = activation_peak(d, n, L)
        total = w_dn + w_te + p_vae * 1e9 * 2 + act + 1.5 * GB
        print(f"{name:<20} {dtype:<5} {w_dn / GB:>6.1f} GB {w_te / GB:>7.1f} GB {act / GB:>6.1f} GB {total / GB:>16.1f} GB")
```

```text title="output"
模型                   精度        去噪权重     文本编码器     一步激活     合计(含 1.5GB 开销)
SDXL 1024²           bf16     4.8 GB     1.5 GB    0.2 GB              8.2 GB
SDXL 1024²           fp8      2.4 GB     0.8 GB    0.2 GB              5.0 GB
SDXL 1024²           nf4      1.4 GB     0.4 GB    0.2 GB              3.6 GB
FLUX.1-dev 1024²     bf16    22.2 GB     9.1 GB    0.4 GB             33.4 GB
FLUX.1-dev 1024²     fp8     11.1 GB     4.6 GB    0.4 GB             17.7 GB
FLUX.1-dev 1024²     nf4      6.2 GB     2.6 GB    0.4 GB             10.9 GB
Wan 2.1-14B 720p     bf16    26.1 GB    10.6 GB   11.6 GB             50.1 GB
Wan 2.1-14B 720p     fp8     13.0 GB     5.3 GB   11.6 GB             31.7 GB
Wan 2.1-14B 720p     nf4      7.3 GB     3.0 GB   11.6 GB             23.7 GB
HunyuanVideo 720p    bf16    23.7 GB    16.0 GB   10.9 GB             52.5 GB
HunyuanVideo 720p    fp8     11.8 GB     8.0 GB   10.9 GB             32.7 GB
HunyuanVideo 720p    nf4      6.7 GB     4.5 GB   10.9 GB             24.0 GB
```

Three readings:

- **FLUX's weights are the hard limit**: in bf16 the denoising network plus T5 is 33 GB, which a 24 GB card cannot hold; FP8 takes it to 17 GB, which just fits in 24 GB; NF4 takes it to 9 GB, which runs on 16 GB.
- **An image model's activations are negligible and a video model's are not**: one Wan 720p step's activations come to over ten gigabytes on this rough estimate. They are proportional to the token count, and video's is more than 16 times an image's. A video model not fitting is usually about the activations rather than the weights.
- **The text encoder is a temporary tenant**: used once at the start, after which every byte it holds is waste. It is the first thing model offload moves away.

!!! note "An LLM's memory budget does not apply here"
    The bulk of memory in LLM inference is the KV cache, which grows with the concurrent requests and the sequence length, hence paging, preemption and computing how much concurrency can be served. A diffusion model has no such item: a step's activations are released as soon as they are used and concurrent requests share no state, so the memory plan is static. It either fits or it does not, and what makes it fit is this chapter's techniques rather than scheduling.

## Three kinds of offload: how much they move and how much they cost {#三种-offload各搬多少拖慢多少}

Put what does not fit in CPU memory and move it in when it is needed. The key is counting **how many bytes cross PCIe per generation**:

```python
PCIE = {"PCIe 4.0 x16": 25e9, "PCIe 5.0 x16": 50e9}     # the bandwidth actually available, in bytes per second (about 80% of the specification)

def offload_cost(w_dn, w_te, steps, compute_per_step, bw):
    """三种策略：每次生成搬运的字节数与额外时间"""
    model = w_te * 2 + w_dn * 2                          # model offload: each component moved in once and out once
    seq = w_te + w_dn * steps                            # per-layer offload: the denoising network moves all of its weights every step
    return {"模型卸载（按组件）": (model, model / bw),
            "逐层卸载（无预取）": (seq, seq / bw),
            "逐层卸载 + 预取": (seq, max(0.0, seq / bw - steps * compute_per_step))}   # the transfer hides behind the computation

w_dn, w_te = 11.9e9 * 2, 4.9e9 * 2                       # FLUX bf16
steps, t_step = 28, 0.17                                 # about 0.17 s per step on an H100 (see the previous chapter)
print(f"FLUX.1-dev bf16，28 步，一步计算约 {t_step} s，纯计算 {steps * t_step:.1f} s")
for bus, bw in PCIE.items():
    print(f"  {bus}")
    for name, (nbytes, extra) in offload_cost(w_dn, w_te, steps, t_step, bw).items():
        print(f"    {name:<14} 搬运 {nbytes / 1e9:>6.0f} GB   额外 {extra:>6.1f} s   一张图 {steps * t_step + extra:>6.1f} s")
```

```text title="output"
FLUX.1-dev bf16，28 步，一步计算约 0.17 s，纯计算 4.8 s
  PCIe 4.0 x16
    模型卸载（按组件）      搬运     67 GB   额外    2.7 s   一张图    7.4 s
    逐层卸载（无预取）      搬运    676 GB   额外   27.0 s   一张图   31.8 s
    逐层卸载 + 预取      搬运    676 GB   额外   22.3 s   一张图   27.0 s
  PCIe 5.0 x16
    模型卸载（按组件）      搬运     67 GB   额外    1.3 s   一张图    6.1 s
    逐层卸载（无预取）      搬运    676 GB   额外   13.5 s   一张图   18.3 s
    逐层卸载 + 预取      搬运    676 GB   额外    8.8 s   一张图   13.5 s
```

- **Model offload**: tens of gigabytes per generation, a few seconds more. Acceptable for interactive generation and not for a throughput service, but it only requires the CPU memory to hold it and carries almost no engineering risk, which is what diffusers' `enable_model_cpu_offload()` does by default.
- **Per-layer offload**: every step takes all 24 GB over PCIe, so 28 steps is 670 GB, 27 seconds of transfer alone, 5 times slower than the computation. `enable_sequential_cpu_offload()` can run FLUX on an 8 GB card, and this is what it costs.
- **Per-layer offload with prefetching**: another CUDA stream moves layer $i+1$ in while layer $i$ computes. As long as moving one layer is no slower than computing one, the transfer hides entirely behind the computation. One FLUX layer is about 420 MB, 17 ms on PCIe 4.0, while computing one layer on an H100 is about 3 ms, so **it cannot be hidden**, and the prefetch row above still carries over twenty seconds of extra time. On a 4090 (about 18 ms per layer) it nearly can. Prefetching requires the CPU memory to be **pinned**, or DMA cannot be used and the bandwidth halves or worse (see [Pinned memory, DMA and NUMA](cs://os/pinned-numa/)).

One variable that is easily forgotten is **the gap between memory bandwidth and PCIe bandwidth**: an H100's HBM is 3.35 TB/s and PCIe 4.0 is only 25 GB/s, a factor of 130. So offloading to the CPU is always a last resort rather than an optimisation; its proper use is to move away what **will not be used again** (the text encoder) and what is **cold** (rarely used LoRAs, ControlNets).

Change the model, the card and the bus, and see what each of the three offloads costs in seconds:

<div class="aig-widget" data-widget="offload-cost"></div>

## Group offload: in between {#分组卸载在两者之间}

Diffusers' group offloading splits the difference between by-component and by-layer: the denoising network is divided into groups of layers, one group is resident and the rest in the CPU, and a stream prefetches the next. The larger the group, the easier it is to hide the prefetch (the group computes for longer), and the more memory stays resident:

```python
layers, per_layer_bytes, t_layer = 57, 11.9e9 * 2 / 57, 0.17 / 57       # FLUX on an H100
bw = 25e9
print(f"{'每组层数':>6} {'常驻显存':>8} {'搬一组':>8} {'算一组':>8}  能否藏住")
for g in (1, 4, 8, 19, 57):
    resident = g * per_layer_bytes * 2                                   # the current group plus the next one being prefetched
    t_move, t_comp = g * per_layer_bytes / bw, g * t_layer
    print(f"{g:>6} {resident / GB:>6.1f} GB {t_move * 1e3:>6.0f} ms {t_comp * 1e3:>6.0f} ms  {'能' if t_move <= t_comp else '不能'}")
print("H100 算得太快，哪种分组都藏不住搬运；慢一些的卡反而能藏住——卸载是给消费级卡准备的手段")
```

```text title="output"
  每组层数     常驻显存      搬一组      算一组  能否藏住
     1    0.8 GB     17 ms      3 ms  不能
     4    3.1 GB     67 ms     12 ms  不能
     8    6.2 GB    134 ms     24 ms  不能
    19   14.8 GB    317 ms     57 ms  不能
    57   44.3 GB    952 ms    170 ms  不能
H100 算得太快，哪种分组都藏不住搬运；慢一些的卡反而能藏住——卸载是给消费级卡准备的手段
```

## Weight precision: FP8 and NF4 {#权重精度fp8-与-nf4}

Quantization gets a [chapter of its own](quantization.md); here is only the rough accounting of memory and speed:

| Precision | FLUX's denoising network | Speed (against bf16) | Quality | Suits |
| --- | --- | --- | --- | --- |
| bf16 | 24 GB | 1x | the original | 48 GB and above |
| FP8 (e4m3, scaled per tensor or per block) | 12 GB | about 1.2 to 1.6x on Hopper, Ada and later; older cards fall back to bf16 computation and save memory alone | nearly lossless | 24 GB |
| NF4 / INT4 weight quantization | 6.5 GB | 0.6 to 0.9x (the dequantization overhead) | a slight loss | 16 GB and below, where running at all is the point |
| SVDQuant (W4A4 with a low-rank correction) | 6.5 GB | 2 to 3x | near lossless | needs a dedicated kernel (Nunchaku) |

Remember one thing: **weight quantization is a memory technique first in a diffusion model**. A step is compute-bound, so shrinking the weights alone does not speed the matrix multiplies up; speed requires the computation itself to happen in low precision (FP8 Tensor Cores, a W4A4 kernel).

## How to configure a 16 GB card {#一张-16-gb-的卡怎么配}

Putting this chapter's accounting together for FLUX.1-dev on a 16 GB card like an RTX 5070 Ti or 4080:

```python
BUDGET = 16 * GB - 1.5 * GB                              # with the context and fragmentation deducted
plans = [
    ("bf16 全部常驻",                 11.9e9 * 2 + 4.9e9 * 2 + 0.08e9 * 2, 0),
    ("去噪 FP8，T5 FP8 常驻",          11.9e9 * 1 + 4.9e9 * 1 + 0.08e9 * 2, 0),
    ("去噪 FP8，T5 编码完卸载",         11.9e9 * 1 + 0.08e9 * 2, 2 * 4.9e9 / 25e9),
    ("去噪 NF4，T5 NF4 常驻",          11.9e9 * 0.5625 + 4.9e9 * 0.5625 + 0.08e9 * 2, 0),
    ("去噪 bf16 分组卸载(8 层)，T5 卸载", 8 * 2 * 11.9e9 * 2 / 57 + 0.08e9 * 2, 28 * 11.9e9 * 2 / 25e9 + 2 * 4.9e9 / 25e9),
]
act = activation_peak(3072, 4608, 57)
print(f"预算 {BUDGET / GB:.1f} GB，一步激活 {act / GB:.1f} GB，VAE 分块解码后峰值可压到 1 GB 以内")
for name, weights, extra_s in plans:
    need = weights + act + 1 * GB
    print(f"  {name:<28} 需要 {need / GB:>5.1f} GB  {'放得下' if need <= BUDGET else '放不下'}  每张图额外搬运 {extra_s:>5.1f} s")
```

```text title="output"
预算 14.5 GB，一步激活 0.4 GB，VAE 分块解码后峰值可压到 1 GB 以内
  bf16 全部常驻                    需要  32.9 GB  放不下  每张图额外搬运   0.0 s
  去噪 FP8，T5 FP8 常驻             需要  17.2 GB  放不下  每张图额外搬运   0.0 s
  去噪 FP8，T5 编码完卸载              需要  12.7 GB  放得下  每张图额外搬运   0.4 s
  去噪 NF4，T5 NF4 常驻             需要  10.4 GB  放得下  每张图额外搬运   0.0 s
  去噪 bf16 分组卸载(8 层)，T5 卸载      需要   7.8 GB  放得下  每张图额外搬运  27.0 s
```

The conclusion is direct: on a 16 GB card, **the denoising network in FP8 with T5 offloaded after encoding** is the optimum. It fits, it barely costs quality, and each image only moves T5 once. NF4 also fits but is slower; group offload can run bf16 but adds over twenty seconds per image and is only for when bf16 is mandatory. This is the logic behind every vendor's low-memory mode.

!!! interview "How to answer in an interview"
    Asked how to run FLUX on a 24 GB or 16 GB card, give the accounting first: the bf16 denoising network is 24 GB plus T5's 9.5, FP8 halves each and NF4 halves again. Then each policy's cost: model offload moves tens of gigabytes per generation for a few extra seconds, per-layer offload moves all of the weights every step and is several times slower, and prefetching can hide it but not on a card as fast as an H100, and it needs pinned memory for DMA. Then the plan: the denoising network in FP8 or NF4, T5 offloaded after use, the VAE tiled. Add the contrast with an LLM: no KV cache, so the memory plan is static and what makes it fit is these techniques rather than scheduling.

## Exercises {#练习}

1. Use this chapter's functions for HunyuanVideo (12.7B, with an 8.6B text encoder) on one 80 GB H100: does it all fit resident in bf16? How large are one step's activations at 720p over 129 frames? If it does not fit, what do you address first?

??? success "Answer"
    The weights are about 43 GB in bf16, which fits; but one step's activations exceed 20 GB on the rough estimate (119,000 tokens x 3072 dimensions x 8 copies), and with the VAE decode's peak on top, 80 GB is very tight. What to address first is the activations rather than the weights: chunked attention and sequence parallelism spread the tokens over several cards (see the chapter on multi-GPU parallelism), and offloading the text encoder after it has encoded saves another 17 GB.

2. Replace `compute_per_step` in `offload_cost` with a 4090's value (about 1.0 s per FLUX step). Does per-layer offload with prefetching still cost extra time? Explain why the same policy gives the opposite conclusion on different cards.

??? success "Answer"
    On a 4090, 28 steps compute for 28 seconds against 27 seconds of transfer (PCIe 4.0), so with prefetching the extra time is near zero: the transfer hides almost entirely behind the computation. The reason is that prefetching can only hide what is no slower than one layer's computation, so the slower the card and the longer a layer takes, the easier it is to hide; an H100 computes too fast and the transfer becomes the bottleneck. So offload with prefetching is a technique for consumer cards, while a data-centre card should use FP8 or several GPUs.

3. A service deploys FLUX alongside three ControlNets (about 3.6 GB each in bf16), and each user uses only one of them. How do you arrange the memory?

??? success "Answer"
    The ControlNets are cold weights: all three resident is 11 GB while only one is used at a time. Put them in pinned CPU memory and move one into the GPU on demand when a request arrives (3.6 GB is about 0.15 seconds over PCIe 4.0), keeping the most recently used one resident (LRU); or route requests by ControlNet type to different instances so that each keeps one resident. This has the same root as multi-LoRA serving for an LLM (see the chapter on [Multiple LoRAs and ControlNet](../serving/lora-controlnet.md)).

## Summary {#小结}

- [x] A diffusion model's memory is the weights (denoising network, text encoder, VAE) plus one step's activations plus the VAE decode's peak plus the overhead; there is no KV cache, so the plan is static.
- [x] An image model's activations are negligible while a video model's are proportional to the token count and may exceed the weights; the text encoder should be moved away as soon as it is done.
- [x] Model offload moves tens of gigabytes per generation for a few extra seconds; per-layer offload moves all of the weights every step and is several times slower; prefetching can hide the transfer, but only on a slower card and only with pinned memory.
- [x] Weight quantization is a memory technique first: FP8 halves it and speeds newer cards up, NF4 halves it again but is slower; running FLUX on 16 GB means the denoising network in FP8, T5 offloaded after use, and the VAE tiled.
