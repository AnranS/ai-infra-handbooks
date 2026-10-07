# Kernel acceleration: attention, compilation and fusion

<p class="lead">Without changing the model or taking one step fewer, getting each step's kernels right alone makes SDXL and FLUX 1.5 to 3 times faster, which is the most nearly free layer of all the accelerations. This chapter covers three things: the attention kernel (why FlashAttention and SageAttention are especially effective for diffusion models), torch.compile and operator fusion (dealing with the memory-bound small operations of AdaLN, normalisation and activation functions), and CUDA graphs (the natural advantage of diffusion's fixed shapes). Each one's "what it computes and what it saves" is worked out on a CPU first, then given its typical gain on a real card.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. How much memory and traffic do naive attention and FlashAttention each take? What does that mean for a video model's hundred thousand tokens?
    2. What does SageAttention quantize to INT8 or FP8? Why can a diffusion model tolerate that precision?
    3. Which operators in a DiT block are memory-bound? How much of the time do they take? How does torch.compile handle them?
    4. What does a CUDA graph save? Why does it suit a diffusion model better than an LLM?
    5. With compilation on, why is the first image especially slow, and slow again on a different resolution? What do you do about it?

??? success "Answers for the self-test (answer first, then open this)"
    1. Naive attention writes the $N \times N$ score matrix to memory and reads it back for the softmax, which is $O(N^2)$ memory and $O(N^2)$ traffic; FlashAttention computes the softmax and the weighted sum in blocks on chip, which is $O(N)$ memory and traffic down to $O(N^2 d / M)$ (with $M$ the on-chip cache). At a hundred thousand tokens the $N^2$ matrix is 20 GB for a single head: without FlashAttention a video model simply cannot run.
    2. $Q$ and $K$ are quantized to INT8 (after smoothing by block) and $P$ and $V$ accumulate in FP8 or FP16, so the matrix multiplies use INT8 or FP8 Tensor Cores and run 2 to 3 times faster than FlashAttention-2. A diffusion model's output at each step is a direction rather than an exact value, and dozens of denoising steps average the per-step error out, so attention's quantization error is invisible.
    3. LayerNorm and RMSNorm, AdaLN's scale and shift, GELU and SiLU, the residual additions and RoPE. Their FLOPs are few but they read and write the whole activation, and uncompiled they take 30% to 50% of a step's time. torch.compile (Inductor) fuses neighbouring elementwise operations into one kernel for a single read and write, and also removes the Python and kernel-launch overhead.
    4. It saves the CPU-side kernel launch overhead: a step launches hundreds to thousands of kernels, and a few microseconds each adds up to a great deal, up to half a step for a small model at a low resolution. A diffusion model's every step has exactly the same shape (no growing KV), so one graph is captured once and replayed dozens of times, far easier than an LLM's need for several graphs across batch sizes and sequence lengths.
    5. Compilation is per shape: the first time a shape appears it has to compile (tens of seconds to minutes), and a different resolution or batch is a new shape. The remedies: fix the resolutions the service supports, warm the compilation up at startup, enable Inductor's persistent cache, and use `dynamic=False` to avoid repeated recompilation on dynamic shapes.

## Attention: from $O(N^2)$ memory to blocks {#注意力从-on2-显存到分块}

![Figure: FlashAttention, with blocking and an online softmax, never materialising the N x N attention matrix](../assets/figures/flash-attention.svg){.aig-svg}

Naive attention computes the whole of $S = QK^\top$ and multiplies by $V$ after the softmax. When $N$ is large that $N \times N$ matrix is itself a disaster. The arithmetic first:

```python
def attn_bytes(n, d_head, heads, dtype=2):
    """一层注意力的显存与访存（字节）：朴素实现要落地 N×N 的分数矩阵"""
    scores = n * n * heads * dtype                       # the result of QKᵀ, one per head
    qkv = 3 * n * heads * d_head * dtype
    naive = scores * 2 + qkv + n * heads * d_head * dtype   # write the scores, read them back for the softmax, write P
    flash = qkv + n * heads * d_head * dtype             # read QKV and write the output only; the scores stay on chip
    return scores, naive, flash

GB = 1024 ** 3
print(f"{'模型':<26} {'token':>8} {'N×N 分数矩阵':>12} {'朴素访存':>10} {'Flash 访存':>10}")
for name, n, dh, h in [("FLUX 1024²", 4608, 128, 24), ("FLUX 2048²", 16896, 128, 24),
                       ("Wan 2.1 720p", 76112, 128, 40), ("HunyuanVideo 720p", 119056, 128, 24)]:
    s, nv, fl = attn_bytes(n, dh, h)
    print(f"{name:<26} {n:>8,} {s / GB:>10.1f} GB {nv / GB:>8.1f} GB {fl / GB:>8.2f} GB")
```

```text title="output"
模型                            token     N×N 分数矩阵       朴素访存   Flash 访存
FLUX 1024²                    4,608        0.9 GB      2.0 GB     0.11 GB
FLUX 2048²                   16,896       12.8 GB     25.9 GB     0.39 GB
Wan 2.1 720p                 76,112      431.6 GB    866.1 GB     2.90 GB
HunyuanVideo 720p           119,056      633.6 GB   1270.0 GB     2.72 GB
```

One HunyuanVideo layer's score matrix is 630 GB: a naive implementation is impossible on any card. FlashAttention computes the softmax for each block of $Q$ against each block of $K$ and $V$ on chip (an online softmax, see [FlashAttention](cuda://advanced/attention/)), so the score matrix is never materialised, the memory goes from $O(N^2)$ to $O(N)$ and the traffic falls two orders of magnitude. **For a diffusion model it is not an optimisation but a precondition.**

In PyTorch, `F.scaled_dot_product_attention` selects the FlashAttention or memory-efficient backend automatically. A small example confirms that the three forms agree and shows the intermediate tensors' sizes:

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, H, N, D = 1, 4, 256, 32
q, k, v = (torch.randn(B, H, N, D) for _ in range(3))

def naive(q, k, v):
    s = (q @ k.transpose(-1, -2)) / D ** 0.5            # [B, H, N, N] materialised
    return s.softmax(-1) @ v, s.numel() * 4

def blocked(q, k, v, block=64):                           # blocked with an online softmax: the score matrix is only block x N
    out = torch.zeros_like(q)
    peak = 0
    for i in range(0, N, block):
        qi = q[:, :, i:i + block]
        m = torch.full((B, H, qi.shape[2], 1), float("-inf"))
        l = torch.zeros(B, H, qi.shape[2], 1)
        acc = torch.zeros(B, H, qi.shape[2], D)
        for j in range(0, N, block):
            s = (qi @ k[:, :, j:j + block].transpose(-1, -2)) / D ** 0.5
            m_new = torch.maximum(m, s.amax(-1, keepdim=True))
            p = (s - m_new).exp()
            l = l * (m - m_new).exp() + p.sum(-1, keepdim=True)
            acc = acc * (m - m_new).exp() + p @ v[:, :, j:j + block]
            m = m_new
            peak = max(peak, s.numel() * 4)
        out[:, :, i:i + block] = acc / l
    return out, peak

o1, bytes1 = naive(q, k, v)
o2, bytes2 = blocked(q, k, v)
o3 = F.scaled_dot_product_attention(q, k, v)
print(f"三种写法一致：{torch.allclose(o1, o2, atol=1e-5)} {torch.allclose(o1, o3, atol=1e-5)}")
print(f"朴素实现落地的分数矩阵 {bytes1 / 1024:.0f} KB，分块实现任一时刻只有 {bytes2 / 1024:.0f} KB（{bytes1 / bytes2:.0f} 倍）")
```

```text title="output"
三种写法一致：True True
朴素实现落地的分数矩阵 1024 KB，分块实现任一时刻只有 64 KB（16 倍）
```

### SageAttention: quantizing attention too {#sageattention把注意力也量化}

Attention's two matrix multiplies, $QK^\top$ and $PV$, are still FP16 or BF16 under FlashAttention. What SageAttention does: subtract the block means from $Q$ and $K$ (smoothing $K$'s outlier channels) and quantize them to INT8, putting $QK^\top$ on INT8 Tensor Cores, while $P$ and $V$ accumulate in FP8 (SageAttention2) or FP16. The result is 2 to 3 times faster than FlashAttention-2 with the same memory and no visible difference in the output.

Why it works for a diffusion model while an LLM has to be careful: a diffusion model's prediction at each step is which way to go, and dozens of iterations average the per-step error out; an LLM's decode has each step's output decide the next token directly, where the error is amplified by sampling. Attention is eighty percent of a video model's time, so SageAttention is all but their default backend.

## The memory-bound small operations: compilation and fusion {#访存受限的小算子编译与融合}

Besides the matrix multiplies and attention, a DiT block has a string of operations that compute little but read and write the whole activation. Listing their FLOPs and traffic:

```python
def block_profile(n, d, dtype=2):
    """一个 DiT 块里各算子的 FLOP 与访存字节数（粗估：逐元素算子读一次写一次）"""
    act = n * d * dtype
    ops = [
        ("AdaLN 调制 (×2)",      2 * 3 * n * d,            2 * 2 * act),
        ("RMSNorm (×2)",         2 * 4 * n * d,            2 * 2 * act),
        ("QKV 投影",             2 * n * d * 3 * d,        act + 3 * act + 3 * d * d * dtype),
        ("RoPE",                 6 * n * d,                2 * 2 * act),
        ("注意力 (Flash)",        4 * n * n * d,            4 * act),
        ("输出投影",             2 * n * d * d,            2 * act + d * d * dtype),
        ("残差相加 + 门控 (×2)",  2 * 2 * n * d,            2 * 3 * act),
        ("MLP 升维",             2 * n * d * 4 * d,        act + 4 * act + 4 * d * d * dtype),
        ("GELU",                 8 * n * 4 * d,            2 * 4 * act),
        ("MLP 降维",             2 * n * 4 * d * d,        4 * act + act + 4 * d * d * dtype),
    ]
    return ops

PEAK, BW = 989e12, 3.35e12                                 # the H100: bf16 throughput and memory bandwidth
n, d = 4608, 3072                                          # FLUX 1024²
print(f"{'算子':<22} {'GFLOP':>8} {'访存 MB':>8} {'强度':>6} {'受限于':>6} {'时间 μs':>8}")
tot = 0
for name, flops, byts in block_profile(n, d):
    t = max(flops / PEAK, byts / BW) * 1e6                 # the roofline: take the slower of the two
    tot += t
    print(f"{name:<22} {flops / 1e9:>8.1f} {byts / 2 ** 20:>8.0f} {flops / byts:>6.0f} {'算力' if flops / PEAK > byts / BW else '带宽':>6} {t:>8.0f}")
small = sum(max(f / PEAK, b / BW) for nm, f, b in block_profile(n, d) if nm.split()[0] in ("AdaLN", "RMSNorm", "RoPE", "残差相加", "GELU")) * 1e6
print(f"合计 {tot:.0f} μs，其中逐元素小算子 {small:.0f} μs（{small / tot:.0%}）——它们 FLOP 不到 1%，时间却占这么多")
```

```text title="output"
算子                        GFLOP    访存 MB     强度    受限于    时间 μs
AdaLN 调制 (×2)               0.1      108      1     带宽       34
RMSNorm (×2)                0.1      108      1     带宽       34
QKV 投影                    260.9      162   1536     算力      264
RoPE                        0.1      108      1     带宽       34
注意力 (Flash)               260.9      108   2304     算力      264
输出投影                       87.0       72   1152     算力       88
残差相加 + 门控 (×2)              0.1      162      0     带宽       51
MLP 升维                    347.9      207   1603     算力      352
GELU                        0.5      216      2     带宽       68
MLP 降维                    347.9      207   1603     算力      352
合计 1539 μs，其中逐元素小算子 220 μs（14%）——它们 FLOP 不到 1%，时间却占这么多
```

These are the ideal values under the roofline model (without kernel launches or the L2), but the conclusion holds: **the small operations are under 1% of the FLOPs and over a tenth of the time**; in an uncompiled implementation the share is higher still, because each small operation is its own kernel with its own launch and its own reads and writes. What torch.compile (Inductor) does:

- **Fusion**: neighbouring elementwise operations (modulation, normalisation, activation) become one kernel and the activation is read and written once.
- **Removing the Python overhead**: a step's hundreds of operations dispatched from Python become one block of generated code.
- **Choosing better matrix-multiply configurations** (`mode="max-autotune"` tries several kernels per shape).

The typical gains: 1.3 to 1.8 times for a DiT like FLUX or SD3, 1.2 to 1.5 for SDXL's UNet. The price is **compiling per shape**: the first sight of a shape takes tens of seconds to minutes, and a different resolution or batch takes another. What services do: fix the list of supported resolutions, warm all of them at startup, set `TORCHINDUCTOR_CACHE_DIR` for a persistent cache, and use `dynamic=False`.

## CUDA graphs: the dividend of a fixed shape {#cuda-graph固定形状的红利}

How large the launch overhead of a thousand small kernels is, and how much a CUDA graph and fusion each save, dialled with the same time model as in the inference-systems handbook:

<div class="aig-widget" data-widget="launch-overhead"></div>

One denoising step launches hundreds to thousands of kernels, and a few microseconds of CPU-side overhead each can be half a step for a small model at a low resolution, with the GPU waiting for the CPU to issue work. A CUDA graph captures all of a step's kernels and submits the whole graph at once (see [CUDA graphs and torch.compile](serving://engine/graphs-compile/)).

A diffusion model is the ideal user of CUDA graphs: **every step has exactly the same shape**. An LLM needs a graph per batch size and sequence length and has to handle a growing KV, while a diffusion model needs one (guidance's two paths are a fixed batch too). What it is worth:

```python
def step_time(kernels, launch_us, gpu_us, graph):
    cpu = 0 if graph else kernels * launch_us                # with a graph: the launch overhead is nearly zero
    return max(cpu, gpu_us) if not graph else gpu_us + 10     # without a graph: the GPU idles when the CPU cannot issue fast enough

print(f"{'情形':<30} {'kernel 数':>8} {'GPU 计算':>9} {'无 graph':>9} {'有 graph':>9}  收益")
for name, kernels, gpu in [("SD 1.5 512²（H100）", 600, 8000), ("SDXL 1024²（H100）", 900, 45000),
                           ("FLUX 1024²（H100）", 1200, 170000), ("SD 1.5 512² · batch 4 · 小卡", 600, 30000)]:
    a, b = step_time(kernels, 30, gpu, False), step_time(kernels, 30, gpu, True)
    print(f"{name:<30} {kernels:>8} {gpu / 1e3:>7.1f} ms {a / 1e3:>7.1f} ms {b / 1e3:>7.1f} ms  {a / b:>4.2f}×")
print("（PyTorch eager 里每个算子的 CPU 侧开销（Python + 分发 + 启动）按 30 μs 估；GPU 快、模型小时 CPU 发不过来，graph 的收益最大）")
```

```text title="output"
情形                             kernel 数    GPU 计算   无 graph   有 graph  收益
SD 1.5 512²（H100）                   600     8.0 ms    18.0 ms     8.0 ms  2.25×
SDXL 1024²（H100）                    900    45.0 ms    45.0 ms    45.0 ms  1.00×
FLUX 1024²（H100）                   1200   170.0 ms   170.0 ms   170.0 ms  1.00×
SD 1.5 512² · batch 4 · 小卡          600    30.0 ms    30.0 ms    30.0 ms  1.00×
（PyTorch eager 里每个算子的 CPU 侧开销（Python + 分发 + 启动）按 30 μs 估；GPU 快、模型小时 CPU 发不过来，graph 的收益最大）
```

A large model like FLUX takes 170 ms per step, so those thirty-odd milliseconds on the CPU are entirely covered by the GPU's computation; a small model like SD 1.5 on a fast card takes only 8 ms of GPU time against 18 ms of CPU issuing, so the GPU waits half the time, and a graph cuts the step by more than half. torch.compile also lowers each operation's CPU overhead, so the graph's additional gain on top of it is smaller. So CUDA graphs pay most for small models on fast cards in interactive use (real-time painting, few-step distilled models), and are a bonus for large models. torch.compile's `mode="reduce-overhead"` uses them automatically.

## Stacking them {#把它们叠起来}

The three act in different places and stack:

| Technique | What it acts on | Typical gain | Cost |
| --- | --- | --- | --- |
| FlashAttention (`sdpa`) | attention's memory and traffic | from unable to run to running; 2 to 4x on long sequences | none |
| SageAttention | attention's matrix-multiply precision | 2 to 3x on the attention, 1.5 to 2x overall for a video model | needs the kernel package; a few models need tuning |
| torch.compile | elementwise small operations, Python overhead, matrix-multiply configuration | 1.3 to 1.8x for a DiT | compiles per shape, slow the first time, needs warming and caching |
| CUDA graphs (reduce-overhead) | kernel launch overhead | 1.5 to 2x for a small model on a fast card, a few percent for a large one | the shape has to be fixed, and it holds an extra copy of memory |
| channels_last (UNet) | the convolutions' memory layout | 1.1 to 1.3x for SDXL | none |
| Fused QKV / fused AdaLN | the linear layers' kernel count | a few percent | the model code has to change |

The suggested order: **sdpa, compile, SageAttention (mandatory for video), reduce-overhead (small models)**. The first two cost almost nothing and the rest depend on the model and the case.

!!! interview "How to answer in an interview"
    Asked how to accelerate diffusion inference without changing the model, answer by layer. The attention layer: FlashAttention keeps the $N^2$ score matrix off memory, which is a precondition for video models; SageAttention quantizes $QK^\top$ to INT8, which works because diffusion averages its per-step error, giving 1.5 to 2x for a video model. The operator layer: AdaLN, normalisation and activation functions are under 1% of the FLOPs but a tenth or two of the time, and torch.compile fuses them for 1.3 to 1.8x on a DiT. The launch layer: a CUDA graph removes the launch overhead, and diffusion's fixed per-step shape means one graph covers everything, paying most for a small model on a fast card. Finish with the cost: compilation is per shape, so a service fixes its resolutions, warms up and caches persistently.

## Exercises {#练习}

1. Change `blocked`'s `block` from 64 to 16 and then 256. What is the score matrix's peak in each case? Why does a real FlashAttention not make the block as small as possible?

??? success "Answer"
    The peak is proportional to block x N: a quarter of 64's at 16 and 4 times at 256. But the smaller the block, the more often $K$ and $V$ are read from memory into the chip (traffic $\propto N^2 d / \text{block}$), and small blocks do not fill the Tensor Cores. Real implementations set the block at the size where the on-chip SRAM just holds the $Q$, $K$ and $V$ blocks and the intermediates, which is 64 to 128.

2. Use `block_profile` for Wan 2.1 at 720p ($n = 76112$, $d = 5120$) to get one block's time breakdown. How does the small operations' share compare with FLUX at 1024²? Why?

??? success "Answer"
    Attention's $4N^2 d$ term grows with $N^2$ and becomes the outright bulk (over eighty percent), with the small operations' share down to single digits. The larger the token count, the more attention dominates, the smaller fusion's relative gain and the larger the attention kernel's. That is why video models treat SageAttention as mandatory and compilation as a bonus.

3. A service supports 512², 768² and 1024² at batches of 1 to 4, with `torch.compile(dynamic=False)` on. How many shapes have to be warmed at startup? And if users can pass arbitrary aspect ratios?

??? success "Answer"
    3 resolutions x 4 batch sizes (guidance's concatenation doubles the batch, but fixedly) = 12 shapes, each taking tens of seconds to compile, so warming up takes ten-odd minutes at startup and a persistent cache is needed so that the second start reuses it. Arbitrary aspect ratios means the shapes cannot be enumerated: either align the resolution to a few fixed sizes (what most services do), accept some performance loss with `dynamic=True` on the backbone, or fall back to an uncompiled path for unusual shapes.

## Summary {#小结}

- [x] FlashAttention keeps the $N \times N$ score matrix off memory: $O(N)$ memory and two orders of magnitude less traffic, which is a precondition rather than an optimisation for video models.
- [x] SageAttention quantizes $QK^\top$ to INT8 or FP8 for 2 to 3x; diffusion's many iterations average the per-step error out, so it can bear it.
- [x] AdaLN, normalisation and activation functions are under 1% of the FLOPs but a tenth or two of the time, and torch.compile fuses them for 1.3 to 1.8x on a DiT; the price is compiling per shape, so a service fixes its resolutions, warms up and caches persistently.
- [x] A CUDA graph removes the kernel launch overhead, and diffusion's fixed per-step shape means one graph covers everything, paying most for a small model on a fast card.
