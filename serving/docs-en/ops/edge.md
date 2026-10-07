# On-device inference: llama.cpp and GGUF, MLX, ExecuTorch

<p class="lead">Server-side inference pursues "as many requests per GPU as possible"; on-device (laptops, phones, edge devices) serves a single user, with entirely different constraints: a few GB to tens of GB of memory shared with the system, bandwidth of tens to hundreds of GB/s, plus power and heat. This chapter looks at how three mainstream on-device frameworks each work, focusing on llama.cpp's GGUF quantization formats (why some "4-bit" formats are clearly more accurate than others) and how to estimate how fast a device can go.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. What sets the speed limit of on-device decode? Roughly how fast is an 8B model on a phone?
    2. How do GGUF's Q4_0, Q4_1 and Q4_K differ? Why is Q4_K more accurate?
    3. Which platforms do llama.cpp, MLX and ExecuTorch each suit?
    4. What limits on-device prefill and decode respectively? Which one can an NPU help with?

??? success "Answers (try first, then expand to compare)"
    1. Single-user decode reads all the weights every step, so the speed limit = effective bandwidth ÷ the quantized weights' bytes. A 4-bit 8B model is about 4.5 GB, and a phone has about 60 GB/s (about 70% effective), giving under 10 tokens per second.
    2. Q4_0: blocks of 32 with one fp16 scale, symmetric 4-bit; Q4_1: adds an fp16 minimum for asymmetry, 5 bits per weight; Q4_K: superblocks of 256 with 8 sub-blocks that each have their own scale and minimum, and those two numbers are themselves quantized to 6 bits and restored by the superblock's fp16 scales. Two levels of scaling achieve "asymmetric per sub-block" accuracy at an overhead of 4.5 bits.
    3. llama.cpp: every platform (CPU, all kinds of GPUs, phones), with the broadest ecosystem; MLX: Apple Silicon, using unified memory; ExecuTorch: exports PyTorch models as graphs and delegates to hardware backends such as a phone's NPU, suited to embedding in apps.
    4. Prefill is compute-bound, decode memory-bound. A phone's NPU has considerable compute and can speed up prefill; decode's bottleneck is memory bandwidth shared with the CPU / GPU, which the NPU can't help much with. So a common division is the NPU doing prefill and the CPU / GPU doing decode.

## The speed limit: bandwidth ÷ the weights' bytes {#速度上限带宽--权重字节数}

Single-user decode reads all the weights every step with a batch of 1, entirely memory-bound. So the two most important numbers on-device are **memory bandwidth** and **the model's quantized bytes**: the former set by hardware, the latter by the quantization format.

## GGUF's quantization formats {#gguf-的量化格式}

In llama.cpp's GGUF files, weights are quantized in blocks, each with its own scale. Several common formats:

- **Q8_0**: blocks of 32 weights, one fp16 scale, symmetric int8 quantization, 8.5 bits per weight;
- **Q4_0**: blocks of 32, one fp16 scale, symmetric 4-bit (the scale set from the largest-magnitude value), 4.5 bits per weight;
- **Q4_1**: blocks of 32, an fp16 scale + an fp16 minimum, asymmetric 4-bit, 5 bits per weight;
- **Q4_K** (K-quants): superblocks of 256 split into 8 sub-blocks of 32; each sub-block has its own scale and minimum, but those two numbers are themselves quantized to 6 bits and restored together by the superblock's fp16 scales, so two levels of scaling squeeze "asymmetric per sub-block" accuracy into an overhead of 4.5 bits.

Compare them on long-tailed weights (with Q4_K implemented in simplified form along the lines above):

```python
import torch

torch.manual_seed(0)
w = torch.distributions.StudentT(5.0).sample((4096 * 256,)) * 0.02     # long-tailed weights, 1M of them


def q8_0(x):                                    # blocks of 32, an fp16 scale, symmetric int8
    b = x.view(-1, 32)
    d = b.abs().amax(1, keepdim=True) / 127
    return ((b / d).round().clamp(-127, 127) * d).flatten(), 8 + 16 / 32


def q4_0(x):                                    # blocks of 32, an fp16 scale, symmetric 4-bit (offset 8)
    b = x.view(-1, 32)
    idx = b.abs().argmax(1, keepdim=True)
    d = b.gather(1, idx) / -8                   # set the scale from the largest-magnitude value (signed) so it lands on -8
    return ((b / d).round().clamp(-8, 7) * d).flatten(), 4 + 16 / 32


def q4_1(x):                                    # blocks of 32, an fp16 scale + an fp16 minimum, asymmetric 4-bit
    b = x.view(-1, 32)
    lo, hi = b.amin(1, keepdim=True), b.amax(1, keepdim=True)
    d = (hi - lo) / 15
    return (((b - lo) / d).round().clamp(0, 15) * d + lo).flatten(), 4 + 32 / 32


def q4_k(x):                                    # superblocks of 256 split into 8 sub-blocks of 32: each sub-block's scale and minimum are themselves quantized to 6 bits
    b = x.view(-1, 8, 32)
    lo, hi = b.amin(2, keepdim=True).clamp(max=0), b.amax(2, keepdim=True)
    sc, mn = (hi - lo) / 15, -lo
    dsc, dmn = sc.amax(1, keepdim=True) / 63, mn.amax(1, keepdim=True) / 63                 # the superblock's fp16 scales
    sc_q, mn_q = (sc / dsc).round().clamp(1, 63) * dsc, (mn / dmn).round().clamp(0, 63) * dmn
    q = ((b + mn_q) / sc_q).round().clamp(0, 15)
    return (q * sc_q - mn_q).flatten(), 4 + (6 + 6) * 8 / 256 + 2 * 16 / 256


for name, fn in (("Q8_0", q8_0), ("Q4_0", q4_0), ("Q4_1", q4_1), ("Q4_K（简化）", q4_k)):
    deq, bpw = fn(w)
    print(f"{name:<11} {bpw:.2f} 比特/权重，相对误差 {((deq - w).norm() / w.norm()).item():.4f}")

print("8B 模型 Q4_K（约 4.5 GB）在不同设备上 decode 的速度上限（带宽 × 70% / 权重字节数）：")
for dev, bw in (("手机（LPDDR5X，约 60 GB/s）", 60e9), ("轻薄本（约 120 GB/s）", 120e9), ("M 系列 Max 芯片（约 400 GB/s）", 400e9), ("RTX 4090（约 1 TB/s）", 1008e9)):
    print(f"  {dev}：约 {bw * 0.7 / 4.5e9:.0f} token/s")
```

```text title="输出"
Q8_0        8.50 比特/权重，相对误差 0.0068
Q4_0        4.50 比特/权重，相对误差 0.1079
Q4_1        5.00 比特/权重，相对误差 0.0901
Q4_K（简化）    4.50 比特/权重，相对误差 0.0903
8B 模型 Q4_K（约 4.5 GB）在不同设备上 decode 的速度上限（带宽 × 70% / 权重字节数）：
  手机（LPDDR5X，约 60 GB/s）：约 9 token/s
  轻薄本（约 120 GB/s）：约 19 token/s
  M 系列 Max 芯片（约 400 GB/s）：约 62 token/s
  RTX 4090（约 1 TB/s）：约 157 token/s
```

- Q4_K reaches Q4_1's accuracy (5 bits) at the same 4.5 bits as Q4_0: two levels of scaling push the metadata overhead down;
- In practice, mixed-precision "recipes" are common (for example Q4_K_M, using higher precision for more sensitive layers, attention's V projection and the FFN's down projection), along with an imatrix that measures each weight's importance from calibration data to reduce error further;
- The speed limit follows directly from bandwidth: the same 8B model gets under 10 tokens per second on a phone and over sixty on a laptop with a Max chip. Going faster on a phone requires a smaller model, fewer bits, or speculative decoding (a smaller draft model verifying several tokens at once, amortizing memory-bound decode).

## Three frameworks {#三个框架}

| Framework | Platforms | Approach |
| --- | --- | --- |
| **llama.cpp** (and Ollama and others built on it) | almost every platform: x86 / ARM CPUs, Apple Metal, CUDA, Vulkan and more | the C / C++ tensor library ggml; the single-file GGUF format (weights + tokenizer + config) mapped directly with mmap; a rich set of quantization formats; quantized matrix multiplies with SIMD on CPUs |
| **MLX** | Apple Silicon | Apple's array framework: unified memory (CPU and GPU share the same memory, with no copies), lazy evaluation, and a NumPy / PyTorch-like interface; mlx-lm provides model conversion, quantization and inference |
| **ExecuTorch** | phones and embedded devices (Android, iOS, microcontrollers) | PyTorch's official on-device solution: `torch.export` exports the computation graph, compiled into a `.pte` file with a tiny runtime; "delegates" hand subgraphs to hardware backends such as XNNPACK (CPU), Core ML and Qualcomm QNN |

## Problems specific to on-device {#端侧特有的问题}

- **Prefill and decode have different bottlenecks**: decode is memory-bound; prefill (processing long prompts) is compute-bound. A phone's NPU has considerable compute but shares bandwidth with the CPU / GPU, so a common division is the NPU doing prefill and the CPU / GPU doing decode, or running the whole model on the NPU but only for prefill-heavy tasks;
- **Memory is shared with the system**: a phone's 8–16 GB must hold the system and other apps too, so the model plus KV Cache usually gets only a fraction of it; hence on-device models are mostly 1B–8B, with the KV Cache quantized as well;
- **Power and heat**: running at full load continuously triggers throttling, and the later part of a long output may slow down noticeably; benchmarks must measure sustained speed, not the first few seconds;
- **The first load**: mmap makes a model "open instantly", but the first access to each page reads from storage, so the first few tokens of a cold start are slow.

!!! interview "In an interview"
    For on-device questions, grab "single-user decode = bandwidth ÷ the weights' bytes" first: a 4-bit 8B model is about 4.5 GB, and a phone's 60 GB/s gives only about 9 tokens per second. Then quantization formats: GGUF quantizes in blocks, and Q4_K uses two levels of scaling to reach a 5-bit format's accuracy at 4.5 bits, plus per-layer mixed precision and an imatrix. Finally platforms: llama.cpp everywhere, MLX using unified memory, ExecuTorch exporting graphs and delegating to hardware backends; plus on-device constraints like the NPU doing prefill, and throttling from power and heat.

## Exercises {#练习}

**1. Running a 3B model on a phone, what quantization is needed for 20 tokens per second?**

??? success "Answer"
    Bandwidth of 60 GB/s at 70% effective is about 42 GB/s, so 20 tokens/s requires reading no more than 2.1 GB of weights per step. A 3B model at 4.5 bits is about 1.7 GB, which works; at 8 bits (about 3.2 GB) it only reaches about 13 tokens/s. Adding the KV Cache reads (not negligible with long contexts) and thermal throttling, leave headroom in practice, so around 4 bits is the right choice.

## Summary {#小结}

- [x] On-device single-user decode's speed limit = effective bandwidth ÷ the quantized weights' bytes; a 4-bit 8B model gets under 10 tokens per second on a phone.
- [x] GGUF quantizes in blocks: Q8_0 / Q4_0 / Q4_1 use one level of scaling, while Q4_K uses two levels over superblocks to reach a 5-bit format's accuracy at 4.5 bits; in practice per-layer mixed precision and an imatrix are common.
- [x] llama.cpp covers every platform, MLX uses Apple's unified memory, and ExecuTorch exports graphs and delegates to backends such as NPUs; on-device also means NPU division of labor, shared memory, thermal throttling and cold starts.
