# Quantization in deployment

<p class="lead">The quantization chapter of the LLM handbook covered the principles: group-wise quantization, outliers, GPTQ/AWQ, SmoothQuant. Deployment raises a different set of questions: what scaling granularity for FP8? How do MXFP4, NVFP4 and INT4 on Blackwell differ? Does quantizing the KV Cache hurt accuracy? How much more traffic can a service actually take with quantization? This chapter compares the accuracy of these formats on Qwen3-0.6B with fake quantization (contrasting with the previous generation's Qwen2.5-0.5B), estimates their effect on capacity with the previous chapter's simulator, and finally lays out how to use them in vLLM and SGLang.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. How do FP8 E4M3 and E5M2 differ? Which is used in inference?
    2. Why does DeepSeek-V3's FP8 use 128×128 block scaling?
    3. How do MXFP4 and NVFP4 scale differently? Which is more accurate?
    4. When quantizing the KV Cache to FP8, which is more sensitive, K or V? Why?
    5. W4A16 makes decode about 3× faster; why doesn't the service's capacity grow 3×?

??? success "Answers (try first, then expand to compare)"
    1. E4M3: 4 exponent bits and 3 mantissa bits, more precision, smaller range (max 448); E5M2: 5 exponent bits and 2 mantissa bits, larger range, less precision. Inference uses E4M3 (E5M2 is mainly for gradients in training).
    2. The model was trained at this granularity, and its activations and weights have extreme outliers; per-tensor scaling would crush the normal values. Block scaling lets an outlier affect only its own block, and inference must quantize at the same granularity to keep accuracy.
    3. MXFP4: one scale per 32 elements, a power of 2 (E8M0); NVFP4: one scale per 16 elements, in FP8 (E4M3), plus a tensor-level scale. NVFP4 has smaller groups and finer scales, so it is more accurate.
    4. K is often more sensitive: K tends to have a "constant per-channel bias + small variation" structure, so with per-tensor quantization the bias fills the quantization range and the small variations that actually distinguish tokens drown in quantization error (in this chapter Qwen2.5's K is very sensitive, while Qwen3, with QK-Norm, is nearly lossless). Remedies: calibrated scales, finer granularity, or per-channel quantization with zero points.
    5. W4A16 only cuts the bytes of weights read, lowering TPOT at low load; near saturation every step includes prefill, which is compute-bound, and weight-only quantization cannot speed up compute, so the service's capacity barely changes.

## Format cheat sheet {#格式速查}

| Format | Values | Scaling | What it speeds up | Hardware |
| --- | --- | --- | --- | --- |
| W8A8 FP8 (E4M3) | 1 sign + 4 exponent + 3 mantissa, max 448 | per tensor, per channel/per token, block (128×128) | halves weight reads, **doubles compute** (FP8 Tensor Cores) | Hopper, Blackwell, MI300 |
| W8A8 INT8 | integers −128 to 127 | per channel + per token, often with SmoothQuant | same as above | Ampere and later |
| W4A16 (INT4 GPTQ/AWQ) | integers −8 to 7 | one scale per 128 numbers (+ zero point) | weight reads drop to about 1/4, compute stays BF16 | all kinds of GPUs (Marlin and other kernels) |
| MXFP4 | E2M1: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6} | one power-of-2 scale per 32 numbers (E8M0) | weight reads about 1/4; FP4 compute on Blackwell | native on Blackwell; gpt-oss's release format |
| NVFP4 | E2M1 | one FP8 (E4M3) scale per 16 numbers + a tensor-level FP32 scale | same as above | native on Blackwell |
| KV Cache FP8 | E4M3 | per tensor (calibrated) or finer | halves KV memory and reads | depends on the attention backend |

FP8 in inference is almost always **E4M3**: it has one more mantissa bit than E5M2 for more precision, and its range (±448) is enough together with scales. E5M2 has a larger range and less precision, mainly for gradients in training.

First get a feel for how outliers and granularity affect error with the widget from the LLM handbook:

<div class="aig-widget" data-widget="quant"></div>

## Fake quantization {#伪量化实现}

```python title="formats.py"
"""formats.py —— 部署中常见的低精度格式（伪量化实现）：FP8 的几种缩放粒度、MXFP4、NVFP4，以及 FP8 KV Cache。

所有函数都是"量化后立刻反量化"，返回与输入同形状的 FP32 张量，用来评估精度损失。
"""

import torch

from mini_llm import KVCache

FP8_MAX = 448.0                                                     # the largest value E4M3 can represent
FP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])  # all non-negative values E2M1 can represent


def fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (x / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * scale


def fp8_per_tensor(x):
    return fp8(x, x.abs().max().clamp(min=1e-12) / FP8_MAX)


def fp8_per_row(x):
    """权重按输出通道、激活按 token：每行一个缩放因子。"""
    return fp8(x, x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / FP8_MAX)


def fp8_blockwise(x, block=(128, 128)):
    """DeepSeek-V3 的做法：权重每 128×128 一个缩放因子；激活用 (1, 128)，即每个 token 每 128 个通道一个。"""
    rows, cols = x.shape[-2], x.shape[-1]
    br, bc = min(block[0], rows), block[1]
    g = x.reshape(*x.shape[:-2], rows // br, br, cols // bc, bc)
    scale = g.abs().amax(dim=(-3, -1), keepdim=True).clamp(min=1e-12) / FP8_MAX
    return fp8(g, scale).reshape(x.shape)


def fp4(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """把 x / scale 舍入到最近的 E2M1 值。"""
    y = (x / scale).clamp(-6.0, 6.0)
    idx = (y.abs().unsqueeze(-1) - FP4_VALUES).abs().argmin(-1)
    return FP4_VALUES[idx] * y.sign() * scale


def mxfp4(w, block=32):
    """OCP MX 格式：每 32 个数共享一个 2 的幂次缩放（E8M0），使块内最大值落在 E2M1 的范围内。"""
    g = w.reshape(*w.shape[:-1], -1, block)
    amax = g.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    scale = 2.0 ** (torch.floor(torch.log2(amax)) - 2)                  # E2M1's largest exponent is 2
    return fp4(g, scale).reshape(w.shape)


def nvfp4(w, block=16):
    """NVFP4：每 16 个数一个 FP8（E4M3）缩放因子，外加一个整个张量的 FP32 缩放因子。"""
    g = w.reshape(*w.shape[:-1], -1, block)
    global_scale = w.abs().max().clamp(min=1e-12) / (FP8_MAX * 6.0)
    block_scale = (g.abs().amax(-1, keepdim=True) / 6.0 / global_scale).to(torch.float8_e4m3fn).float()
    return fp4(g, block_scale.clamp(min=1e-12) * global_scale).reshape(w.shape)


class FP8KVCache(KVCache):
    """写入 KV Cache 时量化成 FP8（每层一个缩放因子，由写入的数据决定）。"""

    def update(self, layer, k, v):
        return super().update(layer, fp8_per_tensor(k), fp8_per_tensor(v))
```

The core difference between MXFP4 and NVFP4 is the scale: MXFP4's scale can only be a power of 2, so the block's maximum and the largest representable value can differ by nearly 2×, wasting precision; NVFP4's scale is itself an FP8 number that fits the block's maximum precisely, and its blocks are smaller (16 numbers), so an outlier affects a smaller range.

## Comparing accuracy {#精度比较}

Fake-quantize all linear layers (or the KV Cache) in each format and measure perplexity on the same passage used in the LLM handbook (FP32 baseline 25.94; group-wise INT4 in the LLM handbook gave 41.49):

```python ci="loose"
import copy
import math
import torch
from transformers import AutoTokenizer
from mini_llm import KVCache, Transformer
from quant import fake_quant_int
from formats import FP8KVCache, fp8, fp8_blockwise, fp8_per_row, fp8_per_tensor, mxfp4, nvfp4

torch.set_num_threads(16)
path = "models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(path)
model = Transformer.from_pretrained(path)
text = ("大语言模型的推理过程分为两个阶段。在预填充阶段，模型一次性处理用户输入的全部提示词，计算每个位置的键和值并写入缓存。"
        "在解码阶段，模型每次只生成一个新的词元，需要读取全部的模型权重和已经缓存的键值。由于每一步的计算量很小而读取的数据量很大，"
        "解码阶段通常受限于显存带宽。为了提高吞吐量，推理系统会把多个请求合并成一个批次，让它们共享同一次权重读取。"
        "量化通过降低权重和激活的数值精度来减少需要读取的字节数，是加速解码的常用方法。")
ids = tok(text, return_tensors="pt").input_ids

@torch.no_grad()
def perplexity(m, cache=None):
    logits = m(ids, cache)
    return math.exp(torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:]).item())

def quantized(weight_fn, act_fn=None):
    m = copy.deepcopy(model)
    for layer in m.layers:
        a, f = layer.self_attn, layer.mlp
        for lin in (a.q_proj, a.k_proj, a.v_proj, a.o_proj, f.gate_proj, f.up_proj, f.down_proj):
            lin.weight.data = weight_fn(lin.weight.data)
            if act_fn is not None:                          # activations are quantized on entering a linear layer
                lin.register_forward_pre_hook(lambda mod, args: (act_fn(args[0]),))
    return m

results = {
    "FP32 基线": perplexity(model),
    "W8A8 FP8 按张量": perplexity(quantized(fp8_per_tensor, fp8_per_tensor)),
    "W8A8 FP8 按通道/按 token": perplexity(quantized(fp8_per_row, fp8_per_row)),
    "W8A8 FP8 分块（DeepSeek 式）": perplexity(quantized(fp8_blockwise, lambda x: fp8_blockwise(x, (1, 128)))),
    "W4A16 INT4 按组 128": perplexity(quantized(lambda w: fake_quant_int(w, 4, "group"))),
    "W4A16 MXFP4": perplexity(quantized(mxfp4)),
    "W4A16 NVFP4": perplexity(quantized(nvfp4)),
    "KV Cache FP8 按张量": perplexity(model, FP8KVCache(model.cfg.num_hidden_layers)),
}
for name, v in results.items():
    print(f"{name:26s} 困惑度 {v:6.2f}（{v / results['FP32 基线'] - 1:+.1%}）")
```

```text title="输出"
FP32 基线                    困惑度  25.94（+0.0%）
W8A8 FP8 按张量               困惑度  26.66（+2.8%）
W8A8 FP8 按通道/按 token       困惑度  26.55（+2.3%）
W8A8 FP8 分块（DeepSeek 式）    困惑度  26.33（+1.5%）
W4A16 INT4 按组 128          困惑度  41.49（+59.9%）
W4A16 MXFP4                困惑度  31.23（+20.4%）
W4A16 NVFP4                困惑度  28.43（+9.6%）
KV Cache FP8 按张量           困惑度  25.97（+0.1%）
```

Conclusions:

- **For FP8, the finer the scaling granularity the better**: per tensor +2.8%, per channel/per token +2.3%, DeepSeek-style block scaling +1.5%. Block scaling gives every 128 numbers their own quantization range, so an outlier affects only its block; this is also the foundation of DeepSeek-V3's FP8 training and inference.
- **Among 4-bit formats, the smaller the block the better**: NVFP4 (blocks of 16, FP8 scales) +10%, MXFP4 (blocks of 32, power-of-2 scales) +20%, and group-wise INT4 with groups of 128 the worst (+60%). MXFP4's scale can only be a power of 2, so a block's maximum may use only a little over half the representable range, wasting precision. Qwen3 is unusually sensitive to naive rounding (the same experiment on Qwen2.5-0.5B loses only 22% with group-wise INT4; empirical studies have found Qwen3 degrades noticeably more than the previous generation at low bit widths, presumably because more thorough pretraining leaves less redundancy in the weights). In real deployments, 4-bit quantization usually comes with GPTQ, AWQ or quantization-aware training (QAT, as done for gpt-oss and Kimi K2 Thinking) rather than this chapter's naive rounding, and accuracy is much better.
- **FP8 for the KV Cache is nearly lossless** (+0.1%). This depends on the model's architecture and does not hold for every model; it is analyzed separately below.

## KV Cache quantization: compare the error with the signal {#kv-cache-量化误差要和信号比}

Quantize K and V separately, try finer scaling granularity, and look at K's value distribution:

```python
class QuantKVCache(KVCache):
    def __init__(self, num_layers, quant_k, quant_v):
        super().__init__(num_layers)
        self.quant_k, self.quant_v = quant_k, quant_v

    def update(self, layer, k, v):                          # k, v: [B, heads, T, head_dim]
        return super().update(layer, self.quant_k(k), self.quant_v(v))

def per_head(x):
    return fp8(x, x.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-12) / 448)

identity, L = (lambda x: x), model.cfg.num_hidden_layers
for name, qk, qv in [("只量化 K（按张量）", fp8_per_tensor, identity), ("只量化 V（按张量）", identity, fp8_per_tensor),
                     ("K、V 都按头缩放", per_head, per_head)]:
    print(f"{name:14s} 困惑度 {perplexity(model, QuantKVCache(L, qk, qv)):6.2f}")

cache = KVCache(L)
with torch.no_grad():
    model(ids, cache)
for layer in (0, 12, 23):
    k, v = cache.k[layer][0], cache.v[layer][0]                 # [heads, T, head_dim]
    spread = k.std(dim=1).median()                              # how much each dimension varies across tokens (median)
    print(f"第 {layer:2d} 层：|K| 最大 {k.abs().max():6.1f}、中位数 {k.abs().median():4.2f}、"
          f"各维随 token 的标准差 {spread:4.2f}；|V| 最大 {v.abs().max():5.2f}")
```

```text title="输出"
只量化 K（按张量）     困惑度  25.93
只量化 V（按张量）     困惑度  25.87
K、V 都按头缩放      困惑度  25.55
第  0 层：|K| 最大  490.7、中位数 0.98、各维随 token 的标准差 1.17；|V| 最大  1.37
第 12 层：|K| 最大   59.2、中位数 0.69、各维随 token 的标准差 0.96；|V| 最大 39.05
第 23 层：|K| 最大   27.7、中位数 0.82、各维随 token 的标准差 1.00；|V| 最大 42.96
```

However K and V are quantized, perplexity barely changes. Yet layer 0's maximum K is 500 times its median, so with per-tensor scaling most values use only a tiny slice of FP8's range; why is that fine?

Because FP8 is a floating-point format, its 3 mantissa bits set the relative error: as long as a value is not too small (not in the subnormal range), the error is about 3%–6% **of the value itself**. So what matters is the error compared with **the useful signal**: what a dimension really carries is its variation across tokens, not its absolute size.

- **Qwen3**: K passes through QK-Norm before RoPE, and each dimension varies around 0 (layer 0 median 0.98, standard deviation across tokens 1.17), so error and signal scale together and quantization is nearly lossless. The largest dimensions come from the per-dimension weights of `k_norm` (layer 0 max 96.5, median 2.3), which also vary with tokens and also carry information;
- **Qwen2.5** (K projection with a bias, no QK-Norm): the same experiment quantizing only K raises perplexity by 12% (22.14 → 24.73), while quantizing only V barely changes it. Its layer-0 K has a median of 5.7 but a standard deviation across tokens of only 0.36: every dimension is "a large constant bias + a very small variation" (the largest dimensions happen to sit at the frequencies RoPE rotates slowest, so they remain nearly constant after rotation). A constant bias has no effect on softmax (every key gets the same number added), but the error is produced at the scale of 5.7, which is large next to an effective signal of 0.36. Per-head scaling helps somewhat (the loss falls to about 2%); the more thorough fix is to quantize K **per channel with zero points**, absorbing each dimension's constant bias into the zero point. That is why KV quantization methods such as KIVI quantize K per channel and V per token.

In real deployments:

- Compute K and V scales offline with calibration data (written into the quantized checkpoint), which is steadier than computing them at runtime from the current batch;
- The attention backend must support the corresponding granularity (per tensor is the most common, with per-head support growing);
- MLA's latent vectors, sliding-window models and others have their own special handling (vLLM's `fp8_ds_mla` and so on);
- Always evaluate accuracy on long contexts before going live: KV error accumulates with context.

## The effect on capacity {#对容量的影响}

How much more traffic does quantization let a service take? Use the [previous chapter](benchmark.md)'s simulator to estimate the maximum request rate of Qwen2.5-7B on one H100 under the same SLO:

```python
from dataclasses import replace
from sim import Setup, simulate, summarize

bf16 = Setup(params=7.6e9, weight_bytes=15.2e9, kv_bytes_per_token=57344, kv_capacity_tokens=int(55e9 / 57344))
variants = {
    "BF16": bf16,
    "W4A16（INT4/FP4 weight-only）": replace(bf16, weight_bytes=4.5e9, kv_capacity_tokens=int(65e9 / 57344)),
    "W8A8 FP8": replace(bf16, weight_bytes=7.6e9, peak_flops=1979e12, kv_capacity_tokens=int(62e9 / 57344)),
    "W8A8 FP8 + KV FP8": replace(bf16, weight_bytes=7.6e9, peak_flops=1979e12, kv_bytes_per_token=28672,
                                 kv_capacity_tokens=int(62e9 / 28672)),
}

def max_rate(s):
    lo, hi = 1.0, 80.0
    for _ in range(12):
        mid = (lo + hi) / 2
        ok = summarize(simulate(s, mid, 2000, 1024, 256), 1.0, 0.04)["slo_ok"] >= 0.99
        lo, hi = (mid, hi) if ok else (lo, mid)
    return lo

print("格式                            低负载 TTFT   低负载 TPOT   满足 SLO 的最大速率")
for name, s in variants.items():
    low = summarize(simulate(s, 1, 500, 1024, 256), 1.0, 0.04)
    print(f"{name:30s} {low['ttft_p50'] * 1e3:7.1f} ms   {low['tpot_p50'] * 1e3:7.1f} ms   {max_rate(s):8.1f} req/s")
```

```text title="输出"
格式                            低负载 TTFT   低负载 TPOT   满足 SLO 的最大速率
BF16                              33.8 ms       5.2 ms       22.5 req/s
W4A16（INT4/FP4 weight-only）       32.0 ms       1.9 ms       24.6 req/s
W8A8 FP8                          16.2 ms       2.8 ms       46.5 req/s
W8A8 FP8 + KV FP8                 16.2 ms       2.8 ms       49.3 req/s
```

This table shows nicely "what quantization speeds up":

- **W4A16 cuts low-load TPOT to about 1/3** (reading 1/3 of the bytes), very valuable for latency-sensitive, low-concurrency scenarios; but **service capacity barely changes**: near saturation, every step includes prefill, which is compute-bound, and weight-only quantization cannot speed up compute;
- **W8A8 FP8 halves TTFT and doubles capacity**: it cuts both the bytes read and the compute time (FP8 Tensor Cores deliver 2× the compute of BF16);
- **KV FP8** adds a bit more on top: KV reads are halved and the context that fits doubles.

So choosing a quantization scheme starts with the bottleneck: for latency first with low concurrency, W4A16 fits well; for throughput first under high load, choose W8A8 FP8 (or FP4 compute on Blackwell).

## Using it in inference engines {#在推理引擎中使用}

```bash
# vLLM: quantize BF16 weights to FP8 online (no calibration, the easiest)
vllm serve Qwen/Qwen2.5-7B-Instruct --quantization fp8 --kv-cache-dtype fp8
# vLLM: load a pre-quantized checkpoint (the format is detected from the checkpoint's config)
vllm serve /path/to/Qwen2.5-7B-Instruct-W4A16               # e.g. the compressed-tensors format exported by LLM Compressor
# SGLang
python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --quantization fp8 --kv-cache-dtype fp8_e4m3
```

Pre-quantized checkpoints usually come from these tools: **LLM Compressor** (outputs the compressed-tensors format, supporting GPTQ, AWQ, SmoothQuant, FP8, NVFP4 and more), **NVIDIA ModelOpt** (FP8, NVFP4, targeting TensorRT-LLM and vLLM/SGLang), AutoAWQ/GPTQModel and others. Accuracy evaluation of quantization is best done in three steps: a quick perplexity check → general benchmarks (MMLU, GSM8K and so on, with lm-evaluation-harness) → your own business evaluation set, with separate evaluation of long contexts and sensitive tasks like code and math.

!!! source "Source code"
    - **vLLM**: quantization methods are in `vllm/model_executor/layers/quantization/`: `fp8.py` (including block FP8), `compressed_tensors/` (LLM Compressor's format), `modelopt.py` (ModelOpt's FP8/NVFP4), `mxfp4.py` (gpt-oss), `auto_gptq.py`, `auto_awq.py`, `kv_cache.py` (KV scales) and more; each method plugs its own kernels into the linear and MoE layers. `--kv-cache-dtype` accepts `fp8` (i.e. `fp8_e4m3`), `fp8_e5m2`, and for DeepSeek MLA `fp8_ds_mla`, `nvfp4_ds_mla` and others.
    - **SGLang**: `srt/layers/quantization/` has the corresponding implementations, and `--quantization` and `--kv-cache-dtype` (`fp8_e4m3`, `fp8_e5m2` and others) work similarly.

!!! interview "In an interview"
    "Should a production service quantize? Which kind?" First ask about the bottleneck and goal: latency first or throughput first, and what hardware. Then answer with this chapter's logic: weight-only (W4A16) cuts weight reads and improves low-load TPOT, but neither speeds up prefill nor raises saturated throughput; W8A8 (FP8/INT8) also speeds up compute and raises capacity; KV quantization increases concurrency and long-context capability, but depends on K's value distribution (constant biases, outliers). Always finish with the accuracy validation process and the rollback plan. Details like "block scaling", "the difference between NVFP4 and MXFP4", and "why some models' K is very sensitive to quantization and others' is not" earn bonus points.

!!! info "Related chapters"
    - [Quantization](llm://inference/quantization/) (LLM Internals: the math and error of quantization)
    - [Quantization and GEMV](cuda://advanced/quantization/) (Advanced CUDA: how to write quantized kernels)
    - [FP8 fine-grained quantization and grouped GEMM](../moe/fp8-gemm.md), [low-bit inference for very large MoE](../frontier/low-bit.md) (this book)

## Exercises {#练习}

**1. Why not quantize these layers?** Many quantization schemes keep the embedding layer, the output layer (lm_head) and the MoE router in BF16. Why?

??? success "Answer"
    - The embedding layer is a table lookup, so quantization cannot speed up compute, only save memory, while embedding errors affect everything downstream;
    - The output layer directly determines the logits, so errors directly change the ranking of tokens, and with a large vocabulary its value distribution is wide; in small models the output layer is a large share of the parameters and is sometimes quantized, which needs separate evaluation;
    - The MoE router's output decides which experts a token goes to, and a tiny error can change the top-k choice, producing a discrete large error; the router has very few parameters, so keeping it in high precision costs almost nothing.

**2. The cost of block FP8.** DeepSeek-style block FP8 is the most accurate. What does it cost?

??? success "Approach"
    GEMM accumulation must handle the scales per block: Tensor Cores multiply and accumulate in FP8 within a 128-wide K block, then multiply by that block's activation and weight scales, and accumulate into the FP32 result (DeepSeek's DeepGEMM does this promotion step on CUDA Cores). This is more complex than a per-tensor-scaled GEMM, needs a dedicated kernel, and is slightly less efficient; the scales also take storage and bandwidth (very little). Moreover, the (1, 128) scaling of activations must be computed at the end of the previous kernel to avoid extra reads and writes.

## Summary {#小结}

- [x] FP8 in inference uses E4M3; scaling granularity sets the accuracy: per tensor < per channel/per token < block.
- [x] Among 4-bit formats, NVFP4 (one FP8 scale per 16 numbers) beats group-wise INT4, and MXFP4 (power-of-2 scales) is the worst; real deployments pair them with GPTQ/AWQ/QAT.
- [x] For KV Cache quantization, compare the error with the signal: K is very sensitive when it is "constant bias + small variation" (Qwen2.5), and nearly lossless with QK-Norm (Qwen3); remedies are calibrated scales, finer granularity or per-channel quantization with zero points, plus long-context evaluation.
- [x] W4A16 lowers low-load latency but barely raises saturated capacity; W8A8 FP8 also speeds up compute and doubles capacity; find the bottleneck before choosing a scheme.
