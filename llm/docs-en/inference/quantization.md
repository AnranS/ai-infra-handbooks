# How quantization works

<p class="lead">Quantization represents weights and activations with fewer bits: weights go from 16 bits to 8 or 4, and the bytes read during decode shrink in proportion; quantize the activations too, and the INT8/FP8 Tensor Cores can speed up the computation. This chapter explains the math, the granularity and the mainstream methods of quantization, and experiments on a real model: how much the error differs between granularities, why activation outliers make quantization hard, and how SmoothQuant deals with them.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. How are the scale and zero point computed in symmetric and asymmetric quantization?
    2. What is the difference between per-tensor, per-channel and per-group quantization? Why is INT4 almost always quantized per group?
    3. What do weight-only quantization and W8A8 quantization each speed up?
    4. Why do activation outliers make quantization hard? How does SmoothQuant deal with them?
    5. What are the core ideas of GPTQ and AWQ?

??? success "Answers (try first, then expand to compare)"
    1. Symmetric: scale $s = \max|x| / (2^{b-1} - 1)$ with zero point 0; asymmetric: $s = (\max - \min) / (2^b - 1)$ with zero point $z = \mathrm{round}(-\min / s)$, which uses the full range at the cost of storing a zero point.
    2. Per tensor shares one scale across the whole group, per channel has one per output channel, and per group one for every run of consecutive values (say 128); the finer the granularity, the fewer values an outlier affects. INT4 has only 16 levels, and one scale per row gives too much error, so it is almost always quantized per group.
    3. Weight-only (such as W4A16) reduces the bytes of weights read, speeding up memory-bound decode while still computing in 16 bits; W8A8 also runs the matrix multiplications on INT8 Tensor Cores, speeding up the computation, which helps prefill and large batches.
    4. A few channels of activations are tens or hundreds of times larger than the rest; under per-tensor or per-token quantization they set the scale, and the other values get squeezed into a handful of levels. SmoothQuant divides the activations per channel by $s$ and multiplies the weights by $s$ (mathematically equivalent), moving the difficulty from the activations to the weights, which quantize more easily.
    5. GPTQ: quantize layer by layer, using the second-order information of the activations, $H = X^\top X$, to compensate each weight's quantization error on the weights not yet quantized; AWQ: find the important channels by the magnitude of the activations and scale them up before quantization (an equivalent scaling) to reduce their relative error.

<!-- comic ../assets/comics/quantization.webp is in Chinese; put it back once the English version exists -->

## The math of quantization {#量化的数学}

Map a floating-point number $x$ to a b-bit integer. **Symmetric quantization**:

$$
s = \frac{\max|x|}{2^{b-1} - 1},\qquad q = \text{clamp}\left(\text{round}\left(\frac{x}{s}\right), -2^{b-1}, 2^{b-1}-1\right),\qquad \hat{x} = q \cdot s
$$

**Asymmetric quantization** adds a zero point $z$ and maps $[\min, \max]$ to $[0, 2^b - 1]$, which suits data not centered on 0 (the [INT4 GEMV](cuda://advanced/quantization/) in the CUDA book uses asymmetric per-group quantization).

The error has two sources: **rounding error** (each number is off by at most half a step, $s/2$) and **clipping error** (numbers beyond the range are clipped, usually avoided by choosing a suitable $s$). The step $s$ is set by the **largest absolute value** in the group, so one large outlier makes every ordinary number less precise; this is the key to understanding quantization.

Put one outlier into a row of 1024 weights and watch it stretch the quantization grid and squash the other values to 0, then switch to per-group quantization to see it isolated:

<div class="aig-widget" data-widget="quant"></div>

## Granularity: how many numbers share one scale {#粒度多少个数共用一个缩放因子}

| Granularity | What shares a scale | Extra cost | Precision |
| --- | --- | --- | --- |
| Per tensor | the whole matrix | negligible | worst: one outlier affects everything |
| Per channel | each row of the weight matrix (each output channel) | very small | good |
| Per group | every G consecutive elements within a row (G is often 128) | one scale per G numbers | better; the standard for INT4 |
| Block-wise | blocks of, say, 128×128 | small | used by DeepSeek-V3's FP8 weights |

Compare them on a real weight matrix of Qwen3-0.6B:

```python title="quant.py"
"""quant.py —— 对称的伪量化（量化后立刻反量化），用于评估误差。"""

import torch


def fake_quant_int(w: torch.Tensor, bits: int, granularity: str = "channel", group: int = 128) -> torch.Tensor:
    out_f, in_f = w.shape
    if granularity == "tensor":
        g = w.reshape(1, 1, -1)
    elif granularity == "channel":
        g = w.reshape(out_f, 1, in_f)
    else:                                                     # "group"
        g = w.reshape(out_f, in_f // group, group)
    qmax = 2 ** (bits - 1) - 1
    scale = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    q = (g / scale).round().clamp(-qmax - 1, qmax)
    return (q * scale).reshape(w.shape)


def fake_quant_fp8(w: torch.Tensor) -> torch.Tensor:
    scale = w.abs().max() / 448.0                             # E4M3's maximum is 448
    return (w / scale).to(torch.float8_e4m3fn).float() * scale


def rel_error(approx: torch.Tensor, exact: torch.Tensor) -> float:
    return ((approx - exact).norm() / exact.norm()).item()
```

```python
import torch
from mini_llm import Transformer
from quant import fake_quant_fp8, fake_quant_int, rel_error

model = Transformer.from_pretrained("models/Qwen3-0.6B")
W = model.layers[0].mlp.down_proj.weight.data                  # [1024, 3072]
results = {
    "INT8 按张量": rel_error(fake_quant_int(W, 8, "tensor"), W),
    "INT8 按通道": rel_error(fake_quant_int(W, 8, "channel"), W),
    "FP8 E4M3 按张量": rel_error(fake_quant_fp8(W), W),
    "INT4 按通道": rel_error(fake_quant_int(W, 4, "channel"), W),
    "INT4 按组（G=128）": rel_error(fake_quant_int(W, 4, "group"), W),
}
for name, e in results.items():
    print(f"{name:18s} 相对误差 {e:.4f}")
assert results["INT8 按通道"] < results["INT8 按张量"] and results["INT4 按组（G=128）"] < results["INT4 按通道"]
```

In this handbook's environment:

```text
INT8 按张量           相对误差 0.0362
INT8 按通道           相对误差 0.0097
FP8 E4M3 按张量       相对误差 0.0265
INT4 按通道           相对误差 0.1762
INT4 按组（G=128）     相对误差 0.1240
```

A few observations: INT8 per channel has only about 1/4 the error of per tensor; FP8, even per tensor, has less error than INT8 per tensor, because a floating-point format has precision that adapts to numbers of different sizes; INT4 has only 16 levels, its error is an order of magnitude larger, and per group is clearly better than per channel.

## Weight-only quantization of a whole model {#对整个模型做-weight-only-量化}

Fake-quantize every linear layer (the 4 attention projections and the 3 FFN projections) and compare perplexity on a piece of text:

```python
import copy
import math
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")
text = ("大语言模型的推理过程分为两个阶段。在预填充阶段，模型一次性处理用户输入的全部提示词，计算每个位置的键和值并写入缓存。"
        "在解码阶段，模型每次只生成一个新的词元，需要读取全部的模型权重和已经缓存的键值。由于每一步的计算量很小而读取的数据量很大，"
        "解码阶段通常受限于显存带宽。为了提高吞吐量，推理系统会把多个请求合并成一个批次，让它们共享同一次权重读取。"
        "量化通过降低权重和激活的数值精度来减少需要读取的字节数，是加速解码的常用方法。")
ids = tok(text, return_tensors="pt").input_ids

@torch.no_grad()
def perplexity(m):
    logits = m(ids)
    return math.exp(torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:]).item())

def quantize_model(m, fn):
    m = copy.deepcopy(m)
    for layer in m.layers:
        a, f = layer.self_attn, layer.mlp
        for lin in (a.q_proj, a.k_proj, a.v_proj, a.o_proj, f.gate_proj, f.up_proj, f.down_proj):
            lin.weight.data = fn(lin.weight.data)
    return m

ppl = {"FP32": perplexity(model)}
for name, fn in [("W8 按通道", lambda w: fake_quant_int(w, 8, "channel")),
                 ("W4 按组 G=128", lambda w: fake_quant_int(w, 4, "group")),
                 ("W4 按通道", lambda w: fake_quant_int(w, 4, "channel")),
                 ("W3 按组 G=128", lambda w: fake_quant_int(w, 3, "group"))]:
    ppl[name] = perplexity(quantize_model(model, fn))
for name, v in ppl.items():
    print(f"{name:14s} 困惑度 {v:8.2f}")
assert abs(ppl["W8 按通道"] - ppl["FP32"]) / ppl["FP32"] < 0.02
assert ppl["W4 按组 G=128"] < ppl["W4 按通道"] < ppl["W3 按组 G=128"]
```

In this handbook's environment:

```text
FP32           困惑度    25.94
W8 按通道         困惑度    25.98
W4 按组 G=128    困惑度    41.49
W4 按通道         困惑度    55.02
W3 按组 G=128    困惑度   442.45
```

- **INT8 per channel is nearly lossless**, which is why W8 quantization is so widespread;
- **INT4 with the most naive round-to-nearest (RTN) quantization raises perplexity by about 60%**, and a small model like 0.6B is especially sensitive to quantization (large models have more redundancy and usually tolerate it better); per group (41.49) beats per channel (55.02);
- Naive 3-bit quantization is essentially unusable.

That is why INT4 needs methods smarter than RTN.

## GPTQ and AWQ {#gptq-与-awq}

**GPTQ** (Frantar et al., 2022): quantize the weights column by column, and after each column, use the second-order information of the calibration data (the covariance of the inputs) to adjust **the columns not yet quantized** to compensate for the error this column introduced. The goal is to keep the layer's **output** after quantization (rather than the weights themselves) as close as possible to the original.

**AWQ** (Lin et al., 2023): observes that only about 1% of weight channels matter especially, and their **input activations** are usually large. Instead of shielding these weights from quantization, multiply these channels by a factor greater than 1 before quantization (equivalently dividing the matching activations by the same factor), so they get higher relative precision when quantized. The factors are found by a search over a small amount of calibration data.

Both are done **offline**: the quantized model has the same format as RTN (integer weights + scales + possibly zero points), and the inference kernel does not need to know which method was used.

## Activation quantization and outliers {#激活量化与离群值}

Weight-only quantization speeds up **reading weights**, while the computation still runs in FP16/BF16. To speed up the computation with INT8/FP8 Tensor Cores, **the activations must be quantized too** (W8A8). The difficulty is activation outliers: the [normalization and residual stream](../transformer/norm-residual.md#看看真实的残差流) chapter showed that a few fixed dimensions of the residual stream have especially large values.

Capture the input activations of layer 12's FFN (after RMSNorm) to look at their distribution and the W8A8 error:

```python
from quant import rel_error

captured = {}
hook = model.layers[12].mlp.gate_proj.register_forward_hook(
    lambda mod, inp, out: captured.__setitem__("x", inp[0][0].detach()))
with torch.no_grad():
    model(ids)
hook.remove()
X = captured["x"]                                         # [T, 1024]
W = model.layers[12].mlp.gate_proj.weight.data            # [3072, 1024]
top = X.abs().amax(dim=0).topk(4)
print("最大的几个通道:", [(i, round(v, 1)) for v, i in zip(top.values.tolist(), top.indices.tolist())],
      " 中位数:", round(X.abs().median().item(), 3))

def q8_per_tensor(x):
    s = x.abs().max() / 127
    return (x / s).round().clamp(-128, 127) * s

def q8_per_row(w):
    s = w.abs().amax(dim=1, keepdim=True) / 127
    return (w / s).round().clamp(-128, 127) * s

Y = X @ W.T
err_w8 = rel_error(X @ q8_per_row(W).T, Y)
err_w8a8 = rel_error(q8_per_tensor(X) @ q8_per_row(W).T, Y)

# SmoothQuant: divide each activation channel by s_j and multiply the matching weight column by s_j; the product is unchanged
alpha = 0.5
s = X.abs().amax(dim=0).clamp(min=1e-5) ** alpha / W.abs().amax(dim=0).clamp(min=1e-5) ** (1 - alpha)
Xs, Ws = X / s, W * s
assert torch.allclose(Xs @ Ws.T, Y, atol=1e-3)              # mathematically identical
err_smooth = rel_error(q8_per_tensor(Xs) @ q8_per_row(Ws).T, Y)
print(f"只量化权重 {err_w8:.4f}；W8A8 {err_w8a8:.4f}；SmoothQuant 后 W8A8 {err_smooth:.4f}")
assert err_w8 < err_smooth < err_w8a8
```

In this handbook's environment:

```text
最大的几个通道: [(277, 21.3), (16, 12.0), (23, 11.6), (3, 11.6)]  中位数: 0.411
只量化权重 0.0065；W8A8 0.0400；SmoothQuant 后 W8A8 0.0130
```

The largest, dimension 277, is also one of the outlier dimensions seen earlier in the residual stream (after RMSNorm each dimension is multiplied by a different scale, so the ranking changes). With per-tensor activation quantization, the step is set by dimension 277's 21.3, so ordinary values with a median of only 0.4 can use only a few quantization levels, and the error is 6 times that of quantizing the weights alone.

**SmoothQuant** (Xiao et al., 2022) uses a simple equivalence: $XW^\top = (X \operatorname{diag}(s)^{-1})(W\operatorname{diag}(s))^\top$. "Shrink" the activations' outlier channels and "grow" the matching weight columns, moving part of the difficulty from the activations to the weights (which are quantized per channel and can take it). $\alpha$ controls how much is moved. The scaling can be merged offline into the weights of the preceding RMSNorm, so it costs nothing at inference time. The error drops to about 1/3.

## Mainstream schemes at a glance {#主流方案小结}

| Scheme | Weights | Activations | What it speeds up | Common use |
| --- | --- | --- | --- | --- |
| W8A16 / W4A16 | INT8 / INT4 (GPTQ, AWQ) | BF16 | weight reads (decode) | small-batch, memory-limited deployments |
| W8A8 INT8 | INT8 | INT8 (SmoothQuant and others) | weight reads + INT8 Tensor Core compute | GPUs with INT8 support such as Ampere |
| FP8 (W8A8) | FP8 | FP8 (dynamic or static scales) | weight reads + FP8 Tensor Core compute | the mainstream choice on Hopper/Ada, with little accuracy loss |
| FP4 (NVFP4 / MXFP4) | FP4 | FP4 | as above, one step further | Blackwell |
| KV cache quantization | — | K, V in FP8/INT8 | KV reads, KV capacity | long contexts, high concurrency |

!!! inference "Inference view"
    - **Decode is about bytes, prefill about compute**: weight-only quantization speeds up small-batch decode almost in proportion to the compression ratio, but the gain shrinks or even turns into a slowdown for large batches and prefill (dequantization has a cost, and the computation is still BF16); W8A8/FP8 helps in both;
    - **Dequantization happens inside the kernel**: quantized weights are unpacked in registers, multiplied by their scales, and fed to the Tensor Cores. Kernels such as Marlin carefully design an offline reordering of the weights and bit tricks for this; see [quantization and GEMV](cuda://advanced/quantization/) in the CUDA book;
    - **Always evaluate accuracy**: perplexity is only a rough signal; before deploying, compare on an evaluation set for the target task, since small models, long contexts and reasoning (math) tasks are often more sensitive to quantization.

!!! interview "How to explain it"
    To explain quantization, first sort out "what it speeds up": weight-only (W4A16, W8A16) reduces the bytes of weights read, speeding up memory-bound decode; W8A8 / FP8 runs the matrix multiplications on low-precision Tensor Cores too, speeding up prefill and large batches. Then precision: the step is set by the largest absolute value in a group, finer granularity is more accurate, and INT4 must be quantized per group (usually 128); activation outliers are the hard part, and SmoothQuant moves the difficulty to the weights with an equivalent scaling; GPTQ compensates errors using second-order information, and AWQ protects the important channels by activation magnitude. Finish by stressing evaluation on the target task before going live.

## Exercises {#练习}

**1. The quantization step.** A group of weights has a maximum absolute value of 0.8. With symmetric INT4 quantization, what is the step? What does 0.05 quantize to? What is the error?

??? success "Answer"
    The largest integer in symmetric INT4 is 7, so the step is s = 0.8 / 7 ≈ 0.1143. 0.05 / 0.1143 ≈ 0.44, which rounds to 0, so it dequantizes to 0 with an error of 0.05 (100%). Under INT4, numbers an order of magnitude smaller than the maximum are almost all quantized to 0. That is why INT4 must narrow the range that shares a scale (per-group quantization).

    ```python
    s = 0.8 / 7
    assert round(0.05 / s) == 0
    ```

**2. Estimate.** A 70B model quantized with W4A16 (G = 128, each group with one FP16 scale and one 4-bit zero point): how many GB of weights in total? How much smaller than BF16?

??? success "Answer"
    Each weight is 0.5 bytes; every 128 weights add 2 + 0.5 = 2.5 bytes, about 0.02 bytes per weight on average. That totals about 0.52 bytes per parameter, about 36.4 GB for 70B (the embedding and output layers are usually not quantized, so in practice it is a bit larger), versus 140 GB in BF16, about 26% of it.

    ```python
    per_param = 0.5 + 2.5 / 128
    print(f"{70e9 * per_param / 1e9:.1f} GB")
    ```

## Summary {#小结}

- [x] The quantization step is set by the largest absolute value in a group, so outliers hurt the precision of every ordinary value.
- [x] The finer the granularity, the smaller the error: INT8 per channel is nearly lossless, while INT4 needs per-group quantization and smarter methods such as GPTQ/AWQ.
- [x] Weight-only quantization speeds up weight reads (decode); W8A8/FP8 speeds up the computation as well.
- [x] Activation outliers are the main difficulty of activation quantization; SmoothQuant moves the difficulty to the weights with an equivalent scaling.
- [x] Before deployment, the quantized model's accuracy must be evaluated on the target task.

For the related math (why rotation removes outliers, see [linear algebra](math://linear-algebra/); measuring quantization loss with KL divergence, see [information theory](math://information-theory/); the derivation and implementation of GPTQ, see [calculus and backpropagation](math://calculus/); quantization noise at 6 dB per bit, see [floating point and numerical computation](math://floating-point/)).
