# 量化部署实战

<p class="lead">大模型手册的量化一章讲了原理：按组量化、离群值、GPTQ/AWQ、SmoothQuant。部署时面对的是另一组问题：FP8 用什么缩放粒度？Blackwell 上的 MXFP4、NVFP4 和 INT4 有什么区别？KV Cache 量化会不会伤精度？量化到底能让服务多扛多少流量？这一章用伪量化在 Qwen3-0.6B 上比较这些格式的精度（并和上一代的 Qwen2.5-0.5B 对照），用上一章的模拟器估算它们对容量的影响，最后整理出在 vLLM 和 SGLang 中的具体用法。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. FP8 E4M3 和 E5M2 的区别？推理中用哪个？
    2. DeepSeek-V3 的 FP8 为什么用 128×128 的分块缩放？
    3. MXFP4 和 NVFP4 的缩放方式有什么不同？哪个精度更好？
    4. KV Cache 做 FP8 量化时，K 和 V 哪个更敏感？为什么？
    5. W4A16 能让 decode 快约 3 倍，为什么服务的容量没有涨 3 倍？

??? success "自测参考答案（先自己答，再展开对照）"
    1. E4M3：4 位指数、3 位尾数，精度更高、范围小（最大 448）；E5M2：5 位指数、2 位尾数，范围大、精度低。推理中用 E4M3（E5M2 主要用于训练时的梯度）。
    2. 模型用这种粒度训练出来，激活和权重里有极端的离群值，逐张量缩放会把正常的数值压没；分块缩放只让离群值影响它所在的那一块，推理也必须按同样的粒度量化，精度才不会掉。
    3. MXFP4：每 32 个元素一个缩放，缩放是 2 的幂（E8M0）；NVFP4：每 16 个元素一个缩放，缩放是 FP8（E4M3），还有一个张量级的缩放。NVFP4 的组更小、缩放更精细，精度更好。
    4. K 往往更敏感：K 常常带有"每个通道一个常数偏置 + 小变化"的结构，按张量量化时偏置占满了量化范围，真正区分不同 token 的小变化被量化误差淹没（本章里 Qwen2.5 的 K 很敏感，有 QK-Norm 的 Qwen3 几乎无损）。对策：校准的缩放因子、更细的粒度，或者按通道带零点量化。
    5. W4A16 只减少读权重的字节，降低低负载时的 TPOT；接近饱和时每一步都混有 prefill，而 prefill 是计算受限的，weight-only 量化不能加速计算，所以服务的容量几乎没变。

## 格式速查

| 格式 | 数值 | 缩放 | 加速了什么 | 硬件 |
| --- | --- | --- | --- | --- |
| W8A8 FP8（E4M3） | 1 符号 + 4 指数 + 3 尾数，最大 448 | 按张量、按通道/按 token、分块（128×128） | 读权重减半，**计算翻倍**（FP8 Tensor Core） | Hopper、Blackwell、MI300 |
| W8A8 INT8 | 整数 −128～127 | 按通道 + 按 token，常配合 SmoothQuant | 同上 | Ampere 及以后 |
| W4A16（INT4 GPTQ/AWQ） | 整数 −8～7 | 每 128 个数一个缩放（+ 零点） | 读权重降为约 1/4，计算仍是 BF16 | 各种 GPU（Marlin 等 kernel） |
| MXFP4 | E2M1：±{0, 0.5, 1, 1.5, 2, 3, 4, 6} | 每 32 个数一个 2 的幂次缩放（E8M0） | 读权重约 1/4；在 Blackwell 上可以用 FP4 计算 | Blackwell 原生；gpt-oss 的发布格式 |
| NVFP4 | E2M1 | 每 16 个数一个 FP8（E4M3）缩放 + 张量级 FP32 缩放 | 同上 | Blackwell 原生 |
| KV Cache FP8 | E4M3 | 按张量（校准）或更细 | KV 显存与读取减半 | 取决于注意力后端 |

推理中的 FP8 几乎都用 **E4M3**：它比 E5M2 多一位尾数，精度更高，范围（±448）配合缩放因子已经够用。E5M2 范围更大、精度更低，主要用于训练中的梯度。

## 伪量化实现

```python title="formats.py"
"""formats.py —— 部署中常见的低精度格式（伪量化实现）：FP8 的几种缩放粒度、MXFP4、NVFP4，以及 FP8 KV Cache。

所有函数都是"量化后立刻反量化"，返回与输入同形状的 FP32 张量，用来评估精度损失。
"""

import torch

from mini_llm import KVCache

FP8_MAX = 448.0                                                     # E4M3 能表示的最大值
FP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])  # E2M1 能表示的全部非负值


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
    scale = 2.0 ** (torch.floor(torch.log2(amax)) - 2)                  # E2M1 的最大指数是 2
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

MXFP4 和 NVFP4 的核心区别在缩放因子：MXFP4 的缩放只能是 2 的幂次，块内最大值与可表示的最大值之间可能差将近一倍，浪费了精度；NVFP4 的缩放本身是一个 FP8 数，可以精确贴合块内最大值，而且块更小（16 个数），离群值影响的范围更小。

## 精度比较

把所有线性层（或 KV Cache）按不同格式伪量化，在大模型手册用过的同一段文本上测困惑度（FP32 基线 25.94，大模型手册中 INT4 按组的结果是 41.49）：

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
            if act_fn is not None:                          # 激活在进入线性层时量化
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

结论：

- **FP8 的缩放粒度越细越好**：按张量 +2.8%，按通道/按 token +2.3%，DeepSeek 式的分块缩放 +1.5%。分块缩放让每 128 个数有自己的量化范围，离群值只影响它所在的块；这也是 DeepSeek-V3 用 FP8 训练与推理的基础。
- **4 位格式里，块越小越好**：NVFP4（16 个数一块、FP8 缩放）+10%，MXFP4（32 个数一块、2 的幂次缩放）+20%，INT4 按组 128 最差（+60%）。MXFP4 的缩放只能取 2 的幂次，块内最大值可能只用到可表示范围的一半多，浪费了精度。Qwen3 对朴素舍入格外敏感（同样的实验在 Qwen2.5-0.5B 上，INT4 按组只损失 22%；有实证研究发现 Qwen3 在低比特下的退化比上一代明显，推测是预训练更充分、权重里的冗余更少）。实际部署中，4 位量化通常配合 GPTQ、AWQ 或量化感知训练（QAT，例如 gpt-oss 与 Kimi K2 Thinking 的做法），而不是本章这样的朴素舍入，精度会好得多。
- **KV Cache 的 FP8 几乎无损**（+0.1%）。这和模型结构有关，并不是所有模型都这样，下面单独分析。

## KV Cache 量化：误差要和信号比

把 K 和 V 分开量化，并试试更细的缩放粒度，再看看 K 的数值分布：

```python
class QuantKVCache(KVCache):
    def __init__(self, num_layers, quant_k, quant_v):
        super().__init__(num_layers)
        self.quant_k, self.quant_v = quant_k, quant_v

    def update(self, layer, k, v):                          # k, v: [B, 头数, T, head_dim]
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
    k, v = cache.k[layer][0], cache.v[layer][0]                 # [头数, T, head_dim]
    spread = k.std(dim=1).median()                              # 每一维在不同 token 之间变化多少（取中位数）
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

K、V 怎么量化，困惑度都几乎不变。可是第 0 层 K 的最大值是中位数的 500 倍，按张量缩放时，大多数值只能用到 FP8 很小的一段范围，为什么没事？

因为 FP8 是浮点格式，3 位尾数决定了相对误差：只要数值不太小（不落进非规格化数区间），误差大约是**数值本身的** 3%～6%。所以要比较的是误差和**有用的信号**：一个维度真正携带信息的是它在不同 token 之间的变化，而不是它的绝对大小。

- **Qwen3**：K 在 RoPE 之前经过 QK-Norm，每一维的数值围绕 0 变化（第 0 层中位数 0.98，随 token 的标准差 1.17），误差和信号同比例，量化几乎无损。最大的那几维来自 `k_norm` 的逐维权重（第 0 层最大 96.5，中位数 2.3），同样随 token 变化，也携带信息；
- **Qwen2.5**（K 投影带偏置、没有 QK-Norm）：同样的实验，只量化 K，困惑度上升 12%（22.14 → 24.73），只量化 V 几乎不变。它第 0 层 K 的中位数是 5.7，随 token 的标准差却只有 0.36：每一维都是"一个大的常数偏置 + 很小的变化"（数值最大的几维恰好落在 RoPE 转得最慢的频率上，旋转之后仍然几乎是常数）。常数偏置对 softmax 没有影响（每个 key 都加了同一个数），误差却按 5.7 的大小产生，和 0.36 的有效信号相比就很大了。按头缩放能缓解一些（损失降到 2% 左右）；更彻底的办法是对 K **按通道、带零点**量化，把每一维的常数偏置吸收进零点——KIVI 等 KV 量化方法对 K 按通道、对 V 按 token 量化，就是这个道理。

实际部署时：

- 用校准数据离线计算 K、V 的缩放因子（写进量化 checkpoint），比运行时按当前批次计算更稳定；
- 注意力后端要支持相应的粒度（按张量最普遍，按头的支持越来越多）；
- MLA 的潜在向量、滑动窗口模型等有各自的特殊处理（vLLM 的 `fp8_ds_mla` 等）；
- 上线前一定要做长上下文的精度评测：KV 误差会随上下文累积。

## 对容量的影响

量化让服务多扛多少流量？用[上一章](benchmark.md)的模拟器估算 Qwen2.5-7B 在单张 H100 上、同样 SLO 下的最大请求速率：

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

这张表很好地说明了"量化加速什么"：

- **W4A16 让低负载的 TPOT 降到约 1/3**（只读 1/3 的字节），对延迟敏感、并发不高的场景价值很大；但**服务容量几乎没变**：接近饱和时，每一步都混有 prefill，而 prefill 是计算受限的，weight-only 量化不能加速计算；
- **W8A8 FP8 让 TTFT 减半、容量翻倍**：它同时减少了读取的字节和计算时间（FP8 Tensor Core 的算力是 BF16 的 2 倍）；
- **KV FP8** 在此基础上再多扛一些：KV 读取减半、能放下的上下文翻倍。

所以选择量化方案要先看瓶颈：延迟优先、并发低，W4A16 很合适；吞吐优先、负载高，要选 W8A8 FP8（或 Blackwell 上的 FP4 计算）。

## 在推理引擎中使用

```bash
# vLLM：在线把 BF16 权重量化成 FP8（无需校准，最省事）
vllm serve Qwen/Qwen2.5-7B-Instruct --quantization fp8 --kv-cache-dtype fp8
# vLLM：加载预先量化好的 checkpoint（格式由 checkpoint 的 config 自动识别）
vllm serve /path/to/Qwen2.5-7B-Instruct-W4A16               # 例如 LLM Compressor 导出的 compressed-tensors 格式
# SGLang
python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --quantization fp8 --kv-cache-dtype fp8_e4m3
```

预量化的 checkpoint 通常由这些工具产生：**LLM Compressor**（输出 compressed-tensors 格式，支持 GPTQ、AWQ、SmoothQuant、FP8、NVFP4 等）、**NVIDIA ModelOpt**（FP8、NVFP4，面向 TensorRT-LLM 与 vLLM/SGLang）、AutoAWQ/GPTQModel 等。量化的精度评测建议分三步：困惑度快速检查 → 通用基准（MMLU、GSM8K 等，用 lm-evaluation-harness）→ 业务自己的评测集，并单独评测长上下文和代码、数学等敏感任务。

!!! source "源码对照"
    - **vLLM**：量化方法在 `vllm/model_executor/layers/quantization/`：`fp8.py`（包括分块 FP8）、`compressed_tensors/`（LLM Compressor 的格式）、`modelopt.py`（ModelOpt 的 FP8/NVFP4）、`mxfp4.py`（gpt-oss）、`auto_gptq.py`、`auto_awq.py`、`kv_cache.py`（KV 缩放因子）等；每种方法把自己的 kernel 接入线性层和 MoE 层。`--kv-cache-dtype` 可选 `fp8`（即 `fp8_e4m3`）、`fp8_e5m2`，以及面向 DeepSeek MLA 的 `fp8_ds_mla`、`nvfp4_ds_mla` 等。
    - **SGLang**：`srt/layers/quantization/` 下有对应的实现，`--quantization` 与 `--kv-cache-dtype`（`fp8_e4m3`、`fp8_e5m2` 等）用法类似。

!!! interview "面试怎么答"
    "线上服务要不要量化？选哪种？"——先问瓶颈与目标：延迟优先还是吞吐优先，硬件是什么。然后用本章的逻辑回答：weight-only（W4A16）减少读权重，改善低负载的 TPOT，不加速 prefill、不提高饱和吞吐；W8A8（FP8/INT8）同时加速计算，提高容量；KV 量化增加并发与长上下文能力，但要看 K 的数值分布（常数偏置、离群值）。最后一定要说精度验证流程和回滚方案。能说出"分块缩放""NVFP4 与 MXFP4 的区别""为什么有的模型 K 对量化很敏感、有的不敏感"这类细节，是加分项。

!!! info "相关章节"
    - [量化](llm://inference/quantization/)（大模型原理：量化的数学与误差）
    - [量化与 GEMV](cuda://advanced/quantization/)（CUDA：量化 kernel 怎么写）
    - [FP8 细粒度量化与分组 GEMM](../moe/fp8-gemm.md)、[超大 MoE 的低比特推理](../frontier/low-bit.md)（本书）

## 练习

**1. 为什么不量化这些层？** 很多量化方案会保留嵌入层、输出层（lm_head）、MoE 的路由器为 BF16。为什么？

??? success "参考答案"
    - 嵌入层是查表，量化不能加速计算，只能省显存，而嵌入误差会影响所有后续计算；
    - 输出层直接决定 logits，误差会直接改变 token 的排序，而且词表很大时它的数值分布很宽；小模型的输出层占参数比例大，有时会量化，需要单独评测；
    - MoE 路由器的输出决定 token 去哪个专家，微小的误差可能改变 top-k 的选择，导致离散的大误差；路由器的参数量很小，保留高精度几乎没有代价。

**2. 分块 FP8 的代价。** DeepSeek 式的分块 FP8 精度最好，它的代价是什么？

??? success "参考思路"
    GEMM 的累加必须按块处理缩放因子：Tensor Core 在一个 128 的 K 维块内用 FP8 相乘、累加，然后乘上这一块激活和权重的缩放因子，再累加到 FP32 的结果中（DeepSeek 的 DeepGEMM 在 CUDA Core 上做这一步提升）。这比按张量缩放的 GEMM 复杂，kernel 需要专门实现，效率也略低；缩放因子本身也要占用存储和带宽（很小）。另外，激活的按 (1, 128) 缩放需要在上一个 kernel 的末尾顺便算出，才能避免额外的读写。

## 小结

- [x] 推理中的 FP8 用 E4M3；缩放粒度决定精度：按张量 < 按通道/按 token < 分块。
- [x] 4 位格式中 NVFP4（16 个数一个 FP8 缩放）优于 INT4 按组，MXFP4（2 的幂次缩放）最差；实际部署配合 GPTQ/AWQ/QAT。
- [x] KV Cache 量化的误差要和信号比：K 是"常数偏置 + 小变化"时（Qwen2.5）很敏感，有 QK-Norm 时（Qwen3）几乎无损；对策是校准的缩放因子、更细的粒度或按通道带零点量化，并做长上下文评测。
- [x] W4A16 降低低负载延迟、几乎不提高饱和容量；W8A8 FP8 同时加速计算，容量翻倍；先找瓶颈再选方案。
