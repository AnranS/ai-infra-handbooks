# 参数量、算力与显存估算

<p class="lead">"这个模型要几张卡？""这台机器能撑多少并发？""decode 最快能多快？"这些问题在推理工作中每天都会遇到，而且都能用几个简单的公式估算出来。这一章把前面各章的计算整理成一套估算方法，写成一个小工具，并用官方模型配置核对。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 给你一个 `config.json`，怎么算出参数量？
    2. 每个 token 的前向计算量是多少？注意力部分怎么算？
    3. 部署一个模型，显存由哪几部分组成？
    4. decode 每个 token 的延迟下限怎么估算？prefill 呢？
    5. 为什么 70B 模型在 8 张卡上做张量并行后，decode 延迟可以和单卡跑 8B 差不多？
    6. 实测的 decode 速度只有带宽下限的一半，时间可能花在哪里？怎么查？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 嵌入层 $V \times d$（不共享时输出层再算一份）+ 每层（q、k、v、o 四个投影按头数和头维算 + FFN 的 $3 \cdot d \cdot d_{ff}$ + 两个归一化的 $2d$）× 层数 + 最后的归一化 $d$；MoE 按专家数展开 FFN。
    2. 约 $2N$（N 是参与计算的参数量，嵌入查表不算）；注意力的点积和加权求和另算，每层约 $4 \cdot d \cdot$ 上下文长度，与上下文长度成正比。
    3. 权重 + KV Cache + 激活（中间结果）+ 运行时开销（CUDA 上下文、通信缓冲区、CUDA Graph 等）；剩下给 KV 的空间决定了能有多少并发。
    4. decode 下限 = (权重字节 + 本步要读的 KV 字节) ÷ 显存带宽；prefill ≈ $2N \times$ token 数 ÷ (峰值算力 × MFU)。
    5. 张量并行后每张卡只存、只读 1/8 的权重：70B × 2 字节 ÷ 8 ≈ 17.5 GB，和单卡读 8B 的 16 GB 差不多；再加上每层两次 all-reduce 的开销。
    6. 先拆开：把所有线性层单独跑一遍，看读权重本身离下限多远；剩下的时间花在 RMSNorm、RoPE、softmax、写缓存这些小算子和调度上，它们几乎不做计算，但个数多、每个都有固定开销。GPU 上靠 CUDA Graph 一次性发射一步的所有 kernel、再把小算子融合起来去掉这部分开销。本章在 CPU 上实测，线性层只比下限多一成多，另外一半时间都是这类开销。

先看一个六格小剧场，再读正文：

![漫画：从 config.json 算到延迟下限](../assets/comics/estimation.webp){.aig-comic}

## 参数量

LLaMA/Qwen 结构的稠密模型，参数由这几部分组成：

$$
N = \underbrace{Vd}_{\text{嵌入}} + \underbrace{Vd}_{\text{输出层（不共享时）}} + L \Big[\underbrace{d\,n_h d_h + 2\,d\,n_{kv} d_h + n_h d_h\, d}_{\text{注意力 q, k, v, o}} + \underbrace{3\,d\,d_{ff}}_{\text{SwiGLU}} + \underbrace{2d}_{\text{两个 RMSNorm}}\Big] + \underbrace{d}_{\text{最终 RMSNorm}}
$$

Qwen2 的 q、k、v 投影还有偏置（$n_h d_h + 2 n_{kv} d_h$）；Qwen3 去掉了偏置，但每层多了 QK-Norm 的两个权重（$2 d_h$）。写成代码，直接读取 `config.json` 的字段：

```python title="estimate.py"
"""estimate.py —— 从模型配置估算参数量、计算量、显存和延迟下限（稠密的 LLaMA / Qwen2 / Qwen3 结构）。"""

from dataclasses import dataclass


def count_params(c: dict) -> int:
    V, d, L, dff = c["vocab_size"], c["hidden_size"], c["num_hidden_layers"], c["intermediate_size"]
    nh = c["num_attention_heads"]
    nkv = c.get("num_key_value_heads", nh)
    hd = c.get("head_dim") or d // nh
    attn = d * nh * hd + 2 * d * nkv * hd + nh * hd * d
    if c.get("attention_bias", c.get("model_type") == "qwen2"):
        attn += nh * hd + 2 * nkv * hd
    if c.get("model_type") == "qwen3":
        attn += 2 * hd                              # QK-Norm：q_norm、k_norm 各一个长度为 d_h 的权重
    layer = attn + 3 * d * dff + 2 * d
    emb = V * d * (1 if c.get("tie_word_embeddings", False) else 2)
    return emb + L * layer + d


def kv_bytes_per_token(c: dict, bytes_per_elem: float = 2) -> float:
    nh = c["num_attention_heads"]
    hd = c.get("head_dim") or c["hidden_size"] // nh
    return 2 * c["num_hidden_layers"] * c.get("num_key_value_heads", nh) * hd * bytes_per_elem


def flops_per_token(c: dict, context: int) -> float:
    """前向计算量：线性层 2 × 参数（不含嵌入查表），加上注意力的 QK^T 与 PV（与上下文长度成正比）。"""
    V, d = c["vocab_size"], c["hidden_size"]
    nh = c["num_attention_heads"]
    hd = c.get("head_dim") or d // nh
    # 嵌入是查表，不算乘法；输出层要算。共享词嵌入时只有一份 V × d，它还要当输出层做乘法，不能减
    lookup = 0 if c.get("tie_word_embeddings", False) else V * d
    linear = 2 * (count_params(c) - lookup)
    attention = 2 * 2 * c["num_hidden_layers"] * context * nh * hd
    return linear + attention


@dataclass
class GPU:
    name: str
    mem_gb: float
    bw_tbs: float        # 显存带宽 TB/s
    tflops: float        # BF16 稠密峰值


H100 = GPU("H100 SXM", 80, 3.35, 989)
A100 = GPU("A100 80GB", 80, 2.039, 312)


def decode_latency_ms(c, batch, context, gpu: GPU, n_gpus=1, weight_bytes=2, kv_bytes=2):
    """decode 一步的延迟下限：每步至少读一遍权重和全部请求的 KV Cache（假设完美的张量并行）。"""
    total = count_params(c) * weight_bytes + batch * context * kv_bytes_per_token(c, kv_bytes)
    return total / (gpu.bw_tbs * 1e12 * n_gpus) * 1e3


def prefill_ms(c, tokens, gpu: GPU, n_gpus=1, mfu=0.5):
    """prefill 的延迟估计：总计算量 / (峰值 × 利用率)。注意力按平均上下文 tokens/2 估算。"""
    return tokens * flops_per_token(c, tokens // 2) / (gpu.tflops * 1e12 * mfu * n_gpus) * 1e3
```

用官方配置核对：用 transformers 在 meta 设备上构建模型（不分配内存），数出真实的参数量，和公式比较：

```python
import json
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from estimate import count_params

configs = {
    "Qwen3-0.6B": json.load(open("models/Qwen3-0.6B/config.json")),
    "LLaMA-2-7B": dict(model_type="llama", vocab_size=32000, hidden_size=4096, intermediate_size=11008,
                       num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=32),
    "LLaMA-3-8B": dict(model_type="llama", vocab_size=128256, hidden_size=4096, intermediate_size=14336,
                       num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8),
    "LLaMA-3-70B": dict(model_type="llama", vocab_size=128256, hidden_size=8192, intermediate_size=28672,
                        num_hidden_layers=80, num_attention_heads=64, num_key_value_heads=8),
    "Qwen2.5-7B": dict(model_type="qwen2", vocab_size=152064, hidden_size=3584, intermediate_size=18944,
                       num_hidden_layers=28, num_attention_heads=28, num_key_value_heads=4),
    "Qwen2.5-72B": dict(model_type="qwen2", vocab_size=152064, hidden_size=8192, intermediate_size=29568,
                        num_hidden_layers=80, num_attention_heads=64, num_key_value_heads=8),
}
for name, c in configs.items():
    cfg = AutoConfig.for_model(**{k: v for k, v in c.items() if k not in ("architectures", "torch_dtype", "transformers_version")})
    with torch.device("meta"):
        real = sum(p.numel() for p in AutoModelForCausalLM.from_config(cfg).parameters())
    ours = count_params(c)
    print(f"{name:14s} 公式 {ours / 1e9:7.3f}B   transformers {real / 1e9:7.3f}B")
    assert ours == real
```

```text title="输出"
Qwen3-0.6B     公式   0.596B   transformers   0.596B
LLaMA-2-7B     公式   6.738B   transformers   6.738B
LLaMA-3-8B     公式   8.030B   transformers   8.030B
LLaMA-3-70B    公式  70.554B   transformers  70.554B
Qwen2.5-7B     公式   7.616B   transformers   7.616B
Qwen2.5-72B    公式  72.706B   transformers  72.706B
```

每一个都与 transformers 构建出的模型精确相等。熟练之后，看一眼配置就能心算出大概：**层数 × (4d² 左右的注意力 + 3d·d_ff 的 FFN) + 词表 × d**。

## 计算量

[前面](../basics/math-torch.md#矩阵乘法线性层)说过线性层每个参数每个 token 贡献 2 次运算。再加上注意力本身（每层 $QK^\top$ 和 $PV$ 各 $2 \times \text{上下文长度} \times n_h d_h$）：

$$
\text{FLOPs/token} \approx 2N + 4 L \cdot S \cdot n_h d_h
$$

S 是当前的上下文长度。上下文较短时第一项占绝对主导，常说的"每个 token 约 2N 次运算"就是这么来的；上下文很长时第二项不可忽略：

```python
from estimate import flops_per_token

c = configs["LLaMA-3-8B"]
for S in (1_000, 8_000, 32_000, 128_000):
    f = flops_per_token(c, S)
    attn_share = 1 - flops_per_token(c, 0) / f
    print(f"上下文 {S:>7,d}: {f / 1e9:6.1f} GFLOPs/token，其中注意力占 {attn_share:.0%}")
```

```text title="输出"
上下文   1,000:   15.5 GFLOPs/token，其中注意力占 3%
上下文   8,000:   19.2 GFLOPs/token，其中注意力占 22%
上下文  32,000:   31.8 GFLOPs/token，其中注意力占 53%
上下文 128,000:   82.1 GFLOPs/token，其中注意力占 82%
```

LLaMA-3-8B 的 2N 约 15 GFLOPs。上下文 1000 时注意力只占 3%，到 32K 就超过了所有线性层，128K 时占八成以上。长上下文的 prefill 越来越慢，主要慢在这一项上。

### 实验：让 PyTorch 数一数

公式对不对，可以让 PyTorch 自己数：`FlopCounterMode` 会记下前向计算中每一次矩阵乘的运算量。拿第 4 章写的 [mini_llm](../transformer/build-llm.md) 加载 Qwen3-0.6B，前面放好 S 个 token 的 KV Cache，数 decode 一步的计算量，和公式比：

```python
from torch.utils.flop_counter import FlopCounterMode
from mini_llm import KVCache, Transformer

model = Transformer.from_pretrained("models/Qwen3-0.6B")
c = configs["Qwen3-0.6B"]
L, n_kv, d_h = c["num_hidden_layers"], c["num_key_value_heads"], c["head_dim"]
for S in (16, 1024, 8192):
    cache = KVCache(L)
    cache.k = [torch.randn(1, n_kv, S, d_h) for _ in range(L)]      # 只数计算量，缓存里放随机数就行
    cache.v = [torch.randn(1, n_kv, S, d_h) for _ in range(L)]
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        model(torch.tensor([[0]]), cache)                          # decode 一步：1 个新 token，前面已有 S 个
    counts = counter.get_flop_counts()["Global"]
    real = counter.get_total_flops()
    ours = flops_per_token(c, S + 1)                               # 加上新 token 自己，注意力看 S + 1 个位置
    print(f"上下文 {S + 1:5d}：PyTorch 数出 {real / 1e9:.3f} GFLOPs（线性层 {counts[torch.ops.aten.mm] / 1e9:.3f}，"
          f"注意力 {counts[torch.ops.aten.bmm] / 1e9:.3f}），公式 {ours / 1e9:.3f}，相差 {ours / real - 1:+.2%}")
```

```text title="输出"
上下文    17：PyTorch 数出 1.196 GFLOPs（线性层 1.192，注意力 0.004），公式 1.196，相差 +0.01%
上下文  1025：PyTorch 数出 1.427 GFLOPs（线性层 1.192，注意力 0.235），公式 1.427，相差 +0.01%
上下文  8193：PyTorch 数出 3.071 GFLOPs（线性层 1.192，注意力 1.879），公式 3.071，相差 +0.00%
```

- 线性层的计算量不随上下文变，正好是 2 × 参数量；注意力随上下文线性增长，8K 上下文时已经超过所有线性层加起来；
- 公式比 PyTorch 数的多 0.01%：公式把 RMSNorm 的权重也算进了参数量，它们做的是逐元素乘法，不在矩阵乘的计数里；
- 共享词嵌入（`tie_word_embeddings`）的模型要小心：嵌入矩阵只存一份，但输出层还要拿它做一次矩阵乘法。如果按"减掉嵌入"来算，就把 Qwen3-0.6B 的输出层（1.56 亿参数）也减掉了，计算量少算约四分之一。`flops_per_token` 里的 `lookup` 就是处理这件事的。

## 显存

部署时的显存大致由四部分组成：

| 部分 | 大小 | 说明 |
| --- | --- | --- |
| 权重 | 参数量 × 每参数字节数 | BF16 为 2，FP8/INT8 为 1，INT4 约 0.5（另加缩放因子） |
| KV Cache | 并发请求数 × 上下文长度 × 每 token KV 字节数 | 占用剩余显存的大部分，决定并发能力 |
| 激活 | 与 batch 中的 token 数成正比 | 推理时不保存中间结果，通常只需几 GB 的工作区 |
| 其他 | CUDA 上下文、CUDA Graphs、通信缓冲区等 | 通常预留几 GB |

vLLM 的 `gpu_memory_utilization`（默认 0.9）就是"权重 + 激活 + KV Cache 总共最多用多少比例的显存"，扣掉权重和激活之后，剩下的全部预分配给 KV Cache 的块池。

## 延迟的下限

**decode**：每一步至少要把权重和所有请求的 KV Cache 从显存读一遍：

$$
\text{TPOT} \ge \frac{\text{权重字节数} + \text{batch} \times \text{上下文} \times \text{每 token KV 字节数}}{\text{显存带宽}}
$$

**prefill**：主要受算力限制：

$$
\text{TTFT} \approx \frac{T \times \text{FLOPs/token}}{\text{峰值算力} \times \text{MFU}}
$$

用这两个公式评估几个部署方案：

```python
from estimate import H100, decode_latency_ms, kv_bytes_per_token, prefill_ms

def report(name, c, batch, ctx, n_gpus=1, weight_bytes=2):
    w_gb = count_params(c) * weight_bytes / 1e9
    kv_gb = batch * ctx * kv_bytes_per_token(c) / 1e9
    tpot = decode_latency_ms(c, batch, ctx, H100, n_gpus, weight_bytes)
    print(f"{name:28s} 权重 {w_gb:6.1f} GB，KV {kv_gb:6.1f} GB，TPOT ≥ {tpot:5.1f} ms，"
          f"单请求 ≤ {1000 / tpot:5.0f} tok/s，总吞吐 ≤ {batch * 1000 / tpot:6.0f} tok/s")

report("8B，1×H100，batch 1", configs["LLaMA-3-8B"], 1, 4096)
report("8B，1×H100，batch 64", configs["LLaMA-3-8B"], 64, 4096)
report("8B INT4 权重，batch 1", configs["LLaMA-3-8B"], 1, 4096, weight_bytes=0.5)
report("70B，8×H100 TP，batch 1", configs["LLaMA-3-70B"], 1, 4096, n_gpus=8)
report("70B，8×H100 TP，batch 64", configs["LLaMA-3-70B"], 64, 4096, n_gpus=8)
print(f"8B prefill 2000 个 token，1×H100，MFU 50%：TTFT ≈ {prefill_ms(configs['LLaMA-3-8B'], 2000, H100):.0f} ms")
```

```text title="输出"
8B，1×H100，batch 1            权重   16.1 GB，KV    0.5 GB，TPOT ≥   5.0 ms，单请求 ≤   202 tok/s，总吞吐 ≤    202 tok/s
8B，1×H100，batch 64           权重   16.1 GB，KV   34.4 GB，TPOT ≥  15.1 ms，单请求 ≤    66 tok/s，总吞吐 ≤   4252 tok/s
8B INT4 权重，batch 1           权重    4.0 GB，KV    0.5 GB，TPOT ≥   1.4 ms，单请求 ≤   736 tok/s，总吞吐 ≤    736 tok/s
70B，8×H100 TP，batch 1        权重  141.1 GB，KV    1.3 GB，TPOT ≥   5.3 ms，单请求 ≤   188 tok/s，总吞吐 ≤    188 tok/s
70B，8×H100 TP，batch 64       权重  141.1 GB，KV   85.9 GB，TPOT ≥   8.5 ms，单请求 ≤   118 tok/s，总吞吐 ≤   7556 tok/s
8B prefill 2000 个 token，1×H100，MFU 50%：TTFT ≈ 63 ms
```

几个值得注意的结论：

- **batch 从 1 增加到 64，总吞吐提高了约 21 倍（202 → 4252 tok/s），TPOT 下限只从 5 ms 增加到 15 ms**，这就是[批处理](kv-cache.md#批处理让多个请求分摊权重读取)的威力。增加的延迟几乎全部来自 KV Cache：64 个 4K 上下文的请求，KV 共 34 GB，是权重的两倍多；
- **INT4 权重量化让 batch 1 的 decode 下限从 5.0 ms 降到 1.4 ms**，因为此时几乎全部时间都在读权重，见[量化原理](quantization.md)；
- **70B 在 8 卡张量并行下，单请求的 decode 下限（5.3 ms）和单卡跑 8B（5.0 ms）几乎一样**：8 张卡的带宽加在一起，每张卡只读 1/8 的权重。实际上还要加上每层两次 all-reduce 的通信延迟，所以 TP 不是免费的，参见 CUDA 手册的[多 GPU 与 NCCL](cuda://tools/multi-gpu/)；
- 这些都是**理论下限**。实际系统能达到带宽利用率的 70%-85% 就很不错了。

把这几个公式做成一个计算器：换模型、换卡、换精度、拉 batch 和上下文，看参数量、每 token 的计算量、decode 一步要读多少字节、TPOT 和 TTFT 的下限怎么变：

<div class="aig-widget" data-widget="estimator"></div>

### 实验：下限和实测差多少

公式给的是下限，真实的程序能跑到下限的几成？在本手册用的这台 CPU 上走一遍完整的流程：先测出机器的峰值算力和内存带宽，算出 Qwen3-0.6B decode 一步和 prefill 的下限，再和 mini_llm 的实测比：

```python
import time

torch.set_num_threads(16)

def best_ms(f, reps=5):
    f()                                                              # 预热
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - t0)
    return best * 1000

a, b = torch.randn(8192, 8192), torch.randn(8192, 8192)
peak = 2 * 8192**3 / best_ms(lambda: a @ b) / 1e6                   # GFLOP/s：大矩阵乘能把算力用满
W, v = torch.randn(16384, 16384), torch.randn(16384)
bw = W.numel() * 4 / best_ms(lambda: W @ v, reps=20) / 1e6          # GB/s：1 GiB 的矩阵乘向量，每个数只用一次，和 decode 读权重一样
del a, b, W
print(f"这台机器：峰值算力 {peak:.0f} GFLOP/s，内存带宽 {bw:.0f} GB/s，脊点 {peak / bw:.0f} FLOP/Byte")

weight_bytes = count_params(c) * 4                                   # mini_llm 用 FP32，每个参数 4 字节

@torch.no_grad()
def decode_step_ms(steps=10):
    cache = KVCache(L)
    nxt = model(torch.randint(0, 150000, (1, 64)), cache)[:, -1].argmax(-1)
    t0 = time.perf_counter()
    for _ in range(steps):
        nxt = model(nxt[:, None], cache)[:, -1].argmax(-1)
    return (time.perf_counter() - t0) / steps * 1000

decode_step_ms(2)                                                    # 预热
t_dec, bound = decode_step_ms(), weight_bytes / bw / 1e6
print(f"decode 一步：下限 {bound:.1f} ms（读一遍 {weight_bytes / 1e9:.2f} GB 权重），实测 {t_dec:.1f} ms，带宽利用率 {bound / t_dec:.0%}")

linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]   # 28 层 × 7 个，加上输出层
xs = [torch.randn(1, m.in_features) for m in linears]
with torch.no_grad():
    t_mm = best_ms(lambda: [m(x) for m, x in zip(linears, xs)], reps=10)
print(f"  其中 {len(linears)} 个线性层单独跑要 {t_mm:.1f} ms，剩下的 {t_dec - t_mm:.1f} ms 花在其他小算子和 Python 上")

T = 512
ids = torch.randint(0, 150000, (1, T))
with torch.no_grad():
    t_pre = best_ms(lambda: model(ids, KVCache(L)), reps=3)
bound = T * flops_per_token(c, T // 2) / peak / 1e6
print(f"prefill {T} 个 token：下限 {bound:.0f} ms，实测 {t_pre:.0f} ms，算力利用率 {bound / t_pre:.0%}")
```

在本手册的环境里（数值因机器而异）：

```text
这台机器：峰值算力 2347 GFLOP/s，内存带宽 104 GB/s，脊点 23 FLOP/Byte
decode 一步：下限 22.9 ms（读一遍 2.38 GB 权重），实测 46.6 ms，带宽利用率 49%
  其中 197 个线性层单独跑要 26.7 ms，剩下的 19.9 ms 花在其他小算子和 Python 上
prefill 512 个 token：下限 273 ms，实测 538 ms，算力利用率 51%
```

- 这台 CPU 的脊点约 23 FLOP/Byte。mini_llm 的权重是 FP32，每个参数 4 字节，batch 为 B 时每读 4 字节做 2B 次运算，算术强度是 B/2，要 batch 四五十才到脊点。[KV Cache](kv-cache.md#batch-加到多大就不再免费) 一章实测，batch 过了 64，吞吐就涨得很慢了，和这个估计大致对得上；
- decode 只跑到了下限的一半。把 197 个线性层单独拿出来跑要 26.7 ms，只比下限多一成多：读权重这部分已经接近跑满带宽。多出来的 20 ms 左右花在 RMSNorm、RoPE、softmax、拼接缓存这些小算子和 Python 的调度上：它们几乎不做计算，但个数多，每一个都有固定的开销。GPU 上也一样，decode 一步要发射几百个 kernel，每个都有启动开销，所以推理引擎要用 CUDA Graph 把一步的 kernel 录下来一次性发射，还要把小算子融合起来；
- prefill 的算力利用率也只有一半左右：mini_llm 的注意力先算出整个 T × T 的分数，再把一半掩掉，softmax、RMSNorm 这些逐元素的算子也要花时间。FlashAttention 这类 kernel 就是为了把注意力这部分做快；
- 差距主要因为 mini_llm 是教学实现，不是推理引擎。估算的用法正是这样：先算出下限，再和实测比，差得多就去找时间花在了哪里。

!!! inference "推理视角"
    估算是推理优化工作的起点：先算出理论上限，再和实测比较。如果实测 TPOT 是理论下限的 3 倍，说明还有很大的优化空间（kernel 不够快？调度开销？CPU 瓶颈？）；如果已经达到下限的 80%，想再快就只能改变"读多少字节"本身：量化、更大的 batch、投机解码。

!!! interview "怎么讲清楚"
    讲估算：按模板算就行——参数量从 `config.json` 算（嵌入 + 每层注意力与 FFN）；每个 token 的计算约 2N，再加上与上下文成正比的注意力项；显存 = 权重 + KV + 激活 + 运行时开销；decode 延迟下限 = (权重 + KV) 字节 ÷ 带宽，prefill ≈ 计算量 ÷ (算力 × MFU)。例如 LLaMA-3-8B 在 H100 上 batch 1：16 GB ÷ 3.35 TB/s ≈ 4.8 ms/token。70B 做 8 卡张量并行后每张卡只读 1/8 的权重，所以 decode 延迟和单卡 8B 相近，再加上 all-reduce 的开销。

!!! note "延伸阅读"
    - [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)（kipply）：本章这套估算最常被引用的出处，从 KV Cache 的大小一路算到 batch、张量并行和延迟，并和 FasterTransformer 的实测对照；
    - [LLM Inference Performance Engineering: Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)（Databricks）：提出 MBU（模型带宽利用率），也就是上面实验里的"带宽利用率"，并给出了真实 GPU 上的测量；
    - [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)（Horace He）：把耗时分成计算、访存、额外开销三类，上面 decode 实验里那十几毫秒就属于第三类；
    - [All About Rooflines](https://jax-ml.github.io/scaling-book/roofline/)（How To Scale Your Model）：屋顶线模型的系统讲解，配有大量练习；
    - [Transformer Math 101](https://blog.eleuther.ai/transformer-math/)（EleutherAI）：训练侧的同一套估算，计算量 6N、优化器状态和激活的显存；
    - [LLM Inference Unveiled: Survey and Roofline Model Insights](https://arxiv.org/abs/2402.16363)：用屋顶线模型逐层分析各种推理优化手段，附带一个分析工具 LLM-Viewer。

## 练习

**1. 部署规划。** 要用 BF16 部署 Qwen2.5-72B，要求支持 32 个并发请求、每个请求 8K 上下文。至少需要几张 80 GB 的 GPU？（为激活和其他开销每张卡预留 8 GB。）

??? success "参考答案"
    ```python
    import math
    c = configs["Qwen2.5-72B"]
    need = count_params(c) * 2 + 32 * 8192 * kv_bytes_per_token(c)
    n = math.ceil(need / ((80 - 8) * 1e9))
    print(f"权重 {count_params(c) * 2 / 1e9:.0f} GB + KV {32 * 8192 * kv_bytes_per_token(c) / 1e9:.0f} GB -> 至少 {n} 张")
    ```

    权重约 145 GB，KV 约 86 GB，总共约 231 GB，至少需要 4 张卡；张量并行的卡数一般取 2 的幂且能整除注意力头数，所以实际会用 4 张或 8 张。

**2. 思考题。** 为什么说"decode 的延迟下限和模型的算力无关"？什么时候这句话不成立？

??? success "参考答案"
    decode 的 batch 较小时，计算量远小于算力上限，耗时由读取的字节数决定，算力再强也无济于事。当 batch 大到每步的计算时间超过读取时间时（算术强度超过脊点，比如 H100 上 BF16 的 batch 达到几百），decode 就变成了计算瓶颈，这时算力才重要。MoE 模型、MLA、投机解码都会让 decode 的算术强度更高，更早地进入计算瓶颈区域。

## 小结

- [x] 稠密模型参数量 = 词表 × d（×2 若不共享）+ 层数 × (注意力 + 3·d·d_ff + 2d) + d，可以精确计算。
- [x] 每个 token 的计算量约 2N，再加上与上下文长度成正比的注意力项。
- [x] 用 PyTorch 的 `FlopCounterMode` 核对计算量公式；共享词嵌入的模型，输出层的那次矩阵乘法也要算上。
- [x] 显存 = 权重 + KV Cache + 激活 + 其他；KV Cache 决定并发。
- [x] decode 延迟下限 = (权重 + KV) 字节数 / 带宽；prefill 延迟 ≈ 计算量 / (算力 × MFU)。
- [x] 先算理论上限再看实测，是推理优化的基本方法。本章在 CPU 上实测：decode 只到带宽下限的一半，线性层本身接近下限，另一半是小算子和调度的开销。

相关的数学：算术强度与屋顶线、Amdahl 定律见[性能与服务中的数学](math://performance-math/)。
