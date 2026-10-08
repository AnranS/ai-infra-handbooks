# Estimating parameters, compute and memory

<p class="lead">"How many GPUs does this model need?" "How much concurrency can this machine handle?" "How fast can decode possibly be?" Questions like these come up every day in inference work, and all of them can be estimated with a few simple formulas. This chapter organizes the calculations of the previous chapters into one estimation method, writes it up as a small tool, and checks it against official model configs.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. Given a `config.json`, how do you compute the parameter count?
    2. What is the forward compute per token? How is the attention part computed?
    3. When deploying a model, what parts make up the memory?
    4. How do you estimate the lower bound on decode latency per token? And prefill?
    5. Why can a 70B model under 8-way tensor parallelism decode about as fast as an 8B model on one GPU?

??? success "Answers (try first, then expand to compare)"
    1. Embedding $V \times d$ (counted again for the output layer if not tied) + per layer (the four projections q, k, v, o by head count and head dimension + the FFN's $3 \cdot d \cdot d_{ff}$ + $2d$ for the two norms) × layers + $d$ for the final norm; for MoE, expand the FFN by the number of experts.
    2. About $2N$ (N being the parameters that take part in computation; the embedding lookup does not count); the attention dot products and weighted sums are counted separately, about $4 \cdot d \cdot$ context length per layer, proportional to the context length.
    3. Weights + KV cache + activations (intermediate results) + runtime overhead (CUDA context, communication buffers, CUDA Graphs and so on); the space left for KV decides how much concurrency there can be.
    4. The decode lower bound = (weight bytes + KV bytes read this step) ÷ memory bandwidth; prefill ≈ $2N \times$ tokens ÷ (peak compute × MFU).
    5. With tensor parallelism each GPU stores and reads only 1/8 of the weights: 70B × 2 bytes ÷ 8 ≈ 17.5 GB, about the same as the 16 GB one GPU reads for 8B; on top of that come two all-reduces per layer.

<!-- comic ../assets/comics/estimation.webp is in Chinese; put it back once the English version exists -->

## Parameter count {#参数量}

The parameters of a dense model with the LLaMA/Qwen architecture consist of:

$$
N = \underbrace{Vd}_{\text{embedding}} + \underbrace{Vd}_{\text{output layer (if not tied)}} + L \Big[\underbrace{d\,n_h d_h + 2\,d\,n_{kv} d_h + n_h d_h\, d}_{\text{attention q, k, v, o}} + \underbrace{3\,d\,d_{ff}}_{\text{SwiGLU}} + \underbrace{2d}_{\text{two RMSNorms}}\Big] + \underbrace{d}_{\text{final RMSNorm}}
$$

Qwen2's q, k and v projections also have biases ($n_h d_h + 2 n_{kv} d_h$); Qwen3 drops the biases but adds QK-Norm's two weights per layer ($2 d_h$). As code that reads the fields of `config.json` directly:

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
        attn += 2 * hd                              # QK-Norm: q_norm and k_norm each have a weight of length d_h
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
    linear = 2 * (count_params(c) - V * d)          # the embedding is a lookup with no multiplication; the output layer counts
    attention = 2 * 2 * c["num_hidden_layers"] * context * nh * hd
    return linear + attention


@dataclass
class GPU:
    name: str
    mem_gb: float
    bw_tbs: float        # memory bandwidth, TB/s
    tflops: float        # dense BF16 peak


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

Check against official configs: build each model on the meta device with transformers (no memory allocated), count the real parameters, and compare with the formula:

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

Every one matches the model transformers builds exactly. With practice you can estimate it in your head from a glance at the config: **layers × (attention of about 4d² + FFN of 3d·d_ff) + vocabulary × d**.

## Compute {#计算量}

As [mentioned earlier](../basics/math-torch.md#矩阵乘法线性层), each parameter of a linear layer contributes 2 operations per token. Add attention itself ($QK^\top$ and $PV$ at $2 \times \text{context length} \times n_h d_h$ each, per layer):

$$
\text{FLOPs/token} \approx 2N + 4 L \cdot S \cdot n_h d_h
$$

S is the current context length. With short contexts the first term dominates completely, which is where the common "about 2N operations per token" comes from; with long contexts the second term cannot be ignored:

```python
from estimate import flops_per_token

c = configs["LLaMA-3-8B"]
for S in (1_000, 8_000, 32_000, 128_000):
    f = flops_per_token(c, S)
    attn_share = 1 - flops_per_token(c, 0) / f
    print(f"上下文 {S:>7,d}: {f / 1e9:6.1f} GFLOPs/token，其中注意力占 {attn_share:.0%}")
```

## Memory {#显存}

At deployment, memory roughly consists of four parts:

| Part | Size | Notes |
| --- | --- | --- |
| Weights | parameters × bytes per parameter | 2 for BF16, 1 for FP8/INT8, about 0.5 for INT4 (plus scales) |
| KV cache | concurrent requests × context length × KV bytes per token | takes most of the remaining memory and decides concurrency |
| Activations | proportional to the number of tokens in the batch | inference keeps no intermediates, so usually only a few GB of workspace |
| Other | CUDA context, CUDA Graphs, communication buffers, etc. | usually a few GB reserved |

vLLM's `gpu_memory_utilization` (default 0.9) is "the maximum share of memory that weights + activations + KV cache may use"; after subtracting weights and activations, everything left is preallocated to the KV cache block pool.

## Lower bounds on latency {#延迟的下限}

**Decode**: every step must read at least the weights and the KV cache of all requests from memory once:

$$
\text{TPOT} \ge \frac{\text{weight bytes} + \text{batch} \times \text{context} \times \text{KV bytes per token}}{\text{memory bandwidth}}
$$

**Prefill**: mainly limited by compute:

$$
\text{TTFT} \approx \frac{T \times \text{FLOPs/token}}{\text{peak compute} \times \text{MFU}}
$$

Use these two formulas to evaluate a few deployment options:

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

A few conclusions worth noting:

- **Going from batch 1 to batch 64 raises total throughput about 21-fold (202 → 4252 tok/s), while the TPOT lower bound only rises from 5 ms to 15 ms**: this is the power of [batching](kv-cache.md#批处理让多个请求分摊权重读取). Nearly all the added latency comes from the KV cache: 64 requests with 4K contexts have 34 GB of KV in total, more than twice the weights;
- **INT4 weight quantization lowers the batch-1 decode lower bound from 5.0 ms to 1.4 ms**, because then nearly all the time goes to reading weights; see [how quantization works](quantization.md);
- **70B under 8-GPU tensor parallelism has a single-request decode lower bound (5.3 ms) almost the same as 8B on one GPU (5.0 ms)**: the bandwidth of 8 GPUs adds up and each GPU reads only 1/8 of the weights. In practice you must add the communication latency of two all-reduces per layer, so TP is not free; see [multi-GPU and NCCL](cuda://tools/multi-gpu/) in the CUDA book;
- These are all **theoretical lower bounds**. A real system that reaches 70–85% of bandwidth utilization is doing very well.

These formulas as a calculator: change the model, GPU and precision, move the batch and context, and see how the parameter count, compute per token, bytes read per decode step and the lower bounds on TPOT and TTFT change:

<div class="aig-widget" data-widget="estimator"></div>

!!! inference "Inference view"
    Estimation is where inference optimization starts: compute the theoretical limit first, then compare with measurements. If the measured TPOT is 3 times the theoretical lower bound, there is a lot of room to optimize (kernels not fast enough? scheduling overhead? a CPU bottleneck?); if it already reaches 80% of the bound, the only way to go faster is to change "how many bytes are read" itself: quantization, larger batches, speculative decoding.

!!! interview "How to explain it"
    Estimation is a basic skill; work it with a template: compute the parameter count from `config.json` (embedding + attention and FFN per layer); compute per token is about 2N, plus an attention term proportional to the context; memory = weights + KV + activations + runtime overhead; the decode latency lower bound = (weights + KV) bytes ÷ bandwidth, and prefill ≈ compute ÷ (peak × MFU). For example, LLaMA-3-8B on an H100 at batch 1: 16 GB ÷ 3.35 TB/s ≈ 4.8 ms per token. With 70B under 8-GPU tensor parallelism each GPU reads only 1/8 of the weights, so decode latency is close to 8B on one GPU, plus the all-reduce overhead.

## Exercises {#练习}

**1. Deployment planning.** To serve Qwen2.5-72B in BF16 with 32 concurrent requests at 8K context each, how many 80 GB GPUs are needed at least? (Reserve 8 GB per GPU for activations and other overhead.)

??? success "Answer"
    ```python
    import math
    c = configs["Qwen2.5-72B"]
    need = count_params(c) * 2 + 32 * 8192 * kv_bytes_per_token(c)
    n = math.ceil(need / ((80 - 8) * 1e9))
    print(f"权重 {count_params(c) * 2 / 1e9:.0f} GB + KV {32 * 8192 * kv_bytes_per_token(c) / 1e9:.0f} GB -> 至少 {n} 张")
    ```

    About 145 GB of weights and about 86 GB of KV, about 231 GB in total, so at least 4 GPUs; the tensor-parallel degree is usually a power of 2 that divides the number of attention heads, so in practice you would use 4 or 8.

**2. Food for thought.** Why do we say "the decode latency lower bound has nothing to do with the model's compute"? When does this stop being true?

??? success "Answer"
    With a small decode batch, the computation is far below the compute limit and the time is decided by the number of bytes read, so more compute does not help. Once the batch is large enough that each step's compute time exceeds its read time (the arithmetic intensity passes the ridge point, for example a batch of several hundred in BF16 on an H100), decode becomes compute bound, and only then does compute matter. MoE models, MLA and speculative decoding all raise decode's arithmetic intensity and push it into the compute-bound region sooner.

## Summary {#小结}

- [x] A dense model's parameter count = vocabulary × d (×2 if not tied) + layers × (attention + 3·d·d_ff + 2d) + d, and can be computed exactly.
- [x] Compute per token is about 2N, plus an attention term proportional to the context length.
- [x] Memory = weights + KV cache + activations + other; the KV cache decides concurrency.
- [x] The decode latency lower bound = (weights + KV) bytes / bandwidth; prefill latency ≈ compute / (peak × MFU).
- [x] Computing the theoretical limit first and then looking at measurements is the basic method of inference optimization.

For the related math (arithmetic intensity and rooflines, Amdahl's law), see [math in performance and serving](math://performance-math/).
