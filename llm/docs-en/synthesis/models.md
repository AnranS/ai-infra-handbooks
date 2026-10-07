# A tour of mainstream model architectures

<p class="lead">LLaMA, Qwen, DeepSeek, Gemma, gpt-oss: many names, but almost the same skeleton: a Pre-Norm decoder, RoPE, SwiGLU, GQA, plus optional MoE. Their differences come down to a few "knobs". This chapter first teaches you to read a <code>config.json</code>, then uses the meta device to count the parameters and KV cache of ten mainstream models exactly, and finally goes through each family's changes and what each change means for inference engines.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. Given a new model's `config.json`, can you state its parameter count, active parameters and KV cache size per token within a minute?
    2. What do Gemma 3, gpt-oss and Qwen3-Next each do to reduce the KV cache of long contexts?
    3. Why can't Qwen2.5-7B run with 8-way tensor parallelism?
    4. Why are DeepSeek models usually deployed with DP attention + expert parallelism rather than pure tensor parallelism?

??? success "Answers (try first, then expand to compare)"
    1. Yes: the parameter count is the embedding + per-layer attention and FFN (MoE by number of experts); active parameters count only the experts a token actually passes through; KV per token = 2 × layers × KV heads × head dimension × bytes (for MLA, by the latent dimension; sliding-window and linear layers counted separately).
    2. Gemma 3: local (sliding window of 1024) and global attention alternate 5 : 1, so most layers' KV does not grow with the context; gpt-oss: full-attention layers alternate with sliding-window layers of window 128, plus attention sinks; Qwen3-Next: in every 4 layers, 3 are Gated DeltaNet (linear attention, a fixed-size state) and 1 is gated standard attention.
    3. It has 28 attention heads (4 KV heads), and 28 is not divisible by 8, so the heads cannot be split evenly across 8 GPUs (with even fewer KV heads, those would have to be replicated too).
    4. MLA's latent vector is shared by all heads, so tensor parallelism cannot split it by head and every GPU must store a full copy of the KV, wasting more with more GPUs; so the attention part uses data parallelism (DP attention, each GPU serving different requests), and the MoE part uses expert parallelism.

## Reading config.json {#读懂-configjson}

This is the config of Qwen3-0.6B, which this handbook has used throughout (the important fields):

<!-- i18n:diagram 1ca4289248 -->
```text
"hidden_size": 1024,                d: the width of the residual stream
"intermediate_size": 3072,          FFN intermediate width, 3 d
"num_hidden_layers": 28,            number of layers L
"num_attention_heads": 16,          number of query heads
"num_key_value_heads": 8,           number of KV heads: GQA, every 2 query heads share one set of K, V
"head_dim": 128,                    dimension per head; note 16 × 128 = 2048 ≠ d, the head dimension is configured separately
"vocab_size": 151936,               vocabulary size
"tie_word_embeddings": true,        the output layer shares weights with the embedding
"rope_theta": 1000000,              RoPE base; larger suits longer contexts
"max_position_embeddings": 40960,   the longest context supported in training
"rms_norm_eps": 1e-06,              RMSNorm's ε
"hidden_act": "silu",               the activation in SwiGLU
"torch_dtype": "bfloat16"           the precision of the released weights
```

Structural differences that config.json does not show require reading the model code: Qwen3 applies an RMSNorm to q and k before RoPE (QK-Norm), and it drops the biases of the q, k and v projections (Qwen2 has them).

Each field corresponds to a chapter of this handbook and to a cost at inference time:

| Field | Chapter | What it decides at inference time |
| --- | --- | --- |
| `hidden_size`, `intermediate_size`, `num_hidden_layers` | [assembling a large model](../transformer/build-llm.md#各部分的参数量) | parameter count → weight memory, the weights read in decode |
| `num_attention_heads`, `num_key_value_heads`, `head_dim` | [attention variants](../transformer/attention-variants.md) | KV cache size; how and how far tensor parallelism can split |
| `vocab_size`, `tie_word_embeddings` | [embedding and output layers](../transformer/embedding.md) | the output layer's compute and logits memory; the cost of sampling |
| `rope_theta`, `rope_scaling`, `max_position_embeddings` | [positional encoding](../transformer/position.md#长上下文扩展) | the supported context length; the RoPE variant the attention kernel must implement |
| `num_experts`, `num_experts_per_tok`, `moe_intermediate_size` | [MoE](../transformer/moe.md) | total and active parameters; whether expert parallelism is needed |
| `sliding_window`, `layer_types` | this chapter | which layers' KV cache grows with the context |
| `torch_dtype`, `quantization_config` | [quantization](../inference/quantization.md) | bytes per weight, the GEMM kernels needed |

In an inference engine, these fields decide which model class to load, how much KV cache to allocate and which attention backend to choose. Whether an engine can support a model largely comes down to whether there are kernels for the structure these fields describe.

## A checkup of ten models {#十个模型的体检表}

Build each model on the meta device with transformers' own config classes, count the total parameters and the parameters each token activates exactly, then compute the KV cache size. For sliding-window layers, the KV cache holds at most the tokens in the window; linear-attention layers have only a fixed-size state that does not grow with the context:

```python
import torch
import transformers as T
from transformers import AutoModelForCausalLM

configs = {
    "LLaMA-3-8B": T.LlamaConfig(vocab_size=128256, hidden_size=4096, intermediate_size=14336, num_hidden_layers=32,
                                num_attention_heads=32, num_key_value_heads=8),
    "Qwen2.5-7B": T.Qwen2Config(vocab_size=152064, hidden_size=3584, intermediate_size=18944, num_hidden_layers=28,
                                num_attention_heads=28, num_key_value_heads=4),
    "Qwen3-8B": T.Qwen3Config(vocab_size=151936, hidden_size=4096, intermediate_size=12288, num_hidden_layers=36,
                              num_attention_heads=32, num_key_value_heads=8, head_dim=128),
    "Gemma-3-27B": T.Gemma3TextConfig(vocab_size=262208, hidden_size=5376, intermediate_size=21504, num_hidden_layers=62,
                                      num_attention_heads=32, num_key_value_heads=16, head_dim=128, sliding_window=1024),
    "Mixtral-8x7B": T.MixtralConfig(),                      # the defaults of the following config classes are those models' configs
    "Qwen3-30B-A3B": T.Qwen3MoeConfig(hidden_size=2048, num_hidden_layers=48, head_dim=128, moe_intermediate_size=768),
    "Qwen3-235B-A22B": T.Qwen3MoeConfig(hidden_size=4096, num_hidden_layers=94, num_attention_heads=64, head_dim=128,
                                        moe_intermediate_size=1536),
    "gpt-oss-120b": T.GptOssConfig(),
    "DeepSeek-V3": T.DeepseekV3Config(),
    "Qwen3-Next-80B-A3B": T.Qwen3NextConfig(),
}

def count(cfg):
    """在 meta 设备上构建模型，返回（总参数，每个 token 激活的参数）。"""
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    total = sum(p.numel() for p in model.parameters())
    routed = sum(p.numel() for n, p in model.named_parameters() if ".experts." in n)   # routed experts (excluding shared experts)
    if not routed:
        return total, total
    n_experts = getattr(cfg, "n_routed_experts", None) or getattr(cfg, "num_local_experts", None) or cfg.num_experts
    return total, total - routed * (1 - cfg.num_experts_per_tok / n_experts)

def kv_bytes(cfg, context, bytes_per_elem=2):
    """一个请求在给定上下文长度下的 KV Cache 字节数。"""
    if hasattr(cfg, "kv_lora_rank"):                                   # MLA: each layer stores only the latent vector and the RoPE part
        per_token_layer = cfg.kv_lora_rank + cfg.qk_rope_head_dim
    else:
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        per_token_layer = 2 * cfg.num_key_value_heads * head_dim
    layer_types = getattr(cfg, "layer_types", None) or ["full_attention"] * cfg.num_hidden_layers
    window = getattr(cfg, "sliding_window", None) or context
    tokens = {"full_attention": context, "sliding_attention": min(context, window),
              "linear_attention": 0}                                    # linear attention has only a fixed-size state
    return sum(tokens[t] for t in layer_types) * per_token_layer * bytes_per_elem

ctx = 131072
print(f"{'模型':20s}{'总参数':>9s}{'激活':>9s}{'KV/token':>10s}{'128K 上下文的 KV':>18s}")
results = {}
for name, cfg in configs.items():
    total, active = count(cfg)
    results[name] = (total, active)
    print(f"{name:20s}{total / 1e9:8.1f}B{active / 1e9:8.1f}B{kv_bytes(cfg, ctx) / ctx / 1024:8.0f}KB"
          f"{kv_bytes(cfg, ctx) / 1e9:14.1f} GB")
assert round(results["DeepSeek-V3"][0] / 1e9) == 671 and round(results["gpt-oss-120b"][0] / 1e9, 2) == 116.83
```

```text title="output"
模型                        总参数       激活  KV/token      128K 上下文的 KV
LLaMA-3-8B               8.0B     8.0B     128KB          17.2 GB
Qwen2.5-7B               7.6B     7.6B      56KB           7.5 GB
Qwen3-8B                 8.2B     8.2B     144KB          19.3 GB
Gemma-3-27B             27.0B    27.0B      83KB          11.2 GB
Mixtral-8x7B            46.7B    12.9B     128KB          17.2 GB
Qwen3-30B-A3B           30.5B     3.4B      96KB          12.9 GB
Qwen3-235B-A22B        235.1B    22.2B     188KB          25.2 GB
gpt-oss-120b           116.8B     5.7B      36KB           4.8 GB
DeepSeek-V3            671.0B    37.6B      69KB           9.2 GB
Qwen3-Next-80B-A3B      79.7B     3.9B      24KB           3.2 GB
```

The total parameters all match the officially published numbers. The active parameters come out a bit larger than the official figures (gpt-oss-120b's official figure is 5.1B, for example), because the embedding table is counted here. The embedding layer is only a lookup and takes no part in computation, so some official counts exclude it: gpt-oss's embedding table is 201088 × 2880 ≈ 0.58B, exactly the difference.

Put these ten models on one chart and the two main threads (MoE makes active parameters far smaller than total parameters; GQA, MLA, sliding windows and linear attention make the KV ever smaller) jump out:

<div class="aig-widget" data-widget="model-map"></div>

The table shows the two main threads of model design in recent years:

1. **MoE keeps getting sparser**: the active share went from 28% in Mixtral to 5.6% in DeepSeek-V3, and about 5% in gpt-oss and Qwen3-Next. Compute goes by active parameters, memory by total parameters.
2. **The KV cache keeps getting cheaper**: at the same 128K context, the full-attention Qwen3-8B needs 19 GB; DeepSeek-V3, with MLA and 80 times the parameters, needs only half that; gpt-oss and Qwen3-Next bring the KV down to 3–5 GB by having some layers "not keep the whole history".

Now look at each family's specific designs.

## LLaMA: the standard template {#llama标准模板}

LLaMA (2023) established the template of nearly every open model today: **Pre-Norm + RMSNorm, SwiGLU, RoPE, no biases**. What came after:

- LLaMA 2: GQA for the 70B model;
- LLaMA 3: GQA at all sizes (8 KV heads), the vocabulary grown from 32K to 128K (fewer tokens for the same text, but a larger output layer), `rope_theta` raised to 500000;
- LLaMA 3.1: `rope_scaling` (`rope_type: llama3`, band-wise scaling of frequencies) extends the context to 128K.

!!! inference "Inference view"
    The number of KV heads sets the "comfortable limit" of tensor parallelism. LLaMA-3-8B has 8 KV heads, so at TP = 8 each GPU gets exactly one KV head; if TP exceeds the number of KV heads, KV heads must be replicated across GPUs and the total KV cache memory doubles.

## Qwen: from biases to QK-Norm, then to hybrid attention {#qwen从偏置到-qk-norm再到混合注意力}

- **Qwen2 / 2.5**: add biases to the Q, K and V projections of the LLaMA template (`attention_bias` in `mini_llm`), tie the embedding weights in small sizes, `rope_theta` of 1e6.
- **Qwen3**: drop the QKV biases and add **QK-Norm**: before RoPE, apply an RMSNorm to each head's q and k to keep attention scores from growing too large, which makes training more stable. The MoE versions (30B-A3B, 235B-A22B) use 128 experts with 8 chosen per token, and no shared expert.
- **Qwen3-Next**: in every 4 layers, 3 use **Gated DeltaNet (linear attention)** and 1 gated standard attention; 512 experts with 10 chosen, plus 1 shared expert; with an MTP module.

Linear attention replaces "attending to every past token" with a fixed-size state matrix that is updated with each new token. Its decode cost does not depend on the context length, but it is less expressive than standard attention, so it is usually mixed with standard attention layers.

!!! inference "Inference view"
    - QK-Norm has to be fused into the kernel after the QKV projection and before RoPE;
    - Hybrid architectures complicate cache management in inference engines: standard attention layers have a KV cache that grows per token and can be paged, while linear-attention layers have one fixed-size state per request. Prefix caching gets harder too: a linear-attention state corresponds to "a prefix of one exact length", so reuse requires saving snapshots of the state at specific positions. vLLM and SGLang both implement cache management for this kind of state separately.
    - Qwen2.5-7B has 28 query heads, not divisible by 8, so it cannot run with 8-GPU tensor parallelism (inference engines require the number of heads to be divisible by TP).

## Mixtral: MoE goes mainstream {#mixtralmoe-进入主流}

Mixtral-8x7B (2023) took Mistral-7B's skeleton and replaced each layer's FFN with 8 experts, 2 chosen per token. 46.7B total parameters with 12.9B active: inference speed close to a 13B dense model, quality close to larger models. It was the first MoE model widely used in the open community, and it popularized fused MoE kernels.

## DeepSeek-V3: MLA + fine-grained MoE {#deepseek-v3mla--细粒度-moe}

DeepSeek-V3 (late 2024) brought together several designs that affect inference:

- **[MLA](../transformer/attention-variants.md#mla多头潜在注意力)**: 128 heads share one 512-dimensional latent KV vector (plus a 64-dimensional RoPE part), so each layer stores only 576 numbers per token;
- **Fine-grained MoE**: of 61 layers, the first 3 have dense FFNs, and every other layer has 256 routed experts with 8 chosen, plus 1 shared expert; routing scores with sigmoid and balances load by adding a dynamically adjusted bias to each expert (no auxiliary loss);
- **MTP**: also predicts the token after the next one, which can serve as the draft for speculative decoding at inference time;
- Trained with FP8 mixed precision, and the released weights are FP8 (quantized in 128×128 blocks).

The later DeepSeek-R1 kept the same structure; DeepSeek-V3.2 added sparse attention (DSA) on top, using a lightweight indexer to choose the most relevant subset of tokens for each query to attend to. Kimi K2 also largely kept V3's structure (see the exercises).

!!! inference "Inference view"
    - MLA uses weight absorption in decode, so attention becomes "128 query heads against one shared 576-dimensional latent KV", with high intensity, needing dedicated kernels (such as FlashMLA);
    - MLA's latent KV exists as a single copy and cannot be split by head, so under tensor parallelism every GPU stores the full KV, wasting more the larger TP is. So **DP attention** is common: attention runs data-parallel by request (each GPU handles different requests and keeps its own KV), and MoE runs expert-parallel;
    - 256 experts spread over many GPUs need two all-to-alls per layer, communication libraries like DeepEP, and redundant experts with load-balanced placement (EPLB);
    - Combining PD disaggregation with large-scale EP (different parallel scales for prefill and decode) is the mainstream way to deploy this kind of model today.

## Gemma: a large vocabulary, soft-capping, sliding windows {#gemma大词表软截断滑动窗口}

The hallmarks of the Gemma family:

- A 256K vocabulary, `head_dim` of 256, embeddings multiplied by $\sqrt{d}$ on input, and GeGLU (the GELU version of gating) in the FFN;
- An RMSNorm before and after each sublayer ("sandwich" Pre + Post Norm);
- **Gemma 2**: local attention (sliding window of 4096) and global attention alternate layer by layer; attention scores and final logits are **soft-capped**: $\text{cap} \cdot \tanh(x / \text{cap})$, smoothly limiting values to $(-\text{cap}, \text{cap})$;
- **Gemma 3**: the local-to-global ratio rises to 5 : 1, the window shrinks to 1024, QK-Norm replaces soft-capping, and the context is 128K.

What soft-capping does:

```pycon
>>> import torch
>>> x = torch.tensor([10.0, 50.0, 100.0, 1000.0])
>>> [round(v, 2) for v in (30 * torch.tanh(x / 30)).tolist()]
[9.65, 27.93, 29.92, 30.0]
```

Small values barely change, and large ones are squeezed below 30.

Sliding windows have a dramatic effect on the KV cache. Only 10 of Gemma-3-27B's 62 layers are global attention, and the other 52 keep only the most recent 1024 tokens:

```python
g = configs["Gemma-3-27B"]
all_global = ctx * g.num_hidden_layers * 2 * g.num_key_value_heads * g.head_dim * 2
print(f"128K 上下文：实际 {kv_bytes(g, ctx) / 1e9:.1f} GB，若全部是全局注意力则为 {all_global / 1e9:.1f} GB")
```

```text title="output"
128K 上下文：实际 11.2 GB，若全部是全局注意力则为 66.6 GB
```

!!! inference "Inference view"
    Soft-capping changes how attention scores are computed, so the attention kernel must support it natively (both FlashAttention 2 and FlashInfer added an option specifically for it), or fall back to a slow implementation that materializes the scores. A `head_dim` of 256 also needs kernel support. Sliding-window layers' KV can be overwritten cyclically, but combined with paged KV and prefix caching, the inference engine has to manage caches separately for different kinds of layers.

## gpt-oss: a natively 4-bit MoE {#gpt-oss原生-4-位的-moe}

OpenAI's gpt-oss (2025) comes in two sizes, 120b and 20b:

- MoE: 120b has 36 layers, each with 128 experts and 4 chosen;
- Attention: full-attention layers alternate with sliding-window layers whose window is only 128 tokens; 64 query heads and 8 KV heads;
- **Attention sinks**: each head has a learnable scalar that takes part in the softmax normalization as a "virtual token" but corresponds to no value. This lets a head "look at nothing" instead of piling attention onto the first token as in the [attention chapter](../transformer/attention.md#真实模型里的注意力注意力汇聚);
- Expert weights are released in **MXFP4** (4-bit floating point, with one shared scale per 32 numbers), so the 120b model fits on one 80 GB GPU;
- It uses a new chat format, harmony.

How attention sinks are computed:

```pycon
>>> scores = torch.tensor([2.0, 1.0, 0.5, 0.2])   # the current query's scores for 4 keys
>>> sink = torch.tensor([3.0])                     # the sink score this head has learned
>>> p = torch.cat([scores, sink]).softmax(-1)[:-1] # the sink joins the normalization, then is dropped
>>> [round(v, 3) for v in p.tolist()], round(p.sum().item(), 3)
([0.223, 0.082, 0.05, 0.037], 0.393)
```

This head's attention on the real tokens sums to only 0.39; the rest of the "attention" is absorbed by the sink.

!!! inference "Inference view"
    Attention sinks also need kernel support: at the end of the online softmax, add the sink term to the denominator, which costs very little. MXFP4 MoE weights have native Tensor Core support on Blackwell, while on Hopper they must be dequantized inside the kernel first.

## Trends at a glance {#趋势小结}

| Direction | Examples | What inference engines must keep up with |
| --- | --- | --- |
| Less KV cache | GQA → MLA; local/global mixes (Gemma 3, gpt-oss); linear-attention hybrids (Qwen3-Next); sparse attention (DeepSeek-V3.2) | new attention kernels; caches managed per layer type; new implementations of prefix caching |
| Sparser MoE | experts 8 → 128 → 256 → 512, active share below 5% | fused MoE, expert parallelism, all-to-all communication, load balancing |
| Native low precision | FP8 training (DeepSeek-V3), MXFP4 releases (gpt-oss) | GEMM kernels for those formats |
| Training-stability tricks | QK-Norm, soft-capping, attention sinks, gated attention | extra computation inside kernels |
| Designed for inference | MTP (usable as a speculative decoding draft), fewer attention heads | speculative decoding frameworks |

Model architectures and inference systems shape each other: models are increasingly designed for inference efficiency, and inference engines must keep supporting new structures. When you read a new model's paper or config, give it this chapter's "checkup": parameters, active parameters, KV cache, attention type, special operators, and you can tell what it means for an inference system.

!!! interview "In an interview"
    Asked "what do you look at in a new model's `config.json`": within a minute, work out the parameter count, active parameters and KV per token; then look at the attention type (GQA, MLA, sliding windows, linear-attention hybrids, sparse attention), the MoE config (number of experts, top-k, shared experts), RoPE and context length, and special structures (soft-capping, attention sinks, QK-Norm). Then derive the deployment implications: with fewer KV heads than tensor-parallel GPUs, KV must be replicated; MLA models use DP attention + expert parallelism; linear attention needs a per-request state pool.

## Exercises {#练习}

**1. Kimi K2.** Kimi K2 largely keeps DeepSeek-V3's structure; the main changes are: routed experts up from 256 to 384, attention heads down from 128 to 64, only the first layer with a dense FFN, and a vocabulary of 163840. Count its total parameters, active parameters and KV cache per token, and think about why fewer attention heads helps inference.

??? success "Answer"
    ```python
    k2 = T.DeepseekV3Config(vocab_size=163840, num_attention_heads=64, n_routed_experts=384, first_k_dense_replace=1)
    total, active = count(k2)
    print(f"总参数 {total / 1e9:.0f}B，激活 {active / 1e9:.1f}B，KV {kv_bytes(k2, ctx) / ctx / 1024:.0f} KB/token")
    ```

    ```text title="output"
    总参数 1026B，激活 32.9B，KV 69 KB/token
    ```

    About 1T total parameters and about 33B active. The KV cache is exactly the same as DeepSeek-V3's: MLA stores a latent vector shared by all heads, independent of the number of heads.

    Fewer heads help the **computation** of attention: in MLA decode, every head takes dot products with the 576-dimensional latent KV, so attention compute is proportional to the number of heads; when prefilling long contexts, the $O(n^2)$ part of attention is also proportional to the number of heads. Halving the heads halves the attention cost at long contexts. It is a design choice made for inference efficiency.

**2. Deployment plan.** You are to deploy Qwen3-235B-A22B (BF16) on 8 H100s (80 GB each). Do the weights fit? What about FP8? Besides tensor parallelism, what parallelism would you consider?

??? success "Answer"
    BF16 weights are about 235 × 2 = 470 GB, and 8 GPUs have 640 GB, so they fit, but only about 170 GB remain for the KV cache and activations, while this model's KV at a 128K context alone is 25 GB, so concurrency would be very limited. FP8 weights are about 235 GB, leaving far more memory, and decode reads half the weights.

    It has 4 KV heads, so at TP = 8 each KV head is replicated on two GPUs, halving the KV cache's memory efficiency. A better plan uses TP = 4 or DP attention for the attention part and expert parallelism for MoE (128 experts over 8 GPUs, 16 each). The exact choice should come from measurements under the real load.

## Summary {#小结}

- [x] Every field of `config.json` corresponds to a chapter of principles and a cost at inference time; from the config you can estimate parameters, active parameters and the KV cache.
- [x] Mainstream models share the skeleton of Pre-Norm + RoPE + SwiGLU + GQA, and differ mainly in attention type, MoE config and a few stability tricks.
- [x] The road to less KV: GQA → MLA → local/global mixes → linear-attention hybrids → sparse attention.
- [x] MoE keeps getting sparser, and inference needs fused MoE, expert parallelism and load balancing.
- [x] Every new structure (soft-capping, attention sinks, MLA, linear attention) requires corresponding kernels and cache management in the inference engine.
