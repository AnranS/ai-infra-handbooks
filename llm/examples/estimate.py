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
    linear = 2 * (count_params(c) - V * d)          # 嵌入是查表，不算乘法；输出层要算
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
