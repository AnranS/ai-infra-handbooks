"""第 6 章（四）：门控注意力。用真实权重跑一层，把注意力矩阵和输出门拿出来看"""
import math
import os

import torch
from tokenizers import Tokenizer

from qwen.attention import Attention
from qwen.embedding import Embedding
from qwen.norm import RMSNorm
from qwen.rope import apply_rope, rope_tables
from qwen.weights import Config, find_model, load, load_weights

torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
cfg = Config.load(path)
W = load_weights(path)
attn = load(Attention(cfg), W, "layers.3.self_attn.")                                          # 第 3 层是第一层全注意力
n = sum(p.numel() for p in attn.parameters())
print(f"一层注意力 {n / 1e6:.1f}M 参数，其中 q_proj {attn.q_proj.weight.numel() / 1e6:.1f}M（一半是门）")

tk = Tokenizer.from_file(str(path / "tokenizer.json"))         # 用第 2、3 章的零件造出一段真实输入
ids = torch.tensor([tk.encode("今天天气不错，我们去公园").ids])
T = ids.shape[1]
h = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, "layers.3.input_layernorm.")(load(Embedding(cfg.vocab_size, cfg.hidden_size), W, "embed_tokens.")(ids))
cos, sin = rope_tables(cfg, torch.arange(T))
y = attn(h, cos, sin)
print(f"输入 {tuple(h.shape)} -> 输出 {tuple(y.shape)}")

with torch.no_grad():                                                                          # 1. 把注意力矩阵算出来看：每行加起来是 1，上三角是 0
    q, gate = attn.q_proj(h).view(1, T, attn.n_head, 2 * attn.d).chunk(2, dim=-1)
    k = attn.k_proj(h).view(1, T, attn.n_kv, attn.d)
    q, k = apply_rope(attn.q_norm(q), cos, sin), apply_rope(attn.k_norm(k), cos, sin)
    scores = q[0, :, 0] @ k[0, :, 0].T / math.sqrt(attn.d)                                     # 第 0 个 query 头 × 它那组 key
    probs = torch.softmax(scores.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf")), dim=-1)
    words = [tk.decode([i]) for i in ids[0].tolist()]
    print("第 0 个头的注意力矩阵（行 = query，列 = key）：")
    for w, row in zip(words, probs):
        print(f"  {w:<4}", " ".join(f"{p:.2f}" for p in row.tolist()))
    g = torch.sigmoid(gate)                                                                    # 2. 输出门：每个 token、每个头、每一维一个
    print(f"输出门的形状 {tuple(g.shape)}：{T} 个 token × {attn.n_head} 个头 × {attn.d} 维，每个数都在 0～1 之间")
print(f"GQA：{attn.n_head} 个 query 头共用 {attn.n_kv} 组 KV，每个 token 要存的 KV 是 {2 * attn.n_kv * attn.d} 个数，不用 GQA 要存 {2 * attn.n_head * attn.d} 个")
