"""第 6 章：换掉 6 层全注意力；看一眼注意力矩阵和输出门"""
import os

import torch

from qwen.attention import Attention
from qwen.embedding import Embedding, LMHead
from qwen.harness import AttentionAdapter, RotaryAdapter, load_hf, logits, max_diff, status
from qwen.mlp import MLP
from qwen.norm import RMSNorm
from qwen.rope import rope_tables
from qwen.weights import Config, find_model, load, load_weights

torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
cfg = Config.load(path)
W = load_weights(path)
hf, tok = load_hf(path)
ids = tok("今天天气不错，我们去公园", return_tensors="pt")["input_ids"]
ref = logits(hf, ids)

hf.model.embed_tokens = load(Embedding(cfg.vocab_size, cfg.hidden_size), W, "embed_tokens.")   # 第 2–5 章换过的
hf.lm_head = LMHead(hf.model.embed_tokens.weight)
hf.model.norm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, "norm.")
hf.model.rotary_emb = RotaryAdapter(cfg, rope_tables)
for i, layer in enumerate(hf.model.layers):
    layer.input_layernorm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, f"layers.{i}.input_layernorm.")
    layer.post_attention_layernorm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, f"layers.{i}.post_attention_layernorm.")
    layer.mlp = load(MLP(cfg), W, f"layers.{i}.mlp.")
    if cfg.layer_types[i] == "full_attention":                                                 # 这一章：6 层
        layer.self_attn = AttentionAdapter(load(Attention(cfg), W, f"layers.{i}.self_attn."))
print(f"换掉 6 层全注意力后，logits 最大差 {max_diff(ref, logits(hf, ids)):.1e}")
print(status(hf))

attn = hf.model.layers[3].self_attn.mine                           # 第 3 层是第一层全注意力
n = sum(p.numel() for p in attn.parameters())
print(f"一层注意力 {n / 1e6:.1f}M 参数，其中 q_proj {attn.q_proj.weight.numel() / 1e6:.1f}M（一半是门）")

torch.manual_seed(0)
with torch.no_grad():                                              # 把注意力矩阵和门拿出来看
    x = torch.randn(1, ids.shape[1], cfg.hidden_size)
    cos, sin = rope_tables(cfg, torch.arange(ids.shape[1]))
    T = ids.shape[1]
    q, gate = attn.q_proj(x).view(1, T, attn.n_head, 2 * attn.d).chunk(2, dim=-1)
    g = torch.sigmoid(gate)
    print(f"输出门 sigmoid(gate)：均值 {g.mean():.2f}，小于 0.1 的比例 {(g < 0.1).float().mean():.0%}，大于 0.9 的比例 {(g > 0.9).float().mean():.0%}")
    causal = torch.ones(T, T, dtype=torch.bool).triu(1)
    print("因果掩码（1 = 看不到）：")
    for row in causal.int()[:4]:
        print("   ", " ".join(str(v) for v in row.tolist()))
