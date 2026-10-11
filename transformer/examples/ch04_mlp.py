"""第 4 章：换掉 24 层前馈网络；量一下它占多少参数、silu 门开了多少"""
import os

import torch
import torch.nn.functional as F

from qwen.embedding import Embedding, LMHead
from qwen.harness import load_hf, logits, max_diff, status
from qwen.mlp import MLP
from qwen.norm import RMSNorm
from qwen.weights import Config, find_model, load, load_weights

torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
cfg = Config.load(path)
W = load_weights(path)
hf, tok = load_hf(path)
ids = tok("今天天气不错，我们去公园", return_tensors="pt")["input_ids"]
ref = logits(hf, ids)

hf.model.embed_tokens = load(Embedding(cfg.vocab_size, cfg.hidden_size), W, "embed_tokens.")   # 第 2、3 章换过的
hf.lm_head = LMHead(hf.model.embed_tokens.weight)
hf.model.norm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, "norm.")
for i, layer in enumerate(hf.model.layers):
    layer.input_layernorm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, f"layers.{i}.input_layernorm.")
    layer.post_attention_layernorm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, f"layers.{i}.post_attention_layernorm.")
    layer.mlp = load(MLP(cfg), W, f"layers.{i}.mlp.")                                          # 这一章
print(f"换掉 24 层前馈网络后，logits 最大差 {max_diff(ref, logits(hf, ids)):.1e}")
print(status(hf))

n = sum(p.numel() for p in hf.model.layers[0].mlp.parameters())
print(f"一层前馈 {n / 1e6:.2f}M 参数，24 层 {24 * n / 1e6:.0f}M，占全模型 {24 * n / sum(p.numel() for p in hf.parameters()):.0%}")

torch.manual_seed(0)
with torch.no_grad():                                               # 闸门开了多少：silu(gate) 明显大于 0 的比例
    x = torch.randn(1, 8, cfg.hidden_size)
    for i in (0, 11, 23):
        g = F.silu(hf.model.layers[i].mlp.gate_proj(x))
        print(f"  第 {i:2d} 层：silu(gate) > 0.1 的比例 {(g > 0.1).float().mean():.0%}，< -0.1 的比例 {(g < -0.1).float().mean():.0%}（silu 最小约 -0.28）")
