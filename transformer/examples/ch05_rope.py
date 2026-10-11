"""第 5 章：自己算 cos/sin 表，和 HF 的逐位对；换掉 rotary_emb；看一眼"相对位置"是怎么出现在点积里的"""
import os

import torch

from qwen.embedding import Embedding, LMHead
from qwen.harness import RotaryAdapter, load_hf, logits, max_diff, status
from qwen.mlp import MLP
from qwen.norm import RMSNorm
from qwen.rope import apply_rope, rope_tables
from qwen.weights import Config, find_model, load, load_weights

torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
cfg = Config.load(path)
W = load_weights(path)
hf, tok = load_hf(path)
ids = tok("今天天气不错，我们去公园", return_tensors="pt")["input_ids"]
ref = logits(hf, ids)

pos = torch.arange(ids.shape[1])
cos, sin = rope_tables(cfg, pos)
hf_cos, hf_sin = hf.model.rotary_emb(torch.zeros(1), pos.view(1, 1, -1).expand(3, 1, -1))   # HF 要三轴位置 [3, B, T]
print(f"cos 表 {tuple(cos.shape)}（HF 的是 {tuple(hf_cos.shape)}），和 HF 最大差 {max(max_diff(cos, hf_cos[0]), max_diff(sin, hf_sin[0])):.1e}")
rot = int(cfg.head_dim * cfg.partial_rotary_factor)
print(f"每个头 {cfg.head_dim} 维，只旋转前 {rot} 维（{rot // 2} 对）；最快的一对每个位置转 {1:.2f} 弧度，最慢的一对转 {1 / cfg.rope_theta ** ((rot - 2) / rot):.1e} 弧度")

hf.model.embed_tokens = load(Embedding(cfg.vocab_size, cfg.hidden_size), W, "embed_tokens.")   # 第 2–4 章换过的
hf.lm_head = LMHead(hf.model.embed_tokens.weight)
hf.model.norm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, "norm.")
for i, layer in enumerate(hf.model.layers):
    layer.input_layernorm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, f"layers.{i}.input_layernorm.")
    layer.post_attention_layernorm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, f"layers.{i}.post_attention_layernorm.")
    layer.mlp = load(MLP(cfg), W, f"layers.{i}.mlp.")
hf.model.rotary_emb = RotaryAdapter(cfg, rope_tables)                                          # 这一章
print(f"换掉 rotary_emb 后，logits 最大差 {max_diff(ref, logits(hf, ids)):.1e}")
print(status(hf))

torch.manual_seed(0)                                              # 相对位置：同一对 q、k，放在不同位置但距离相同，点积相同
q, k = torch.randn(1, 1, 1, cfg.head_dim), torch.randn(1, 1, 1, cfg.head_dim)
def dot(pq, pk):
    cq, sq = rope_tables(cfg, torch.tensor([pq]))
    ck, sk = rope_tables(cfg, torch.tensor([pk]))
    return (apply_rope(q, cq, sq) * apply_rope(k, ck, sk)).sum().item()
print(f"q 在 3、k 在 1：{dot(3, 1):.4f}；q 在 103、k 在 101：{dot(103, 101):.4f}；q 在 3、k 在 2：{dot(3, 2):.4f}")
