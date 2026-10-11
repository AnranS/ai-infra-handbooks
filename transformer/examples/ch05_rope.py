"""第 5 章：自己算 cos/sin 表，和 HF 的 rotary_emb 逐位对；看一眼"相对位置"是怎么出现在点积里的"""
import os

import torch
from transformers import AutoModelForCausalLM

from qwen.rope import apply_rope, rope_tables
from qwen.weights import Config, find_model

torch.set_num_threads(max(1, os.cpu_count() // 2))
diff = lambda a, b: (a - b).abs().max().item()
path = find_model()
cfg = Config.load(path)
hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()

pos = torch.arange(6)
cos, sin = rope_tables(cfg, pos)
hf_cos, hf_sin = hf.model.rotary_emb(torch.zeros(1), pos.view(1, 1, -1).expand(3, 1, -1))   # HF 要三轴位置 [3, B, T]
print(f"cos 表 {tuple(cos.shape)}（HF 的是 {tuple(hf_cos.shape)}），和 HF 的 rotary_emb 最大差 {max(diff(cos, hf_cos[0]), diff(sin, hf_sin[0])):.1e}")
rot = int(cfg.head_dim * cfg.partial_rotary_factor)
print(f"每个头 {cfg.head_dim} 维，只旋转前 {rot} 维（{rot // 2} 对）；最快的一对每个位置转 {1:.2f} 弧度，最慢的一对转 {1 / cfg.rope_theta ** ((rot - 2) / rot):.1e} 弧度")

torch.manual_seed(0)                                              # 相对位置：同一对 q、k，放在不同位置但距离相同，点积相同
q, k = torch.randn(1, 1, 1, cfg.head_dim), torch.randn(1, 1, 1, cfg.head_dim)


def dot(pq, pk):
    cq, sq = rope_tables(cfg, torch.tensor([pq]))
    ck, sk = rope_tables(cfg, torch.tensor([pk]))
    return (apply_rope(q, cq, sq) * apply_rope(k, ck, sk)).sum().item()


print(f"q 在 3、k 在 1：{dot(3, 1):.4f}；q 在 103、k 在 101：{dot(103, 101):.4f}；q 在 3、k 在 2：{dot(3, 2):.4f}")
