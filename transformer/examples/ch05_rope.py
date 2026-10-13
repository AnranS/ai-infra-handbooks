"""第 5 章：自己算 cos/sin 表，看每一对转得多快、"相对位置"是怎么出现在点积里的、没旋转的维度原样不动"""
import math
import os

import torch

from qwen.rope import apply_rope, rope_tables
from qwen.weights import Config, find_model

torch.set_num_threads(max(1, os.cpu_count() // 2))
cfg = Config.load(find_model())
rot = int(cfg.head_dim * cfg.partial_rotary_factor)
cos, sin = rope_tables(cfg, torch.arange(6))
print(f"cos 表 {tuple(cos.shape)}：每个头 {cfg.head_dim} 维只旋转前 {rot} 维（{rot // 2} 对），位置 0 的 cos 全是 1：{cos[0, :4].tolist()}")

inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, rot, 2).float() / rot))   # 1. 每一对转多快：转满一圈要多少个位置
for i in (0, 8, 16, 24, 31):
    print(f"  第 {i:2d} 对：每个位置转 {inv_freq[i]:.2e} 弧度，转满一圈要 {2 * math.pi / inv_freq[i]:.3g} 个位置")

torch.manual_seed(0)                                              # 2. 相对位置：同一对 q、k，放在不同位置但距离相同，点积相同
q, k = torch.randn(1, 1, 1, cfg.head_dim), torch.randn(1, 1, 1, cfg.head_dim)


def dot(pq, pk):
    cq, sq = rope_tables(cfg, torch.tensor([pq]))
    ck, sk = rope_tables(cfg, torch.tensor([pk]))
    return (apply_rope(q, cq, sq) * apply_rope(k, ck, sk)).sum().item()


print(f"q 在 3、k 在 1：{dot(3, 1):.4f}；q 在 103、k 在 101：{dot(103, 101):.4f}；q 在 3、k 在 2：{dot(3, 2):.4f}")
print(f"距离从 0 拉到 2000，点积怎么变：", [f"{dot(d, 0):.1f}" for d in (0, 1, 2, 10, 100, 2000)])

x = torch.randn(1, 6, 8, cfg.head_dim)                            # 3. 部分旋转：后 192 维原样不动
y = apply_rope(x, cos, sin)
print(f"旋转后前 {rot} 维变了（最大差 {(y[..., :rot] - x[..., :rot]).abs().max():.2f}），后 {cfg.head_dim - rot} 维没动（最大差 {(y[..., rot:] - x[..., rot:]).abs().max():.1e}）")
