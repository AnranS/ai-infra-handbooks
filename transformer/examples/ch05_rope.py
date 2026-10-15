"""第 5 章：先在两维里转一个角，再看 Qwen3.5 的 32 对各转多快、哪两维是一对、点积为什么只看距离"""
import math
import os

import torch

from qwen.rope import apply_rope, rope_tables
from qwen.weights import Config, find_model

torch.set_num_threads(max(1, os.cpu_count() // 2))


def rotate2d(v, angle):                                           # 1. 两维里转一个角：(a, b) -> (a cos - b sin, a sin + b cos)
    a, b = v
    return (a * math.cos(angle) - b * math.sin(angle), a * math.sin(angle) + b * math.cos(angle))


step = math.radians(30)                                           # 玩具设定：每往后一个位置，多转 30 度
for pq, pk in ((3, 1), (103, 101), (3, 2)):
    rq, rk = rotate2d((1.0, 0.0), pq * step), rotate2d((1.0, 0.0), pk * step)
    print(f"q 在 {pq:3d}（转到 {pq * 30 % 360:3d} 度），k 在 {pk:3d}（转到 {pk * 30 % 360:3d} 度）：夹角 {(pq - pk) * 30:2d} 度，点积 {rq[0] * rk[0] + rq[1] * rk[1]:.3f}")

cfg = Config.load(find_model())
rot = int(cfg.head_dim * cfg.partial_rotary_factor)
inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, rot, 2).float() / rot))   # 2. Qwen3.5 的 32 对：每一对转多快
print(f"每个头 {cfg.head_dim} 维，前 {rot} 维参与旋转，共 {rot // 2} 对；后一对比前一对慢 {inv_freq[0] / inv_freq[1]:.2f} 倍")
for i in (0, 1, 2, 8, 16, 24, 31):
    print(f"  第 {i:2d} 对：每个位置转 {inv_freq[i]:.2e} 弧度，转满一圈要 {2 * math.pi / inv_freq[i]:.3g} 个位置")
print(f"基频换成经典的 10000：最慢的一对 {2 * math.pi * 1e4 ** ((rot - 2) / rot):.3g} 个位置就转满一圈，比 262144 的上下文短")   # 262144 是 config.json 里的最大长度

x = torch.zeros(1, 1, 1, cfg.head_dim)                            # 3. 哪两维是一对：只在第 0 维放一个 1，放到位置 1 上转一下
x[..., 0] = 1.0
cos, sin = rope_tables(cfg, torch.tensor([1]))
y = apply_rope(x, cos, sin)[0, 0, 0]
print("转完之后不为 0 的维度：", {j: round(y[j].item(), 4) for j in range(cfg.head_dim) if y[j].abs() > 1e-6}, f"= cos 1 和 sin 1：第 0 维和第 {rot // 2} 维是一对")

torch.manual_seed(0)                                              # 4. 相对位置：随机的 q、k，距离一样点积就一样
q, k = torch.randn(1, 1, 1, cfg.head_dim), torch.randn(1, 1, 1, cfg.head_dim)


def dot(a, b, pa, pb):
    ca, sa = rope_tables(cfg, torch.tensor([pa]))
    cb, sb = rope_tables(cfg, torch.tensor([pb]))
    return (apply_rope(a, ca, sa) * apply_rope(b, cb, sb)).sum().item()


print(f"q 在 3、k 在 1：{dot(q, k, 3, 1):.4f}；q 在 103、k 在 101：{dot(q, k, 103, 101):.4f}；q 在 3、k 在 2：{dot(q, k, 3, 2):.4f}")

same = torch.zeros(1, 1, 1, cfg.head_dim)                         # 5. 同样的内容，离得越远点积越小（只看参与旋转的 64 维）
same[..., :rot] = 1.0
print("q 和 k 一样时，点积随距离：", ", ".join(f"{d}: {dot(same, same, d, 0):.1f}" for d in (0, 1, 10, 100, 1000, 10000, 100000)))

x = torch.randn(1, 6, 8, cfg.head_dim)                            # 6. 部分旋转：后 192 维原样不动
cos, sin = rope_tables(cfg, torch.arange(6))
y = apply_rope(x, cos, sin)
print(f"cos 表 {tuple(cos.shape)}；旋转后前 {rot} 维变了（最大差 {(y[..., :rot] - x[..., :rot]).abs().max():.2f}），后 {cfg.head_dim - rot} 维没动（最大差 {(y[..., rot:] - x[..., rot:]).abs().max():.1e}）")
