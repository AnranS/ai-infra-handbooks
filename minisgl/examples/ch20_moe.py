"""第 20 章：MoE 的路由、按专家排序补齐，Triton fused MoE 与参考实现的比较。

Triton kernel 在 CPU 上用解释器模式运行（TRITON_INTERPRET=1 必须在导入 triton 之前设置）。
"""

import os

os.environ["TRITON_INTERPRET"] = "1"

import torch  # noqa: E402
from minisgl.moe.base import select_experts  # noqa: E402
from minisgl.moe.fused import FusedMoeBackend, moe_align_block_size  # noqa: E402
from minisgl.moe.torch_backend import TorchMoeBackend  # noqa: E402

torch.manual_seed(0)
T, H, I, E, K = 5, 64, 48, 8, 2
x, gating = torch.randn(T, H), torch.randn(T, E)
weights, ids = select_experts(gating, K, renormalize=True)
print("每个 token 选中的专家:", ids.tolist())
print("路由权重（重新归一化后）:", [[round(w, 3) for w in row] for row in weights.tolist()])
sorted_ids, expert_ids = moe_align_block_size(ids, block_m=4, num_experts=E)
print("按专家排序、补齐到 4 的倍数（10 是占位符）:", sorted_ids.tolist())
print("每个块属于哪个专家:", expert_ids.tolist())

x, gating = torch.randn(37, H), torch.randn(37, E)
w1, w2 = torch.randn(E, 2 * I, H) * 0.1, torch.randn(E, H, I) * 0.1
a = FusedMoeBackend().forward(x, w1, w2, gating, K, renormalize=True)
b = TorchMoeBackend().forward(x, w1, w2, gating, K, renormalize=True)
print(f"37 个 token：fused（Triton 解释器）与参考实现的最大误差 {(a - b).abs().max().item():.1e}")
