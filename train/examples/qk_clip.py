import math

import torch

torch.manual_seed(0)
H, d, D, T, tau = 4, 32, 128, 256, 100.0                      # 4 个头；阈值 τ = 100（Kimi K2 的取值）
x = torch.randn(T, D)
Wq = torch.randn(H, D, d) / math.sqrt(D)
Wk = torch.randn(H, D, d) / math.sqrt(D)
Wq[1] *= 6                                                    # 假装训练中第 1、3 个头的 q、k 权重长大了
Wk[1] *= 5
Wq[3] *= 3
Wk[3] *= 2.5


def max_logits():
    q, k = torch.einsum("td,hde->hte", x, Wq), torch.einsum("td,hde->hte", x, Wk)
    return (q @ k.transpose(1, 2) / math.sqrt(d)).amax(dim=(1, 2))   # 每个头在这批数据上的最大 logit


def qk_clip(alpha=0.5):
    """优化器更新之后执行：最大 logit 超过 τ 的头，把 W_q、W_k 分别乘 γ^α、γ^(1-α)，γ = τ / S_max"""
    s = max_logits()
    gamma = (tau / s).clamp(max=1.0)                          # 没超过阈值的头 γ = 1，不受影响
    Wq.mul_(gamma[:, None, None] ** alpha)
    Wk.mul_(gamma[:, None, None] ** (1 - alpha))
    return gamma


before = max_logits()
gamma = qk_clip()
after = max_logits()
for h in range(H):
    print(f"头 {h}：最大 logit {before[h]:6.1f} → {after[h]:6.1f}（γ = {gamma[h]:.3f}）")
