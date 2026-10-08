import torch
from muon import newton_schulz

torch.manual_seed(0)
# 造一个"梯度"：奇异值从 1 到 0.01 按指数衰减（真实的权重梯度往往被少数几个方向主导）
U, _ = torch.linalg.qr(torch.randn(256, 256))
V, _ = torch.linalg.qr(torch.randn(1024, 256))
S = torch.logspace(0, -2, 256)
G = U @ torch.diag(S) @ V.T                                  # 256 × 1024

for steps in (1, 3, 5):
    s = torch.linalg.svdvals(newton_schulz(G, steps))
    print(f"Newton-Schulz {steps} 步：奇异值在 [{s.min():.2f}, {s.max():.2f}]，落在 [0.7, 1.2] 的占 {((s > 0.7) & (s < 1.2)).float().mean():.0%}")
X = newton_schulz(G)
top = (X @ V[:, :8]).norm() ** 2 / X.norm() ** 2
print(f"梯度最大的 8 个方向（共 256 个）占的能量：原始梯度 {(S[:8] ** 2).sum() / (S ** 2).sum():.0%}，正交化之后 {top:.1%}")
print(f"正交化之后的均方根 {X.pow(2).mean().sqrt():.4f} ≈ 1/√1024 = {1024 ** -0.5:.4f}；乘上 0.2·√1024 之后是 {(X * 0.2 * 1024 ** 0.5).pow(2).mean().sqrt():.2f}")
