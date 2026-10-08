# 流匹配（flow matching）最小实现：训练目标只有一行，在 CPU 上几秒钟就能看到它学会一个分布
import math

import torch
import torch.nn as nn

torch.manual_seed(0)
K, R = 8, 2.0                                    # 目标分布：半径 2 的圆上 8 个高斯


def sample_data(n):
    idx = torch.randint(0, K, (n,))
    ang = idx.float() * (2 * math.pi / K)
    center = torch.stack([R * ang.cos(), R * ang.sin()], dim=1)
    return center + 0.08 * torch.randn(n, 2), idx


class Velocity(nn.Module):
    """v(x, t)：给定位置和时间，预测该往哪个方向走"""

    def __init__(self, h=96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


model = Velocity()
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
recent = []
for step in range(1, 4001):
    x1, _ = sample_data(256)                     # 数据端
    x0 = torch.randn(256, 2)                     # 噪声端
    t = torch.rand(256, 1)
    xt = (1 - t) * x0 + t * x1                   # 直线插值：t=0 是噪声，t=1 是数据
    loss = ((model(xt, t) - (x1 - x0)) ** 2).mean()   # 目标速度就是两端之差，整个方法只有这一行
    opt.zero_grad()
    loss.backward()
    opt.step()
    recent.append(loss.item())
    if step in (1, 200, 1000, 4000):
        print(f"step {step:>4}  最近 {len(recent):>4} 步的平均 loss {sum(recent) / len(recent):.2f}")
        recent = []


@torch.no_grad()
def sample(n, steps=100):                        # 采样：从噪声出发，用欧拉法沿速度场积分到 t=1
    x = torch.randn(n, 2)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n, 1), i * dt)
        x = x + model(x, t) * dt
    return x


gen = sample(2000)
ang = torch.arange(K).float() * (2 * math.pi / K)
centers = torch.stack([R * ang.cos(), R * ang.sin()], dim=1)
dist = torch.cdist(gen, centers)                 # 每个样本到最近模态的距离
near = dist.min(dim=1)
hit = (near.values < 0.3).float().mean().item()
counts = torch.bincount(near.indices, minlength=K)
print(f"落在某个模态 0.3 半径内的样本：{hit:.0%}")
print(f"八个模态各分到：{counts.tolist()}（理想是每个 250）")
