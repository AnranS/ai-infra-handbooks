import torch

lr, wd, steps = 1e-2, 0.1, 1000
scale = torch.tensor([0.01, 10.0])                     # 两个参数：梯度的量级一个很小、一个很大


def train(decoupled):
    w = torch.ones(2, requires_grad=True)
    if decoupled:
        opt = torch.optim.AdamW([w], lr=lr, weight_decay=wd)            # 衰减直接作用在权重上
    else:
        opt = torch.optim.Adam([w], lr=lr, weight_decay=wd)             # 衰减加进梯度（L2 正则），再被 Adam 归一化
    for t in range(steps):
        opt.zero_grad()
        w.grad = scale * (-1) ** t                                      # 正负交替、平均为 0 的梯度：只让 Adam 的 v 有量级
        opt.step()
    return w.detach()


for name, decoupled in [("Adam + L2 正则", False), ("AdamW（解耦的权重衰减）", True)]:
    w = train(decoupled)
    print(f"{name:16s} 小梯度的参数 {w[0]:.3f}，大梯度的参数 {w[1]:.3f}")
print(f"只有衰减时的理论值：(1 - lr·wd)^{steps} = {(1 - lr * wd) ** steps:.3f}")
