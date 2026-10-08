"""一个完整的训练循环：五行主干，外加评估、学习率调度和梯度裁剪"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)

# 造一个两类的玩具数据：两个高斯团
g = torch.Generator().manual_seed(0)
n = 512
x = torch.cat([torch.randn(n, 2, generator=g) + 1.5, torch.randn(n, 2, generator=g) - 1.5])
y = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
perm = torch.randperm(len(x), generator=g)
x, y = x[perm], y[perm]
train = TensorDataset(x[:800], y[:800])
val = TensorDataset(x[800:], y[800:])
loader = DataLoader(train, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(1))

model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2))
opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
loss_fn = nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate():
    model.eval()                                              # 关掉 dropout / BN 的训练行为
    xb, yb = val.tensors
    logits = model(xb)
    acc = (logits.argmax(1) == yb).float().mean().item()
    model.train()                                             # 记得切回来
    return loss_fn(logits, yb).item(), acc


print("epoch   训练 loss   验证 loss   验证准确率   学习率")
for epoch in range(10):
    total = 0.0
    for xb, yb in loader:                                     # ——— 五行主干 ———
        loss = loss_fn(model(xb), yb)                         # 1. 前向 + 算损失
        opt.zero_grad(set_to_none=True)                       # 2. 清空上一步的梯度
        loss.backward()                                       # 3. 反向
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)     # 4. 梯度裁剪（可选，但建议有）
        opt.step()                                            # 5. 更新参数
        total += loss.item() * len(xb)                        # .item() 会同步，放在循环里只是为了打日志
    sched.step()                                              # 调度器按 epoch 走，别放进内层循环
    vl, acc = evaluate()
    print(f"{epoch + 1:5d}   {total / len(train):9.4f}   {vl:9.4f}   {acc:10.1%}   {sched.get_last_lr()[0]:.2e}")

print("\n几条容易写错的：")
print("- zero_grad 要在 backward **之前**；忘了它，梯度会一直累加")
print("- set_to_none=True 把梯度置空而不是填零，省一次写显存，也能让「没收到梯度的参数」暴露出来")
print("- loss.item() 会强制同步 GPU，别在每一步都调；要累计就先留在张量里，最后再取")
print("- scheduler.step() 按 epoch 调用；按步调度的（warmup）要放进内层循环，两种别混")
