"""保存、加载与复现：checkpoint 里到底要存什么，以及种子怎么管"""
import torch
import torch.nn as nn


def make():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    return model, opt


def steps(model, opt, k, gen):
    for _ in range(k):
        x = torch.randn(16, 4, generator=gen)
        loss = (model(x) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return loss.item()


print("—— 只存权重，续训就对不上 ——")
model, opt = make()
gen = torch.Generator().manual_seed(1)
steps(model, opt, 20, gen)
only_weights = {k: v.clone() for k, v in model.state_dict().items()}
full = {"model": model.state_dict(), "optimizer": opt.state_dict(),
        "step": 20, "rng": torch.get_rng_state(), "data_rng": gen.get_state()}
torch.save(full, "ckpt.pt")
ref = steps(model, opt, 10, gen)                              # 不中断，再走 10 步

m1, o1 = make()                                               # ① 只恢复权重
m1.load_state_dict(only_weights)
g1 = torch.Generator().manual_seed(1)
loss1 = steps(m1, o1, 10, g1)

ck = torch.load("ckpt.pt", weights_only=False)                # ② 全部恢复
m2, o2 = make()
m2.load_state_dict(ck["model"])
o2.load_state_dict(ck["optimizer"])
torch.set_rng_state(ck["rng"])
g2 = torch.Generator()
g2.set_state(ck["data_rng"])
loss2 = steps(m2, o2, 10, g2)

print(f"不中断走完 30 步的 loss：{ref:.6f}")
print(f"① 只恢复权重：          {loss1:.6f}  一致：{loss1 == ref}")
print(f"② 恢复权重+优化器+随机数：{loss2:.6f}  一致：{loss2 == ref}")
print("差别来自 Adam 的一阶、二阶矩和步数：优化器状态丢了，等于重新 warmup 一遍")

print("\n—— checkpoint 的最小集合 ——")
for k in full:
    print(f"  {k}")
print("再加上：学习率调度器的状态、数据读到了第几条、以及（混合精度时）GradScaler 的状态")

print("\n—— weights_only ——")
print("torch.load 默认 weights_only=True：只还原张量和一小撮白名单类型，不执行文件里的任意代码")
print("我们这份 checkpoint 全是张量和整数，默认模式也读得出来：", list(torch.load("ckpt.pt")))
print("但只要存了自定义的类（比如整个 model 对象、argparse 的 Namespace），默认模式就会拒绝，")
print("这时候的正确做法是**改成只存 state_dict**，而不是顺手把 weights_only 关掉——")
print("关掉它等于允许反序列化时执行任意代码，加载来路不明的权重时这是真实的攻击面")

print("\n—— 种子 ——")
torch.manual_seed(0)
a = torch.randn(3)
torch.manual_seed(0)
print("同一个种子，两次结果一样：", torch.equal(a, torch.randn(3)))
print("全局种子会被任何一处 randn 推着走；要隔离就各用各的 Generator（上一章 DataLoader 就是这么做的）")
print("想完全确定：torch.use_deterministic_algorithms(True) + 固定 cuBLAS 工作区，代价是变慢，而且有些算子没有确定性实现")
