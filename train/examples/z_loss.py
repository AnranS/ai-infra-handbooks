import math

import torch
import torch.nn.functional as F

torch.manual_seed(0)
V = 32000
base = torch.randn(V) * 3                                     # 某个位置的输出 logits

print("logits 整体平移后用 bf16 存：")
p = base.softmax(-1)
for shift in (0, 50, 500):
    q = (base + shift).bfloat16().float().softmax(-1)        # 平移不改变 softmax，但 bf16 的间隔随数值变大
    gap = 2.0 ** (math.floor(math.log2(max(shift, 1))) - 7)  # bf16 只有 7 位尾数：[2^e, 2^(e+1)) 里的间隔是 2^(e-7)
    print(f"  平移 {shift:>3}：bf16 在这附近的间隔 {gap:5.3f}，概率的总变差距离 {0.5 * (q - p).abs().sum():.3f}")

logits = (base + 20).requires_grad_()                          # 假设 log Z 已经漂到了 20 附近
target = torch.tensor([7])
ce = F.cross_entropy(logits[None], target)
log_z = torch.logsumexp(logits, -1)
g_ce, = torch.autograd.grad(ce, logits, retain_graph=True)
g_z, = torch.autograd.grad(1e-4 * log_z ** 2, logits)
print(f"log Z = {log_z:.1f}")
print(f"交叉熵对 logits 的梯度之和 {abs(g_ce.sum()):.4f}：整体平移不改变交叉熵，没有力量把 log Z 拉回来")
print(f"z-loss（1e-4·log²Z）的梯度之和 {g_z.sum():+.1e} = 2·1e-4·log Z：把所有 logits 一起往下推")
