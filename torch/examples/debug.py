"""调试与提速：读懂报错、三个经典的坑、inference_mode、autocast 和 profiler"""
import time

import torch
import torch.nn as nn

torch.manual_seed(0)

print("—— 读懂形状报错 ——")
try:
    torch.randn(4, 8) @ torch.randn(16, 2)
except RuntimeError as e:
    print(" ", e)
print("报错里的两个形状就是答案：(4,8) 和 (16,2)，8 ≠ 16。矩阵乘错位几乎都是少了一次 transpose 或者 batch 维没对齐")

print("\n—— 坑一：把带梯度的张量攒进 list ——")
model = nn.Linear(16, 16)
x = torch.randn(8, 16)
bad, good = [], []
for _ in range(3):
    out = model(x).sum()
    bad.append(out)                                           # 每一个都拖着一整张计算图
    good.append(out.detach())                                 # 或者 .item()
print("攒进去的张量还连着图吗：", [t.requires_grad for t in bad], "->", [t.requires_grad for t in good])
print("日志里 running_loss += loss 而不是 loss.item()，就是这样把显存吃光的")

print("\n—— 坑二：忘了切 eval ——")
net = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))
z = torch.randn(2, 4)
net.train()
print("train 模式下两次推理不同：", not torch.allclose(net(z), net(z)))
net.eval()
print("eval 模式下才稳定：   ", torch.allclose(net(z), net(z)))

print("\n—— 坑三：原地操作和视图 ——")
t = torch.arange(6.).reshape(2, 3)
row = t[0]                                                    # 视图
row += 100
print("改视图会改到原张量：", t[0].tolist())
print("要独立的一份就 .clone()；传参数给函数时尤其注意，调用方的张量可能被就地改掉")

print("\n—— no_grad 与 inference_mode ——")
big = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512))
xb = torch.randn(256, 512)
with torch.no_grad():
    a = big(xb)
with torch.inference_mode():
    b = big(xb)
print("no_grad 的结果能不能再进计算图：", (a * 1).requires_grad is False and a.requires_grad is False)
try:
    (b * torch.ones(1, requires_grad=True)).sum().backward()
except RuntimeError as e:
    print("inference_mode 的结果不行：", str(e).split(".")[0])
print("两者都不建图；inference_mode 更彻底（连版本计数都不记，推理更快），代价是产出的张量不能再参与求导")
print("只想临时关梯度、之后还要接着训练，用 no_grad；纯推理服务用 inference_mode")

print("\n—— 自动混合精度 ——")
with torch.autocast("cpu", dtype=torch.bfloat16):
    out = big(xb)
print("autocast 里矩阵乘的输出 dtype：", out.dtype, "；归一化、softmax 这类敏感算子仍然留在 fp32")
print("GPU 上要配 GradScaler（fp16）或者直接用 bf16（不用 scaler）；CPU 上没有专门的矩阵指令时反而更慢")

print("\n—— profiler：先看是谁最慢 ——")
from torch.profiler import ProfilerActivity, profile

with profile(activities=[ProfilerActivity.CPU]) as prof:
    big(xb).sum().backward()
top = [e for e in prof.key_averages() if e.key.startswith("aten::")]
top.sort(key=lambda e: -e.self_cpu_time_total)
print("占用 CPU 时间最多的两个算子：", [e.key for e in top[:2]])
print("（具体耗时每台机器、每次运行都不一样，这里只看排名：矩阵乘一定在最前面）")
print("完整的表用 prof.key_averages().table(sort_by=\"self_cpu_time_total\", row_limit=10) 打出来")
print("GPU 上把 activities 加上 CUDA，再按 self_cuda_time_total 排序；要看时间线就用 Nsight Systems")
