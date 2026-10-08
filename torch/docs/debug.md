# （八）调试与提速

<p class="lead">最后一章讲"出了问题怎么办"：形状报错怎么读、三个最常见的坑长什么样、`no_grad` 和 `inference_mode` 到底差在哪、autocast 怎么用，以及用 profiler 找出最慢的算子。所有例子都能跑，包括那些报错。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. `mat1 and mat2 shapes cannot be multiplied (4x8 and 16x2)` 要怎么读？
    2. 为什么 `running_loss += loss` 会吃显存，而 `+= loss.item()` 不会？
    3. `no_grad` 和 `inference_mode` 有什么区别？
    4. autocast 会把所有算子都变成 bf16 吗？
    5. profiler 的 self time 和 total time 有什么区别？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 两个形状就写在里面：`(4,8) @ (16,2)`，8 ≠ 16。矩阵乘错位基本都是少了一次 transpose 或者 batch 维没对齐。
    2. 前者攒的是带梯度的张量，每一个都拖着一整张计算图，图上的中间激活全都释放不掉。
    3. 两者都不建图；`inference_mode` 更彻底（连版本计数都不记录），推理更快，但产出的张量不能再参与求导。
    4. 不会。只有矩阵乘、卷积这类对精度不敏感又受益明显的算子会被转；归一化、softmax、损失函数仍然留在 fp32。
    5. self time 只算这个算子自己花的时间，total time 包含它调用的子算子。找热点看 self time，看调用结构看 total。

## 报错、坑与提速

```python title="debug.py" ci="loose"
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
```

```text title="输出"
—— 读懂形状报错 ——
  mat1 and mat2 shapes cannot be multiplied (4x8 and 16x2)
报错里的两个形状就是答案：(4,8) 和 (16,2)，8 ≠ 16。矩阵乘错位几乎都是少了一次 transpose 或者 batch 维没对齐

—— 坑一：把带梯度的张量攒进 list ——
攒进去的张量还连着图吗： [True, True, True] -> [False, False, False]
日志里 running_loss += loss 而不是 loss.item()，就是这样把显存吃光的

—— 坑二：忘了切 eval ——
train 模式下两次推理不同： True
eval 模式下才稳定：    True

—— 坑三：原地操作和视图 ——
改视图会改到原张量： [100.0, 101.0, 102.0]
要独立的一份就 .clone()；传参数给函数时尤其注意，调用方的张量可能被就地改掉

—— no_grad 与 inference_mode ——
no_grad 的结果能不能再进计算图： True
inference_mode 的结果不行： Inference tensors cannot be saved for backward
两者都不建图；inference_mode 更彻底（连版本计数都不记，推理更快），代价是产出的张量不能再参与求导
只想临时关梯度、之后还要接着训练，用 no_grad；纯推理服务用 inference_mode

—— 自动混合精度 ——
autocast 里矩阵乘的输出 dtype： torch.bfloat16 ；归一化、softmax 这类敏感算子仍然留在 fp32
GPU 上要配 GradScaler（fp16）或者直接用 bf16（不用 scaler）；CPU 上没有专门的矩阵指令时反而更慢

—— profiler：先看是谁最慢 ——
占用 CPU 时间最多的两个算子： ['aten::mm', 'aten::addmm']
（具体耗时每台机器、每次运行都不一样，这里只看排名：矩阵乘一定在最前面）
完整的表用 prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10) 打出来
GPU 上把 activities 加上 CUDA，再按 self_cuda_time_total 排序；要看时间线就用 Nsight Systems
```

**怎么读报错**。PyTorch 的形状报错里一般直接写着两个形状，对一下哪一维不匹配就知道少了什么。另外两类常见的：

- `expected scalar type Float but found Double`：dtype 不一致，多半是某个张量来自 NumPy（默认 float64）；
- `Expected all tensors to be on the same device`：有东西没跟着 `.to(device)` 搬，十有八九是放进普通 list 的子模块或者忘了注册的 buffer（第五章）。

**三个坑**：

1. **把带梯度的张量攒进容器**。日志里 `running_loss += loss` 看起来无害，但每一个 loss 都拖着一整张图，一个 epoch 下来显存就没了。要么 `.item()`，要么 `.detach()`；
2. **忘了 `eval()`**：dropout 还在随机丢，同一个输入两次结果不一样，评估指标会莫名其妙地抖；
3. **原地改了视图**：切片是视图，`row += 100` 会改到原张量。函数接收别人的张量时，要么明确文档说明自己会原地修改，要么先 `.clone()`。

**提速的三件小事**：

- **`no_grad` / `inference_mode`**：推理时必加。后者更彻底，但它产出的张量不能再进计算图——所以训练中途的评估用 `no_grad`，纯推理服务用 `inference_mode`；
- **autocast**：让矩阵乘跑在 bf16/fp16 上，精度敏感的算子留在 fp32。GPU 上这是免费的两倍速；CPU 上没有对应的矩阵指令时反而更慢（第三章《从零训练》里实测过）；
- **profiler**：先量再优化。`torch.profiler` 列出每个算子的耗时，按 self time 排序就是热点列表。真正要看时间线、看 GPU 空隙，用 Nsight Systems（见 [CUDA 手册的 Profiling 一章](cuda://tools/profiling/)）。

!!! tip "量之前先锁住变量"
    测两版代码哪个快，先确认它们算的是同一件事（输出对得上）、跑的是同样的输入、并且都预热过。CUDA 手册里[「让测出来的数字可信」](cuda://basics/first-kernel/#让测出来的数字可信)那一节把这件事讲得更细：计时口径、统计量、锁频，一样都不能少。

!!! interview "怎么讲清楚"
    讲"PyTorch 代码怎么调"：先**读报错**——形状错误里直接写着两个形状；dtype 错误多半来自 NumPy 的 float64；设备错误多半是子模块放进了普通 list 或者 buffer 没注册。再讲三个高频坑：**带梯度的张量攒进容器**会把显存吃光（要 `.item()` 或 `.detach()`）、**忘了 `eval()`** 让评估结果抖、**原地改视图**会改到调用方的数据。性能上给三件事：推理用 `no_grad` 或更彻底的 `inference_mode`、矩阵乘用 autocast、优化前先用 profiler 按 self time 找热点。

## 练习

**1. 复现一次显存泄漏。** 在 GPU 上跑一个循环，把 `loss` 直接 append 进 list，用 `torch.cuda.memory_allocated()` 观察显存怎么涨；换成 `.item()` 再看一次。

??? success "参考思路"
    不 detach 时显存会线性增长，因为每个 loss 都保留着自己那张图上的全部中间激活。换成 `.item()` 之后显存稳定。这是实际项目里最常见的"显存莫名其妙满了"。

**2. 量一次 autocast 的收益。** 在有 GPU 的机器上，对同一个线性层堆叠分别用 fp32 和 bf16 autocast 计时，算出加速比，再检查输出的最大误差。

??? success "参考思路"
    Tensor Core 上 bf16 的矩阵乘算力是 fp32 的十几倍，但端到端通常只快一两倍——因为还有访存和那些留在 fp32 的算子。误差在 1e-2 量级是正常的，bf16 的尾数只有 7 位。

**3. 用 profiler 找一次热点。** 给一个你手上的模型跑一次 profiler，按 self time 列出前五个算子，判断它是算力瓶颈还是访存瓶颈。

??? success "参考思路"
    如果前几名是 `mm` / `addmm` / `conv`，大概率是算力瓶颈，考虑 autocast、更大的 batch、`torch.compile`；如果前几名是 `copy_`、`contiguous`、各种 elementwise，那是访存和布局问题，考虑减少不必要的 `permute`、用融合算子。

## 小结

- [x] 形状报错里写着两个形状；dtype 错误多半来自 NumPy，设备错误多半是没注册的子模块或 buffer。
- [x] 带梯度的张量别攒进容器（`.item()` 或 `.detach()`）；推理前记得 `eval()`；视图会被原地改掉。
- [x] 推理用 `no_grad`，纯推理服务用更彻底的 `inference_mode`；矩阵乘用 autocast。
- [x] 优化之前先用 profiler 按 self time 找热点，再决定是算力问题还是访存问题。
