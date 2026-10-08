# 8. Debugging and speed

<p class="lead">The last chapter is about what to do when something goes wrong: how to read a shape error, what the three most common traps look like, how <code>no_grad</code> and <code>inference_mode</code> really differ, how to use autocast, and how to find the slowest operator with the profiler. Every example runs, including the ones that raise.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. How do you read `mat1 and mat2 shapes cannot be multiplied (4x8 and 16x2)`?
    2. Why does `running_loss += loss` eat memory while `+= loss.item()` does not?
    3. How do `no_grad` and `inference_mode` differ?
    4. Does autocast turn every operator into bf16?
    5. What is the difference between the profiler's self time and total time?

??? success "Answers (try first, then expand to compare)"
    1. Both shapes are right there: `(4,8) @ (16,2)`, and 8 ≠ 16. A misaligned matrix multiply is almost always a missing transpose or a batch dimension that does not line up.
    2. The first accumulates tensors that still carry gradients, each dragging a whole graph, so none of the intermediate activations can be freed.
    3. Neither builds a graph; `inference_mode` goes further (it does not even track version counters) and is faster, but its outputs cannot take part in autograd afterwards.
    4. No. Only matrix multiplies, convolutions and the like — insensitive to precision and clearly faster — are converted; normalisation, softmax and losses stay in fp32.
    5. Self time counts only what the operator itself spent; total time includes the operators it called. Look at self time for hot spots, total time for structure.

## Errors, traps and speed {#报错坑与提速}

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
    bad.append(out)                                           # each one drags a whole graph along with it
    good.append(out.detach())                                 # or .item()
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
row = t[0]                                                    # a view
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

```text title="output"
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

**How to read an error**. PyTorch's shape errors generally print both shapes; check which dimension does not match and you know what is missing. Two other common ones:

- `expected scalar type Float but found Double`: a dtype mismatch, usually a tensor that came from NumPy (float64 by default);
- `Expected all tensors to be on the same device`: something did not travel with `.to(device)` — nine times out of ten a submodule in a plain list or a buffer that was never registered (chapter 5).

**Three traps**:

1. **Accumulating gradient-carrying tensors in a container**. `running_loss += loss` looks harmless in a logging line, but every loss drags a whole graph and one epoch is enough to exhaust memory. Use `.item()` or `.detach()`;
2. **Forgetting `eval()`**: dropout is still dropping, the same input gives two different answers, and the evaluation metric jitters for no visible reason;
3. **Modifying a view in place**: a slice is a view, and `row += 100` changes the original. A function that takes someone else's tensor should either document that it modifies in place or `.clone()` first.

**Three small things for speed**:

- **`no_grad` / `inference_mode`**: mandatory at inference. The latter goes further, but its outputs cannot re-enter a graph — so use `no_grad` for evaluation during training and `inference_mode` for a pure inference service;
- **autocast**: run the matrix multiplies in bf16/fp16 and leave the precision-sensitive operators in fp32. On a GPU this is a free near-doubling; on a CPU without the matching matrix instructions it is slower (measured in chapter 3 of *Train a Small Model*);
- **profiler**: measure before optimizing. `torch.profiler` lists the time per operator, and sorting by self time is the hot-spot list. For a timeline and GPU gaps, use Nsight Systems (see [the CUDA handbook's profiling chapter](cuda://tools/profiling/)).

!!! tip "Pin the variables before measuring"
    Before comparing two versions, make sure they compute the same thing (the outputs agree), run on the same input, and are both warmed up. [Making the numbers trustworthy](cuda://basics/first-kernel/#让测出来的数字可信) in the CUDA handbook goes into this in more detail: the timing scope, the statistic and locked clocks all matter.

!!! interview "How to explain it"
    On debugging PyTorch code: start by **reading the error** — shape errors print both shapes; dtype errors usually come from NumPy's float64; device errors usually come from a submodule in a plain list or an unregistered buffer. Then three frequent traps: **accumulating gradient-carrying tensors** exhausts memory (use `.item()` or `.detach()`), **forgetting `eval()`** makes evaluation jitter, and **modifying a view in place** changes the caller's data. For performance, give three things: `no_grad` or the stricter `inference_mode` at inference, autocast for the matrix multiplies, and the profiler sorted by self time before optimizing anything.

## Exercises {#练习}

**1. Reproduce a memory leak.** On a GPU, append `loss` directly to a list in a loop and watch `torch.cuda.memory_allocated()`; then switch to `.item()` and watch again.

??? success "An approach"
    Without detaching, memory grows linearly, because each loss keeps every intermediate activation of its own graph. With `.item()` it stays flat. This is the most common "memory mysteriously filled up" in real projects.

**2. Measure autocast's benefit.** On a machine with a GPU, time the same stack of linear layers in fp32 and under bf16 autocast, compute the speedup, and check the maximum output difference.

??? success "An approach"
    bf16 matrix multiplies on Tensor Cores have more than ten times fp32's throughput, but end to end you usually see one to two times — memory traffic and the operators still in fp32 take the rest. A difference on the order of 1e-2 is normal; bf16 has only 7 mantissa bits.

**3. Find a hot spot with the profiler.** Run the profiler on a model of your own, list the top five operators by self time, and decide whether it is compute-bound or memory-bound.

??? success "An approach"
    If the top entries are `mm` / `addmm` / `conv`, it is most likely compute-bound: consider autocast, a larger batch, `torch.compile`. If they are `copy_`, `contiguous` and assorted elementwise operators, it is a memory and layout problem: consider removing unnecessary `permute`s and using fused operators.

## Summary {#小结}

- [x] Shape errors print both shapes; dtype errors usually come from NumPy, device errors from unregistered submodules or buffers.
- [x] Do not accumulate gradient-carrying tensors (`.item()` or `.detach()`); remember `eval()` before evaluating; views get modified in place.
- [x] Use `no_grad` at inference and the stricter `inference_mode` for a pure inference service; use autocast for matrix multiplies.
- [x] Profile by self time before optimizing, then decide whether the problem is compute or memory.
