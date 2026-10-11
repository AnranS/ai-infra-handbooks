# （十五）这台机器能跑多快

<p class="lead">前面十四章只关心"对不对"。这一章量"多快"：线程数、递推还是分块、fp32 还是 bf16，以及 CPU / Apple MPS / NVIDIA 的差别。脚本会自动选设备，在你的机器上跑一遍就得到同样的表——下面贴的是一台 32 核 Xeon 上的结果。</p>

!!! note "这一章跑什么"
    `ch15_speed.py`，CPU 上约 2 分钟（线程扫描最费时）。

## 量四件事

```python title="ch15_speed.py" ci="loose"
"""第 15 章：这台机器上能跑多快。线程数、递推 vs 分块、fp32 vs bf16、CPU / MPS / CUDA。"""
import os
import platform
import time

import torch

from qwen.cache import State
from qwen.chat import Chat, build_prompt
from qwen.chunk import chunked_delta_rule
from qwen.deltanet import delta_rule
from qwen.model import Model
from qwen.weights import find_model

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"{platform.machine()} / {device} / torch {torch.__version__} / {os.cpu_count()} 个逻辑核")


def decode_speed(model, ids, n=32):
    state = State(model)
    model(ids, state)
    tok = ids[:, -1:]
    t = time.time()
    for _ in range(n):
        tok = model(tok, state)[:, -1:].argmax(-1)
    if device == "cuda":
        torch.cuda.synchronize()
    return n / (time.time() - t)


path = find_model()
model = Model.from_pretrained(path)
chat = Chat(model, path)
ids = chat.encode(build_prompt([{"role": "user", "content": "写一段 200 字的短文，介绍长江。"}]))

if device == "cpu":                                                              # 1. 线程数：不是越多越好
    for n in sorted({1, 2, 4, 8, os.cpu_count() // 2, os.cpu_count()}):
        torch.set_num_threads(n)
        print(f"{n:>3} 线程：decode {decode_speed(model, ids):.1f} tok/s")
    torch.set_num_threads(max(1, os.cpu_count() // 2))

torch.manual_seed(0)                                                             # 2. 递推 vs 分块：prefill 一段 512 token
B, T, H, d = 1, 512, 16, 128
q, k, v = (torch.randn(B, T, H, d, device=device) for _ in range(3))
g, beta = -torch.rand(B, T, H, device=device) * 0.5, torch.rand(B, T, H, device=device)
for name, fn in (("递推", delta_rule), ("分块", chunked_delta_rule)):
    t = time.time()
    out, _ = fn(q, k, v, g, beta)
    print(f"一层线性注意力，{T} 个 token：{name} {(time.time() - t) * 1000:.0f} ms")

for dtype in (torch.float32, torch.bfloat16):                                    # 3. fp32 vs bf16
    m = Model.from_pretrained(path, dtype=dtype).to(device)
    x = ids.to(device)
    t = time.time()
    m(x, State(m))
    pre = (time.time() - t) * 1000
    print(f"{str(dtype).split('.')[-1]:>8}：prefill {ids.shape[1]} token {pre:.0f} ms，decode {decode_speed(m, x):.1f} tok/s")
```

```text title="输出"
x86_64 / cpu / torch 2.14.0+cpu / 32 个逻辑核
  1 线程：decode 4.3 tok/s
  2 线程：decode 7.4 tok/s
  4 线程：decode 10.6 tok/s
  8 线程：decode 15.9 tok/s
 16 线程：decode 18.4 tok/s
 32 线程：decode 0.3 tok/s
一层线性注意力，512 个 token：递推 129 ms
一层线性注意力，512 个 token：分块 11 ms
 float32：prefill 25 token 157 ms，decode 18.4 tok/s
bfloat16：prefill 25 token 327 ms，decode 18.0 tok/s
```

| 测什么 | 结论 |
| --- | --- |
| **线程数** | 16 线程最快，32 线程（用满所有逻辑核）慢 60 倍——超线程 + 内存带宽争抢，PyTorch 的默认值（全部核）在这台机器上是灾难。所以全书的脚本都写了 `set_num_threads(cpu_count() // 2)`。Mac 上不用管：macOS 的 PyTorch 默认只用性能核。 |
| **递推 vs 分块** | 一层线性注意力处理 512 个 token：129 ms vs 11 ms。prefill 必须用分块；decode 一次一个 token，递推更快。 |
| **fp32 vs bf16** | CPU 上 bf16 不占便宜：decode 持平（权重读取减半，但每个算子都要先转成 fp32 算），prefill 慢一倍（这台 CPU 没有 bf16 的矩阵乘指令）。GPU 上 bf16 全面快一倍，而且是部署的默认精度。 |
| **设备** | 脚本在 MPS / CUDA 上会自动用对应设备。 |

!!! tip "在你的机器上"
    `python ch15_speed.py` 在 Mac（MPS）或 NVIDIA 显卡上直接跑。线性注意力的递推版在 GPU 上尤其吃亏（几百次小核函数发射），分块版的优势会比 CPU 上更大；decode 的 tok/s 则主要看显存带宽。

## 为什么是这些数字

decode 一个 token 的下限是**把权重读一遍**：fp32 的 752M 参数 = 3 GB，这台机器的内存带宽约 100 GB/s，理论上限 30 多 tok/s；我们跑到 18，剩下的是 24 层 × 几十个小算子的调度开销和 Python 循环。GPU 上同样的账：bf16 1.5 GB，一张 5070 Ti 的 896 GB/s 带宽对应几百 tok/s，但小算子的发射开销会先成为瓶颈——这就是 CUDA Graph 和算子融合存在的理由（《手写 mini-sglang》里讲）。

prefill 则是算力瓶颈：T 个 token 的矩阵乘是 T × 1.5 GFLOP，长输入下 GPU 的 TFLOPS 才是决定因素。

!!! interview "怎么讲清楚"
    讲推理性能，先分 prefill（算力瓶颈）和 decode（访存瓶颈）两个阶段，再给出 decode 的下限公式：权重字节数 / 带宽。然后说三个最便宜的优化：bf16（权重减半）、批处理（一次读权重出 B 个 token）、投机解码（一次读权重出 >1 个 token）。线程数那一条是 CPU 特有的坑，但它说明"默认配置未必最优"是通用的教训。

## 小结

- [x] CPU：线程数开满反而慢几十倍；fp32 decode 约 18 tok/s，bf16 在 CPU 上不占便宜。
- [x] prefill 用分块、decode 用递推；长输入分块快 10 倍以上。
- [x] decode 的理论上限 = 权重字节数 / 内存带宽。
