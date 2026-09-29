# 端侧推理：llama.cpp 与 GGUF、MLX、ExecuTorch

<p class="lead">服务端推理追求的是"每张卡服务尽量多的请求"；端侧（笔记本、手机、边缘设备）只服务一个用户，约束完全不同：内存只有几 GB 到几十 GB 且与系统共享，带宽几十到几百 GB/s，还有功耗和发热。这一章看端侧推理的三个主流框架各自的做法，重点是 llama.cpp 的 GGUF 量化格式——为什么同样是"4 比特"，有的格式明显更准——以及怎样估算一台设备能跑多快。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 端侧 decode 的速度上限由什么决定？一台 8B 模型在手机上大约多快？
    2. GGUF 的 Q4_0、Q4_1、Q4_K 有什么区别？为什么 Q4_K 更准？
    3. llama.cpp、MLX、ExecuTorch 各适合什么平台？
    4. 端侧的 prefill 和 decode 分别受什么限制？NPU 能帮上哪一个？

## 速度上限：带宽 ÷ 权重字节数

单用户的 decode 每一步都要把全部权重读一遍，batch 为 1，完全是访存瓶颈。所以端侧最重要的两个数字是**内存带宽**和**模型量化后的字节数**：前者由硬件决定，后者由量化格式决定。

## GGUF 的量化格式

llama.cpp 的 GGUF 文件里，权重按块量化，每块带自己的缩放。几种常见的格式：

- **Q8_0**：每 32 个权重一块，一个 fp16 缩放、int8 对称量化，8.5 比特 / 权重；
- **Q4_0**：每 32 个一块，一个 fp16 缩放，4 比特对称（用绝对值最大的数定缩放），4.5 比特 / 权重；
- **Q4_1**：每 32 个一块，fp16 缩放 + fp16 最小值，4 比特非对称，5 比特 / 权重；
- **Q4_K**（K-quants）：256 个一个超块，内部分成 8 个 32 的子块；每个子块有自己的缩放和最小值，但这两个数本身再量化成 6 比特、由超块的 fp16 缩放统一还原——用两级缩放把"每个子块都非对称"的精度，压到 4.5 比特的开销里。

用长尾分布的权重比较（Q4_K 按上面的思路简化实现）：

```python
import torch

torch.manual_seed(0)
w = torch.distributions.StudentT(5.0).sample((4096 * 256,)) * 0.02     # 长尾分布的权重，1M 个


def q8_0(x):                                    # 每 32 个一块，fp16 缩放，int8 对称
    b = x.view(-1, 32)
    d = b.abs().amax(1, keepdim=True) / 127
    return ((b / d).round().clamp(-127, 127) * d).flatten(), 8 + 16 / 32


def q4_0(x):                                    # 每 32 个一块，fp16 缩放，4 比特对称（偏移 8）
    b = x.view(-1, 32)
    idx = b.abs().argmax(1, keepdim=True)
    d = b.gather(1, idx) / -8                   # 用绝对值最大的那个数（带符号）定缩放，让它落在 -8
    return ((b / d).round().clamp(-8, 7) * d).flatten(), 4 + 16 / 32


def q4_1(x):                                    # 每 32 个一块，fp16 缩放 + fp16 最小值，4 比特非对称
    b = x.view(-1, 32)
    lo, hi = b.amin(1, keepdim=True), b.amax(1, keepdim=True)
    d = (hi - lo) / 15
    return (((b - lo) / d).round().clamp(0, 15) * d + lo).flatten(), 4 + 32 / 32


def q4_k(x):                                    # 256 个一个超块，内分 8 个 32 的子块：子块的缩放和最小值再量化成 6 比特
    b = x.view(-1, 8, 32)
    lo, hi = b.amin(2, keepdim=True).clamp(max=0), b.amax(2, keepdim=True)
    sc, mn = (hi - lo) / 15, -lo
    dsc, dmn = sc.amax(1, keepdim=True) / 63, mn.amax(1, keepdim=True) / 63                 # 超块的 fp16 缩放
    sc_q, mn_q = (sc / dsc).round().clamp(1, 63) * dsc, (mn / dmn).round().clamp(0, 63) * dmn
    q = ((b + mn_q) / sc_q).round().clamp(0, 15)
    return (q * sc_q - mn_q).flatten(), 4 + (6 + 6) * 8 / 256 + 2 * 16 / 256


for name, fn in (("Q8_0", q8_0), ("Q4_0", q4_0), ("Q4_1", q4_1), ("Q4_K（简化）", q4_k)):
    deq, bpw = fn(w)
    print(f"{name:<11} {bpw:.2f} 比特/权重，相对误差 {((deq - w).norm() / w.norm()).item():.4f}")

print("8B 模型 Q4_K（约 4.5 GB）在不同设备上 decode 的速度上限（带宽 × 70% / 权重字节数）：")
for dev, bw in (("手机（LPDDR5X，约 60 GB/s）", 60e9), ("轻薄本（约 120 GB/s）", 120e9), ("M 系列 Max 芯片（约 400 GB/s）", 400e9), ("RTX 4090（约 1 TB/s）", 1008e9)):
    print(f"  {dev}：约 {bw * 0.7 / 4.5e9:.0f} token/s")
```

```text title="输出"
Q8_0        8.50 比特/权重，相对误差 0.0068
Q4_0        4.50 比特/权重，相对误差 0.1079
Q4_1        5.00 比特/权重，相对误差 0.0901
Q4_K（简化）    4.50 比特/权重，相对误差 0.0903
8B 模型 Q4_K（约 4.5 GB）在不同设备上 decode 的速度上限（带宽 × 70% / 权重字节数）：
  手机（LPDDR5X，约 60 GB/s）：约 9 token/s
  轻薄本（约 120 GB/s）：约 19 token/s
  M 系列 Max 芯片（约 400 GB/s）：约 62 token/s
  RTX 4090（约 1 TB/s）：约 157 token/s
```

- Q4_K 用和 Q4_0 一样的 4.5 比特，达到了 Q4_1（5 比特）的精度：两级缩放把元数据的开销压了下来；
- 实际使用中常见的是混合精度的"配方"（比如 Q4_K_M：对更敏感的层——注意力的 V 投影、FFN 的下投影——用更高的精度），以及用校准数据统计每个权重重要性的 imatrix，进一步减小误差；
- 速度上限直接由带宽决定：同一个 8B 模型，手机上每秒不到 10 个 token，Max 芯片的笔记本上六十多个。想在手机上更快，只能用更小的模型、更低的比特，或者投机解码（用更小的草稿模型，一次验证多个 token，把访存受限的 decode 摊薄）。

## 三个框架

| 框架 | 平台 | 做法 |
| --- | --- | --- |
| **llama.cpp**（以及基于它的 Ollama 等） | 几乎所有平台：x86 / ARM CPU、Apple Metal、CUDA、Vulkan 等 | C / C++ 的张量库 ggml；GGUF 单文件格式（权重 + 分词器 + 配置），mmap 直接映射；丰富的量化格式；CPU 上用 SIMD 做量化矩阵乘 |
| **MLX** | Apple Silicon | Apple 的数组框架：统一内存（CPU 和 GPU 共享同一块内存，没有拷贝）、惰性求值、类似 NumPy / PyTorch 的接口；mlx-lm 提供模型转换、量化和推理 |
| **ExecuTorch** | 手机与嵌入式（Android、iOS、微控制器） | PyTorch 官方的端侧方案：`torch.export` 导出计算图，编译成 `.pte` 文件，运行时很小；通过"委托"（delegate）把子图交给 XNNPACK（CPU）、Core ML、高通 QNN 等硬件后端 |

## 端侧特有的问题

- **prefill 与 decode 的瓶颈不同**：decode 受带宽限制；prefill（处理长提示词）受算力限制。手机的 NPU 算力可观但带宽和 CPU / GPU 共享，常见的分工是 NPU 做 prefill、CPU / GPU 做 decode，或者整个模型都跑在 NPU 上但只用于 prefill 较重的任务；
- **内存与系统共享**：手机上 8～16 GB 的内存要同时容纳系统和其他应用，模型加上 KV Cache 通常只能用到其中的一小部分，所以端侧模型多在 1B～8B，KV Cache 也要量化；
- **功耗与发热**：持续满负荷运行会触发降频，长输出的后半段可能明显变慢；基准测试要测持续的速度，而不是前几秒；
- **首次加载**：mmap 让模型"秒开"，但第一次访问每一页都要从存储读取，冷启动的前几个 token 会慢。

!!! interview "面试怎么答"
    端侧问题先抓住"单用户 decode = 带宽 ÷ 权重字节数"：8B 的 4 比特模型约 4.5 GB，手机 60 GB/s 只有每秒约 9 个 token。再讲量化格式：GGUF 按块量化，Q4_K 用两级缩放在 4.5 比特下达到 5 比特格式的精度，加上按层混合精度和 imatrix。最后讲平台：llama.cpp 全平台、MLX 用统一内存、ExecuTorch 导出计算图并委托给硬件后端；以及 NPU 做 prefill、功耗降频这些端侧特有的约束。

## 练习

**1. 手机上跑 3B 模型，想达到每秒 20 个 token，需要什么量化？**

??? success "参考答案"
    带宽 60 GB/s、有效 70% 约 42 GB/s，20 token/s 要求每步读的权重不超过 2.1 GB。3B 模型在 4.5 比特下约 1.7 GB，满足；8 比特（约 3.2 GB）只能跑到约 13 token/s。再加上 KV Cache 的读取（上下文长时不可忽略）和发热降频，实际要留余量，4 比特左右是合适的选择。

## 小结

- [x] 端侧单用户 decode 的速度上限 = 有效带宽 ÷ 量化后的权重字节数；手机上 8B 的 4 比特模型每秒不到 10 个 token。
- [x] GGUF 按块量化：Q8_0 / Q4_0 / Q4_1 单级缩放，Q4_K 用超块两级缩放，在 4.5 比特下达到 5 比特格式的精度；实际常用按层混合精度与 imatrix。
- [x] llama.cpp 覆盖全平台、MLX 用 Apple 的统一内存、ExecuTorch 导出图并委托给 NPU 等后端；端侧还要考虑 NPU 分工、内存共享、功耗降频与冷启动。
