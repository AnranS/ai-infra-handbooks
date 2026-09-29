# AI 编译器全景

<p class="lead">模型结构每个月都在变，硬件每一两年换一代，靠人手为每一种"算子 × 形状 × 硬件"写 kernel 是跟不上的。AI 编译器要做的，就是把一张计算图自动变成在目标硬件上跑得快的代码。这一章给出一张地图：编译器分几层、每层做什么优化、主流的编译器各自在哪一层，以及推理工程师什么时候靠编译器、什么时候手写 kernel。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 图级优化和算子级（循环级）优化分别做什么？各举两个例子。
    2. 为什么"算子融合"对推理特别重要？什么样的算子容易融合、什么样的难？
    3. Halide 提出的"算法与调度分离"是什么意思？Triton 把哪些调度决策交给了编译器？
    4. 一个 Triton kernel 从 Python 到 GPU 机器码要经过哪几层表示？
    5. TileLang、CuTe DSL 这一类新的 kernel 语言，想解决 Triton 的什么问题？

## 编译器的三层

| 层次 | 输入 | 典型优化 | 代表 |
| --- | --- | --- | --- |
| 图级 | 计算图（FX、XLA HLO、ONNX、MLIR 的高层方言） | 常量折叠、公共子表达式消除、死代码删除、**算子融合**、布局转换、量化和精度转换 | Dynamo + Inductor 的图 pass、XLA、TVM Relax、TensorRT 的图优化 |
| 算子级（循环级） | 一个（融合后的）算子的循环嵌套 | 分块（tiling）、向量化、循环交换与展开、利用共享内存和 Tensor Core、流水线 | Inductor 的循环 IR、TVM TensorIR、Halide、MLIR linalg / affine、Triton |
| 代码生成 | 低层 IR | 寄存器分配、指令选择与调度 | LLVM → PTX → SASS（`ptxas`） |

推理工程师最常接触的是前两层：图级决定"哪些运算合成一个 kernel"，算子级决定"一个 kernel 内部怎么切块、怎么搬数据"。

## 图级：融合省下的是访存

推理（尤其是 decode）大多是带宽瓶颈，所以图级最重要的优化是**融合**：让中间结果留在寄存器或共享内存里，而不是写回显存再读出来。下面在 FX 图上写一个最简化的"融合分析"，量一下 RMSNorm + SiLU + 乘法（SwiGLU 的一半）融合前后的访存：

```python title="fx_fusion.py"
import operator

import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp

# 逐元素运算和归约：本例里所有算子都属于这两类
POINTWISE_FN = {operator.add, operator.mul, operator.truediv, torch.rsqrt, torch.nn.functional.silu}
POINTWISE_METHOD = {"pow"}
REDUCE_METHOD = {"mean", "sum"}


def rmsnorm_silu_mul(x, w, up):
    h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    return torch.nn.functional.silu(h) * up


def nbytes(node):
    meta = node.meta.get("tensor_meta")
    return meta.shape.numel() * meta.dtype.itemsize if meta is not None else 0


def fusible(node):
    if node.op == "call_function":
        return node.target in POINTWISE_FN
    if node.op == "call_method":
        return node.target in POINTWISE_METHOD | REDUCE_METHOD
    return False


gm = fx.symbolic_trace(rmsnorm_silu_mul)
x, w, up = torch.randn(8, 4096), torch.randn(4096), torch.randn(8, 4096)
ShapeProp(gm).propagate(x, w, up)   # 在图上推导每个节点的形状和类型

eager = 0
ops = [n for n in gm.graph.nodes if fusible(n)]
for n in ops:   # eager：每个算子读自己的全部输入、写自己的输出
    read = sum(nbytes(a) for a in n.all_input_nodes)
    eager += read + nbytes(n)
    name = n.target if isinstance(n.target, str) else n.target.__name__
    print(f"{name:<8} 读 {read:>7} 字节，写 {nbytes(n):>7} 字节")

# 融合：整条链只读图的输入、只写最终输出，中间结果留在寄存器里
group = set(ops)
inputs = {a for n in ops for a in n.all_input_nodes if a not in group}
outputs = [n for n in ops if any(u not in group for u in n.users)]
fused = sum(nbytes(a) for a in inputs) + sum(nbytes(o) for o in outputs)
print(f"{len(ops)} 个算子逐个执行：共 {eager} 字节；融合成 1 个 kernel：{fused} 字节，是原来的 {fused / eager:.1%}")
```

```text title="输出"
pow      读  131072 字节，写  131072 字节
mean     读  131072 字节，写      32 字节
add      读      32 字节，写      32 字节
rsqrt    读      32 字节，写      32 字节
mul      读  131104 字节，写  131072 字节
mul      读  147456 字节，写  131072 字节
silu     读  131072 字节，写  131072 字节
mul      读  262144 字节，写  131072 字节
8 个算子逐个执行：共 1589440 字节；融合成 1 个 kernel：409600 字节，是原来的 25.8%
```

融合把访存降到四分之一；在带宽瓶颈下，时间也差不多降到四分之一，还省掉了 7 次 kernel 启动。真实编译器的融合要复杂得多，需要回答：

- **什么能融合**：逐元素运算之间最容易；逐元素运算可以融进归约的前面（prologue）和后面（epilogue）；矩阵乘的输出可以接逐元素运算（偏置、激活、量化）作为 epilogue。
- **什么不该融合**：两个都需要"全局视野"的归约（比如先按行 softmax、再按列求和）；融合后寄存器压力太大、导致并行度下降；一个中间结果被多个消费者使用时，重复计算可能比写回显存还贵。
- 所以融合是一个**代价模型**驱动的决策：Inductor 按访存节省和寄存器压力打分，TensorRT 有一组预设的融合模式。

## 算子级：算法与调度分离

同一个算子（比如矩阵乘）的**算法**是固定的，但**调度**——循环怎么切块、按什么顺序、用哪一级存储、怎么并行——有无数种，性能能差几十倍。Halide 首先提出把两者分开写：算法只写一次，调度单独描述、可以反复尝试。TVM 继承了这个思路，并用**自动调优**（在调度空间里搜索、用代价模型或实测挑最快的）代替人手写调度。

Triton 走了另一条路：程序员以**块**（tile）为单位写 kernel，决定每个程序实例处理哪一块数据；块**内部**的事情交给编译器——每个线程负责哪些元素、怎么合并访存、怎么用共享内存、怎么调用 Tensor Core、怎么做多级流水线。块大小和 warp 数等参数用 `triton.autotune` 在几组候选里实测挑选。
这把门槛降到了"会写 NumPy 风格的块运算"，代价是一部分底层控制交给了编译器（见 CUDA 手册的 [Triton](../tools/triton.md)）。

### 一个 Triton kernel 的编译流程

```text
Python 函数（@triton.jit）
  → Triton IR（ttir）：块级的张量运算，和硬件无关
  → TritonGPU IR（ttgir）：给每个张量指定在线程之间的分布（layout），插入共享内存、异步拷贝和流水线
  → LLVM IR（llir）
  → PTX
  → cubin（ptxas 生成的 GPU 机器码）
```

编译好的 kernel 对象可以取出每一层（`compiled.asm["ttgir"]`、`["ptx"]` 等），性能不符合预期时，看 ttgir 里的 layout 和流水线、看 PTX 里有没有用上 `mma` / `wgmma` 指令，是定位问题的常用办法。

### 新一代的 tile 语言

Hopper 和 Blackwell 带来了 TMA（张量内存加速器）、`wgmma`（warpgroup 级矩阵乘）、线程块集群、Blackwell 的 Tensor Memory 等新硬件特性。要把它们用到极致，需要精确控制"哪个 warp 做什么"（warp 专用化：一部分 warp 只负责搬数据、一部分只负责计算）——这恰恰是 Triton 交给编译器、而编译器不一定做得好的部分。于是出现了一批"比 CUDA 高一层、比 Triton 低一层"的语言：

| 语言 | 思路 |
| --- | --- |
| CUTLASS / CuTe（C++）与 CuTe DSL（Python） | 用"布局代数"精确描述张量在线程、寄存器、共享内存之间的分布；FlashAttention-3 基于它，DeepGEMM 也借鉴了它的概念 |
| TileLang | 在 TVM 之上的 tile 级 DSL，允许显式控制流水线和 warp 分工；DeepSeek-V3.2 开源的部分算子提供了 TileLang 版本 |
| Triton Gluon | Triton 的底层扩展，允许直接写 layout 和 warp 专用化 |
| ThunderKittens | 以 16×16 的寄存器 tile 为基本单位的 C++ 库 |

趋势是：常规算子交给编译器（Inductor 自动生成），性能关键的少数算子（注意力、GEMM、MoE）用这些 tile 语言手工精调。

## 其他编译器与推理引擎

- **XLA**：TensorFlow / JAX 的编译器，以 HLO 为中间表示，是 TPU 的主要编程路径；JAX 的 `jit` 就是把函数编译成 XLA；
- **MLIR**：不是一个编译器，而是构建编译器的基础设施——一套可扩展的多层中间表示（方言）。Triton、IREE、torch-mlir 以及很多国产加速卡的编译器都基于它；
- **TensorRT / TensorRT-LLM**：NVIDIA 的推理优化器，做图融合、精度校准（INT8 / FP8）、为每个算子在多个实现（tactic）里实测选最快的；TensorRT-LLM 在它之上加了 LLM 专用的插件（分页注意力、飞行中批处理）；
- **TVM**：端到端的开源编译栈，从图级到代码生成，自动调优是其核心，也是很多端侧、非 NVIDIA 硬件部署的选择。

## 靠编译器还是手写 kernel

| 情况 | 选择 |
| --- | --- |
| 逐元素运算、归约、它们的组合（归一化、激活、残差、量化的前后处理） | 交给 `torch.compile` / Inductor，通常已经接近带宽上限 |
| 标准形状的矩阵乘 | cuBLAS / cuBLASLt，或 Inductor 的 `max-autotune` |
| 注意力（分页、变长、GQA / MLA、稀疏）、MoE 的分组 GEMM、非标准量化的 GEMM | 用成熟的库（FlashAttention、FlashInfer、FlashMLA、DeepGEMM），或用 Triton / CuTe / TileLang 手写 |
| 通信与计算融合、跨设备的算子 | 手写（NVSHMEM、自定义 all-reduce），编译器目前很难自动生成 |

面试里如果被问到"你会怎么优化某个算子"，一个好的回答顺序是：先用屋顶线算出它是带宽还是算力瓶颈、离上限多远；再看能否靠融合（编译器）解决；最后才是手写 kernel 以及用哪种语言、为什么。

!!! interview "面试怎么答"
    AI 编译器题：分图级、算子级、代码生成三层；推理（尤其 decode）受带宽限制，最受益的是图级的融合，省下的是访存和 kernel 启动（本章 RMSNorm + SiLU + 乘法融合后访存降到约四分之一）。融合由代价模型决定：逐元素运算、按行归约的前后处理容易融，归约方向不一致、寄存器压力过大时不融。算子级的核心是"算法与调度分离"：Halide / TVM 搜索调度，Triton 让程序员管块、编译器管块内；Hopper / Blackwell 催生了 CuTe DSL、TileLang 这类更底层的 tile 语言。

## 练习

1. 在 `fx_fusion.py` 的基础上加一个 `torch.softmax` 调用（作用在最后一维），把它放进融合组里是否合理？如果 softmax 作用在第 0 维呢？

??? success "参考答案"
    最后一维的 softmax 是"按行"的归约 + 逐元素运算，和 RMSNorm 一样，一行数据可以在一个线程块里完成，适合融合（这正是 online softmax、FlashAttention 的基础）。
    作用在第 0 维时，归约方向和 RMSNorm 的归约方向（最后一维）不同：一个线程块处理一行时拿不到整列的数据，融合需要把整个张量放在片上或者跨线程块同步，通常不划算，编译器会在这里切开成两个 kernel。

2. 为什么 Triton 的 `autotune` 通常要按"输入形状"作为键分别调优？推理服务里这会带来什么问题？

??? success "参考答案"
    最佳的块大小、warp 数、流水线级数取决于问题的形状：小 M（decode 的 batch 小）适合沿 N、K 切得更细甚至 split-K，大 M（prefill）适合大块。所以不同形状要分别调优并缓存结果。
    推理服务里 batch 和序列长度一直在变：每遇到一个新形状就调优一次，会在线上引入长时间的停顿。常见的做法是按区间分桶（比如 batch 向上取整到 2 的幂）、启动时预先调优常见的桶、或者离线调优后把配置表写进代码（很多推理库的 MoE、量化 GEMM 就带着按 GPU 型号生成的配置文件）。

## 小结

- [x] AI 编译器分图级、算子级、代码生成三层；推理最受益的是图级的融合，它省下的是访存和 kernel 启动。
- [x] 融合是代价模型驱动的：逐元素运算、按行归约的前后处理容易融合；归约方向不一致、寄存器压力过大时不融合。
- [x] 算子级优化的核心是"算法与调度分离"：Halide / TVM 搜索调度，Triton 让程序员管块、编译器管块内。
- [x] Triton 的编译链：ttir → ttgir → LLVM IR → PTX → cubin；性能问题看 ttgir 的 layout 和 PTX 的指令。
- [x] Hopper / Blackwell 催生了 CuTe DSL、TileLang、Gluon 这类更底层的 tile 语言；常规算子靠编译器，注意力、GEMM、MoE、通信融合靠库或手写。
